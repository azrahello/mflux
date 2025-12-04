import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

import mlx.core as mx
from mlx import nn
from mlx.utils import tree_flatten
from tqdm import tqdm
from transformers import PreTrainedTokenizer

from mflux.utils.version_util import VersionUtil

if TYPE_CHECKING:
    from mflux.models.common.weights.loading.weight_definition import WeightDefinitionType


class ModelSaver:
    @staticmethod
    def save_model(
        model: Any,
        bits: int,
        base_path: str,
        weight_definition: "WeightDefinitionType",
    ) -> None:
        # Save tokenizers from model.tokenizers dict
        tokenizer_defs = weight_definition.get_tokenizers()
        for t in tokenizer_defs:
            if hasattr(model, "tokenizers") and t.name in model.tokenizers:
                tokenizer_wrapper = model.tokenizers[t.name]
                if hasattr(tokenizer_wrapper, "tokenizer"):
                    ModelSaver._save_tokenizer(base_path, tokenizer_wrapper.tokenizer, t.hf_subdir)

        # Save model components with progress bar
        components = weight_definition.get_components()
        for component_def in tqdm(components, desc="Saving components", unit="component"):
            attr_name = component_def.model_attr or component_def.name
            component = getattr(model, attr_name, None)
            if component is not None:
                # Respect skip_quantization flag for each component
                component_bits = None if component_def.skip_quantization else bits
                ModelSaver._save_weights(base_path, component_bits, component, component_def.hf_subdir)

    @staticmethod
    def _save_tokenizer(base_path: str, tokenizer: PreTrainedTokenizer, subdir: str) -> None:
        path = Path(base_path) / subdir
        path.mkdir(parents=True, exist_ok=True)
        tokenizer.save_pretrained(path)

    @staticmethod
    def _save_weights(base_path: str, bits: int, model: nn.Module, subdir: str) -> None:
        from mflux.models.common.lora.baking.lora_baker import LoRABaker

        path = Path(base_path) / subdir
        path.mkdir(parents=True, exist_ok=True)

        # Bake LoRAs before saving (if any exist)
        baked_count = LoRABaker.bake_loras_inplace(model)
        if baked_count > 0:
            print(f"  🔥 Baked {baked_count} LoRA layer(s) into base weights")

            # Verify no LoRA layers remain
            from mflux.models.common.lora.layer.fused_linear_lora_layer import FusedLoRALinear
            from mflux.models.common.lora.layer.linear_lora_layer import LoRALinear

            remaining_loras = sum(1 for _, m in model.named_modules() if isinstance(m, (LoRALinear, FusedLoRALinear)))
            if remaining_loras > 0:
                print(f"  ⚠️  Warning: {remaining_loras} LoRA layer(s) still present after baking")

        # Quantize the entire model if requested (after baking LoRAs)
        if bits is not None:
            print(f"  ⚙️  Quantizing model to {bits}-bit...")
            nn.quantize(model, bits=bits)
            print("  ✓ Quantization complete")

        # Aggressive cleanup after baking
        mx.clear_cache()
        import gc

        gc.collect()

        # Force re-evaluation of the model's parameter tree
        # This ensures tree_flatten only sees the current state
        mx.eval(model.parameters())

        # Save shards directly without keeping them all in memory
        weight_map = ModelSaver._save_shards_streaming(model, path, bits)

        # Final cleanup
        gc.collect()
        mx.clear_cache()

        # Write model.safetensors.index.json for HuggingFace compatibility
        # This ensures the saved model works even if custom metadata is stripped
        index_data = {
            "metadata": {
                "quantization_level": str(bits),
                "mflux_version": VersionUtil.get_mflux_version(),
            },
            "weight_map": weight_map,
        }
        with open(path / "model.safetensors.index.json", "w") as f:
            json.dump(index_data, f, indent=2)

    @staticmethod
    def _save_shards_streaming(
        model: nn.Module, path: Path, bits: int | None, max_file_size_gb: int = 2
    ) -> dict[str, str]:
        """
        Stream-save model weights directly to disk without keeping all shards in memory.
        This is the most memory-efficient approach for saving large models.
        """
        import gc

        max_file_size_bytes = max_file_size_gb << 30
        weight_map = {}
        shard: dict = {}
        shard_size = 0
        shard_index = 0

        # Use tqdm to show progress
        # First count total parameters to show accurate progress
        param_count = sum(1 for key, _ in tree_flatten(model.parameters()) if "lora" not in key.lower())
        param_iter = tqdm(
            tree_flatten(model.parameters()),
            total=param_count,
            desc=f"  {path.name}",
            unit="weight",
            leave=False,
        )

        for key, value in param_iter:
            # Skip LoRA-related parameters
            if "lora" in key.lower():
                continue

            # Evaluate this specific weight to materialize it
            mx.eval(value)

            # Check if adding this weight would exceed shard size
            if shard_size + value.nbytes > max_file_size_bytes and shard:
                # Save current shard immediately
                shard_filename = f"{shard_index}.safetensors"
                mx.save_safetensors(
                    str(path / shard_filename),
                    shard,
                    {
                        "quantization_level": str(bits),
                        "mflux_version": VersionUtil.get_mflux_version(),
                    },
                )

                # Update weight_map
                for shard_key in shard.keys():
                    weight_map[shard_key] = shard_filename

                # Clear shard and cleanup
                del shard
                shard = {}
                shard_size = 0
                shard_index += 1

                # Aggressive cleanup
                gc.collect()
                mx.clear_cache()

            shard[key] = value
            shard_size += value.nbytes

        # Don't forget to save the last shard
        if shard:
            shard_filename = f"{shard_index}.safetensors"
            mx.save_safetensors(
                str(path / shard_filename),
                shard,
                {
                    "quantization_level": str(bits),
                    "mflux_version": VersionUtil.get_mflux_version(),
                },
            )

            for shard_key in shard.keys():
                weight_map[shard_key] = shard_filename

            del shard
            gc.collect()
            mx.clear_cache()

        return weight_map

    @staticmethod
    def _split_weights_incremental(model: nn.Module, max_file_size_gb: int = 2) -> list[dict]:
        """
        Split model weights into shards incrementally to minimize memory usage.
        Processes parameters one at a time instead of loading all into memory.
        """
        import gc

        max_file_size_bytes = max_file_size_gb << 30
        shards: list[dict] = []
        shard: dict = {}
        shard_size = 0

        # Get parameters as an iterator (lazy evaluation)
        for key, value in tree_flatten(model.parameters()):
            # Skip LoRA-related parameters
            if "lora" in key.lower():
                continue

            # Evaluate this specific weight to materialize it
            mx.eval(value)

            # Check if adding this weight would exceed shard size
            if shard_size + value.nbytes > max_file_size_bytes and shard:
                shards.append(shard)
                shard = {}
                shard_size = 0

                # Cleanup after creating a shard
                gc.collect()
                mx.clear_cache()

            shard[key] = value
            shard_size += value.nbytes

        # Don't forget the last shard
        if shard:
            shards.append(shard)

        return shards

    @staticmethod
    def _split_weights(weights: dict, max_file_size_gb: int = 2) -> list[dict]:
        max_file_size_bytes = max_file_size_gb << 30
        shards: list[dict] = []
        shard: dict = {}
        shard_size = 0
        for k, v in weights.items():
            if shard_size + v.nbytes > max_file_size_bytes and shard:
                shards.append(shard)
                shard, shard_size = {}, 0
            shard[k] = v
            shard_size += v.nbytes
        if shard:  # Don't append empty shard
            shards.append(shard)
        return shards
