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
        baked_count = LoRABaker.bake_loras_inplace(model, bits=bits)
        if baked_count > 0:
            if bits is not None:
                print(f"  🔥 Baked {baked_count} LoRA layer(s) into base weights (quantized to {bits}-bit)")
            else:
                print(f"  🔥 Baked {baked_count} LoRA layer(s) into base weights")

            # Verify no LoRA layers remain
            from mflux.models.common.lora.layer.fused_linear_lora_layer import FusedLoRALinear
            from mflux.models.common.lora.layer.linear_lora_layer import LoRALinear

            remaining_loras = sum(1 for _, m in model.named_modules() if isinstance(m, (LoRALinear, FusedLoRALinear)))
            if remaining_loras > 0:
                print(f"  ⚠️  Warning: {remaining_loras} LoRA layer(s) still present after baking")

        # Aggressive cleanup after baking
        mx.clear_cache()
        import gc

        gc.collect()

        # Force re-evaluation of the model's parameter tree
        # This ensures tree_flatten only sees the current state
        mx.eval(model.parameters())

        # Collect weights, filtering out any LoRA-related parameters
        all_weights = dict(tree_flatten(model.parameters()))
        weights = {k: v for k, v in all_weights.items() if "lora" not in k.lower()}

        shards = ModelSaver._split_weights(weights)

        # Build weight_map for index.json (maps each weight key to its shard file)
        weight_map = {}
        shard_iter = tqdm(enumerate(shards), total=len(shards), desc=f"  {subdir}", unit="shard", leave=False)
        for i, shard in shard_iter:
            shard_filename = f"{i}.safetensors"
            mx.save_safetensors(
                str(path / shard_filename),
                shard,
                {
                    "quantization_level": str(bits),
                    "mflux_version": VersionUtil.get_mflux_version(),
                },
            )
            # Record which file each weight belongs to
            for key in shard.keys():
                weight_map[key] = shard_filename

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
