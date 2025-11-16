import json
from pathlib import Path

import mlx.core as mx
from mlx import nn
from mlx.utils import tree_flatten
from transformers import Qwen2Tokenizer

from mflux.models.common.lora.layer.fused_linear_lora_layer import FusedLoRALinear
from mflux.models.common.lora.layer.linear_lora_layer import LoRALinear
from mflux.utils.version_util import VersionUtil


class QwenModelSaver:
    @staticmethod
    def save_model(model, bits: int, base_path: str):
        import gc

        # Save the tokenizer
        QwenModelSaver._save_tokenizer(base_path, model.qwen_tokenizer.tokenizer, "tokenizer")

        # CRITICAL: Replace LoRA layers with their base Linear layers before saving
        # This ensures we save only the base weights, not the LoRA parameters
        save_bits = bits
        if model.lora_paths:
            print(f"⚠️ Model has {len(model.lora_paths)} LoRA(s) loaded")
            print("🔥 Baking LoRA weights into the model (permanent merge)")
            QwenModelSaver._merge_loras_into_base_weights(model.transformer)
            QwenModelSaver._merge_loras_into_base_weights(model.text_encoder)
            # After baking, weights are de-quantized, so save without quantization metadata
            save_bits = None
            print("⚠️ Model will be saved in de-quantized format (bfloat16) with LoRAs baked in")

        # Save the models ONE AT A TIME with aggressive memory cleanup between
        # NOTE: Models are saved in bfloat16 precision
        # Quantization metadata is preserved for runtime loading (except when LoRAs are baked)

        print("\n💾 Saving VAE...")
        QwenModelSaver.save_weights(base_path, bits, model.vae, "vae")
        mx.clear_cache()
        gc.collect()

        print("\n💾 Saving Transformer (this will take a while)...")
        QwenModelSaver.save_weights(base_path, save_bits, model.transformer, "transformer")
        mx.clear_cache()
        gc.collect()

        print("\n💾 Saving Text Encoder...")
        QwenModelSaver.save_weights(base_path, save_bits, model.text_encoder, "text_encoder")
        mx.clear_cache()
        gc.collect()

    @staticmethod
    def _merge_loras_into_base_weights(module: nn.Module):
        """Merge LoRA weights into base Linear layers using W = W + (A @ B).T * scale.

        Merged weights are converted to bfloat16 (model's default precision).
        Memory is aggressively cleaned up after each layer to minimize peak usage.
        """
        layers_merged = 0

        for name, child in module.named_modules():
            # Skip LoRA internal subpaths
            if ".loras." in name or name.endswith(".linear") or name.endswith(".base_linear"):
                continue

            # --- LoRALinear -----------------------------------------------------
            if isinstance(child, LoRALinear):
                parent, attr = QwenModelSaver._get_parent_and_attr(module, name)
                lin = child.linear

                # W_merged = W_base + (A @ B).T * scale
                # Note: Transpose needed because weight is (out_dims, in_dims) but A@B is (in_dims, out_dims)
                update = mx.matmul(child.lora_A, child.lora_B).T * child.scale
                merged_weight = lin.weight + update

                # Convert to bfloat16 (model's default precision)
                lin.weight = merged_weight.astype(mx.bfloat16)

                # Force evaluation and cleanup memory after each layer
                mx.eval(lin.weight)
                mx.clear_cache()

                QwenModelSaver._set_module_attr(parent, attr, lin)
                layers_merged += 1

            # --- FusedLoRALinear ------------------------------------------------
            elif isinstance(child, FusedLoRALinear):
                parent, attr = QwenModelSaver._get_parent_and_attr(module, name)

                # Start with base weight instead of zeros_like to save memory
                merged_weight = child.base_linear.weight

                for lora in child.loras:
                    # Note: Transpose needed because weight is (out_dims, in_dims) but A@B is (in_dims, out_dims)
                    lora_update = lora.scale * mx.matmul(lora.lora_A, lora.lora_B).T
                    merged_weight = merged_weight + lora_update

                    # Free memory immediately after each LoRA
                    del lora_update
                    mx.clear_cache()

                # Convert to bfloat16 (model's default precision)
                child.base_linear.weight = merged_weight.astype(mx.bfloat16)

                # Force evaluation and cleanup memory after each layer
                mx.eval(child.base_linear.weight)
                mx.clear_cache()

                # Replace FusedLoRALinear with the clean base_linear to save memory
                QwenModelSaver._set_module_attr(parent, attr, child.base_linear)
                layers_merged += 1

        print(f"✅ Merged {layers_merged} LoRA layer(s) using W = W + (A @ B).T * scale")

        # Final cleanup
        mx.clear_cache()

    @staticmethod
    def _get_parent_and_attr(module: nn.Module, path: str):
        """Get parent module and attribute name from a path"""
        parts = path.split(".")
        parent = module
        for part in parts[:-1]:
            if part.isdigit():
                parent = parent[int(part)]
            else:
                parent = getattr(parent, part)

        final_attr = parts[-1]
        # Return both parent and final_attr (which might be numeric for lists)
        return parent, final_attr

    @staticmethod
    def _set_module_attr(parent, attr, value):
        """Set an attribute or list element on parent"""
        if attr.isdigit():
            parent[int(attr)] = value
        else:
            setattr(parent, attr, value)

    @staticmethod
    def _save_tokenizer(base_path: str, tokenizer: Qwen2Tokenizer, subdir: str):
        path = Path(base_path) / subdir
        tokenizer.save_pretrained(path)

    @staticmethod
    def save_weights(base_path: str, bits: int, model: nn.Module, subdir: str):
        path = Path(base_path) / subdir
        path.mkdir(parents=True, exist_ok=True)

        # Get model parameters
        weights_dict = dict(tree_flatten(model.parameters()))

        # Split weights into shards FIRST
        weights_shards = QwenModelSaver._split_weights(base_path, weights_dict)

        # Free the full weights dict immediately to reduce memory pressure
        del weights_dict
        mx.clear_cache()

        # Save each shard and track for index
        weight_map = {}
        total_size = 0

        print(f"Saving {subdir} ({len(weights_shards)} shards)...")

        for i, weight_shard in enumerate(weights_shards):
            shard_filename = f"{i}.safetensors"
            shard_path = path / shard_filename

            # NOTE: Weights are saved in full precision (bfloat16/float16)
            # Quantization metadata is included and will be applied at load time
            # This is the standard MLX approach - safetensors cannot serialize quantized arrays
            # The metadata ensures the model is quantized correctly when loaded
            if bits is not None:
                print(
                    f"  Preparing shard {i + 1}/{len(weights_shards)} (will be quantized to {bits}-bit at load time)..."
                )

            # Force evaluation and free memory BEFORE saving to reduce memory pressure
            mx.eval(list(weight_shard.values()))
            # Clear temporary allocations
            mx.clear_cache()

            # Save the shard
            print(f"  Writing shard {i + 1}/{len(weights_shards)}...")
            # usage: save_safetensors(file: str, arrays: dict[str, array], metadata: Optional[dict[str, str]] = None)
            mx.save_safetensors(
                str(shard_path),
                # arrays (dict(str, array)): The dictionary of names to arrays to be saved.
                weight_shard,
                # [save_safetensors] Metadata must be a dictionary with string keys and values
                # i.e. 'None' and other special values are string-ified and need to be parsed by readers
                {
                    "quantization_level": str(bits) if bits is not None else "None",
                    "mflux_version": VersionUtil.get_mflux_version(),
                },
            )

            # Track weights for index
            for key in weight_shard.keys():
                weight_map[key] = shard_filename

            # Calculate size
            total_size += shard_path.stat().st_size

            # Clear memory after each shard
            del weight_shard
            mx.clear_cache()

        # CRITICAL FIX: Create model.safetensors.index.json for multi-shard models
        if len(weights_shards) > 1 or subdir in ["transformer", "text_encoder"]:
            index = {
                "metadata": {
                    "total_size": total_size,
                    "quantization_level": str(bits) if bits is not None else "None",
                    "mflux_version": VersionUtil.get_mflux_version(),
                },
                "weight_map": weight_map,
            }

            index_path = path / "model.safetensors.index.json"
            with open(index_path, "w") as f:
                json.dump(index, f, indent=2)

            print(
                f"✅ Created {subdir}: {len(weight_map)} tensors, {len(weights_shards)} shards, {total_size / 1024**3:.2f} GB"
            )

    @staticmethod
    def _split_weights(base_path: str, weights: dict, max_file_size_gb: float = 0.5) -> list:
        # Use smaller shards (512MB instead of 2GB) to reduce memory peaks during save
        max_file_size_bytes = int(max_file_size_gb * (1 << 30))
        shards = []
        shard, shard_size = {}, 0
        for k, v in weights.items():
            if shard_size + v.nbytes > max_file_size_bytes:
                shards.append(shard)
                shard, shard_size = {}, 0
            shard[k] = v
            shard_size += v.nbytes
        shards.append(shard)
        return shards
