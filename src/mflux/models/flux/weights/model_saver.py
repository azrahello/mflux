import json
from pathlib import Path

import mlx.core as mx
from mlx import nn
from mlx.utils import tree_flatten
from transformers import CLIPTokenizer, T5Tokenizer

from mflux.utils.version_util import VersionUtil


class ModelSaver:
    @staticmethod
    def save_model(model, bits: int, base_path: str):
        import gc

        # Save the tokenizers
        ModelSaver._save_tokenizer(base_path, model.clip_tokenizer.tokenizer, "tokenizer")
        ModelSaver._save_tokenizer(base_path, model.t5_tokenizer.tokenizer, "tokenizer_2")

        # Save the models ONE AT A TIME with aggressive memory cleanup between
        print("\n💾 Saving VAE...")
        ModelSaver.save_weights(base_path, bits, model.vae, "vae")
        mx.clear_cache()
        gc.collect()

        print("\n💾 Saving Transformer (this will take a while)...")
        ModelSaver.save_weights(base_path, bits, model.transformer, "transformer")
        mx.clear_cache()
        gc.collect()

        print("\n💾 Saving CLIP Text Encoder...")
        ModelSaver.save_weights(base_path, bits, model.clip_text_encoder, "text_encoder")
        mx.clear_cache()
        gc.collect()

        print("\n💾 Saving T5 Text Encoder...")
        ModelSaver.save_weights(base_path, bits, model.t5_text_encoder, "text_encoder_2")
        mx.clear_cache()
        gc.collect()

    @staticmethod
    def _save_tokenizer(base_path: str, tokenizer: CLIPTokenizer | T5Tokenizer, subdir: str):
        path = Path(base_path) / subdir
        path.mkdir(parents=True, exist_ok=True)
        tokenizer.save_pretrained(path)

    @staticmethod
    def save_weights(base_path: str, bits: int, model: nn.Module, subdir: str):
        path = Path(base_path) / subdir
        path.mkdir(parents=True, exist_ok=True)

        # Get model parameters
        weights_dict = dict(tree_flatten(model.parameters()))

        # Split weights into shards FIRST
        weights_shards = ModelSaver._split_weights(base_path, weights_dict)

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
        if len(weights_shards) > 1 or subdir in ["transformer", "text_encoder", "text_encoder_2"]:
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
