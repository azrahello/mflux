"""
LoRA Baking Utility

This module provides functionality to permanently merge (bake) LoRA weights into base model weights.
When LoRAs are baked, the resulting model has no LoRA overhead and can be saved/loaded as a standard model.
"""

import mlx.core as mx
from mlx import nn

from mflux.models.common.lora.layer.fused_linear_lora_layer import FusedLoRALinear
from mflux.models.common.lora.layer.linear_lora_layer import LoRALinear


class LoRABaker:
    """Utility class for baking LoRA weights into base model weights"""

    @staticmethod
    def bake_loras_inplace(module: nn.Module) -> int:
        """
        Recursively bake all LoRA layers in a module by replacing them with regular Linear layers
        containing the merged weights.

        Args:
            module: The module to bake LoRAs in (modified in-place)

        Returns:
            Number of LoRA layers that were baked

        Formula:
            W_merged = W_base + scale * (lora_A @ lora_B).T

        Note:
            After baking, the module should be quantized separately using nn.quantize()
            if quantization is desired.
        """
        baked_count = 0
        lora_layers = []

        # First pass: collect all LoRA layers
        for name, submodule in module.named_modules():
            if isinstance(submodule, (LoRALinear, FusedLoRALinear)):
                lora_layers.append((name, submodule))

        # Second pass: bake and replace with memory cleanup
        for i, (name, submodule) in enumerate(lora_layers):
            # Get the parent module and attribute name
            parent, attr_name = LoRABaker._get_parent_and_attr(module, name)
            if parent is None:
                continue

            # Bake the LoRA layer
            baked_linear = LoRABaker._bake_single_lora_layer(submodule)

            # Explicitly clear references to old LoRA layers to prevent memory leaks
            if isinstance(submodule, FusedLoRALinear):
                # Clear references to prevent tree_flatten from picking them up
                submodule.loras = []
                submodule.base_linear = None
            elif isinstance(submodule, LoRALinear):
                submodule.lora_A = None
                submodule.lora_B = None
                submodule.linear = None

            # Replace the LoRA layer with the baked linear layer
            if attr_name.isdigit():
                # Parent is a list/ModuleList, use indexing
                parent[int(attr_name)] = baked_linear
            else:
                # Parent is a regular module, use setattr
                setattr(parent, attr_name, baked_linear)
            baked_count += 1

            # Delete the old submodule reference
            del submodule

            # Aggressive periodic cleanup to reduce memory pressure
            if (i + 1) % 10 == 0:
                import gc

                mx.eval(baked_linear.parameters())
                mx.clear_cache()
                gc.collect()
                print(f"  Baked {i + 1}/{len(lora_layers)} layers...")

        # Final cleanup after all baking
        import gc

        gc.collect()
        mx.clear_cache()

        return baked_count

    @staticmethod
    def _bake_single_lora_layer(lora_layer: LoRALinear | FusedLoRALinear) -> nn.Linear:
        """
        Bake a single LoRA layer into a regular Linear layer

        Args:
            lora_layer: LoRALinear or FusedLoRALinear layer to bake

        Returns:
            nn.Linear layer with merged weights
        """
        if isinstance(lora_layer, FusedLoRALinear):
            # FusedLoRALinear: merge all LoRAs into the base linear IN-PLACE
            base_linear = lora_layer.base_linear

            # Get base weight - dequantize if needed
            if isinstance(base_linear, nn.QuantizedLinear):
                # Dequantize to full precision for baking
                base_weight = mx.dequantize(
                    base_linear.weight, base_linear.scales, base_linear.biases, base_linear.group_size, base_linear.bits
                )
                original_dtype = mx.float16
            else:
                base_weight = base_linear.weight
                original_dtype = base_weight.dtype

            # Calculate merged update from all LoRAs
            merged_update = mx.zeros_like(base_weight)
            for lora in lora_layer.loras:
                # W += scale * (lora_A @ lora_B).T
                merged_update = merged_update + lora.scale * mx.matmul(lora.lora_A, lora.lora_B).T

            # Bake into base weight
            merged_weight = (base_weight + merged_update).astype(original_dtype)

            # Create new Linear layer with merged weights
            output_dims, input_dims = merged_weight.shape
            new_linear = nn.Linear(input_dims, output_dims, bias=hasattr(base_linear, "bias"))
            new_linear.weight = merged_weight
            if hasattr(base_linear, "bias") and base_linear.bias is not None:
                new_linear.bias = base_linear.bias

            # Force evaluation to free computation graph
            mx.eval(new_linear.parameters())

            return new_linear

        elif isinstance(lora_layer, LoRALinear):
            # LoRALinear: merge single LoRA into the base linear
            base_linear = lora_layer.linear

            # Get base weight - dequantize if needed
            if isinstance(base_linear, nn.QuantizedLinear):
                # Dequantize to full precision for baking
                base_weight = mx.dequantize(
                    base_linear.weight, base_linear.scales, base_linear.biases, base_linear.group_size, base_linear.bits
                )
                original_dtype = mx.float16
            else:
                base_weight = base_linear.weight
                original_dtype = base_weight.dtype

            # W += scale * (lora_A @ lora_B).T
            lora_delta = mx.matmul(lora_layer.lora_A, lora_layer.lora_B).T
            merged_weight = (base_weight + lora_layer.scale * lora_delta).astype(original_dtype)

            # Create new Linear layer with merged weights
            output_dims, input_dims = merged_weight.shape
            new_linear = nn.Linear(input_dims, output_dims, bias=hasattr(base_linear, "bias"))
            new_linear.weight = merged_weight
            if hasattr(base_linear, "bias") and base_linear.bias is not None:
                new_linear.bias = base_linear.bias

            # Force evaluation to free computation graph
            mx.eval(new_linear.parameters())

            return new_linear

        else:
            raise TypeError(f"Unsupported layer type: {type(lora_layer)}")

    @staticmethod
    def _get_parent_and_attr(module: nn.Module, full_path: str) -> tuple[nn.Module | None, str | None]:
        """
        Get the parent module and attribute name from a full path

        Args:
            module: Root module
            full_path: Full path to the target module (e.g., "transformer.blocks.0.attn.qkv")

        Returns:
            Tuple of (parent_module, attribute_name) or (None, None) if not found
        """
        if not full_path:
            return None, None

        parts = full_path.split(".")
        if len(parts) == 1:
            return module, parts[0]

        # Navigate to parent
        current = module
        for part in parts[:-1]:
            if part.isdigit():
                current = current[int(part)]
            else:
                current = getattr(current, part, None)
                if current is None:
                    return None, None

        return current, parts[-1]
