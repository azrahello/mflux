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
        """
        baked_count = 0
        lora_layers = []

        # First pass: collect all LoRA layers
        for name, submodule in module.named_modules():
            if isinstance(submodule, (LoRALinear, FusedLoRALinear)):
                lora_layers.append((name, submodule))

        # Second pass: bake and replace
        for name, submodule in lora_layers:
            # Get the parent module and attribute name
            parent, attr_name = LoRABaker._get_parent_and_attr(module, name)
            if parent is None:
                continue

            # Bake the LoRA layer
            baked_linear = LoRABaker._bake_single_lora_layer(submodule)

            # Replace the LoRA layer with the baked linear layer
            if attr_name.isdigit():
                # Parent is a list/ModuleList, use indexing
                parent[int(attr_name)] = baked_linear
            else:
                # Parent is a regular module, use setattr
                setattr(parent, attr_name, baked_linear)
            baked_count += 1

        return baked_count

    @staticmethod
    def _bake_single_lora_layer(lora_layer: LoRALinear | FusedLoRALinear) -> nn.Linear:
        """
        Bake a single LoRA layer into a regular Linear layer

        Args:
            lora_layer: LoRALinear or FusedLoRALinear layer to bake

        Returns:
            Regular nn.Linear layer with merged weights
        """
        if isinstance(lora_layer, FusedLoRALinear):
            # FusedLoRALinear: merge all LoRAs into the base linear
            base_linear = lora_layer.base_linear
            base_weight = base_linear.weight

            # Start with base weights
            merged_weight = base_weight

            # Add each LoRA's contribution: W += scale * (lora_A @ lora_B).T
            # Note: lora_A is (input_dims, r), lora_B is (r, output_dims)
            # So (lora_A @ lora_B) is (input_dims, output_dims)
            # We transpose to get (output_dims, input_dims) to match weight shape
            for lora in lora_layer.loras:
                lora_delta = mx.matmul(lora.lora_A, lora.lora_B).T
                merged_weight = merged_weight + lora.scale * lora_delta

            # Create new linear layer with merged weights
            output_dims, input_dims = merged_weight.shape
            baked = nn.Linear(input_dims, output_dims, bias=hasattr(base_linear, "bias"))
            baked.weight = merged_weight
            if hasattr(base_linear, "bias"):
                baked.bias = base_linear.bias

            return baked

        elif isinstance(lora_layer, LoRALinear):
            # LoRALinear: merge single LoRA into the base linear
            base_linear = lora_layer.linear
            base_weight = base_linear.weight

            # Compute LoRA contribution: scale * (lora_A @ lora_B).T
            # Note: lora_A is (input_dims, r), lora_B is (r, output_dims)
            # So (lora_A @ lora_B) is (input_dims, output_dims)
            # We transpose to get (output_dims, input_dims) to match weight shape
            lora_delta = mx.matmul(lora_layer.lora_A, lora_layer.lora_B).T
            merged_weight = base_weight + lora_layer.scale * lora_delta

            # Create new linear layer with merged weights
            output_dims, input_dims = merged_weight.shape
            baked = nn.Linear(input_dims, output_dims, bias=hasattr(base_linear, "bias"))
            baked.weight = merged_weight
            if hasattr(base_linear, "bias"):
                baked.bias = base_linear.bias

            return baked

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
