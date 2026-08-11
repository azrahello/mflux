import math

import mlx.core as mx
from mlx import nn


class LoRALinear(nn.Module):
    @staticmethod
    def from_linear(
        linear: nn.Linear | nn.QuantizedLinear,
        r: int = 16,
        scale: float = 1.0,
    ):
        output_dims, input_dims = linear.weight.shape
        if isinstance(linear, nn.QuantizedLinear):
            input_dims *= 32 // linear.bits
        lora_lin = LoRALinear(
            input_dims=input_dims,
            output_dims=output_dims,
            r=r,
            scale=scale,
        )
        lora_lin.linear = linear
        return lora_lin

    def __init__(
        self,
        input_dims: int,
        output_dims: int,
        r: int = 8,
        scale: float = 1.0,
        bias: bool = False,
    ):
        super().__init__()
        self.linear = nn.Linear(input_dims, output_dims, bias=bias)
        self.scale = scale
        scale = 1 / math.sqrt(input_dims)

        self.lora_A = mx.random.uniform(
            low=-scale,
            high=scale,
            shape=(input_dims, r),
        )
        # B starts at zero so ΔW = A @ B is zero at init and training begins from the exact
        # base model. Random B injects a rank-r perturbation into every patched layer that the
        # optimizer has to undo first — it grows with rank and compounds across blocks.
        # (Loading a saved adapter overwrites both factors, so this only affects fresh training.)
        self.lora_B = mx.zeros((r, output_dims))

    def __call__(self, x):
        base_out = self.linear(x)
        # The factors stay float32 as master weights (fp16 LoRA params underflow the
        # gradient at typical learning rates), but the matmuls must run in the activation
        # dtype: fp16 @ fp32 promotes to fp32, and since every patched layer feeds the
        # next block, one un-cast factor drags the whole transformer to fp32 and silently
        # cancels --dtype. Casting (in_dims, r) is free next to the base GEMM.
        lora_out = mx.matmul(mx.matmul(x, self.lora_A.astype(x.dtype)), self.lora_B.astype(x.dtype))
        return base_out + self.scale * lora_out
