import mlx.core as mx
import pytest
from mlx import nn

from mflux.models.common.lora.layer.fused_linear_lora_layer import FusedLoRALinear
from mflux.models.common.lora.layer.linear_lokr_layer import LoKrLinear
from mflux.models.common.lora.layer.linear_lora_layer import LoRALinear

DTYPES = [mx.float16, mx.bfloat16, mx.float32]


def _base(dtype, dims=64):
    linear = nn.Linear(dims, dims, bias=False)
    linear.weight = linear.weight.astype(dtype)
    return linear


# The adapter factors are float32 master weights, but every forward must come out in the
# activation dtype. Regression guard: an un-cast factor promotes the output to float32,
# and since patched layers feed each other it drags the whole transformer to float32,
# silently cancelling --dtype (measured ~20% slower per step on fp16).


@pytest.mark.parametrize("dtype", DTYPES)
def test_lora_keeps_activation_dtype(dtype):
    layer = LoRALinear.from_linear(_base(dtype), r=8)
    assert layer.lora_A.dtype == mx.float32
    assert layer.lora_B.dtype == mx.float32
    assert layer(mx.zeros((1, 4, 64), dtype)).dtype == dtype


@pytest.mark.parametrize("dtype", DTYPES)
def test_fused_lora_keeps_activation_dtype(dtype):
    base = _base(dtype)
    loras = [LoRALinear.from_linear(base, r=8), LoRALinear.from_linear(base, r=4)]
    fused = FusedLoRALinear(base_linear=base, loras=loras)
    assert fused(mx.zeros((1, 4, 64), dtype)).dtype == dtype


@pytest.mark.parametrize("dtype", DTYPES)
def test_lokr_keeps_activation_dtype(dtype):
    # kron((8,8), (8,8)) -> the (64, 64) delta the base layer expects
    layer = LoKrLinear.from_linear(
        _base(dtype),
        lokr_w1=mx.random.normal((8, 8)) * 0.01,
        lokr_w2=mx.random.normal((8, 8)) * 0.01,
    )
    assert layer.lokr_w1.dtype == mx.float32
    assert layer(mx.zeros((1, 4, 64), dtype)).dtype == dtype
