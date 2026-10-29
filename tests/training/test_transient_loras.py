import mlx.core as mx
from mlx import nn

from mflux.models.common.lora.layer.fused_linear_lora_layer import FusedLoRALinear
from mflux.models.common.lora.layer.linear_lora_layer import LoRALinear
from mflux.models.common.training.utils import TrainingUtil


class _Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = [nn.Linear(4, 4), nn.Linear(4, 4)]
        self.table = mx.zeros((2, 4))


def test_transient_loras_restores_modules_and_raw_params():
    model = _Tiny()
    train = LoRALinear.from_linear(model.blocks[0], r=2)
    model.blocks[0] = train
    plain = model.blocks[1]
    table = model.table

    with TrainingUtil.transient_loras(model):
        # What the loader does: fuse over the train LoRA, wrap a plain linear, bake a raw tensor.
        model.blocks[0] = FusedLoRALinear(
            base_linear=train.linear, loras=[train, LoRALinear.from_linear(train.linear, r=2)]
        )
        model.blocks[1] = LoRALinear.from_linear(plain, r=2)
        model.table = table + 1.0

    assert model.blocks[0] is train
    assert model.blocks[1] is plain
    assert model.table is table
