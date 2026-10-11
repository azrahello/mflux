import mlx.core as mx
import pytest
from mlx import nn
from mlx.utils import tree_flatten

from mflux.models.common.lora.layer.linear_lora_layer import LoRALinear
from mflux.models.common.training.statistics.verbose_log import VerboseLog
from mflux.models.common.training.trainer import TrainingTrainer


class _Block(nn.Module):
    def __init__(self, lora):
        super().__init__()
        self.to_q = lora


class _Transformer(nn.Module):
    def __init__(self, lora):
        super().__init__()
        self.blocks = [_Block(lora)]


@pytest.mark.fast
def test_perturbation_norm_matches_dense_delta(monkeypatch):
    lora = LoRALinear.from_linear(nn.Linear(12, 8), r=4, scale=0.5)
    lora.lora_B = mx.random.normal((4, 8))
    lora._mflux_lora_role = "train"
    lines = []
    monkeypatch.setattr(VerboseLog, "write", lines.append)

    VerboseLog.perturbation(step=1, transformer=_Transformer(lora))

    dense = 0.5 * mx.linalg.norm(lora.lora_A @ lora.lora_B).item()
    assert "  blocks.{block}.to_q: 100.0%" in lines
    reported = float(next(line for line in lines if "blocks.0.to_q:" in line).split("||ΔW||=")[1].split()[0])
    assert reported == pytest.approx(dense, rel=1e-3)


@pytest.mark.fast
def test_grad_norm_matches_clip_grad_norm():
    from mlx.optimizers import clip_grad_norm

    grads = {"a": mx.array([3.0, 0.0]), "b": {"c": mx.array([4.0])}}
    _, expected = clip_grad_norm(grads, 1e9)
    assert TrainingTrainer._grad_norm(grads).item() == pytest.approx(expected.item())


@pytest.mark.fast
def test_activation_probe_records_layer_output_dtype_once_and_restores():
    base = nn.Linear(12, 8)
    base.set_dtype(mx.bfloat16)  # like the real model; the fp32 LoRA factors must not promote
    lora = LoRALinear.from_linear(base, r=4)
    lora._mflux_lora_role = "train"
    model = _Transformer(lora)
    params_before = len(tree_flatten(model.parameters()))

    seen = VerboseLog.probe_activation_dtypes(model)
    model.blocks[0].to_q(mx.ones((2, 12), dtype=mx.bfloat16))

    assert seen == {"blocks.0.to_q": mx.bfloat16}
    assert type(model.blocks[0].to_q) is LoRALinear
    assert len(tree_flatten(model.parameters())) == params_before
