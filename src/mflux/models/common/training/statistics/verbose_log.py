from __future__ import annotations

import re
from collections import defaultdict

import mlx.core as mx
from mlx import nn
from mlx.utils import tree_flatten
from tqdm import tqdm

from mflux.models.common.lora.layer.fused_linear_lora_layer import FusedLoRALinear
from mflux.models.common.lora.layer.linear_lora_layer import LoRALinear

_GB = 1e9
_PROBES: dict[int, tuple] = {}
# "blocks.7.attn.wq" -> "blocks.{block}.attn.wq", the config's module_path. Keeping the prefix
# stops targets that share a suffix (blocks.*.attn.wv vs txtfusion.*.attn.wv) from merging.
_BLOCK_INDEX = re.compile(r"\.\d+(?=\.|$)")


def _target(name: str) -> str:
    return _BLOCK_INDEX.sub(".{block}", name, count=1)


class VerboseLog:
    """Diagnostics printed by `mflux-train -v`. Everything per step reads values the step has
    already evaluated, so the only extra GPU work is the perturbation report, which runs at
    checkpoint cadence."""

    @staticmethod
    def write(msg: str) -> None:
        tqdm.write(msg)  # keeps the progress bar intact

    @staticmethod
    def train_loras(transformer) -> list[tuple[str, LoRALinear, nn.Module]]:
        # (layer path, train LoRA, module that runs it). named_modules also yields the LoRAs
        # inside a FusedLoRALinear as "<path>.loras.N"; those are reached through the fused
        # parent instead, or every fused layer would be counted twice.
        found = []
        for name, child in transformer.named_modules():
            if isinstance(child, FusedLoRALinear):
                found += [(name, lora, child) for lora in child.loras if getattr(lora, "_mflux_lora_role", None) == "train"]  # fmt: off
            elif isinstance(child, LoRALinear) and ".loras." not in name:
                if getattr(child, "_mflux_lora_role", None) == "train":
                    found.append((name, child, child))
        return found

    @staticmethod
    def probe_activation_dtypes(transformer) -> dict[str, mx.Dtype]:
        # The adapter casts the DiT output back to the latent dtype, so the prediction says
        # nothing about the activations. Record instead the output dtype of every patched
        # layer on its next call, then put the layer back: an un-cast fp32 factor shows up
        # here as the layer where fp32 starts.
        seen: dict[str, mx.Dtype] = {}
        for name, _, owner in VerboseLog.train_loras(transformer):
            if isinstance(owner, _Probe):
                continue
            original = owner.__class__
            owner.__class__ = type("Probe" + original.__name__, (_Probe, original), {})
            # Kept off the module: an nn.Module attribute would join its parameter tree.
            _PROBES[id(owner)] = (name, original, seen)
        return seen

    @staticmethod
    def activation_dtypes(seen: dict[str, mx.Dtype]) -> None:
        by_dtype: dict[mx.Dtype, list[str]] = defaultdict(list)
        for name, dtype in seen.items():
            by_dtype[dtype].append(name)
        summary = ", ".join(f"{dtype}: {len(names)} layers" for dtype, names in by_dtype.items())
        VerboseLog.write(f"activation dtype (patched layer outputs) {summary or 'not observed'}")
        if len(by_dtype) > 1:
            for dtype, names in by_dtype.items():
                VerboseLog.write(f"  {dtype}: first {names[0]}")

    @staticmethod
    def startup(*, adapter, training_spec, base_config, gradient_checkpointing: bool) -> None:
        loras = VerboseLog.train_loras(adapter.transformer())
        per_target: dict[str, list[int]] = defaultdict(list)
        for name, lora, _ in loras:
            per_target[_target(name)].append(lora.lora_A.shape[1])
        trainable = sum(v.size for _, v in tree_flatten(adapter.model().trainable_parameters()))
        opt = training_spec.optimizer
        accum = max(1, opt.gradient_accumulation_steps)
        sigmas = base_config.scheduler.sigmas
        w = VerboseLog.write
        w("── verbose: training setup ──")
        w(f"model={training_spec.model} dtype={base_config.precision} quantize={training_spec.quantize}")
        w(f"gradient_checkpointing={gradient_checkpointing} low_ram={training_spec.low_ram} cache_limit_gb={training_spec.cache_limit_gb}")  # fmt: off
        w(f"LoRA layers={len(loras)} trainable params={trainable:,}")
        for target, ranks in sorted(per_target.items()):
            w(f"  {target}: {len(ranks)} layers, rank {sorted(set(ranks))}")
        if loras:
            lora = loras[0][1]  # fp32 is expected: master weights, cast at the matmul
            w(f"LoRA factor dtype={lora.lora_A.dtype} scale={lora.scale}")
        w(f"batch={training_spec.training_loop.batch_size} x accum={accum} = effective {training_spec.training_loop.batch_size * accum}")  # fmt: off
        w(f"optimizer={opt.name} lr={opt.learning_rate} schedule={opt.lr_schedule} warmup={opt.lr_warmup_steps} max_grad_norm={opt.max_grad_norm}")  # fmt: off
        loop = training_spec.training_loop
        w(f"timesteps={loop.timestep_type or 'uniform'} [{loop.timestep_low}, {loop.timestep_high or len(sigmas) - 1}) of {len(sigmas) - 1}")  # fmt: off
        w(f"sigma shift={base_config.model_config.requires_sigma_shift} sigmas at preview size (training shifts per image)={[round(float(s), 3) for s in sigmas.tolist()]}")  # fmt: off

    @staticmethod
    def samples(batch, name: str | None = None) -> str:
        return " ".join(f"t={t}/σ={float(s):.3f}" for n, t, s in batch.samples if name in (None, n))

    @staticmethod
    def step(*, step: int, loss: float, batch, grad_norm, lr, timings: dict[str, float]) -> None:
        parts = [f"step {step}", f"loss={loss:.5f}", VerboseLog.samples(batch)]
        if grad_norm is not None:
            parts.append(f"grad_norm={float(grad_norm):.4g}")
        if lr is not None:
            parts.append(f"lr={float(lr):.3g}")
        parts.append(
            f"mem active={mx.get_active_memory() / _GB:.2f} peak={mx.get_peak_memory() / _GB:.2f} "
            f"cache={mx.get_cache_memory() / _GB:.2f} GB"
        )
        parts.append(" ".join(f"{k}={v:.2f}s" for k, v in timings.items()))
        VerboseLog.write(" | ".join(parts))

    @staticmethod
    def nonfinite(*, step: int, batch) -> None:
        items = ", ".join(
            f"{name} ({VerboseLog.samples(batch, name)})" for name in dict.fromkeys(s[0] for s in batch.samples)
        )
        VerboseLog.write(f"⚠ non-finite loss at step {step}, skipped: {items}")

    @staticmethod
    def perturbation(*, step: int, transformer, top: int = 5) -> None:
        # ||scale·A@B||_F without materialising A@B: ||AB||² = sum((AᵀA) ⊙ (BBᵀ)), both r×r.
        rows = []
        for name, lora, _ in VerboseLog.train_loras(transformer):
            a, b = lora.lora_A, lora.lora_B
            delta = abs(lora.scale) * mx.sqrt(mx.maximum(mx.sum((a.T @ a) * (b @ b.T)), 0.0))
            base = lora.linear.weight if type(lora.linear) is nn.Linear else None
            rows.append((name, delta, None if base is None else mx.linalg.norm(base.astype(mx.float32))))
        if not rows:
            return
        mx.eval([r[1] for r in rows] + [r[2] for r in rows if r[2] is not None])
        values = [(name, delta.item(), None if base is None else base.item()) for name, delta, base in rows]
        total = sum(d**2 for _, d, _ in values) or 1.0
        per_target: dict[str, float] = defaultdict(float)
        for name, d, _ in values:
            per_target[_target(name)] += d**2
        w = VerboseLog.write
        w(f"── verbose: LoRA perturbation at step {step} (share of total ||ΔW||²) ──")
        for target, sq in sorted(per_target.items(), key=lambda kv: -kv[1]):
            w(f"  {target}: {100 * sq / total:.1f}%")
        w(f"  top {top} layers:")
        for name, d, base in sorted(values, key=lambda v: -v[1])[:top]:
            ratio = f" ||ΔW||/||W||={d / base:.2e}" if base else ""
            w(f"    {name}: ||ΔW||={d:.4g}{ratio}")


class _Probe:
    def __call__(self, x):
        out = super().__call__(x)
        name, original, seen = _PROBES.pop(id(self))
        seen[name] = out.dtype
        self.__class__ = original
        return out
