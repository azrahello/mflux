#!/usr/bin/env python3
"""Numerical parity check: this MLX PidNet against NVIDIA's PyTorch reference, on CPU.

`validate_pid_decoder.py` compares end images against NVIDIA's `from_clean.py`, which needs
CUDA -- so it was never actually run, and the port shipped with no ground truth at all. This
runs the reference network itself on CPU instead, which needs nothing but torch, and compares
tensors rather than pixels.

That distinction matters: a wrong RoPE reference grid (fixed 2026-07-26) left images looking
perfectly plausible while diverging from the reference by 2.3e-01 relative. Only a tensor
comparison catches that class of bug.

Usage:

    git clone https://github.com/nv-tlabs/PiD /somewhere/PiD
    python scripts/validate_pid_reference.py --pid-repo /somewhere/PiD
    python scripts/validate_pid_reference.py --pid-repo /somewhere/PiD --layers

Expect a relative error around 1e-04. That is float32 accumulation across 1.4B parameters in
two frameworks, not a defect. Anything at 1e-02 or worse is a real divergence -- rerun with
--layers, which reports per-module error so the first diverging layer is the culprit and
everything after it is downstream noise.

The reference is NOT vendored: it is NVIDIA code under NSCLv1 (non-commercial). Point
--pid-repo at your own clone. Its two `imaginaire` imports are stubbed in-process here, so
the clone stays untouched.
"""

import argparse
import sys
import types
from pathlib import Path

import mlx.core as mx
import numpy as np
from mlx.utils import tree_unflatten

from mflux.models.common.pid_decoder.pid_decoder import PID_CHECKPOINT_VARIANTS
from mflux.models.common.pid_decoder.pid_weight_mapping import convert_checkpoint
from mflux.models.common.pid_decoder.pixdit.pixdit_network import PidNet as MlxPidNet

# The production config, mirrored from PidDecoder.from_pretrained. Both sides get this, so a
# drift there is exactly what we want the comparison to expose.
# Every key here must be one BOTH constructors accept, or the two sides silently fall back to
# their own defaults and the comparison certifies a config the checkpoint was never trained
# with -- which is exactly how rope_ref_h stayed at 1024 (grid 64) while PID_SR4X_V1PT5 sets
# 2048 (grid 128), passing at 9.1e-05 the whole time.
CONFIG = dict(
    rope_ref_h=2048, rope_ref_w=2048,
    patch_depth=14, pixel_depth=2, hidden_size=1536, pixel_hidden_size=16,
    pixel_attn_hidden_size=1152, num_groups=24, pixel_num_groups=16, patch_size=16,
    lq_gate_type="sigma_aware_per_token", lq_interval=2, lq_hidden_dim=1024,
    lq_conv_padding_mode="replicate", pit_lq_inject=True, sr_scale=4,
    latent_spatial_down_factor=8, txt_embed_dim=2304, txt_max_length=300,
    use_text_rope=True, lq_num_res_blocks=4,
)  # fmt: skip


def _stub_imaginaire() -> None:
    """The reference imports two helpers from its vendored `imaginaire`: a logger and a rank
    query. Both are no-ops at rank 0, so stub them rather than pulling in the dependency."""
    import importlib

    for name, attrs in (
        ("pid._ext.imaginaire.utils.log", {"info": lambda *a, **k: None}),
        ("pid._ext.imaginaire.utils.distributed", {"get_rank": lambda *a, **k: 0}),
    ):
        parent_name = name.rsplit(".", 1)[0]
        # Import the real parent packages from the clone -- creating our own would shadow the
        # clone's `pid` package and break every other import from it.
        parent = importlib.import_module(parent_name)
        mod = types.ModuleType(name)
        for k, v in attrs.items():
            setattr(mod, k, v)
        sys.modules[name] = mod
        setattr(parent, name.rsplit(".", 1)[1], mod)


def resolve_checkpoint(variant: str, explicit: str | None) -> str:
    if explicit:
        return explicit
    from huggingface_hub import hf_hub_download

    return hf_hub_download(repo_id="nvidia/PiD", filename=PID_CHECKPOINT_VARIANTS[variant][0])


def build_reference(pid_repo: Path, checkpoint: str, lq_latent_channels: int):
    import torch

    sys.path.insert(0, str(pid_repo))
    _stub_imaginaire()
    from pid._src.networks.pid_net import PidNet as RefPidNet

    # lq_in_channels=0: the released checkpoints are latent-only, and the reference's default
    # of 3 would build an unused RGB branch that no key in the checkpoint fills.
    net = RefPidNet(**CONFIG, lq_in_channels=0, lq_latent_channels=lq_latent_channels).eval()
    sd = torch.load(checkpoint, map_location="cpu", mmap=True, weights_only=True)
    net.load_state_dict({k.removeprefix("net."): v.float() for k, v in sd.items()}, strict=True)
    return net


def build_ours(checkpoint: str, lq_latent_channels: int) -> MlxPidNet:
    net = MlxPidNet(**CONFIG, lq_latent_channels=lq_latent_channels)
    weights = {k.removeprefix("pid_net."): v.astype(mx.float32) for k, v in convert_checkpoint(checkpoint).items()}
    net.update(tree_unflatten(list(weights.items())), strict=False)
    return net


def make_inputs(size: int, lq_latent_channels: int) -> dict[str, np.ndarray]:
    # Fixed, not sampled: the two frameworks must see bit-identical inputs.
    rng = np.random.default_rng(0)
    zh = size // (CONFIG["sr_scale"] * CONFIG["latent_spatial_down_factor"])
    return {
        "x": rng.standard_normal((1, 3, size, size)).astype(np.float32),
        "t": np.array([500.0], np.float32),
        "caption": (rng.standard_normal((1, CONFIG["txt_max_length"], CONFIG["txt_embed_dim"])) * 0.1).astype(
            np.float32
        ),
        "lq": rng.standard_normal((1, lq_latent_channels, zh, zh)).astype(np.float32),
        "sigma": np.array([0.0], np.float32),
    }


def report(name: str, a: np.ndarray, b: np.ndarray) -> float:
    if a.shape != b.shape:
        print(f"  {name:<24} SHAPE reference={a.shape} ours={b.shape}  <-- DIVERGES")
        return float("inf")
    rel = float(np.abs(a - b).max() / max(np.abs(a).max(), 1e-9))
    print(f"  {name:<24} {str(a.shape):<22} {rel:.3e}  {'<-- DIVERGES' if rel > 1e-3 else 'ok'}")
    return rel


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pid-repo", required=True, type=Path, help="Clone of https://github.com/nv-tlabs/PiD")
    p.add_argument("--variant", default="flux", choices=sorted(PID_CHECKPOINT_VARIANTS))
    p.add_argument("--checkpoint", default=None, help="Path to model_ema_bf16.pth (default: fetch from HF cache)")
    p.add_argument("--size", type=int, default=256, help="Output size; 256 keeps a CPU forward to ~minutes")
    p.add_argument("--layers", action="store_true", help="Also report per-module error to locate the first divergence")
    p.add_argument("--tolerance", type=float, default=1e-3, help="Relative error above which to fail")
    args = p.parse_args()

    if not (args.pid_repo / "pid" / "_src" / "networks" / "pid_net.py").is_file():
        p.error(f"{args.pid_repo} does not look like a PiD clone (pid/_src/networks/pid_net.py not found)")

    lq_channels = PID_CHECKPOINT_VARIANTS[args.variant][1]
    checkpoint = resolve_checkpoint(args.variant, args.checkpoint)
    inputs = make_inputs(args.size, lq_channels)
    print(f"variant={args.variant}  lq_latent_channels={lq_channels}  size={args.size}\ncheckpoint={checkpoint}\n")

    import torch

    ref = build_reference(args.pid_repo, checkpoint, lq_channels)
    grabbed: dict[str, np.ndarray] = {}
    if args.layers:

        def hook(name):
            def fn(_m, _i, out):
                first = out[0] if isinstance(out, (tuple, list)) else out
                if torch.is_tensor(first):
                    grabbed[name] = first.detach().numpy()

            return fn

        for name, mod in _probe_points(ref):
            mod.register_forward_hook(hook(name))

    with torch.no_grad():
        out_ref = ref(
            torch.from_numpy(inputs["x"]), torch.from_numpy(inputs["t"]), torch.from_numpy(inputs["caption"]),
            lq_latent=torch.from_numpy(inputs["lq"]), degrade_sigma=torch.from_numpy(inputs["sigma"]),
        )  # fmt: skip
    out_ref = (out_ref[0] if isinstance(out_ref, tuple) else out_ref).numpy()
    del ref

    ours = build_ours(checkpoint, lq_channels)
    mine: dict[str, np.ndarray] = {}
    if args.layers:
        _instrument(ours, mine)
    out_ours = ours(
        mx.array(inputs["x"]), mx.array(inputs["t"]), mx.array(inputs["caption"]),
        mx.array(inputs["lq"]), mx.array(inputs["sigma"]),
    )  # fmt: skip
    mx.eval(out_ours)
    out_ours = np.array(out_ours, copy=False)

    if args.layers:
        print("per-module relative error (first divergence is the culprit; later ones inherit it):")
        for name in ("pixel_embedder", "patch_block[0]", "pixel_block[0]", "final_layer"):
            if name in grabbed and name in mine:
                report(name, grabbed[name], mine[name])
        print()

    print("end to end:")
    rel = report("output", out_ref, out_ours)
    corr = float(np.corrcoef(out_ref.ravel(), out_ours.ravel())[0, 1])
    print(f"  correlation {corr:.6f}")
    if rel > args.tolerance:
        print(f"\nFAIL: relative error {rel:.3e} exceeds {args.tolerance:.0e}. Rerun with --layers.")
        return 1
    print(f"\nPASS: within {args.tolerance:.0e}.")
    return 0


def _probe_points(net):
    """Same attribute names on both sides, so one accessor serves reference and port alike."""
    return [
        ("pixel_embedder", net.pixel_embedder),
        ("patch_block[0]", net.patch_blocks[0]),
        ("pixel_block[0]", net.pixel_blocks[0]),
        ("final_layer", net.final_layer),
    ]


def _instrument(net: MlxPidNet, sink: dict[str, np.ndarray]) -> None:
    """MLX resolves __call__ on the class, so patch there and key by instance identity."""
    wanted = {id(mod): name for name, mod in _probe_points(net)}
    for cls in {type(mod) for _, mod in _probe_points(net)}:
        original = cls.__call__

        def wrapper(original):
            def call(self, *a, **k):
                out = original(self, *a, **k)
                if id(self) in wanted:
                    first = out[0] if isinstance(out, (tuple, list)) else out
                    if isinstance(first, mx.array):
                        mx.eval(first)
                        sink[wanted[id(self)]] = np.array(first, copy=False)
                return out

            return call

        cls.__call__ = wrapper(original)


if __name__ == "__main__":
    raise SystemExit(main())
