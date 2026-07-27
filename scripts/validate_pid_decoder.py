"""Task 12: end-to-end numerical validation of the PiD decoder port against NVIDIA's own reference.

Dev-only script (not shipped in `src/`, not a pytest test -- comparing against NVIDIA's
reference needs `torch` + the multi-GB nv-tlabs/PiD checkout, neither an mflux runtime
dependency). Run manually per the plan's Testing section.

Workflow (5 steps):

  1. `export` subcommand (THIS machine, Apple Silicon/MLX) -- generate a real image through
     the normal mflux CLI/model with the *standard* VAE decoder, capture the exact latent
     tensor handed to that decoder, and save it + a small JSON sidecar (prompt/seed/backbone).

        python scripts/validate_pid_decoder.py export \\
            --backbone z-image-turbo --prompt "a red fox in a snowy forest" \\
            --seed 42 --width 1024 --height 1024 --steps 8 \\
            --out-dir /tmp/pid_validate

  2. NVIDIA's own reference decode -- REQUIRES CUDA. This sandbox has no NVIDIA GPU
     (`python3 -c "import torch; print(torch.cuda.is_available())"` -> False here, checked
     2026-07-24). Run this step on a separate CUDA machine:

        git clone https://github.com/nv-tlabs/PiD /tmp/PiD-reference
        cd /tmp/PiD-reference
        # follow the repo's own Installation + Download Checkpoints steps, then:
        PYTHONPATH=. python -m pid._src.inference.from_clean \\
            --backbone qwenimage --pid_ckpt_type 2kto4k_v1pt5 \\
            --image /tmp/pid_validate/export.png \\
            --prompt "<the exact prompt printed by step 1's sidecar JSON>" \\
            --pid_inference_steps 4 --output_dir /tmp/pid_reference_output

     (Use `--backbone zimage`/whatever nv-tlabs/PiD calls the Flux/Z-Image checkpoint family
     if step 1 was run with `--backbone z-image-turbo` -- the exported --image must match
     the backbone the reference checkpoint was distilled against.) Copy the resulting PNG
     back to this machine as the `--reference` argument to `compare` below.

  3. `compare` subcommand (THIS machine) -- load the exported latent, run it through this
     port's PidDecoder.decode, and diff pixel-wise (MAE, PSNR) against the reference PNG
     from step 2.

        python scripts/validate_pid_decoder.py compare \\
            --latent /tmp/pid_validate/latent.safetensors \\
            --reference /tmp/pid_reference_output/xxx.png \\
            --out /tmp/pid_validate/mflux_decoded.png

     There is no pre-set pass/fail threshold (first real numerical check of a from-scratch
     port) -- use judgment on the printed MAE/PSNR and a side-by-side look at both PNGs. If
     the match looks wrong, pass `--dump-intermediates DIR` to also save this port's caption
     embeddings and the sampler's `x` after step 1, for bisecting against equivalent dumps
     added to NVIDIA's `from_clean.py` on the CUDA machine (see mflux-debugging's
     export-then-compare pattern).

  4. Manual visual check -- run the real CLI twice (with/without --pid-decode, same
     prompt/seed) and `Read` both PNGs. Not part of this script; see task-12-report.md for
     what was actually observed.

  5. Commit this script only (not generated images/checkpoints).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import mlx.core as mx
import numpy as np
import PIL.Image

from mflux.models.common.pid_decoder.pid_decoder import PID_CHECKPOINT_VARIANTS, PidDecoder
from mflux.models.common.vae.vae_util import VAEUtil
from mflux.utils.image_util import ImageUtil

# Backbones this script can drive on this machine today. qwen-image is listed for
# completeness (it's what the brief's Step 1 command uses) but as of 2026-07-24 this
# environment's HF cache holds only Qwen-Image's config/index JSON (61KB), not its
# safetensors weight shards -- `export --backbone qwen-image` will fail with a download/
# missing-file error until real weights are present. z-image-turbo's weights (32.8GB) are
# fully cached and is what Task 12 actually validated against.
_BACKBONES = {
    "qwen-image": ("mflux.models.qwen.variants.txt2img.qwen_image", "QwenImage", "qwen-image"),
    "z-image-turbo": ("mflux.models.z_image.variants.z_image", "ZImage", "flux"),
    "z-image": ("mflux.models.z_image.variants.z_image", "ZImage", "flux"),
}


def _load_model_class(backbone: str):
    import importlib

    module_path, class_name, pid_variant = _BACKBONES[backbone]
    module = importlib.import_module(module_path)
    return getattr(module, class_name), pid_variant


def cmd_export(args: argparse.Namespace) -> None:
    model_cls, pid_variant = _load_model_class(args.backbone)
    model = model_cls()

    captured: dict[str, mx.array] = {}
    original_decode = VAEUtil.decode

    def spy_decode(vae, latent, tiling_config=None):
        captured["latent"] = latent
        return original_decode(vae, latent, tiling_config)

    VAEUtil.decode = staticmethod(spy_decode)
    try:
        image = model.generate_image(
            seed=args.seed,
            prompt=args.prompt,
            num_inference_steps=args.steps,
            width=args.width,
            height=args.height,
            pid_decode=False,  # standard VAE decode -- we only want the latent it consumed
        )
    finally:
        VAEUtil.decode = original_decode

    if "latent" not in captured:
        raise RuntimeError("VAEUtil.decode was never called -- generation path changed?")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    image.save(path=str(out_dir / "export.png"))
    mx.save_safetensors(str(out_dir / "latent.safetensors"), {"latent": captured["latent"]})
    (out_dir / "export_meta.json").write_text(
        json.dumps(
            {
                "backbone": args.backbone,
                "pid_variant": pid_variant,
                "prompt": args.prompt,
                "seed": args.seed,
                "width": args.width,
                "height": args.height,
                "steps": args.steps,
                "latent_shape": list(captured["latent"].shape),
            },
            indent=2,
        )
    )
    print(f"Exported latent {list(captured['latent'].shape)} + export.png -> {out_dir}")


def _mae_psnr(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    mae = float(np.mean(np.abs(a - b)))
    mse = float(np.mean((a - b) ** 2))
    psnr = float("inf") if mse == 0 else 10.0 * np.log10((255.0**2) / mse)
    return mae, psnr


def cmd_compare(args: argparse.Namespace) -> None:
    meta_path = Path(args.latent).with_name("export_meta.json")
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    caption = args.caption or meta.get("prompt")
    variant = args.variant or meta.get("pid_variant", "qwen-image")
    if not caption:
        raise ValueError("No caption available -- pass --caption or run `export` first (writes export_meta.json)")

    latent = mx.load(args.latent)["latent"]
    decoder = PidDecoder.from_pretrained(variant=variant)

    if args.dump_intermediates:
        dump_dir = Path(args.dump_intermediates)
        dump_dir.mkdir(parents=True, exist_ok=True)
        caption_embs = decoder.caption_encoder(caption)
        mx.save_safetensors(str(dump_dir / "caption_embs.safetensors"), {"caption_embs": caption_embs})
        print(f"Dumped caption_embs {list(caption_embs.shape)} -> {dump_dir}")
        # "x after sampler step 1" is inlined here (not exposed by pid_sampler.sample) --
        # see pid_sampler.py for the reference this replicates.
        from mflux.models.common.pid_decoder.pid_sampler import STUDENT_T_LIST, _velocity_to_x0

        _, _, zH, zW = latent.shape
        target_h, target_w = zH * decoder.VAE_COMPRESSION * decoder.SR_SCALE, zW * decoder.VAE_COMPRESSION * decoder.SR_SCALE
        mx.random.seed(args.seed)
        x = mx.random.normal((latent.shape[0], 3, target_h, target_w))
        sigma = mx.zeros((latent.shape[0],))
        t_cur = mx.full((latent.shape[0],), STUDENT_T_LIST[0])
        v_pred = decoder.pid_net(x, t_cur * 1000.0, caption_embs, latent, sigma)
        x0_pred = _velocity_to_x0(x, v_pred, t_cur)
        eps = mx.random.normal(x0_pred.shape)
        x_after_step1 = (1.0 - STUDENT_T_LIST[1]) * x0_pred + STUDENT_T_LIST[1] * eps
        mx.save_safetensors(str(dump_dir / "x_after_step1.safetensors"), {"x_after_step1": x_after_step1})
        print(f"Dumped x_after_step1 {list(x_after_step1.shape)} -> {dump_dir}")

    decoded = decoder.decode(latent=latent, caption=caption, seed=args.seed)
    normalized = ImageUtil._to_numpy(ImageUtil._denormalize(decoded))
    decoded_img = ImageUtil._numpy_to_pil(normalized)
    decoded_img.save(args.out)
    print(f"Decoded -> {args.out} ({decoded_img.size})")

    if args.reference:
        ref_img = PIL.Image.open(args.reference).convert("RGB")
        if ref_img.size != decoded_img.size:
            print(f"WARNING: size mismatch, resizing reference {ref_img.size} -> {decoded_img.size}")
            ref_img = ref_img.resize(decoded_img.size, PIL.Image.BICUBIC)
        mae, psnr = _mae_psnr(np.array(decoded_img.convert("RGB")), np.array(ref_img))
        print(f"MAE: {mae:.4f} (0-255 scale)   PSNR: {psnr:.2f} dB")
    else:
        print("No --reference given (expected until Step 2 is run on a CUDA machine) -- skipped diff.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_export = sub.add_parser("export", help="Generate an image, capture the pre-decode latent (Step 1)")
    p_export.add_argument("--backbone", choices=sorted(_BACKBONES), default="z-image-turbo")
    p_export.add_argument("--prompt", required=True)
    p_export.add_argument("--seed", type=int, default=42)
    p_export.add_argument("--width", type=int, default=1024)
    p_export.add_argument("--height", type=int, default=1024)
    p_export.add_argument("--steps", type=int, default=8)
    p_export.add_argument("--out-dir", required=True)
    p_export.set_defaults(func=cmd_export)

    p_compare = sub.add_parser("compare", help="Run PidDecoder.decode on an exported latent and diff vs. reference (Step 3)")
    p_compare.add_argument("--latent", required=True, help="Path to latent.safetensors from `export`")
    p_compare.add_argument("--reference", default=None, help="Reference PNG from NVIDIA's from_clean.py (Step 2)")
    p_compare.add_argument("--caption", default=None, help="Overrides the prompt from export_meta.json")
    p_compare.add_argument("--variant", default=None, choices=[None, *sorted(PID_CHECKPOINT_VARIANTS)])
    p_compare.add_argument("--seed", type=int, default=0)
    p_compare.add_argument("--out", required=True, help="Where to save this port's decoded PNG")
    p_compare.add_argument("--dump-intermediates", default=None, help="Dir to also dump caption_embs / x_after_step1")
    p_compare.set_defaults(func=cmd_compare)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
