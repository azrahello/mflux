#!/usr/bin/env python3
"""End-to-end memory + time profile of an mflux generation: per phase and per step.

Reports two tables:
  1. per phase   -- import, backbone load, PiD load, decode, save
  2. per step    -- one row per denoise step and per PiD sampler step, so memory growth
                    across steps (a leak) is immediately visible against a flat profile

Steps are measured by hooking the actual forwards (backbone transformer, PidNet), not by
sampling on a timer -- so every row corresponds to a real unit of work.

All measurements use mlx.core's own counters -- no numpy anywhere.

Usage:
  python profile_generation.py --size 512                 # VAE decode (baseline)
  python profile_generation.py --size 512 --pid           # PiD decode
  python profile_generation.py --width 832 --height 1248 --pid -m /path/to/model
"""

from __future__ import annotations

import argparse
import contextlib
import resource
import sys
import time

import mlx.core as mx

GB = float(1 << 30)


def rss_gb() -> float:
    """Peak process RSS. macOS reports ru_maxrss in bytes, Linux in KiB -- decide by platform,
    not by magnitude (a magnitude guess misreads any process under 4 GB)."""
    raw = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return raw / GB if sys.platform == "darwin" else raw * 1024 / GB


class Log:
    """Phase and step records. Every read is preceded by mx.synchronize(): MLX is lazy, so
    without settling the graph the counters describe work that hasn't happened yet."""

    def __init__(self):
        self.phases: list[tuple[str, float, float, float, float]] = []
        self.steps: list[tuple[str, float, float, float]] = []
        self._last_step_t = None

    @contextlib.contextmanager
    def phase(self, name: str):
        mx.synchronize()
        mx.reset_peak_memory()
        held_before = mx.get_active_memory() / GB
        t0 = time.time()
        try:
            yield
        finally:
            mx.synchronize()
            dt = time.time() - t0
            self.phases.append(
                (name, dt, mx.get_peak_memory() / GB, mx.get_active_memory() / GB - held_before, rss_gb())
            )
            print(f"  {name:<30s} {dt:7.1f}s  peak {mx.get_peak_memory()/GB:6.2f} GB"
                  f"  held {mx.get_active_memory()/GB:6.2f} GB", flush=True)

    def step(self, label: str):
        """Record one unit of work (a denoise step, a PiD sampler step)."""
        mx.synchronize()
        now = time.time()
        dt = 0.0 if self._last_step_t is None else now - self._last_step_t
        self._last_step_t = now
        self.steps.append((label, dt, mx.get_active_memory() / GB, mx.get_peak_memory() / GB))

    def mark_step_start(self):
        self._last_step_t = time.time()

    def report(self):
        print("\n" + "=" * 78)
        print(f"{'phase':<30s} {'time':>8s} {'MLX peak':>10s} {'MLX delta':>11s} {'RSS':>9s}")
        print("-" * 78)
        for name, dt, peak, delta, rss in self.phases:
            print(f"{name:<30s} {dt:7.1f}s {peak:9.2f}G {delta:+10.2f}G {rss:8.2f}G")
        print("-" * 78)
        total = sum(p[1] for p in self.phases)
        worst = max(self.phases, key=lambda p: p[2])
        print(f"{'TOTAL':<30s} {total:7.1f}s   peak phase: {worst[0]} ({worst[2]:.2f} GB)")

        if not self.steps:
            print("=" * 78)
            return
        print("\n" + "=" * 78)
        print(f"{'step':<30s} {'time':>8s} {'active':>10s} {'peak':>10s} {'growth':>10s}")
        print("-" * 78)
        base = self.steps[0][2]
        for label, dt, act, peak in self.steps:
            print(f"{label:<30s} {dt:7.2f}s {act:9.2f}G {peak:9.2f}G {act - base:+9.2f}G")
        print("-" * 78)
        # Compare only steps of the same kind: loading PiD between the denoise and the
        # sampler is a real allocation, not a leak, and would otherwise dominate the verdict.
        for kind in ("denoise fwd", "PiD sampler step"):
            same = [s for s in self.steps if s[0].startswith(kind)]
            if len(same) < 2:
                continue
            drift = same[-1][2] - same[0][2]
            verdict = "flat (no leak)" if abs(drift) < 0.25 else f"GROWING {drift:+.2f} GB"
            print(f"{kind}: active {same[0][2]:.2f} -> {same[-1][2]:.2f} GB over "
                  f"{len(same)} steps : {verdict}")
            transient = max(s[3] for s in same) - max(s[2] for s in same)
            print(f"{'':>{len(kind)}}  transient inside one forward: {transient:+.2f} GB "
                  f"(peak {max(s[3] for s in same):.2f} vs held {max(s[2] for s in same):.2f})")

        # Point straight at the spike: the biggest jump in active memory between consecutive
        # records is what actually allocated, which is the thing worth reading first.
        jumps = [
            (self.steps[i][2] - self.steps[i - 1][2], self.steps[i - 1][0], self.steps[i][0])
            for i in range(1, len(self.steps))
        ]
        print("\nlargest allocations between consecutive records:")
        for delta, prev, cur in sorted(jumps, reverse=True)[:5]:
            if delta <= 0.01:
                break
            print(f"  {delta:+6.2f} GB   {prev}  ->  {cur}")
        hi = max(self.steps, key=lambda s: s[3])
        print(f"highest peak seen: {hi[3]:.2f} GB at '{hi[0]}'")
        print("=" * 78)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", "-m", default=None, help="Local model path (default: Z-Image-Turbo from HF cache)")
    p.add_argument("--prompt", default="a red fox in a snowy forest, warm afternoon light")
    p.add_argument("--size", type=int, default=None, help="Square size; overrides --width/--height")
    p.add_argument("--width", type=int, default=512)
    p.add_argument("--height", type=int, default=512)
    p.add_argument("--steps", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--pid", action="store_true", help="Decode with PiD instead of the VAE")
    p.add_argument("--deep", action="store_true", help="Also trace inside each PidNet forward (per LQ proj / patch block / pixel block), to attribute a spike to the block that caused it. Adds a sync per block, so timings are perturbed.")
    p.add_argument("--out", default="/tmp/profile_out.png")
    args = p.parse_args()

    if args.size:
        args.width = args.height = args.size
    log = Log()

    print(f"\n{args.width}x{args.height} | steps={args.steps} | seed={args.seed} | "
          f"decode={'PiD' if args.pid else 'VAE'}", flush=True)
    if args.pid:
        out_h, out_w = args.height * 4, args.width * 4
        print(f"PiD output: {out_w}x{out_h} ({out_w*out_h/1e6:.1f} MP), "
              f"L = {(out_h//16)*(out_w//16):,} patch tokens", flush=True)
    print()

    with log.phase("import + mlx init"):
        from mflux.models.common.config import ModelConfig
        from mflux.models.z_image.variants.z_image import ZImage

    with log.phase("load backbone"):
        model = ZImage(model_config=ModelConfig.z_image_turbo(), model_path=args.model)
        mx.eval(model.transformer.parameters())

    # --- per-step hooks: one row per real forward, not per timer tick -------------------
    counters = {"denoise": 0, "pid": 0}

    transformer_cls = type(model.transformer)
    _orig_transformer_call = transformer_cls.__call__

    def traced_transformer(self, *a, **kw):
        out = _orig_transformer_call(self, *a, **kw)
        counters["denoise"] += 1
        # Z-Image runs cond+uncond per step; count forwards, label by step.
        log.step(f"denoise fwd {counters['denoise']}")
        return out

    transformer_cls.__call__ = traced_transformer

    if args.pid:
        from mflux.models.common.pid_decoder.caption_encoder import PidCaptionEncoder
        from mflux.models.common.pid_decoder.pid_decoder import PidDecoder
        from mflux.models.common.pid_decoder.pixdit.pixdit_lq_projection import LQProjection2D
        from mflux.models.common.pid_decoder.pixdit.pixdit_mmdit_block import MMDiTBlockT2I
        from mflux.models.common.pid_decoder.pixdit.pixdit_network import PidNet
        from mflux.models.common.pid_decoder.pixdit.pixdit_pit_block import PiTBlock

        _orig_pidnet_call = PidNet.__call__

        def traced_pidnet(self, *a, **kw):
            out = _orig_pidnet_call(self, *a, **kw)
            counters["pid"] += 1
            log.step(f"PiD sampler step {counters['pid']}")
            return out

        PidNet.__call__ = traced_pidnet

        # The caption encoder runs once per decode, before the sampler -- a Gemma-2 forward
        # that the step hooks above would otherwise miss entirely.
        _orig_caption = PidCaptionEncoder.__call__

        def traced_caption(self, *a, **kw):
            out = _orig_caption(self, *a, **kw)
            log.step("caption encode (gemma-2)")
            return out

        PidCaptionEncoder.__call__ = traced_caption

        # --deep: hook inside the forward so a spike is attributed to the block that caused
        # it, not just to the step. Each hook syncs, which perturbs timing -- hence opt-in.
        if args.deep:
            for cls, tag in ((LQProjection2D, "lq_proj"), (MMDiTBlockT2I, "patch_blk"), (PiTBlock, "pixel_blk")):
                orig = cls.__call__
                counters[tag] = 0

                def make(orig=orig, tag=tag):
                    def traced(self, *a, **kw):
                        out = orig(self, *a, **kw)
                        counters[tag] += 1
                        log.step(f"  {tag} #{counters[tag]}")
                        return out

                    return traced

                cls.__call__ = make()

        _orig_from_pretrained = PidDecoder.from_pretrained.__func__
        _orig_decode = PidDecoder.decode

        def timed_from_pretrained(cls, *a, **kw):
            with log.phase("load PiD (net + gemma2)"):
                return _orig_from_pretrained(cls, *a, **kw)

        def timed_decode(self, *a, **kw):
            with log.phase("PiD decode"):
                log.mark_step_start()
                out = _orig_decode(self, *a, **kw)
                mx.eval(out)
                return out

        PidDecoder.from_pretrained = classmethod(timed_from_pretrained)
        PidDecoder.decode = timed_decode
    else:
        from mflux.models.common.vae.vae_util import VAEUtil

        _orig_vae_decode = VAEUtil.decode

        def timed_vae_decode(vae, latent, tiling_config=None):
            with log.phase("VAE decode"):
                out = _orig_vae_decode(vae, latent, tiling_config)
                mx.eval(out)
                return out

        VAEUtil.decode = staticmethod(timed_vae_decode)

    with log.phase("generate (encode+denoise+decode)"):
        log.mark_step_start()
        image = model.generate_image(
            seed=args.seed,
            prompt=args.prompt,
            width=args.width,
            height=args.height,
            num_inference_steps=args.steps,
            **({"pid_decode": True} if args.pid else {}),
        )

    with log.phase("save png"):
        image.save(path=args.out)

    log.report()
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
