import os
from pathlib import Path

import platformdirs

BATTERY_PERCENTAGE_STOP_LIMIT = 5
CONTROLNET_STRENGTH = 0.4
DEFAULT_DEV_FILL_GUIDANCE = 30
DEFAULT_DEPTH_GUIDANCE = 10
DIMENSION_STEP_PIXELS = 16
GUIDANCE_SCALE = 3.5
GUIDANCE_SCALE_KONTEXT = 2.5
HEIGHT, WIDTH = 1024, 1024
IMAGE_STRENGTH = 0.4
MODEL_CHOICES = [
    "dev",
    "schnell",
    "krea-dev",
    "dev-krea",
    "krea-2",
    "krea2",
    "qwen",
    "fibo",
    "fibo-lite",
    "fibo-edit",
    "fibo-edit-rmbg",
    "z-image",
    "z-image-turbo",
    "flux2-klein-4b",
    "flux2-klein-9b",
    "flux2-klein-9b-kv",
    "flux2-klein-base-4b",
    "flux2-klein-base-9b",
    "ernie-image-turbo",
    "ernie-image",
    "ideogram4",
]
MODEL_INFERENCE_STEPS = {
    "dev": 25,
    "schnell": 4,
    "krea-dev": 25,
    "qwen": 20,
    "qwen-image": 20,
    "qwen-image-edit": 20,
    "fibo": 50,
    "fibo-lite": 8,
    "fibo-edit": 50,
    "fibo-edit-rmbg": 10,
    "z-image": 50,
    "z-image-turbo": 9,
    "krea-2": 8,
    "krea2": 8,
    "ernie-image-turbo": 8,
    "ernie-image": 50,
    "flux2-klein-4b": 4,
    "flux2-klein-9b": 4,
    "flux2-klein-9b-kv": 4,
    "flux2-klein-base-4b": 50,
    "flux2-klein-base-9b": 50,
    "ideogram4": 20,
    "ideogram-4-fp8": 20,
}
QUANTIZE_CHOICES = [3, 5, 4, 6, 8]

# Per-layer weights applied to the stacked multi-tap text-encoder conditioning
# (Krea 2: 12 taps, Ideogram 4: 13 taps). All-1.0 = no effect on generation.
# Edit these lists directly to change the default for all future generations
# of that model without passing --conditioning-weights every time. Keyed by
# the single canonical name each mflux-generate-<model> CLI uses internally
# (not by every --model alias).
CONDITIONING_WEIGHTS_DEFAULT = {
    "krea2": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
    "ideogram4": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
}

# Uniform gain on the whole conditioning tensor, applied after the per-layer
# weights (and after --conditioning-renormalize). Mirrors the reference
# rebalance node's 'multiplier' input. 1.0 = no effect.
CONDITIONING_MULTIPLIER_DEFAULT = {
    "krea2": 1.0,
    "ideogram4": 1.0,
}

# Clamp applied to |conditioning value| after weights/multiplier, guarding against
# fp16 overflow on extreme settings (see --conditioning-clamp). 0 = no clamp.
CONDITIONING_CLAMP_DEFAULT = {
    "krea2": 0.0,
    "ideogram4": 0.0,
}

# Default per-layer diffs for --projector-rebalance-weights (the Krea 2
# identity-edit recipe values, ported from the ComfyUI AzKrea2ProjectorRebalance
# node). Scaled by --projector-rebalance-strength (default 0.05); pass
# --projector-rebalance-weights none to disable the patch entirely.
PROJECTOR_REBALANCE_WEIGHTS_DEFAULT = {
    "krea2": "-24.195,-32.266,92.695,125.977,176.379,98.633,99.555,-359.75,-127.92,-190.32,-152.17,28.199",
}

# Optional default guidance schedule per model, format "start-end:value;...".
# None = use the model's normal constant/preset guidance behavior.
GUIDANCE_SCHEDULE_DEFAULT = {
    "krea2": None,
    "ideogram4": None,
}

# Optional default time-gating for CONDITIONING_WEIGHTS_DEFAULT (see
# --conditioning-crossover/--conditioning-overlap). None = no gating, weights
# apply to every step (current default behavior). krea2-only: ideogram4 has
# no per-step conditioning plan to gate against.
CONDITIONING_CROSSOVER_DEFAULT = {
    "krea2": None,
}
CONDITIONING_OVERLAP_DEFAULT = {
    "krea2": 0.0,
}

if os.environ.get("MFLUX_CACHE_DIR"):
    MFLUX_CACHE_DIR = Path(os.environ["MFLUX_CACHE_DIR"]).resolve()
else:
    MFLUX_CACHE_DIR = Path(platformdirs.user_cache_dir(appname="mflux"))

MFLUX_LORA_CACHE_DIR = MFLUX_CACHE_DIR / "loras"
