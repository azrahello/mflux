#!/usr/bin/env python3
"""
Test script to compare different DDIM configurations.
"""

import subprocess
from pathlib import Path

# Base configuration
PROMPT = "A 27-year-old rebellious woman, partially clothed in a green bra, neon magenta fishnet top, and yellow underwear, kneels on a bed with blue sheets and yellow pillows, legs spread wide in a dynamic pose that maximizes geometric depth; her face bears a mischievous smile, sharp eyes conveying malice, with a beautiful voluminous hairstyle cascading over her shoulders; the scene is lit by soft diffused light creating gentle shadows across pastel-colored bed linens while the neon magenta fishnet top contrasts sharply against pale blue sheets and yellow pillows; an 85mm lens captures the scene with shallow depth of field, isolating the subject against the softly blurred pastel background, as the composition emphasizes spatial depth through the subject's kneeling stance and open legs forming leading lines that guide the viewer's eye; the pastel color palette of soft blue and pale yellow is punctuated by the vibrant neon magenta top, creating chromatic tension that reinforces the rebellious and erotic mood, with the subject's confident posture and intense gaze embodying playful defiance through subtle anatomical details such as bent knees, arched back, and tension in the thighs, while the bed's blue sheets and yellow pillows establish a visual anchor for the scene's spatial stratification with foreground, middle ground, and background elements seamlessly integrated. vibrant colors, well-lit, high contrast, vivid"

SEED = 1010
STEPS = 13
GUIDANCE = 1.0
MODEL = "AlessandroRizzo/whiteQWEN-alpha02"
PATH = "/Volumes/NewHome/test/mflux_models/qwenwhitefranky_fp16_sab128-03_8step_samsung06_jib02_meta015_skinfix02_ghost06_sabri02_leti02"
WIDTH = 832
HEIGHT = 1248

# Test configurations
CONFIGS = [
    {"name": "01_linear_baseline", "scheduler": "linear", "steps": 20},
    {"name": "02_ddim_default", "scheduler": "ddim"},
    {"name": "03_ddim_scaled_linear", "scheduler": "ddim", "beta_schedule": "scaled_linear"},
    {"name": "04_ddim_cosine", "scheduler": "ddim", "beta_schedule": "squaredcos_cap_v2"},
    {"name": "05_ddim_trailing", "scheduler": "ddim", "timestep_spacing": "trailing"},
    {"name": "06_ddim_linspace", "scheduler": "ddim", "timestep_spacing": "linspace"},
    {"name": "07_ddim_zero_snr", "scheduler": "ddim", "rescale_betas_zero_snr": True},
    {
        "name": "08_ddim_cosine_trailing",
        "scheduler": "ddim",
        "beta_schedule": "squaredcos_cap_v2",
        "timestep_spacing": "trailing",
    },
    {
        "name": "09_ddim_scaled_trailing",
        "scheduler": "ddim",
        "beta_schedule": "scaled_linear",
        "timestep_spacing": "trailing",
    },
    {
        "name": "10_ddim_best_combo",
        "scheduler": "ddim",
        "beta_schedule": "squaredcos_cap_v2",
        "timestep_spacing": "trailing",
        "rescale_betas_zero_snr": True,
    },
]


def main():
    output_dir = Path("ddim_test")
    output_dir.mkdir(exist_ok=True)

    print(f"🧪 Testing {len(CONFIGS)} DDIM configurations...")
    print(f"📁 Output: {output_dir}")
    print(f"🌱 Seed: {SEED}")
    print(f"📐 Resolution: {WIDTH}x{HEIGHT}")
    print(f"🎯 Steps: {STEPS} (except baseline: 20)")
    print()

    for i, config in enumerate(CONFIGS, 1):
        print(f"\n{'=' * 60}")
        print(f"[{i}/{len(CONFIGS)}] {config['name']}")
        print(f"{'=' * 60}")

        output = output_dir / f"{config['name']}.png"

        cmd = [
            "mflux-generate-qwen",
            "--model",
            MODEL,
            "--path",
            PATH,
            "--metadata",
            "--low-ram",
            "--width",
            str(WIDTH),
            "--height",
            str(HEIGHT),
            "--prompt",
            PROMPT,
            "--seed",
            str(SEED),
            "--steps",
            str(config.get("steps", STEPS)),
            "--guidance",
            str(GUIDANCE),
            "--scheduler",
            config["scheduler"],
            "--output",
            str(output),
        ]

        # Build scheduler kwargs
        kwargs = {}
        if "beta_schedule" in config:
            kwargs["beta_schedule"] = config["beta_schedule"]
        if "timestep_spacing" in config:
            kwargs["timestep_spacing"] = config["timestep_spacing"]
        if "rescale_betas_zero_snr" in config:
            kwargs["rescale_betas_zero_snr"] = config["rescale_betas_zero_snr"]

        if kwargs:
            # Create proper JSON string
            import json

            kwargs_str = json.dumps(kwargs)
            cmd.extend(["--scheduler-kwargs", kwargs_str])

            print("📋 Config:")
            print(f"   Scheduler: {config['scheduler']}")
            for key, value in kwargs.items():
                print(f"   {key}: {value}")
        else:
            print(f"📋 Config: {config['scheduler']} (default)")

        print("\n⏳ Generating...")

        try:
            result = subprocess.run(
                cmd,
                check=True,
                timeout=600,  # 10 minutes
                capture_output=True,
                text=True,
            )
            print("✅ Success!")

            # Extract generation time if available
            if "Generation time:" in result.stderr or "Generation time:" in result.stdout:
                for line in (result.stderr + result.stdout).split("\n"):
                    if "Generation time:" in line or "time:" in line.lower():
                        print(f"⏱️  {line.strip()}")
                        break

        except subprocess.TimeoutExpired:
            print("⏱️ TIMEOUT (>10 min)")
        except subprocess.CalledProcessError as e:
            print("❌ FAILED")
            print(f"Error: {e.stderr[:200] if e.stderr else 'Unknown error'}")
        except Exception as e:
            print(f"💥 ERROR: {e}")

    print(f"\n{'=' * 60}")
    print("✨ DONE!")
    print(f"{'=' * 60}")
    print(f"\n📁 All images saved to: {output_dir.absolute()}")
    print("\n💡 Compare the images to find:")
    print("   - Most natural skin tones")
    print("   - Best lighting/contrast balance")
    print("   - Realistic vs over-processed look")
    print("   - Which config gives the 'photographic' quality you want")
    print()


if __name__ == "__main__":
    main()
