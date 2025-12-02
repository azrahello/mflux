#!/usr/bin/env python3
"""
Example script demonstrating STORK scheduler usage with mflux.

STORK (Stabilized Runge-Kutta) is a training-free sampling method that
accelerates flow matching inference while maintaining quality.
"""

from mflux import Config, Flux1


def generate_with_stork_2():
    """Generate an image using STORK-2 (faster variant)."""
    print("=" * 60)
    print("STORK-2 Example (Second-Order, Faster)")
    print("=" * 60)

    # Load Flux model
    print("\nLoading Flux.1-schnell model...")
    flux = Flux1.from_alias("schnell")

    # Generate with STORK-2
    print("\nGenerating with STORK-2 scheduler (4 steps)...")
    image = flux.generate_image(
        seed=42,
        prompt="A serene mountain lake at sunset, photorealistic, 4k",
        config=Config(num_inference_steps=4, height=1024, width=1024, scheduler="stork-2"),
    )

    output_path = "output_stork2.png"
    image.save(output_path)
    print(f"✓ Image saved to {output_path}")


def generate_with_stork_4():
    """Generate an image using STORK-4 (higher quality variant)."""
    print("\n" + "=" * 60)
    print("STORK-4 Example (Fourth-Order, Higher Quality)")
    print("=" * 60)

    # Load Flux model
    print("\nLoading Flux.1-dev model...")
    flux = Flux1.from_alias("dev")

    # Generate with STORK-4
    print("\nGenerating with STORK-4 scheduler (8 steps)...")
    image = flux.generate_image(
        seed=42,
        prompt="A serene mountain lake at sunset, photorealistic, 4k",
        config=Config(num_inference_steps=8, height=1024, width=1024, scheduler="stork-4"),
    )

    output_path = "output_stork4.png"
    image.save(output_path)
    print(f"✓ Image saved to {output_path}")


def compare_schedulers():
    """Compare different schedulers side by side."""
    print("\n" + "=" * 60)
    print("Scheduler Comparison")
    print("=" * 60)

    prompt = "A futuristic cityscape at night, neon lights, cyberpunk"
    seed = 123

    flux = Flux1.from_alias("schnell")

    schedulers = [
        ("euler", 20, "Euler (baseline)"),
        ("ddim", 10, "DDIM (accelerated)"),
        ("stork-2", 6, "STORK-2 (fast)"),
        ("stork-4", 10, "STORK-4 (quality)"),
    ]

    for scheduler_name, steps, description in schedulers:
        print(f"\n{description}: {steps} steps")

        image = flux.generate_image(
            seed=seed,
            prompt=prompt,
            config=Config(num_inference_steps=steps, height=512, width=512, scheduler=scheduler_name),
        )

        output_path = f"comparison_{scheduler_name}.png"
        image.save(output_path)
        print(f"  ✓ Saved to {output_path}")


def main():
    """Run all examples."""
    import argparse

    parser = argparse.ArgumentParser(description="STORK Scheduler Examples")
    parser.add_argument(
        "--example", choices=["stork2", "stork4", "compare", "all"], default="all", help="Which example to run"
    )

    args = parser.parse_args()

    if args.example in ["stork2", "all"]:
        generate_with_stork_2()

    if args.example in ["stork4", "all"]:
        generate_with_stork_4()

    if args.example == "compare":
        compare_schedulers()

    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)


if __name__ == "__main__":
    main()
