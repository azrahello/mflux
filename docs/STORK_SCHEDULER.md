# STORK Scheduler

STORK (Stabilized Runge-Kutta) is a training-free sampling method that accelerates diffusion and flow-matching model inference by addressing ODE stiffness and structural limitations of existing solvers.

## Overview

Based on the paper ["STORK: Faster Diffusion and Flow Matching Sampling by Resolving both Stiffness and Structure-Dependence"](https://arxiv.org/html/2505.24210v2) by Tan et al. (UCLA, 2025).

### Key Features

- **Training-free**: Works directly with pre-trained Flux and Qwen models
- **Structure-independent**: Compatible with both diffusion and flow matching models
- **Stiff problem handling**: Uses orthogonal polynomials for stability
- **Virtual NFEs**: Approximates intermediate evaluations using Taylor expansions to reduce computational cost
- **Two variants**: STORK-2 (second-order, faster) and STORK-4 (fourth-order, more accurate)

## Usage

### With Flux Models

```bash
# STORK-2 (faster, recommended for most use cases)
mflux-generate \
    --model schnell \
    --prompt "A beautiful sunset over mountains" \
    --scheduler stork-2 \
    --steps 4

# STORK-4 (higher quality, slightly slower)
mflux-generate \
    --model dev \
    --prompt "A beautiful sunset over mountains" \
    --scheduler stork-4 \
    --steps 8
```

### With Qwen Models

```bash
# STORK-2
mflux-generate-qwen \
    --prompt "A serene lake with reflections" \
    --scheduler stork-2 \
    --steps 10

# STORK-4
mflux-generate-qwen \
    --prompt "A serene lake with reflections" \
    --scheduler stork-4 \
    --steps 15
```

### Python API

```python
from mflux import Flux1, Config

# Create model
flux = Flux1.from_alias("schnell")

# Generate with STORK-2
image = flux.generate_image(
    seed=42,
    prompt="A beautiful sunset over mountains",
    config=Config(
        num_inference_steps=4,
        height=1024,
        width=1024,
        scheduler="stork-2"  # or "stork-4"
    )
)

image.save("output.png")
```

## Performance Recommendations

### STORK-2 (Second-Order)
- **Best for**: Fast generation, real-time applications
- **Recommended steps**: 4-10 steps
- **Speed**: Fastest STORK variant
- **Quality**: Good quality with minimal steps

### STORK-4 (Fourth-Order)
- **Best for**: High-quality generation, complex prompts
- **Recommended steps**: 8-20 steps
- **Speed**: Slightly slower than STORK-2
- **Quality**: Higher accuracy, especially with more steps

## Comparison with Other Schedulers

| Scheduler | Steps (typical) | Speed | Quality | Training-free |
|-----------|----------------|-------|---------|---------------|
| Euler (flow_match_euler_discrete) | 20-50 | Baseline | Good | ✅ |
| DDIM (ddim) | 10-25 | Fast | Good | ✅ |
| **STORK-2** | **4-10** | **Very Fast** | **Good** | ✅ |
| **STORK-4** | **8-20** | **Fast** | **Very Good** | ✅ |

## Technical Details

### How STORK Works

1. **Stabilized Runge-Kutta Integration**: Uses orthogonal polynomials (Gegenbauer/Chebyshev) to construct stable RK schemes that handle stiff ODEs without requiring smaller timesteps.

2. **Virtual NFEs (Neural Function Evaluations)**: Instead of calling the model multiple times per step, STORK approximates intermediate velocities using Taylor expansions of previously computed values.

3. **Multi-stage Integration**:
   - STORK-2: 2-stage RK method with first-order Taylor approximations
   - STORK-4: 4-stage RK method with second-order Taylor approximations

### Compatibility

STORK works with any flow matching model, including:
- ✅ Flux.1 [schnell]
- ✅ Flux.1 [dev]
- ✅ Flux.1 [pro]
- ✅ Qwen Image models
- ✅ All Flux variants (Fill, Redux, Depth, ControlNet, etc.)

## Advanced Options

You can customize STORK behavior using scheduler kwargs:

```python
config = Config(
    scheduler="stork-2",
    scheduler_kwargs={
        "order": 2,  # Explicitly set order (2 or 4)
    }
)
```

## Tips for Best Results

1. **Start with fewer steps**: STORK is designed for efficiency. Try 4-6 steps with STORK-2 first.

2. **Use STORK-2 for iteration**: When experimenting with prompts, use STORK-2 for fast iterations.

3. **Use STORK-4 for final renders**: When you've settled on a prompt, use STORK-4 with more steps for the final output.

4. **Adjust based on model**:
   - Flux schnell: 4-6 steps with STORK-2
   - Flux dev: 8-12 steps with STORK-2 or STORK-4
   - Qwen: 10-15 steps with STORK-4

## Troubleshooting

### Images look noisy or incomplete
- Increase the number of steps
- Try STORK-4 instead of STORK-2
- Check that your prompt is clear and detailed

### Generation is slower than expected
- Verify you're using STORK-2 (faster) not STORK-4
- Reduce the number of steps
- Check system resources (memory, GPU usage)

### Errors during generation
- Ensure you're using a compatible model (Flux or Qwen)
- Update mflux to the latest version
- Check that scheduler name is correct: "stork-2" or "stork-4"

## Citation

If you use STORK in your research or projects, please cite:

```bibtex
@article{tan2025stork,
  title={STORK: Faster Diffusion and Flow Matching Sampling by Resolving both Stiffness and Structure-Dependence},
  author={Tan, Zheng and Wang, Weizhen and Bertozzi, Andrea L. and Ryu, Ernest K.},
  journal={arXiv preprint arXiv:2505.24210},
  year={2025}
}
```

## References

- [STORK Paper](https://arxiv.org/html/2505.24210v2)
- [mflux Documentation](../README.md)
- [Scheduler Comparison Guide](./SCHEDULERS.md)
