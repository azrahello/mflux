# mflux Schedulers - Complete Guide

This is the complete, unified collection of all schedulers available in mflux after merging all experimental branches.

## 📦 Available Schedulers (6 Total)

### 1. **LinearScheduler** (Basic)
```bash
--scheduler linear
```

**Description**: Simple linear sigma schedule from 1.0 to 0.0

**Characteristics**:
- Most basic scheduler
- Good baseline for comparisons
- Predictable behavior

**Recommended Steps**: 20-50

**Use When**: You need a simple, predictable baseline

---

### 2. **FlowMatchEulerDiscreteScheduler** (Standard)
```bash
--scheduler flow_match_euler_discrete
```

**Description**: Standard Euler method with time-shifting for resolution scaling

**Characteristics**:
- Industry standard for Flow Matching
- Resolution-adaptive time shifting
- Reliable and well-tested

**Recommended Steps**: 20-50

**Use When**: You want the standard, reliable approach

---

### 3. **DDIMFlowScheduler** (Accelerated)
```bash
--scheduler ddim
```

**Description**: DDIM-style accelerated sampling for Flow Matching

**Characteristics**:
- 10x-50x faster than standard Euler
- Deterministic sampling (eta=0)
- Strategic timestep selection

**Recommended Steps**: 10-25

**Parameters**:
```python
scheduler_kwargs={"eta": 0.0}  # 0.0 = deterministic, 1.0 = stochastic
```

**Use When**: You need faster sampling without sacrificing much quality

**Reference**: [DDIM Paper](https://diffusionflow.github.io/)

---

### 4. **ERSDEBetaScheduler** ⭐ (High Quality)
```bash
--scheduler er_sde_beta
```

**Description**: Extended Reverse-Time SDE with Beta timestep distribution

**Characteristics**:
- Superior detail preservation
- Natural, non-plastic appearance
- Controllable ODE/SDE balance
- Beta-distributed timesteps for optimal sampling

**Recommended Steps**: 15-30

**Parameters**:
```python
scheduler_kwargs={
    "gamma": 0.0,              # 0.0 = pure ODE, 0.3 = balanced SDE
    "use_beta": True,          # Use Beta distribution
    "beta_strength": 1.0,      # 1.0 = gentle, 3.0+ = aggressive
    "use_subsequence": False,  # DDIM-style subsequence
}
```

**Use When**: You want maximum quality and natural appearance

**References**:
- Extended Reverse-Time SDE: [arXiv:2309.06169](https://arxiv.org/abs/2309.06169)
- Beta Sampling: [arXiv:2407.12173](https://arxiv.org/abs/2407.12173)

---

### 5. **FlowMatchAdvancedScheduler** (Experimental)
```bash
--scheduler advanced
```

**Description**: Multiple advanced scheduling strategies (Cosine, Exponential, Beta, etc.)

**Characteristics**:
- Multiple schedule types in one scheduler
- Highly configurable
- Experimental features

**Recommended Steps**: Varies by schedule type

**Parameters**:
```python
scheduler_kwargs={
    "schedule_type": "cosine",  # "linear", "cosine", "sqrt", "exponential", "beta"
    # Additional parameters depending on schedule_type
}
```

**Use When**: You're experimenting with different schedule strategies

---

### 6. **STORKScheduler** ⭐ (Ultra Fast)
```bash
--scheduler stork-2  # or stork-4
```

**Description**: Stabilized Runge-Kutta with virtual NFEs

**Variants**:
- **STORK-2** (`stork-2` or `stork`): Second-order, ultra-fast
- **STORK-4** (`stork-4`): Fourth-order, higher quality

**Characteristics**:
- Training-free (works with any model)
- Virtual NFEs via Taylor approximations
- Handles stiff ODEs with stability polynomials
- Fastest scheduler available

**Recommended Steps**:
- STORK-2: 4-10 steps
- STORK-4: 8-20 steps

**Parameters**:
```python
scheduler_kwargs={
    "order": 2  # or 4
}
```

**Use When**: You need maximum speed with good quality

**Reference**: [STORK Paper](https://arxiv.org/html/2505.24210v2)

---

## 📊 Performance Comparison

| Scheduler | Alias | Min Steps | Speed | Quality | Training-free |
|-----------|-------|-----------|-------|---------|---------------|
| Linear | `linear` | 20-50 | ⚡ | ⭐⭐⭐ | ✅ |
| Euler | `flow_match_euler_discrete` | 20-50 | ⚡ | ⭐⭐⭐ | ✅ |
| DDIM | `ddim` | 10-25 | ⚡⚡ | ⭐⭐⭐ | ✅ |
| ER-SDE-Beta | `er_sde_beta` | 15-30 | ⚡⚡ | ⭐⭐⭐⭐⭐ | ✅ |
| Advanced | `advanced` | Varies | ⚡⚡ | ⭐⭐⭐⭐ | ✅ |
| **STORK-2** | `stork-2`, `stork` | **4-10** | **⚡⚡⚡** | **⭐⭐⭐⭐** | ✅ |
| **STORK-4** | `stork-4` | **8-20** | **⚡⚡** | **⭐⭐⭐⭐⭐** | ✅ |

## 🎯 Recommendation Matrix

### For Maximum Speed
```bash
--scheduler stork-2 --steps 4
```
**Best for**: Real-time, iteration, prototyping

### For Balanced Speed/Quality
```bash
--scheduler stork-4 --steps 10
# or
--scheduler ddim --steps 15
```
**Best for**: Most use cases, production

### For Maximum Quality
```bash
--scheduler er_sde_beta --steps 25 --scheduler-kwargs '{"gamma": 0.0, "beta_strength": 2.0}'
# or
--scheduler stork-4 --steps 20
```
**Best for**: Final renders, showcases, critical applications

### For Natural Appearance
```bash
--scheduler er_sde_beta --steps 20 --scheduler-kwargs '{"gamma": 0.3, "use_beta": true}'
```
**Best for**: Photorealistic images, avoiding "plastic" look

### For Experimentation
```bash
--scheduler advanced --scheduler-kwargs '{"schedule_type": "cosine"}'
```
**Best for**: Research, finding optimal settings

## 🔧 Usage Examples

### CLI - Flux
```bash
# STORK-2 (fastest)
mflux-generate \
    --model schnell \
    --prompt "A beautiful sunset" \
    --scheduler stork-2 \
    --steps 6

# ER-SDE-Beta (highest quality)
mflux-generate \
    --model dev \
    --prompt "A beautiful sunset" \
    --scheduler er_sde_beta \
    --steps 25 \
    --scheduler-kwargs '{"gamma": 0.0, "beta_strength": 2.0}'

# DDIM (balanced)
mflux-generate \
    --model dev \
    --prompt "A beautiful sunset" \
    --scheduler ddim \
    --steps 15
```

### CLI - Qwen
```bash
# STORK-4 (fast + quality)
mflux-generate-qwen \
    --prompt "A serene lake" \
    --scheduler stork-4 \
    --steps 13

# ER-SDE-Beta (natural appearance)
mflux-generate-qwen \
    --prompt "A serene lake" \
    --scheduler er_sde_beta \
    --steps 20 \
    --scheduler-kwargs '{"gamma": 0.2, "use_beta": true}'
```

### Python API
```python
from mflux import Flux1, Config

flux = Flux1.from_alias("dev")

# STORK-2
image = flux.generate_image(
    seed=42,
    prompt="A beautiful sunset",
    config=Config(
        num_inference_steps=6,
        scheduler="stork-2"
    )
)

# ER-SDE-Beta with custom parameters
image = flux.generate_image(
    seed=42,
    prompt="A beautiful sunset",
    config=Config(
        num_inference_steps=25,
        scheduler="er_sde_beta",
        scheduler_kwargs={
            "gamma": 0.0,
            "beta_strength": 2.0,
            "use_beta": True
        }
    )
)

# Advanced scheduler with cosine schedule
image = flux.generate_image(
    seed=42,
    prompt="A beautiful sunset",
    config=Config(
        num_inference_steps=20,
        scheduler="advanced",
        scheduler_kwargs={
            "schedule_type": "cosine"
        }
    )
)
```

## 📖 Complete Registry

All schedulers are registered with multiple aliases:

```python
SCHEDULER_REGISTRY = {
    # Basic
    "linear": LinearScheduler,
    "LinearScheduler": LinearScheduler,

    # Standard
    "flow_match_euler_discrete": FlowMatchEulerDiscreteScheduler,
    "FlowMatchEulerDiscreteScheduler": FlowMatchEulerDiscreteScheduler,

    # Accelerated
    "ddim": DDIMFlowScheduler,
    "DDIMFlowScheduler": DDIMFlowScheduler,

    # High Quality
    "er_sde_beta": ERSDEBetaScheduler,
    "ERSDEBetaScheduler": ERSDEBetaScheduler,

    # Experimental
    "advanced": FlowMatchAdvancedScheduler,
    "FlowMatchAdvancedScheduler": FlowMatchAdvancedScheduler,

    # Ultra Fast
    "stork": _create_stork_2,        # Default to STORK-2
    "stork-2": _create_stork_2,      # Explicit STORK-2
    "stork-4": _create_stork_4,      # STORK-4
    "STORKScheduler": STORKScheduler, # Direct class
}
```

## 🚀 Installation

To use all these schedulers, install from the unified branch:

```bash
cd ~/.claude-worktrees/mflux/condescending-lehmann
pip uninstall mflux -y
pip install -e .
```

## 📚 Additional Documentation

- [STORK Detailed Guide](./STORK_SCHEDULER.md)
- [Main README](../README.md)
- [Examples](../examples/)

## 🆘 Troubleshooting

### Scheduler not found
```
NotImplementedError: The scheduler 'stork-4' is not implemented
```
**Solution**: Reinstall mflux from the unified branch (see Installation above)

### Images look weird/plastic
**Solution**: Try ER-SDE-Beta with `gamma > 0.0` for natural stochasticity

### Too slow
**Solution**: Use STORK-2 with 4-6 steps

### Not enough quality
**Solution**: Increase steps or switch to STORK-4 or ER-SDE-Beta

## 🎓 Tips & Tricks

1. **Start fast, finish slow**: Use STORK-2 for iterations, then STORK-4 or ER-SDE-Beta for finals
2. **Model matters**: Flux schnell works great with STORK-2 at 4-6 steps
3. **Resolution scaling**: All schedulers (except Linear) adapt to resolution automatically
4. **Combine techniques**: ER-SDE-Beta with `use_beta=True` + low `gamma` = best quality
5. **Experiment**: Use `advanced` scheduler to try different mathematical schedules

## 📝 Summary

You now have **6 powerful schedulers** unified in one branch:
- ✅ 3 basic/standard (Linear, Euler, DDIM)
- ✅ 1 high-quality (ER-SDE-Beta)
- ✅ 1 experimental (Advanced)
- ✅ 1 ultra-fast (STORK with 2 variants)

All are training-free, production-ready, and work with both Flux and Qwen models!
