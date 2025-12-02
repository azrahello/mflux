# mflux Documentation

This directory contains documentation for various features and components of mflux.

## 📚 Available Documentation

### Schedulers
- **[Complete Scheduler Guide](ALL_SCHEDULERS.md)** - All 6 schedulers unified in one place
- **[STORK Scheduler](STORK_SCHEDULER.md)** - Training-free accelerated sampling using Stabilized Runge-Kutta methods

## 🚀 Quick Start

### See All Available Schedulers
Check out the [Complete Scheduler Guide](ALL_SCHEDULERS.md) for:
- 6 schedulers: Linear, Euler, DDIM, ER-SDE-Beta, Advanced, STORK (2 & 4)
- Performance comparisons
- Usage examples
- Best practices

### Fast Generation
```bash
mflux-generate --model schnell --prompt "sunset" --scheduler stork-2 --steps 4
```

### High Quality Generation
```bash
mflux-generate --model dev --prompt "sunset" --scheduler er_sde_beta --steps 25
```

## 🔗 Quick Links

- [Main Repository](../README.md)
- [Examples](../examples/)
- [Tests](../tests/)
