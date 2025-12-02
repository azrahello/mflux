# Qwen MLX Performance Benchmarks

## Test Configuration
- **Prompt**: "A serene mountain landscape at sunset"
- **Steps**: 10
- **Resolution**: 1024x1024
- **Quantization**: 8-bit
- **Seed**: 42

## Benchmark Script
```bash
#!/bin/bash
# Run this script to benchmark: ./benchmark.sh

PROMPT="A serene mountain landscape at sunset"
STEPS=10
SEED=42

echo "Running MPS benchmark..."
time qwen-image-mps generate -p "$PROMPT" -s $STEPS --seed $SEED

echo ""
echo "Running MLX benchmark..."
time mflux-generate-qwen --prompt "$PROMPT" --steps $STEPS --seed $SEED -q 8
```

## Results

### Baseline (Before Optimizations)
| Implementation | Time | Relative |
|----------------|------|----------|
| MPS            | TBD  | 1.0x     |
| MLX Original   | TBD  | 2.0x     |

### After Patch 1 (Vision Attention)
| Implementation | Time | Relative | Improvement |
|----------------|------|----------|-------------|
| MLX Patched    | TBD  | TBDx     | TBD%        |

### After Patch 2 (Type Conversions)
| Implementation | Time | Relative | Improvement |
|----------------|------|----------|-------------|
| MLX Patched    | TBD  | TBDx     | TBD%        |

### After Patch 3 (Cache Management)
| Implementation | Time | Relative | Improvement |
|----------------|------|----------|-------------|
| MLX Patched    | TBD  | TBDx     | TBD%        |

### After Patch 4 (Compilation)
| Implementation | Time | Relative | Improvement |
|----------------|------|----------|-------------|
| MLX Patched    | TBD  | TBDx     | TBD%        |

### Final Result
| Implementation | Time | Relative | Total Improvement |
|----------------|------|----------|-------------------|
| MPS            | TBD  | 1.0x     | -                 |
| MLX Optimized  | TBD  | TBDx     | TBD%              |

---

## Memory Usage

### Before Optimizations
```
Peak Memory: TBD GB
Cache Size: TBD GB
```

### After Optimizations
```
Peak Memory: TBD GB
Cache Size: TBD GB
Improvement: TBD%
```
