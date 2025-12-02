# Qwen Performance Optimizations Changelog

## Branch: qwen-performance-optimizations

### Goal
Reduce the 2x performance gap between MLX and MPS implementations of Qwen Image.

### Baseline Performance
- **MPS (qwen-image-mps)**: X seconds (reference)
- **MLX (mflux original)**: 2X seconds (2x slower)
- **Target**: <1.5X seconds (within 50% of MPS)

---

## Applied Optimizations

### [NOT APPLIED YET] Patch 1: Vision Attention Chunking
**File**: `src/mflux/models/qwen/model/qwen_text_encoder/qwen_vision_attention.py`
**Issue**: Python loop processing chunks sequentially
**Fix**: Use mx.split and vectorized operations
**Expected Gain**: 10-15%
**Status**: ⏳ Pending

### [NOT APPLIED YET] Patch 2: Type Conversion Optimization  
**File**: `src/mflux/models/qwen/model/qwen_text_encoder/qwen_attention.py`
**Issue**: Redundant float32 conversions in hot path
**Fix**: Conditional type casting
**Expected Gain**: 5-8%
**Status**: ⏳ Pending

### [NOT APPLIED YET] Patch 3: Cache Management
**File**: `src/mflux/models/qwen/variants/txt2img/qwen_image.py`
**Issue**: MLX cache not properly managed
**Fix**: Explicit cache clearing and limits
**Expected Gain**: 5-10% (memory), stability improvement
**Status**: ⏳ Pending

### [NOT APPLIED YET] Patch 4: Function Compilation
**Files**: Various attention modules
**Issue**: Functions not compiled to Metal kernels
**Fix**: Add @mx.compile decorators
**Expected Gain**: 8-12%
**Status**: ⏳ Pending

---

## Benchmark Results

### Pre-Optimization (Baseline)
```
Configuration: 10 steps, 1024x1024, q8, seed=42
Prompt: "A serene mountain landscape at sunset"

MPS:  X.XX seconds
MLX:  X.XX seconds (2.0x slower)
```

### Post-Optimization (Target)
```
[Results will be added after applying patches]
```

---

## Notes
- Each patch should be tested individually
- Benchmark after each patch to measure impact
- Document any unexpected behavior
- Keep patches small and focused
