# Fix: LoRA baking and quantization support for large models

## Summary

This PR implements complete LoRA baking and quantization support for large models (Qwen), fixing several critical issues with model saving and loading.

## Problem Statement

When saving large models with LoRAs:
1. Individual weight tensors > 2GB caused empty shards or save failures
2. Models were saved at 2x expected size (82GB instead of 56GB)
3. Quantization with `-q` flag was not working correctly
4. 240 LoRA keys were unmatched for Qwen models (img_mod, txt_mod)
5. Text encoder was incorrectly quantized despite `skip_quantization=True`
6. Loading models failed with ValueError on "None" string in metadata

## Changes

### 1. Fix oversized weight handling (2e640ca)
- Modified `_split_weights()` to handle individual weights > 2GB
- Added `and shard` check to prevent empty shard creation

### 2. LoRA baking implementation (6a04b05)
- Created new `LoRABaker` class in `src/mflux/models/common/lora/baking/lora_baker.py`
- Supports both `LoRALinear` and `FusedLoRALinear` layers
- Handles quantized models (dequantizes before baking)
- Memory optimizations with periodic cleanup every 50 layers
- Formula: `W_merged = W_base + scale * (lora_A @ lora_B).T`

### 3. Quantization support (eea66d3, 794a041)
- Quantization applied to entire model after baking using `nn.quantize(model, bits=bits)`
- Fixed incorrect per-layer quantization attempts

### 4. Qwen LoRA mappings (854f492)
- Added missing mappings for `txt_mod_linear` and `img_mod_linear`
- Increased matched layers from 720 to 840 (1680/1680 keys)

### 5. Respect skip_quantization flag (d50fdc9)
- Text encoder now correctly skipped during quantization
- Only transformer component is quantized when using `-q` flag

### 6. Metadata loading fix (796dddf)
- Handle both `None` and string `"None"` in quantization_level metadata

### 7. Memory optimization during saving (23c92d4)
- Aggressive memory management to prevent swapping
- Clear intermediate weight dictionaries immediately after use
- Evaluate each shard before saving to materialize lazy computations
- Delete shards immediately after saving
- Cleanup every 5 shards (~10GB) with gc.collect() and mx.clear_cache()
- Significantly reduces memory pressure, especially for text_encoder

## Results

- ✅ Saving with `-q 8`: ~14-17GB instead of ~56GB
- ✅ LoRA weights permanently merged into base weights
- ✅ All LoRA layers matched: 840/840 (1680/1680 keys)
- ✅ Model loading works correctly
- ✅ Image generation functional

## Testing

Tested with:
- Model: `AlessandroRizzo/whiteQWEN-alpha02`
- 10 LoRAs with various scales
- Quantization levels: 8-bit
- Result: ~14GB quantized model with baked LoRAs, fully functional

## Breaking Changes

None. All changes are backward compatible.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude <noreply@anthropic.com>
