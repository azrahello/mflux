"""
KV-cache container for the FLUX.2-klein-9b-kv variant.

The 9B-KV checkpoint distilled from FLUX.2 [klein] 9B at 4 inference steps with
an attention-side optimisation: reference-image (and text) K/V tensors are
computed once on step 0 and re-used on steps 1-3, skipping redundant
reference-token processing. BFL's reference implementation lives in the
``Flux2KleinKVPipeline`` in upstream diffusers.

This container holds the per-layer K/V slots for both the double-stream
(``transformer_blocks``) and single-stream (``single_transformer_blocks``)
stacks, plus the slice metadata each attention layer needs to know.

mflux's concat order inside the transformer differs from diffusers'. In
mflux the post-concat sequence is ``[txt, target, ref]`` (because
``Flux2KleinEdit._predict`` concatenates ``[latents, image_latents]`` and then
``Flux2Attention`` prepends the text-derived stream). Diffusers uses
``[txt, ref, target]``. The cache protocol is identical; only the slice
indices differ — we slice ``[:, :, num_txt + num_target :, :]`` in extract
mode and splice the cached ref K/V at the *end* of the fresh K/V in cached
mode.

K/V slots are pre-allocated as mx.zeros buffers and updated in-place via
``[:] =`` on each extract pass.  This preserves Python object identity so
that ``mx.compile(fn, inputs=kv_state)`` can reuse a single compiled Metal
graph across generations while reading fresh K/V values each time.
"""

from __future__ import annotations

from typing import Literal

import mlx.core as mx

from mflux.models.common.config.model_config import ModelConfig

CacheMode = Literal["extract", "cached"]
StreamType = Literal["double", "single"]


class Flux2KVCache:
    """Per-layer K/V slot store for FLUX.2-klein-9b-kv.

    Attributes
    ----------
    mode :
        ``"extract"`` populates the cache during the first denoise step,
        ``"cached"`` reads from it on subsequent steps.
    num_ref_tokens :
        Count of reference-image tokens (the static slice we cache).
    num_txt_tokens :
        Count of text-encoder tokens (also static across steps; needed in
        both modes so attention layers know where target tokens start).
    double_keys, double_values :
        Pre-allocated buffer lists for double-stream layers.
    single_keys, single_values :
        Pre-allocated buffer lists for single-stream layers.
    """

    def __init__(
        self,
        num_double_layers: int,
        num_single_layers: int,
        num_ref_tokens: int,
        num_heads: int,
        head_dim: int,
    ) -> None:
        shape = (1, num_heads, num_ref_tokens, head_dim)
        dtype = ModelConfig.precision
        self.double_keys:   list[mx.array] = [mx.zeros(shape, dtype=dtype) for _ in range(num_double_layers)]
        self.double_values: list[mx.array] = [mx.zeros(shape, dtype=dtype) for _ in range(num_double_layers)]
        self.single_keys:   list[mx.array] = [mx.zeros(shape, dtype=dtype) for _ in range(num_single_layers)]
        self.single_values: list[mx.array] = [mx.zeros(shape, dtype=dtype) for _ in range(num_single_layers)]
        self.num_ref_tokens: int = num_ref_tokens
        self.num_txt_tokens: int = 0
        self.mode: CacheMode | None = None

    def configure(
        self,
        *,
        mode: CacheMode,
        num_ref_tokens: int,
        num_txt_tokens: int,
    ) -> None:
        self.mode = mode
        self.num_ref_tokens = int(num_ref_tokens)
        self.num_txt_tokens = int(num_txt_tokens)

    # ------- store (extract mode) ----------------------------------------

    def store(self, stream: StreamType, layer_idx: int, key: mx.array, value: mx.array) -> None:
        # [:] preserves Python object identity (required for mx.compile inputs= tracking)
        # while updating the underlying data. [...] would be equivalent but MLX does
        # not support Ellipsis as an index type for __setitem__.
        if stream == "double":
            self.double_keys[layer_idx][:] = key
            self.double_values[layer_idx][:] = value
        elif stream == "single":
            self.single_keys[layer_idx][:] = key
            self.single_values[layer_idx][:] = value
        else:
            raise ValueError(f"Unknown stream {stream!r}")

    # ------- load (cached mode) ------------------------------------------

    def load(self, stream: StreamType, layer_idx: int) -> tuple[mx.array, mx.array]:
        if stream == "double":
            return self.double_keys[layer_idx], self.double_values[layer_idx]
        return self.single_keys[layer_idx], self.single_values[layer_idx]

    # ------- inspection --------------------------------------------------

    def is_populated(self) -> bool:
        return self.mode == "cached"


class Flux2KVExtractMeta:
    """Minimal metadata object for the compiled extract pass.

    Attention layers read mode and num_ref_tokens from this object to decide
    which tokens to extract. K/V arrays are NOT stored here — they are
    returned as explicit outputs by the transformer and collected by the
    caller (_make_extract_predict).
    """

    mode: str = "extract"

    def __init__(self, num_ref_tokens: int, num_txt_tokens: int = 0) -> None:
        self.num_ref_tokens = num_ref_tokens
        self.num_txt_tokens = num_txt_tokens


class Flux2KVReadOnlyView:
    """Lightweight read-only K/V view for the compiled cached predict.

    K/V arrays are passed as explicit function arguments to the compiled
    predict closure and wrapped here so the attention layers can call the
    familiar load() interface.  Creating this object inside the compiled
    function is pure Python — MLX only traces the array operations that
    follow (mx.concatenate inside the attention layers).

    mode is hardcoded to "cached"; store() is intentionally unsupported.
    """

    mode: str = "cached"

    def __init__(
        self,
        double_keys: list[mx.array],
        double_values: list[mx.array],
        single_keys: list[mx.array],
        single_values: list[mx.array],
    ) -> None:
        self._dk = double_keys
        self._dv = double_values
        self._sk = single_keys
        self._sv = single_values

    def load(self, stream: StreamType, layer_idx: int) -> tuple[mx.array, mx.array]:
        if stream == "double":
            return self._dk[layer_idx], self._dv[layer_idx]
        return self._sk[layer_idx], self._sv[layer_idx]

    def store(self, *_) -> None:
        raise RuntimeError("store() is not allowed on a read-only KV view")
