import gc

import mlx.core as mx
import PIL.Image
from tqdm import tqdm

from mflux.callbacks.callback import AfterLoopCallback, BeforeLoopCallback, InLoopCallback
from mflux.models.common.config.config import Config
from mflux.models.common.vae.tiling_config import TilingConfig


class MemorySaver(BeforeLoopCallback, InLoopCallback, AfterLoopCallback):
    def __init__(
        self,
        model,
        keep_transformer: bool = True,
        cache_limit_bytes: int | None = 1000**3,
        args=None,
        num_seeds: int = 1,
    ):
        self.model = model
        self.keep_transformer = keep_transformer
        self._cache_limit_bytes = cache_limit_bytes
        self._num_seeds = num_seeds
        self.peak_memory: int = 0
        if cache_limit_bytes is not None:
            # Only set tiling if the model has not already configured it.
            # Some models (e.g. ERNIE-Image) explicitly disable tiling by setting
            # tiling_config to a non-None sentinel; overwriting it here would cause
            # VAE decode artifacts such as red-channel banding.
            if model.tiling_config is None:
                self.model.tiling_config = TilingConfig()
            mx.set_cache_limit(cache_limit_bytes)
            mx.clear_cache()
            mx.reset_peak_memory()

    def call_before_loop(
        self,
        seed: int,
        prompt: str,
        latents: mx.array,
        config: Config,
        canny_image: PIL.Image.Image | None = None,
        depth_image: PIL.Image.Image | None = None,
    ) -> None:
        self.peak_memory = mx.get_peak_memory()
        self._free_stale_kv_cache()
        # Only evict when safe: single-seed run, or embeddings are cached in prompt_cache
        # (Flux1 caches embeddings so encoder isn't needed for subsequent seeds;
        #  Flux2 re-encodes each call and has no prompt_cache — keep encoder for multi-seed)
        has_cached_embeds = hasattr(self.model, "prompt_cache") and prompt in (self.model.prompt_cache or {})
        if self._num_seeds <= 1 or has_cached_embeds:
            self._delete_text_encoders()

    def call_in_loop(
        self,
        t: int,
        seed: int,
        prompt: str,
        latents: mx.array,
        config: Config,
        time_steps: tqdm,
    ) -> None:
        self.peak_memory = mx.get_peak_memory()

    def call_after_loop(
        self,
        seed: int,
        prompt: str,
        latents: mx.array,
        config: Config,
    ) -> None:
        self.peak_memory = mx.get_peak_memory()
        # Free K/V buffers here — before VAE decode — so they don't inflate peak
        # memory during decode. call_before_loop also calls this as a safety net
        # for generations interrupted by an exception.
        self._free_stale_kv_cache()
        if not self.keep_transformer:
            # Tighten the cache limit only now: the transformer is being dropped
            # so the remaining budget is just the VAE. Applying this limit during
            # the denoising loop is counterproductive on large models (9B+) because
            # the 1 GB free-list budget forces fresh OS allocations for every
            # intermediate activation, which can raise peak memory rather than
            # lower it.
            mx.set_cache_limit(self._cache_limit_bytes)
            mx.clear_cache()
            self._delete_transformer()
        else:
            # Clear Metal cache between seeds — without a cache limit, MLX accumulates
            # freed tensor buffers indefinitely across sequential denoising passes.
            gc.collect()
            mx.clear_cache()

    def _free_stale_kv_cache(self) -> None:
        """Free K/V arrays left over from the previous generation.

        The persistent kv_cache on KV-cache models keeps the ref K/V arrays from
        the last extract pass alive between generate_image calls. Zero them out
        in-place ([:] = 0) rather than setting to None: the [:] assignment drops
        the reference to the old Metal buffer (freeing it) while preserving Python
        object identity, which is required by both mx.compile(outputs=kv_state) in
        _make_extract_predict and mx.compile(inputs=kv_state) in _make_cached_predict.
        """
        kv = getattr(self.model, "_kv_cache", None)
        if kv is None:
            return
        for slot in (kv.double_keys, kv.double_values, kv.single_keys, kv.single_values):
            for i, arr in enumerate(slot):
                if isinstance(arr, mx.array):
                    slot[i][:] = 0
        mx.clear_cache()

    def _delete_text_encoders(self) -> None:
        # repeated image generation only works with the same prompt (cache)
        if hasattr(self.model, "clip_text_encoder"):
            self.model.clip_text_encoder = None
        if hasattr(self.model, "t5_text_encoder"):
            self.model.t5_text_encoder = None
        if hasattr(self.model, "image_encoder") and self.model.image_encoder is not None:
            self.model.image_encoder = None
        if hasattr(self.model, "image_embedder") and self.model.image_embedder is not None:
            self.model.image_embedder = None
        if hasattr(self.model, "depth_pro") and self.model.depth_pro is not None:
            self.model.depth_pro = None
        if hasattr(self.model, "qwen_vl_encoder") and self.model.qwen_vl_encoder is not None:
            self.model.qwen_vl_encoder = None
        # Clear VLM tokenizers from the tokenizers dict if present
        if hasattr(self.model, "tokenizers") and "qwen_vl" in self.model.tokenizers:
            self.model.tokenizers["qwen_vl"] = None
        gc.collect()
        mx.clear_cache()

    def _delete_transformer(self) -> None:
        self.model.transformer = None
        if hasattr(self.model, "transformer_controlnet"):
            self.model.transformer_controlnet = None
        gc.collect()
        mx.clear_cache()

    def memory_stats(self) -> str:
        self.peak_memory = mx.get_peak_memory()
        return f"Peak MLX memory: {self.peak_memory / 10**9:.2f} GB"
