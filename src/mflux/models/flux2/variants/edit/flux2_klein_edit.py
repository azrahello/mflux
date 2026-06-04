from pathlib import Path

import mlx.core as mx
from mlx import nn

from mflux.models.common.config.config import Config
from mflux.models.common.config.model_config import ModelConfig
from mflux.models.flux2.flux2_initializer import Flux2Initializer
from mflux.models.flux2.model.flux2_text_encoder.qwen3_text_encoder import Qwen3TextEncoder
from mflux.models.flux2.model.flux2_transformer.flux2_kv_cache import Flux2KVCache
from mflux.models.flux2.model.flux2_transformer.transformer import Flux2Transformer
from mflux.models.flux2.model.flux2_vae.vae import Flux2VAE
from mflux.models.flux2.variants.edit.flux2_klein_edit_helpers import _Flux2KleinEditHelpers
from mflux.utils.apple_silicon import AppleSiliconUtil
from mflux.utils.exceptions import StopImageGenerationException
from mflux.utils.generated_image import GeneratedImage
from mflux.utils.image_util import ImageUtil


class Flux2KleinEdit(nn.Module):
    vae: Flux2VAE
    transformer: Flux2Transformer
    text_encoder: Qwen3TextEncoder

    def __init__(
        self,
        quantize: int | None = None,
        model_path: str | None = None,
        lora_paths: list[str] | None = None,
        lora_scales: list[float] | None = None,
        model_config: ModelConfig | None = None,
    ):
        super().__init__()
        Flux2Initializer.init(
            model=self,
            quantize=quantize,
            model_path=model_path,
            lora_paths=lora_paths,
            lora_scales=lora_scales,
            model_config=model_config or ModelConfig.flux2_klein_4b(),
        )
        self._compiled_predict = None

        # Persistent KV-cache state — one pre-allocated buffer set per output resolution.
        # Buffers are updated in-place each generation; _compiled_cached_predict is rebuilt
        # after each extract pass so it captures fresh K/V constants.
        self._kv_cache: Flux2KVCache | None = None
        self._kv_resolution: tuple[int, int] | None = None
        self._compiled_cached_predict = None

    def generate_image(
        self,
        seed: int,
        prompt: str,
        num_inference_steps: int = 4,
        height: int = 1024,
        width: int = 1024,
        guidance: float = 1.0,
        image_paths: list[Path | str] | None = None,
        image_strength: float | None = None,
        scheduler: str = "flow_match_euler_discrete",
        use_kv_cache: bool | None = None,
    ) -> GeneratedImage:
        # For metadata + dimension inference purposes, pick a primary reference image (if any).
        primary_image_path = None
        if image_paths:
            primary_image_path = image_paths[0]

        # 0. Create a new config based on the model type and input parameters
        config = Config(
            model_config=self.model_config,
            num_inference_steps=num_inference_steps,
            height=height,
            width=width,
            guidance=guidance,
            image_path=primary_image_path,
            image_strength=image_strength,
            scheduler=scheduler,
        )
        # 1. Encode prompt(s)
        prompt_embeds, text_ids, negative_prompt_embeds, negative_text_ids = self._encode_prompt_pair(
            prompt=prompt,
            negative_prompt=" ",
            guidance=guidance,
        )

        # 2. Prepare latents
        latents, latent_ids, latent_height, latent_width = _Flux2KleinEditHelpers.prepare_generation_latents(
            seed=seed,
            height=config.height,
            width=config.width,
        )

        # 3. Reference image conditioning (edit-style, concat reference tokens)
        image_latents, image_latent_ids = _Flux2KleinEditHelpers.prepare_reference_image_conditioning(
            vae=self.vae,
            tiling_config=self.tiling_config,
            image_paths=image_paths,
            height=config.height,
            width=config.width,
            batch_size=latents.shape[0],
        )

        # KV-cache opt-in. Defaults to ON when the model declares support
        # (FLUX.2-klein-9b-kv) AND a reference image is present, since the
        # cache is meaningless without static reference tokens to skip.
        cache_enabled = (
            (use_kv_cache if use_kv_cache is not None else self.model_config.supports_kv_cache)
            and image_latents is not None
            and image_latents.shape[1] > 0
        )

        # Initialise or reuse the persistent KV-cache. A new object (and new
        # compiled cached-predict) is needed only when the output resolution
        # changes (different K/V buffer shape).
        if cache_enabled:
            resolution = (config.height, config.width)
            if self._kv_cache is None or self._kv_resolution != resolution:
                ref_attn = self.transformer.transformer_blocks[0].attn
                self._kv_cache = Flux2KVCache(
                    num_double_layers=len(self.transformer.transformer_blocks),
                    num_single_layers=len(self.transformer.single_transformer_blocks),
                    num_ref_tokens=image_latents.shape[1],
                    num_heads=ref_attn.heads,
                    head_dim=ref_attn.dim_head,
                )
                self._kv_resolution = resolution
                self._compiled_cached_predict = None

        kv_cache = self._kv_cache if cache_enabled else None

        # 4. Denoising loop
        ctx = self.callbacks.start(seed=seed, prompt=prompt, config=config)
        ctx.before_loop(latents)
        predict = None
        if not cache_enabled:
            if self._compiled_predict is None:
                self._compiled_predict = self._predict(self.transformer)
            predict = self._compiled_predict
        for step_idx, t in enumerate(config.time_steps):
            try:
                if cache_enabled and step_idx == 0:
                    # Step 0: full forward [txt, target, ref]; populates KV cache.
                    kv_cache.configure(
                        mode="extract",
                        num_ref_tokens=image_latents.shape[1],
                        num_txt_tokens=prompt_embeds.shape[1],
                    )
                    extract_predict = self._make_extract_predict(self.transformer, kv_cache)
                    noise = extract_predict(
                        latents=latents,
                        image_latents=image_latents,
                        latent_ids=latent_ids,
                        image_latent_ids=image_latent_ids,
                        prompt_embeds=prompt_embeds,
                        text_ids=text_ids,
                        negative_prompt_embeds=negative_prompt_embeds,
                        negative_text_ids=negative_text_ids,
                        guidance=guidance,
                        timestep=config.scheduler.timesteps[t],
                    )
                elif cache_enabled:
                    # Steps 1+: target-only input; ref K/V spliced from cache.
                    noise = self._compiled_cached_predict(
                        latents=latents,
                        latent_ids=latent_ids,
                        prompt_embeds=prompt_embeds,
                        text_ids=text_ids,
                        negative_prompt_embeds=negative_prompt_embeds,
                        negative_text_ids=negative_text_ids,
                        guidance=guidance,
                        timestep=config.scheduler.timesteps[t],
                    )
                else:
                    noise = predict(
                        latents=latents,
                        image_latents=image_latents,
                        latent_ids=latent_ids,
                        image_latent_ids=image_latent_ids,
                        prompt_embeds=prompt_embeds,
                        text_ids=text_ids,
                        negative_prompt_embeds=negative_prompt_embeds,
                        negative_text_ids=negative_text_ids,
                        guidance=guidance,
                        timestep=config.scheduler.timesteps[t],
                    )

                # 5.t Take one denoise step
                latents = config.scheduler.step(
                    noise=noise, timestep=t, latents=latents, sigmas=config.scheduler.sigmas
                )

                ctx.in_loop(t, latents)
                if cache_enabled and step_idx == 0:
                    # Materialise latents + all K/V buffers in one Metal pass.
                    # The buffers must be concrete before mx.compile captures them
                    # as inputs in _make_cached_predict.
                    mx.eval(
                        latents,
                        *kv_cache.double_keys,
                        *kv_cache.double_values,
                        *kv_cache.single_keys,
                        *kv_cache.single_values,
                    )
                    kv_cache.configure(
                        mode="cached",
                        num_ref_tokens=image_latents.shape[1],
                        num_txt_tokens=prompt_embeds.shape[1],
                    )
                    # Rebuild every generation: K/V constants differ per reference image,
                    # so the compiled graph from the previous generation is stale.
                    self._compiled_cached_predict = self._make_cached_predict(self.transformer, kv_cache)
                else:
                    mx.eval(latents)
            except KeyboardInterrupt:  # noqa: PERF203
                ctx.interruption(t, latents)
                raise StopImageGenerationException(
                    f"Stopping image generation at step {t + 1}/{config.num_inference_steps}"
                )

        ctx.after_loop(latents)

        # 6. Decode latents
        packed_latents = latents.reshape(latents.shape[0], latent_height, latent_width, latents.shape[-1]).transpose(0, 3, 1, 2)  # fmt: off
        decoded = self.vae.decode_packed_latents(packed_latents)
        return ImageUtil.to_image(
            decoded_latents=decoded,
            config=config,
            seed=seed,
            prompt=prompt,
            negative_prompt=None,
            quantization=self.bits,
            image_paths=image_paths,
            image_path=config.image_path,
            generation_time=config.time_steps.format_dict["elapsed"],
        )

    def _encode_prompt_pair(
        self,
        *,
        prompt: str,
        negative_prompt: str | None,
        guidance: float,
    ) -> tuple[mx.array, mx.array, mx.array | None, mx.array | None]:
        prompt_embeds, text_ids = _Flux2KleinEditHelpers.encode_text(
            prompt,
            tokenizer=self.tokenizers["qwen3"],
            text_encoder=self.text_encoder,
        )
        negative_prompt_embeds = None
        negative_text_ids = None
        if guidance is not None and guidance > 1.0 and negative_prompt is not None:
            negative_prompt_embeds, negative_text_ids = _Flux2KleinEditHelpers.encode_text(
                negative_prompt,
                tokenizer=self.tokenizers["qwen3"],
                text_encoder=self.text_encoder,
            )
        return prompt_embeds, text_ids, negative_prompt_embeds, negative_text_ids

    def _predict(self, transformer):
        """Full-input predict for the no-KV-cache path. All args are arrays → compilable."""

        def predict(
            latents: mx.array,
            image_latents: mx.array,
            latent_ids: mx.array,
            image_latent_ids: mx.array,
            prompt_embeds: mx.array,
            text_ids: mx.array,
            negative_prompt_embeds: mx.array | None,
            negative_text_ids: mx.array | None,
            guidance: float,
            timestep: mx.array,
        ) -> mx.array:
            hidden_states = mx.concatenate([latents, image_latents], axis=1)
            img_ids = mx.concatenate([latent_ids, image_latent_ids], axis=1)

            noise = transformer(
                hidden_states=hidden_states,
                encoder_hidden_states=prompt_embeds,
                timestep=timestep,
                img_ids=img_ids,
                txt_ids=text_ids,
                guidance=None,
            )
            noise = noise[:, : latents.shape[1]]
            if negative_prompt_embeds is not None and negative_text_ids is not None:
                negative_noise = transformer(
                    hidden_states=hidden_states,
                    encoder_hidden_states=negative_prompt_embeds,
                    timestep=timestep,
                    img_ids=img_ids,
                    txt_ids=negative_text_ids,
                    guidance=None,
                )
                negative_noise = negative_noise[:, : latents.shape[1]]
                noise = negative_noise + guidance * (noise - negative_noise)
            return noise

        if AppleSiliconUtil.is_m1_or_m2():
            return predict
        return mx.compile(predict)

    @staticmethod
    def _make_extract_predict(transformer, kv_cache: Flux2KVCache):
        """Build the step-0 predict closure (KV extract pass).

        Not compiled: this path runs exactly once per generation, so there is
        nothing to amortise. Running uncompiled also avoids the mx.compile
        limitation where [:] side-effects on captured arrays are not visible
        to the Python wrappers after the compiled call returns.
        """

        def predict(
            latents: mx.array,
            image_latents: mx.array,
            latent_ids: mx.array,
            image_latent_ids: mx.array,
            prompt_embeds: mx.array,
            text_ids: mx.array,
            negative_prompt_embeds: mx.array | None,
            negative_text_ids: mx.array | None,
            guidance: float,
            timestep: mx.array,
        ) -> mx.array:
            hidden_states = mx.concatenate([latents, image_latents], axis=1)
            img_ids = mx.concatenate([latent_ids, image_latent_ids], axis=1)

            noise = transformer(
                hidden_states=hidden_states,
                encoder_hidden_states=prompt_embeds,
                timestep=timestep,
                img_ids=img_ids,
                txt_ids=text_ids,
                guidance=None,
                kv_cache=kv_cache,
            )
            noise = noise[:, : latents.shape[1]]
            if negative_prompt_embeds is not None and negative_text_ids is not None:
                negative_noise = transformer(
                    hidden_states=hidden_states,
                    encoder_hidden_states=negative_prompt_embeds,
                    timestep=timestep,
                    img_ids=img_ids,
                    txt_ids=negative_text_ids,
                    guidance=None,
                    kv_cache=None,  # negative forward must not overwrite cached ref K/V
                )
                negative_noise = negative_noise[:, : latents.shape[1]]
                noise = negative_noise + guidance * (noise - negative_noise)
            return noise

        return predict

    @staticmethod
    def _make_cached_predict(transformer, kv_cache: Flux2KVCache):
        """Build the cached-mode predict closure for steps 1+.

        kv_cache is captured in the closure; all K/V arrays are frozen as
        constants at the first call (trace time). The closure is rebuilt after
        each extract pass so the constants reflect the current reference image.
        """

        def predict(
            latents: mx.array,
            latent_ids: mx.array,
            prompt_embeds: mx.array,
            text_ids: mx.array,
            negative_prompt_embeds: mx.array | None,
            negative_text_ids: mx.array | None,
            guidance: float,
            timestep: mx.array,
        ) -> mx.array:
            noise = transformer(
                hidden_states=latents,
                encoder_hidden_states=prompt_embeds,
                timestep=timestep,
                img_ids=latent_ids,
                txt_ids=text_ids,
                guidance=None,
                kv_cache=kv_cache,
            )
            noise = noise[:, : latents.shape[1]]
            if negative_prompt_embeds is not None and negative_text_ids is not None:
                negative_noise = transformer(
                    hidden_states=latents,
                    encoder_hidden_states=negative_prompt_embeds,
                    timestep=timestep,
                    img_ids=latent_ids,
                    txt_ids=negative_text_ids,
                    guidance=None,
                    kv_cache=kv_cache,
                )
                negative_noise = negative_noise[:, : latents.shape[1]]
                noise = negative_noise + guidance * (noise - negative_noise)
            return noise

        if AppleSiliconUtil.is_m1_or_m2():
            return predict
        return mx.compile(predict)
