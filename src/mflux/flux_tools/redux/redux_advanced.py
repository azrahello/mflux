from pathlib import Path
from typing import Literal

import mlx.core as mx
from mlx import nn
from tqdm import tqdm

from mflux.callbacks.callbacks import Callbacks
from mflux.config.config import Config
from mflux.config.model_config import ModelConfig
from mflux.config.runtime_config import RuntimeConfig
from mflux.error.exceptions import StopImageGenerationException
from mflux.flux.flux_initializer import FluxInitializer
from mflux.flux_tools.redux.redux_util_patch_downsampling import ReduxUtilDownsampling
from mflux.latent_creator.latent_creator import LatentCreator
from mflux.models.redux_encoder.redux_encoder import ReduxEncoder
from mflux.models.siglip_vision_transformer.siglip_vision_transformer import SiglipVisionTransformer
from mflux.models.text_encoder.clip_encoder.clip_encoder import CLIPEncoder
from mflux.models.text_encoder.prompt_encoder import PromptEncoder
from mflux.models.text_encoder.t5_encoder.t5_encoder import T5Encoder
from mflux.models.transformer.transformer import Transformer
from mflux.models.vae.vae import VAE
from mflux.post_processing.array_util import ArrayUtil
from mflux.post_processing.generated_image import GeneratedImage
from mflux.post_processing.image_util import ImageUtil
from mflux.tokenizer.clip_tokenizer import TokenizerCLIP
from mflux.tokenizer.t5_tokenizer import TokenizerT5


class Flux1ReduxAdvanced(nn.Module):
    """
    Advanced FLUX Redux model with enhanced control over image influence.

    This version extends the standard Redux functionality with different modes
    that control how strongly the reference image influences the generation.
    """

    vae: VAE
    image_encoder: SiglipVisionTransformer
    image_embedder: ReduxEncoder
    transformer: Transformer
    t5_text_encoder: T5Encoder
    clip_text_encoder: CLIPEncoder

    def __init__(
        self,
        model_config: ModelConfig,
        quantize: int | None = None,
        local_path: str | None = None,
        lora_paths: list[str] | None = None,
        lora_scales: list[float] | None = None,
    ):
        super().__init__()
        FluxInitializer.init_redux(
            flux_model=self,
            quantize=quantize,
            local_path=local_path,
            lora_paths=lora_paths,
            lora_scales=lora_scales,
        )

    def generate_image(
        self,
        seed: int,
        prompt: str,
        config: Config,
        redux_mode: Literal["lowest", "low", "medium", "high", "highest"] = "medium",
        interpolation_mode: Literal["area", "bicubic", "nearest"] = "area",
    ) -> GeneratedImage:
        """
        Generate an image using advanced redux control with patch downsampling.

        Args:
            seed: Random seed for generation
            prompt: Text prompt
            config: Generation configuration including image paths and strengths
            redux_mode: Control mode for image influence (ComfyUI compatible)
                - "lowest": Minimal image influence (5x downsampling) - prompt-driven
                - "low": Low image influence (4x downsampling)
                - "medium": Balanced influence (3x downsampling) - ComfyUI default
                - "high": High image influence (2x downsampling)
                - "highest": Maximum image influence (1x downsampling) - image-focused
            interpolation_mode: Interpolation method for downsampling (ComfyUI compatible)
                - "area": Weighted average (smooth, default)
                - "bicubic": Cubic interpolation (very smooth, may blur)
                - "nearest": Nearest neighbor (sharp, preserves details)

        Returns:
            Generated image with metadata
        """
        # 0. Create a new runtime config based on the model type and input parameters
        runtime_config = RuntimeConfig(config, self.model_config)
        time_steps = tqdm(range(runtime_config.init_time_step, runtime_config.num_inference_steps))

        # 1. Create the initial latents
        latents = LatentCreator.create(
            seed=seed,
            height=runtime_config.height,
            width=runtime_config.width,
        )

        # 2. Get prompt embeddings by fusing the prompt and image embeddings with advanced control
        prompt_embeds, pooled_prompt_embeds = Flux1ReduxAdvanced._get_prompt_embeddings(
            prompt=prompt,
            prompt_cache=self.prompt_cache,
            t5_tokenizer=self.t5_tokenizer,
            clip_tokenizer=self.clip_tokenizer,
            t5_text_encoder=self.t5_text_encoder,
            clip_text_encoder=self.clip_text_encoder,
            image_paths=runtime_config.redux_image_paths,
            image_encoder=self.image_encoder,
            image_embedder=self.image_embedder,
            image_strengths=runtime_config.redux_image_strengths,
            redux_mode=redux_mode,
            interpolation_mode=interpolation_mode,
        )  # fmt: off

        # (Optional) Call subscribers for beginning of loop
        Callbacks.before_loop(
            seed=seed,
            prompt=prompt,
            latents=latents,
            config=runtime_config,
        )  # fmt: off

        for t in time_steps:
            try:
                # 3.t Predict the noise
                noise = self.transformer(
                    t=t,
                    config=runtime_config,
                    hidden_states=latents,
                    prompt_embeds=prompt_embeds,
                    pooled_prompt_embeds=pooled_prompt_embeds,
                )

                # 4.t Take one denoise step
                dt = runtime_config.sigmas[t + 1] - runtime_config.sigmas[t]
                latents += noise * dt

                # (Optional) Call subscribers in-loop
                Callbacks.in_loop(
                    t=t,
                    seed=seed,
                    prompt=prompt,
                    latents=latents,
                    config=runtime_config,
                    time_steps=time_steps,
                )  # fmt: off

                # (Optional) Evaluate to enable progress tracking
                mx.eval(latents)

            except KeyboardInterrupt:  # noqa: PERF203
                Callbacks.interruption(
                    t=t,
                    seed=seed,
                    prompt=prompt,
                    latents=latents,
                    config=runtime_config,
                    time_steps=time_steps,
                )
                raise StopImageGenerationException(f"Stopping image generation at step {t + 1}/{len(time_steps)}")

        # (Optional) Call subscribers after loop
        Callbacks.after_loop(
            seed=seed,
            prompt=prompt,
            latents=latents,
            config=runtime_config,
        )  # fmt: off

        # 7. Decode the latent array and return the image
        latents = ArrayUtil.unpack_latents(latents=latents, height=runtime_config.height, width=runtime_config.width)
        decoded = self.vae.decode(latents)
        return ImageUtil.to_image(
            decoded_latents=decoded,
            config=runtime_config,
            seed=seed,
            prompt=prompt,
            quantization=self.bits,
            lora_paths=self.lora_paths,
            lora_scales=self.lora_scales,
            redux_image_paths=runtime_config.redux_image_paths,
            redux_image_strengths=runtime_config.redux_image_strengths,
            image_strength=runtime_config.image_strength,
            generation_time=time_steps.format_dict["elapsed"],
            redux_mode=redux_mode,
        )

    @staticmethod
    def _get_prompt_embeddings(
        prompt: str,
        prompt_cache: dict[str, tuple[mx.array, mx.array]],
        t5_tokenizer: TokenizerT5,
        clip_tokenizer: TokenizerCLIP,
        t5_text_encoder: T5Encoder,
        clip_text_encoder: CLIPEncoder,
        image_paths: list[str] | list[Path],
        image_encoder: SiglipVisionTransformer,
        image_embedder: ReduxEncoder,
        image_strengths: list[float] | None = None,
        redux_mode: Literal["lowest", "low", "medium", "high", "highest"] = "medium",
        interpolation_mode: Literal["area", "bicubic", "nearest"] = "area",
    ) -> tuple[mx.array, mx.array]:
        """
        Get text and image embeddings with patch downsampling control.

        Args:
            prompt: Text prompt
            prompt_cache: Cache for text embeddings
            t5_tokenizer: T5 tokenizer
            clip_tokenizer: CLIP tokenizer
            t5_text_encoder: T5 text encoder
            clip_text_encoder: CLIP text encoder
            image_paths: Paths to reference images
            image_encoder: SigLIP vision encoder
            image_embedder: Redux image embedder
            image_strengths: Optional per-image strength multipliers
            redux_mode: Redux downsampling mode
            interpolation_mode: Interpolation method for downsampling

        Returns:
            Tuple of (combined_embeddings, pooled_text_embeddings)
        """
        # 1. Encode the prompt
        prompt_embeds_txt, pooled_prompt_embeds = PromptEncoder.encode_prompt(
            prompt=prompt,
            prompt_cache=prompt_cache,
            t5_tokenizer=t5_tokenizer,
            clip_tokenizer=clip_tokenizer,
            t5_text_encoder=t5_text_encoder,
            clip_text_encoder=clip_text_encoder,
        )

        # 2. Encode the image(s) using patch downsampling
        image_embeds = ReduxUtilDownsampling.embed_images(
            image_paths=image_paths,
            image_encoder=image_encoder,
            image_embedder=image_embedder,
            image_strengths=image_strengths,
            redux_mode=redux_mode,
            interpolation_mode=interpolation_mode,
        )

        # 3. Join text embeddings with all image embeddings
        prompt_embeds = mx.concatenate([prompt_embeds_txt] + image_embeds, axis=1)

        return prompt_embeds, pooled_prompt_embeds
