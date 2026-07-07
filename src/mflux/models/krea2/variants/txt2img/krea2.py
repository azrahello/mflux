from pathlib import Path

import mlx.core as mx
from mlx import nn
from PIL import Image

from mflux.models.common.conditioning import ConditioningBands, GuidanceSchedule
from mflux.models.common.config import ModelConfig
from mflux.models.common.config.config import Config
from mflux.models.common.latent_creator.latent_creator import LatentCreator
from mflux.models.common.weights.saving.model_saver import ModelSaver
from mflux.models.krea2.krea2_initializer import Krea2Initializer
from mflux.models.krea2.latent_creator.krea2_latent_creator import Krea2LatentCreator
from mflux.models.krea2.model.krea2_edit_rebalance import Krea2EditRebalance
from mflux.models.krea2.model.krea2_sampler import Krea2Sampler
from mflux.models.krea2.model.krea2_text_encoder.prompt_encoder import Krea2PromptEncoder
from mflux.models.krea2.model.krea2_text_encoder.text_encoder import Krea2TextEncoder
from mflux.models.krea2.model.krea2_transformer.transformer import Krea2Transformer
from mflux.models.krea2.weights.krea2_weight_definition import Krea2WeightDefinition
from mflux.models.qwen.model.qwen_vae.qwen_vae import QwenVAE
from mflux.utils.apple_silicon import AppleSiliconUtil
from mflux.utils.exceptions import StopImageGenerationException
from mflux.utils.generated_image import GeneratedImage
from mflux.utils.image_util import ImageUtil

IMG_REF_TIER_TARGET_SIZE = {"low": 256, "normal": 512, "high": 1024, "max": 1280}


class Krea2(nn.Module):
    vae: QwenVAE
    transformer: Krea2Transformer
    text_encoder: Krea2TextEncoder

    def __init__(
        self,
        quantize: int | None = None,
        model_path: str | None = None,
        model_config: ModelConfig | None = None,
        lora_paths: list[str] | None = None,
        lora_scales: list[float] | None = None,
    ):
        super().__init__()
        Krea2Initializer.init(
            model=self,
            model_config=model_config or ModelConfig.krea2(),
            quantize=quantize,
            model_path=model_path,
            lora_paths=lora_paths,
            lora_scales=lora_scales,
        )

    def generate_image(
        self,
        seed: int,
        prompt: str,
        num_inference_steps: int = 8,
        height: int = 1024,
        width: int = 1024,
        guidance: float = 1.0,
        negative_prompt: str | None = None,
        image_path: Path | str | None = None,
        image_strength: float | None = None,
        scheduler: str | None = None,
        conditioning_weights: list[float] | None = None,
        conditioning_renormalize: bool = False,
        conditioning_multiplier: float = 1.0,
        guidance_schedule: str | None = None,
        img_ref_paths: list[Path | str] | None = None,
        img_ref_details: list[str] | None = None,
        img_ref_rebalance: bool = False,
    ) -> GeneratedImage:
        resolved_scheduler = Krea2._resolve_scheduler(scheduler)
        if image_strength is not None:
            image_strength = min(max(image_strength, 0.0), 0.999)
        is_img2img = image_path is not None and image_strength is not None and image_strength > 0.0

        # img2img keeps ComfyUI's denoise semantics: all num_inference_steps run,
        # compressed into the reduced noise range (Config's image_strength would
        # instead truncate the step grid, wasting steps a turbo model can't spare).
        config = Config(
            model_config=self.model_config,
            num_inference_steps=num_inference_steps,
            height=height,
            width=width,
            guidance=guidance,
            image_path=image_path,
            image_strength=None,
            scheduler=resolved_scheduler,
        )

        if is_img2img:
            sigmas = Krea2Sampler.flow_sigmas(
                num_inference_steps,
                self.model_config.sigma_max_shift,
                start=1.0 - image_strength,
            )
        else:
            sigmas = config.scheduler.sigmas
        latents = self._prepare_latents(seed=seed, config=config, sigmas=sigmas, is_img2img=is_img2img)
        if img_ref_rebalance and img_ref_paths:
            images = Krea2._load_img_ref_images(img_ref_paths, img_ref_details)
            cond_raw, cond_main, cond_ref, neg_embeds = Krea2PromptEncoder.encode_edit_rebalance_prompts(
                prompt=prompt,
                negative_prompt=negative_prompt,
                guidance=guidance,
                images=images,
                tokenizer=self.tokenizers["qwen3vl"],
                vision_tokenizer=self.tokenizers["qwen3vl_vision"],
                text_encoder=self.text_encoder,
            )
            compiled = Krea2EditRebalance.compile_conditioning(cond_raw, cond_main, cond_ref)
            scheduled_embeds = [cond_raw, compiled]
            plan = Krea2EditRebalance.per_step_plan(sigmas, num_inference_steps)
        else:
            embeds, neg_embeds = self._encode_prompts(
                prompt=prompt,
                negative_prompt=negative_prompt,
                guidance=guidance,
                img_ref_paths=img_ref_paths,
                img_ref_details=img_ref_details,
            )
            scheduled_embeds = [embeds]
            plan = [(0, 1.0)] * num_inference_steps
        if conditioning_weights is not None:
            scheduled_embeds = [
                ConditioningBands.scale_bands(e, conditioning_weights, conditioning_renormalize, conditioning_multiplier)
                for e in scheduled_embeds
            ]
            if neg_embeds is not None:
                neg_embeds = ConditioningBands.scale_bands(
                    neg_embeds, conditioning_weights, conditioning_renormalize, conditioning_multiplier
                )
        elif conditioning_multiplier != 1.0:
            scheduled_embeds = [e * conditioning_multiplier for e in scheduled_embeds]
            if neg_embeds is not None:
                neg_embeds = neg_embeds * conditioning_multiplier
        guidance_values = Krea2._resolve_guidance_values(guidance, guidance_schedule, num_inference_steps)
        mx.eval(latents, *scheduled_embeds)
        if neg_embeds is not None:
            mx.eval(neg_embeds)

        stepper = Krea2Sampler.make_stepper(resolved_scheduler, sigmas, seed)
        ctx = self.callbacks.start(seed=seed, prompt=prompt, config=config)
        ctx.before_loop(latents)
        predict = self._predict(self.transformer, neg_embeds)

        for t in config.time_steps:
            try:
                ts = sigmas[t].reshape(1)
                gv = mx.array(guidance_values[t]).reshape(1)
                cond_index, cond_multiplier = plan[t]
                step_embeds = scheduled_embeds[cond_index]
                if cond_multiplier != 1.0:
                    step_embeds = step_embeds * cond_multiplier
                v = predict(latents=latents, timestep=ts, guidance_value=gv, embeds=step_embeds)
                denoised = latents - sigmas[t] * v
                latents = stepper.step(t, latents, v, denoised)
                ctx.in_loop(t, latents)
                mx.eval(latents)
            except KeyboardInterrupt:  # noqa: PERF203
                ctx.interruption(t, latents)
                raise StopImageGenerationException(
                    f"Stopping image generation at step {t + 1}/{config.num_inference_steps}"
                )
        ctx.after_loop(latents)

        decoded = self._decode_latents(latents=latents)
        return ImageUtil.to_image(
            decoded_latents=decoded,
            config=config,
            seed=seed,
            prompt=prompt,
            quantization=self.bits,
            generation_time=config.time_steps.format_dict["elapsed"],
            lora_paths=self.lora_paths,
            lora_scales=self.lora_scales,
            negative_prompt=negative_prompt,
            image_path=config.image_path,
            image_strength=image_strength,
        )

    def save_model(self, base_path: str) -> None:
        ModelSaver.save_model(
            model=self,
            bits=self.bits,
            base_path=base_path,
            weight_definition=Krea2WeightDefinition,
        )

    def _encode_prompts(
        self,
        *,
        prompt: str,
        negative_prompt: str | None,
        guidance: float,
        img_ref_paths: list[Path | str] | None = None,
        img_ref_details: list[str] | None = None,
    ) -> tuple[mx.array, mx.array | None]:
        if img_ref_paths:
            images = Krea2._load_img_ref_images(img_ref_paths, img_ref_details)
            return Krea2PromptEncoder.encode_prompt_pair_with_images(
                prompt=prompt,
                negative_prompt=negative_prompt,
                guidance=guidance,
                images=images,
                tokenizer=self.tokenizers["qwen3vl"],
                vision_tokenizer=self.tokenizers["qwen3vl_vision"],
                text_encoder=self.text_encoder,
            )
        return Krea2PromptEncoder.encode_prompt_pair(
            prompt=prompt,
            negative_prompt=negative_prompt,
            guidance=guidance,
            tokenizer=self.tokenizers["qwen3vl"],
            text_encoder=self.text_encoder,
            prompt_cache=self.prompt_cache,
        )

    @staticmethod
    def _load_img_ref_images(paths: list[Path | str], details: list[str] | None) -> list[Image.Image]:
        details = details or []
        images = []
        for i, path in enumerate(paths):
            tier = details[i] if i < len(details) else "normal"
            target = IMG_REF_TIER_TARGET_SIZE.get(tier, IMG_REF_TIER_TARGET_SIZE["normal"])
            image = ImageUtil.load_image(path).convert("RGB")
            images.append(Krea2._resize_to_longest_side(image, target))
        return images

    @staticmethod
    def _resize_to_longest_side(image: Image.Image, target: int) -> Image.Image:
        width, height = image.size
        scale = target / max(width, height)
        new_width = max(1, round(width * scale))
        new_height = max(1, round(height * scale))
        # BOX = area averaging, matching the reference's common_upscale(..., "area").
        return image.resize((new_width, new_height), Image.BOX)

    def _prepare_latents(self, *, seed: int, config: Config, sigmas: mx.array, is_img2img: bool) -> mx.array:
        pure_noise = Krea2LatentCreator.create_noise(seed, config.height, config.width)
        if not is_img2img:
            return pure_noise

        encoded = LatentCreator.encode_image(
            vae=self.vae,
            image_path=config.image_path,
            height=config.height,
            width=config.width,
            tiling_config=self.tiling_config,
        )
        clean_latents = Krea2LatentCreator.pack_latents(encoded, config.height, config.width)
        return LatentCreator.add_noise_by_interpolation(clean=clean_latents, noise=pure_noise, sigma=float(sigmas[0]))

    def _decode_latents(self, *, latents: mx.array) -> mx.array:
        return self.vae.decode(latents)

    @staticmethod
    def _predict(
        transformer: Krea2Transformer,
        neg_embeds: mx.array | None,
    ):
        def predict(latents: mx.array, timestep: mx.array, guidance_value: mx.array, embeds: mx.array) -> mx.array:
            v = transformer(latents, timestep, embeds)
            if neg_embeds is not None:
                v_neg = transformer(latents, timestep, neg_embeds)
                v = v_neg + guidance_value * (v - v_neg)
            return v

        if AppleSiliconUtil.is_m1_or_m2():
            return predict
        return mx.compile(predict)

    @staticmethod
    def _resolve_guidance_values(
        guidance: float, guidance_schedule: str | None, num_inference_steps: int
    ) -> list[float]:
        if guidance_schedule is None:
            return [guidance] * num_inference_steps
        points = GuidanceSchedule.parse_schedule(guidance_schedule)
        return GuidanceSchedule.schedule_to_per_step(points, num_inference_steps)

    @staticmethod
    def _resolve_scheduler(scheduler: str | None) -> str:
        if scheduler is None or scheduler == "linear":
            return "er_sde"
        if scheduler in ("er_sde", "euler"):
            return scheduler
        raise ValueError(f"Unknown Krea-2 scheduler {scheduler!r}. Expected 'er_sde' or 'euler'.")
