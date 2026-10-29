from __future__ import annotations

import json
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import mlx.core as mx

from mflux.models.common.config.config import Config
from mflux.models.common.config.model_config import ModelConfig
from mflux.models.common.latent_creator.latent_creator import LatentCreator
from mflux.models.common.lora.mapping.lora_loader import LoRALoader
from mflux.models.common.training.adapters.base import TrainingAdapter
from mflux.models.common.training.state.training_spec import TrainingSpec
from mflux.models.common.training.utils import TrainingUtil
from mflux.models.krea2.krea2_initializer import Krea2Initializer
from mflux.models.krea2.latent_creator.krea2_latent_creator import Krea2LatentCreator
from mflux.models.krea2.model.krea2_text_encoder.prompt_encoder import Krea2PromptEncoder
from mflux.models.krea2.variants.txt2img.krea2 import Krea2
from mflux.models.krea2.weights.krea2_lora_mapping import Krea2LoRAMapping
from mflux.utils.version_util import VersionUtil


class Krea2TrainingAdapter(TrainingAdapter):
    """LoRA training adapter for Krea 2.

    Krea 2 is a single-stream MMDiT with no separate unconditional branch, so training is plain
    flow matching: the trainer noises a latent with z = (1 - sigma) * clean + sigma * noise and the
    target velocity is noise - clean, which is exactly what the transformer predicts at timestep =
    sigma (the same value the sampler feeds at inference). predict_noise therefore returns the
    transformer output unchanged. The transformer patchifies and builds RoPE position ids internally,
    so the adapter only has to VAE-encode the image and encode the caption. The base is quantized and
    the LoRA trains over it (QLoRA); gradient checkpointing keeps the 28-block graph within memory.
    """

    _PROJECTOR = "txtfusion.projector"

    def __init__(
        self,
        *,
        model_config: ModelConfig,
        quantize: int | None,
        model_path: str | None = None,
        projector_rebalance_weights: str | None = None,
        projector_rebalance_strength: float | None = None,
    ):
        self._model_config = model_config
        # The rebalance patches the base projector weight, so training and previews both run on it.
        self._projector_rebalance = Krea2Initializer.projector_rebalance_kwargs(
            projector_rebalance_weights, projector_rebalance_strength
        )
        self._krea2 = Krea2(
            quantize=quantize, model_config=model_config, model_path=model_path, **self._projector_rebalance
        )
        self._guidance: float | None = None
        self._preview_lora_path: str | None = None

    def model(self):
        return self._krea2

    def transformer(self):
        return self._krea2.transformer

    def create_config(self, training_spec: TrainingSpec, *, width: int, height: int) -> Config:
        self._guidance = training_spec.guidance
        return Config(
            model_config=self._model_config,
            num_inference_steps=training_spec.steps,
            width=width,
            height=height,
            guidance=training_spec.guidance,
        )

    def freeze_base(self) -> None:
        self._krea2.vae.freeze()
        self._krea2.transformer.freeze()
        if getattr(self._krea2, "text_encoder", None) is not None:
            self._krea2.text_encoder.freeze()
        # Recompute each block's activations in backward rather than storing them, so the ~12B
        # transformer's activation graph fits in memory. Only the LoRA matrices are trainable.
        self._krea2.transformer.gradient_checkpointing = True

    def set_gradient_checkpointing(self, enabled: bool) -> None:  # noqa: ARG002
        # Always on for Krea 2 (see freeze_base); the config flag cannot turn it off.
        self._krea2.transformer.gradient_checkpointing = True

    def encode_data(
        self,
        *,
        data_id: int,  # noqa: ARG002
        image_path: Path,
        prompt: str,
        width: int,
        height: int,
        input_image_path: Path | None = None,  # txt2img: unused  # noqa: ARG002
    ) -> tuple[mx.array, Any]:
        encoded = LatentCreator.encode_image(
            vae=self._krea2.vae,
            image_path=image_path,
            height=height,
            width=width,
            tiling_config=getattr(self._krea2, "tiling_config", None),
        )
        # Tiled VAE encode (low_ram) returns 5D (B, C, 1, H, W); the transformer expects 4D.
        if encoded.ndim == 5 and encoded.shape[2] == 1:
            encoded = encoded[:, :, 0, :, :]
        clean_latents = Krea2LatentCreator.pack_latents(encoded, height, width).astype(mx.float32)

        caption = Krea2TrainingAdapter._caption_text(prompt)
        embeds = Krea2PromptEncoder.encode_prompt(
            prompt=caption,
            tokenizer=self._krea2.tokenizers["qwen3vl"],
            text_encoder=self._krea2.text_encoder,
        )

        mx.eval(clean_latents, embeds)
        return clean_latents, {"embeds": embeds}

    def predict_noise(self, *, t: int, latents_t: mx.array, sigmas: mx.array, cond: Any, config: Config) -> mx.array:  # noqa: ARG002
        # The transformer takes the sigma value as its timestep (sampler feeds sigmas[t] at inference)
        # and predicts the flow-matching velocity noise - clean, which is the target the loss expects.
        timestep = sigmas[t].reshape(1)
        velocity = self._krea2.transformer(latents_t.astype(ModelConfig.precision), timestep, cond["embeds"])
        return velocity.astype(latents_t.dtype)

    def generate_preview_image(
        self,
        *,
        seed: int,
        prompt: str,
        width: int,
        height: int,
        steps: int,
        image_paths: list[Path | str] | None = None,  # txt2img: unused  # noqa: ARG002
    ):
        with self._assistant_disabled(), self._preview_lora():
            return self._krea2.generate_image(
                seed=seed,
                prompt=prompt,
                num_inference_steps=steps,
                height=height,
                width=width,
                guidance=self._guidance if self._guidance is not None else 1.0,
            )

    def save_lora_adapter(self, *, path: Path, training_spec: TrainingSpec) -> None:
        weights: dict[str, mx.array] = {}
        for target in training_spec.lora_layers.targets:
            if target.blocks is not None:
                for b in target.blocks.get_blocks():
                    self._append_train_lora_weights(weights, target.module_path.format(block=b))
            else:
                self._append_train_lora_weights(weights, target.module_path)
        metadata = {"mflux_version": VersionUtil.get_mflux_version(), "model": training_spec.model}
        if self._projector_rebalance:
            # The rebalance as an exact rank-1 LoRA (B @ A = strength * diff, no alpha), so the saved
            # adapter carries it and inference needs no --projector-rebalance-* flag.
            weights.update(self._projector_rebalance_lora())
            metadata["projector_rebalance"] = json.dumps(self._projector_rebalance)
        mx.save_safetensors(str(path), weights, metadata=metadata)

    def _projector_rebalance_lora(self) -> dict[str, mx.array]:
        values = self._projector_rebalance["projector_rebalance_weights"].split(",")
        diff = mx.array([float(v) for v in values], dtype=mx.float32).reshape(1, -1)
        strength = self._projector_rebalance["projector_rebalance_strength"]
        return {
            f"transformer.{self._PROJECTOR}.lora_A.weight": strength * diff,  # [rank=1, in=12]
            f"transformer.{self._PROJECTOR}.lora_B.weight": mx.ones((1, 1), dtype=mx.float32),  # [out=1, rank=1]
        }

    def load_lora_adapter(self, *, path: str | Path) -> None:
        if self._projector_rebalance:
            # Resume: the config re-applies the rebalance to the base, so the rank-1 copy saved in the
            # adapter is dropped; loading it too would patch the projector twice.
            weights = {k: v for k, v in mx.load(str(path)).items() if f".{self._PROJECTOR}." not in k}
            if not weights:
                return
            with tempfile.TemporaryDirectory() as tmp:
                filtered = Path(tmp) / Path(path).name
                mx.save_safetensors(str(filtered), weights)
                self._load_train_lora(filtered)
                mx.eval(self._krea2.transformer.parameters())
            return
        self._load_train_lora(path)

    def _load_train_lora(self, path: str | Path) -> None:
        LoRALoader.load_and_apply_lora(
            lora_mapping=Krea2LoRAMapping.get_mapping(),
            transformer=self._krea2.transformer,
            lora_paths=[str(path)],
            lora_scales=[1.0],
            role="train",
            bake_lora=False,
        )

    def load_training_adapter(self, *, path: str | Path, scale: float = 1.0) -> None:
        LoRALoader.load_and_apply_lora(
            lora_mapping=Krea2LoRAMapping.get_mapping(),
            transformer=self._krea2.transformer,
            lora_paths=[str(path)],
            lora_scales=[float(scale)],
            role="assistant",
            bake_lora=False,
        )

    def set_preview_adapter(self, *, path: str | Path) -> None:
        # Loaded at the start of every preview and removed after it, never present while training.
        self._preview_lora_path = str(path)

    @contextmanager
    def _preview_lora(self):
        if self._preview_lora_path is None:
            yield
            return
        with TrainingUtil.transient_loras(self._krea2.transformer):
            LoRALoader.load_and_apply_lora(
                lora_mapping=Krea2LoRAMapping.get_mapping(),
                transformer=self._krea2.transformer,
                lora_paths=[self._preview_lora_path],
                lora_scales=[1.0],
                role="preview",
                bake_lora=False,
            )
            yield

    def _assistant_disabled(self):
        return TrainingUtil.assistant_disabled(self._krea2.transformer)

    def _append_train_lora_weights(self, weights: dict[str, mx.array], module_path: str) -> None:
        train_lora = TrainingUtil.get_train_lora(self._krea2.transformer, module_path)
        weights[f"transformer.{module_path}.lora_A.weight"] = mx.transpose(train_lora.lora_A)
        weights[f"transformer.{module_path}.lora_B.weight"] = mx.transpose(train_lora.lora_B)

    @staticmethod
    def _caption_text(prompt: str) -> str:
        # Captions may be plain text or an Ideogram-style JSON object; Krea 2's text encoder takes a
        # natural-language describe-the-image prompt, so extract high_level_description from JSON.
        text = prompt.strip()
        if not (text.startswith("{") and text.endswith("}")):
            return text
        try:
            data = json.loads(text)
        except (ValueError, TypeError):
            return text
        if isinstance(data, dict):
            desc = data.get("high_level_description")
            if isinstance(desc, str) and desc.strip():
                return desc.strip()
        return text
