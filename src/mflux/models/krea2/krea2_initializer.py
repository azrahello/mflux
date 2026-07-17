import mlx.core as mx
from mlx import nn

import mflux.models.krea2.model.krea2_scheduler  # noqa: F401 — register er_sde/euler schedulers
from mflux.callbacks.callback_registry import CallbackRegistry
from mflux.models.common.config import ModelConfig
from mflux.models.common.conditioning import ConditioningBands
from mflux.models.common.lora.mapping.lora_loader import LoRALoader
from mflux.models.common.tokenizer import TokenizerLoader
from mflux.models.common.weights.loading.loaded_weights import LoadedWeights
from mflux.models.common.weights.loading.weight_applier import WeightApplier
from mflux.models.common.weights.loading.weight_loader import WeightLoader
from mflux.models.krea2.model.krea2_text_encoder.text_encoder import Krea2TextEncoder
from mflux.models.krea2.model.krea2_transformer.transformer import Krea2Transformer
from mflux.models.krea2.weights.krea2_lora_mapping import Krea2LoRAMapping
from mflux.models.krea2.weights.krea2_weight_definition import Krea2WeightDefinition
from mflux.models.qwen.model.qwen_vae.qwen_vae import QwenVAE


class Krea2Initializer:
    @staticmethod
    def init(
        model,
        model_config: ModelConfig,
        quantize: int | None,
        model_path: str | None = None,
        lora_paths: list[str] | None = None,
        lora_scales: list[float] | None = None,
        projector_rebalance_weights: str | None = None,
        projector_rebalance_strength: float = 0.05,
    ) -> None:
        path = model_path if model_path else model_config.model_name
        Krea2Initializer._init_config(model, model_config)
        weights = Krea2Initializer._load_weights(path)
        Krea2Initializer._init_tokenizers(model, path)
        Krea2Initializer._init_models(model, model_config)
        Krea2Initializer._apply_weights(model, weights, quantize)
        # Materialize the model and drop the raw file weights before LoRA:
        # baking evaluates the patched layers, and with `weights` still alive
        # every raw tensor it touches stays pinned alongside the merged copy
        # (~+17 GB peak on the dense model).
        del weights
        mx.eval(model)
        Krea2Initializer._apply_lora(model, lora_paths, lora_scales)
        Krea2Initializer._apply_projector_rebalance(model, projector_rebalance_weights, projector_rebalance_strength)
        mx.eval(model)
        mx.clear_cache()

    @staticmethod
    def _init_tokenizers(model, model_path: str) -> None:
        model.tokenizers = TokenizerLoader.load_all(
            definitions=Krea2WeightDefinition.get_tokenizers(),
            model_path=model_path,
        )

    @staticmethod
    def _init_config(model, model_config: ModelConfig) -> None:
        model.prompt_cache = {}
        model.model_config = model_config
        model.callbacks = CallbackRegistry()
        model.tiling_config = None
        model.lora_paths = None
        model.lora_scales = None

    @staticmethod
    def _load_weights(model_path: str) -> LoadedWeights:
        return WeightLoader.load(weight_definition=Krea2WeightDefinition, model_path=model_path)

    @staticmethod
    def _init_models(model, model_config: ModelConfig) -> None:
        model.vae = QwenVAE()
        model.transformer = Krea2Transformer(**(model_config.transformer_overrides or {}))
        model.text_encoder = Krea2TextEncoder()

    @staticmethod
    def _apply_weights(model, weights: LoadedWeights, quantize: int | None) -> None:
        model.bits = WeightApplier.apply_and_quantize(
            weights=weights,
            quantize_arg=quantize,
            weight_definition=Krea2WeightDefinition,
            models={
                "vae": model.vae,
                "transformer": model.transformer,
                "text_encoder": model.text_encoder,
            },
        )

    @staticmethod
    def _apply_lora(model, lora_paths: list[str] | None, lora_scales: list[float] | None) -> None:
        model.lora_paths, model.lora_scales = LoRALoader.load_and_apply_lora(
            lora_mapping=Krea2LoRAMapping.get_mapping(),
            transformer=model.transformer,
            lora_paths=lora_paths,
            lora_scales=lora_scales,
        )

    @staticmethod
    def _apply_projector_rebalance(model, weights: str | None, strength: float) -> None:
        # Reversible, LoRA-style diff on the learned txtfusion.projector (a
        # Linear(txtlayers -> 1) that fuses the tapped Qwen3-VL layers into one
        # conditioning): new_weight = weight + strength * diff. Ported from the
        # ComfyUI AzKrea2ProjectorRebalance node -- ComfyUI's model patcher applies
        # diffs to full-precision weights, so a quantized projector is dequantized
        # back to a plain Linear here rather than patched in its packed form.
        #
        # The applied patch is recorded on the model so generated images carry
        # it in their metadata (EXIF/JSON/XMP) and --config-from-metadata can
        # recreate the exact state; "none" = explicitly unpatched.
        model.projector_rebalance_weights = weights if weights else "none"
        model.projector_rebalance_strength = strength if weights else None
        if not weights:
            return
        diffs = ConditioningBands.parse_weights(weights)
        projector = model.transformer.txtfusion.projector
        if isinstance(projector, nn.QuantizedLinear):
            weight = mx.dequantize(
                projector.weight,
                scales=projector.scales,
                biases=projector.biases,
                group_size=projector.group_size,
                bits=projector.bits,
                mode=projector.mode,
            )
            projector = nn.Linear(weight.shape[1], weight.shape[0], bias=False)
            projector.weight = weight
            model.transformer.txtfusion.projector = projector
        if len(diffs) != projector.weight.shape[1]:
            raise ValueError(
                f"--projector-rebalance-weights must have {projector.weight.shape[1]} "
                f"comma-separated numbers, got {len(diffs)}"
            )
        diff = mx.array(diffs, dtype=mx.float32).reshape(1, -1)
        orig_dtype = projector.weight.dtype
        projector.weight = (projector.weight.astype(mx.float32) + strength * diff).astype(orig_dtype)
