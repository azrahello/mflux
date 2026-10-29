from contextlib import nullcontext
from types import SimpleNamespace

import mlx.core as mx
import pytest

from mflux.models.common.config.model_config import ModelConfig
from mflux.models.krea2.training_adapter import krea2_training_adapter as krea2_adapter_module

Krea2TrainingAdapter = krea2_adapter_module.Krea2TrainingAdapter


class _FakeKrea2:
    def __init__(self, *, quantize, model_config, model_path=None):  # noqa: ARG002
        self.recorded = []
        self.last_generate_kwargs = None

        def transformer(latents, timestep, embeds):
            self.recorded.append(SimpleNamespace(latents=latents, timestep=timestep, embeds=embeds))
            return latents * 2

        self.transformer = transformer

    def generate_image(self, **kwargs):
        self.last_generate_kwargs = kwargs
        return "preview-image"


@pytest.mark.fast
def test_predict_noise_feeds_sigma_as_timestep_and_returns_velocity(monkeypatch):
    # Krea 2 is single-stream flow matching: the transformer takes the sigma value as its timestep
    # and predicts the velocity directly, so predict_noise must pass sigmas[t] through and return
    # the transformer output unchanged (cast back to the latent dtype).
    monkeypatch.setattr(krea2_adapter_module, "Krea2", _FakeKrea2)
    adapter = Krea2TrainingAdapter(model_config=ModelConfig.krea2_raw(), quantize=None)

    latents_t = mx.ones((1, 16, 8, 8), dtype=mx.float32)
    sigmas = mx.array([0.9, 0.4, 0.0])
    cond = {"embeds": mx.zeros((1, 4, 8))}

    velocity = adapter.predict_noise(t=1, latents_t=latents_t, sigmas=sigmas, cond=cond, config=None)

    call = adapter._krea2.recorded[-1]
    assert call.timestep.shape == (1,)
    assert float(call.timestep[0]) == pytest.approx(0.4, abs=1e-3)
    assert velocity.dtype == latents_t.dtype
    assert velocity.shape == latents_t.shape
    assert mx.allclose(velocity, latents_t * 2, atol=1e-2).item()


@pytest.mark.fast
def test_caption_text_plain_and_ideogram_json():
    caption = Krea2TrainingAdapter._caption_text
    # Plain text passes through (stripped)
    assert caption("a red sports car") == "a red sports car"
    assert caption("  spaced caption  ") == "spaced caption"
    # Ideogram-style JSON caption: extract high_level_description
    js = '{"high_level_description": "a fox in the snow", "elements": []}'
    assert caption(js) == "a fox in the snow"
    # Malformed JSON is returned as-is
    assert caption("{not valid json}") == "{not valid json}"
    # JSON without the field is returned as-is
    assert caption('{"foo": "bar"}') == '{"foo": "bar"}'


@pytest.mark.fast
def test_preview_image_uses_training_guidance(monkeypatch):
    monkeypatch.setattr(krea2_adapter_module, "Krea2", _FakeKrea2)
    adapter = Krea2TrainingAdapter(model_config=ModelConfig.krea2_raw(), quantize=None)
    monkeypatch.setattr(adapter, "_assistant_disabled", lambda: nullcontext())

    training_spec = SimpleNamespace(steps=8, guidance=3.0)
    adapter.create_config(training_spec=training_spec, width=512, height=512)
    adapter.generate_preview_image(seed=7, prompt="test", width=512, height=512, steps=6)

    assert adapter._krea2.last_generate_kwargs is not None
    assert adapter._krea2.last_generate_kwargs["guidance"] == 3.0


class _TinyKrea2:
    # A real (tiny) transformer with the projector rebalance applied as Krea2Initializer does.
    def __init__(self, *, quantize, model_config, model_path=None, **projector):  # noqa: ARG002
        from mflux.models.krea2.krea2_initializer import Krea2Initializer
        from tests.test_krea2_lora_mapping import _tiny_transformer

        self.transformer = _tiny_transformer()
        self.transformer.txtfusion.projector.weight = mx.arange(1, 13, dtype=mx.float32).reshape(1, 12) / 10
        if projector:
            Krea2Initializer._apply_projector_rebalance(
                self, projector["projector_rebalance_weights"], projector["projector_rebalance_strength"]
            )


@pytest.mark.fast
def test_saved_adapter_carries_the_projector_rebalance_and_resume_does_not_apply_it_twice(monkeypatch, tmp_path):
    import json

    from mflux.models.common.lora.mapping.lora_loader import LoRALoader
    from mflux.models.common.training.lora.target_injector import inject_lora_targets
    from mflux.models.common.training.state.training_spec import LoraTargetSpec
    from mflux.models.krea2.weights.krea2_lora_mapping import Krea2LoRAMapping

    monkeypatch.setattr(krea2_adapter_module, "Krea2", _TinyKrea2)
    adapter = Krea2TrainingAdapter(
        model_config=ModelConfig.krea2_raw(), quantize=None, projector_rebalance_strength=0.05
    )
    patched = adapter.transformer().txtfusion.projector.weight
    targets = [LoraTargetSpec(module_path="blocks.0.attn.wq", rank=2)]
    inject_lora_targets(adapter.transformer(), targets)
    path = tmp_path / "adapter.safetensors"
    adapter.save_lora_adapter(
        path=path, training_spec=SimpleNamespace(lora_layers=SimpleNamespace(targets=targets), model="krea-2-raw")
    )

    weights, metadata = mx.load(str(path), return_metadata=True)
    assert set(weights) == {
        "transformer.blocks.0.attn.wq.lora_A.weight",
        "transformer.blocks.0.attn.wq.lora_B.weight",
        "transformer.txtfusion.projector.lora_A.weight",
        "transformer.txtfusion.projector.lora_B.weight",
    }
    assert json.loads(metadata["projector_rebalance"])["projector_rebalance_strength"] == 0.05

    # Inference on an unpatched model: loading the adapter alone reproduces the patch exactly.
    plain = _TinyKrea2(quantize=None, model_config=None).transformer
    LoRALoader.load_and_apply_lora(Krea2LoRAMapping.get_mapping(), plain, [str(path)], [1.0], bake_lora=True)
    assert mx.allclose(plain.txtfusion.projector.weight, patched, atol=1e-6).item()

    # Resume: the config re-patches the base, so the saved rank-1 is dropped instead of stacked.
    resumed = Krea2TrainingAdapter(
        model_config=ModelConfig.krea2_raw(), quantize=None, projector_rebalance_strength=0.05
    )
    resumed.load_lora_adapter(path=path)
    projector = resumed.transformer().txtfusion.projector
    assert type(projector).__name__ == "Linear"
    assert mx.array_equal(projector.weight, patched).item()
    resumed_wq = resumed.transformer().blocks[0].attn.wq
    assert type(resumed_wq).__name__ == "LoRALinear"
    assert mx.array_equal(resumed_wq.lora_A, adapter.transformer().blocks[0].attn.wq.lora_A).item()
