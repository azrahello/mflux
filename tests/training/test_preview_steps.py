from pathlib import Path
from unittest.mock import MagicMock

import pytest

from mflux.models.common.training.state.training_spec import TrainingSpec
from mflux.models.common.training.trainer import TrainingTrainer


def _spec(*, steps: int, preview_steps: int | None) -> MagicMock:
    spec = MagicMock(spec=TrainingSpec)
    spec.steps = steps
    spec.preview_steps = preview_steps
    spec.is_edit = False
    spec.seed = 0
    spec.monitoring = MagicMock()
    spec.monitoring.preview_prompts = ["a prompt"]
    spec.monitoring.preview_prompt_names = ["preview"]
    spec.monitoring.preview_images = None
    spec.monitoring.preview_width = 512
    spec.monitoring.preview_height = 512
    return spec


def _preview_steps_used(spec: MagicMock) -> int:
    adapter = MagicMock()
    TrainingTrainer._generate_previews(adapter, spec, MagicMock())
    return adapter.generate_preview_image.call_args.kwargs["steps"]


@pytest.mark.fast
def test_preview_steps_overrides_the_training_grid():
    # 'steps' sizes the sigma grid training samples from; previews must use their own count
    # so a large training grid does not turn every preview into a 1000-step generation.
    assert _preview_steps_used(_spec(steps=1000, preview_steps=8)) == 8


@pytest.mark.fast
def test_preview_falls_back_to_steps_when_unset():
    assert _preview_steps_used(_spec(steps=8, preview_steps=None)) == 8


def _conf(tmp_path: Path, **overrides) -> dict:
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / "01.txt").write_text("photo of sks dog\n", encoding="utf-8")
    (data_dir / "01.jpeg").write_bytes(b"")
    return {
        "model": "dev",
        "seed": 42,
        "steps": 20,
        "guidance": 3.0,
        "training_loop": {"num_epochs": 1, "batch_size": 1},
        "optimizer": {"name": "AdamW", "learning_rate": 1e-4},
        "checkpoint": {"output_path": str(tmp_path / "out"), "save_frequency": 10},
        "lora_layers": {
            "targets": [
                {"module_path": "transformer_blocks.{block}.attn.to_q", "blocks": {"start": 0, "end": 1}, "rank": 4}
            ]
        },
        "data": "data",
        **overrides,
    }


@pytest.mark.fast
def test_preview_steps_is_read_from_the_config(tmp_path: Path):
    spec = TrainingSpec.from_conf(_conf(tmp_path, preview_steps=8), str(tmp_path / "train.json"), new_folder=False)
    assert spec.preview_steps == 8
    assert spec.steps == 20


@pytest.mark.fast
def test_preview_steps_defaults_to_none(tmp_path: Path):
    spec = TrainingSpec.from_conf(_conf(tmp_path), str(tmp_path / "train.json"), new_folder=False)
    assert spec.preview_steps is None


@pytest.mark.fast
def test_preview_steps_must_be_positive(tmp_path: Path):
    with pytest.raises(ValueError, match="preview_steps"):
        TrainingSpec.from_conf(_conf(tmp_path, preview_steps=0), str(tmp_path / "train.json"), new_folder=False)
