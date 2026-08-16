import random

from mflux.models.common.training.trainer import TrainingTrainer


def _mean_index(timestep_type: str, low: int = 1, high: int = 8, n: int = 20000) -> float:
    rng = random.Random(0)
    return sum(TrainingTrainer._sample_timestep_index(timestep_type, low, high, rng) for _ in range(n)) / n


def test_content_is_noisier_than_style():
    # The sigma schedule descends, so a low index is high noise. Per ai-toolkit's content_or_style
    # (and T2I-Adapter 3.4) structure is learned at high noise and texture at low noise: "content"
    # must land on the noisy end, "style" on the clean end, "sigmoid" between them.
    assert _mean_index("content") < _mean_index("sigmoid") < _mean_index("style")


def test_indices_stay_in_range():
    rng = random.Random(0)
    for timestep_type in ("style", "content", "sigmoid", "uniform"):
        for _ in range(2000):
            assert 1 <= TrainingTrainer._sample_timestep_index(timestep_type, 1, 8, rng) <= 7
