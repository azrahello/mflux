import mlx.core as mx
import numpy as np
import pytest

from mflux.models.krea2.model.krea2_edit_rebalance import (
    KREA2_EDIT_N_BANDS,
    KREA2_EDIT_SUBJECT_WEIGHTS,
    Krea2EditRebalance,
)
from mflux.models.krea2.model.krea2_sampler import Krea2Sampler

BAND_DIM = 8
FEATURE_DIM = KREA2_EDIT_N_BANDS * BAND_DIM


class TestKrea2EditRebalance:
    @pytest.mark.fast
    def test_subject_weights_isolate_band_8(self):
        assert len(KREA2_EDIT_SUBJECT_WEIGHTS) == KREA2_EDIT_N_BANDS
        assert KREA2_EDIT_SUBJECT_WEIGHTS[8] == 12.0
        assert sum(w != 0.0 for w in KREA2_EDIT_SUBJECT_WEIGHTS) == 1

    @pytest.mark.fast
    def test_compiled_conditioning_lives_in_subject_band(self):
        cond_raw = mx.random.normal((1, 5, FEATURE_DIM))
        cond_main = mx.random.normal((1, 9, FEATURE_DIM))
        cond_ref = mx.random.normal((1, 7, FEATURE_DIM))

        compiled = Krea2EditRebalance.compile_conditioning(cond_raw, cond_main, cond_ref)

        assert compiled.shape == cond_main.shape
        bands = np.array(compiled).reshape(1, 9, KREA2_EDIT_N_BANDS, BAND_DIM)
        # The recipe zeroes every band but the subject band before the guidance
        # chain, and dissimilarity projection never re-populates a zero band.
        for band in range(KREA2_EDIT_N_BANDS):
            if band == 8:
                assert np.abs(bands[..., band, :]).max() > 0.0
            else:
                assert np.allclose(bands[..., band, :], 0.0, atol=1e-5)

    @pytest.mark.fast
    def test_per_step_plan_matches_comfyui_timing_at_8_steps(self):
        sigmas = Krea2Sampler.flow_sigmas(8)

        plan = Krea2EditRebalance.per_step_plan(sigmas, num_steps=8)

        # ComfyUI schedule percents are sigma thresholds under its linear 1.15
        # shift; mapped onto mflux's exp(1.15) sigmas the raw phase covers the
        # first 4 of 8 steps and the extreme x12/x20.5 tail never fires.
        assert [index for index, _ in plan] == [0, 0, 0, 0, 1, 1, 1, 1]
        multipliers = [multiplier for _, multiplier in plan]
        assert multipliers[:4] == [1.0] * 4
        assert multipliers[4] == pytest.approx(0.8375)
        assert multipliers[5] == pytest.approx(0.9875)
        assert multipliers[6] == pytest.approx(1.1375)
        assert multipliers[7] == pytest.approx(1.3625)
