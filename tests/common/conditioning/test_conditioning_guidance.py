import mlx.core as mx
import numpy as np
import pytest

from mflux.models.common.conditioning import ConditioningGuidance


def numpy_project(cond: np.ndarray, ref: np.ndarray, strength: float, n_bands: int) -> np.ndarray:
    band_dim = cond.shape[-1] // n_bands
    c = cond.reshape(*cond.shape[:-1], n_bands, band_dim).astype(np.float32)
    r = ref.reshape(*ref.shape[:-1], n_bands, band_dim).astype(np.float32)
    direction = c.mean(axis=1) - r.mean(axis=1)
    direction = direction / np.sqrt((direction**2).sum(axis=-1, keepdims=True) + 1e-8)
    projection = (c * direction[:, None]).sum(axis=-1, keepdims=True)
    out = c + strength * projection * direction[:, None]
    return out.reshape(cond.shape)


class TestConditioningGuidanceProject:
    @pytest.mark.fast
    def test_matches_reference_math(self):
        cond = mx.random.normal((2, 6, 4 * 5))
        ref = mx.random.normal((2, 9, 4 * 5))

        out = ConditioningGuidance.project(cond, ref, strength=0.7, n_bands=4)
        expected = numpy_project(np.array(cond), np.array(ref), strength=0.7, n_bands=4)

        assert np.allclose(np.array(out), expected, atol=1e-5)

    @pytest.mark.fast
    def test_zero_strength_is_no_op(self):
        cond = mx.random.normal((1, 8, 12 * 16))
        ref = mx.random.normal((1, 8, 12 * 16))

        out = ConditioningGuidance.project(cond, ref, strength=0.0, n_bands=12)

        assert np.allclose(np.array(out), np.array(cond), atol=1e-6)

    @pytest.mark.fast
    def test_identical_reference_is_no_op(self):
        cond = mx.random.normal((1, 8, 12 * 16))

        out = ConditioningGuidance.project(cond, cond, strength=2.0, n_bands=12)

        assert np.allclose(np.array(out), np.array(cond), atol=1e-5)

    @pytest.mark.fast
    def test_zeroed_bands_stay_zero(self):
        n_bands, band_dim = 4, 3
        cond = np.random.default_rng(0).normal(size=(1, 5, n_bands * band_dim)).astype(np.float32)
        cond.reshape(1, 5, n_bands, band_dim)[..., 2, :] = 0.0
        ref = mx.random.normal((1, 5, n_bands * band_dim))

        out = np.array(ConditioningGuidance.project(mx.array(cond), ref, strength=1.0, n_bands=n_bands))

        assert np.allclose(out.reshape(1, 5, n_bands, band_dim)[..., 2, :], 0.0, atol=1e-6)

    @pytest.mark.fast
    def test_mismatched_dims_raise(self):
        cond = mx.zeros((1, 4, 12 * 16))
        ref = mx.zeros((1, 4, 12 * 16 + 1))

        with pytest.raises(ValueError):
            ConditioningGuidance.project(cond, ref, strength=1.0, n_bands=12)
        with pytest.raises(ValueError):
            ConditioningGuidance.project(cond, cond, strength=1.0, n_bands=7)
