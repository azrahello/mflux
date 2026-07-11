import mlx.core as mx
import numpy as np
import pytest

from mflux.models.common.conditioning import ConditioningBands


class TestConditioningBandsScaleBands:
    @pytest.mark.fast
    def test_all_ones_is_no_op(self):
        embeds = mx.random.normal((1, 8, 12 * 16))
        weights = [1.0] * 12

        scaled = ConditioningBands.scale_bands(embeds, weights)

        assert np.allclose(np.array(scaled), np.array(embeds), atol=1e-6)

    @pytest.mark.fast
    def test_scales_only_targeted_band(self):
        n_bands, band_dim = 4, 3
        embeds = mx.ones((1, 2, n_bands * band_dim))
        weights = [1.0, 2.0, 1.0, 1.0]

        scaled = np.array(ConditioningBands.scale_bands(embeds, weights))
        reshaped = scaled.reshape(1, 2, n_bands, band_dim)

        assert np.allclose(reshaped[..., 1, :], 2.0)
        assert np.allclose(reshaped[..., 0, :], 1.0)
        assert np.allclose(reshaped[..., 2, :], 1.0)
        assert np.allclose(reshaped[..., 3, :], 1.0)

    @pytest.mark.fast
    def test_mismatched_weight_count_raises(self):
        embeds = mx.zeros((1, 4, 12 * 16))

        with pytest.raises(ValueError):
            ConditioningBands.scale_bands(embeds, [1.0] * 5)

    @pytest.mark.fast
    def test_renormalize_all_ones_is_still_no_op(self):
        embeds = mx.random.normal((1, 8, 12 * 16))
        weights = [1.0] * 12

        scaled = ConditioningBands.scale_bands(embeds, weights, renormalize=True)

        assert np.allclose(np.array(scaled), np.array(embeds), atol=1e-5)

    @pytest.mark.fast
    def test_renormalize_preserves_overall_rms(self):
        embeds = mx.random.normal((2, 8, 4 * 16))
        weights = [1.0, 5.0, 0.2, 1.0]

        scaled = np.array(ConditioningBands.scale_bands(embeds, weights, renormalize=True))
        original = np.array(embeds)

        original_rms = np.sqrt(np.mean(original**2, axis=(1, 2)))
        scaled_rms = np.sqrt(np.mean(scaled**2, axis=(1, 2)))
        assert np.allclose(scaled_rms, original_rms, rtol=1e-4)

    @pytest.mark.fast
    def test_renormalize_still_redistributes_band_energy(self):
        n_bands, band_dim = 4, 3
        embeds = mx.ones((1, 2, n_bands * band_dim))
        weights = [1.0, 2.0, 1.0, 1.0]

        without = np.array(ConditioningBands.scale_bands(embeds, weights, renormalize=False))
        with_renorm = np.array(ConditioningBands.scale_bands(embeds, weights, renormalize=True))

        with_reshaped = with_renorm.reshape(1, 2, n_bands, band_dim)
        # Band 1 still stands out as the loudest band after renormalization...
        assert with_reshaped[..., 1, :].mean() > with_reshaped[..., 0, :].mean()
        # ...but the absolute values differ from the non-renormalized version.
        assert not np.allclose(with_renorm, without)

    @pytest.mark.fast
    def test_multiplier_scales_whole_tensor(self):
        embeds = mx.random.normal((1, 8, 12 * 16))
        weights = [1.0] * 12

        scaled = ConditioningBands.scale_bands(embeds, weights, multiplier=4.0)

        assert np.allclose(np.array(scaled), np.array(embeds) * 4.0, atol=1e-5)

    @pytest.mark.fast
    def test_multiplier_composes_with_weights(self):
        n_bands, band_dim = 4, 3
        embeds = mx.ones((1, 2, n_bands * band_dim))
        weights = [1.0, 2.0, 1.0, 1.0]

        scaled = np.array(ConditioningBands.scale_bands(embeds, weights, multiplier=3.0))
        reshaped = scaled.reshape(1, 2, n_bands, band_dim)

        assert np.allclose(reshaped[..., 1, :], 6.0)
        assert np.allclose(reshaped[..., 0, :], 3.0)

    @pytest.mark.fast
    def test_multiplier_survives_renormalize(self):
        embeds = mx.random.normal((2, 8, 4 * 16))
        weights = [1.0, 5.0, 0.2, 1.0]

        scaled = np.array(ConditioningBands.scale_bands(embeds, weights, renormalize=True, multiplier=4.0))
        original = np.array(embeds)

        original_rms = np.sqrt(np.mean(original**2, axis=(1, 2)))
        scaled_rms = np.sqrt(np.mean(scaled**2, axis=(1, 2)))
        assert np.allclose(scaled_rms, original_rms * 4.0, rtol=1e-4)

    @pytest.mark.fast
    def test_clamp_zero_is_no_op(self):
        embeds = mx.random.normal((1, 8, 4 * 16)) * 1000.0
        weights = [1.0, 32.0, 1.0, 1.0]

        unclamped = ConditioningBands.scale_bands(embeds, weights, multiplier=4.0)
        clamped = ConditioningBands.scale_bands(embeds, weights, multiplier=4.0, clamp=0.0)

        assert np.allclose(np.array(unclamped), np.array(clamped))

    @pytest.mark.fast
    def test_clamp_bounds_extreme_values(self):
        embeds = mx.random.normal((1, 8, 4 * 16)) * 1000.0
        weights = [1.0, 32.0, 1.0, 1.0]

        scaled = np.array(ConditioningBands.scale_bands(embeds, weights, multiplier=4.0, clamp=50.0))

        assert np.max(np.abs(scaled)) <= 50.0 + 1e-4

    @pytest.mark.fast
    def test_clamp_prevents_fp16_overflow(self):
        embeds = (mx.random.normal((1, 8, 4 * 16)).astype(mx.float16) * 1000.0)
        weights = [1.0, 32.0, 1.0, 1.0]

        overflowed = ConditioningBands.scale_bands(embeds, weights, multiplier=4.0)
        guarded = ConditioningBands.scale_bands(embeds, weights, multiplier=4.0, clamp=50.0)

        assert bool(mx.any(mx.isinf(overflowed.astype(mx.float32))).item())
        assert not bool(mx.any(mx.isinf(guarded.astype(mx.float32))).item())


class TestConditioningBandsParseWeights:
    @pytest.mark.fast
    def test_parses_csv(self):
        assert ConditioningBands.parse_weights("1.0,2.5,0.0") == [1.0, 2.5, 0.0]

    @pytest.mark.fast
    def test_invalid_value_raises(self):
        with pytest.raises(ValueError):
            ConditioningBands.parse_weights("1.0,abc,2.0")
