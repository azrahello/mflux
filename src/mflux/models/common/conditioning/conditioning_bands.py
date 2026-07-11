import mlx.core as mx


class ConditioningBands:
    @staticmethod
    def parse_weights(spec: str) -> list[float]:
        try:
            return [float(v) for v in spec.split(",") if v.strip() != ""]
        except ValueError:
            raise ValueError(f"invalid conditioning weights {spec!r}, expected comma-separated floats")

    @staticmethod
    def scale_bands(
        embeds: mx.array,
        weights: list[float],
        renormalize: bool = False,
        multiplier: float = 1.0,
    ) -> mx.array:
        n_bands = len(weights)
        feature_dim = embeds.shape[-1]
        if feature_dim % n_bands != 0:
            raise ValueError(f"conditioning last dim {feature_dim} is not divisible by {n_bands} weights")
        band_dim = feature_dim // n_bands
        embeds_f = embeds.astype(mx.float32)
        reduce_axes = tuple(range(1, embeds_f.ndim))
        ref_rms = mx.sqrt(mx.mean(embeds_f**2, axis=reduce_axes)) if renormalize else None

        shaped = embeds_f.reshape(*embeds.shape[:-1], n_bands, band_dim)
        gains = mx.array(weights, dtype=mx.float32).reshape(*([1] * (shaped.ndim - 2)), n_bands, 1)
        scaled = (shaped * gains).reshape(embeds.shape)

        if ref_rms is not None:
            # Redistribute per-band energy without changing the overall conditioning
            # magnitude, so uneven weights don't cause magnitude-collapse quality loss.
            new_rms = mx.maximum(mx.sqrt(mx.mean(scaled**2, axis=reduce_axes)), 1e-8)
            scale = (ref_rms / new_rms).reshape(-1, *([1] * (embeds_f.ndim - 1)))
            scaled = scaled * scale

        # Uniform gain on the whole tensor, applied after renormalization so the
        # two compose: weights shape the band balance, the multiplier sets volume.
        if multiplier != 1.0:
            scaled = scaled * multiplier

        return scaled.astype(embeds.dtype)
