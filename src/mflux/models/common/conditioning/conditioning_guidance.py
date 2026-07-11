import mlx.core as mx


class ConditioningGuidance:
    @staticmethod
    def project(embeds: mx.array, reference: mx.array, strength: float, n_bands: int) -> mx.array:
        # Dissimilarity projection: per band, amplify (strength > 0) or suppress
        # (strength < 0) the component of the conditioning that lies along the
        # direction separating it from the reference conditioning. The sequence
        # lengths of embeds and reference may differ; only their per-band means
        # are compared.
        feature_dim = embeds.shape[-1]
        if feature_dim % n_bands != 0 or reference.shape[-1] != feature_dim:
            raise ValueError(
                f"conditioning dims {feature_dim} / {reference.shape[-1]} not divisible into {n_bands} matching bands"
            )
        band_dim = feature_dim // n_bands

        cond = embeds.astype(mx.float32).reshape(*embeds.shape[:-1], n_bands, band_dim)
        ref = reference.astype(mx.float32).reshape(*reference.shape[:-1], n_bands, band_dim)

        cond_mean = mx.mean(cond, axis=1)  # (B, n_bands, band_dim)
        ref_mean = mx.mean(ref, axis=1)
        if ref_mean.shape[0] != cond_mean.shape[0]:
            ref_mean = mx.broadcast_to(mx.mean(ref_mean, axis=0, keepdims=True), cond_mean.shape)

        direction = cond_mean - ref_mean
        norm = mx.sqrt(mx.sum(direction**2, axis=-1, keepdims=True) + 1e-8)
        direction = (direction / norm)[:, None]  # (B, 1, n_bands, band_dim)

        projection = mx.sum(cond * direction, axis=-1, keepdims=True)
        out = cond + strength * projection * direction
        return out.reshape(embeds.shape).astype(embeds.dtype)
