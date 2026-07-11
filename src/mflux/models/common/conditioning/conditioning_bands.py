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
        clamp: float = 0.0,
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
            new_rms = mx.maximum(mx.sqrt(mx.mean(scaled**2, axis=reduce_axes)), 1e-8)
            scale = (ref_rms / new_rms).reshape(-1, *([1] * (embeds_f.ndim - 1)))
            scaled = scaled * scale

        if multiplier != 1.0:
            scaled = scaled * multiplier

        if clamp > 0.0:
            scaled = mx.clip(scaled, -clamp, clamp)

        return scaled.astype(embeds.dtype)


# ---------------------------------------------------------------------------
# Krea2 Pinot Rebalance (filter knobs + step gating, all-in-one)
# ---------------------------------------------------------------------------

N_LAYERS = 12

# 1-indexed knob -> 0-indexed layer in the 12*2560 stack.
_KNOB_LAYER = {9: 8, 10: 9, 11: 10}


def _apply_knobs(conditioning, knob_map, clamp=0.0):
    """Multiply only the magic knobs (9/10 primary, 11 secondary).

    Returns a list of [conditioning_array, dict] pairs (same shape as input).
    """
    out = []
    for t, d in conditioning:
        if t.shape[-1] % N_LAYERS != 0:
            raise ValueError(
                f"conditioning last dim {t.shape[-1]} not divisible by {N_LAYERS} "
                "-- is this a Krea2 (CLIPLoader type=krea2) conditioning?"
            )
        layer_dim = t.shape[-1] // N_LAYERS
        orig = t.dtype

        # View as [*, layers, layer_dim]
        x = mx.array(t).astype(mx.float32)
        x = x.reshape(*x.shape[:-1], N_LAYERS, layer_dim)

        for idx, m in knob_map.items():
            x = mx.where(
                mx.arange(N_LAYERS) == idx,
                x[..., idx:idx + 1, :] * m,
                x
            )

        # Clamp guards against fp16 overflow (vision tokens carry large norms)
        if clamp and clamp > 0:
            x = mx.clip(x, -clamp, clamp)

        out.append([x.reshape(t.shape).astype(orig), dict(d)])
    return out


def conditioning_set_values(conditioning, values):
    """Set conditioning parameters (start_percent / end_percent)."""
    return [[c, {**d, **values}] for c, d in conditioning]


class MagicRebalance:
    """Multiply only the magic knobs, gate rebalanced cond to early steps
    and clean cond to late steps.  Style/anatomy priors are never touched."""

    @classmethod
    def define_schema(cls):
        return {
            "node_id": "MagicRebalance",
            "display_name": "Krea2 Gated Rebalance",
            "category": "AZ_Nodes",
            "description": (
                "Multiply only the magic knobs (9/10 primary, 11 secondary), gate the "
                "rebalanced cond to early steps and the clean cond to late steps. Style/anatomy "
                "priors are never touched (no plasticky drift). 0 = leave that knob untouched."
            ),
            "inputs": [
                ("conditioning", "Conditioning"),
                ("knob9", 0.4883, -100.0, 100.0, 0.0001),
                ("knob10", 0.1094, -100.0, 100.0, 0.0001),
                ("knob11", 0.0, -100.0, 100.0, 0.0001),
                ("multiplier", 1.0, -100.0, 100.0, 0.05),
                ("crossover", 0.5, 0.0, 1.0, 0.01),
                ("overlap", 0.0, 0.0, 0.5, 0.01),
                ("clamp", 0.0, 0.0, 1000.0, 1.0),
            ],
            "outputs": [("conditioning",)],
        }

    @classmethod
    def execute(cls, conditioning, knob9=0.4883, knob10=0.1094, knob11=0.0,
                multiplier=1.0, crossover=0.5, overlap=0.0, clamp=0.0):
        knob_map = {}
        for knob, val in ((9, knob9), (10, knob10), (11, knob11)):
            if val != 0.0:
                knob_map[_KNOB_LAYER[knob]] = 1.0 + multiplier * (val - 1.0)

        # No knobs engaged → pass clean cond straight through
        if not knob_map:
            return conditioning

        rebalanced = _apply_knobs(conditioning, knob_map, clamp)

        early_end = min(1.0, crossover + overlap)
        late_start = max(0.0, crossover - overlap)

        early = conditioning_set_values(rebalanced, {
            "start_percent": 0.0,
            "end_percent": early_end,
        })
        late = conditioning_set_values(conditioning, {
            "start_percent": late_start,
            "end_percent": 1.0,
        })

        return early + late
