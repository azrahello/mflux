import mlx.core as mx
from mlx import nn


class Qwen3TextRotaryEmbedding(nn.Module):
    def __init__(
        self,
        dim: int,
        max_position_embeddings: int = 40960,
        base: float = 1000000.0,
        scaling_factor: float = 1.0,
        mrope_section: list[int] | None = None,
    ):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        self.scaling_factor = scaling_factor
        self.inv_freq = 1.0 / (base ** (mx.arange(0, dim, 2, dtype=mx.float32) / dim))
        # mrope_section (e.g. [24, 20, 20] temporal/height/width channel counts): only
        # set for vision-language callers (Krea2's image-grounded encode). None keeps
        # the plain single-axis behavior every other caller (Flux2, Ideogram4) relies on.
        self.mrope_section = mrope_section
        if mrope_section is not None:
            self._axis_id = Qwen3TextRotaryEmbedding._build_axis_id(dim // 2, mrope_section)

    def __call__(self, x: mx.array, position_ids: mx.array) -> tuple[mx.array, mx.array]:
        if position_ids.ndim == 1:
            position_ids = mx.expand_dims(position_ids, axis=0)
        if self.mrope_section is not None and position_ids.ndim == 2:
            # Plain positions (no multimodal input this call): identical on all 3 axes,
            # so the interleaved merge is a no-op -- skip straight to the 1D path.
            pass
        elif self.mrope_section is not None:
            # position_ids: (3, batch, seq) -- temporal/height/width axes.
            inv_freq = self.inv_freq[None, None, None, :]
            pos = position_ids.astype(mx.float32)[..., None]
            freqs = pos * inv_freq  # (3, batch, seq, dim/2)
            freqs = Qwen3TextRotaryEmbedding._apply_interleaved_mrope(freqs, self._axis_id)
            emb = mx.concatenate([freqs, freqs], axis=-1)
            cos = mx.cos(emb) * self.scaling_factor
            sin = mx.sin(emb) * self.scaling_factor
            return cos.astype(x.dtype), sin.astype(x.dtype)

        inv_freq = mx.expand_dims(mx.expand_dims(self.inv_freq, axis=0), axis=0)
        pos = mx.expand_dims(position_ids.astype(mx.float32), axis=-1)
        freqs = pos * inv_freq
        emb = mx.concatenate([freqs, freqs], axis=-1)
        cos = mx.cos(emb) * self.scaling_factor
        sin = mx.sin(emb) * self.scaling_factor
        return cos.astype(x.dtype), sin.astype(x.dtype)

    @staticmethod
    def _build_axis_id(dim_half: int, mrope_section: list[int]) -> mx.array:
        # Reference layout (Qwen3-VL "interleaved" MRoPE): channel i defaults to the
        # temporal axis (0); channels i % 3 == 1 (up to mrope_section[1]*3) belong to
        # height (1); channels i % 3 == 2 (up to mrope_section[2]*3) belong to width (2).
        axis_id = [0] * dim_half
        for dim, offset in ((1, 1), (2, 2)):
            length = mrope_section[dim] * 3
            for i in range(offset, min(length, dim_half), 3):
                axis_id[i] = dim
        return mx.array(axis_id, dtype=mx.int32)

    @staticmethod
    def _apply_interleaved_mrope(freqs: mx.array, axis_id: mx.array) -> mx.array:
        # freqs: (3, batch, seq, dim/2) -> merged (batch, seq, dim/2), picking each
        # channel's value from its designated axis (temporal/height/width).
        axis_id = axis_id[None, None, :]
        merged = freqs[0]
        merged = mx.where(axis_id == 1, freqs[1], merged)
        merged = mx.where(axis_id == 2, freqs[2], merged)
        return merged
