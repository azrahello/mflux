import mlx.core as mx
from mlx import nn

from mflux.models.krea2.model.krea2_transformer.attention import Krea2Attention
from mflux.models.krea2.model.krea2_transformer.common import Krea2RMSNorm
from mflux.models.krea2.model.krea2_transformer.feed_forward import Krea2SwiGLU
from mflux.models.krea2.model.krea2_transformer.modulation import DoubleSharedModulation


class SingleStreamBlock(nn.Module):
    def __init__(self, features: int, heads: int, multiplier: int, bias: bool = False, kvheads: int | None = None):
        super().__init__()
        self.mod = DoubleSharedModulation(features)
        self.prenorm = Krea2RMSNorm(features)
        self.postnorm = Krea2RMSNorm(features)
        self.attn = Krea2Attention(features, heads, kvheads=kvheads, bias=bias)
        self.mlp = Krea2SwiGLU(features, multiplier, bias)

    def __call__(
        self,
        x: mx.array,
        vec: mx.array,
        freqs: mx.array,
        mask: mx.array | None = None,
        refvec: mx.array | None = None,
        split: int | None = None,
    ) -> mx.array:
        if refvec is None:
            prescale, preshift, pregate, postscale, postshift, postgate = self.mod(vec)
            x = x + pregate * self.attn((1 + prescale) * self.prenorm(x) + preshift, freqs=freqs, mask=mask)
            x = x + postgate * self.mlp((1 + postscale) * self.postnorm(x) + postshift)
            return x

        # In-context edit ("index_timestep_zero"): tokens [:split] (text + noisy
        # target) modulate at the real timestep, tokens [split:] (clean reference
        # tokens) at t=0. Attention still runs over the full sequence.
        m = self.mod(vec)
        r = self.mod(refvec)

        def modulate(h: mx.array, scale: int, shift: int) -> mx.array:
            return mx.concatenate(
                [(1 + m[scale]) * h[:, :split] + m[shift], (1 + r[scale]) * h[:, split:] + r[shift]],
                axis=1,
            )

        def gate(h: mx.array, g: int) -> mx.array:
            return mx.concatenate([m[g] * h[:, :split], r[g] * h[:, split:]], axis=1)

        x = x + gate(self.attn(modulate(self.prenorm(x), 0, 1), freqs=freqs, mask=mask), 2)
        x = x + gate(self.mlp(modulate(self.postnorm(x), 3, 4)), 5)
        return x
