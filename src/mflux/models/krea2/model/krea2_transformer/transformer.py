from typing import NamedTuple

import mlx.core as mx
from mlx import nn

from mflux.models.krea2.model.krea2_transformer.final_layer import LastLayer
from mflux.models.krea2.model.krea2_transformer.rope_embedder import Krea2RopeEmbedder
from mflux.models.krea2.model.krea2_transformer.text_fusion import TextFusionTransformer
from mflux.models.krea2.model.krea2_transformer.text_mlp import Krea2TextMLP
from mflux.models.krea2.model.krea2_transformer.timestep_embedder import Krea2TimestepMLP, Krea2TimestepProj
from mflux.models.krea2.model.krea2_transformer.transformer_block import SingleStreamBlock


class Krea2PreparedRefs(NamedTuple):
    # Patchified reference tokens and RoPE ids, constant across denoise steps.
    tokens: mx.array  # (bs, reflen, channels*patch*patch), pre-`first` projection
    pos: mx.array  # (bs, reflen, 3)
    reflen: int


class Krea2Transformer(nn.Module):
    def __init__(
        self,
        features: int = 6144,
        tdim: int = 256,
        txtdim: int = 2560,
        heads: int = 48,
        kvheads: int = 12,
        multiplier: int = 4,
        layers: int = 28,
        patch: int = 2,
        channels: int = 16,
        bias: bool = False,
        theta: int = 1000,
        txtlayers: int = 12,
        txtheads: int = 20,
        txtkvheads: int = 20,
    ):
        super().__init__()
        self.patch = patch
        self.channels = channels
        self.features = features
        self.tdim = tdim
        self.heads = heads
        self.txtdim = txtdim
        self.txtlayers = txtlayers

        head_dim = features // heads
        axes = [head_dim - 12 * (head_dim // 16), 6 * (head_dim // 16), 6 * (head_dim // 16)]
        self.pe_embedder = Krea2RopeEmbedder(head_dim=head_dim, theta=theta, axes_dim=axes)

        self.first = nn.Linear(channels * patch**2, features, bias=True)
        self.blocks = [SingleStreamBlock(features, heads, multiplier, bias, kvheads) for _ in range(layers)]
        self.tmlp = Krea2TimestepMLP(tdim, features)
        self.tproj = Krea2TimestepProj(features)
        self.txtfusion = TextFusionTransformer(txtlayers, txtdim, txtheads, multiplier, bias, txtkvheads)
        self.txtmlp = Krea2TextMLP(txtdim, features)
        self.last = LastLayer(features, patch, channels)

    def __call__(
        self,
        hidden_states: mx.array,
        timestep: mx.array,
        context: mx.array,
        attention_mask: mx.array | None = None,
        ref_latents: list[mx.array] | None = None,
        prepared_refs: Krea2PreparedRefs | None = None,
    ) -> mx.array:
        bs, c, H_orig, W_orig = hidden_states.shape
        patch = self.patch

        x = Krea2Transformer._pad_to_multiple(hidden_states, patch)
        H, W = x.shape[-2], x.shape[-1]
        h_, w_ = H // patch, W // patch

        context = self.fuse_context(context)

        # Patchify: (b, c, h*ph, w*pw) -> (b, h*w, c*ph*pw)
        img = x.reshape(bs, c, h_, patch, w_, patch).transpose(0, 2, 4, 1, 3, 5).reshape(bs, h_ * w_, c * patch * patch)

        # In-context edit references (constant across steps; callers on the hot
        # path pass prepared_refs from prepare_refs() instead of re-patchifying).
        if prepared_refs is None and ref_latents:
            prepared_refs = self.prepare_refs(ref_latents, bs=bs, dtype=x.dtype)
        reflen = prepared_refs.reflen if prepared_refs else 0
        if reflen:
            img = mx.concatenate([img, prepared_refs.tokens.astype(x.dtype)], axis=1)
        img = self.first(img)

        t = self.tmlp(Krea2TimestepMLP.timestep_embedding(timestep, self.tdim)[:, None, :].astype(img.dtype))
        tvec = self.tproj(t)
        refvec = None
        if reflen:
            t0 = self.tmlp(
                Krea2TimestepMLP.timestep_embedding(mx.zeros_like(timestep), self.tdim)[:, None, :].astype(img.dtype)
            )
            refvec = self.tproj(t0)

        txtlen, imglen = context.shape[1], img.shape[1]
        combined = mx.concatenate([context, img], axis=1)
        split = txtlen + imglen - reflen  # start of the reference span

        # Position ids: text at (0,0,0); target image at (0, h_idx, w_idx); refs at (i+1, h_idx, w_idx).
        txtpos = mx.zeros((bs, txtlen, 3), dtype=mx.float32)
        gh, gw = mx.meshgrid(mx.arange(h_, dtype=mx.float32), mx.arange(w_, dtype=mx.float32), indexing="ij")
        imgids = mx.stack([mx.zeros_like(gh), gh, gw], axis=-1).reshape(1, h_ * w_, 3)
        imgpos = mx.broadcast_to(imgids, (bs, h_ * w_, 3))
        pos_parts = [txtpos, imgpos] + ([prepared_refs.pos] if reflen else [])
        freqs = self.pe_embedder(mx.concatenate(pos_parts, axis=1))

        # Optional gradient checkpointing: recompute each block in backward instead of storing its
        # activations, trading compute for a large drop in peak memory during training. Off by
        # default, so inference is unaffected; the training adapter turns it on.
        gradient_checkpointing = getattr(self, "gradient_checkpointing", False)
        for block in self.blocks:
            run = nn.utils.checkpoint(block) if gradient_checkpointing else block
            combined = run(combined, tvec, freqs, attention_mask, refvec, split)

        final = self.last(combined, t)
        out = final[:, txtlen:split, :]  # noisy target tokens only
        # Unpatchify: (b, h*w, c*ph*pw) -> (b, c, h*ph, w*pw)
        out = out.reshape(bs, h_, w_, self.channels, patch, patch).transpose(0, 3, 1, 4, 2, 5)
        out = out.reshape(bs, self.channels, H, W)
        return out[:, :, :H_orig, :W_orig]

    def fuse_context(self, context: mx.array) -> mx.array:
        # Fuse the stacked per-layer conditioning (B, seq, txtlayers*txtdim) into
        # the DiT's text stream (B, seq, features). Constant across denoise steps,
        # so callers may pre-fuse once and pass the result through unchanged
        # (recognized by its feature dim).
        b, seq, dim = context.shape
        if dim == self.features:
            return context  # already fused
        if dim != self.txtlayers * self.txtdim:
            raise ValueError(
                f"Krea2 expects conditioning with {self.txtlayers}x{self.txtdim}="
                f"{self.txtlayers * self.txtdim} features (a {self.txtlayers}-layer Qwen3-VL stack) but got {dim}."
            )
        # The fusion runs in fp32 regardless of the activation dtype: a rebalanced
        # txtfusion.projector (--projector-rebalance-strength) can amplify the
        # stream far past float16 range mid-fusion, and the fusion is off the
        # per-step hot path so full precision here is effectively free. When the
        # result goes back to float16, saturate at its finite max instead of
        # overflowing to inf (the DiT blocks renormalize magnitudes via prenorm,
        # so saturated-but-finite values keep rendering).
        out_dtype = context.dtype
        context = context.reshape(b, seq, self.txtlayers, self.txtdim).astype(mx.float32)
        context = self.txtfusion(context, mask=None)
        context = self.txtmlp(context)
        if out_dtype == mx.float16:
            f16_max = float(mx.finfo(mx.float16).max)
            context = mx.clip(context, -f16_max, f16_max)
        return context.astype(out_dtype)

    def prepare_refs(self, ref_latents: list[mx.array], bs: int, dtype: mx.Dtype) -> Krea2PreparedRefs:
        # Patchify the clean reference latents once: each keeps its own y/x grid
        # with RoPE axis-0 frame index i+1 (target = 0). The blocks modulate the
        # reference span at t=0.
        c, patch = self.channels, self.patch
        ref_tokens: list[mx.array] = []
        ref_pos: list[mx.array] = []
        for i, ref in enumerate(ref_latents):
            ref = Krea2Transformer._pad_to_multiple(ref.astype(dtype), patch)
            ref = mx.broadcast_to(ref, (bs, *ref.shape[1:]))
            rh, rw = ref.shape[-2] // patch, ref.shape[-1] // patch
            rtok = ref.reshape(bs, c, rh, patch, rw, patch).transpose(0, 2, 4, 1, 3, 5)
            ref_tokens.append(rtok.reshape(bs, rh * rw, c * patch * patch))
            rgh, rgw = mx.meshgrid(mx.arange(rh, dtype=mx.float32), mx.arange(rw, dtype=mx.float32), indexing="ij")
            rid = mx.stack([mx.full(rgh.shape, i + 1.0, dtype=mx.float32), rgh, rgw], axis=-1)
            ref_pos.append(mx.broadcast_to(rid.reshape(1, rh * rw, 3), (bs, rh * rw, 3)))
        tokens = mx.concatenate(ref_tokens, axis=1)
        pos = mx.concatenate(ref_pos, axis=1)
        return Krea2PreparedRefs(tokens=tokens, pos=pos, reflen=tokens.shape[1])

    @staticmethod
    def _pad_to_multiple(x: mx.array, patch: int) -> mx.array:
        H, W = x.shape[-2], x.shape[-1]
        pad_h = (patch - H % patch) % patch
        pad_w = (patch - W % patch) % patch
        if pad_h == 0 and pad_w == 0:
            return x
        return mx.pad(x, [(0, 0), (0, 0), (0, pad_h), (0, pad_w)])
