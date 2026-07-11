import mlx.core as mx
from PIL import Image

from mflux.models.krea2.model.krea2_transformer.transformer import Krea2Transformer
from mflux.models.krea2.model.krea2_transformer.transformer_block import SingleStreamBlock
from mflux.models.krea2.variants.txt2img.krea2 import Krea2


def tiny_transformer() -> Krea2Transformer:
    return Krea2Transformer(
        features=64,
        tdim=16,
        txtdim=8,
        heads=4,
        kvheads=2,
        multiplier=2,
        layers=2,
        patch=2,
        channels=4,
        txtlayers=2,
        txtheads=2,
        txtkvheads=2,
    )


def test_no_refs_path_unchanged():
    # ref_latents=None and ref_latents=[] must produce the exact same output.
    t = tiny_transformer()
    x = mx.random.normal(shape=(1, 4, 8, 8), key=mx.random.key(0))
    ctx = mx.random.normal(shape=(1, 5, 16), key=mx.random.key(1))
    ts = mx.array([0.5])
    out_none = t(x, ts, ctx)
    out_empty = t(x, ts, ctx, ref_latents=[])
    assert out_none.shape == (1, 4, 8, 8)
    assert mx.allclose(out_none, out_empty).item()


def test_refs_output_covers_target_only():
    # With references attached, the output stays the target's shape, and the
    # reference content changes the prediction (full attention sees the refs).
    t = tiny_transformer()
    x = mx.random.normal(shape=(1, 4, 8, 8), key=mx.random.key(0))
    ctx = mx.random.normal(shape=(1, 5, 16), key=mx.random.key(1))
    ts = mx.array([0.5])
    ref_a = mx.random.normal(shape=(1, 4, 6, 6), key=mx.random.key(2))
    ref_b = mx.random.normal(shape=(1, 4, 4, 10), key=mx.random.key(3))
    out = t(x, ts, ctx, ref_latents=[ref_a, ref_b])
    assert out.shape == (1, 4, 8, 8)
    out_other = t(x, ts, ctx, ref_latents=[ref_b, ref_a])
    assert not mx.allclose(out, out_other).item()


def test_block_span_modulation_matches_uniform_when_refvec_equals_vec():
    # With refvec == vec the per-span path must reduce to the uniform path.
    block = SingleStreamBlock(features=64, heads=4, multiplier=2, kvheads=2)
    x = mx.random.normal(shape=(1, 10, 64), key=mx.random.key(4))
    vec = mx.random.normal(shape=(1, 1, 6 * 64), key=mx.random.key(5))
    freqs = None
    uniform = block(x, vec, freqs)
    spanned = block(x, vec, freqs, refvec=vec, split=6)
    assert mx.allclose(uniform, spanned, atol=1e-5).item()


def test_fit_area_downscales_never_upscales():
    big = Image.new("RGB", (2000, 1500))
    fitted = Krea2._fit_area(big, 1024 * 1024, snap=16)
    w, h = fitted.size
    assert w % 16 == 0 and h % 16 == 0
    assert w * h <= 1024 * 1024 * 1.05  # snapping tolerance
    assert abs((w / h) - (2000 / 1500)) < 0.05

    small = Image.new("RGB", (300, 200))
    assert Krea2._fit_area(small, 1024 * 1024, snap=1).size == (300, 200)
