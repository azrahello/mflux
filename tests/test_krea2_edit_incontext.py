import mlx.core as mx

from mflux.models.krea2.model.krea2_transformer.transformer import Krea2Transformer


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


def test_refs_modulated_at_current_timestep():
    # The reference implementation modulates the whole [text|target|refs] sequence
    # with the current timestep's vector: changing the timestep must change the
    # output even when only refs distinguish the runs (no t=0 special-casing).
    t = tiny_transformer()
    x = mx.random.normal(shape=(1, 4, 8, 8), key=mx.random.key(0))
    ctx = mx.random.normal(shape=(1, 5, 16), key=mx.random.key(1))
    ref = mx.random.normal(shape=(1, 4, 8, 8), key=mx.random.key(2))
    out_a = t(x, mx.array([0.9]), ctx, ref_latents=[ref])
    out_b = t(x, mx.array([0.1]), ctx, ref_latents=[ref])
    assert not mx.allclose(out_a, out_b).item()
