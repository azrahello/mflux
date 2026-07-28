import mlx.core as mx

from mflux.models.common.pid_decoder.pid_decoder import _resize_latent


class _FakeVae:
    """Just the 8x spatial contract -- the pixels don't matter, only the geometry does."""

    spatial_scale = 8

    def decode(self, latent: mx.array) -> mx.array:
        b, _, zh, zw = latent.shape
        return mx.zeros((b, 3, zh * 8, zw * 8))

    def encode(self, image: mx.array) -> mx.array:
        b, _, h, w = image.shape
        return mx.zeros((b, 16, h // 8, w // 8))


def test_resize_lands_on_the_requested_output_size():
    # 832x1248 generation, asked for a 768px long side -> 512x768 into PiD -> 2048x3072 out.
    resized = _resize_latent(vae=_FakeVae(), latent=mx.zeros((1, 16, 104, 156)), long_side=768)
    assert resized.shape == (1, 16, 64, 96)
    assert tuple(dim * 8 * 4 for dim in resized.shape[2:]) == (2048, 3072)


def test_resize_is_a_no_op_when_the_generation_is_already_that_size():
    latent = mx.zeros((1, 16, 64, 96))
    assert _resize_latent(vae=_FakeVae(), latent=latent, long_side=768) is latent
