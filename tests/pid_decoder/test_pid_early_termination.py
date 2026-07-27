"""PiD early termination: stop the base LDM early and hand PiD the residual-noise latent.

The one thing here that fails silently is the sigma index. PiD's own capture callback uses
`sigma_idx = min(step_index + 1, len(sigmas) - 1)` (step_capture.py) and mflux's scheduler
advances sigmas[t] -> sigmas[t+1] in `step(timestep=t)`, so the two must line up exactly --
an off-by-one still produces an image, just a subtly wrong one.
"""

import pytest

from mflux.models.common.config.config import PID_MAX_TRAINED_SIGMA, Config
from mflux.models.common.config.model_config import ModelConfig


def _config(steps: int = 8, skip: int = 0) -> Config:
    return Config(
        model_config=ModelConfig.krea2(),
        num_inference_steps=steps,
        height=512,
        width=512,
        pid_skip_steps=skip,
    )


def test_default_is_the_fully_denoised_latent():
    # No skipping must land on the schedule's final sigma (0.0) -- the pre-existing behaviour,
    # reached without a special case.
    config = _config(skip=0)
    assert config.pid_sigma == 0.0
    assert len(list(config.time_steps)) == 8


def test_skipping_shortens_the_loop_by_exactly_that_many_steps():
    for skip in (1, 2, 3):
        assert len(list(_config(skip=skip).time_steps)) == 8 - skip


def test_sigma_matches_the_latent_the_loop_actually_leaves_behind():
    # Running steps [0, 8-skip) leaves the latent at sigmas[8-skip], because step(timestep=t)
    # advances the latent from sigmas[t] to sigmas[t+1].
    sigmas = _config().scheduler.sigmas
    for skip in (0, 1, 2, 3):
        config = _config(skip=skip)
        last_t = list(config.time_steps)[-1]
        assert config.pid_sigma == pytest.approx(float(sigmas[last_t + 1])), f"skip={skip}"


def test_more_skipping_means_a_noisier_latent():
    # Stopping earlier leaves more residual noise, so sigma rises with the skip count.
    sigmas = [_config(skip=s).pid_sigma for s in (0, 1, 2, 3)]
    assert sigmas == sorted(sigmas), sigmas
    assert sigmas[0] == 0.0


def test_rejects_skipping_past_pids_trained_sigma_range():
    # 8-step turbo schedules jump fast: skipping 6 of 8 leaves sigma=0.849, past the sigma<=0.8
    # PiD was trained on, where it has never seen the input distribution. Must refuse rather
    # than decode noise. (Skipping 5 still lands at 0.758 -- inside the range, allowed.)
    assert _config(steps=8, skip=5).pid_sigma <= PID_MAX_TRAINED_SIGMA
    config = _config(steps=8, skip=6)
    assert float(config.scheduler.sigmas[2]) > PID_MAX_TRAINED_SIGMA, "premise: this skip is out of range"
    with pytest.raises(ValueError, match="trained on"):
        _ = config.pid_sigma


def test_rejects_skipping_every_step():
    with pytest.raises(ValueError, match="no denoising steps"):
        _config(steps=8, skip=8)


def test_rejects_negative_skip():
    with pytest.raises(ValueError, match=">= 0"):
        _config(skip=-1)


def test_skipping_is_inert_without_pid_decode():
    """--pid-skip-steps only makes sense with --pid-decode: PidNet finishes the denoising in
    pixel space. Without it a shortened loop just hands the VAE a latent that is still noisy,
    silently -- no error, just a worse image. Every model gates it at the Config call, so the
    guarantee holds for library callers too, not only the CLI.
    """
    import inspect

    from mflux.models.z_image.variants.z_image import ZImage

    source = inspect.getsource(ZImage.generate_image)
    assert "pid_skip_steps=pid_skip_steps if pid_decode else 0" in source

    # And the underlying behaviour: gated to 0, the loop keeps every step.
    assert len(list(_config(steps=8, skip=0).time_steps)) == 8
