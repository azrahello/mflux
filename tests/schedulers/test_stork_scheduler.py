import mlx.core as mx
import pytest

from mflux.config.config import Config
from mflux.config.model_config import ModelConfig
from mflux.config.runtime_config import RuntimeConfig
from mflux.schedulers import SCHEDULER_REGISTRY, try_import_external_scheduler
from mflux.schedulers.stork_scheduler import STORKScheduler


def test_stork_scheduler_import_by_name():
    """Test that STORK scheduler can be imported by name."""
    assert try_import_external_scheduler("mflux.schedulers.stork_scheduler.STORKScheduler") == STORKScheduler


@pytest.fixture
def test_runtime_config():
    """Create a test runtime configuration."""
    return RuntimeConfig(
        Config(
            num_inference_steps=10,
            width=1024,
            height=1024,
            scheduler="stork-2",
        ),
        ModelConfig.dev(),
    )


def test_stork_scheduler_initialization_order_2(test_runtime_config):
    """Test the initialization of STORK-2 scheduler."""
    scheduler = STORKScheduler(runtime_config=test_runtime_config, order=2)
    assert scheduler.order == 2
    assert scheduler.rk_stages == 2
    assert scheduler.sigmas is not None
    assert isinstance(scheduler.sigmas, mx.array)
    assert len(scheduler.sigmas) > 0


def test_stork_scheduler_initialization_order_4(test_runtime_config):
    """Test the initialization of STORK-4 scheduler."""
    scheduler = STORKScheduler(runtime_config=test_runtime_config, order=4)
    assert scheduler.order == 4
    assert scheduler.rk_stages == 4
    assert scheduler.sigmas is not None
    assert isinstance(scheduler.sigmas, mx.array)
    assert len(scheduler.sigmas) > 0


def test_stork_scheduler_invalid_order(test_runtime_config):
    """Test that invalid order raises ValueError."""
    with pytest.raises(ValueError, match="STORK order must be 2 or 4"):
        STORKScheduler(runtime_config=test_runtime_config, order=3)


def test_stork_scheduler_sigmas_shape(test_runtime_config):
    """Test that sigmas have the correct shape."""
    scheduler = STORKScheduler(runtime_config=test_runtime_config, order=2)
    expected_length = test_runtime_config.num_inference_steps + 1
    assert scheduler.sigmas.shape == (expected_length,)
    assert scheduler.timesteps.shape == (test_runtime_config.num_inference_steps,)


def test_stork_scheduler_sigmas_bounds(test_runtime_config):
    """Test that sigmas are within expected bounds."""
    scheduler = STORKScheduler(runtime_config=test_runtime_config, order=2)
    sigmas = scheduler.sigmas

    # First sigma should be close to 1.0 (pure noise)
    assert sigmas[0] > 0.9
    assert sigmas[0] <= 1.0

    # Last sigma should be 0.0 (clean data)
    assert sigmas[-1] == 0.0

    # Sigmas should be monotonically decreasing
    for i in range(len(sigmas) - 1):
        assert sigmas[i] >= sigmas[i + 1]


def test_stork_scheduler_step_basic(test_runtime_config):
    """Test basic step functionality."""
    scheduler = STORKScheduler(runtime_config=test_runtime_config, order=2)

    # Create dummy inputs
    batch_size = 1
    latent_dim = 64
    height = 64
    width = 64

    sample = mx.random.normal((batch_size, height, width, latent_dim))
    model_output = mx.random.normal((batch_size, height, width, latent_dim))

    # Perform one step
    next_sample = scheduler.step(model_output=model_output, timestep=0, sample=sample)

    # Check output shape
    assert next_sample.shape == sample.shape

    # Check that output is different from input
    assert not mx.array_equal(next_sample, sample)


def test_stork_scheduler_step_history(test_runtime_config):
    """Test that velocity history is maintained correctly."""
    scheduler = STORKScheduler(runtime_config=test_runtime_config, order=2)

    sample = mx.random.normal((1, 64, 64, 64))
    model_output = mx.random.normal((1, 64, 64, 64))

    # Initially, history should be empty
    assert len(scheduler.velocity_history) == 0

    # After first step, history should have one entry
    scheduler.step(model_output=model_output, timestep=0, sample=sample)
    assert len(scheduler.velocity_history) == 1

    # After second step, history should have two entries
    scheduler.step(model_output=model_output, timestep=1, sample=sample)
    assert len(scheduler.velocity_history) == 2

    # History should be limited to max_history (3)
    scheduler.step(model_output=model_output, timestep=2, sample=sample)
    scheduler.step(model_output=model_output, timestep=3, sample=sample)
    assert len(scheduler.velocity_history) <= 3


def test_stork_scheduler_scale_model_input(test_runtime_config):
    """Test that scale_model_input is a no-op for flow matching."""
    scheduler = STORKScheduler(runtime_config=test_runtime_config, order=2)

    latents = mx.random.normal((1, 64, 64, 64))
    scaled = scheduler.scale_model_input(latents, t=0)

    # Should return unchanged latents
    assert mx.array_equal(scaled, latents)


def test_stork_2_registry():
    """Test that stork-2 is registered correctly."""
    assert "stork-2" in SCHEDULER_REGISTRY
    factory = SCHEDULER_REGISTRY["stork-2"]

    # Create a test config
    config = RuntimeConfig(
        Config(num_inference_steps=10, width=1024, height=1024, scheduler="stork-2"),
        ModelConfig.dev(),
    )

    # Create scheduler using factory
    scheduler = factory(config)
    assert isinstance(scheduler, STORKScheduler)
    assert scheduler.order == 2


def test_stork_4_registry():
    """Test that stork-4 is registered correctly."""
    assert "stork-4" in SCHEDULER_REGISTRY
    factory = SCHEDULER_REGISTRY["stork-4"]

    # Create a test config
    config = RuntimeConfig(
        Config(num_inference_steps=10, width=1024, height=1024, scheduler="stork-4"),
        ModelConfig.dev(),
    )

    # Create scheduler using factory
    scheduler = factory(config)
    assert isinstance(scheduler, STORKScheduler)
    assert scheduler.order == 4


def test_stork_default_registry():
    """Test that stork defaults to order 2."""
    assert "stork" in SCHEDULER_REGISTRY
    factory = SCHEDULER_REGISTRY["stork"]

    # Create a test config
    config = RuntimeConfig(
        Config(num_inference_steps=10, width=1024, height=1024, scheduler="stork"),
        ModelConfig.dev(),
    )

    # Create scheduler using factory
    scheduler = factory(config)
    assert isinstance(scheduler, STORKScheduler)
    assert scheduler.order == 2  # Default should be 2


def test_stork_rk_coefficients_order_2():
    """Test that RK coefficients are set correctly for order 2."""
    config = RuntimeConfig(
        Config(num_inference_steps=10, width=1024, height=1024, scheduler="stork-2"),
        ModelConfig.dev(),
    )
    scheduler = STORKScheduler(runtime_config=config, order=2)

    assert scheduler.rk_stages == 2
    assert len(scheduler.rk_b) == 2
    assert len(scheduler.rk_c) == 2
    assert sum(scheduler.rk_b) == 1.0  # RK weights should sum to 1


def test_stork_rk_coefficients_order_4():
    """Test that RK coefficients are set correctly for order 4."""
    config = RuntimeConfig(
        Config(num_inference_steps=10, width=1024, height=1024, scheduler="stork-4"),
        ModelConfig.dev(),
    )
    scheduler = STORKScheduler(runtime_config=config, order=4)

    assert scheduler.rk_stages == 4
    assert len(scheduler.rk_b) == 4
    assert len(scheduler.rk_c) == 4
    # RK4 weights: [1/6, 1/3, 1/3, 1/6] sum to 1
    assert abs(sum(scheduler.rk_b) - 1.0) < 1e-6
