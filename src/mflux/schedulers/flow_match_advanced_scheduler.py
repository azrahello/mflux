"""
Flow Matching Advanced Scheduler with Multiple Noise Schedules

Supports various noise schedules for Flow Matching models:
- Linear, Cosine, Exponential, Sqrt, Scaled Linear (simple, effective)
- Beta distribution (complex, from arXiv:2407.12173)

Based on:
- DDPM noise schedules (Cosine, Exponential, Sqrt, Scaled Linear)
- Beta Sampling (arXiv:2407.12173, RES4LYF ComfyUI)

This scheduler separates NOISE SCHEDULE from SAMPLING METHOD,
allowing different timestep distributions with Euler integration.
"""

import math
from typing import TYPE_CHECKING, Literal

import mlx.core as mx

if TYPE_CHECKING:
    from mflux.config.runtime_config import RuntimeConfig

from mflux.schedulers.base_scheduler import BaseScheduler


class FlowMatchAdvancedScheduler(BaseScheduler):
    """
    Flow Matching scheduler with multiple noise schedule options.

    Separates noise schedule from sampling method (Euler integration).

    Args:
        runtime_config: Runtime configuration
        schedule: Noise schedule type
                 - "linear": Uniform spacing (baseline, fast)
                 - "cosine": Smoother transitions, better perceptual quality
                 - "exponential": Faster early denoising, refined details at end
                 - "sqrt": Preserves structure, good detail in complex areas
                 - "scaled_linear": Adaptive scaling for different image types
                 - "beta": Beta distribution (complex, concentrates steps at edges)
        exponential_beta: Beta parameter for exponential schedule (default: 5.0)
        beta_alpha: Alpha parameter for beta schedule (default: 0.6)
        beta_beta: Beta parameter for beta schedule (default: 0.6)

    Examples:
        # Cosine (smooth, good perceptual quality)
        --scheduler advanced --scheduler-kwargs '{"schedule": "cosine"}'

        # Exponential (fast early, refined end)
        --scheduler advanced --scheduler-kwargs '{"schedule": "exponential"}'

        # Sqrt (structure preservation)
        --scheduler advanced --scheduler-kwargs '{"schedule": "sqrt"}'

        # Beta RES_2M (from ComfyUI/Reddit)
        --scheduler advanced --scheduler-kwargs '{"schedule": "beta", "beta_alpha": 2.0, "beta_beta": 1.0}'
    """

    def __init__(
        self,
        runtime_config: "RuntimeConfig",
        schedule: Literal["linear", "cosine", "exponential", "sqrt", "scaled_linear", "beta"] = "linear",
        exponential_beta: float = 5.0,
        beta_alpha: float = 0.6,
        beta_beta: float = 0.6,
        **kwargs,
    ):
        self.runtime_config = runtime_config
        self.model_config = runtime_config.model_config
        self.schedule = schedule
        self.exponential_beta = exponential_beta
        self.beta_alpha = beta_alpha
        self.beta_beta = beta_beta

        # Compute sigma schedule
        self._sigmas, self._timesteps = self._compute_timesteps_and_sigmas()

    def _compute_timesteps_and_sigmas(self) -> tuple[mx.array, mx.array]:
        """Compute timesteps and sigmas using selected distribution."""
        num_steps = self.runtime_config.num_inference_steps

        # Generate normalized timesteps [0, 1] using selected schedule
        if self.schedule == "beta":
            timesteps_normalized = self._beta_schedule(num_steps)
        elif self.schedule == "cosine":
            timesteps_normalized = self._cosine_schedule(num_steps)
        elif self.schedule == "exponential":
            timesteps_normalized = self._exponential_schedule(num_steps)
        elif self.schedule == "sqrt":
            timesteps_normalized = self._sqrt_schedule(num_steps)
        elif self.schedule == "scaled_linear":
            timesteps_normalized = self._scaled_linear_schedule(num_steps)
        else:  # linear
            timesteps_normalized = self._linear_schedule(num_steps)

        # Convert to sigmas for Flow Matching
        # sigmas go from 1.0 (pure noise) to 0.0 (clean data)
        sigmas = [1.0 - t for t in timesteps_normalized]

        # Apply sigma shift if required by the model (e.g., for Qwen)
        if self.model_config.requires_sigma_shift:
            sigmas = self._apply_sigma_shift(sigmas)

        sigmas_arr = mx.array(sigmas, dtype=mx.float32)
        timesteps_arr = mx.arange(num_steps, dtype=mx.float32)

        return sigmas_arr, timesteps_arr

    def _linear_schedule(self, num_steps: int) -> list[float]:
        """Linear (uniform) timestep distribution."""
        return [i / num_steps for i in range(num_steps + 1)]

    def _cosine_schedule(self, num_steps: int) -> list[float]:
        """
        Cosine noise schedule.

        Provides smoother transitions between noise levels using cosine function.
        Often results in better perceptual quality.

        Formula: cos(((t + s) / (1 + s)) * π/2)²
        where s = 0.008 (small offset to avoid singularities)
        """
        import math

        s = 0.008  # Small offset
        timesteps = []

        for i in range(num_steps + 1):
            t = i / num_steps
            # Cosine schedule: cos((t+s)/(1+s) * π/2)²
            value = math.cos(((t + s) / (1.0 + s)) * math.pi * 0.5) ** 2
            timesteps.append(value)

        # Normalize to [0, 1]
        first = timesteps[0]
        timesteps = [t / first for t in timesteps]

        # Invert so it goes from 0 to 1 (not 1 to 0)
        timesteps = [1.0 - t for t in timesteps]

        return timesteps

    def _exponential_schedule(self, num_steps: int) -> list[float]:
        """
        Exponential noise schedule.

        Accelerates denoising at the beginning and slows at the end for
        more detail refinement.

        Uses a gentler exponential curve that's more compatible with Flow Matching.
        """

        timesteps = []
        for i in range(num_steps + 1):
            t = i / num_steps
            # Gentler exponential: use (1 - t)^beta instead of exp(-beta*t)
            # This gives more control and better convergence
            value = (1.0 - t) ** self.exponential_beta
            timesteps.append(value)

        # These values go from 1.0 to 0.0, need to invert for timesteps
        timesteps = [1.0 - t for t in timesteps]

        return timesteps

    def _sqrt_schedule(self, num_steps: int) -> list[float]:
        """
        Square root noise schedule.

        Helps preserve global structure while ensuring good detail
        preservation in complex areas.

        Formula: sqrt(1 - t)
        """
        import math

        timesteps = []
        for i in range(num_steps + 1):
            t = i / num_steps
            # Square root transformation
            value = math.sqrt(1.0 - t)
            timesteps.append(value)

        # Invert so it goes from 0 to 1
        timesteps = [1.0 - t for t in timesteps]

        return timesteps

    def _scaled_linear_schedule(self, num_steps: int) -> list[float]:
        """
        Scaled linear noise schedule.

        Adaptive scaling for different image types. Used in Stable Diffusion.
        Slower initial denoising and faster final denoising.

        Formula: (sqrt(beta_start) to sqrt(beta_end))²
        """
        import math

        # Standard beta values from Stable Diffusion
        beta_start = 0.00085
        beta_end = 0.012

        timesteps = []
        for i in range(num_steps + 1):
            t = i / num_steps
            # Scaled linear: interpolate sqrt of betas, then square
            sqrt_beta = math.sqrt(beta_start) + t * (math.sqrt(beta_end) - math.sqrt(beta_start))
            beta = sqrt_beta**2
            timesteps.append(beta)

        # Normalize to [0, 1]
        first = timesteps[0]
        last = timesteps[-1]
        timesteps = [(t - first) / (last - first) for t in timesteps]

        return timesteps

    def _beta_schedule(self, num_steps: int) -> list[float]:
        """
        Beta distribution timestep schedule using Beta ppf approximation.

        Based on arXiv:2407.12173 Beta sampling. Matches ComfyUI implementation:
        ts = 1 - linspace(0, 1, steps)
        ts = beta.ppf(ts, alpha, beta)

        For Beta(α, β):
        - When α > β: concentrates steps at high noise (start of denoising)
        - When α < β: concentrates steps at low noise (end of denoising)
        - When α = β: symmetric distribution

        Returns timesteps from 0.0 to 1.0 (will be inverted to sigmas 1.0 to 0.0 later).
        """

        # Generate uniform samples from 1.0 to 0.0 (matching ComfyUI: 1 - linspace(0,1))
        # This is the CDF values we want to invert
        timesteps_beta = []

        for i in range(num_steps + 1):
            # Uniform value from 1.0 to 0.0
            u = 1.0 - (i / num_steps)

            # Approximate beta.ppf(u, alpha, beta)
            if u <= 0.0:
                t = 0.0
            elif u >= 1.0:
                t = 1.0
            else:
                t = self._beta_ppf_scalar(u, self.beta_alpha, self.beta_beta)

            timesteps_beta.append(t)

        # timesteps_beta now goes from 1.0 to 0.0 (beta ppf output)
        # But we need to return 0.0 to 1.0 (to match _linear_schedule convention)
        # So invert it
        timesteps_normalized = [1.0 - t for t in timesteps_beta]

        return timesteps_normalized

    def _beta_ppf_scalar(self, p: float, alpha: float, beta: float) -> float:
        """
        Approximate Beta inverse CDF (ppf) for a scalar value.
        Uses iterative refinement with Newton's method.
        """

        # Clamp p to valid range
        p = max(1e-10, min(1.0 - 1e-10, p))

        # Initial guess using moment matching
        # Mean = alpha / (alpha + beta)
        # Use a better initial guess based on parameter values
        if alpha > 1.0 and beta > 1.0:
            # For both > 1, use mode as initial guess weighted by p
            mode = (alpha - 1.0) / (alpha + beta - 2.0)
            # Adjust towards 0 or 1 based on p
            if p < 0.5:
                x = mode * (2.0 * p) ** (1.0 / alpha)
            else:
                x = 1.0 - (1.0 - mode) * (2.0 * (1.0 - p)) ** (1.0 / beta)
        elif alpha <= 1.0 and beta > 1.0:
            # Concentrated near 0
            x = p ** (1.0 / alpha)
        elif alpha > 1.0 and beta <= 1.0:
            # Concentrated near 1
            x = 1.0 - (1.0 - p) ** (1.0 / beta)
        else:
            # Both <= 1, use simple power law
            x = p

        x = max(0.01, min(0.99, x))

        # Newton-Raphson refinement (5 iterations should be enough)
        for _ in range(8):
            # Compute CDF and PDF at current x
            cdf = self._beta_cdf_scalar(x, alpha, beta)
            pdf = self._beta_pdf_scalar(x, alpha, beta)

            if pdf < 1e-10:
                break

            # Newton step
            error = cdf - p
            x = x - error / pdf

            # Clamp to valid range
            x = max(1e-8, min(1.0 - 1e-8, x))

            # Check convergence
            if abs(error) < 1e-6:
                break

        return max(0.0, min(1.0, x))

    def _beta_cdf_scalar(self, x: float, alpha: float, beta: float) -> float:
        """
        Regularized incomplete beta function I_x(a,b) using continued fraction.
        This is the standard method from Numerical Recipes.
        """
        import math

        if x <= 0.0:
            return 0.0
        if x >= 1.0:
            return 1.0

        # Use symmetry for better convergence: I_x(a,b) = 1 - I_{1-x}(b,a)
        if x > (alpha + 1.0) / (alpha + beta + 2.0):
            return 1.0 - self._beta_cdf_scalar(1.0 - x, beta, alpha)

        # Compute the leading factor
        log_beta = math.lgamma(alpha) + math.lgamma(beta) - math.lgamma(alpha + beta)
        bt = math.exp(alpha * math.log(x) + beta * math.log(1.0 - x) - log_beta - math.log(alpha))

        # Compute continued fraction
        cf = self._beta_continued_fraction(x, alpha, beta)

        return bt * cf

    def _beta_continued_fraction(self, x: float, alpha: float, beta: float, max_iter: int = 200) -> float:
        """
        Evaluate continued fraction for incomplete beta function.
        Uses modified Lentz's method.
        """

        qab = alpha + beta
        qap = alpha + 1.0
        qam = alpha - 1.0

        # First step
        c = 1.0
        d = 1.0 - qab * x / qap

        if abs(d) < 1e-30:
            d = 1e-30
        d = 1.0 / d
        h = d

        for m in range(1, max_iter):
            m2 = 2 * m

            # Even step
            aa = m * (beta - m) * x / ((qam + m2) * (alpha + m2))
            d = 1.0 + aa * d
            if abs(d) < 1e-30:
                d = 1e-30
            c = 1.0 + aa / c
            if abs(c) < 1e-30:
                c = 1e-30
            d = 1.0 / d
            h *= d * c

            # Odd step
            aa = -(alpha + m) * (qab + m) * x / ((alpha + m2) * (qap + m2))
            d = 1.0 + aa * d
            if abs(d) < 1e-30:
                d = 1e-30
            c = 1.0 + aa / c
            if abs(c) < 1e-30:
                c = 1e-30
            d = 1.0 / d
            delta = d * c
            h *= delta

            if abs(delta - 1.0) < 3e-7:
                break

        return h

    def _beta_pdf_scalar(self, x: float, alpha: float, beta: float) -> float:
        """
        Beta PDF for scalar x.
        PDF(x) = x^(α-1) * (1-x)^(β-1) / B(α,β)
        """
        import math

        if x <= 0.0 or x >= 1.0:
            return 0.0

        # Compute in log space for numerical stability
        log_beta = math.lgamma(alpha) + math.lgamma(beta) - math.lgamma(alpha + beta)
        log_pdf = (alpha - 1.0) * math.log(x) + (beta - 1.0) * math.log(1.0 - x) - log_beta

        return math.exp(log_pdf)

    def _apply_sigma_shift(self, sigmas: list[float]) -> list[float]:
        """
        Apply exponential sigma shift for resolution-dependent adjustment.
        Required for models like Qwen that need resolution-based scaling.
        """
        # Calculate mu based on resolution
        y1 = 0.5
        x1 = 256
        m = (1.15 - y1) / (4096 - x1)
        b = y1 - m * x1
        mu = m * self.runtime_config.width * self.runtime_config.height / 256 + b

        # Apply exponential shift
        shifted_sigmas = []
        for s in sigmas:
            if s > 0:
                shifted = math.exp(mu) / (math.exp(mu) + (1 / s - 1))
                shifted_sigmas.append(shifted)
            else:
                shifted_sigmas.append(0.0)

        return shifted_sigmas

    @property
    def sigmas(self) -> mx.array:
        """Return the sigma schedule."""
        return self._sigmas

    @property
    def timesteps(self) -> mx.array:
        """Return the timestep indices."""
        return self._timesteps

    def step(self, model_output: mx.array, timestep: int, sample: mx.array, **kwargs) -> mx.array:
        """
        Perform one denoising step using Euler method for Flow Matching.

        For Flow Matching with velocity prediction v(x_t, t):
        x_{t+dt} = x_t + dt * v(x_t, t)

        Args:
            model_output: Predicted velocity v(x_t, t) from the model
            timestep: Current timestep index (0 to num_steps-1)
            sample: Current sample x_t

        Returns:
            Updated sample x_{t+dt}
        """
        # Get current and next sigma values
        sigma_t = self._sigmas[timestep]
        sigma_t_plus_1 = self._sigmas[timestep + 1]

        # Compute step size (dt)
        dt = sigma_t_plus_1 - sigma_t

        # Euler step for Flow Matching
        pred_sample = sample + dt * model_output

        return pred_sample

    def scale_model_input(self, latents: mx.array, t: int) -> mx.array:
        """
        Scale the model input. Flow Matching doesn't require input scaling.
        """
        return latents
