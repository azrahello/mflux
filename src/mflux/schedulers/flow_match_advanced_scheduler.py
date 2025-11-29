"""
Flow Matching Advanced Scheduler with Multiple Timestep Distributions

Supports various timestep schedules (Beta, Tangent, Karras, Linear) with Euler sampler.
Based on research and community-tested configurations from:
- Beta Sampling (arXiv:2407.12173)
- RES4LYF ComfyUI extension
- Karras et al. noise schedules

This scheduler separates TIMESTEP DISTRIBUTION from SAMPLING METHOD,
following the ComfyUI paradigm where they are independent choices.
"""

import math
from typing import TYPE_CHECKING, Literal

import mlx.core as mx

if TYPE_CHECKING:
    from mflux.config.runtime_config import RuntimeConfig

from mflux.schedulers.base_scheduler import BaseScheduler


class FlowMatchAdvancedScheduler(BaseScheduler):
    """
    Flow Matching scheduler with advanced timestep distributions.

    Separates timestep schedule (Beta, Tangent, Karras, Linear) from
    sampling method (Euler integration for Flow Matching).

    Args:
        runtime_config: Runtime configuration
        schedule: Timestep distribution to use
                 - "linear": Uniform spacing (baseline)
                 - "beta": Beta distribution (concentrates steps at edges)
                 - "tangent": Arctangent-based (smooth non-linear)
                 - "karras": Karras et al. schedule (power-law)
        beta_alpha: Alpha parameter for beta schedule (default: 0.6)
                    Higher values → more steps at high noise
        beta_beta: Beta parameter for beta schedule (default: 0.6)
                   Higher values → more steps at low noise
        tangent_slope: Slope for tangent schedule (default: 1.0)
        tangent_offset: Offset for tangent schedule (default: 0.0)
        karras_rho: Rho parameter for Karras schedule (default: 7.0)

    Popular configurations:
        - RES_2M-like: schedule="beta", beta_alpha=2.0, beta_beta=1.0
        - Beta57: schedule="beta", beta_alpha=0.5, beta_beta=0.7
        - Standard Karras: schedule="karras", karras_rho=7.0
        - Bong Tangent: schedule="tangent" (with default params)
    """

    def __init__(
        self,
        runtime_config: "RuntimeConfig",
        schedule: Literal["linear", "beta", "tangent", "karras"] = "linear",
        beta_alpha: float = 0.6,
        beta_beta: float = 0.6,
        tangent_slope: float = 1.0,
        tangent_offset: float = 0.0,
        karras_rho: float = 7.0,
        **kwargs,
    ):
        self.runtime_config = runtime_config
        self.model_config = runtime_config.model_config
        self.schedule = schedule
        self.beta_alpha = beta_alpha
        self.beta_beta = beta_beta
        self.tangent_slope = tangent_slope
        self.tangent_offset = tangent_offset
        self.karras_rho = karras_rho

        # Compute sigma schedule
        self._sigmas, self._timesteps = self._compute_timesteps_and_sigmas()

    def _compute_timesteps_and_sigmas(self) -> tuple[mx.array, mx.array]:
        """Compute timesteps and sigmas using selected distribution."""
        num_steps = self.runtime_config.num_inference_steps

        # Generate normalized timesteps [0, 1] using selected schedule
        if self.schedule == "beta":
            timesteps_normalized = self._beta_schedule(num_steps)
        elif self.schedule == "tangent":
            timesteps_normalized = self._tangent_schedule(num_steps)
        elif self.schedule == "karras":
            timesteps_normalized = self._karras_schedule(num_steps)
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

    def _beta_schedule(self, num_steps: int) -> list[float]:
        """
        Beta distribution timestep schedule using Newton-Raphson approximation.

        Based on arXiv:2407.12173 and ComfyUI implementation.
        Uses Beta(α, β) distribution to concentrate timesteps at edges.

        Implements inverse CDF (ppf) of Beta distribution without scipy dependency.
        """
        # Generate uniform samples in reverse order (1 → 0)
        uniform_samples = [1.0 - (i / num_steps) for i in range(num_steps + 1)]

        timesteps = []
        for p in uniform_samples:
            if p <= 0.0:
                t = 0.0
            elif p >= 1.0:
                t = 1.0
            else:
                # Newton-Raphson iteration for Beta inverse CDF
                # Find x such that I_x(α, β) = p
                # where I_x is the regularized incomplete beta function
                t = self._beta_ppf_newton(p, self.beta_alpha, self.beta_beta)
            timesteps.append(t)

        return timesteps

    def _beta_ppf_newton(self, p: float, alpha: float, beta: float, tol: float = 1e-6, max_iter: int = 100) -> float:
        """
        Compute Beta distribution inverse CDF using Newton-Raphson.

        Finds x such that CDF(x) = p, where CDF is the incomplete beta function.
        """
        # Initial guess using simple approximation
        if alpha > 1 and beta > 1:
            # Mode-based initial guess
            x = (alpha - 1) / (alpha + beta - 2)
        elif alpha <= 1 and beta > 1:
            x = 0.1
        elif alpha > 1 and beta <= 1:
            x = 0.9
        else:
            x = 0.5

        # Newton-Raphson iteration
        for _ in range(max_iter):
            # Compute CDF and PDF at current x
            cdf = self._beta_cdf(x, alpha, beta)
            pdf = self._beta_pdf(x, alpha, beta)

            if abs(pdf) < 1e-10:
                break

            # Newton step: x_new = x - f(x)/f'(x)
            # where f(x) = CDF(x) - p
            x_new = x - (cdf - p) / pdf

            # Clamp to [0, 1]
            x_new = max(0.0, min(1.0, x_new))

            if abs(x_new - x) < tol:
                return x_new

            x = x_new

        return x

    def _beta_cdf(self, x: float, alpha: float, beta: float) -> float:
        """Regularized incomplete beta function (CDF of Beta distribution)."""
        if x <= 0:
            return 0.0
        if x >= 1:
            return 1.0

        # Use continued fraction approximation
        return self._incomplete_beta(x, alpha, beta)

    def _beta_pdf(self, x: float, alpha: float, beta: float) -> float:
        """PDF of Beta distribution."""
        if x <= 0 or x >= 1:
            return 0.0

        # PDF: x^(α-1) * (1-x)^(β-1) / B(α,β)
        # where B(α,β) is the beta function
        log_pdf = (alpha - 1) * math.log(x) + (beta - 1) * math.log(1 - x)
        log_beta = self._log_beta_function(alpha, beta)

        return math.exp(log_pdf - log_beta)

    def _log_beta_function(self, alpha: float, beta: float) -> float:
        """Logarithm of Beta function using lgamma."""
        return math.lgamma(alpha) + math.lgamma(beta) - math.lgamma(alpha + beta)

    def _incomplete_beta(self, x: float, alpha: float, beta: float) -> float:
        """
        Regularized incomplete beta function using continued fraction.

        This is a simplified implementation. For production use, consider
        using scipy or a more robust numerical library.
        """
        # Symmetry transformation if needed
        if x > (alpha + 1) / (alpha + beta + 2):
            return 1.0 - self._incomplete_beta(1 - x, beta, alpha)

        # Continued fraction expansion
        bt = math.exp((alpha - 1) * math.log(x) + (beta - 1) * math.log(1 - x) - self._log_beta_function(alpha, beta))

        if x < (alpha + 1) / (alpha + beta + 2):
            # Use continued fraction directly
            cf = self._beta_continued_fraction(x, alpha, beta)
            return bt * cf / alpha
        else:
            # Use symmetry
            cf = self._beta_continued_fraction(1 - x, beta, alpha)
            return 1.0 - bt * cf / beta

    def _beta_continued_fraction(self, x: float, alpha: float, beta: float, max_iter: int = 200) -> float:
        """Evaluate continued fraction for incomplete beta function."""
        qab = alpha + beta
        qap = alpha + 1.0
        qam = alpha - 1.0
        c = 1.0
        d = 1.0 - qab * x / qap

        if abs(d) < 1e-30:
            d = 1e-30
        d = 1.0 / d
        h = d

        for m in range(1, max_iter):
            m2 = 2 * m
            aa = m * (beta - m) * x / ((qam + m2) * (alpha + m2))
            d = 1.0 + aa * d
            if abs(d) < 1e-30:
                d = 1e-30
            c = 1.0 + aa / c
            if abs(c) < 1e-30:
                c = 1e-30
            d = 1.0 / d
            h *= d * c

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

            if abs(delta - 1.0) < 1e-7:
                break

        return h

    def _tangent_schedule(self, num_steps: int) -> list[float]:
        """
        Arctangent-based timestep distribution (Bong Tangent style).

        Uses arctan transformation for smooth non-linear spacing.
        Formula: ((2/π) * arctan(-slope * (x - offset)) + 1) / 2
        """
        # Calculate range for normalization
        smax = ((2 / math.pi) * math.atan(-self.tangent_slope * (0 - self.tangent_offset)) + 1) / 2
        smin = ((2 / math.pi) * math.atan(-self.tangent_slope * ((num_steps - 1) - self.tangent_offset)) + 1) / 2
        srange = smax - smin

        timesteps = []
        for x in range(num_steps + 1):
            # Apply arctan transformation
            t = ((2 / math.pi) * math.atan(-self.tangent_slope * (x - self.tangent_offset)) + 1) / 2
            # Normalize to [0, 1]
            t_norm = (t - smin) / srange if srange > 0 else 0.0
            timesteps.append(t_norm)

        return timesteps

    def _karras_schedule(self, num_steps: int) -> list[float]:
        """
        Karras et al. timestep distribution.

        Uses power-law spacing: σ(t) = (σ_max^(1/ρ) + t(σ_min^(1/ρ) - σ_max^(1/ρ)))^ρ

        Default rho=7.0 concentrates steps at high noise levels.
        """
        sigma_min = 0.0
        sigma_max = 1.0

        # Generate power-law distributed sigmas
        ramp = [i / num_steps for i in range(num_steps + 1)]
        min_inv_rho = sigma_min ** (1 / self.karras_rho)
        max_inv_rho = sigma_max ** (1 / self.karras_rho)

        sigmas_karras = [(max_inv_rho + t * (min_inv_rho - max_inv_rho)) ** self.karras_rho for t in ramp]

        # Convert sigmas back to timesteps (invert: t = 1 - σ)
        timesteps = [1.0 - s for s in sigmas_karras]
        return timesteps

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
