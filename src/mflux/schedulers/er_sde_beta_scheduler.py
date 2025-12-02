"""
ER-SDE Beta Scheduler for Flow Matching

Combines two powerful techniques:
1. Extended Reverse-Time SDE (arXiv:2309.06169) - high-quality sampling with ODE/SDE balance
2. Beta Sampling (arXiv:2407.12173) - optimal timestep distribution for detail preservation

This scheduler is optimized for Flow Matching models (FLUX, Qwen) and provides:
- Superior detail preservation through Beta-distributed timesteps
- Natural, non-plastic appearance through controlled stochasticity
- Controllable quality/speed trade-off via gamma parameter
"""

from typing import TYPE_CHECKING

import mlx.core as mx

if TYPE_CHECKING:
    from mflux.config.runtime_config import RuntimeConfig

from mflux.schedulers.base_scheduler import BaseScheduler


class ERSDEBetaScheduler(BaseScheduler):
    """
    Flow Matching scheduler with Beta timestep distribution and optional SDE component.

    Args:
        runtime_config: Runtime configuration
        gamma: Controls SDE noise injection (0.0 = pure ODE/deterministic, 0.3 = balanced)
               Default: 0.0 (deterministic)
        use_beta: Whether to use Beta distribution for timesteps (default: True)
        beta_strength: Controls how aggressive the Beta distribution is
                       1.0 = gentle (sin²), 2.0 = moderate, 3.0+ = aggressive
                       Default: 1.0
        use_subsequence: Use DDIM-style subsequence approach (creates different "character")
                         Default: False
        num_train_timesteps: Total timesteps for subsequence (only if use_subsequence=True)
                            Default: 1000
        subsequence_intensity: Blend between linear (0.0) and full subsequence (1.0)
                              0.0 = no effect, 0.5 = moderate character, 1.0 = full DDIM
                              Default: 1.0
        subsequence_start_point: At which normalized timestep (0-1) to start applying subsequence
                                0.0 = from beginning, 0.5 = from halfway, 1.0 = never
                                Useful: 0.0 for full effect, 0.3-0.5 for "texture only in late steps"
                                Default: 0.0 (apply throughout)
    """

    def __init__(
        self,
        runtime_config: "RuntimeConfig",
        gamma: float = 0.0,
        use_beta: bool = True,
        beta_strength: float = 1.0,
        use_subsequence: bool = False,
        num_train_timesteps: int = 1000,
        subsequence_intensity: float = 1.0,
        subsequence_start_point: float = 0.0,
        **kwargs,
    ):
        self.runtime_config = runtime_config
        self.model_config = runtime_config.model_config
        self.gamma = gamma
        self.use_beta = use_beta
        self.beta_strength = beta_strength
        self.use_subsequence = use_subsequence
        self.num_train_timesteps = num_train_timesteps
        self.subsequence_intensity = max(0.0, min(1.0, subsequence_intensity))  # Clamp to [0, 1]
        self.subsequence_start_point = max(0.0, min(1.0, subsequence_start_point))  # Clamp to [0, 1]

        # Compute sigma schedule
        self._sigmas, self._timesteps = self._compute_timesteps_and_sigmas()

    def _compute_timesteps_and_sigmas(self) -> tuple[mx.array, mx.array]:
        """
        Compute timesteps and sigmas using various distribution strategies.
        Can combine subsequence + beta for advanced control.
        """
        num_steps = self.runtime_config.num_inference_steps

        # Step 1: Choose base distribution
        if self.use_subsequence:
            # DDIM-style subsequence approach (like DDIM backup)
            # This creates the "vintage character" look
            step_ratio = self.num_train_timesteps // num_steps

            # Create subsequence: [0, step_ratio, 2*step_ratio, ..., num_train_timesteps]
            timesteps_indices = mx.arange(0, self.num_train_timesteps, step_ratio, dtype=mx.int32)
            timesteps_indices = timesteps_indices[:num_steps]
            timesteps_indices = timesteps_indices[::-1]  # Reverse for denoising

            # Convert timesteps to sigmas (DDIM-style: keep compressed range!)
            # This is the KEY difference from linear!
            sigmas_subsequence_raw = timesteps_indices.astype(mx.float32) / self.num_train_timesteps
            sigmas_subsequence_raw = mx.concatenate([sigmas_subsequence_raw, mx.zeros(1)])

            # IMPORTANT: Scale to full [0, 1] range based on intensity
            # intensity=1.0 -> keep DDIM compressed range (e.g., 0.923->0.0)
            # intensity=0.0 -> expand to full linear range (1.0->0.0)

            # Get max sigma from subsequence (e.g., 0.923)
            max_sigma_subseq = float(sigmas_subsequence_raw.max())

            # Scale subsequence to full range for blending
            # When intensity=1.0, we want the compressed range
            # When intensity=0.0, we want full 1.0->0.0 range
            target_max_sigma = max_sigma_subseq + (1.0 - max_sigma_subseq) * (1.0 - self.subsequence_intensity)

            # Scale subsequence to target range
            if max_sigma_subseq > 0:
                scale_factor = target_max_sigma / max_sigma_subseq
                sigmas_base = sigmas_subsequence_raw * scale_factor
            else:
                sigmas_base = sigmas_subsequence_raw

            # Convert to list for further processing
            sigmas = sigmas_base.tolist()

        elif self.use_beta:
            # Beta-inspired distribution: concentrates steps at edges
            # This improves detail preservation
            import math

            timesteps_normalized = []
            for i in range(num_steps + 1):
                t = i / num_steps
                # Use sin transformation with adjustable strength
                # beta_strength controls how aggressive the distribution is:
                # 1.0 = gentle (original sin²)
                # 2.0 = moderate (sin⁴)
                # 3.0+ = aggressive (concentrates more at edges)
                t_base = math.sin(math.pi * t / 2) ** 2
                t_transformed = t_base**self.beta_strength
                timesteps_normalized.append(t_transformed)

            # For Flow Matching: sigmas go from 1.0 (pure noise) to 0.0 (clean data)
            sigmas = [1.0 - t for t in timesteps_normalized]
        else:
            # Linear distribution (uniform spacing)
            timesteps_normalized = [i / num_steps for i in range(num_steps + 1)]
            sigmas = [1.0 - t for t in timesteps_normalized]

        # Step 2: Apply Beta transformation on top if both are enabled
        # This allows subsequence (for character) + beta (for detail focus)
        if self.use_subsequence and self.use_beta and self.beta_strength != 1.0:
            import math

            # Apply beta transformation to the subsequence result
            # Normalize sigmas to [0, 1] range
            sigma_min = min(sigmas)
            sigma_max = max(sigmas)
            sigma_range = sigma_max - sigma_min

            if sigma_range > 0:
                beta_transformed = []
                for s in sigmas:
                    # Normalize to [0, 1]
                    t_norm = (s - sigma_min) / sigma_range
                    # Apply beta transformation
                    t_base = math.sin(math.pi * t_norm / 2) ** 2
                    t_transformed = t_base**self.beta_strength
                    # Scale back to original range
                    s_new = sigma_min + t_transformed * sigma_range
                    beta_transformed.append(s_new)
                sigmas = beta_transformed

        # Apply sigma shift if required by the model
        if self.model_config.requires_sigma_shift:
            sigmas = self._apply_sigma_shift(sigmas)

        sigmas_arr = mx.array(sigmas, dtype=mx.float32)
        timesteps_arr = mx.arange(num_steps, dtype=mx.float32)

        return sigmas_arr, timesteps_arr

    def _apply_sigma_shift(self, sigmas: list[float]) -> list[float]:
        """
        Apply exponential sigma shift for resolution-dependent adjustment.
        Same logic as LinearScheduler for consistency.
        """
        import math

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
        Perform one denoising step for Flow Matching.

        For Flow Matching with velocity prediction v(x_t, t):
        - ODE update: x_{t+dt} = x_t + dt * v(x_t, t)
        - Optional SDE: add controlled noise based on gamma

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

        # ODE component: Euler step for Flow Matching
        # This is the base deterministic update
        pred_sample = sample + dt * model_output

        # Optional SDE component: add controlled noise
        # This can improve quality but adds stochasticity
        if self.gamma > 0 and timestep < len(self._timesteps) - 1:
            noise = mx.random.normal(sample.shape)
            # Scale noise by gamma and step size
            noise_scale = self.gamma * mx.sqrt(mx.abs(dt))
            pred_sample = pred_sample + noise_scale * noise

        return pred_sample

    def scale_model_input(self, latents: mx.array, t: int) -> mx.array:
        """
        Scale the model input. Flow Matching doesn't require input scaling.
        """
        return latents
