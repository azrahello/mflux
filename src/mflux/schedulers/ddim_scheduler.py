"""
DDIM Scheduler - Denoising Diffusion Implicit Models

Based on the paper: "Denoising Diffusion Implicit Models" by Jiaming Song et al.
https://arxiv.org/abs/2010.02502

Key features:
- Deterministic sampling when eta=0
- Can reduce sampling steps from 1000 to 10-50 while maintaining quality
- 10x-50x speedup compared to DDPM
- Supports both deterministic (DDIM) and stochastic (DDPM-like) sampling
"""

import math
from typing import TYPE_CHECKING

import mlx.core as mx

if TYPE_CHECKING:
    from mflux.config.runtime_config import RuntimeConfig

from mflux.schedulers.base_scheduler import BaseScheduler


class DDIMScheduler(BaseScheduler):
    """
    DDIM (Denoising Diffusion Implicit Models) scheduler for faster sampling.

    Args:
        runtime_config: Runtime configuration containing inference steps and other params
        num_train_timesteps: Number of timesteps used during training (default: 1000)
        eta: Stochasticity parameter (0.0 = fully deterministic DDIM, 1.0 = DDPM-like)
        schedule_type: Type of alpha schedule ("linear" or "cosine")
    """

    def __init__(
        self,
        runtime_config: "RuntimeConfig",
        num_train_timesteps: int = 1000,
        eta: float = 0.0,
        schedule_type: str = "linear",
    ):
        self.runtime_config = runtime_config
        self.model_config = runtime_config.model_config
        self.num_train_timesteps = num_train_timesteps
        self.eta = eta  # Stochasticity parameter: 0.0 = deterministic, 1.0 = DDPM
        self.schedule_type = schedule_type

        # Compute alpha schedule
        self._alphas = self._compute_alpha_schedule()
        self._alphas_cumprod = mx.cumprod(self._alphas)

        # Compute timestep subsequence for accelerated sampling
        self._timesteps = self._compute_timestep_subsequence()

        # Compute sigmas for the accelerated timestep subsequence
        self._sigmas = self._compute_sigmas()

    def _compute_alpha_schedule(self) -> mx.array:
        """
        Compute the alpha schedule (αt in the paper).
        Supports both linear and cosine schedules.
        """
        if self.schedule_type == "linear":
            # Linear schedule: β goes from 0.0001 to 0.02
            beta_start = 0.0001
            beta_end = 0.02
            betas = mx.linspace(beta_start, beta_end, self.num_train_timesteps)
            alphas = 1.0 - betas
        elif self.schedule_type == "cosine":
            # Cosine schedule as in improved DDPM
            s = 0.008  # small offset to prevent beta from being too small
            steps = mx.arange(self.num_train_timesteps + 1, dtype=mx.float32)
            alphas_cumprod = mx.cos(((steps / self.num_train_timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
            alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
            betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
            betas = mx.clip(betas, 0.0001, 0.9999)
            alphas = 1.0 - betas
        else:
            raise ValueError(f"Unknown schedule_type: {self.schedule_type}")

        return alphas

    def _compute_timestep_subsequence(self) -> mx.array:
        """
        Compute the timestep subsequence τ for accelerated sampling.
        Uses a linear spacing strategy.
        """
        num_inference_steps = self.runtime_config.num_inference_steps

        # Create linear spacing from 0 to num_train_timesteps-1
        # This gives us a subsequence of the full timestep sequence
        step_ratio = self.num_train_timesteps // num_inference_steps
        timesteps = mx.arange(0, self.num_train_timesteps, step_ratio, dtype=mx.int32)

        # Ensure we have exactly num_inference_steps timesteps
        timesteps = timesteps[:num_inference_steps]

        # Reverse to go from high noise to low noise
        timesteps = timesteps[::-1]

        return timesteps

    def _compute_sigmas(self) -> mx.array:
        """
        Compute the sigma values for each timestep in the subsequence.

        From the paper (Eq. 16):
        σ_τi(η) = η * sqrt((1 - α_τi-1) / (1 - α_τi)) * sqrt(1 - α_τi / α_τi-1)
        """
        alphas_cumprod = self._alphas_cumprod
        timesteps = self._timesteps

        sigmas = []

        for i in range(len(timesteps)):
            t_curr = int(timesteps[i])

            # For the first step (highest noise), use previous timestep
            # Otherwise use the actual previous timestep in the subsequence
            if i == len(timesteps) - 1:
                t_prev = 0
            else:
                t_prev = int(timesteps[i + 1])

            alpha_prod_t = alphas_cumprod[t_curr]
            alpha_prod_t_prev = alphas_cumprod[t_prev] if t_prev > 0 else mx.array(1.0)

            # Compute sigma according to DDIM paper (Eq. 16)
            beta_prod_t = 1 - alpha_prod_t
            beta_prod_t_prev = 1 - alpha_prod_t_prev

            # σ_t = η * sqrt((1 - α_{t-1}) / (1 - α_t)) * sqrt(1 - α_t / α_{t-1})
            variance = (beta_prod_t_prev / beta_prod_t) * (1 - alpha_prod_t / alpha_prod_t_prev)
            sigma = self.eta * mx.sqrt(mx.maximum(variance, 0.0))

            sigmas.append(sigma)

        # Add final sigma of 0
        sigmas.append(mx.array(0.0))

        return mx.array(sigmas)

    @property
    def sigmas(self) -> mx.array:
        """Return the sigma schedule."""
        return self._sigmas

    @property
    def timesteps(self) -> mx.array:
        """Return the timestep subsequence."""
        return self._timesteps

    def step(self, model_output: mx.array, timestep: int, sample: mx.array, **kwargs) -> mx.array:
        """
        Perform one denoising step using DDIM sampling.

        From the paper (Eq. 12):
        x_{t-1} = sqrt(α_{t-1}) * ("predicted x0") +
                  sqrt(1 - α_{t-1} - σ_t^2) * ("direction pointing to x_t") +
                  σ_t * ε

        Args:
            model_output: The predicted noise ε_θ(x_t) from the model
            timestep: Index in the loop (0 to num_inference_steps-1)
            sample: Current sample x_t

        Returns:
            Previous sample x_{t-1}
        """
        # timestep is the index in the generation loop
        # We use it to index into our timesteps and sigmas arrays

        # Get current timestep from our subsequence
        t = int(self._timesteps[timestep])

        # Handle the case of last timestep in the loop
        if timestep == len(self._timesteps) - 1:
            t_prev = 0
        else:
            t_prev = int(self._timesteps[timestep + 1])

        # Get alpha values from the full training schedule
        alpha_prod_t = self._alphas_cumprod[t]
        alpha_prod_t_prev = self._alphas_cumprod[t_prev] if t_prev > 0 else mx.array(1.0)

        beta_prod_t = 1 - alpha_prod_t

        # Compute predicted original sample x_0 from x_t and predicted noise
        # x_0 = (x_t - sqrt(1 - α_t) * ε_θ) / sqrt(α_t)
        pred_original_sample = (sample - mx.sqrt(beta_prod_t) * model_output) / mx.sqrt(alpha_prod_t)

        # Compute coefficients for pred_original_sample and current sample
        # These come from equation 12 in the DDIM paper

        # Coefficient of predicted x_0
        pred_original_sample_coeff = mx.sqrt(alpha_prod_t_prev)

        # Coefficient of "direction pointing to x_t"
        current_sample_coeff = mx.sqrt(1 - alpha_prod_t_prev - self._sigmas[timestep] ** 2)

        # Compute the predicted previous sample mean
        pred_prev_sample = pred_original_sample_coeff * pred_original_sample + current_sample_coeff * model_output

        # Add noise for stochastic sampling (when eta > 0)
        if self.eta > 0:
            noise = mx.random.normal(sample.shape)
            variance = self._sigmas[timestep]
            pred_prev_sample = pred_prev_sample + variance * noise

        return pred_prev_sample

    def scale_model_input(self, latents: mx.array, t: int) -> mx.array:
        """
        Scale the model input. DDIM doesn't require input scaling.
        """
        return latents

    def get_alpha_prod_t(self, timestep: int) -> mx.array:
        """
        Get cumulative product of alphas at a given timestep.
        Useful for other operations like adding noise.
        """
        return self._alphas_cumprod[timestep]

    def add_noise(self, original_samples: mx.array, noise: mx.array, timestep: int) -> mx.array:
        """
        Add noise to the original samples according to the noise schedule.

        From the paper (Eq. 4):
        x_t = sqrt(α_t) * x_0 + sqrt(1 - α_t) * ε
        """
        alpha_prod = self._alphas_cumprod[timestep]
        sqrt_alpha_prod = mx.sqrt(alpha_prod)
        sqrt_one_minus_alpha_prod = mx.sqrt(1 - alpha_prod)

        noisy_samples = sqrt_alpha_prod * original_samples + sqrt_one_minus_alpha_prod * noise
        return noisy_samples
