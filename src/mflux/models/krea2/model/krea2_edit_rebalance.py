import mlx.core as mx

from mflux.models.common.conditioning import ConditioningBands, ConditioningGuidance, ConditioningSchedule

# ComfyUI samples Krea 2 with a linear time-snr shift of 1.15 (supported_models
# Krea2.sampling_settings), and schedule percents there are sigma thresholds
# under that shift. mflux uses the reference exponential schedule (exp(1.15)),
# so the same step fraction lands on a very different noise level; percents must
# be recovered from the actual sigmas through the ComfyUI shift to reproduce the
# node's effective timing (otherwise the late x20.5 phase fires steps too early
# and heavily stylizes the output).
KREA2_COMFY_REFERENCE_SHIFT = 1.15

# Edit-rebalance recipe ported from ComfyUI-Conditioning-Rebalance (krea2.py,
# Krea2EditRebalance node with its default strengths). Band 8 (tap layer 26)
# carries the "subject" signal the recipe isolates and re-amplifies.
KREA2_EDIT_N_BANDS = 12
KREA2_EDIT_SUBJECT_WEIGHTS = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 12.0, 0.0, 0.0, 0.0]

# Time-windowed conditioning: the plain text encoding drives the first 20% of
# the denoise, then the compiled edit conditioning takes over with a ramping
# multiplier (0.8 -> 1.4 -> 20.5, "gradual" interpolation in 8 plateaus).
KREA2_EDIT_RAW_SCHEDULE = "0.000-0.200:1.00"
KREA2_EDIT_COMPILED_SCHEDULE = "0.200-0.750:0.80;0.750-0.875:1.40;0.875-1.000:20.50"
KREA2_EDIT_SUB_STEPS = 8


class Krea2EditRebalance:
    @staticmethod
    def compile_conditioning(
        cond_raw: mx.array,
        cond_image_main: mx.array,
        cond_image_ref: mx.array,
        guidance_strength: float = 1.0,
    ) -> mx.array:
        # Isolate the subject band on the image-conditioned main encoding. The
        # reference's refocus in the original node uses all-1.0 weights (a no-op)
        # and is omitted here.
        subject = ConditioningBands.scale_bands(cond_image_main, KREA2_EDIT_SUBJECT_WEIGHTS)
        first = ConditioningGuidance.project(subject, cond_image_ref, guidance_strength, KREA2_EDIT_N_BANDS)
        compiled = ConditioningBands.scale_bands(first, KREA2_EDIT_SUBJECT_WEIGHTS)
        second = ConditioningGuidance.project(cond_raw, compiled, -0.5, KREA2_EDIT_N_BANDS)
        return ConditioningGuidance.project(first, second, -0.5, KREA2_EDIT_N_BANDS)

    @staticmethod
    def per_step_plan(sigmas: mx.array, num_steps: int) -> list[tuple[int, float]]:
        # Index 0 = raw text conditioning, index 1 = compiled edit conditioning.
        progress = [Krea2EditRebalance._sigma_to_percent(float(sigmas[t])) for t in range(num_steps)]
        return ConditioningSchedule.per_step_plan(
            schedules=[KREA2_EDIT_RAW_SCHEDULE, KREA2_EDIT_COMPILED_SCHEDULE],
            num_steps=num_steps,
            interpolation="gradual",
            sub_steps=KREA2_EDIT_SUB_STEPS,
            progress_per_step=progress,
        )

    @staticmethod
    def _sigma_to_percent(sigma: float) -> float:
        # Invert ComfyUI's percent_to_sigma for flow models: sigma =
        # shift*t/(1+(shift-1)*t) with t = 1-percent and shift = 1.15.
        shift = KREA2_COMFY_REFERENCE_SHIFT
        t = sigma / (shift - (shift - 1.0) * sigma)
        return 1.0 - t
