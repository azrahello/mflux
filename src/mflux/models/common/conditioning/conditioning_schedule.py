from mflux.models.common.conditioning.guidance_schedule import GuidanceSchedule


class ConditioningSchedule:
    @staticmethod
    def per_step_plan(
        schedules: list[str],
        num_steps: int,
        interpolation: str = "gradual",
        sub_steps: int = 8,
        progress_per_step: list[float] | None = None,
    ) -> list[tuple[int, float]]:
        # Each schedule string ("start-end:multiplier;...") describes when the
        # conditioning with the same list index is active and how strongly it is
        # scaled. Together the schedules must cover every step; where two overlap,
        # the earlier conditioning in the list wins. Returns one
        # (conditioning_index, multiplier) pair per denoise step.
        #
        # progress_per_step overrides the default uniform step-midpoint progress
        # for schedules whose start/end points live in a different time variable
        # (e.g. sigma-derived denoise percent rather than step fraction).
        parsed = [GuidanceSchedule.parse_schedule(s) for s in schedules]
        sub_steps = min(max(int(sub_steps), 1), 64)
        if progress_per_step is not None and len(progress_per_step) != num_steps:
            raise ValueError(f"progress_per_step has {len(progress_per_step)} entries, expected {num_steps}")

        plan: list[tuple[int, float]] = []
        for i in range(num_steps):
            progress = progress_per_step[i] if progress_per_step is not None else (i + 0.5) / num_steps
            for cond_index, points in enumerate(parsed):
                multiplier = ConditioningSchedule._value_at(points, progress, interpolation, sub_steps)
                if multiplier is not None:
                    plan.append((cond_index, multiplier))
                    break
            else:
                raise ValueError(
                    f"conditioning schedules {schedules!r} leave step {i + 1}/{num_steps} "
                    f"(progress {progress:.3f}) uncovered"
                )
        return plan

    @staticmethod
    def _value_at(
        points: list[tuple[float, float, float]],
        progress: float,
        interpolation: str,
        sub_steps: int,
    ) -> float | None:
        # Float dust from sigma-derived progress must not push a step landing
        # exactly on a segment boundary into the next segment; with the
        # tolerance, ties go to the earlier segment (first match wins).
        eps = 1e-4
        for i, (start, end, value) in enumerate(points):
            if not (start - eps <= progress <= end + eps):
                continue
            next_value = points[i + 1][2] if i + 1 < len(points) else value
            if interpolation == "gradual" and sub_steps > 1 and next_value != value and end > start:
                # Reference ramps each segment toward the next segment's value in
                # sub_steps plateaus, each held at its midpoint interpolant.
                fraction = (progress - start) / (end - start)
                k = min(int(fraction * sub_steps), sub_steps - 1)
                return value + (next_value - value) * ((k + 0.5) / sub_steps)
            return value
        return None
