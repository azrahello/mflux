import pytest

from mflux.models.common.conditioning import ConditioningSchedule

KREA2_RAW = "0.000-0.200:1.00"
KREA2_COMPILED = "0.200-0.750:0.80;0.750-0.875:1.40;0.875-1.000:20.50"


class TestConditioningSchedulePerStepPlan:
    @pytest.mark.fast
    def test_single_constant_schedule(self):
        plan = ConditioningSchedule.per_step_plan(["0.0-1.0:2.5"], num_steps=4)

        assert plan == [(0, 2.5)] * 4

    @pytest.mark.fast
    def test_earlier_schedule_wins_on_overlap(self):
        plan = ConditioningSchedule.per_step_plan(["0.0-1.0:2.0", "0.0-1.0:3.0"], num_steps=3)

        assert plan == [(0, 2.0)] * 3

    @pytest.mark.fast
    def test_uncovered_step_raises(self):
        with pytest.raises(ValueError):
            ConditioningSchedule.per_step_plan(["0.0-0.3:1.0"], num_steps=4)

    @pytest.mark.fast
    def test_krea2_edit_recipe_plan_at_8_steps(self):
        plan = ConditioningSchedule.per_step_plan(
            [KREA2_RAW, KREA2_COMPILED], num_steps=8, interpolation="gradual", sub_steps=8
        )

        indices = [index for index, _ in plan]
        assert indices == [0, 0, 1, 1, 1, 1, 1, 1]

        # Gradual mode holds each of 8 sub-plateaus at its midpoint interpolant
        # between the segment's value and the next segment's value.
        multipliers = [multiplier for _, multiplier in plan]
        assert multipliers[0] == pytest.approx(1.0)
        assert multipliers[1] == pytest.approx(1.0)
        assert multipliers[2] == pytest.approx(0.8 + 0.6 * (1.5 / 8))  # 0.9125
        assert multipliers[3] == pytest.approx(0.8 + 0.6 * (3.5 / 8))  # 1.0625
        assert multipliers[4] == pytest.approx(0.8 + 0.6 * (5.5 / 8))  # 1.2125
        assert multipliers[5] == pytest.approx(0.8 + 0.6 * (7.5 / 8))  # 1.3625
        assert multipliers[6] == pytest.approx(1.4 + 19.1 * (4.5 / 8))  # 12.14375
        assert multipliers[7] == pytest.approx(20.5)

    @pytest.mark.fast
    def test_sharp_interpolation_holds_segment_values(self):
        plan = ConditioningSchedule.per_step_plan([KREA2_RAW, KREA2_COMPILED], num_steps=8, interpolation="sharp")

        multipliers = [multiplier for _, multiplier in plan]
        assert multipliers == [1.0, 1.0, 0.8, 0.8, 0.8, 0.8, 1.4, 20.5]
