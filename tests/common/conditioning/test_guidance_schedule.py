import pytest

from mflux.models.common.conditioning import GuidanceSchedule


class TestGuidanceScheduleParseSchedule:
    @pytest.mark.fast
    def test_parses_multiple_segments(self):
        points = GuidanceSchedule.parse_schedule("0.0-0.5:1.0;0.5-1.0:0.8")

        assert points == [(0.0, 0.5, 1.0), (0.5, 1.0, 0.8)]

    @pytest.mark.fast
    def test_sorts_out_of_order_segments(self):
        points = GuidanceSchedule.parse_schedule("0.5-1.0:0.8;0.0-0.5:1.0")

        assert points == [(0.0, 0.5, 1.0), (0.5, 1.0, 0.8)]

    @pytest.mark.fast
    def test_empty_spec_raises(self):
        with pytest.raises(ValueError):
            GuidanceSchedule.parse_schedule("")

    @pytest.mark.fast
    def test_malformed_segment_raises(self):
        with pytest.raises(ValueError):
            GuidanceSchedule.parse_schedule("not-a-schedule")


class TestGuidanceScheduleToPerStep:
    @pytest.mark.fast
    def test_maps_steps_to_segment_values(self):
        points = GuidanceSchedule.parse_schedule("0.0-0.5:1.0;0.5-1.0:0.8")

        values = GuidanceSchedule.schedule_to_per_step(points, num_steps=4)

        assert values == [1.0, 1.0, 0.8, 0.8]

    @pytest.mark.fast
    def test_single_segment_covers_all_steps(self):
        points = GuidanceSchedule.parse_schedule("0.0-1.0:3.5")

        values = GuidanceSchedule.schedule_to_per_step(points, num_steps=5)

        assert values == [3.5] * 5
