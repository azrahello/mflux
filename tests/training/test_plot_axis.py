import datetime

from mflux.models.common.training.statistics.plotter import Plotter


def _elapsed(minutes: list[float]) -> list[float]:
    start = datetime.datetime(2026, 1, 1, 12, 0, 0)
    pairs = [(i * 10, start + datetime.timedelta(minutes=m)) for i, m in enumerate(minutes)]
    return Plotter._axis_timeline(pairs)[1]


def test_elapsed_tracks_wall_clock_and_never_goes_backwards():
    elapsed = _elapsed([i * 0.5 for i in range(240)])
    assert elapsed[0] == 0.0
    assert elapsed[-1] == 119.5
    assert all(b >= a for a, b in zip(elapsed, elapsed[1:]))


def test_a_pause_between_runs_costs_one_typical_step():
    # Steps take a minute; the run is stopped and resumed three hours later.
    resumed = _elapsed([0.0, 1.0, 2.0, 182.0, 183.0, 184.0])
    assert resumed[3] - resumed[2] == Plotter.FOCUS_GAP_FACTOR * 1.0
    assert resumed[-1] == 4.0 + Plotter.FOCUS_GAP_FACTOR * 1.0


def test_the_cap_follows_the_step_time_of_the_run():
    # Same three hour pause, but steps take ten times as long: the pause costs ten times more.
    slow = _elapsed([0.0, 10.0, 20.0, 200.0, 210.0, 220.0])
    assert slow[3] - slow[2] == Plotter.FOCUS_GAP_FACTOR * 10.0


def test_clock_going_backwards_does_not_move_the_axis():
    assert _elapsed([0.0, 1.0, 0.5, 2.0]) == [0.0, 1.0, 1.0, 2.5]


def test_degenerate_inputs():
    assert Plotter._axis_timeline([]) == ([], [])
    assert Plotter._axis_timeline([(0, datetime.datetime(2026, 1, 1))]) == ([0], [0.0])
    same = datetime.datetime(2026, 1, 1)
    assert Plotter._axis_timeline([(0, same), (5, same), (9, same)])[1] == [0.0, 1.0, 2.0]
