from src.lane_teacher import LaneTeacherConfig, select_lane_teacher_action, teacher_probability
from src.state import DuckThreat, RawState, TileType


def _state(d=0.0, phi=0.0, tile=TileType.STRAIGHT, stop=None, sigma=False, duck=DuckThreat.NONE):
    return RawState(d, phi, 0.17, tile, stop, sigma, duck)


def test_teacher_schedule_decays_after_full_control():
    cfg = LaneTeacherConfig(True, 10, 20, 0.10)
    assert teacher_probability(9, cfg) == 1.0
    assert teacher_probability(20, cfg) == 0.55
    assert teacher_probability(30, cfg) == 0.10


def test_teacher_uses_relative_curvature_and_lane_error():
    cfg = LaneTeacherConfig(enabled=True)
    assert select_lane_teacher_action(_state(tile=TileType.CURVE_LEFT), cfg) == 0
    assert select_lane_teacher_action(_state(tile=TileType.CURVE_RIGHT), cfg) == 5
    assert select_lane_teacher_action(_state(d=0.15), cfg) == 3
    assert select_lane_teacher_action(_state(d=-0.15), cfg) == 5


def test_teacher_brakes_for_unmet_stop_then_releases():
    cfg = LaneTeacherConfig(enabled=True)
    assert select_lane_teacher_action(_state(stop=0.30), cfg) == 6
    assert select_lane_teacher_action(_state(stop=0.30, sigma=True), cfg) == 1


def test_teacher_brakes_for_crossing_duck():
    cfg = LaneTeacherConfig(enabled=True)
    assert select_lane_teacher_action(_state(duck=DuckThreat.CROSSING_FAR), cfg) == 6
    assert select_lane_teacher_action(_state(duck=DuckThreat.CROSSING_NEAR), cfg) == 6
