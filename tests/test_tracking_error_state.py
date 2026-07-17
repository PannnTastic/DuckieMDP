from src.discretizer import discretize
from src.state import DuckThreat, RawState, TileType


def _state(d, phi):
    return RawState(d, phi, 0.17, TileType.STRAIGHT, None, False, DuckThreat.NONE)


def test_tracking_error_separates_same_heading_when_lateral_error_is_unsafe():
    safe = discretize(_state(0.00, 0.08))
    correction_needed = discretize(_state(0.06, 0.08))
    assert safe[1] == 2
    assert correction_needed[1] == 3
