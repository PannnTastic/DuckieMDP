from math import pi
from types import SimpleNamespace

import numpy as np

from src.state import (
    StateConfig,
    TileType,
    next_stop_candidate,
    tile_ahead,
)


class FakeEnv:
    def __init__(self):
        self.cur_pos = np.array([0.0, 0.0, 0.0])
        self.cur_angle = 0.0
        self.objects = []

    def closest_curve_point(self, pos, angle):
        return np.array([0.0, 0.0, 0.0]), np.array([1.0, 0.0, 0.0])

    def get_grid_coords(self, pos):
        return (1, 0) if pos[0] >= 0.2 else (0, 0)

    def _get_tile(self, i, j):
        kind = "curve_right" if i == 1 else "straight"
        return {"kind": kind, "drivable": True}


def sign(pos, angle):
    return SimpleNamespace(kind="sign_stop", visible=True, pos=np.array(pos), angle=angle)


def test_stop_filter_uses_lateral_and_orientation():
    env = FakeEnv()
    env.objects = [
        sign([1.0, 0.0, 0.2], pi),
        sign([0.5, 0.0, 0.1], 0.0),
        sign([0.4, 0.0, 0.6], pi),
    ]
    distance, stop_id = next_stop_candidate(env, StateConfig())
    assert stop_id == 0
    assert distance == 0.8


def test_tile_lookahead_uses_lane_tangent():
    env = FakeEnv()
    assert tile_ahead(env, StateConfig(tile_lookahead=0.3)) == TileType.CURVE_RIGHT

