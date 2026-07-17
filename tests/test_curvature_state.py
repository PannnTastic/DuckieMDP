import numpy as np

from src.state import StateConfig, TileType, _ego_relative_curve


def test_same_map_kind_can_encode_opposite_ego_relative_curves():
    left = np.array([
        [0.0, 0.0, 0.0], [0.3, 0.0, 0.0],
        [1.0, 0.0, -0.7], [1.0, 0.0, -1.0],
    ])
    right = left[::-1].copy()
    tile = {"kind": "curve_left", "drivable": True, "curves": np.stack([left, right])}
    cfg = StateConfig()
    assert _ego_relative_curve(tile, np.array([1.0, 0.0, 0.0]), cfg.curvature_threshold) == TileType.CURVE_LEFT
    assert _ego_relative_curve(tile, np.array([0.0, 0.0, 1.0]), cfg.curvature_threshold) == TileType.CURVE_RIGHT

