import numpy as np
import pytest

from src.continuous_state import curve_signed_curvature


def _left_curve():
    return np.array(
        [
            [0.0, 0.0, 0.0],
            [0.3, 0.0, 0.0],
            [1.0, 0.0, -0.7],
            [1.0, 0.0, -1.0],
        ],
        dtype=float,
    )


def test_opposite_directed_curves_have_opposite_curvature():
    left = curve_signed_curvature(_left_curve())
    right = curve_signed_curvature(_left_curve()[::-1].copy())
    assert left > 0.0
    assert right < 0.0
    assert abs(left) == pytest.approx(abs(right), rel=1e-6)


def test_straight_curve_has_zero_curvature():
    straight = np.array(
        [[0, 0, 0], [0.3, 0, 0], [0.7, 0, 0], [1.0, 0, 0]],
        dtype=float,
    )
    assert curve_signed_curvature(straight) == pytest.approx(0.0)


def test_rotation_does_not_change_curvature():
    curve = _left_curve()
    angle = np.pi / 2.0
    rotation = np.array(
        [
            [np.cos(angle), 0.0, np.sin(angle)],
            [0.0, 1.0, 0.0],
            [-np.sin(angle), 0.0, np.cos(angle)],
        ]
    )
    rotated = curve @ rotation.T
    assert curve_signed_curvature(rotated) == pytest.approx(
        curve_signed_curvature(curve), rel=1e-6
    )