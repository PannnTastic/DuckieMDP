from math import radians

import pytest

from src.env_wrapper import position_in_bounds_xz, route_circulation_score


@pytest.mark.parametrize(
    "position, angle_deg",
    [
        ((0.16, 0.0, 1.24), 270.0),  # left side, heading north
        ((0.88, 0.0, 1.55), 0.0),    # top side, heading east
        ((1.55, 0.0, 0.88), 90.0),   # right side, heading south
        ((0.88, 0.0, 0.20), 180.0),  # bottom side, heading west
    ],
)
def test_clockwise_route_has_positive_alignment(position, angle_deg):
    score = route_circulation_score(position, radians(angle_deg), (0.8775, 0.8775))
    assert score > 0.80


@pytest.mark.parametrize(
    "position, angle_deg",
    [
        ((0.16, 0.0, 1.24), 90.0),
        ((0.88, 0.0, 1.55), 180.0),
        ((1.55, 0.0, 0.88), 270.0),
        ((0.88, 0.0, 0.20), 0.0),
    ],
)
def test_counterclockwise_route_has_negative_alignment(position, angle_deg):
    score = route_circulation_score(position, radians(angle_deg), (0.8775, 0.8775))
    assert score < -0.80


def test_known_q_and_sac_render_spawns_have_opposite_circulation():
    center = (0.8775, 0.8775)
    q_score = route_circulation_score(
        (0.16171704, 0.0, 1.24021731), radians(270.7205), center
    )
    old_sac_score = route_circulation_score(
        (1.29545096, 0.0, 0.56399118), radians(273.7307), center
    )
    assert q_score > 0.50
    assert old_sac_score < -0.50


def test_before_stop_spawn_bounds_accept_left_approach_only():
    bounds = (0.10, 0.45, 0.45, 1.30)
    assert position_in_bounds_xz((0.16, 0.0, 1.24), bounds)
    assert position_in_bounds_xz((0.30, 0.80), bounds)
    assert not position_in_bounds_xz((0.35, 0.0, 1.55), bounds)
    assert not position_in_bounds_xz((1.30, 0.0, 0.56), bounds)
