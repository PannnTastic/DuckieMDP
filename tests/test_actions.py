import numpy as np
import pytest
from src.actions import ActionConfig, action_to_wheels


def test_straight_has_equal_wheels():
    wheels = action_to_wheels(1)
    assert wheels[0] == pytest.approx(wheels[1])


def test_left_turn_has_faster_right_wheel():
    wheels = action_to_wheels(0)
    assert wheels[1] > wheels[0]


def test_brake_is_zero():
    assert np.array_equal(action_to_wheels(6), np.zeros(2))


def test_invalid_action():
    with pytest.raises(ValueError):
        action_to_wheels(7, ActionConfig())
