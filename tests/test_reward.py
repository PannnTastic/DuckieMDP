import pytest

from src.reward import EventFlags, RewardConfig, StopTracker, compute_reward
from src.state import DuckThreat, RawState, TileType


def state(v=0.0, stop=None, sigma=False, duck=DuckThreat.NONE):
    return RawState(0.0, 0.0, v, TileType.STRAIGHT, stop, sigma, duck)


def test_full_stop_bonus_once():
    tracker = StopTracker()
    sigma, first = tracker.update(state(0.1, 0.3), state(0.0, 0.3), 2, 2)
    assert sigma and first.full_stop
    sigma, second = tracker.update(state(0.0, 0.3, True), state(0.0, 0.3, True), 2, 2)
    assert sigma and not second.full_stop


def test_stop_violation_when_sign_identity_changes():
    tracker = StopTracker()
    sigma, event = tracker.update(state(0.1, 0.4), state(0.1, 1.2), 2, 3)
    assert not sigma and event.stop_violation and event.passed_stop


def test_compliant_stop_does_not_become_violation():
    tracker = StopTracker()
    tracker.update(state(0.1, 0.3), state(0.0, 0.3), 2, 2)
    sigma, event = tracker.update(state(0.1, 0.2, True), state(0.1, None, True), 2, None)
    assert not sigma and event.passed_stop and not event.stop_violation


def test_event_rewards():
    full = compute_reward(state(), EventFlags(full_stop=True))
    violation = compute_reward(state(), EventFlags(stop_violation=True))
    collision = compute_reward(state(), EventFlags(collision_duck=True))
    other = compute_reward(state(), EventFlags(other_collision=True))
    timeout = compute_reward(state(), EventFlags(timeout=True))
    assert full.events == pytest.approx(10.0)
    assert violation.events == pytest.approx(-20.0)
    assert collision.events == pytest.approx(-100.0)
    assert other.events == pytest.approx(-50.0)
    assert timeout.events == pytest.approx(0.0)


def test_teacher_free_pedestrian_shaping_distinguishes_yield_and_unsafe_motion():
    cfg = RewardConfig(duck_yield=2.0, duck_unsafe=-5.0, duck_yield_speed=0.04)
    yielding = compute_reward(state(v=0.0, duck=DuckThreat.CROSSING_NEAR), EventFlags(), cfg)
    unsafe = compute_reward(state(v=0.2, duck=DuckThreat.CROSSING_NEAR), EventFlags(), cfg)
    absent = compute_reward(state(v=0.0, duck=DuckThreat.NONE), EventFlags(), cfg)

    assert yielding.pedestrian == pytest.approx(2.0)
    assert unsafe.pedestrian == pytest.approx(-5.0)
    assert absent.pedestrian == pytest.approx(0.0)


def test_unnecessary_stop_penalty_exempts_crossing_and_required_stop():
    cfg = RewardConfig(unnecessary_stop=-0.5, idle_speed=0.04)
    idle = compute_reward(state(v=0.0), EventFlags(), cfg)
    crossing = compute_reward(state(v=0.0, duck=DuckThreat.CROSSING_NEAR), EventFlags(), cfg)
    required_stop = compute_reward(state(v=0.0, stop=0.3, sigma=False), EventFlags(), cfg)

    assert idle.stagnation == pytest.approx(-0.5)
    assert crossing.stagnation == pytest.approx(0.0)
    assert required_stop.stagnation == pytest.approx(0.0)
