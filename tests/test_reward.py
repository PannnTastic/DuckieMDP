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


def test_stop_requires_configured_consecutive_hold_steps():
    tracker = StopTracker(hold_steps=3)
    previous = state(0.1, 0.3)
    for expected_steps in (1, 2):
        sigma, event = tracker.update(previous, state(0.0, 0.3), 2, 2)
        assert not sigma and not event.full_stop
        assert tracker.hold_steps == expected_steps
        assert tracker.hold_progress == pytest.approx(expected_steps / 3.0)
        previous = state(0.0, 0.3)
    sigma, event = tracker.update(previous, state(0.0, 0.3), 2, 2)
    assert sigma and event.full_stop
    assert tracker.hold_progress == pytest.approx(1.0)


def test_stop_hold_must_be_consecutive():
    tracker = StopTracker(hold_steps=3)
    tracker.update(state(0.1, 0.3), state(0.0, 0.3), 2, 2)
    tracker.update(state(0.0, 0.3), state(0.1, 0.3), 2, 2)
    assert tracker.hold_steps == 0
    assert tracker.hold_progress == 0.0


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


def test_stop_approach_shaping_is_off_by_default_and_penalises_speed_when_enabled():
    # Disabled by default: no distance configured -> no shaping for any policy.
    off = compute_reward(state(v=0.30, stop=0.4, sigma=False), EventFlags())
    assert off.stop_approach == pytest.approx(0.0)

    cfg = RewardConfig(
        stop_approach_distance=0.60,
        stop_approach_speed=0.02,
        stop_approach_yield=1.0,
        stop_approach_unsafe=-5.0,
        unnecessary_stop=-0.5,
    )
    # Carrying speed toward an unsatisfied stop is penalised...
    fast = compute_reward(state(v=0.30, stop=0.4, sigma=False), EventFlags(), cfg)
    assert fast.stop_approach == pytest.approx(-5.0)
    # ...being at the full-stop speed earns credit...
    stopped = compute_reward(state(v=0.0, stop=0.4, sigma=False), EventFlags(), cfg)
    assert stopped.stop_approach == pytest.approx(1.0)
    # ...it stops once the stop is satisfied, and never fires outside the zone.
    satisfied = compute_reward(state(v=0.0, stop=0.4, sigma=True), EventFlags(), cfg)
    assert satisfied.stop_approach == pytest.approx(0.0)
    far = compute_reward(state(v=0.30, stop=1.5, sigma=False), EventFlags(), cfg)
    assert far.stop_approach == pytest.approx(0.0)
    # The widened zone also exempts the slow agent from the idle penalty.
    assert stopped.stagnation == pytest.approx(0.0)


def test_straight_steering_penalty_uses_action_but_exempts_curves():
    cfg = RewardConfig(straight_steer_penalty=0.5, max_steer_command=1.5)
    straight = compute_reward(
        state(v=0.2), EventFlags(), cfg, action_omega=1.5, curvature=0.0
    )
    gentle = compute_reward(
        state(v=0.2), EventFlags(), cfg, action_omega=0.3, curvature=0.0
    )
    curve = compute_reward(
        state(v=0.2), EventFlags(), cfg, action_omega=1.5, curvature=2.0
    )
    assert straight.steering == pytest.approx(-0.5)
    assert gentle.steering == pytest.approx(-0.5 * (0.3 / 1.5) ** 2)
    assert curve.steering == pytest.approx(0.0)
