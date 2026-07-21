from dataclasses import replace

import numpy as np
import pytest

from src.discretizer import Q_SHAPE
from src.explainability.primitives import DrivingPrimitive, PrimitiveLabel
from src.explainability.q_policy_adapter import QPolicyAdapter
from src.explainability.schema import CanonicalState
from src.explainability.temporal_outcomes import (
    compute_physical_outcome,
    compute_reward_profile,
)
from src.explainability.trajectory import TrajectoryRecorder


def _state(**overrides):
    values = dict(
        d=0.0,
        phi=0.0,
        v=0.2,
        curvature=None,
        curvature_class="straight",
        stop_present=False,
        stop_distance=None,
        stop_satisfied=False,
        stop_hold_progress=0.0,
        duck_present=False,
        duck_threat="none",
        duck_longitudinal=None,
        duck_lateral=None,
        duck_v_longitudinal_relative=None,
        duck_v_lateral_relative=None,
        duck_active=None,
        duck_crossing_available=None,
        source_representation="unit_test",
    )
    values.update(overrides)
    return CanonicalState(**values)


def _label(value):
    return PrimitiveLabel(value, "test trigger", "test.rule", False)


def _trajectory():
    table = np.zeros(Q_SHAPE, dtype=np.float32)
    policy = QPolicyAdapter(table)
    base = policy.decide(_state())
    recorder = TrajectoryRecorder("temporal", {}, decision_dt_seconds=0.2)
    states = [
        _state(d=0.10, phi=0.10, v=0.20, duck_present=True,
               duck_threat="crossing_far", duck_longitudinal=0.5,
               duck_lateral=0.2),
        _state(d=0.05, phi=0.05, v=0.15, duck_present=True,
               duck_threat="crossing_near", duck_longitudinal=0.3,
               duck_lateral=0.1),
        _state(d=0.01, phi=0.01, v=0.0),
    ]
    rewards = [1.0, 2.0, -1.0]
    primitives = [
        DrivingPrimitive.YIELD_DECELERATE,
        DrivingPrimitive.YIELD_HOLD,
        DrivingPrimitive.RESUME_AFTER_YIELD,
    ]
    omegas = [0.5, -0.5, 0.0]
    for index, (state, reward, primitive, omega) in enumerate(
        zip(states, rewards, primitives, omegas)
    ):
        decision = replace(
            base,
            state=state,
            action=replace(base.action, v_cmd=state.v, omega_cmd=omega),
        )
        recorder.append(
            decision,
            _label(primitive),
            reward,
            info={
                "reward_terms": {
                    "progress": reward * 0.75,
                    "safety": reward * 0.25,
                    "total": reward,
                },
                "events": {
                    "full_stop": index == 1,
                    "passed_stop": False,
                    "stop_violation": False,
                    "offroad": False,
                    "collision_duck": False,
                    "other_collision": False,
                },
                "termination_reason": "in_progress",
            },
        )
    return recorder.finalize()


def test_reward_profile_separates_discounted_terms_and_horizons():
    profile = compute_reward_profile(_trajectory(), gamma=0.5, horizons=(1, 2, 5))
    assert [point.horizon_steps for point in profile] == [1, 2, 3]
    assert profile[1].discounted_total == pytest.approx(2.0)
    assert profile[1].undiscounted_total == pytest.approx(3.0)
    assert profile[1].discounted_terms["progress"] == pytest.approx(1.5)
    assert profile[-1].discounted_total == pytest.approx(1.75)


def test_physical_profile_reports_semantics_not_only_reward():
    outcome = compute_physical_outcome(
        _trajectory(), decision_dt_seconds=0.2, brake_command_threshold=0.04
    )
    assert outcome.steps == 3
    assert outcome.forward_progress_m > 0.0
    assert outcome.max_abs_lateral_error_m == pytest.approx(0.10)
    assert outcome.minimum_duck_clearance_m == pytest.approx((0.3**2 + 0.1**2) ** 0.5)
    assert outcome.full_stops == 1
    assert outcome.brake_steps == 1
    assert outcome.steering_reversals == 1
    assert outcome.primitive_sequence == (
        "YieldDecelerate",
        "YieldHold",
        "ResumeAfterYield",
    )


def test_invalid_temporal_profile_parameters_are_rejected():
    with pytest.raises(ValueError, match="gamma"):
        compute_reward_profile(_trajectory(), gamma=1.1)
    with pytest.raises(ValueError, match="positive horizon"):
        compute_reward_profile(_trajectory(), horizons=(0, -1))
