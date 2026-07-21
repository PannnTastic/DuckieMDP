from dataclasses import asdict

import pytest

from src.explainability.primitives import (
    PRIMITIVE_SCHEMA_VERSION,
    DrivingPrimitive,
    PrimitiveThresholds,
    label_primitive,
)
from src.explainability.schema import CanonicalAction, CanonicalState, SolverKind


def _state(**overrides):
    values = dict(
        d=0.0,
        phi=0.0,
        v=0.30,
        curvature=0.0,
        curvature_class="straight",
        stop_present=False,
        stop_distance=None,
        stop_satisfied=False,
        stop_hold_progress=0.0,
        duck_present=False,
        duck_threat=None,
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


def _sac_action(v=0.30, omega=0.0):
    return CanonicalAction(
        solver=SolverKind.SAC,
        v_cmd=v,
        omega_cmd=omega,
    )


def _q_action(action_id, name, v, omega):
    return CanonicalAction(
        solver=SolverKind.Q_LEARNING,
        action_id=action_id,
        action_name=name,
        v_cmd=v,
        omega_cmd=omega,
    )


def _label(state=None, action=None, **kwargs):
    result = label_primitive(
        state or _state(),
        action or _sac_action(),
        **kwargs,
    )
    assert result.trigger
    assert result.rule_id
    assert result.schema_version == PRIMITIVE_SCHEMA_VERSION
    return result.primitive


def test_brake_semantics_depend_on_context_not_action_name():
    brake = _sac_action(v=0.0)
    assert _label(action=brake) == DrivingPrimitive.UNNECESSARY_BRAKE
    assert _label(
        _state(stop_present=True, stop_distance=0.2), brake
    ) == DrivingPrimitive.STOP_HOLD
    assert _label(
        _state(
            duck_present=True,
            duck_longitudinal=0.2,
            duck_lateral=0.0,
            duck_active=True,
            duck_crossing_available=True,
        ),
        brake,
    ) == DrivingPrimitive.YIELD_HOLD
    assert _label(
        _state(d=0.20), brake
    ) == DrivingPrimitive.EMERGENCY_LANE_RECOVERY


def test_unsatisfied_stop_outranks_inactive_side_duck():
    state = _state(
        stop_present=True,
        stop_distance=0.2,
        duck_present=True,
        duck_longitudinal=0.7,
        duck_lateral=0.8,
        duck_active=False,
        duck_crossing_available=True,
    )
    assert _label(state, _sac_action(v=0.0)) == DrivingPrimitive.STOP_HOLD


def test_identical_semantics_are_solver_independent():
    state = _state(
        duck_present=True,
        duck_threat="crossing_near",
        duck_active=True,
    )
    q_label = _label(state, _q_action(6, "brake", 0.0, 0.0))
    sac_label = _label(state, _sac_action(0.0, 0.0))
    assert q_label == sac_label == DrivingPrimitive.YIELD_HOLD


@pytest.mark.parametrize(
    "state,action,expected",
    [
        (_state(), _sac_action(), DrivingPrimitive.CRUISE_STRAIGHT),
        (
            _state(curvature=1.0, curvature_class="curve_left"),
            _sac_action(0.30, 0.8),
            DrivingPrimitive.CRUISE_CURVE_LEFT,
        ),
        (
            _state(curvature=-1.0, curvature_class="curve_right"),
            _sac_action(0.30, -0.8),
            DrivingPrimitive.CRUISE_CURVE_RIGHT,
        ),
        (
            _state(d=0.08, phi=0.05),
            _sac_action(0.20, 0.8),
            DrivingPrimitive.LANE_CORRECT_LEFT,
        ),
        (
            _state(d=-0.08, phi=-0.05),
            _sac_action(0.20, -0.8),
            DrivingPrimitive.LANE_CORRECT_RIGHT,
        ),
        (
            _state(v=0.35),
            _sac_action(0.15, 0.0),
            DrivingPrimitive.DECELERATE_LANE,
        ),
        (
            _state(d=0.20),
            _sac_action(0.15, 0.8),
            DrivingPrimitive.EMERGENCY_LANE_RECOVERY,
        ),
        (
            _state(d=0.20),
            _sac_action(0.30, -0.8),
            DrivingPrimitive.LANE_DEPARTURE,
        ),
    ],
)
def test_lane_primitive_rules(state, action, expected):
    assert _label(state, action) == expected


def test_oscillation_and_lane_departure_event_have_explicit_precedence():
    result = _label(
        _state(),
        _sac_action(0.25, -1.0),
        previous_action=_sac_action(0.25, 1.0),
    )
    assert result == DrivingPrimitive.OSCILLATORY_STEERING
    assert _label(
        _state(),
        _sac_action(),
        events={"offroad": True},
    ) == DrivingPrimitive.LANE_DEPARTURE


def test_stop_sign_lifecycle_and_violation():
    assert _label(
        _state(stop_present=True, stop_distance=1.5),
        _sac_action(0.30),
    ) == DrivingPrimitive.APPROACH_STOP
    assert _label(
        _state(stop_present=True, stop_distance=0.8, v=0.30),
        _sac_action(0.15),
    ) == DrivingPrimitive.DECELERATE_STOP
    assert _label(
        _state(stop_present=True, stop_distance=0.2),
        _sac_action(0.0),
    ) == DrivingPrimitive.STOP_HOLD
    assert _label(
        _state(
            stop_present=True,
            stop_distance=0.2,
            stop_satisfied=True,
            stop_hold_progress=1.0,
        ),
        _sac_action(0.0),
        events={"full_stop": True},
    ) == DrivingPrimitive.STOP_SATISFIED
    assert _label(
        _state(
            stop_present=True,
            stop_distance=0.2,
            stop_satisfied=True,
            stop_hold_progress=1.0,
        ),
        _sac_action(0.15),
        previous_primitive=DrivingPrimitive.STOP_SATISFIED,
    ) == DrivingPrimitive.RESUME_AFTER_STOP
    assert _label(
        _state(),
        _sac_action(),
        events={"stop_violation": True},
    ) == DrivingPrimitive.STOP_VIOLATION


def test_pedestrian_lifecycle_and_unsafe_modes():
    side = _state(
        duck_present=True,
        duck_longitudinal=0.7,
        duck_lateral=0.8,
        duck_active=False,
        duck_crossing_available=True,
    )
    crossing = _state(
        duck_present=True,
        duck_longitudinal=0.3,
        duck_lateral=0.1,
        duck_active=True,
        duck_crossing_available=True,
    )
    assert _label(side, _sac_action(0.25)) == DrivingPrimitive.APPROACH_CROSSING
    assert _label(
        side,
        _sac_action(0.25),
        previous_primitive=DrivingPrimitive.APPROACH_CROSSING,
    ) == DrivingPrimitive.APPROACH_CROSSING
    far_duck = _state(
        duck_present=True,
        duck_longitudinal=2.0,
        duck_lateral=2.0,
        duck_active=False,
        duck_crossing_available=True,
    )
    assert _label(far_duck, _sac_action(0.25)) == DrivingPrimitive.CRUISE_STRAIGHT
    assert _label(crossing, _sac_action(0.15)) == DrivingPrimitive.YIELD_DECELERATE
    assert _label(crossing, _sac_action(0.0)) == DrivingPrimitive.YIELD_HOLD
    assert _label(
        side,
        _sac_action(0.0),
        previous_primitive=DrivingPrimitive.YIELD_HOLD,
    ) == DrivingPrimitive.WAIT_FOR_CLEARANCE
    assert _label(
        _state(),
        _sac_action(0.15),
        previous_primitive=DrivingPrimitive.WAIT_FOR_CLEARANCE,
    ) == DrivingPrimitive.RESUME_AFTER_YIELD
    assert _label(
        crossing,
        _sac_action(0.15),
        previous_primitive=DrivingPrimitive.YIELD_HOLD,
    ) == DrivingPrimitive.PREMATURE_RESUME
    assert _label(crossing, _sac_action(0.30)) == DrivingPrimitive.UNSAFE_PROCEED
    assert _label(
        _state(),
        _sac_action(),
        events={"collision_duck": True},
    ) == DrivingPrimitive.UNSAFE_PROCEED


def test_lexicon_and_threshold_configuration_are_complete_and_serializable():
    expected = {
        "CruiseStraight",
        "CruiseCurveLeft",
        "CruiseCurveRight",
        "LaneCorrectLeft",
        "LaneCorrectRight",
        "DecelerateLane",
        "EmergencyLaneRecovery",
        "ApproachStop",
        "DecelerateStop",
        "StopHold",
        "StopSatisfied",
        "ResumeAfterStop",
        "ApproachCrossing",
        "YieldDecelerate",
        "YieldHold",
        "WaitForClearance",
        "ResumeAfterYield",
        "UnnecessaryBrake",
        "UnsafeProceed",
        "StopViolation",
        "LaneDeparture",
        "OscillatorySteering",
        "PrematureResume",
        "Unknown",
    }
    assert {primitive.value for primitive in DrivingPrimitive} == expected
    thresholds = asdict(PrimitiveThresholds())
    assert thresholds["stop_hold_distance"] == pytest.approx(0.45)
    assert thresholds["duck_corridor_half_width"] == pytest.approx(0.40)
    assert thresholds["duck_approach_radius"] == pytest.approx(1.50)
    assert thresholds["duck_approach_longitudinal_min"] == pytest.approx(-0.30)
