from types import SimpleNamespace

import pytest

from src.explainability.compare_policies import (
    comparable_influence_subset,
    primitive_durations,
    primitive_frequency,
    primitive_transitions,
    q_supported_influence_signature,
    sac_ig_influence_signature,
    summarize_policy,
)
from src.explainability.primitives import DrivingPrimitive
from src.explainability.schema import PolicyMode, SolverKind


def _step(primitive, v, omega=0.0, stop=False, distance=None, satisfied=False,
          duck_active=False, events=None, phi=0.0, d=0.0,
          curvature=0.0, curvature_class="straight"):
    state = SimpleNamespace(
        stop_present=stop,
        stop_satisfied=satisfied,
        stop_distance=distance,
        duck_present=duck_active,
        duck_active=duck_active,
        curvature=curvature,
        curvature_class=curvature_class,
        phi=phi,
        d=d,
    )
    action = SimpleNamespace(v_cmd=v, omega_cmd=omega)
    label = SimpleNamespace(
        primitive=DrivingPrimitive(primitive),
        undesirable=DrivingPrimitive(primitive) in {
            DrivingPrimitive.UNNECESSARY_BRAKE,
            DrivingPrimitive.UNSAFE_PROCEED,
        },
    )
    return SimpleNamespace(
        decision=SimpleNamespace(state=state, action=action),
        primitive=label,
        events=dict(events or {}),
    )


def _segment(primitive, duration):
    return SimpleNamespace(
        primitive=DrivingPrimitive(primitive), duration_steps=duration
    )


def _record(steps, segments):
    return SimpleNamespace(
        solver=SolverKind.SAC,
        policy_mode=PolicyMode.DETERMINISTIC_ACTOR_MEAN,
        steps=tuple(steps),
        segments=tuple(segments),
        total_reward=10.0,
        termination_reason="timeout",
    )


def test_primitive_frequency_transitions_and_durations():
    record = _record(
        [
            _step("CruiseStraight", 0.2),
            _step("CruiseStraight", 0.2),
            _step("StopHold", 0.0, stop=True, distance=0.2),
        ],
        [_segment("CruiseStraight", 2), _segment("StopHold", 1)],
    )
    frequency = primitive_frequency([record])
    assert frequency["CruiseStraight"]["count"] == 2
    assert frequency["StopHold"]["rate"] == pytest.approx(1 / 3)
    transitions = primitive_transitions([record])
    assert transitions == ({
        "source": "CruiseStraight", "target": "StopHold", "count": 1
    },)
    durations = primitive_durations([record])
    assert durations["CruiseStraight"]["mean_steps"] == pytest.approx(2.0)


def test_policy_summary_reports_safety_and_response_metrics():
    record = _record(
        [
            _step("DecelerateStop", 0.1, stop=True, distance=0.4),
            _step("StopHold", 0.0, stop=True, distance=0.2,
                  events={"full_stop": True}),
            _step("YieldHold", 0.0, duck_active=True),
            _step("UnsafeProceed", 0.2, duck_active=True),
            _step("UnnecessaryBrake", 0.0),
        ],
        [
            _segment("DecelerateStop", 1),
            _segment("StopHold", 1),
            _segment("YieldHold", 1),
            _segment("UnsafeProceed", 1),
            _segment("UnnecessaryBrake", 1),
        ],
    )
    summary = summarize_policy([record])
    assert summary["stop_compliance_rate"] == pytest.approx(1.0)
    assert summary["pedestrian_yield_command_rate"] == pytest.approx(0.5)
    assert summary["unnecessary_brake_rate"] == pytest.approx(0.2)
    assert summary["unsafe_proceed_rate"] == pytest.approx(0.2)
    assert summary["first_stop_brake_distance"]["mean_m"] == pytest.approx(0.2)


def test_q_curve_class_is_not_counted_as_straight_when_curvature_is_absent():
    record = _record(
        [
            _step("CruiseStraight", 0.2, omega=0.1, curvature=None,
                  curvature_class="straight"),
            _step("CruiseCurveLeft", 0.2, omega=1.5, curvature=None,
                  curvature_class="curve_left"),
        ],
        [_segment("CruiseStraight", 1), _segment("CruiseCurveLeft", 1)],
    )
    summary = summarize_policy([record])
    assert summary["steering_response"]["straight_steps"] == 1
    assert summary["steering_response"]["mean_abs_omega"] == pytest.approx(0.1)


def test_influence_signatures_preserve_entangled_q_semantics():
    dimensions = (
        "d_bin", "tracking_error_bin", "speed_bin", "curvature_bin",
        "duck_threat_bin", "stop_distance_bin", "stop_satisfied_bin",
    )
    m8 = {"exact_characterization": {"one_bin": {}}}
    for index, dimension in enumerate(dimensions):
        m8["exact_characterization"]["one_bin"][dimension + "/supported"] = {
            "comparisons": 10,
            "flips": index,
            "flip_rate": index / 10,
        }
    q_signature = q_supported_influence_signature(m8)
    assert "lane_heading_entangled" in q_signature["raw"]
    assert sum(q_signature["normalized_l1"].values()) == pytest.approx(1.0)

    concepts = {
        "lane": 1.0, "heading": 2.0, "speed": 3.0,
        "road": 4.0, "stop": 5.0, "pedestrian": 6.0,
    }
    m9 = {"integrated_gradients": {
        "lane": {"neutral": {"concept_absolute": {
            "v_cmd": concepts, "omega_cmd": concepts,
        }}}
    }}
    sac_signature = sac_ig_influence_signature(m9)
    subset = comparable_influence_subset(q_signature, sac_signature)
    assert subset["concepts"] == ["speed", "road", "stop", "pedestrian"]
    assert "lane_heading_entangled" in subset["excluded"]
    assert sum(subset["sac_normalized"].values()) == pytest.approx(1.0)
