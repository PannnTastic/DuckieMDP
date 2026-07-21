from dataclasses import replace

from src.explainability.counterfactual import (
    ControllerSemantics,
    make_counterfactual,
    project_state_for_solver,
    state_anchor_id,
    validate_state,
)
from src.explainability.schema import CanonicalState, SolverKind


def _sac_anchor(**overrides):
    values = dict(
        d=0.02,
        phi=-0.04,
        v=0.20,
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
        source_representation="real_rollout_test",
    )
    values.update(overrides)
    return CanonicalState(**values)


def test_real_like_sac_anchor_is_valid_and_fingerprint_is_stable():
    anchor = _sac_anchor()
    result = validate_state(anchor, SolverKind.SAC)
    assert result.valid
    assert result.encoded_observation_valid is True
    assert state_anchor_id(anchor) == state_anchor_id(replace(anchor))


def test_stop_absence_and_satisfaction_dependencies_are_repaired():
    anchor = _sac_anchor(
        stop_present=True,
        stop_distance=0.2,
        stop_satisfied=True,
        stop_hold_progress=1.0,
    )
    record = make_counterfactual(
        anchor,
        SolverKind.SAC,
        "remove_stop",
        {"stop_present": False},
    )
    assert record.validation.valid
    assert record.state.stop_distance is None
    assert record.state.stop_satisfied is False
    assert record.state.stop_hold_progress == 0.0
    assert "cleared dependent stop fields" in record.repair_notes


def test_stop_hold_progress_projects_lossily_for_q_learning():
    anchor = _sac_anchor(stop_present=True, stop_distance=0.2)
    partial = make_counterfactual(
        anchor,
        SolverKind.Q_LEARNING,
        "partial_dwell",
        {"stop_hold_progress": 0.66},
    )
    complete = make_counterfactual(
        anchor,
        SolverKind.Q_LEARNING,
        "complete_dwell",
        {"stop_hold_progress": 1.0},
    )
    assert partial.validation.valid and complete.validation.valid
    assert partial.state.stop_hold_progress == 0.0
    assert partial.state.stop_satisfied is False
    assert complete.state.stop_hold_progress == 1.0
    assert complete.state.stop_satisfied is True


def test_metric_duck_is_projected_to_q_threat_categories():
    anchor = _sac_anchor(
        duck_present=True,
        duck_longitudinal=0.30,
        duck_lateral=0.10,
        duck_v_longitudinal_relative=0.0,
        duck_v_lateral_relative=0.0,
        duck_active=True,
        duck_crossing_available=False,
    )
    near = project_state_for_solver(anchor, SolverKind.Q_LEARNING)
    assert near.duck_threat == "crossing_near"
    assert near.duck_longitudinal is None
    assert validate_state(near, SolverKind.Q_LEARNING).valid

    far = make_counterfactual(
        anchor,
        SolverKind.Q_LEARNING,
        "duck_longitudinal",
        {"duck_longitudinal": 0.9},
    )
    assert far.validation.valid
    assert far.state.duck_threat == "crossing_far"

    outside = make_counterfactual(
        anchor,
        SolverKind.Q_LEARNING,
        "duck_lateral",
        {"duck_lateral": 0.9},
    )
    assert outside.validation.valid
    assert outside.state.duck_present is False
    assert outside.state.duck_threat == "none"


def test_active_duck_repairs_crossing_available_under_canonical_controller():
    anchor = _sac_anchor(
        duck_present=True,
        duck_longitudinal=0.5,
        duck_lateral=0.2,
        duck_v_longitudinal_relative=0.0,
        duck_v_lateral_relative=0.0,
        duck_active=False,
        duck_crossing_available=True,
    )
    record = make_counterfactual(
        anchor,
        SolverKind.SAC,
        "activate_duck",
        {"duck_active": True},
    )
    assert record.validation.valid
    assert record.state.duck_active is True
    assert record.state.duck_crossing_available is False


def test_validator_rejects_off_manifold_state_and_preserves_reason_codes():
    invalid = _sac_anchor(
        d=0.30,
        stop_present=False,
        stop_distance=0.2,
    )
    result = validate_state(invalid, SolverKind.SAC)
    assert not result.valid
    assert "D_OUT_OF_BOUNDS" in result.reason_codes
    assert "STOP_ABSENT_DISTANCE" in result.reason_codes


def test_q_projection_cannot_hide_invalid_physical_intervention():
    anchor = _sac_anchor()
    curvature = make_counterfactual(
        anchor,
        SolverKind.Q_LEARNING,
        "curvature_out_of_bounds",
        {"curvature": 9.0},
    )
    assert not curvature.validation.valid
    assert "CURVATURE_OUT_OF_BOUNDS" in curvature.validation.reason_codes

    lateral = make_counterfactual(
        _sac_anchor(duck_present=False),
        SolverKind.Q_LEARNING,
        "incomplete_duck_geometry",
        {"duck_lateral": 2.2},
    )
    assert not lateral.validation.valid



def test_unlimited_controller_can_allow_active_and_available_flags():
    state = _sac_anchor(
        duck_present=True,
        duck_longitudinal=0.4,
        duck_lateral=0.1,
        duck_v_longitudinal_relative=0.0,
        duck_v_lateral_relative=0.0,
        duck_active=True,
        duck_crossing_available=True,
    )
    strict = validate_state(state, SolverKind.SAC)
    allowed = validate_state(
        state,
        SolverKind.SAC,
        controller=ControllerSemantics(
            allow_active_and_crossing_available=True,
            label="unlimited_without_rearm",
        ),
    )
    assert "DUCK_PHASE_INCONSISTENT" in strict.reason_codes
    assert allowed.valid
