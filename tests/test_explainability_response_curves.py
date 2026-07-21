from pathlib import Path

from src.explainability.response_curves import (
    SweepSpec,
    run_response_curve,
)
from src.explainability.schema import (
    CanonicalAction,
    CanonicalState,
    PolicyDecision,
    PolicyMode,
    SolverKind,
)


def _anchor(**overrides):
    values = dict(
        d=0.0,
        phi=0.0,
        v=0.2,
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


class FakeSACPolicy:
    def __init__(self):
        self.calls = 0

    def decide(self, state):
        self.calls += 1
        omega = -0.8 if state.d > 0.05 else (0.8 if state.d < -0.05 else 0.0)
        action = CanonicalAction(SolverKind.SAC, v_cmd=0.2, omega_cmd=omega)
        return PolicyDecision(
            solver=SolverKind.SAC,
            policy_mode=PolicyMode.DETERMINISTIC_ACTOR_MEAN,
            state=state,
            action=action,
        )


class FakeQPolicy:
    def __init__(self):
        self.calls = 0

    def decide(self, state):
        self.calls += 1
        if state.stop_satisfied:
            action = CanonicalAction(
                SolverKind.Q_LEARNING,
                v_cmd=0.15,
                omega_cmd=0.0,
                action_id=4,
                action_name="slow_straight",
            )
        else:
            action = CanonicalAction(
                SolverKind.Q_LEARNING,
                v_cmd=0.0,
                omega_cmd=0.0,
                action_id=6,
                action_name="brake",
            )
        return PolicyDecision(
            solver=SolverKind.Q_LEARNING,
            policy_mode=PolicyMode.GREEDY,
            state=state,
            action=action,
        )


def test_response_curve_queries_only_valid_manifold_points():
    policy = FakeSACPolicy()
    curve = run_response_curve(
        policy,
        SolverKind.SAC,
        _anchor(),
        SweepSpec("d", (-0.1, 0.0, 0.1, 0.30)),
    )
    assert curve.valid_points == 3
    assert curve.rejected_points == 1
    assert policy.calls == 4  # anchor plus three accepted queries
    rejected = curve.points[-1]
    assert rejected.decision is None
    assert "D_OUT_OF_BOUNDS" in rejected.synthetic.validation.reason_codes


def test_response_curve_finds_nearest_action_flip_and_primitive_change():
    curve = run_response_curve(
        FakeSACPolicy(),
        SolverKind.SAC,
        _anchor(),
        SweepSpec("d", (-0.10, -0.04, 0.0, 0.04, 0.10)),
    )
    assert curve.minimal_action_counterfactual is not None
    assert curve.minimal_action_counterfactual.distance_from_anchor == 0.10
    assert curve.minimal_action_counterfactual.primitive in {
        "LaneCorrectLeft", "LaneCorrectRight"
    }
    assert curve.minimal_primitive_counterfactual is not None
    assert (
        curve.minimal_primitive_counterfactual.distance_from_anchor == 0.10
    )
    assert curve.minimal_primitive_counterfactual.primitive in {
        "LaneCorrectLeft", "LaneCorrectRight"
    }


def test_q_stop_hold_curve_exposes_binary_dwell_representation():
    anchor = _anchor(stop_present=True, stop_distance=0.2)
    curve = run_response_curve(
        FakeQPolicy(),
        SolverKind.Q_LEARNING,
        anchor,
        SweepSpec("stop_hold_progress", (0.0, 0.33, 0.66, 1.0)),
    )
    action_ids = [point.decision.action.action_id for point in curve.points]
    projected = [point.synthetic.state.stop_hold_progress for point in curve.points]
    assert projected == [0.0, 0.0, 0.0, 1.0]
    assert action_ids == [6, 6, 6, 4]
    assert curve.minimal_action_counterfactual.requested_value == 1.0


def test_response_curve_serializes_valid_and_rejected_audits(tmp_path: Path):
    curve = run_response_curve(
        FakeSACPolicy(),
        SolverKind.SAC,
        _anchor(),
        SweepSpec("d", (0.0, 0.30)),
    )
    json_path = tmp_path / "curve.json"
    csv_path = tmp_path / "curve.csv"
    curve.save_json(json_path)
    curve.save_csv(csv_path)
    text = json_path.read_text(encoding="utf-8")
    assert '"rejected_points": 1' in text
    assert "D_OUT_OF_BOUNDS" in text
    rows = csv_path.read_text(encoding="utf-8").splitlines()
    assert len(rows) == 3
    assert "rejection_codes" in rows[0]
