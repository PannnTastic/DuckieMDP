from src.explainability.metamorphic import (
    MetamorphicStatus,
    evaluate_relation,
    speed_level,
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


_Q_ACTIONS = {
    "fast_left": (0, 0.41, 1.5),
    "fast_straight": (1, 0.41, 0.0),
    "fast_right": (2, 0.41, -1.5),
    "slow_left": (3, 0.17, 1.5),
    "slow_straight": (4, 0.17, 0.0),
    "slow_right": (5, 0.17, -1.5),
    "brake": (6, 0.0, 0.0),
}


class FakePolicy:
    def __init__(self, solver, action_fn):
        self.solver = solver
        self.action_fn = action_fn
        self.calls = 0

    def decide(self, state):
        self.calls += 1
        value = self.action_fn(state)
        if self.solver == SolverKind.Q_LEARNING:
            action_id, v_cmd, omega_cmd = _Q_ACTIONS[value]
            action = CanonicalAction(
                self.solver, v_cmd, omega_cmd,
                action_id=action_id, action_name=value,
            )
            mode = PolicyMode.GREEDY
        else:
            action = CanonicalAction(self.solver, value[0], value[1])
            mode = PolicyMode.DETERMINISTIC_ACTOR_MEAN
        return PolicyDecision(self.solver, mode, state, action)


def test_invalid_pair_is_not_queried_and_is_not_applicable():
    policy = FakePolicy(SolverKind.SAC, lambda state: (0.2, 0.0))
    result = evaluate_relation(
        policy,
        SolverKind.SAC,
        _anchor(),
        "MR-STOP",
        {"stop_distance": 1.5},
        {"stop_distance": 3.5},
    )
    assert result.status == MetamorphicStatus.NOT_APPLICABLE
    assert "STOP_DISTANCE_OUT_OF_BOUNDS" in result.reason
    assert policy.calls == 0


def test_false_precondition_is_not_a_pass_and_is_not_queried():
    policy = FakePolicy(SolverKind.SAC, lambda state: (0.2, 0.0))
    result = evaluate_relation(
        policy,
        SolverKind.SAC,
        _anchor(),
        "MR-STOP",
        {"stop_distance": 0.2},
        {"stop_distance": 1.5},
    )
    assert result.status == MetamorphicStatus.NOT_APPLICABLE
    assert result.reason == "stop precondition false"
    assert policy.calls == 0


def test_q_stop_monotonicity_uses_ordinal_speed_level():
    policy = FakePolicy(
        SolverKind.Q_LEARNING,
        lambda state: "brake" if state.stop_distance <= 0.3 else "fast_straight",
    )
    result = evaluate_relation(
        policy,
        SolverKind.Q_LEARNING,
        _anchor(),
        "MR-STOP",
        {"stop_distance": 1.5},
        {"stop_distance": 0.2},
    )
    assert result.status == MetamorphicStatus.PASS
    assert result.measurements["source_speed_level"] == 2
    assert result.measurements["target_speed_level"] == 0
    assert speed_level(result.target_decision) == 0


def test_q_stop_speed_increase_is_a_failure():
    policy = FakePolicy(
        SolverKind.Q_LEARNING,
        lambda state: "fast_straight" if state.stop_distance <= 0.3 else "slow_straight",
    )
    result = evaluate_relation(
        policy,
        SolverKind.Q_LEARNING,
        _anchor(),
        "MR-STOP",
        {"stop_distance": 1.5},
        {"stop_distance": 0.2},
    )
    assert result.status == MetamorphicStatus.FAIL


def test_sac_curvature_respects_and_violates_frozen_tolerance():
    safe = FakePolicy(
        SolverKind.SAC,
        lambda state: (0.19 if abs(state.curvature) > 0.5 else 0.20, 0.0),
    )
    passed = evaluate_relation(
        safe, SolverKind.SAC, _anchor(), "MR-CURVATURE",
        {"curvature": 0.0}, {"curvature": 2.0},
    )
    assert passed.status == MetamorphicStatus.PASS

    unsafe = FakePolicy(
        SolverKind.SAC,
        lambda state: (0.22 if abs(state.curvature) > 0.5 else 0.20, 0.0),
    )
    failed = evaluate_relation(
        unsafe, SolverKind.SAC, _anchor(), "MR-CURVATURE",
        {"curvature": 0.0}, {"curvature": 2.0},
    )
    assert failed.status == MetamorphicStatus.FAIL
    assert failed.measurements["speed_delta"] > 0.01


def test_pedestrian_relation_rejects_unsafe_proceed_even_without_speed_increase():
    policy = FakePolicy(SolverKind.SAC, lambda state: (0.20, 0.0))
    result = evaluate_relation(
        policy,
        SolverKind.SAC,
        _anchor(),
        "MR-PEDESTRIAN",
        {
            "duck_longitudinal": 1.0,
            "duck_lateral": 0.5,
            "duck_v_longitudinal_relative": 0.0,
            "duck_v_lateral_relative": 0.0,
            "duck_active": False,
            "duck_crossing_available": True,
        },
        {
            "duck_longitudinal": 0.2,
            "duck_lateral": 0.1,
            "duck_active": True,
            "duck_crossing_available": False,
        },
    )
    assert result.status == MetamorphicStatus.FAIL
    assert result.target_primitive.primitive.value == "UnsafeProceed"


def test_q_lane_symmetry_requires_exact_mirrored_macro_action():
    policy = FakePolicy(
        SolverKind.Q_LEARNING,
        lambda state: "slow_left" if state.d > 0 else "slow_right",
    )
    result = evaluate_relation(
        policy,
        SolverKind.Q_LEARNING,
        _anchor(),
        "MR-LANE-SYMMETRY",
        {"d": 0.1, "phi": 0.2},
        {"d": -0.1, "phi": -0.2},
    )
    assert result.status == MetamorphicStatus.PASS
    assert result.measurements["expected_target_action"] == "slow_right"

