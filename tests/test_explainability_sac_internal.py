from types import SimpleNamespace

import numpy as np
import pytest
import torch

from src.agents.sac import Critic, SquashedGaussianActor
from src.continuous_state import OBSERVATION_NAMES, ContinuousStateConfig
from src.explainability.explain_sac import (
    ACTION_OUTPUT_NAMES,
    CONCEPT_GROUPS,
    SACInternalDiagnostics,
    attribution_stability,
    compare_baselines,
    concept_aggregate,
    empirical_centroid,
    explanation_distance,
    local_boundary_search,
    neutral_baseline_state,
    normalized_influence,
)
from src.explainability.schema import (
    CanonicalAction,
    CanonicalState,
    PolicyDecision,
    PolicyMode,
    SolverKind,
)
from src.explainability.semantic_state import encode_canonical_for_sac


ACTION_LOW = np.asarray([0.0, -1.5], dtype=np.float32)
ACTION_HIGH = np.asarray([0.41, 1.5], dtype=np.float32)


def _state(**changes):
    values = {
        "d": 0.06,
        "phi": -0.12,
        "v": 0.19,
        "curvature": 0.0,
        "curvature_class": "straight",
        "stop_present": False,
        "stop_distance": None,
        "stop_satisfied": False,
        "stop_hold_progress": 0.0,
        "duck_present": False,
        "duck_threat": None,
        "duck_longitudinal": None,
        "duck_lateral": None,
        "duck_v_longitudinal_relative": None,
        "duck_v_lateral_relative": None,
        "duck_active": None,
        "duck_crossing_available": None,
        "source_representation": "m9_test",
    }
    values.update(changes)
    return CanonicalState(**values)


def _diagnostics(seed=9):
    torch.manual_seed(seed)
    actor = SquashedGaussianActor(15, ACTION_LOW, ACTION_HIGH, hidden=16)
    critic1 = Critic(15, 2, hidden=16)
    critic2 = Critic(15, 2, hidden=16)
    policy = SimpleNamespace(actor=actor)
    return SACInternalDiagnostics(
        policy,
        critic1,
        critic2,
        ACTION_LOW,
        ACTION_HIGH,
        ContinuousStateConfig(),
    )


class _FixedPolicy:
    def __init__(self, action=(0.20, 0.0)):
        self.action = action
        self.queries = []

    def decide(self, state):
        self.queries.append(state)
        action = CanonicalAction(
            solver=SolverKind.SAC,
            v_cmd=self.action[0],
            omega_cmd=self.action[1],
        )
        return PolicyDecision(
            solver=SolverKind.SAC,
            policy_mode=PolicyMode.DETERMINISTIC_ACTOR_MEAN,
            state=state,
            action=action,
        )


def test_neutral_baseline_uses_frozen_absence_sentinels():
    observation = encode_canonical_for_sac(neutral_baseline_state())
    lookup = dict(zip(OBSERVATION_NAMES, observation))
    assert observation.shape == (15,)
    assert lookup["d"] == pytest.approx(0.0)
    assert lookup["phi"] == pytest.approx(0.0)
    assert lookup["v"] == pytest.approx(0.17 / 0.41)
    assert lookup["d_stop"] == pytest.approx(1.0)
    assert lookup["stop_present"] == pytest.approx(0.0)
    assert lookup["duck_present"] == pytest.approx(0.0)
    assert lookup["duck_longitudinal"] == pytest.approx(1.0)
    assert lookup["stop_hold_progress"] == pytest.approx(0.0)


def test_integrated_gradients_has_two_outputs_and_small_completeness_error():
    diagnostics = _diagnostics()
    baseline = encode_canonical_for_sac(neutral_baseline_state())
    result = diagnostics.integrated_gradients(
        _state(), baseline, "neutral", steps=128
    )
    assert set(result.attributions) == set(ACTION_OUTPUT_NAMES)
    assert tuple(result.feature_names) == tuple(OBSERVATION_NAMES)
    for output in ACTION_OUTPUT_NAMES:
        assert len(result.attributions[output]) == 15
        assert set(result.concept_absolute[output]) == set(CONCEPT_GROUPS)
        assert abs(result.completeness_residual[output]) < 1e-3


def test_concept_aggregation_and_baseline_comparison_are_explicit():
    values = np.arange(1.0, 16.0)
    signed, absolute = concept_aggregate(values)
    assert signed["lane"] == pytest.approx(1.0)
    assert signed["stop"] == pytest.approx(5.0 + 6.0 + 7.0 + 15.0)
    assert absolute["pedestrian"] == pytest.approx(sum(range(8, 15)))
    diagnostics = _diagnostics()
    neutral = encode_canonical_for_sac(neutral_baseline_state())
    state = _state()
    primary = diagnostics.integrated_gradients(state, neutral, "neutral", steps=32)
    alternative = diagnostics.integrated_gradients(
        state, empirical_centroid([neutral_baseline_state(), state]),
        "empirical_cruise_centroid", steps=32,
    )
    comparison = compare_baselines(primary, alternative)
    assert set(comparison) == set(ACTION_OUTPUT_NAMES)
    assert all("dominant_concept_stable" in row for row in comparison.values())


def test_critic_probes_label_reference_support_without_claiming_advantage():
    diagnostics = _diagnostics()
    probes = diagnostics.critic_probes(_state())
    assert len(probes) == 6
    actor = next(probe for probe in probes if probe.probe_name == "actor")
    assert actor.normalized_distance_to_actor == pytest.approx(0.0)
    assert actor.support_label == "ACTOR_ACTION"
    assert actor.delta_min_q_vs_actor_probe == pytest.approx(0.0)
    for probe in probes:
        assert np.isfinite(probe.q1)
        assert np.isfinite(probe.q2)
        assert np.isfinite(probe.actor_log_probability)
        if probe.probe_name != "actor":
            assert probe.support_label == "LOW_ACTOR_SUPPORT_NO_REPLAY_SNAPSHOT"
            assert "replay snapshot" in probe.caveat


def test_local_boundary_rejects_invalid_neighbor_before_policy_query():
    policy = _FixedPolicy()
    diagnostics = SimpleNamespace(action_low=ACTION_LOW, action_high=ACTION_HIGH)
    result = local_boundary_search(
        policy, diagnostics, _state(d=0.25), action_threshold=0.05
    )
    rejected = [point for point in result.points if point.decision_action is None]
    assert result.rejected_points >= 1
    assert any(
        "D_OUT_OF_BOUNDS" in point.synthetic.validation.reason_codes
        for point in rejected
    )
    assert all(abs(query.d) <= 0.25 for query in policy.queries)


def test_attribution_stability_separates_boundary_cases():
    diagnostics = _diagnostics()
    policy = _FixedPolicy()
    baseline = encode_canonical_for_sac(neutral_baseline_state())
    result = attribution_stability(
        diagnostics, policy, [_state()], baseline, steps=16
    )
    assert result["anchors"] == 1
    assert result["rows"]
    assert result["disposition"] in {
        "PASS", "FAIL_RETAIN_FOR_AUDIT", "NO_NON_BOUNDARY_ANCHORS"
    }


def test_normalized_influence_and_distance_are_well_defined_at_zero():
    zero = np.zeros(15)
    assert np.allclose(normalized_influence(zero), zero)
    assert explanation_distance(zero, zero) == pytest.approx(0.0)
    assert explanation_distance(zero, np.ones(15)) == pytest.approx(1.0)
