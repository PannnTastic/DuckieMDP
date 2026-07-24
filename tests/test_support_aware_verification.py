"""Regression tests for support-aware C-EDDP metamorphic certification."""

import numpy as np

from src.discretizer import Q_SHAPE
from src.explainability.certified_primitives.schema import FullDecisionAnchor
from src.explainability.eddp.support import SupportOracle
from src.explainability.eddp.verification import verification_profile
from src.explainability.q_policy_adapter import QPolicyAdapter
from src.explainability.schema import SolverKind
from src.explainability.semantic_state import canonical_from_discrete_index


def _policy():
    return QPolicyAdapter(
        np.zeros(Q_SHAPE, dtype=np.float32),
        solver_kind=SolverKind.Q_LEARNING,
    )


def _anchor(policy, index, serial):
    state = canonical_from_discrete_index(index)
    decision = policy.decide(state)
    return FullDecisionAnchor(
        anchor_id="anchor-%d" % serial,
        solver=SolverKind.Q_LEARNING,
        seed=20101,
        episode_id="support-test",
        step_index=serial,
        state=state,
        selected_action=decision.action,
        action_prefix=(),
        config_path="test.yaml",
        checkpoint_path="test.npy",
        policy_mode="greedy_teacher_free",
    )


def test_pedestrian_relation_is_claimable_only_when_both_cells_supported():
    policy = _policy()
    source_index = (2, 2, 1, 0, 0, 0, 0)
    target_index = (2, 2, 1, 0, 0, 0, 1)
    anchors = tuple(
        _anchor(policy, source_index, index)
        for index in range(3)
    ) + tuple(
        _anchor(policy, target_index, index + 3)
        for index in range(3)
    )
    oracle = SupportOracle.from_anchors(
        anchors, {"q_learning": policy},
        tabular_support_threshold=3,
    )

    profile = verification_profile(
        policy,
        canonical_from_discrete_index(source_index),
        oracle,
    )

    assert profile["pedestrian_applicable"]
    assert profile["pedestrian_eligible"]
    assert profile["pedestrian_status"] == "PASS"
    assert profile["pedestrian_pair_stratum"] == "both_supported"


def test_unsupported_counterfactual_target_abstains_instead_of_failing():
    policy = _policy()
    source_index = (2, 2, 1, 0, 0, 0, 0)
    anchors = tuple(
        _anchor(policy, source_index, index)
        for index in range(3)
    )
    oracle = SupportOracle.from_anchors(
        anchors, {"q_learning": policy},
        tabular_support_threshold=3,
    )

    profile = verification_profile(
        policy,
        canonical_from_discrete_index(source_index),
        oracle,
    )

    assert profile["pedestrian_applicable"]
    assert not profile["pedestrian_eligible"]
    assert profile["pedestrian_abstain"]
    assert not profile["pedestrian_pass"]
    assert not profile["pedestrian_fail"]
    assert profile["pedestrian_raw_pass"]
    assert (
        profile["pedestrian_pair_stratum"]
        == "reachable_source_interventional_target"
    )


def test_missing_support_oracle_never_creates_a_claimable_relation():
    policy = _policy()
    state = canonical_from_discrete_index((2, 2, 1, 0, 0, 0, 0))

    profile = verification_profile(policy, state)

    assert profile["pedestrian_applicable"]
    assert profile["pedestrian_status"] == "ABSTAIN"
    assert not profile["pedestrian_eligible"]
