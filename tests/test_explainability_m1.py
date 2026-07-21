import json

import numpy as np
import pytest

from src.actions import ActionConfig
from src.continuous_state import ContinuousState, ContinuousStateConfig
from src.discretizer import Q_SHAPE, discretize
from src.explainability.q_policy_adapter import QPolicyAdapter
from src.explainability.schema import (
    CanonicalState,
    PolicyMode,
    SolverKind,
    to_json,
)
from src.explainability.semantic_state import (
    canonical_from_continuous_state,
    canonical_from_discrete_index,
    raw_state_from_canonical,
)


def _sac_state(**overrides):
    values = dict(
        d=0.02,
        phi=-0.03,
        v=0.14,
        kappa=0.0,
        stop_present=False,
        d_stop=None,
        sigma_stop=False,
        duck_present=False,
        duck_longitudinal=0.0,
        duck_lateral=0.0,
        duck_v_longitudinal_relative=0.0,
        duck_v_lateral_relative=0.0,
        duck_active=False,
        duck_crossing_available=False,
        stop_hold_progress=0.0,
    )
    values.update(overrides)
    return ContinuousState(**values)


def _q_state(**overrides):
    values = dict(
        d=0.02,
        phi=-0.03,
        v=0.14,
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


def test_schema_serializes_enums_and_can_represent_invalid_q_cells():
    # stop-absent + sigma=true is not physically valid, but it is one of the
    # representable Q-table cells and must survive until the M6 validator.
    state = canonical_from_discrete_index((0, 0, 0, 0, 0, 1, 0))
    table = np.zeros(Q_SHAPE, dtype=np.float32)
    decision = QPolicyAdapter(table).decide_index(state.source_index, state)
    payload = json.loads(to_json(decision))
    assert payload["solver"] == "q_learning"
    assert payload["policy_mode"] == "greedy"
    assert payload["state"]["stop_satisfied"] is True


def test_q_adapter_matches_discretization_and_does_not_mutate_source():
    source = np.zeros(Q_SHAPE, dtype=np.float32)
    state = _q_state()
    index = discretize(raw_state_from_canonical(state))
    source[index + (3,)] = 2.0
    source[index + (4,)] = 3.5
    original = source.copy()

    adapter = QPolicyAdapter(
        source,
        allowed_actions=(0, 1, 2, 3, 4, 5),
        action_config=ActionConfig(v_slow=0.12),
    )
    decision = adapter.decide(state)
    assert decision.solver == SolverKind.Q_LEARNING
    assert decision.policy_mode == PolicyMode.GREEDY
    assert decision.action.action_id == 4
    assert decision.action.action_name == "slow_straight"
    assert decision.action.v_cmd == pytest.approx(0.12)
    assert decision.diagnostics["q_margin"] == pytest.approx(1.5)
    assert decision.metadata["teacher_active"] is False
    assert np.array_equal(source, original)
    assert not adapter.q_table.flags.writeable


def test_q_adapter_tie_break_is_reproducible_and_explicit():
    table = np.zeros(Q_SHAPE, dtype=np.float32)
    state = _q_state()
    index = discretize(raw_state_from_canonical(state))
    table[index + (2,)] = 4.0
    table[index + (5,)] = 4.0
    decision = QPolicyAdapter(table).decide(state)
    assert decision.action.action_id == 2
    assert decision.diagnostics["greedy_ties"] == (2, 5)
    assert decision.diagnostics["q_margin"] == 0.0


def test_canonical_continuous_state_preserves_metric_semantics():
    canonical = canonical_from_continuous_state(
        _sac_state(
            kappa=-1.2,
            stop_present=True,
            d_stop=0.4,
            duck_present=True,
            duck_longitudinal=0.5,
            duck_lateral=-0.2,
            duck_active=True,
            duck_crossing_available=True,
        )
    )
    assert canonical.curvature == pytest.approx(-1.2)
    assert canonical.curvature_class == "curve_right"
    assert canonical.stop_distance == pytest.approx(0.4)
    assert canonical.duck_longitudinal == pytest.approx(0.5)
    assert canonical.duck_active is True


def test_sac_adapter_matches_deterministic_agent_without_replay_allocation(tmp_path):
    torch = pytest.importorskip("torch")
    from src.agents.sac import SACAgent, SACConfig
    from src.explainability.sac_policy_adapter import SACPolicyAdapter
    from src.explainability.semantic_state import encode_canonical_for_sac

    cfg = SACConfig(batch_size=8, replay_capacity=64, hidden_size=32)
    agent = SACAgent(
        obs_dim=15,
        action_low=np.array([0.0, -1.5], dtype=np.float32),
        action_high=np.array([0.41, 1.5], dtype=np.float32),
        cfg=cfg,
        seed=7,
    )
    checkpoint = tmp_path / "sac.pt"
    agent.save(checkpoint)
    adapter = SACPolicyAdapter.from_checkpoint(checkpoint)

    canonical = canonical_from_continuous_state(_sac_state())
    observation = encode_canonical_for_sac(canonical, ContinuousStateConfig())
    expected = agent.select_action(observation, deterministic=True)
    decision = adapter.decide(canonical)
    actual = np.array([decision.action.v_cmd, decision.action.omega_cmd])

    assert np.allclose(actual, expected, atol=1e-7, rtol=0.0)
    assert decision.policy_mode == PolicyMode.DETERMINISTIC_ACTOR_MEAN
    assert decision.metadata["actor_sampling"] is False
    assert decision.metadata["teacher_active"] is False
    assert not hasattr(adapter, "replay")
    assert len(decision.diagnostics["observation_names"]) == 15


def test_sac_adapter_14_to_15_expansion_is_explicit_and_preserves_actor(tmp_path):
    pytest.importorskip("torch")
    from src.agents.sac import SACAgent, SACConfig
    from src.explainability.sac_policy_adapter import SACPolicyAdapter
    from src.explainability.semantic_state import encode_canonical_for_sac

    cfg = SACConfig(batch_size=8, replay_capacity=64, hidden_size=32)
    old = SACAgent(
        obs_dim=14,
        action_low=np.array([0.0, -1.5], dtype=np.float32),
        action_high=np.array([0.41, 1.5], dtype=np.float32),
        cfg=cfg,
        seed=11,
    )
    checkpoint = tmp_path / "sac_14d.pt"
    old.save(checkpoint)
    with pytest.raises(ValueError, match="dimension mismatch"):
        SACPolicyAdapter.from_checkpoint(checkpoint)

    adapter = SACPolicyAdapter.from_checkpoint(
        checkpoint,
        allow_observation_expansion=True,
    )
    canonical = canonical_from_continuous_state(_sac_state())
    observation = encode_canonical_for_sac(canonical, ContinuousStateConfig())
    expected = old.select_action(observation[:14], deterministic=True)
    decision = adapter.decide(canonical)
    actual = np.array([decision.action.v_cmd, decision.action.omega_cmd])
    assert np.allclose(actual, expected, atol=1e-7, rtol=0.0)
    assert decision.metadata["observation_expanded"] is True
