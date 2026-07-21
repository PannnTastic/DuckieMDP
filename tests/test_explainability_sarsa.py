import numpy as np

from src.actions import ActionConfig
from src.discretizer import Q_SHAPE
from src.explainability.action_outcomes import q_action
from src.explainability.counterfactual import validate_state
from src.explainability.sarsa_policy_adapter import SarsaPolicyAdapter
from src.explainability.schema import PolicyMode, SolverKind
from src.explainability.sarsa_explanation_report import compare_tabular_policies
from src.explainability.semantic_state import canonical_from_discrete_index


def test_sarsa_adapter_preserves_solver_identity_and_greedy_contract():
    table = np.zeros(Q_SHAPE, dtype=np.float32)
    index = (2, 2, 1, 0, 0, 0, 0)
    table[index + (3,)] = 1.0
    table[index + (4,)] = 2.5
    policy = SarsaPolicyAdapter(
        table,
        action_config=ActionConfig(v_slow=0.17),
        solver_kind=SolverKind.SARSA,
    )

    decision = policy.decide_index(index)
    assert decision.solver == SolverKind.SARSA
    assert decision.action.solver == SolverKind.SARSA
    assert decision.policy_mode == PolicyMode.GREEDY
    assert decision.action.action_id == 4
    assert decision.action.action_name == "slow_straight"
    assert decision.diagnostics["q_margin"] == 1.5
    assert decision.metadata["teacher_active"] is False


def test_sarsa_action_and_manifold_use_tabular_contract():
    policy = SarsaPolicyAdapter(
        np.zeros(Q_SHAPE, dtype=np.float32),
        solver_kind=SolverKind.SARSA,
    )
    action = q_action(policy, 6)
    state = canonical_from_discrete_index((2, 2, 1, 0, 0, 0, 0))

    assert action.solver == SolverKind.SARSA
    assert action.action_name == "brake"
    assert validate_state(state, SolverKind.SARSA).valid


def test_sarsa_checkpoint_is_expected_full_task_table():
    path = "artifacts/ablation/full_task/sarsa_teacher/q_table_best.npy"
    policy = SarsaPolicyAdapter.from_checkpoint(path)
    assert policy.q_table.shape == Q_SHAPE
    assert policy.checkpoint_hash == (
        "0266ad6f6fdae71bf2dfb7c7121f66e038d16f75e214a931f8c9d50bc6ad3313"
    )
    assert policy.solver_kind == SolverKind.SARSA


def test_tabular_policy_comparison_distinguishes_values_from_greedy_actions():
    q_table = np.zeros(Q_SHAPE, dtype=np.float32)
    sarsa_table = np.zeros(Q_SHAPE, dtype=np.float32)
    index = (2, 2, 1, 0, 0, 0, 0)
    q_table[index + (4,)] = 2.0
    sarsa_table[index + (4,)] = 7.0

    comparison = compare_tabular_policies(q_table, sarsa_table)
    assert comparison["tables_numerically_equal"] is False
    assert comparison["greedy_action_disagreement_states"] == 0
    assert comparison["greedy_action_agreement_rate"] == 1.0


def test_tabular_policy_comparison_rejects_wrong_shape():
    wrong = np.zeros((2, 7), dtype=np.float32)
    valid = np.zeros(Q_SHAPE, dtype=np.float32)
    try:
        compare_tabular_policies(wrong, valid)
    except ValueError as error:
        assert "shape" in str(error)
    else:
        raise AssertionError("wrong table shape must fail closed")

