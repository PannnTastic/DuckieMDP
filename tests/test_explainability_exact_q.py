import numpy as np

from src.discretizer import Q_SHAPE, STATE_SHAPE, discretize
from src.explainability.explain_q import (
    analyze_one_bin_flips,
    collect_evaluation_reach_counts,
    enumerate_q_policy,
    verify_safety_properties,
)
from src.explainability.q_policy_adapter import QPolicyAdapter
from src.explainability.semantic_state import raw_state_from_canonical
from src.state import DuckThreat, RawState, TileType


def _policy(table=None):
    if table is None:
        table = np.zeros(Q_SHAPE, dtype=np.float32)
    return QPolicyAdapter(table)


def test_exact_enumeration_covers_9000_and_preserves_every_index():
    records = enumerate_q_policy(_policy())
    assert len(records) == 9000
    assert len({row.index for row in records}) == 9000
    assert sum(row.valid_manifold for row in records) == 7875
    assert all(
        discretize(raw_state_from_canonical(row.state)) == row.index
        for row in records
    )


def test_margin_q2_and_historical_support_are_explicit():
    table = np.zeros(Q_SHAPE, dtype=np.float32)
    table[..., 4] = 1.0
    table[..., 6] = 0.5
    reach = np.zeros(STATE_SHAPE, dtype=np.int64)
    target = (2, 2, 1, 0, 0, 0, 0)
    reach[target] = 3
    records = enumerate_q_policy(_policy(table), evaluation_reach_counts=reach)
    row = next(item for item in records if item.index == target)
    assert row.action.action_id == 4
    assert row.second_action_id == 6
    assert row.q_margin == 0.5
    assert row.supported
    assert row.support_basis == "evaluation_reach_count_historical_proxy"
    assert row.provenance_status == "reached_only"
    assert row.training_visit_count is None


def test_one_bin_analysis_is_exact_and_stratified():
    table = np.zeros(Q_SHAPE, dtype=np.float32)
    table[..., 1] = 1.0
    table[3:, ..., 2] = 2.0
    records = enumerate_q_policy(_policy(table))
    flips, summary = analyze_one_bin_flips(records)
    assert flips
    assert {flip.dimension for flip in flips} == {"d_bin"}
    assert summary["d_bin/representable"]["flips"] == len(flips)
    assert all(not flip.both_reachable for flip in flips)
    assert all(flip.provenance == "unsupported_policy_region" for flip in flips)


def test_safety_checker_reports_exact_strata_not_one_blended_rate():
    table = np.zeros(Q_SHAPE, dtype=np.float32)
    table[..., 1] = 1.0  # fast_straight everywhere
    results = verify_safety_properties(enumerate_q_policy(_policy(table)))
    duck = results["P-DUCK-CROSSING-NEAR-NO-FAST"]["breakdown"]
    stop = results["P-STOP-NEAR-UNSATISFIED-NO-FAST"]["breakdown"]
    assert duck["representable"]["applicable"] == 1800
    assert duck["valid_manifold"]["applicable"] == 1575
    assert duck["representable"]["violations"] == 1800
    assert stop["representable"]["applicable"] == 1125
    assert stop["valid_manifold"]["applicable"] == 1125
    assert stop["representable"]["violations"] == 1125


class _TwoStepEnv:
    def __init__(self):
        self.step_number = 0

    @staticmethod
    def _state(speed):
        return RawState(
            d=0.0,
            phi=0.0,
            v=speed,
            tile=TileType.STRAIGHT,
            d_stop=None,
            sigma_stop=False,
            duck=DuckThreat.NONE,
        )

    def reset(self, seed):
        self.step_number = 0
        return self._state(0.02)

    def step(self, action):
        self.step_number += 1
        done = self.step_number == 2
        return (
            self._state(0.10),
            0.0,
            done,
            {"termination_reason": "timeout" if done else "in_progress"},
        )


def test_reach_count_reconstruction_counts_decision_states_only():
    counts, manifest = collect_evaluation_reach_counts(
        _TwoStepEnv(), _policy(), episodes=2, seeds=(101,)
    )
    assert int(counts.sum()) == 4
    assert np.count_nonzero(counts) == 2
    assert manifest["decision_states_counted"] == 4
    assert manifest["termination_counts"] == {"timeout": 2}
    assert manifest["policy_mode"] == "greedy_teacher_free_lowest_id_tie_break"

