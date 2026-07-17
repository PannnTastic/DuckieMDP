import numpy as np

from src.transition_model import EmpiricalTransitionModel
from src.value_iteration import value_iteration


S0 = (0, 0, 0, 0, 0, 0, 0)
S1 = (1, 0, 0, 0, 0, 0, 0)


def test_empirical_probabilities_and_reward_means(tmp_path):
    model = EmpiricalTransitionModel()
    model.observe(S0, 0, 1.0, S1, False)
    model.observe(S0, 0, 3.0, S1, False)
    model.observe(S0, 0, -2.0, None, True)
    outcomes = model.outcomes(S0, 0)
    assert sum(item[0] for item in outcomes) == 1.0
    path = tmp_path / "model.npz"
    model.save(path)
    loaded = EmpiricalTransitionModel.load(path)
    assert loaded.counts == model.counts
    assert loaded.reward_sums == model.reward_sums


def test_value_iteration_prefers_observed_terminal_reward():
    model = EmpiricalTransitionModel()
    model.observe(S0, 0, 1.0, None, True)
    model.observe(S0, 1, 0.0, S0, False)
    q, report = value_iteration(model, [0, 1], gamma=0.9)
    assert q[S0 + (0,)] > q[S0 + (1,)]
    assert report["observed_state_action_pairs"] == 2
