import numpy as np
import pytest

from src.agents.sarsa import SarsaAgent, SarsaConfig


STATE = (0, 0, 0, 0, 0, 0, 0)
NEXT_STATE = (1, 1, 1, 1, 1, 1, 1)


def test_sarsa_bootstraps_selected_next_action_not_maximum():
    agent = SarsaAgent(SarsaConfig(gamma=0.5, alpha_lr=1.0), seed=1)
    agent.q[NEXT_STATE + (1,)] = 2.0
    agent.q[NEXT_STATE + (2,)] = 100.0

    agent.update(STATE, 0, 1.0, NEXT_STATE, next_action=1, done=False)

    assert agent.q[STATE + (0,)] == pytest.approx(2.0)


def test_sarsa_true_terminal_does_not_bootstrap():
    agent = SarsaAgent(SarsaConfig(gamma=0.5, alpha_lr=1.0), seed=1)
    agent.update(STATE, 0, -5.0, None, next_action=None, done=True)
    assert agent.q[STATE + (0,)] == pytest.approx(-5.0)


def test_sarsa_action_mask_applies_to_current_and_next_action():
    agent = SarsaAgent(SarsaConfig(allowed_actions=[0, 1, 2, 3, 4, 5]), seed=3)
    agent.q[STATE + (6,)] = 100.0
    assert agent.select_action(STATE, greedy=True) != 6

    with pytest.raises(ValueError, match="Next action 6 is masked"):
        agent.update(STATE, 0, 0.0, NEXT_STATE, next_action=6, done=False)


def test_timeout_bootstrap_selection_can_avoid_advancing_epsilon_clock():
    agent = SarsaAgent(SarsaConfig(epsilon_start=0.0, epsilon_end=0.0), seed=4)
    before = agent.steps
    action = agent.select_action(NEXT_STATE, advance_step=False)
    assert action in agent.allowed_actions
    assert agent.steps == before
    assert np.isfinite(agent.epsilon)
