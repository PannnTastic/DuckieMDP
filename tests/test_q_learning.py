import pytest

from src.agents.q_learning import QLearningAgent, QLearningConfig


def test_timeout_transition_bootstraps():
    agent = QLearningAgent(QLearningConfig(gamma=0.5, alpha_lr=1.0), seed=1)
    state = (0, 0, 0, 0, 0, 0, 0)
    next_state = (1, 1, 1, 1, 1, 1, 1)
    agent.q[next_state] = 2.0
    agent.update(state, 0, 1.0, next_state, done=False)
    assert agent.q[state + (0,)] == pytest.approx(2.0)


def test_true_terminal_does_not_bootstrap():
    agent = QLearningAgent(QLearningConfig(gamma=0.5, alpha_lr=1.0), seed=1)
    state = (0, 0, 0, 0, 0, 0, 0)
    agent.update(state, 0, -5.0, None, done=True)
    assert agent.q[state + (0,)] == pytest.approx(-5.0)

