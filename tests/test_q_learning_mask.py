import numpy as np

from src.agents.q_learning import QLearningAgent, QLearningConfig


def test_masked_brake_is_never_selected_or_bootstrapped():
    agent = QLearningAgent(QLearningConfig(allowed_actions=[0, 1, 2, 3, 4, 5]), seed=3)
    state = (0, 0, 0, 0, 0, 0, 0)
    next_state = (1, 1, 1, 1, 1, 1, 1)
    agent.q[state + (6,)] = 100.0
    agent.q[next_state + (6,)] = 100.0
    agent.q[next_state + (2,)] = 2.0
    assert agent.select_action(state, greedy=True) != 6
    agent.update(state, 0, 1.0, next_state, done=False)
    expected = 0.1 * (1.0 + 0.99 * 2.0)
    assert np.isclose(agent.q[state + (0,)], expected)

