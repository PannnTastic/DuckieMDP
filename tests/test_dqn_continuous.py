from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from src.agents.dqn import DQNAgent, DQNConfig
from src.train_dqn import load_config


def test_dqn_config_inherits_canonical_continuous_lane_environment():
    config = load_config(Path("configs/dqn_continuous_state.yaml"))
    assert config["algorithm"] == "dqn"
    assert config["environment"]["frame_skip"] == 6
    assert config["continuous_state"]["curvature_samples"] == 33
    assert config["reward"]["duck_yield"] == 0.0


def test_dqn_uses_14d_observation_action_mask_and_checkpoint(tmp_path):
    cfg = DQNConfig(
        batch_size=8,
        replay_size=64,
        hidden_size=32,
        allowed_actions=(0, 1, 2, 3, 4, 5),
    )
    agent = DQNAgent(cfg, obs_dim=14, seed=7)
    observation = np.zeros(14, dtype=np.float32)
    assert agent.select_action(observation) in cfg.allowed_actions

    rng = np.random.RandomState(8)
    for _ in range(16):
        state = rng.uniform(-1, 1, size=14).astype(np.float32)
        next_state = rng.uniform(-1, 1, size=14).astype(np.float32)
        action = int(rng.choice(cfg.allowed_actions))
        agent.buffer.add(state, action, rng.randn(), next_state, False)
    loss = agent.train_step()
    assert np.isfinite(loss)

    checkpoint = tmp_path / "dqn.pt"
    before = agent.online(
        torch.as_tensor(observation).unsqueeze(0)
    ).detach().numpy()
    agent.save(checkpoint)
    restored = DQNAgent(cfg, obs_dim=14, seed=99)
    restored.load(checkpoint)
    after = restored.online(
        torch.as_tensor(observation).unsqueeze(0)
    ).detach().numpy()
    assert np.allclose(before, after)
