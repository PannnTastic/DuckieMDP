from pathlib import Path

import numpy as np
import yaml

from src.continuous_env import build_continuous_env


def test_real_environment_reset_and_step_fit_spaces():
    with Path("configs/sac_lane.yaml").open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    config["environment"]["max_steps"] = 24
    env = build_continuous_env(config, seed=7)
    try:
        observation = env.reset(7)
        assert env.observation_space.contains(observation)
        action = np.array([0.10, 0.0], dtype=np.float32)
        observation, reward, done, info = env.step(action)
        assert env.observation_space.contains(observation)
        assert np.isfinite(reward)
        assert np.allclose(info["command_v_omega"], [0.10, 0.0])
        assert info["termination_reason"] in {
            "in_progress",
            "timeout",
            "offroad",
            "other_collision",
            "duck_collision",
            "goal",
        }
    finally:
        env.close()