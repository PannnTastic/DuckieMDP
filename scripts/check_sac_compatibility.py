"""Smoke test dependency, wrapper, update SAC, dan checkpoint."""

import argparse
import logging
import tempfile
import warnings
from pathlib import Path

import gym
import numpy as np
import torch
import yaml

from src.agents.sac import SACAgent, SACConfig
from src.continuous_env import build_continuous_env


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/sac_lane.yaml"))
    parser.add_argument("--transitions", type=int, default=1000)
    args = parser.parse_args()

    with args.config.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    warnings.filterwarnings("ignore")
    logging.getLogger().setLevel(logging.ERROR)
    logging.getLogger("gym-duckietown").setLevel(logging.ERROR)
    logging.getLogger("gym_duckietown").setLevel(logging.ERROR)
    seed = int(config["seed"])
    device = str(config["training"].get("device", "cpu"))
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            "Config meminta CUDA, tetapi torch.cuda.is_available() bernilai False"
        )
    env = build_continuous_env(config, seed)
    env.action_space.seed(seed)
    agent = SACAgent(
        obs_dim=env.observation_space.shape[0],
        action_low=env.action_space.low,
        action_high=env.action_space.high,
        cfg=SACConfig(**config["sac"]),
        seed=seed,
        device=device,
    )

    observation = env.reset(seed)
    last_metrics = {}
    try:
        for step in range(args.transitions):
            if not env.observation_space.contains(observation):
                raise AssertionError(f"invalid observation at step {step}")
            action = env.action_space.sample()
            next_observation, reward, done, info = env.step(action)
            agent.replay.add(
                observation,
                action,
                reward,
                next_observation,
                bool(info["terminated"]),
            )
            last_metrics = agent.update() or last_metrics
            observation = next_observation
            if done:
                observation = env.reset(seed + step + 1)

        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "smoke.pt"
            agent.save(checkpoint)
            restored = SACAgent(
                env.observation_space.shape[0],
                env.action_space.low,
                env.action_space.high,
                SACConfig(**config["sac"]),
                seed + 99,
                device=device,
            )
            restored.load(checkpoint)
            action = restored.select_action(observation, deterministic=True)
            if not env.action_space.contains(action):
                raise AssertionError("loaded policy produced invalid action")
    finally:
        env.close()

    print(f"python_torch={torch.__version__}")
    print(f"device={agent.device}")
    print(f"gym={gym.__version__}")
    print(f"transitions={args.transitions}")
    print(f"updates={agent.updates}")
    print(f"last_metrics={last_metrics}")
    print("compatibility_smoke=passed")


if __name__ == "__main__":
    main()
