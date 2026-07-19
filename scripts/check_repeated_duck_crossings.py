"""Audit repeated Duckie crossings with an existing SAC checkpoint.

The canonical YAML is read but never modified. The audit overrides
``max_crossings_per_episode`` to zero and applies a one-meter re-arm hysteresis
in memory. Crossings remain gated by the ego-distance interval and ``ahead``.
"""

import argparse
import json
import logging
import warnings
from math import cos
from pathlib import Path

import numpy as np
import torch
import yaml

from src.agents.sac import SACAgent, SACConfig
from src.continuous_env import build_continuous_env


def audit(config_path: Path, checkpoint: Path, episodes: int):
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config["duck_controller"]["max_crossings_per_episode"] = 0
    config["duck_controller"]["repeat_rearm_distance"] = 1.0
    warnings.filterwarnings("ignore")
    logging.getLogger().setLevel(logging.WARNING)

    env = build_continuous_env(config, seed=20101)
    agent = SACAgent(
        env.observation_space.shape[0],
        env.action_space.low,
        env.action_space.high,
        SACConfig(**config["sac"]),
        seed=int(config["seed"]),
        device=str(config["training"].get("device", "cpu")),
    )
    agent.load(checkpoint)
    decision_dt = env.unwrapped.delta_time * int(
        config["environment"]["frame_skip"]
    )
    brake_threshold = float(config["evaluation"]["brake_command_threshold"])
    records = []

    try:
        for episode in range(episodes):
            seed = 20101 + episode
            observation = env.reset(seed)
            done = False
            decisions = active_steps = yield_steps = violations = 0
            progress = 0.0
            rising_edges = []
            previous_active = env.current_state.duck_active
            info = {"termination_reason": "in_progress"}

            while not done:
                action = agent.select_action(observation, deterministic=True)
                observation, _, done, info = env.step(action)
                decisions += 1
                active = env.current_state.duck_active
                active_steps += int(active)
                yield_steps += int(active and float(action[0]) < brake_threshold)
                progress += max(
                    0.0, env.current_state.v * cos(env.current_state.phi)
                ) * decision_dt
                violations += int(info["events"]["stop_violation"])
                if active and not previous_active:
                    rising_edges.append(
                        {
                            "decision": decisions,
                            "physics_step": int(env.unwrapped.step_count),
                            "ego_xz": [
                                float(env.unwrapped.cur_pos[0]),
                                float(env.unwrapped.cur_pos[2]),
                            ],
                            "duck_longitudinal": float(
                                env.current_state.duck_longitudinal
                            ),
                            "duck_lateral": float(env.current_state.duck_lateral),
                        }
                    )
                previous_active = active

            controller_count = int(env.mdp_env.duck_controller.crossings_started[0])
            records.append(
                {
                    "episode": episode,
                    "seed": seed,
                    "termination_reason": info["termination_reason"],
                    "decisions": decisions,
                    "forward_progress_m": progress,
                    "crossings_started": controller_count,
                    "observed_crossing_rising_edges": rising_edges,
                    "duck_active_steps": active_steps,
                    "duck_yield_step_rate": yield_steps / max(1, active_steps),
                    "stop_violations": violations,
                }
            )
    finally:
        env.close()

    return {
        "override": {
            "max_crossings_per_episode": 0,
            "repeat_rearm_distance": 1.0,
        },
        "episodes": episodes,
        "episodes_with_repeated_crossing": sum(
            record["crossings_started"] >= 2 for record in records
        ),
        "mean_crossings_per_episode": float(
            np.mean([record["crossings_started"] for record in records])
        ),
        "records": records,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = audit(args.config, args.checkpoint, args.episodes)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
