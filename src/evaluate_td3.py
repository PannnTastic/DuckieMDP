"""Deterministic, teacher-free evaluation for a TD3 checkpoint.

Reuses the SAC evaluation metric loop verbatim so both continuous policies are
scored by identical success criteria; only the agent construction differs.
"""

import argparse
import json
from pathlib import Path
from typing import Dict, Optional, Sequence

import torch
import yaml

from .agents.td3 import TD3Agent, TD3Config
from .continuous_env import build_continuous_env
from .evaluate_sac import (
    _quiet_third_party_logs,
    resolve_eval_params,
    run_evaluation,
)


def evaluate_policy(
    config_path: Path,
    checkpoint_path: Path,
    episodes: Optional[int] = None,
    seeds: Optional[Sequence[int]] = None,
) -> Dict[str, float]:
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    _quiet_third_party_logs()
    episodes, seeds = resolve_eval_params(config, episodes, seeds)
    device = str(config["training"].get("device", "cpu"))
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("Evaluasi meminta CUDA tetapi CUDA tidak tersedia")
    env = build_continuous_env(config, int(seeds[0]))
    agent = TD3Agent(
        env.observation_space.shape[0],
        env.action_space.low,
        env.action_space.high,
        TD3Config(**config["td3"]),
        seed=int(config["seed"]),
        device=device,
    )
    agent.load(checkpoint_path)
    return run_evaluation(env, agent, config, episodes, seeds)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = evaluate_policy(args.config, args.checkpoint, args.episodes)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
