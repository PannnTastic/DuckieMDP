"""Optional M10 DQN ablation on the same 14-D privileged observation.

This keeps state/reward/dynamics equal to SAC while replacing continuous
actions with the seven historical differential-drive macro-actions.  It is
teacher-free and is not part of the M9 SAC curriculum.
"""

import argparse
import csv
import logging
import shutil
import warnings
from pathlib import Path

import numpy as np
import torch
import yaml

from .actions import ActionConfig, build_action_table
from .agents.dqn import DQNAgent, DQNConfig
from .continuous_env import build_continuous_env


def _deep_merge(base, override):
    result = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def load_config(path: Path):
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    base_name = config.pop("base_config", None)
    if base_name is None:
        return config
    base_path = (path.parent / base_name).resolve()
    base = yaml.safe_load(base_path.read_text(encoding="utf-8"))
    return _deep_merge(base, config)


def _epsilon(step: int, start: float, end: float, decay_steps: int) -> float:
    ratio = min(1.0, step / max(1, decay_steps))
    return float(start + ratio * (end - start))


def _write_rows(rows, path: Path) -> None:
    if not rows:
        return
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def train(config_path: Path) -> Path:
    warnings.filterwarnings("ignore")
    logging.getLogger().setLevel(logging.WARNING)
    config = load_config(config_path)
    seed = int(config["seed"])
    training = config["training"]
    device = str(training.get("device", "cpu"))
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("DQN config meminta CUDA tetapi CUDA tidak tersedia")

    output = Path(training["output_dir"])
    checkpoints = output / "checkpoints"
    checkpoints.mkdir(parents=True, exist_ok=True)
    shutil.copy2(config_path, output / "config_source.yaml")
    (output / "config_resolved.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
    )

    env = build_continuous_env(config, seed)
    env.action_space.seed(seed)
    dqn_cfg = DQNConfig(**config["dqn"])
    agent = DQNAgent(
        dqn_cfg,
        device=device,
        obs_dim=env.observation_space.shape[0],
        actions=7,
        seed=seed,
    )
    specs = build_action_table(ActionConfig(**config["actions"]))
    rng = np.random.RandomState(seed + 2)

    total_steps = int(training["total_steps"])
    warmup_steps = int(training["warmup_steps"])
    epsilon_start = float(training["epsilon_start"])
    epsilon_end = float(training["epsilon_end"])
    epsilon_decay_steps = int(training["epsilon_decay_steps"])
    checkpoint_interval = int(training["checkpoint_interval"])
    log_interval = int(training["log_interval"])
    rows = []
    episode = 0
    episode_return = 0.0
    episode_steps = 0
    observation = env.reset(seed)
    last_loss = np.nan

    print(
        f"algorithm=dqn stage={config.get('stage')} device={device} "
        f"obs_dim={env.observation_space.shape[0]} total_steps={total_steps}",
        flush=True,
    )
    try:
        for step in range(1, total_steps + 1):
            epsilon = _epsilon(
                step, epsilon_start, epsilon_end, epsilon_decay_steps
            )
            if step <= warmup_steps or rng.random_sample() < epsilon:
                action_id = int(rng.choice(agent.allowed_actions))
            else:
                action_id = agent.select_action(observation)
            spec = specs[action_id]
            command = np.array([spec.v, spec.omega], dtype=np.float32)
            next_observation, reward, done, info = env.step(command)
            agent.buffer.add(
                observation,
                action_id,
                reward,
                next_observation,
                bool(info["terminated"]),
            )
            observation = next_observation
            episode_return += float(reward)
            episode_steps += 1

            if step > warmup_steps:
                last_loss = agent.train_step()

            if done:
                rows.append(
                    {
                        "environment_step": step,
                        "physics_step": step
                        * int(config["environment"]["frame_skip"]),
                        "episode": episode,
                        "episode_return": episode_return,
                        "episode_steps": episode_steps,
                        "termination_reason": info["termination_reason"],
                        "terminated": int(info["terminated"]),
                        "truncated": int(info["truncated"]),
                        "epsilon": epsilon,
                        "loss": last_loss,
                    }
                )
                episode += 1
                episode_return = 0.0
                episode_steps = 0
                observation = env.reset(seed + episode)

            if step % log_interval == 0:
                recent = rows[-20:]
                mean_return = (
                    float(np.mean([row["episode_return"] for row in recent]))
                    if recent
                    else np.nan
                )
                print(
                    f"step={step} episode={episode} mean_return_20={mean_return:.3f} "
                    f"epsilon={epsilon:.4f} loss={last_loss:.4f} ",
                    flush=True,
                )
                _write_rows(rows, output / "training.csv")

            if step % checkpoint_interval == 0:
                agent.save(checkpoints / f"dqn_step_{step:09d}.pt")

        final = output / "dqn_final.pt"
        agent.save(final)
        _write_rows(rows, output / "training.csv")
        return final
    except KeyboardInterrupt:
        agent.save(output / "dqn_interrupted.pt")
        _write_rows(rows, output / "training_partial.csv")
        raise
    finally:
        env.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    final = train(args.config)
    print(f"final_checkpoint={final.resolve()}")


if __name__ == "__main__":
    main()
