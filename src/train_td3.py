"""Teacher-free TD3 training mirroring the SAC training loop.

The behaviour policy is the deterministic TD3 actor plus Gaussian
exploration noise (or random actions during warm-up). No scripted teacher
chooses actions. Logging, checkpointing, and CSV schema follow
``train_sac.py`` so downstream tooling can read both interchangeably; the
only schema difference is that TD3 has no entropy temperature, so the
``alpha`` column is always NaN.
"""

import argparse
from math import cos, hypot
from pathlib import Path
import shutil
from typing import Optional

import numpy as np
import yaml

from .agents.td3 import TD3Agent, TD3Config
from .continuous_env import build_continuous_env
from .train_sac import (
    REWARD_COMPONENTS,
    _empty_reward_totals,
    _init_wandb,
    _log_checkpoint_artifact,
    _quiet_third_party_logs,
    _require_device,
    _write_rows,
)


def train(
    config_path: Path,
    steps_override: Optional[int] = None,
    output_override: Optional[Path] = None,
    random_steps_override: Optional[int] = None,
    wandb_mode_override: Optional[str] = None,
    initial_checkpoint_override: Optional[Path] = None,
) -> Path:
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if wandb_mode_override is not None:
        config.setdefault("wandb", {})["mode"] = wandb_mode_override
    _quiet_third_party_logs()

    seed = int(config["seed"])
    training = config["training"]
    device = str(training.get("device", "cpu"))
    _require_device(device)
    output = output_override or Path(training["output_dir"])
    checkpoints = output / "checkpoints"
    checkpoints.mkdir(parents=True, exist_ok=True)
    shutil.copy2(config_path, output / "config.yaml")
    wandb_run = _init_wandb(config, output)

    env = build_continuous_env(config, seed)
    env.action_space.seed(seed)
    agent = TD3Agent(
        obs_dim=env.observation_space.shape[0],
        action_low=env.action_space.low,
        action_high=env.action_space.high,
        cfg=TD3Config(**config["td3"]),
        seed=seed,
        device=device,
    )
    initial = (
        str(initial_checkpoint_override)
        if initial_checkpoint_override is not None
        else training.get("initial_checkpoint")
    )
    # Resume/warm-start memuat actor, critics, dan optimizer. Replay sengaja
    # mulai kosong karena distribusi behaviour policy berubah.
    if initial and (steps_override is None or initial_checkpoint_override):
        initial_path = Path(initial)
        if not initial_path.exists():
            raise FileNotFoundError(
                f"Initial checkpoint TD3 tidak ditemukan: {initial_path}"
            )
        agent.load(initial_path)
        print(f"initial_checkpoint={initial_path}", flush=True)

    total_steps = int(steps_override or training["total_steps"])
    configured_random_steps = (
        int(random_steps_override)
        if random_steps_override is not None
        else int(training["random_steps"])
    )
    random_steps = min(configured_random_steps, total_steps)
    gradient_steps = int(training.get("gradient_steps", 1))
    checkpoint_interval = int(training["checkpoint_interval"])
    log_interval = min(int(training["log_interval"]), total_steps)
    rows = []
    episode = 0
    episode_return = 0.0
    episode_steps = 0
    episode_progress = 0.0
    episode_brakes = 0
    episode_v_command = 0.0
    episode_abs_omega_command = 0.0
    episode_abs_omega_straight = 0.0
    episode_straight_steps = 0
    episode_max_stop_hold = 0
    episode_full_stops = 0
    episode_stop_violations = 0
    episode_duck_crossing_steps = 0
    episode_duck_yield_steps = 0
    episode_false_stop_steps = 0
    episode_min_duck_distance = np.inf
    episode_reward_totals = _empty_reward_totals()
    observation = env.reset(seed)
    episode_spawn_route_alignment = env.mdp_env.last_spawn_route_alignment
    last_metrics = {}
    decision_dt = env.unwrapped.delta_time * int(
        config["environment"]["frame_skip"]
    )
    evaluation = config.get("evaluation", {})
    brake_threshold = float(evaluation.get("brake_command_threshold", 0.04))

    print(
        f"stage={config.get('stage')} device={device} total_steps={total_steps} "
        f"random_steps={random_steps} frame_skip={config['environment']['frame_skip']}",
        flush=True,
    )
    try:
        for step in range(1, total_steps + 1):
            if step <= random_steps:
                action = env.action_space.sample()
            else:
                action = agent.select_action(observation, deterministic=False)

            state_before = env.current_state
            is_brake = float(action[0]) < brake_threshold
            unmet_stop = (
                state_before.stop_present
                and state_before.d_stop is not None
                and state_before.d_stop
                <= env.mdp_env.reward_cfg.stop_exemption_distance
                and not state_before.sigma_stop
            )
            crossing_before = state_before.duck_active
            is_straight = (
                abs(state_before.kappa)
                <= env.mdp_env.reward_cfg.straight_curvature_threshold
            )
            next_observation, reward, done, info = env.step(action)
            agent.replay.add(
                observation,
                action,
                reward,
                next_observation,
                bool(info["terminated"]),
            )
            observation = next_observation
            episode_return += float(reward)
            reward_terms = info["reward_terms"]
            for name in REWARD_COMPONENTS:
                episode_reward_totals[name] += float(reward_terms[name])
            episode_steps += 1
            episode_brakes += int(is_brake)
            episode_v_command += float(action[0])
            episode_abs_omega_command += abs(float(action[1]))
            episode_abs_omega_straight += int(is_straight) * abs(float(action[1]))
            episode_straight_steps += int(is_straight)
            episode_max_stop_hold = max(
                episode_max_stop_hold,
                int(info.get("stop_hold_steps", 0)),
            )
            episode_progress += max(
                0.0,
                env.current_state.v * cos(env.current_state.phi),
            ) * decision_dt
            events = info["events"]
            episode_full_stops += int(events["full_stop"])
            episode_stop_violations += int(events["stop_violation"])
            episode_duck_crossing_steps += int(crossing_before)
            episode_duck_yield_steps += int(crossing_before and is_brake)
            episode_false_stop_steps += int(
                is_brake and not crossing_before and not unmet_stop
            )
            if env.current_state.duck_present:
                episode_min_duck_distance = min(
                    episode_min_duck_distance,
                    hypot(
                        env.current_state.duck_longitudinal,
                        env.current_state.duck_lateral,
                    ),
                )

            if step > random_steps:
                for _ in range(gradient_steps):
                    metrics = agent.update()
                    if metrics:
                        last_metrics = metrics

            if done:
                row = {
                    "environment_step": step,
                    "physics_step": step
                    * int(config["environment"]["frame_skip"]),
                    "episode": episode,
                    "episode_return": episode_return,
                    "episode_steps": episode_steps,
                    "episode_forward_progress_m": episode_progress,
                    "episode_brake_ratio": episode_brakes / max(1, episode_steps),
                    "episode_mean_v_command": episode_v_command
                    / max(1, episode_steps),
                    "episode_mean_abs_omega_command": episode_abs_omega_command
                    / max(1, episode_steps),
                    "episode_mean_abs_omega_straight": episode_abs_omega_straight
                    / max(1, episode_straight_steps),
                    "episode_max_stop_hold_decisions": episode_max_stop_hold,
                    "spawn_route_alignment": episode_spawn_route_alignment,
                    "full_stop_count": episode_full_stops,
                    "stop_violation_count": episode_stop_violations,
                    "duck_crossing_steps": episode_duck_crossing_steps,
                    "duck_yield_ratio": episode_duck_yield_steps
                    / max(1, episode_duck_crossing_steps),
                    "false_stop_ratio": episode_false_stop_steps
                    / max(1, episode_steps),
                    "minimum_duck_distance_m": (
                        episode_min_duck_distance
                        if np.isfinite(episode_min_duck_distance)
                        else np.nan
                    ),
                    "termination_reason": info["termination_reason"],
                    "terminated": int(info["terminated"]),
                    "truncated": int(info["truncated"]),
                    "alpha": np.nan,
                    "critic_loss": last_metrics.get("critic_loss", np.nan),
                    "actor_loss": last_metrics.get("actor_loss", np.nan),
                }
                for name in REWARD_COMPONENTS:
                    row[f"reward_{name}_sum"] = episode_reward_totals[name]
                    row[f"reward_{name}_mean"] = (
                        episode_reward_totals[name] / max(1, episode_steps)
                    )
                rows.append(row)
                if wandb_run is not None:
                    row = rows[-1]
                    reason = info["termination_reason"]
                    wandb_metrics = {
                        "episode/index": episode,
                        "episode/return": episode_return,
                        "episode/length": episode_steps,
                        "episode/forward_progress_m": episode_progress,
                        "episode/brake_ratio": row["episode_brake_ratio"],
                        "episode/mean_v_command": row["episode_mean_v_command"],
                        "episode/mean_abs_omega_command": row[
                            "episode_mean_abs_omega_command"
                        ],
                        "episode/mean_abs_omega_straight": row[
                            "episode_mean_abs_omega_straight"
                        ],
                        "spawn/route_alignment": episode_spawn_route_alignment,
                        "task/full_stop_count": episode_full_stops,
                        "task/max_stop_hold_decisions": episode_max_stop_hold,
                        "task/stop_violation_count": episode_stop_violations,
                        "task/duck_crossing_steps": episode_duck_crossing_steps,
                        "task/duck_yield_ratio": row["duck_yield_ratio"],
                        "task/false_stop_ratio": row["false_stop_ratio"],
                        "task/minimum_duck_distance_m": row[
                            "minimum_duck_distance_m"
                        ],
                        "termination/timeout": int(reason == "timeout"),
                        "termination/offroad": int(reason == "offroad"),
                        "termination/duck_collision": int(
                            reason == "duck_collision"
                        ),
                        "termination/other_collision": int(
                            reason == "other_collision"
                        ),
                    }
                    for name in REWARD_COMPONENTS:
                        wandb_metrics[f"reward/episode_{name}"] = row[
                            f"reward_{name}_sum"
                        ]
                        wandb_metrics[f"reward/mean_per_decision_{name}"] = row[
                            f"reward_{name}_mean"
                        ]
                    wandb_run.log(wandb_metrics, step=step)
                episode += 1
                episode_return = 0.0
                episode_steps = 0
                episode_progress = 0.0
                episode_brakes = 0
                episode_v_command = 0.0
                episode_abs_omega_command = 0.0
                episode_abs_omega_straight = 0.0
                episode_straight_steps = 0
                episode_max_stop_hold = 0
                episode_full_stops = 0
                episode_stop_violations = 0
                episode_duck_crossing_steps = 0
                episode_duck_yield_steps = 0
                episode_false_stop_steps = 0
                episode_min_duck_distance = np.inf
                episode_reward_totals = _empty_reward_totals()
                observation = env.reset(seed + episode)
                episode_spawn_route_alignment = env.mdp_env.last_spawn_route_alignment

            if step % log_interval == 0:
                recent = rows[-20:]
                mean_return = (
                    float(np.mean([row["episode_return"] for row in recent]))
                    if recent
                    else np.nan
                )
                print(
                    f"step={step} episode={episode} mean_return_20={mean_return:.3f} "
                    f"critic_loss={last_metrics.get('critic_loss', np.nan):.4f} "
                    f"replay={len(agent.replay)}",
                    flush=True,
                )
                if wandb_run is not None:
                    wandb_run.log(
                        {
                            "train/episode": episode,
                            "train/return_ma20": mean_return,
                            "train/critic_loss": last_metrics.get(
                                "critic_loss", np.nan
                            ),
                            "train/actor_loss": last_metrics.get(
                                "actor_loss", np.nan
                            ),
                            "train/mean_q": last_metrics.get("mean_q", np.nan),
                            "train/replay_size": len(agent.replay),
                            "train/physics_step": step
                            * int(config["environment"]["frame_skip"]),
                        },
                        step=step,
                    )
                _write_rows(rows, output / "training.csv")

            if step % checkpoint_interval == 0:
                checkpoint = checkpoints / f"td3_step_{step:09d}.pt"
                agent.save(checkpoint)
                if bool(config.get("wandb", {}).get("log_checkpoints", True)):
                    _log_checkpoint_artifact(
                        wandb_run,
                        checkpoint,
                        str(config.get("stage", "unknown")),
                        step,
                        aliases=(f"step-{step}",),
                    )

        final_checkpoint = output / "td3_final.pt"
        agent.save(final_checkpoint)
        _write_rows(rows, output / "training.csv")
        if wandb_run is not None:
            wandb_run.summary["final_checkpoint"] = str(final_checkpoint)
            wandb_run.summary["total_environment_steps"] = total_steps
            _log_checkpoint_artifact(
                wandb_run,
                final_checkpoint,
                str(config.get("stage", "unknown")),
                total_steps,
                aliases=("final",),
            )
        return final_checkpoint
    except KeyboardInterrupt:
        agent.save(output / "td3_interrupted.pt")
        _write_rows(rows, output / "training_partial.csv")
        raise
    finally:
        env.close()
        if wandb_run is not None:
            wandb_run.finish()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--steps",
        type=int,
        default=None,
        help="Override jumlah step untuk smoke test; config kanonis tidak berubah.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Override output untuk smoke test.",
    )
    parser.add_argument(
        "--random-steps",
        type=int,
        default=None,
        help="Override warm-up random untuk smoke test.",
    )
    parser.add_argument(
        "--wandb-mode",
        choices=("online", "offline", "disabled"),
        default=None,
        help="Override mode W&B, terutama untuk smoke test offline.",
    )
    parser.add_argument(
        "--initial-checkpoint",
        type=Path,
        default=None,
        help="Resume/warm-start dari checkpoint TD3; mengalahkan config.",
    )
    args = parser.parse_args()
    checkpoint = train(
        args.config,
        args.steps,
        args.output_dir,
        args.random_steps,
        args.wandb_mode,
        args.initial_checkpoint,
    )
    print(f"final_checkpoint={checkpoint.resolve()}", flush=True)


if __name__ == "__main__":
    main()
