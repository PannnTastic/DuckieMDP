"""Teacher-free SAC training with replay, curriculum warm-start, and checkpoints.

The behaviour policy is the SAC actor plus entropy (or random actions during
warm-up).  No scripted teacher chooses actions in this training loop.
"""

import argparse
import csv
import logging
import shutil
import warnings
from math import cos, hypot
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
import yaml

from .agents.sac import SACAgent, SACConfig
from .continuous_env import build_continuous_env


REWARD_COMPONENTS = (
    "progress",
    "lateral",
    "heading",
    "time",
    "pedestrian",
    "stagnation",
    "steering",
    "events",
    "total",
)


def _empty_reward_totals() -> Dict[str, float]:
    return {name: 0.0 for name in REWARD_COMPONENTS}


def _quiet_third_party_logs() -> None:
    """Keep long training output focused on experiment metrics."""
    warnings.filterwarnings("ignore")
    logging.getLogger().setLevel(logging.WARNING)
    for name in ("gym-duckietown", "gym_duckietown", "commons", "typing"):
        logging.getLogger(name).setLevel(logging.WARNING)


def _write_rows(rows, path: Path) -> None:
    if not rows:
        return
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _require_device(device_name: str) -> None:
    if device_name.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            "Config meminta CUDA, tetapi build PyTorch/driver tidak menyediakan CUDA"
        )


def _init_wandb(config: Dict[str, Any], output: Path):
    """Start an optional W&B run without ever reading/storing an API key here."""
    wandb_cfg = config.get("wandb", {})
    if not bool(wandb_cfg.get("enabled", False)):
        return None
    try:
        import wandb
    except ImportError as error:
        raise RuntimeError(
            "W&B diaktifkan, tetapi package wandb belum terpasang. "
            "Install requirements-sac.txt terlebih dahulu."
        ) from error

    run = wandb.init(
        entity=wandb_cfg.get("entity"),
        project=wandb_cfg.get("project"),
        name=wandb_cfg.get("name"),
        group=wandb_cfg.get("group"),
        job_type=wandb_cfg.get("job_type", "train"),
        tags=list(wandb_cfg.get("tags", [])),
        notes=wandb_cfg.get("notes"),
        mode=wandb_cfg.get("mode", "online"),
        dir=str(output.resolve()),
        config=config,
    )
    print(f"wandb_run={run.name} url={run.url}", flush=True)
    return run


def _log_checkpoint_artifact(
    wandb_run,
    checkpoint: Path,
    stage: str,
    step: int,
    aliases,
) -> None:
    if wandb_run is None:
        return
    import wandb

    artifact = wandb.Artifact(
        name=f"sac-{stage}-{wandb_run.id}",
        type="model",
        metadata={"environment_step": int(step), "stage": stage},
    )
    artifact.add_file(str(checkpoint), name=checkpoint.name)
    wandb_run.log_artifact(artifact, aliases=list(aliases))


def train(
    config_path: Path,
    steps_override: Optional[int] = None,
    output_override: Optional[Path] = None,
    random_steps_override: Optional[int] = None,
    wandb_mode_override: Optional[str] = None,
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
    agent = SACAgent(
        obs_dim=env.observation_space.shape[0],
        action_low=env.action_space.low,
        action_high=env.action_space.high,
        cfg=SACConfig(**config["sac"]),
        seed=seed,
        device=device,
    )
    initial = training.get("initial_checkpoint")
    # Curriculum transfer loads actor, critics, temperature, and optimizers. The
    # replay buffer intentionally starts empty because the stage dynamics change.
    if initial and steps_override is None:
        initial_path = Path(initial)
        if not initial_path.exists():
            raise FileNotFoundError(
                f"Initial curriculum checkpoint tidak ditemukan: {initial_path}"
            )
        agent.load(
            initial_path,
            allow_observation_expansion=bool(
                training.get("allow_observation_expansion", False)
            ),
        )
        print(f"initial_checkpoint={initial_path}", flush=True)
        if bool(training.get("save_initial_checkpoint", False)):
            step_zero = checkpoints / "sac_step_000000000.pt"
            agent.save(step_zero)
            if bool(config.get("wandb", {}).get("log_checkpoints", True)):
                _log_checkpoint_artifact(
                    wandb_run,
                    step_zero,
                    str(config.get("stage", "unknown")),
                    0,
                    aliases=("warm-start", "step-0"),
                )

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
                    "alpha": last_metrics.get("alpha", np.nan),
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
                    f"alpha={last_metrics.get('alpha', np.nan):.4f} "
                    f"replay={len(agent.replay)}",
                    flush=True,
                )
                if wandb_run is not None:
                    wandb_run.log(
                        {
                            "train/episode": episode,
                            "train/return_ma20": mean_return,
                            "train/alpha": last_metrics.get("alpha", np.nan),
                            "train/critic_loss": last_metrics.get(
                                "critic_loss", np.nan
                            ),
                            "train/actor_loss": last_metrics.get(
                                "actor_loss", np.nan
                            ),
                            "train/alpha_loss": last_metrics.get(
                                "alpha_loss", np.nan
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
                checkpoint = checkpoints / f"sac_step_{step:09d}.pt"
                agent.save(checkpoint)
                if bool(config.get("wandb", {}).get("log_checkpoints", True)):
                    _log_checkpoint_artifact(
                        wandb_run,
                        checkpoint,
                        str(config.get("stage", "unknown")),
                        step,
                        aliases=(f"step-{step}",),
                    )

        final_checkpoint = output / "sac_final.pt"
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
        agent.save(output / "sac_interrupted.pt")
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
    args = parser.parse_args()
    checkpoint = train(
        args.config,
        args.steps,
        args.output_dir,
        args.random_steps,
        args.wandb_mode,
    )
    print(f"final_checkpoint={checkpoint.resolve()}", flush=True)


if __name__ == "__main__":
    main()
