"""Deterministic, teacher-free evaluation for a SAC checkpoint."""

import argparse
import json
import logging
import warnings
from collections import Counter
from math import cos, hypot
from pathlib import Path
from typing import Dict, Optional, Sequence

import numpy as np
import torch
import yaml

from .agents.sac import SACAgent, SACConfig
from .continuous_env import build_continuous_env


def _quiet_third_party_logs() -> None:
    warnings.filterwarnings("ignore")
    logging.getLogger().setLevel(logging.WARNING)
    for name in ("gym-duckietown", "gym_duckietown", "commons", "typing"):
        logging.getLogger(name).setLevel(logging.WARNING)


def resolve_eval_params(config, episodes, seeds):
    """Shared episode/seed defaulting so SAC and TD3 evaluate identically."""
    evaluation = config["evaluation"]
    if episodes is None:
        episodes = int(evaluation["final_episodes"])
    if episodes <= 0:
        raise ValueError("episodes harus lebih besar dari nol")
    if seeds is None:
        seeds = [int(value) for value in evaluation["final_seeds"]]
    if not seeds:
        raise ValueError("seed list tidak boleh kosong")
    return episodes, seeds


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
    agent = SACAgent(
        env.observation_space.shape[0],
        env.action_space.low,
        env.action_space.high,
        SACConfig(**config["sac"]),
        seed=int(config["seed"]),
        device=device,
    )
    agent.load(
        checkpoint_path,
        allow_observation_expansion=bool(
            config["training"].get("allow_observation_expansion", False)
        ),
    )
    return run_evaluation(env, agent, config, episodes, seeds)


def run_evaluation(env, agent, config, episodes, seeds):
    """Deterministic, teacher-free metric loop shared by SAC and TD3."""
    evaluation = config["evaluation"]
    decision_dt = env.unwrapped.delta_time * int(config["environment"]["frame_skip"])
    brake_threshold = float(evaluation["brake_command_threshold"])
    move_threshold = float(evaluation["move_command_threshold"])
    spin_threshold = float(evaluation["spin_omega_threshold"])
    resume_window = int(evaluation["resume_window_steps"])
    min_progress = float(evaluation["success_min_progress_m"])
    max_brake_ratio = float(evaluation["success_max_brake_ratio"])

    reasons = Counter()
    returns = []
    progresses = []
    deviations = []
    brake_ratios = []
    task_success = []
    stop_successes = stop_violations = 0
    episodes_with_stop_opportunity = 0
    episodes_with_compliant_stop = 0
    false_stop_steps = spin_steps = total_steps = 0
    crossing_steps = yield_steps = 0
    resume_opportunities = resume_successes = 0
    minimum_duck_distances = []
    completed_stop_hold_decisions = []
    maximum_stop_hold_decisions = []
    straight_abs_omega_sum = 0.0
    straight_action_steps = 0

    try:
        for episode in range(episodes):
            seed = int(seeds[episode % len(seeds)]) + episode
            observation = env.reset(seed)
            done = False
            total_return = progress = 0.0
            episode_steps = episode_brakes = 0
            pending_resume = 0
            previous_active = env.current_state.duck_active
            episode_stop_successes = episode_stop_violations = 0
            episode_max_stop_hold = 0
            info = {"termination_reason": "in_progress"}

            while not done:
                state_before = env.current_state
                # Evaluation is deterministic and teacher-free: only actor mean.
                action = agent.select_action(observation, deterministic=True)
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
                if is_straight:
                    straight_abs_omega_sum += abs(float(action[1]))
                    straight_action_steps += 1
                if is_brake and not crossing_before and not unmet_stop:
                    false_stop_steps += 1
                if is_brake and abs(float(action[1])) >= spin_threshold:
                    spin_steps += 1

                observation, reward, done, info = env.step(action)
                state_after = env.current_state
                total_return += float(reward)
                episode_steps += 1
                total_steps += 1
                episode_brakes += int(is_brake)
                deviations.append(abs(state_after.d))
                progress += max(0.0, state_after.v * cos(state_after.phi)) * decision_dt

                events = info["events"]
                full_stop = int(events["full_stop"])
                stop_violation = int(events["stop_violation"])
                stop_successes += full_stop
                stop_violations += stop_violation
                episode_stop_successes += full_stop
                episode_stop_violations += stop_violation
                episode_max_stop_hold = max(
                    episode_max_stop_hold,
                    int(info.get("stop_hold_steps", 0)),
                )
                if full_stop:
                    completed_stop_hold_decisions.append(
                        int(info.get("stop_hold_steps", 0))
                    )
                crossing_steps += int(state_after.duck_active)
                yield_steps += int(state_after.duck_active and is_brake)
                if state_after.duck_present:
                    minimum_duck_distances.append(
                        hypot(
                            state_after.duck_longitudinal,
                            state_after.duck_lateral,
                        )
                    )

                if previous_active and not state_after.duck_active:
                    resume_opportunities += 1
                    pending_resume = resume_window
                if pending_resume > 0:
                    if float(action[0]) >= move_threshold:
                        resume_successes += 1
                        pending_resume = 0
                    else:
                        pending_resume -= 1
                previous_active = state_after.duck_active

            reasons[info["termination_reason"]] += 1
            maximum_stop_hold_decisions.append(episode_max_stop_hold)
            returns.append(total_return)
            progresses.append(progress)
            brake_ratio = episode_brakes / max(1, episode_steps)
            brake_ratios.append(brake_ratio)
            has_stop_opportunity = (
                episode_stop_successes + episode_stop_violations
            ) > 0
            episodes_with_stop_opportunity += int(has_stop_opportunity)
            episodes_with_compliant_stop += int(
                has_stop_opportunity and episode_stop_violations == 0
            )
            task_success.append(
                info["termination_reason"] == "timeout"
                and progress >= min_progress
                and brake_ratio <= max_brake_ratio
                and has_stop_opportunity
                and episode_stop_violations == 0
            )
    finally:
        env.close()

    stop_opportunities = stop_successes + stop_violations
    failures = (
        reasons["duck_collision"]
        + reasons["other_collision"]
        + reasons["offroad"]
    )
    return {
        "stage": config.get("stage", "unknown"),
        "episodes": episodes,
        "mean_return": float(np.mean(returns)),
        "mean_forward_progress_m": float(np.mean(progresses)),
        "mean_abs_d": float(np.mean(deviations)),
        "p95_abs_d": float(np.percentile(deviations, 95)),
        "mean_brake_ratio": float(np.mean(brake_ratios)),
        "task_success_rate": float(np.mean(task_success)),
        "timeout_rate": reasons["timeout"] / episodes,
        "offroad_rate": reasons["offroad"] / episodes,
        "duck_collision_rate": reasons["duck_collision"] / episodes,
        "other_collision_rate": reasons["other_collision"] / episodes,
        "total_failure_rate": failures / episodes,
        "stop_compliance_rate": (
            stop_successes / stop_opportunities if stop_opportunities else 1.0
        ),
        "stop_opportunities": stop_opportunities,
        "stop_opportunity_episode_rate": episodes_with_stop_opportunity / episodes,
        "stop_compliant_episode_rate": episodes_with_compliant_stop / episodes,
        "mean_completed_stop_hold_decisions": (
            float(np.mean(completed_stop_hold_decisions))
            if completed_stop_hold_decisions
            else None
        ),
        "mean_completed_stop_hold_seconds": (
            float(np.mean(completed_stop_hold_decisions) * decision_dt)
            if completed_stop_hold_decisions
            else None
        ),
        "mean_episode_max_stop_hold_decisions": float(
            np.mean(maximum_stop_hold_decisions)
        ),
        "mean_abs_omega_on_straight": (
            straight_abs_omega_sum / max(1, straight_action_steps)
        ),
        "false_stop_rate": false_stop_steps / max(1, total_steps),
        "spin_in_place_rate": spin_steps / max(1, total_steps),
        "duck_yield_step_rate": yield_steps / max(1, crossing_steps),
        "resume_after_clear_rate": (
            resume_successes / resume_opportunities if resume_opportunities else 1.0
        ),
        "resume_opportunities": resume_opportunities,
        "minimum_duck_distance_m": (
            float(np.min(minimum_duck_distances))
            if minimum_duck_distances
            else None
        ),
        "termination_counts": dict(reasons),
    }


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
