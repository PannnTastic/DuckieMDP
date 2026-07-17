import argparse
import json
from collections import Counter
from math import cos
from pathlib import Path

import numpy as np
import yaml

from .agents.factory import algorithm_name, build_tabular_agent
from .discretizer import discretize
from .env_wrapper import build_env
from .state import DuckThreat


def evaluate(config_path: Path, q_path: Path) -> None:
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    seeds = config["evaluation"]["seeds"]
    episodes = int(config["evaluation"]["episodes"])
    env = build_env(config, int(seeds[0]))
    agent = build_tabular_agent(config, int(seeds[0]))
    agent.load(q_path)

    returns, lengths, physics_lengths, deviations = [], [], [], []
    progresses, moving_ratios, brake_ratios, tile_transitions = [], [], [], []
    task_successes = []
    reasons = Counter()
    stops = violations = 0
    duck_crossing_steps = duck_yield_steps = duck_encounters = 0
    duck_encounter_episodes = 0
    decision_dt = env.unwrapped.delta_time * env.unwrapped.frame_skip
    try:
        for episode in range(episodes):
            raw = env.reset(int(seeds[episode % len(seeds)]) + episode)
            done, total, length = False, 0.0, 0
            abs_d, progress, moving, brakes, transitions = [], 0.0, 0, 0, 0
            previous_tile = tuple(env.unwrapped.get_grid_coords(env.unwrapped.cur_pos))
            previous_crossing = False
            episode_saw_duck = False
            info = {"termination_reason": "in_progress"}
            while not done:
                action = agent.select_action(discretize(raw), greedy=True)
                raw, reward, done, info = env.step(action)
                total += reward
                length += 1
                abs_d.append(abs(raw.d))
                progress += max(0.0, raw.v * cos(raw.phi)) * decision_dt
                moving += int(raw.v > 0.02)
                brakes += int(action == 6)
                tile = tuple(env.unwrapped.get_grid_coords(env.unwrapped.cur_pos))
                transitions += int(tile != previous_tile)
                previous_tile = tile
                event = info["events"]
                stops += int(event["full_stop"])
                violations += int(event["stop_violation"])
                crossing = raw.duck in {DuckThreat.CROSSING_FAR, DuckThreat.CROSSING_NEAR}
                duck_crossing_steps += int(crossing)
                duck_yield_steps += int(crossing and raw.v < 0.04)
                duck_encounters += int(crossing and not previous_crossing)
                episode_saw_duck = episode_saw_duck or crossing
                previous_crossing = crossing
            duck_encounter_episodes += int(episode_saw_duck)
            reasons[info["termination_reason"]] += 1
            returns.append(total)
            lengths.append(length)
            physics_lengths.append(int(env.unwrapped.step_count))
            deviations.extend(abs_d)
            progresses.append(progress)
            moving_ratios.append(moving / length if length else 0.0)
            brake_ratios.append(brakes / length if length else 0.0)
            tile_transitions.append(transitions)
            min_progress = float(config["evaluation"].get("success_min_progress_m", 5.0))
            max_brake_ratio = float(config["evaluation"].get("success_max_brake_ratio", 0.25))
            task_successes.append(
                info["termination_reason"] == "timeout"
                and progress >= min_progress
                and brake_ratios[-1] <= max_brake_ratio
            )
    finally:
        env.close()

    opportunities = stops + violations
    report = {
        "algorithm": algorithm_name(config),
        "episodes": episodes,
        "mean_return": float(np.mean(returns)),
        "duck_collision_rate": reasons["duck_collision"] / episodes,
        "other_collision_rate": reasons["other_collision"] / episodes,
        "offroad_rate": reasons["offroad"] / episodes,
        "timeout_rate": reasons["timeout"] / episodes,
        # Timeout saja bukan sukses: policy yang brake permanen juga timeout.
        # Task success mensyaratkan kemajuan minimum dan brake tidak berlebihan.
        "task_success_rate": float(np.mean(task_successes)),
        "success_min_progress_m": float(config["evaluation"].get("success_min_progress_m", 5.0)),
        "success_max_brake_ratio": float(config["evaluation"].get("success_max_brake_ratio", 0.25)),
        "goal_rate": reasons["goal"] / episodes,
        "termination_counts": dict(reasons),
        "stop_compliance_rate": stops / opportunities if opportunities else 1.0,
        "stop_opportunities": opportunities,
        "duck_crossing_encounters": duck_encounters,
        "duck_encounter_episode_rate": duck_encounter_episodes / episodes,
        "duck_crossing_exposure_steps": duck_crossing_steps,
        "duck_yield_step_rate": duck_yield_steps / duck_crossing_steps if duck_crossing_steps else 1.0,
        "mean_abs_d": float(np.mean(deviations)) if deviations else 0.0,
        "p95_abs_d": float(np.percentile(deviations, 95)) if deviations else 0.0,
        "mean_episode_length": float(np.mean(lengths)),
        "mean_physics_steps": float(np.mean(physics_lengths)),
        "mean_forward_progress_m": float(np.mean(progresses)),
        "mean_moving_step_ratio": float(np.mean(moving_ratios)),
        "mean_brake_ratio": float(np.mean(brake_ratios)),
        "mean_tile_transitions": float(np.mean(tile_transitions)),
    }
    output = q_path.parent / "evaluation_report.json"
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    parser.add_argument("--q-table", type=Path, default=Path("results/q_table.npy"))
    args = parser.parse_args()
    evaluate(args.config, args.q_table)
