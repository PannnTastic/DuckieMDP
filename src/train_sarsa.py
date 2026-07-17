"""Training teacher-guided on-policy SARSA.

Teacher hanya membentuk behavior policy saat training. Aksi ``a'`` yang masuk
target SARSA adalah aksi aktual hasil campuran epsilon-greedy dan teacher, lalu
aksi yang sama dibawa ke iterasi berikutnya. Setelah teacher decay ke nol,
policy melanjutkan training secara mandiri.
"""

import argparse
import csv
from collections import deque
from pathlib import Path
import shutil
from typing import Tuple

import numpy as np
import yaml

from .agents.sarsa import SarsaAgent, SarsaConfig
from .discretizer import discretize
from .env_wrapper import build_env
from .lane_teacher import LaneTeacherConfig, select_lane_teacher_action, teacher_probability


def _write_csv_atomic(rows, path: Path) -> None:
    if not rows:
        return
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _save_checkpoint(agent: SarsaAgent, rows, output: Path) -> None:
    temporary = output / "q_table_checkpoint.tmp.npy"
    agent.save(temporary)
    temporary.replace(output / "q_table_checkpoint.npy")
    _write_csv_atomic(rows, output / "training_partial.csv")


def select_behavior_action(
    agent: SarsaAgent,
    raw_state,
    discrete_state: Tuple[int, ...],
    teacher_cfg: LaneTeacherConfig,
    teacher_chance: float,
    behavior_rng: np.random.RandomState,
    advance_step: bool = True,
) -> Tuple[int, bool]:
    """Campurkan epsilon-greedy student dan teacher menjadi behavior policy."""
    action = agent.select_action(discrete_state, advance_step=advance_step)
    teacher_used = bool(behavior_rng.random_sample() < teacher_chance)
    if teacher_used:
        action = select_lane_teacher_action(raw_state, teacher_cfg)
    return action, teacher_used


def train(config_path: Path) -> None:
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if str(config.get("algorithm", "")).lower() != "sarsa":
        raise ValueError("SARSA config must set algorithm: sarsa")

    seed = int(config["seed"])
    env = build_env(config, seed)
    agent = SarsaAgent(SarsaConfig(**config["sarsa"]), seed)
    initial_q_table = config["training"].get("initial_q_table")
    if initial_q_table:
        # Curriculum transfer: mulai full task dari kemampuan lane-following
        # SARSA yang sudah dipelajari, lalu isi nilai state stop/duck baru.
        agent.load(Path(initial_q_table))
        print(f"initial_sarsa_q_table={initial_q_table}", flush=True)
    teacher_cfg = LaneTeacherConfig(**config.get("lane_teacher", {}))
    behavior_rng = np.random.RandomState(seed + 10000)
    output = Path(config["training"]["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    shutil.copy2(config_path, output / "config.yaml")

    rows = []
    episodes = int(config["training"]["episodes"])
    ma_window = int(config["training"].get("moving_average_window", 100))
    checkpoint_every = int(config["training"].get("checkpoint_every", 50))
    milestone_episodes = {
        int(value) for value in config["training"].get("milestone_episodes", [])
    }
    recent_returns = deque(maxlen=max(1, ma_window))
    interrupted = False
    try:
        for episode in range(episodes):
            raw = env.reset(seed + episode)
            state = discretize(raw)
            probability = teacher_probability(episode, teacher_cfg)
            action, action_from_teacher = select_behavior_action(
                agent, raw, state, teacher_cfg, probability, behavior_rng
            )
            done, total, steps = False, 0.0, 0
            stops, violations, teacher_steps = 0, 0, 0
            termination_reason = "in_progress"

            while not done:
                next_raw, reward, done, info = env.step(action)
                terminated = bool(info["terminated"])
                truncated = bool(info["truncated"])
                teacher_steps += int(action_from_teacher)

                if terminated:
                    next_state, next_action, next_from_teacher = None, None, False
                else:
                    next_state = discretize(next_raw)
                    # Untuk timeout, a' hanya dipakai sebagai bootstrap dan tidak
                    # dihitung sebagai keputusan/teacher step yang dieksekusi.
                    next_action, next_from_teacher = select_behavior_action(
                        agent,
                        next_raw,
                        next_state,
                        teacher_cfg,
                        probability,
                        behavior_rng,
                        advance_step=not truncated,
                    )

                agent.update(state, action, reward, next_state, next_action, terminated)
                raw, total, steps = next_raw, total + reward, steps + 1
                event = info["events"]
                stops += int(event["full_stop"])
                violations += int(event["stop_violation"])
                termination_reason = info["termination_reason"]

                if not done:
                    state, action, action_from_teacher = next_state, next_action, next_from_teacher

            recent_returns.append(total)
            moving_average = sum(recent_returns) / len(recent_returns)
            opportunities = stops + violations
            rows.append(
                {
                    "episode": episode,
                    "return": total,
                    "return_moving_average": moving_average,
                    "steps": steps,
                    "physics_steps": int(env.unwrapped.step_count),
                    "termination_reason": termination_reason,
                    "terminated": int(info["terminated"]),
                    "truncated": int(info["truncated"]),
                    "duck_collision": int(termination_reason == "duck_collision"),
                    "other_collision": int(termination_reason == "other_collision"),
                    "offroad": int(termination_reason == "offroad"),
                    "timeout": int(termination_reason == "timeout"),
                    "goal": int(termination_reason == "goal"),
                    "stop_compliance": stops / opportunities if opportunities else 1.0,
                    "teacher_probability": probability,
                    "teacher_step_ratio": teacher_steps / max(1, steps),
                    "epsilon": agent.epsilon,
                }
            )
            if (episode + 1) % checkpoint_every == 0:
                _save_checkpoint(agent, rows, output)
            if (episode + 1) in milestone_episodes:
                agent.save(output / f"q_table_ep{episode + 1}.npy")
            if (episode + 1) % int(config["training"]["log_every"]) == 0:
                print(
                    "episode=%d return=%.2f return_ma=%.2f epsilon=%.3f teacher=%.2f reason=%s"
                    % (episode + 1, total, moving_average, agent.epsilon, probability, termination_reason),
                    flush=True,
                )
    except KeyboardInterrupt:
        interrupted = True
        print("SARSA training interrupted; saving atomic checkpoint...", flush=True)
    finally:
        _save_checkpoint(agent, rows, output)
        env.close()

    if interrupted:
        print(f"checkpoint={output / 'q_table_checkpoint.npy'}", flush=True)
        return
    agent.save(output / "q_table.npy")
    _write_csv_atomic(rows, output / "training.csv")
    print(f"sarsa_q_table={output / 'q_table.npy'}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/small_loop_lane_sarsa.yaml"))
    train(parser.parse_args().config)
