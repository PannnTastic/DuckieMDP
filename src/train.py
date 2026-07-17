import argparse
import csv
from collections import deque
from pathlib import Path
import shutil

import numpy as np
import yaml

from .agents.q_learning import QLearningAgent, QLearningConfig
from .discretizer import discretize
from .env_wrapper import build_env
from .lane_teacher import LaneTeacherConfig, select_lane_teacher_action, teacher_probability
from .transition_model import EmpiricalTransitionModel


def _broadcast_lane_prior(q: np.ndarray) -> None:
    """Transfer lane-only Q values to every stop/duck context in-place.

    A lane-only run observes only ``d_stop=0, sigma_stop=0, duck=NONE``.
    The lane-control part of its policy is nevertheless a useful prior for all
    safety contexts. Broadcasting that slice prevents unseen context indices
    from starting at an all-zero Q vector; subsequent Q-learning updates remain
    fully teacher-free and specialize each context from environment rewards.
    """
    lane_slice = q[:, :, :, :, 0, 0, 0, :].copy()
    q[...] = lane_slice[:, :, :, :, None, None, None, :]


def _write_csv_atomic(rows, path: Path) -> None:
    if not rows:
        return
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _save_checkpoint(agent, rows, output, transition_model=None) -> None:
    temporary = output / "q_table_checkpoint.tmp.npy"
    agent.save(temporary)
    temporary.replace(output / "q_table_checkpoint.npy")
    _write_csv_atomic(rows, output / "training_partial.csv")
    if transition_model is not None:
        model_temporary = output / "transition_model.tmp.npz"
        transition_model.save(model_temporary)
        model_temporary.replace(output / "transition_model.npz")


def train(config_path: Path) -> None:
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    seed = int(config["seed"])
    env = build_env(config, seed)
    agent = QLearningAgent(QLearningConfig(**config["q_learning"]), seed)
    initial_q_table = config.get("training", {}).get("initial_q_table")
    if initial_q_table:
        agent.load(Path(initial_q_table))
        print(f"initialized_q_table={Path(initial_q_table)}", flush=True)
        if bool(config.get("training", {}).get("broadcast_lane_prior", False)):
            _broadcast_lane_prior(agent.q)
            print("broadcast_lane_prior=true", flush=True)
    teacher_cfg = LaneTeacherConfig(**config.get("lane_teacher", {}))
    behavior_rng = np.random.RandomState(seed + 10000)
    model_cfg = config.get("transition_model", {})
    transition_model = EmpiricalTransitionModel() if model_cfg.get("enabled", False) else None
    terminal_on_truncation = bool(model_cfg.get("terminal_on_truncation", True))
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
            done, total, steps = False, 0.0, 0
            stops, violations, teacher_steps = 0, 0, 0
            termination_reason = "in_progress"
            probability = teacher_probability(episode, teacher_cfg)
            while not done:
                state = discretize(raw)
                action = agent.select_action(state)
                if behavior_rng.random_sample() < probability:
                    action = select_lane_teacher_action(raw, teacher_cfg)
                    teacher_steps += 1
                next_raw, reward, done, info = env.step(action)
                terminated = bool(info["terminated"])
                truncated = bool(info["truncated"])
                next_state = None if terminated else discretize(next_raw)
                agent.update(state, action, reward, next_state, terminated)
                if transition_model is not None:
                    planning_terminal = terminated or (truncated and terminal_on_truncation)
                    transition_model.observe(
                        state,
                        action,
                        reward,
                        None if planning_terminal else next_state,
                        planning_terminal,
                    )
                raw, total, steps = next_raw, total + reward, steps + 1
                event = info["events"]
                stops += int(event["full_stop"])
                violations += int(event["stop_violation"])
                termination_reason = info["termination_reason"]

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
                _save_checkpoint(agent, rows, output, transition_model)
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
        print("Training interrupted; saving atomic checkpoint...", flush=True)
    finally:
        _save_checkpoint(agent, rows, output, transition_model)
        env.close()

    if interrupted:
        print(f"checkpoint={output / 'q_table_checkpoint.npy'}", flush=True)
        return
    agent.save(output / "q_table.npy")
    _write_csv_atomic(rows, output / "training.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    train(parser.parse_args().config)
