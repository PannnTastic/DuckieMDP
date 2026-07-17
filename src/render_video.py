import argparse
from pathlib import Path
from typing import Any, Dict

import imageio.v2 as imageio
import numpy as np
import yaml

from .agents.factory import build_tabular_agent
from .discretizer import discretize
from .env_wrapper import build_env


def load_config(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"Invalid or empty configuration: {path}")
    return config


def capture_frame(env, view: str) -> np.ndarray:
    simulator = env.unwrapped
    if view == "camera":
        frame = simulator.render_obs()
    elif view == "follow":
        frame = simulator.render(mode="rgb_array")
    else:
        raise ValueError(f"Unsupported view: {view}")
    frame = np.asarray(frame)
    if frame.dtype != np.uint8:
        frame = np.clip(frame, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(frame)


def render_solver_video(
    config_path: Path,
    q_table_path: Path,
    output_path: Path,
    seed: int,
    fps: int,
    max_steps: int,
    view: str,
) -> None:
    if output_path.suffix.lower() != ".mp4":
        raise ValueError("Output file must use the .mp4 extension")
    if not q_table_path.is_file():
        raise FileNotFoundError(f"Q-table not found: {q_table_path}")

    config = load_config(config_path)
    policy_repeat = int(config["environment"].get("frame_skip", 1))
    # Step physics one frame at a time for recording, while holding each policy
    # action for the same number of frames used during training.
    config["environment"]["frame_skip"] = 1
    if max_steps > 0:
        config["environment"]["max_steps"] = max_steps

    env = build_env(config, seed)
    agent = build_tabular_agent(config, seed)
    agent.load(q_table_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    writer = imageio.get_writer(
        str(output_path),
        format="FFMPEG",
        mode="I",
        fps=fps,
        codec="libx264",
        quality=8,
        macro_block_size=16,
        ffmpeg_log_level="warning",
    )

    total_reward = 0.0
    decisions = physics_steps = 0
    final_events: Dict[str, bool] = {}
    video_clock = 0.0
    last_frame_written_at = -1
    try:
        state = env.reset(seed)
        writer.append_data(capture_frame(env, view))
        last_frame_written_at = 0
        done = False
        while not done and (max_steps <= 0 or physics_steps < max_steps):
            action = agent.select_action(discretize(state), greedy=True)
            decisions += 1
            for _ in range(policy_repeat):
                if done or (max_steps > 0 and physics_steps >= max_steps):
                    break
                state, reward, done, info = env.step(action)
                total_reward += float(reward)
                physics_steps += 1
                final_events = info["events"]
                # Sample the 30 Hz physics stream at the requested video FPS.
                video_clock += fps * float(env.unwrapped.delta_time)
                if video_clock >= 1.0:
                    writer.append_data(capture_frame(env, view))
                    last_frame_written_at = physics_steps
                    video_clock -= 1.0
        if last_frame_written_at != physics_steps:
            writer.append_data(capture_frame(env, view))
    finally:
        writer.close()
        env.close()

    print(f"video={output_path.resolve()}")
    print(
        f"fps={fps} decisions={decisions} physics_steps={physics_steps} "
        f"sim_seconds={physics_steps * float(env.unwrapped.delta_time):.3f} "
        f"sampled_reward={total_reward:.3f}"
    )
    print(f"final_events={final_events}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Render a greedy policy rollout to MP4 in simulation time.")
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    parser.add_argument("--q-table", type=Path, default=Path("results/q_table.npy"))
    parser.add_argument("--output", type=Path, default=Path("results/solver.mp4"))
    parser.add_argument("--seed", type=int, default=101)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument(
        "--max-steps",
        type=int,
        default=1500,
        help="Maximum physics steps; use 0 to follow the config value.",
    )
    parser.add_argument("--view", choices=("camera", "follow"), default="camera")
    args = parser.parse_args()
    if args.fps <= 0 or args.max_steps < 0:
        parser.error("--fps must be positive and --max-steps cannot be negative")
    render_solver_video(
        args.config,
        args.q_table,
        args.output,
        args.seed,
        args.fps,
        args.max_steps,
        args.view,
    )


if __name__ == "__main__":
    main()
