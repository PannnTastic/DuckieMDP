"""Render deterministic SAC at 20 FPS without accelerating simulator time."""

import argparse
import logging
import warnings
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import torch
import yaml

from .agents.sac import SACAgent, SACConfig
from .continuous_env import build_continuous_env


def capture_frame(env, view):
    if view == "camera":
        frame = env.unwrapped.render_obs()
    elif view == "follow":
        frame = env.unwrapped.render(mode="rgb_array")
    else:
        raise ValueError(f"unsupported view: {view}")
    frame = np.asarray(frame)
    if frame.dtype != np.uint8:
        frame = np.clip(frame, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(frame)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=10101)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--view", choices=("camera", "follow"), default="camera")
    parser.add_argument(
        "--max-decisions",
        type=int,
        default=None,
        help="Optional limit for a short renderer smoke test.",
    )
    args = parser.parse_args()

    warnings.filterwarnings("ignore")
    logging.getLogger().setLevel(logging.WARNING)
    for name in ("gym-duckietown", "gym_duckietown", "commons", "typing"):
        logging.getLogger(name).setLevel(logging.WARNING)
    with args.config.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    config["environment"]["render_observations"] = True
    device = str(config["training"].get("device", "cpu"))
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("Renderer meminta CUDA tetapi CUDA tidak tersedia")
    env = build_continuous_env(config, args.seed)
    agent = SACAgent(
        env.observation_space.shape[0],
        env.action_space.low,
        env.action_space.high,
        SACConfig(**config["sac"]),
        seed=int(config["seed"]),
        device=device,
    )
    agent.load(
        args.checkpoint,
        allow_observation_expansion=bool(
            config["training"].get("allow_observation_expansion", False)
        ),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)

    writer = imageio.get_writer(
        str(args.output),
        format="FFMPEG",
        mode="I",
        fps=args.fps,
        codec="libx264",
        quality=8,
        macro_block_size=16,
        ffmpeg_log_level="warning",
    )
    decision_seconds = env.unwrapped.delta_time * int(
        config["environment"]["frame_skip"]
    )
    video_clock = 0.0
    decisions = 0
    final_reason = "in_progress"
    try:
        observation = env.reset(args.seed)
        writer.append_data(capture_frame(env, args.view))
        done = False
        while not done and (
            args.max_decisions is None or decisions < args.max_decisions
        ):
            action = agent.select_action(observation, deterministic=True)
            observation, _, done, info = env.step(action)
            decisions += 1
            final_reason = info["termination_reason"]
            frame = capture_frame(env, args.view)
            # At frame_skip=6 and simulator dt=1/30 s, one decision lasts 0.2 s.
            # At 20 FPS that frame must occupy four video frames.
            video_clock += args.fps * decision_seconds
            while video_clock >= 1.0:
                writer.append_data(frame)
                video_clock -= 1.0
    finally:
        writer.close()
        env.close()
    print(f"video={args.output.resolve()}")
    print(f"fps={args.fps} decisions={decisions} reason={final_reason}")


if __name__ == "__main__":
    main()
