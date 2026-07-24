"""Render TD3 with the same multiview diagnostic layout as the SAC renderer.

The deterministic TD3 actor picks one action every configured ``frame_skip``
physics ticks; rendering substeps at frame_skip=1 so the video is smooth while
the action is held for the same simulator time as during training. TD3 has no
entropy temperature or stochastic latent, so the dashboard shows the twin
critic values instead of SAC's alpha/latent diagnostics.
"""

import argparse
import logging
import warnings
from pathlib import Path
from typing import List, Sequence, Tuple

import cv2
import imageio.v2 as imageio
import numpy as np
import torch
import yaml

from .actions import ActionConfig, build_action_table
from .agents.td3 import TD3Agent, TD3Config
from .continuous_env import build_continuous_env
from .decorations import attach_kfupm_small_loop_decorations
from .render_multiview_video import (
    BOTTOM_HEIGHT,
    HEIGHT,
    TOP_HEIGHT,
    WIDTH,
    _put_line,
    add_title,
    letterbox,
    trajectory_panel,
)
from .render_sac_multiview_video import (
    _gauge,
    capture_views,
    compose_frame,
)


def critic_diagnostics(
    agent: TD3Agent, observation: np.ndarray, action: np.ndarray, probes
):
    obs = torch.as_tensor(
        observation, dtype=torch.float32, device=agent.device
    ).unsqueeze(0)
    act = torch.as_tensor(
        action, dtype=torch.float32, device=agent.device
    ).unsqueeze(0)
    probe_actions = torch.as_tensor(
        probes, dtype=torch.float32, device=agent.device
    )
    probe_obs = obs.expand(probe_actions.shape[0], -1)
    with torch.no_grad():
        q1 = float(agent.critic1(obs, act).item())
        q2 = float(agent.critic2(obs, act).item())
        probe_q = torch.minimum(
            agent.critic1(probe_obs, probe_actions),
            agent.critic2(probe_obs, probe_actions),
        ).squeeze(1)
    return {
        "q1": q1,
        "q2": q2,
        "q_min": min(q1, q2),
        "probe_q": probe_q.cpu().numpy(),
    }


def td3_dashboard_panel(
    state,
    observation: np.ndarray,
    action: np.ndarray,
    diagnostics,
    probe_names: Sequence[str],
    position: Tuple[float, float],
    angle: float,
    decision: int,
    physics_step: int,
    cumulative_reward: float,
    termination_reason: str,
    action_low: np.ndarray,
    action_high: np.ndarray,
    width: int = WIDTH // 2,
    height: int = BOTTOM_HEIGHT,
) -> np.ndarray:
    panel = np.full((height, width, 3), (13, 18, 25), dtype=np.uint8)
    panel = add_title(panel, "STATE, TD3 ACTION & CRITIC")
    _put_line(panel, "SOLVER: TD3 / TEACHER-FREE", 640, 29, 0.50, (245, 248, 250), 2)

    stop_distance = "None" if state.d_stop is None else f"{state.d_stop:.3f} m"
    lines_left = [
        f"position (x,z) : ({position[0]:+.3f}, {position[1]:+.3f}) m",
        f"heading psi     : {np.degrees(angle):+.2f} deg",
        f"lateral d       : {state.d:+.4f} m",
        f"heading phi     : {state.phi:+.4f} rad",
        f"speed v         : {state.v:.4f}",
        f"curvature kappa : {state.kappa:+.4f} 1/m",
    ]
    lines_right = [
        f"stop present    : {state.stop_present}",
        f"stop distance   : {stop_distance}",
        f"stop satisfied  : {state.sigma_stop}",
        f"stop dwell      : {state.stop_hold_progress:.2f}",
        f"duck present    : {state.duck_present}",
        f"duck active     : {state.duck_active}",
        f"duck rel (L,D)  : ({state.duck_longitudinal:+.2f}, {state.duck_lateral:+.2f})",
    ]
    for index, line in enumerate(lines_left):
        _put_line(panel, line, 28, 73 + index * 28, 0.53)
    for index, line in enumerate(lines_right):
        _put_line(panel, line, 500, 73 + index * 28, 0.53)

    _put_line(
        panel,
        f"decision={decision}   physics_step={physics_step}   reward={cumulative_reward:+.3f}",
        28,
        260,
        0.57,
    )
    status_color = (
        (65, 220, 110)
        if termination_reason in {"in_progress", "timeout"}
        else (255, 92, 92)
    )
    _put_line(panel, f"status: {termination_reason}", 28, 289, 0.59, status_color, 2)
    _put_line(
        panel,
        "EXECUTED CONTINUOUS ACTION (held for 6 physics ticks)",
        28,
        319,
        0.50,
        (72, 230, 120),
        2,
    )
    _gauge(
        panel, "v_cmd", float(action[0]),
        float(action_low[0]), float(action_high[0]), 28, 349, 410,
    )
    _gauge(
        panel, "omega_cmd", float(action[1]),
        float(action_low[1]), float(action_high[1]), 500, 349, 410, centered=True,
    )
    _put_line(
        panel,
        "Q1={:+.3f}  Q2={:+.3f}  minQ={:+.3f}  (deterministic actor, no entropy temperature)".format(
            diagnostics["q1"], diagnostics["q2"], diagnostics["q_min"],
        ),
        28,
        397,
        0.51,
    )
    _put_line(
        panel,
        "CRITIC PROBES (twin-critic min; actor output is continuous)",
        28,
        425,
        0.47,
        (166, 185, 205),
    )
    values = np.asarray(diagnostics["probe_q"], dtype=float)
    value_min, value_max = float(values.min()), float(values.max())
    span = max(1e-9, value_max - value_min)
    best = int(np.argmax(values))
    bar_x, bar_width = 258, 620
    for row, (name, value) in enumerate(zip(probe_names, values)):
        y = 439 + row * 21
        marker = ">" if row == best else " "
        color = (56, 214, 104) if row == best else (64, 137, 204)
        _put_line(
            panel, f"{marker} {row}: {name:14s}", 28, y + 15, 0.43,
            thickness=2 if row == best else 1,
        )
        cv2.rectangle(panel, (bar_x, y), (bar_x + bar_width, y + 15), (31, 40, 51), -1)
        normalized = (float(value) - value_min) / span
        cv2.rectangle(
            panel, (bar_x, y), (bar_x + int(bar_width * normalized), y + 15), color, -1
        )
        _put_line(panel, f"{value:+.3f}", bar_x + bar_width - 72, y + 13, 0.39)
    return panel


def render_td3_multiview_video(
    config_path: Path,
    checkpoint_path: Path,
    output_path: Path,
    seed: int = 10101,
    fps: int = 20,
    max_steps: int = 1500,
    duration_seconds: float = 0.0,
    repeat_duck: bool = False,
    repeat_rearm_distance: float = 1.0,
    decorate_kfupm: bool = False,
) -> None:
    warnings.filterwarnings("ignore")
    logging.getLogger().setLevel(logging.WARNING)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if repeat_duck:
        config["duck_controller"]["max_crossings_per_episode"] = 0
        config["duck_controller"]["repeat_rearm_distance"] = float(
            repeat_rearm_distance
        )
    policy_repeat = int(config["environment"].get("frame_skip", 1))
    config["environment"]["frame_skip"] = 1
    config["state"]["stop_hold_steps"] = (
        int(config["state"].get("stop_hold_steps", 1)) * policy_repeat
    )
    if max_steps > 0:
        config["environment"]["max_steps"] = max_steps
    config["environment"]["render_observations"] = True
    device = str(config["training"].get("device", "cpu"))
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("Renderer meminta CUDA tetapi CUDA tidak tersedia")

    env = build_continuous_env(config, seed)
    if decorate_kfupm:
        if str(config["environment"].get("map_name", "")) != "small_loop":
            raise ValueError("KFUPM decoration layout is defined only for small_loop")
        asset_dir = Path(__file__).resolve().parents[1] / "assets"
        attach_kfupm_small_loop_decorations(env, asset_dir)
    agent = TD3Agent(
        env.observation_space.shape[0],
        env.action_space.low,
        env.action_space.high,
        TD3Config(**config["td3"]),
        seed=int(config["seed"]),
        device=device,
    )
    agent.load(checkpoint_path)
    action_specs = build_action_table(ActionConfig(**config["actions"]))
    probe_names = [spec.name for spec in action_specs]
    probes = np.asarray(
        [[spec.v, spec.omega] for spec in action_specs], dtype=np.float32
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(
        str(output_path),
        format="FFMPEG",
        mode="I",
        fps=fps,
        codec="libx264",
        quality=8,
        macro_block_size=8,
        ffmpeg_log_level="warning",
        ffmpeg_params=["-preset", "medium", "-pix_fmt", "yuv420p"],
    )

    simulator = env.unwrapped
    observation = None
    trajectory: List[Tuple[float, float]] = []
    video_clock = 0.0
    physics_steps = decisions = frames = 0
    episodes_rendered = 0
    cumulative_reward = 0.0
    termination_reason = "in_progress"
    target_frames = (
        int(round(duration_seconds * fps)) if duration_seconds > 0.0 else 0
    )

    def write_frame(action, decision_observation, diagnostics) -> None:
        nonlocal frames
        if target_frames and frames >= target_frames:
            return
        camera, bev, vantage = capture_views(env)
        position = (float(simulator.cur_pos[0]), float(simulator.cur_pos[2]))
        trajectory_view = trajectory_panel(
            trajectory,
            position,
            float(simulator.cur_angle),
            simulator.grid_width,
            simulator.grid_height,
            float(simulator.road_tile_size),
        )
        dashboard = td3_dashboard_panel(
            env.current_state,
            decision_observation,
            action,
            diagnostics,
            probe_names,
            position,
            float(simulator.cur_angle),
            decisions,
            physics_steps,
            cumulative_reward,
            termination_reason,
            env.action_space.low,
            env.action_space.high,
        )
        writer.append_data(
            compose_frame(camera, bev, vantage, trajectory_view, dashboard)
        )
        frames += 1

    try:
        while not target_frames or frames < target_frames:
            episode_seed = seed + episodes_rendered
            observation = env.reset(episode_seed)
            episodes_rendered += 1
            episode_physics_steps = 0
            trajectory = [
                (float(simulator.cur_pos[0]), float(simulator.cur_pos[2]))
            ]
            cumulative_reward = 0.0
            termination_reason = "in_progress"
            done = False
            while (
                not done
                and (max_steps <= 0 or episode_physics_steps < max_steps)
                and (not target_frames or frames < target_frames)
            ):
                decision_observation = observation.copy()
                action = agent.select_action(
                    decision_observation, deterministic=True
                )
                diagnostics = critic_diagnostics(
                    agent, decision_observation, action, probes
                )
                decisions += 1
                if episode_physics_steps == 0:
                    write_frame(action, decision_observation, diagnostics)
                for _ in range(policy_repeat):
                    if (
                        done
                        or (max_steps > 0 and episode_physics_steps >= max_steps)
                        or (target_frames and frames >= target_frames)
                    ):
                        break
                    observation, reward, done, info = env.step(action)
                    physics_steps += 1
                    episode_physics_steps += 1
                    cumulative_reward += float(reward)
                    termination_reason = info["termination_reason"]
                    trajectory.append(
                        (float(simulator.cur_pos[0]), float(simulator.cur_pos[2]))
                    )
                    video_clock += fps * float(simulator.delta_time)
                    while video_clock >= 1.0:
                        write_frame(action, decision_observation, diagnostics)
                        video_clock -= 1.0
            if not target_frames:
                break
    finally:
        writer.close()
        env.close()
    print(f"video={output_path.resolve()}")
    print(
        f"resolution={WIDTH}x{HEIGHT} fps={fps} frames={frames} "
        f"decisions={decisions} physics_steps={physics_steps} "
        f"episodes={episodes_rendered} status={termination_reason}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render synchronized multiview diagnostics for TD3."
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=10101)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--max-steps", type=int, default=1500)
    parser.add_argument("--duration-seconds", type=float, default=0.0)
    parser.add_argument("--repeat-duck", action="store_true")
    parser.add_argument("--repeat-rearm-distance", type=float, default=1.0)
    parser.add_argument("--decorate-kfupm", action="store_true")
    args = parser.parse_args()
    render_td3_multiview_video(
        args.config,
        args.checkpoint,
        args.output,
        args.seed,
        args.fps,
        args.max_steps,
        args.duration_seconds,
        args.repeat_duck,
        args.repeat_rearm_distance,
        args.decorate_kfupm,
    )


if __name__ == "__main__":
    main()
