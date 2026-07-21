"""Render SAC with the same diagnostic layout as the tabular baseline.

The actor makes one deterministic decision every configured ``frame_skip``
physics ticks. Rendering substeps the simulator at frame_skip=1 so the 20 FPS
video remains smooth while the selected action is held for exactly the same
amount of simulator time as during training.
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
from .agents.sac import SACAgent, SACConfig
from .continuous_env import build_continuous_env
from .decorations import attach_kfupm_small_loop_decorations
from .explainability.schema import CanonicalAction, SolverKind
from .explainability.semantic_state import canonical_from_continuous_state
from .explainability.video_overlay import sac_video_explanation
from .render_multiview_video import (
    BOTTOM_HEIGHT,
    HEIGHT,
    TOP_HEIGHT,
    WIDTH,
    _put_line,
    add_title,
    letterbox,
    explanation_view_panel,
    trajectory_panel,
)


def full_loop_vantage(bev: np.ndarray) -> np.ndarray:
    """Turn the full-map BEV into a fixed oblique full-loop vantage.

    This perspective projection deliberately keeps all four BEV corners in the
    output. Unlike the simulator follow camera, it never crops the small loop.
    """
    source = np.asarray(bev, dtype=np.uint8)
    height, width = source.shape[:2]
    src = np.float32(
        [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]]
    )
    dst = np.float32(
        [
            [0.22 * width, 0.10 * height],
            [0.78 * width, 0.10 * height],
            [0.98 * width, 0.96 * height],
            [0.02 * width, 0.96 * height],
        ]
    )
    transform = cv2.getPerspectiveTransform(src, dst)
    canvas = np.full_like(source, (10, 14, 19))
    warped = cv2.warpPerspective(
        source,
        transform,
        (width, height),
        dst=canvas,
        borderMode=cv2.BORDER_TRANSPARENT,
    )
    cv2.polylines(
        warped,
        [np.rint(dst).astype(np.int32)],
        True,
        (78, 88, 99),
        2,
        cv2.LINE_AA,
    )
    return warped


def capture_views(env) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    simulator = env.unwrapped
    camera = np.ascontiguousarray(simulator.render_obs())
    bev = np.ascontiguousarray(simulator.render(mode="top_down"))
    vantage = np.ascontiguousarray(full_loop_vantage(bev))
    return camera, bev, vantage


def critic_diagnostics(agent: SACAgent, observation: np.ndarray, action: np.ndarray, probes):
    obs = torch.as_tensor(
        observation, dtype=torch.float32, device=agent.device
    ).unsqueeze(0)
    act = torch.as_tensor(action, dtype=torch.float32, device=agent.device).unsqueeze(0)
    probe_actions = torch.as_tensor(probes, dtype=torch.float32, device=agent.device)
    probe_obs = obs.expand(probe_actions.shape[0], -1)
    with torch.no_grad():
        distribution = agent.actor.distribution(obs)
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
        "alpha": float(agent.alpha.detach().item()),
        "latent_mean": distribution.mean.squeeze(0).cpu().numpy(),
        "latent_std": distribution.stddev.squeeze(0).cpu().numpy(),
        "probe_q": probe_q.cpu().numpy(),
    }


def _gauge(
    panel: np.ndarray,
    label: str,
    value: float,
    low: float,
    high: float,
    x: int,
    y: int,
    width: int,
    centered: bool = False,
) -> None:
    _put_line(panel, f"{label}: {value:+.4f}", x, y - 7, 0.49)
    cv2.rectangle(panel, (x, y), (x + width, y + 18), (31, 40, 51), -1)
    ratio = float(np.clip((value - low) / max(high - low, 1e-9), 0.0, 1.0))
    if centered:
        zero = int(round(x + width * ((0.0 - low) / (high - low))))
        endpoint = int(round(x + width * ratio))
        cv2.rectangle(
            panel,
            (min(zero, endpoint), y),
            (max(zero, endpoint), y + 18),
            (52, 196, 235),
            -1,
        )
        cv2.line(panel, (zero, y - 2), (zero, y + 20), (220, 225, 230), 1)
    else:
        cv2.rectangle(
            panel,
            (x, y),
            (x + int(round(width * ratio)), y + 18),
            (52, 196, 235),
            -1,
        )


def sac_dashboard_panel(
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
    explanation=None,
    width: int = WIDTH // 2,
    height: int = BOTTOM_HEIGHT,
) -> np.ndarray:
    panel = np.full((height, width, 3), (13, 18, 25), dtype=np.uint8)
    panel = add_title(panel, "STATE, SAC ACTION & CRITIC")
    _put_line(panel, "SOLVER: SAC / TEACHER-FREE", 670, 29, 0.50, (245, 248, 250), 2)
    if explanation is not None:
        color = (80, 95, 255) if explanation.undesirable else (72, 230, 120)
        _put_line(
            panel, f"PRIMITIVE: {explanation.primitive}", 350, 29,
            0.45, color, 2,
        )

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
    status_text = f"status: {termination_reason}"
    if explanation is not None:
        status_text += f"  |  trigger: {explanation.trigger}"
    _put_line(
        panel, status_text, 28, 289,
        0.43 if explanation is not None else 0.59, status_color, 2,
    )
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
        panel,
        "v_cmd",
        float(action[0]),
        float(action_low[0]),
        float(action_high[0]),
        28,
        349,
        410,
    )
    _gauge(
        panel,
        "omega_cmd",
        float(action[1]),
        float(action_low[1]),
        float(action_high[1]),
        500,
        349,
        410,
        centered=True,
    )
    _put_line(
        panel,
        "Q1={:+.3f}  Q2={:+.3f}  minQ={:+.3f}  alpha={:.4f}  latent_std=({:.3f},{:.3f})".format(
            diagnostics["q1"],
            diagnostics["q2"],
            diagnostics["q_min"],
            diagnostics["alpha"],
            diagnostics["latent_std"][0],
            diagnostics["latent_std"][1],
        ),
        28,
        397,
        0.51,
    )
    probe_header = (
        "CRITIC PROBES (supporting evidence; actor remains continuous)"
        if explanation is None
        else f"FOIL: {explanation.foil_label}  {explanation.separation_label}"
    )
    _put_line(
        panel,
        probe_header,
        28,
        425,
        0.47,
        (166, 185, 205),
    )

    values = np.asarray(diagnostics["probe_q"], dtype=float)
    value_min, value_max = float(values.min()), float(values.max())
    span = max(1e-9, value_max - value_min)
    best = int(np.argmax(values))
    foil = (
        None if explanation is None
        else int(explanation.foil_label.split("/", 1)[0])
    )
    bar_x, bar_width = 258, 620
    for row, (name, value) in enumerate(zip(probe_names, values)):
        y = 439 + row * 21
        marker = "F" if row == foil else (">" if row == best else " ")
        color = (255, 174, 44) if row == foil else ((56, 214, 104) if row == best else (64, 137, 204))
        _put_line(panel, f"{marker} {row}: {name:14s}", 28, y + 15, 0.43, thickness=2 if row in {best, foil} else 1)
        cv2.rectangle(panel, (bar_x, y), (bar_x + bar_width, y + 15), (31, 40, 51), -1)
        normalized = (float(value) - value_min) / span
        cv2.rectangle(panel, (bar_x, y), (bar_x + int(bar_width * normalized), y + 15), color, -1)
        _put_line(panel, f"{value:+.3f}", bar_x + bar_width - 72, y + 13, 0.39)
    return panel


def compose_frame(
    camera, bev, vantage, trajectory, dashboard, explanation_view=None
) -> np.ndarray:
    frame = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
    frame[:TOP_HEIGHT, 0:640] = add_title(
        letterbox(camera, 640, TOP_HEIGHT), "AGENT CAMERA"
    )
    frame[:TOP_HEIGHT, 640:1280] = add_title(
        letterbox(bev, 640, TOP_HEIGHT), "BEV / FULL MAP"
    )
    if explanation_view is None:
        frame[:TOP_HEIGHT, 1280:1920] = add_title(
            letterbox(vantage, 640, TOP_HEIGHT), "VANTAGE / FULL LOOP"
        )
    else:
        frame[:TOP_HEIGHT, 1280:1920] = letterbox(
            explanation_view, 640, TOP_HEIGHT
        )
    frame[TOP_HEIGHT:, :960] = letterbox(trajectory, 960, BOTTOM_HEIGHT)
    frame[TOP_HEIGHT:, 960:] = letterbox(dashboard, 960, BOTTOM_HEIGHT)
    cv2.line(frame, (640, 0), (640, TOP_HEIGHT), (75, 84, 94), 2)
    cv2.line(frame, (1280, 0), (1280, TOP_HEIGHT), (75, 84, 94), 2)
    cv2.line(frame, (960, TOP_HEIGHT), (960, HEIGHT), (75, 84, 94), 2)
    cv2.line(frame, (0, TOP_HEIGHT), (WIDTH, TOP_HEIGHT), (75, 84, 94), 2)
    return frame


def render_sac_multiview_video(
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
    replace_vantage_with_explanation: bool = False,
) -> None:
    warnings.filterwarnings("ignore")
    logging.getLogger().setLevel(logging.WARNING)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if repeat_duck:
        # Render-only task override: the trained actor is unchanged. Duckie is
        # re-armed after ego has left the crossing instead of oscillating back
        # immediately at the same encounter.
        config["duck_controller"]["max_crossings_per_episode"] = 0
        config["duck_controller"]["repeat_rearm_distance"] = float(
            repeat_rearm_distance
        )
    policy_repeat = int(config["environment"].get("frame_skip", 1))
    # Render individual physics ticks, but choose a new actor action only after
    # policy_repeat ticks. This reproduces the training action-hold semantics.
    config["environment"]["frame_skip"] = 1
    # StopTracker diperbarui sekali per env.step. Karena renderer memecah satu
    # macro-decision menjadi physics tick, jumlah hold juga harus dikalikan agar
    # dwell 3 decision saat training tetap 3*6/30 = 0.6 s di video.
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
    action_specs = build_action_table(ActionConfig(**config["actions"]))
    probe_names = [spec.name for spec in action_specs]
    probes = np.asarray([[spec.v, spec.omega] for spec in action_specs], dtype=np.float32)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(
        str(output_path),
        format="FFMPEG",
        mode="I",
        fps=fps,
        codec="libx264",
        quality=8,
        # 1920x1080 is divisible by 8 (but not 16); keep the exact dashboard
        # resolution instead of letting imageio pad the height to 1088.
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

    def write_frame(action, decision_observation, diagnostics, explanation) -> None:
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
        dashboard = sac_dashboard_panel(
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
            explanation=explanation,
        )
        explanation_view = None
        if replace_vantage_with_explanation:
            selected_label = (
                f"v={float(action[0]):+.3f}, "
                f"omega={float(action[1]):+.3f}"
            )
            explanation_view = explanation_view_panel(
                explanation,
                selected_label,
                "SOLVER: SAC / DETERMINISTIC ACTOR MEAN",
            )
        writer.append_data(compose_frame(
            camera, bev, vantage, trajectory_view, dashboard,
            explanation_view=explanation_view,
        ))
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
                canonical_action = CanonicalAction(
                    solver=SolverKind.SAC,
                    v_cmd=float(action[0]),
                    omega_cmd=float(action[1]),
                )
                explanation = sac_video_explanation(
                    canonical_from_continuous_state(env.current_state),
                    canonical_action,
                    probe_names,
                    probes,
                    diagnostics["q_min"],
                    diagnostics["probe_q"],
                )
                decisions += 1
                if episode_physics_steps == 0:
                    write_frame(action, decision_observation, diagnostics, explanation)
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
                        write_frame(action, decision_observation, diagnostics, explanation)
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
        description="Render synchronized multiview diagnostics for SAC."
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=10101)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--max-steps", type=int, default=1500)
    parser.add_argument(
        "--duration-seconds",
        type=float,
        default=0.0,
        help="Jika >0, reset setelah terminal dan terus render sampai durasi ini.",
    )
    parser.add_argument(
        "--repeat-duck",
        action="store_true",
        help="Izinkan Duckie menyeberang kembali setelah ego meninggalkan crossing.",
    )
    parser.add_argument(
        "--repeat-rearm-distance",
        type=float,
        default=1.0,
        help="Jarak ego dari crossing sebelum Duckie boleh menyeberang lagi.",
    )
    parser.add_argument(
        "--decorate-kfupm",
        action="store_true",
        help="Tambahkan logo KFUPM di tengah dan billboard JISR3 di kiri spawn.",
    )
    parser.add_argument(
        "--explanation-panel",
        action="store_true",
        help="Ganti vantage view dengan explanation keputusan saat ini.",
    )
    args = parser.parse_args()
    render_sac_multiview_video(
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
        args.explanation_panel,
    )


if __name__ == "__main__":
    main()
