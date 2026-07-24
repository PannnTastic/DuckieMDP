"""Render SAC with real M1--M13 primitive evidence and colored trajectory."""

import argparse
from collections import Counter
import json
import logging
from pathlib import Path
from typing import Dict, List, Sequence, Tuple
import warnings

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
    world_to_panel,
)
from .render_sac_multiview_video import capture_views, critic_diagnostics


# OpenCV uses BGR colors.
PRIMITIVE_COLORS = {
    "LaneKeeping": (235, 211, 52),
    "CurveNegotiation": (235, 153, 72),
    "StopCompliance": (82, 82, 242),
    "PedestrianYield": (103, 210, 102),
}

# 2x2 layout: four equal quadrants on the 1920x1080 canvas.
QUAD_W, QUAD_H = WIDTH // 2, HEIGHT // 2  # 960 x 540


def primitive_from_state(state) -> str:
    if bool(state.stop_present) and state.d_stop is not None:
        return "StopCompliance"
    if bool(state.duck_present) or bool(state.duck_active):
        return "PedestrianYield"
    if abs(float(state.kappa)) >= 0.8:
        return "CurveNegotiation"
    return "LaneKeeping"


def _wrapped(text: str, limit: int = 65, maximum: int = 2) -> List[str]:
    words = str(text).split()
    lines: List[str] = []
    current = ""
    for word in words:
        candidate = word if not current else current + " " + word
        if len(candidate) <= limit:
            current = candidate
        else:
            if current:
                lines.append(current)
            current = word
    if current:
        lines.append(current)
    if len(lines) > maximum:
        lines = lines[:maximum]
        lines[-1] = lines[-1].rstrip(".") + "..."
    return lines or ["n/a"]


def evidence_panel(
    primitive: str,
    explanation,
    selected_label: str,
    card: Dict,
    width: int = QUAD_W,
    height: int = QUAD_H,
) -> np.ndarray:
    panel = np.full((height, width, 3), (12, 17, 24), dtype=np.uint8)
    panel = add_title(panel, "DRIVING PRIMITIVE + COUNTERFACTUAL")
    color = PRIMITIVE_COLORS[primitive]
    _put_line(panel, "SAC / deterministic actor mean", 24, 74, 0.46, (170, 187, 204))
    _put_line(panel, primitive, 24, 112, 0.80, color, 2)
    _put_line(panel, "SELECTED", 24, 144, 0.42, (242, 198, 79))
    _put_line(panel, selected_label, 150, 144, 0.48, (245, 248, 250), 2)
    _put_line(panel, "FOIL", 24, 172, 0.42, (242, 198, 79))
    _put_line(
        panel,
        "%s | %s" % (explanation.foil_label, explanation.separation_label),
        96,
        172,
        0.44,
        (52, 196, 235),
    )

    _put_line(panel, "WHY", 24, 204, 0.46, (242, 198, 79), 2)
    why = "Live: %s | M1-M13: %s" % (explanation.trigger, card["why"])
    for index, line in enumerate(_wrapped(why, 100, 2)):
        _put_line(panel, line, 24, 228 + index * 22, 0.42)

    _put_line(panel, "WHAT IF / PAIRED OUTCOME", 24, 288, 0.46, (242, 198, 79), 2)
    for index, line in enumerate(_wrapped(card["what_if"], 100, 2)):
        _put_line(panel, line, 24, 312 + index * 22, 0.42)

    _put_line(panel, "VERIFICATION", 24, 372, 0.46, (242, 198, 79), 2)
    for index, line in enumerate(_wrapped(card["verification"], 100, 2)):
        _put_line(panel, line, 24, 396 + index * 22, 0.42)

    _put_line(panel, "TEMPORAL ARC", 24, 456, 0.46, (242, 198, 79), 2)
    for index, line in enumerate(_wrapped(card["temporal"], 100, 2)):
        _put_line(panel, line, 24, 480 + index * 22, 0.42)
    source = card["representative"]
    _put_line(
        panel,
        "REAL EVIDENCE: %s | %s | step %d"
        % (
            source["instance_id"],
            source["solver"],
            source["middle_step"],
        ),
        24,
        528,
        0.36,
        (153, 170, 188),
    )
    return panel


def trajectory_primitive_panel(
    history: Sequence[Tuple[float, float, str]],
    position: Tuple[float, float],
    angle: float,
    active_primitive: str,
    grid_width: int,
    grid_height: int,
    tile_size: float,
    width: int = QUAD_W,
    height: int = QUAD_H,
) -> np.ndarray:
    panel = np.full((height, width, 3), (11, 17, 24), dtype=np.uint8)
    map_width = grid_width * tile_size
    map_height = grid_height * tile_size
    margin = 62
    for gx in range(grid_width + 1):
        x = gx * tile_size
        cv2.line(
            panel,
            world_to_panel(x, 0, map_width, map_height, width, height, margin),
            world_to_panel(
                x, map_height, map_width, map_height, width, height, margin
            ),
            (38, 49, 61),
            1,
            cv2.LINE_AA,
        )
    for gz in range(grid_height + 1):
        z = gz * tile_size
        cv2.line(
            panel,
            world_to_panel(0, z, map_width, map_height, width, height, margin),
            world_to_panel(
                map_width, z, map_width, map_height, width, height, margin
            ),
            (38, 49, 61),
            1,
            cv2.LINE_AA,
        )
    for previous, current in zip(history, history[1:]):
        p0 = world_to_panel(
            previous[0], previous[1],
            map_width, map_height, width, height, margin,
        )
        p1 = world_to_panel(
            current[0], current[1],
            map_width, map_height, width, height, margin,
        )
        cv2.line(
            panel,
            p0,
            p1,
            PRIMITIVE_COLORS[current[2]],
            7,
            cv2.LINE_AA,
        )
    if history:
        start = world_to_panel(
            history[0][0], history[0][1],
            map_width, map_height, width, height, margin,
        )
        cv2.circle(panel, start, 9, (245, 245, 245), -1, cv2.LINE_AA)
    current = world_to_panel(
        position[0], position[1],
        map_width, map_height, width, height, margin,
    )
    cv2.circle(
        panel, current, 12, PRIMITIVE_COLORS[active_primitive],
        -1, cv2.LINE_AA,
    )
    heading_length = 0.18
    heading = world_to_panel(
        position[0] + heading_length * np.cos(angle),
        position[1] - heading_length * np.sin(angle),
        map_width,
        map_height,
        width,
        height,
        margin,
    )
    cv2.arrowedLine(
        panel, current, heading, (255, 245, 160),
        4, cv2.LINE_AA, tipLength=0.35,
    )
    panel = add_title(panel, "TRAJECTORY / DRIVING PRIMITIVES")
    for (primitive, _), lx in zip(PRIMITIVE_COLORS.items(), (18, 258, 498, 738)):
        cv2.line(
            panel, (lx, 62), (lx + 24, 62),
            PRIMITIVE_COLORS[primitive], 6, cv2.LINE_AA,
        )
        _put_line(panel, primitive, lx + 30, 68, 0.40, (230, 235, 240))
    _put_line(
        panel,
        "ACTIVE: " + active_primitive,
        18,
        100,
        0.58,
        PRIMITIVE_COLORS[active_primitive],
        2,
    )
    return panel


def compose_frame(camera, bev, explanation, trajectory) -> np.ndarray:
    # 2x2 grid: camera | BEV on top, trajectory | primitive-evidence on bottom.
    frame = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
    frame[:QUAD_H, :QUAD_W] = add_title(
        letterbox(camera, QUAD_W, QUAD_H), "AGENT CAMERA"
    )
    frame[:QUAD_H, QUAD_W:] = add_title(
        letterbox(bev, QUAD_W, QUAD_H), "BEV / FULL MAP"
    )
    frame[QUAD_H:, :QUAD_W] = trajectory
    frame[QUAD_H:, QUAD_W:] = explanation
    cv2.line(frame, (QUAD_W, 0), (QUAD_W, HEIGHT), (75, 84, 94), 2)
    cv2.line(frame, (0, QUAD_H), (WIDTH, QUAD_H), (75, 84, 94), 2)
    return frame


def render_video(
    config_path: Path,
    checkpoint_path: Path,
    evidence_path: Path,
    output_path: Path,
    seed: int = 20202,
    fps: int = 20,
    duration_seconds: float = 45.0,
    repeat_duck: bool = True,
    repeat_rearm_distance: float = 1.0,
    decorate_kfupm: bool = True,
) -> None:
    warnings.filterwarnings("ignore")
    logging.getLogger().setLevel(logging.WARNING)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))[
        "primitives"
    ]
    policy_repeat = int(config["environment"].get("frame_skip", 1))
    config["environment"]["frame_skip"] = 1
    config["environment"]["render_observations"] = True
    config["state"]["stop_hold_steps"] = (
        int(config["state"].get("stop_hold_steps", 1)) * policy_repeat
    )
    if repeat_duck:
        config["duck_controller"]["max_crossings_per_episode"] = 0
        config["duck_controller"]["repeat_rearm_distance"] = float(
            repeat_rearm_distance
        )
    device = str(config["training"].get("device", "cpu"))
    if device.startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"

    env = build_continuous_env(config, seed)
    if decorate_kfupm:
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
    probes = np.asarray(
        [[spec.v, spec.omega] for spec in action_specs],
        dtype=np.float32,
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
    target_frames = int(round(duration_seconds * fps))
    frame_count = 0
    video_clock = 0.0
    physics_steps = 0
    episodes = 0
    history: List[Tuple[float, float, str]] = []
    primitive_frame_counts = Counter()

    def write_frame(action, diagnostics, explanation, primitive):
        nonlocal frame_count
        primitive_frame_counts[primitive] += 1
        camera, bev, _ = capture_views(env)
        position = (
            float(simulator.cur_pos[0]),
            float(simulator.cur_pos[2]),
        )
        selected = "v=%+.3f, omega=%+.3f" % (
            float(action[0]), float(action[1])
        )
        panel = evidence_panel(
            primitive,
            explanation,
            selected,
            evidence[primitive],
        )
        trajectory = trajectory_primitive_panel(
            history,
            position,
            float(simulator.cur_angle),
            primitive,
            simulator.grid_width,
            simulator.grid_height,
            float(simulator.road_tile_size),
        )
        writer.append_data(compose_frame(camera, bev, panel, trajectory))
        frame_count += 1

    try:
        while frame_count < target_frames:
            observation = env.reset(seed + episodes)
            episodes += 1
            state = env.current_state
            primitive = primitive_from_state(state)
            history = [(
                float(simulator.cur_pos[0]),
                float(simulator.cur_pos[2]),
                primitive,
            )]
            done = False
            while not done and frame_count < target_frames:
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
                canonical_state = canonical_from_continuous_state(
                    env.current_state
                )
                primitive = primitive_from_state(env.current_state)
                explanation = sac_video_explanation(
                    canonical_state,
                    canonical_action,
                    probe_names,
                    probes,
                    diagnostics["q_min"],
                    diagnostics["probe_q"],
                )
                if physics_steps == 0:
                    write_frame(
                        action, diagnostics, explanation, primitive
                    )
                for _ in range(policy_repeat):
                    if done or frame_count >= target_frames:
                        break
                    observation, _, done, _ = env.step(action)
                    physics_steps += 1
                    primitive = primitive_from_state(env.current_state)
                    history.append((
                        float(simulator.cur_pos[0]),
                        float(simulator.cur_pos[2]),
                        primitive,
                    ))
                    video_clock += fps * float(simulator.delta_time)
                    while video_clock >= 1.0 and frame_count < target_frames:
                        write_frame(
                            action, diagnostics, explanation, primitive
                        )
                        video_clock -= 1.0
    finally:
        writer.close()
        env.close()
    print("primitive_frame_counts=%s" % dict(primitive_frame_counts))
    print(
        "video=%s frames=%d fps=%d duration=%.1fs episodes=%d"
        % (
            output_path.resolve(),
            frame_count,
            fps,
            frame_count / float(fps),
            episodes,
        )
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--evidence",
        type=Path,
        default=Path(
            "runs/explanations/cedp_v2_4policy/primitive_real_evidence.json"
        ),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20202)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--duration-seconds", type=float, default=45.0)
    parser.add_argument("--no-repeat-duck", action="store_true")
    parser.add_argument("--repeat-rearm-distance", type=float, default=1.0)
    parser.add_argument("--no-decorations", action="store_true")
    args = parser.parse_args()
    render_video(
        args.config,
        args.checkpoint,
        args.evidence,
        args.output,
        args.seed,
        args.fps,
        args.duration_seconds,
        not args.no_repeat_duck,
        args.repeat_rearm_distance,
        not args.no_decorations,
    )


if __name__ == "__main__":
    main()
