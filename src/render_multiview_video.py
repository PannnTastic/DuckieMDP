import argparse
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import cv2
import imageio.v2 as imageio
import numpy as np
import yaml

from .actions import build_action_table, ActionConfig
from .agents.factory import algorithm_name, build_tabular_agent
from .discretizer import discretize
from .decorations import attach_kfupm_small_loop_decorations
from .env_wrapper import build_env
from .explainability.schema import CanonicalAction, SolverKind
from .explainability.semantic_state import canonical_from_raw_state
from .explainability.video_overlay import q_video_explanation


WIDTH, HEIGHT = 1920, 1080
TOP_HEIGHT = 480
BOTTOM_HEIGHT = HEIGHT - TOP_HEIGHT
ACTION_NAMES = tuple(spec.name for spec in build_action_table(ActionConfig()))


def load_config(path: Path) -> Dict[str, Any]:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError(f"Invalid or empty configuration: {path}")
    return config


def letterbox(image: np.ndarray, width: int, height: int, color=(12, 15, 20)) -> np.ndarray:
    image = np.asarray(image, dtype=np.uint8)
    scale = min(width / image.shape[1], height / image.shape[0])
    resized = cv2.resize(
        image,
        (max(1, round(image.shape[1] * scale)), max(1, round(image.shape[0] * scale))),
        interpolation=cv2.INTER_AREA,
    )
    canvas = np.full((height, width, 3), color, dtype=np.uint8)
    x = (width - resized.shape[1]) // 2
    y = (height - resized.shape[0]) // 2
    canvas[y : y + resized.shape[0], x : x + resized.shape[1]] = resized
    return canvas


def add_title(panel: np.ndarray, title: str) -> np.ndarray:
    result = panel.copy()
    cv2.rectangle(result, (0, 0), (result.shape[1], 42), (8, 12, 18), -1)
    cv2.putText(result, title, (18, 29), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (245, 248, 250), 2, cv2.LINE_AA)
    return result


def capture_views(env) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    simulator = env.unwrapped
    camera = np.ascontiguousarray(simulator.render_obs())
    bev = np.ascontiguousarray(simulator.render(mode="top_down"))
    old_draw_bbox = simulator.draw_bbox
    try:
        simulator.draw_bbox = True
        vantage = np.ascontiguousarray(simulator.render(mode="rgb_array"))
    finally:
        simulator.draw_bbox = old_draw_bbox
    return camera, bev, vantage


def world_to_panel(
    x: float,
    z: float,
    map_width: float,
    map_height: float,
    width: int,
    height: int,
    margin: int = 58,
) -> Tuple[int, int]:
    px = margin + x / max(map_width, 1e-9) * (width - 2 * margin)
    py = height - margin - z / max(map_height, 1e-9) * (height - 2 * margin)
    return int(round(px)), int(round(py))


def trajectory_panel(
    trajectory: Sequence[Tuple[float, float]],
    position: Tuple[float, float],
    angle: float,
    grid_width: int,
    grid_height: int,
    tile_size: float,
    width: int = WIDTH // 2,
    height: int = BOTTOM_HEIGHT,
) -> np.ndarray:
    panel = np.full((height, width, 3), (11, 17, 24), dtype=np.uint8)
    map_width, map_height = grid_width * tile_size, grid_height * tile_size
    margin = 58
    for gx in range(grid_width + 1):
        x = gx * tile_size
        p0 = world_to_panel(x, 0, map_width, map_height, width, height, margin)
        p1 = world_to_panel(x, map_height, map_width, map_height, width, height, margin)
        cv2.line(panel, p0, p1, (38, 49, 61), 1, cv2.LINE_AA)
    for gz in range(grid_height + 1):
        z = gz * tile_size
        p0 = world_to_panel(0, z, map_width, map_height, width, height, margin)
        p1 = world_to_panel(map_width, z, map_width, map_height, width, height, margin)
        cv2.line(panel, p0, p1, (38, 49, 61), 1, cv2.LINE_AA)
    if len(trajectory) >= 2:
        points = np.asarray(
            [world_to_panel(x, z, map_width, map_height, width, height, margin) for x, z in trajectory],
            dtype=np.int32,
        )
        cv2.polylines(panel, [points], False, (52, 211, 235), 4, cv2.LINE_AA)
    if trajectory:
        start = world_to_panel(*trajectory[0], map_width, map_height, width, height, margin)
        cv2.circle(panel, start, 8, (48, 209, 88), -1, cv2.LINE_AA)
    current = world_to_panel(*position, map_width, map_height, width, height, margin)
    cv2.circle(panel, current, 10, (255, 174, 44), -1, cv2.LINE_AA)
    heading_length = 0.18
    hx = position[0] + heading_length * np.cos(angle)
    hz = position[1] - heading_length * np.sin(angle)
    heading = world_to_panel(hx, hz, map_width, map_height, width, height, margin)
    cv2.arrowedLine(panel, current, heading, (255, 245, 160), 4, cv2.LINE_AA, tipLength=0.35)
    cv2.putText(panel, "z", (28, 66), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (160, 176, 192), 1, cv2.LINE_AA)
    cv2.putText(panel, "x", (width - 45, height - 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (160, 176, 192), 1, cv2.LINE_AA)
    return add_title(panel, "TRAJECTORY / WORLD FRAME")


def _put_line(panel: np.ndarray, text: str, x: int, y: int, scale=0.58, color=(224, 231, 239), thickness=1):
    cv2.putText(panel, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def _wrapped_lines(text: str, limit: int = 54) -> List[str]:
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
    return lines or [""]


def explanation_view_panel(
    explanation,
    selected_label: str,
    solver_label: str,
    width: int = 640,
    height: int = TOP_HEIGHT,
) -> np.ndarray:
    """Render a live per-state explanation in place of the vantage view."""
    panel = np.full((height, width, 3), (12, 17, 24), dtype=np.uint8)
    panel = add_title(panel, "EXPLANATION / CURRENT DECISION")
    if explanation is None:
        _put_line(
            panel, "No explanation available for this frame.", 24, 92,
            0.58, (190, 201, 214),
        )
        return panel

    accent = (80, 95, 255) if explanation.undesirable else (72, 230, 120)
    _put_line(panel, solver_label, 24, 70, 0.49, (166, 185, 205), 1)
    _put_line(
        panel, f"PRIMITIVE: {explanation.primitive}", 24, 105,
        0.65, accent, 2,
    )
    _put_line(panel, "SELECTED ACTION", 24, 143, 0.43, (242, 198, 79), 1)
    _put_line(panel, selected_label, 24, 170, 0.58, (245, 248, 250), 2)

    _put_line(panel, "WHY / TRIGGER", 24, 207, 0.43, (242, 198, 79), 1)
    for index, line in enumerate(_wrapped_lines(explanation.trigger)):
        _put_line(panel, line, 24, 234 + index * 25, 0.48)

    contrast_y = 294
    _put_line(panel, "CONTRASTIVE FOIL", 24, contrast_y, 0.43, (242, 198, 79), 1)
    _put_line(
        panel, explanation.foil_label, 24, contrast_y + 27,
        0.53, (224, 231, 239), 2,
    )
    _put_line(
        panel, explanation.separation_label, 24, contrast_y + 56,
        0.49, (52, 196, 235), 1,
    )
    _put_line(
        panel, f"RULE: {explanation.rule_id}", 24, contrast_y + 88,
        0.44, (166, 185, 205), 1,
    )
    caveat_lines = _wrapped_lines(explanation.caveat, limit=60)[:2]
    _put_line(panel, "EVIDENCE SCOPE", 24, 414, 0.40, (242, 198, 79), 1)
    for index, line in enumerate(caveat_lines):

        _put_line(panel, line, 24, 438 + index * 21, 0.40, (166, 185, 205), 1)
    return panel

def dashboard_panel(
    raw_state,
    discrete_state: Tuple[int, ...],
    action: int,
    q_values: np.ndarray,
    allowed_actions: Iterable[int],
    position: Tuple[float, float],
    angle: float,
    decision: int,
    physics_step: int,
    cumulative_reward: float,
    termination_reason: str,
    teacher_active: bool = False,
    explanation=None,
    width: int = WIDTH // 2,
    height: int = BOTTOM_HEIGHT,
) -> np.ndarray:
    panel = np.full((height, width, 3), (13, 18, 25), dtype=np.uint8)
    panel = add_title(panel, "STATE, ACTION & Q VALUES")
    if explanation is not None:
        color = (80, 95, 255) if explanation.undesirable else (72, 230, 120)
        _put_line(
            panel, f"PRIMITIVE: {explanation.primitive}", 330, 29,
            0.47, color, 2,
        )
        _put_line(
            panel, f"trigger: {explanation.trigger}", 28, 250,
            0.43, (242, 198, 79),
        )
    tracking_error = raw_state.phi + raw_state.d
    tile_name = getattr(raw_state.tile, "name", str(raw_state.tile))
    duck_name = getattr(raw_state.duck, "name", str(raw_state.duck))
    lines_left = [
        f"position (x,z) : ({position[0]:+.3f}, {position[1]:+.3f}) m",
        f"heading psi     : {np.degrees(angle):+.2f} deg",
        f"lateral d      : {raw_state.d:+.4f} m",
        f"heading phi    : {raw_state.phi:+.4f} rad",
        f"tracking e     : {tracking_error:+.4f}",
        f"speed v        : {raw_state.v:.4f} m/s",
    ]
    lines_right = [
        f"curvature      : {tile_name}",
        f"stop distance  : {raw_state.d_stop}",
        f"stop satisfied : {raw_state.sigma_stop}",
        f"duck threat    : {duck_name}",
        f"state bins     : {discrete_state}",
        f"teacher active : {teacher_active}",
    ]
    for index, line in enumerate(lines_left):
        _put_line(panel, line, 28, 76 + index * 31)
    for index, line in enumerate(lines_right):
        _put_line(panel, line, 492, 76 + index * 31)

    _put_line(panel, f"decision={decision}   physics_step={physics_step}   cumulative_reward={cumulative_reward:+.3f}", 28, 278, 0.61)
    status_color = (65, 220, 110) if termination_reason in {"in_progress", "timeout"} else (255, 92, 92)
    _put_line(panel, f"status: {termination_reason}", 28, 311, 0.64, status_color, 2)

    allowed = tuple(int(a) for a in allowed_actions)
    finite_values = np.asarray([q_values[a] for a in allowed], dtype=float)
    greedy_action = allowed[int(np.argmax(finite_values))]
    q_min, q_max = float(np.min(finite_values)), float(np.max(finite_values))
    span = max(1e-9, q_max - q_min)
    bar_x, bar_width = 275, 610
    for row, action_id in enumerate(allowed):
        y = 337 + row * 34
        active = action_id == action
        recommended = action_id == greedy_action
        color = (56, 214, 104) if recommended else ((255, 174, 44) if active else (64, 137, 204))
        name = ACTION_NAMES[action_id]
        marker = ">" if recommended else ("*" if active else " ")
        _put_line(
            panel, f"{marker} {action_id}: {name:14s}", 28, y + 18, 0.52,
            (250, 250, 250) if (active or recommended) else (190, 201, 214),
            2 if (active or recommended) else 1,
        )
        normalized = (float(q_values[action_id]) - q_min) / span
        cv2.rectangle(panel, (bar_x, y), (bar_x + bar_width, y + 22), (31, 40, 51), -1)
        cv2.rectangle(panel, (bar_x, y), (bar_x + int(bar_width * normalized), y + 22), color, -1)
        _put_line(panel, f"{float(q_values[action_id]):+.4f}", bar_x + bar_width - 92, y + 17, 0.47, (246, 248, 250))
    _put_line(
        panel,
        (
            f"ACTIVE: {action}/{ACTION_NAMES[action]}  "
            f"GREEDY NEXT: {greedy_action}/{ACTION_NAMES[greedy_action]}"
            + ("" if explanation is None else
               f"  FOIL: {explanation.foil_label}  {explanation.separation_label}")
        ),
        28,
        height - 24,
        0.45,
        (72, 230, 120),
        2,
    )
    return panel


def compose_frame(
    camera: np.ndarray,
    bev: np.ndarray,
    vantage: np.ndarray,
    trajectory: np.ndarray,
    dashboard: np.ndarray,
    explanation_view: np.ndarray = None,
) -> np.ndarray:
    frame = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
    frame[:TOP_HEIGHT, 0:640] = add_title(letterbox(camera, 640, TOP_HEIGHT), "AGENT CAMERA")
    frame[:TOP_HEIGHT, 640:1280] = add_title(letterbox(bev, 640, TOP_HEIGHT), "BEV / FULL MAP")
    if explanation_view is None:
        frame[:TOP_HEIGHT, 1280:1920] = add_title(
            letterbox(vantage, 640, TOP_HEIGHT), "VANTAGE / LOCAL OVERHEAD"
        )
    else:
        frame[:TOP_HEIGHT, 1280:1920] = letterbox(explanation_view, 640, TOP_HEIGHT)
    frame[TOP_HEIGHT:, :960] = letterbox(trajectory, 960, BOTTOM_HEIGHT)
    frame[TOP_HEIGHT:, 960:] = letterbox(dashboard, 960, BOTTOM_HEIGHT)
    cv2.line(frame, (640, 0), (640, TOP_HEIGHT), (75, 84, 94), 2)
    cv2.line(frame, (1280, 0), (1280, TOP_HEIGHT), (75, 84, 94), 2)
    cv2.line(frame, (960, TOP_HEIGHT), (960, HEIGHT), (75, 84, 94), 2)
    cv2.line(frame, (0, TOP_HEIGHT), (WIDTH, TOP_HEIGHT), (75, 84, 94), 2)
    return frame


def render_multiview_video(
    config_path: Path,
    q_table_path: Path,
    output_path: Path,
    seed: int = 101,
    fps: int = 20,
    max_steps: int = 1500,
    decorate_kfupm: bool = False,
    replace_vantage_with_explanation: bool = False,
) -> None:
    config = load_config(config_path)
    algorithm = algorithm_name(config)
    solver_kind = (
        SolverKind.SARSA
        if algorithm == "sarsa"
        else SolverKind.Q_LEARNING
    )
    policy_repeat = int(config["environment"].get("frame_skip", 1))
    config["environment"]["frame_skip"] = 1
    if max_steps > 0:
        config["environment"]["max_steps"] = max_steps
    env = build_env(config, seed)
    if decorate_kfupm:
        if str(config["environment"].get("map_name", "")) != "small_loop":
            raise ValueError("KFUPM decoration layout is defined only for small_loop")
        asset_dir = Path(__file__).resolve().parents[1] / "assets"
        attach_kfupm_small_loop_decorations(env, asset_dir)
    agent = build_tabular_agent(config, seed)
    agent.load(q_table_path)
    allowed_actions = tuple(int(action) for action in agent.allowed_actions)
    action_specs = build_action_table(
        ActionConfig(**config.get("actions", {}))
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(
        str(output_path), format="FFMPEG", mode="I", fps=fps, codec="libx264",
        quality=7,
        macro_block_size=8,
        ffmpeg_log_level="warning",
        ffmpeg_params=["-g", "1", "-bf", "0", "-threads", "1"],
    )

    state = env.reset(seed)
    trajectory: List[Tuple[float, float]] = [(float(env.unwrapped.cur_pos[0]), float(env.unwrapped.cur_pos[2]))]
    video_clock = 0.0
    physics_steps = decisions = frames = 0
    cumulative_reward = 0.0
    termination_reason = "in_progress"

    def explain_decision(raw_state, action_id):
        state_index = discretize(raw_state)
        spec = action_specs[action_id]
        canonical_action = CanonicalAction(
            solver=solver_kind,
            action_id=action_id,
            action_name=spec.name,
            v_cmd=float(spec.v),
            omega_cmd=float(spec.omega),
        )
        return q_video_explanation(
            canonical_from_raw_state(raw_state),
            canonical_action,
            agent.q[state_index],
            allowed_actions,
            action_specs,
        )

    def write_frame(action: int, explanation) -> None:
        nonlocal frames
        simulator = env.unwrapped
        camera, bev, vantage = capture_views(env)
        position = (float(simulator.cur_pos[0]), float(simulator.cur_pos[2]))
        state_index = discretize(state)
        trajectory_view = trajectory_panel(
            trajectory, position, float(simulator.cur_angle), simulator.grid_width,
            simulator.grid_height, float(simulator.road_tile_size)
        )
        dashboard = dashboard_panel(
            state, state_index, action, agent.q[state_index], allowed_actions,
            position, float(simulator.cur_angle), decisions, physics_steps,
            cumulative_reward, termination_reason,
            explanation=explanation,
        )
        teacher_trained = bool(config.get("lane_teacher", {}).get("enabled", False))
        solver_label = f"SOLVER: {algorithm.upper()}"
        if teacher_trained:
            solver_label += " / TEACHER-TRAINED"
        cv2.putText(
            dashboard,
            solver_label,
            (620 if teacher_trained else 735, 29),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (245, 248, 250),
            2,
            cv2.LINE_AA,
        )
        explanation_view = None
        if replace_vantage_with_explanation:
            spec = action_specs[action]
            selected_label = (
                f"{action}/{spec.name}  "
                f"(v={float(spec.v):+.2f}, omega={float(spec.omega):+.2f})"
            )
            explanation_view = explanation_view_panel(
                explanation,
                selected_label,
                solver_label + " / GREEDY EVALUATION",
            )
        writer.append_data(
            compose_frame(
                camera, bev, vantage, trajectory_view, dashboard,
                explanation_view=explanation_view,
            )
        )
        frames += 1

    try:
        initial_action = agent.select_action(discretize(state), greedy=True)
        explanation = explain_decision(state, initial_action)
        write_frame(initial_action, explanation)
        done = False
        while not done and (max_steps <= 0 or physics_steps < max_steps):
            action = agent.select_action(discretize(state), greedy=True)
            explanation = explain_decision(state, action)
            decisions += 1
            for _ in range(policy_repeat):
                if done or (max_steps > 0 and physics_steps >= max_steps):
                    break
                state, reward, done, info = env.step(action)
                physics_steps += 1
                cumulative_reward += float(reward)
                termination_reason = info["termination_reason"]
                trajectory.append((float(env.unwrapped.cur_pos[0]), float(env.unwrapped.cur_pos[2])))
                video_clock += fps * float(env.unwrapped.delta_time)
                if video_clock >= 1.0:
                    write_frame(action, explanation)
                    video_clock -= 1.0
    finally:
        writer.close()
        env.close()
    print(f"video={output_path.resolve()}")
    print(
        f"resolution={WIDTH}x{HEIGHT} fps={fps} frames={frames} decisions={decisions} "
        f"physics_steps={physics_steps} status={termination_reason}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Render synchronized Duckietown policy diagnostics.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--q-table", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=101)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--max-steps", type=int, default=1500)
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
    render_multiview_video(
        args.config,
        args.q_table,
        args.output,
        args.seed,
        args.fps,
        args.max_steps,
        args.decorate_kfupm,
        args.explanation_panel,
    )


if __name__ == "__main__":
    main()
