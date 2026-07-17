"""Build compact README GIFs from the retained multiview MP4 artifacts.

The script intentionally labels every panel with its solver, task, and whether
teacher-guided exploration was used. It never implies that a missing ablation
cell (currently SARSA without teacher) was run.
"""

from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import imageio.v2 as imageio
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "assets" / "gifs"
GIF_FPS = 5
PANEL_WIDTH = 480
PANEL_HEIGHT = 270


SOURCES: Dict[str, Tuple[str, str, float, float]] = {
    "lane_q_teacher": (
        "docs/assets/videos/teacher/lane_q_learning.mp4",
        "Q-LEARNING | LANE | TEACHER-GUIDED",
        0.0,
        10.0,
    ),
    "lane_sarsa_teacher": (
        "docs/assets/videos/teacher/lane_sarsa.mp4",
        "SARSA | LANE | TEACHER-GUIDED",
        0.0,
        10.0,
    ),
    "full_q_teacher": (
        "docs/assets/videos/teacher/full_q_learning.mp4",
        "Q-LEARNING | STOP + DUCKIE | TEACHER-GUIDED",
        8.0,
        18.0,
    ),
    "full_sarsa_teacher": (
        "docs/assets/videos/teacher/full_sarsa.mp4",
        "SARSA | STOP + DUCKIE | TEACHER-GUIDED",
        8.0,
        18.0,
    ),
    "lane_q_no_teacher": (
        "docs/assets/videos/no_teacher/lane_q_learning.mp4",
        "Q-LEARNING | LANE | NO TEACHER",
        0.0,
        10.0,
    ),
    "full_q_no_teacher": (
        "docs/assets/videos/no_teacher/full_q_learning.mp4",
        "Q-LEARNING | STOP + DUCKIE | NO TEACHER",
        8.0,
        18.0,
    ),
}


def _letterbox(frame: np.ndarray) -> np.ndarray:
    height, width = frame.shape[:2]
    scale = min(PANEL_WIDTH / width, PANEL_HEIGHT / height)
    resized = cv2.resize(
        frame,
        (max(1, round(width * scale)), max(1, round(height * scale))),
        interpolation=cv2.INTER_AREA,
    )
    canvas = np.full((PANEL_HEIGHT, PANEL_WIDTH, 3), (10, 14, 19), dtype=np.uint8)
    x = (PANEL_WIDTH - resized.shape[1]) // 2
    y = (PANEL_HEIGHT - resized.shape[0]) // 2
    canvas[y : y + resized.shape[0], x : x + resized.shape[1]] = resized
    return canvas


def _title(frame: np.ndarray, label: str) -> np.ndarray:
    result = frame.copy()
    cv2.rectangle(result, (0, 0), (result.shape[1], 29), (5, 8, 12), -1)
    cv2.putText(
        result,
        label,
        (10, 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.43,
        (245, 248, 250),
        1,
        cv2.LINE_AA,
    )
    return result


def read_clip(relative: str, label: str, start: float, duration: float) -> List[np.ndarray]:
    source = ROOT / relative
    if not source.is_file():
        raise FileNotFoundError(source)
    capture = cv2.VideoCapture(str(source))
    source_fps = float(capture.get(cv2.CAP_PROP_FPS)) or 20.0
    start_frame = round(start * source_fps)
    end_frame = round((start + duration) * source_fps)
    stride = max(1, round(source_fps / GIF_FPS))
    capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    frames: List[np.ndarray] = []
    index = start_frame
    while index < end_frame:
        ok, frame = capture.read()
        if not ok:
            break
        if (index - start_frame) % stride == 0:
            rgb = cv2.cvtColor(_title(_letterbox(frame), label), cv2.COLOR_BGR2RGB)
            frames.append(rgb)
        index += 1
    capture.release()
    if not frames:
        raise RuntimeError(f"No frames decoded from {source}")
    return frames


def save_gif(path: Path, frames: List[np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(
        str(path),
        frames,
        format="GIF",
        duration=1000 // GIF_FPS,
        loop=0,
        palettesize=128,
        subrectangles=True,
    )
    print(f"gif={path.relative_to(ROOT)} frames={len(frames)}")


def side_by_side(left: List[np.ndarray], right: List[np.ndarray]) -> List[np.ndarray]:
    count = min(len(left), len(right))
    return [np.concatenate((left[i], right[i]), axis=1) for i in range(count)]


def main() -> None:
    clips = {
        key: read_clip(relative, label, start, duration)
        for key, (relative, label, start, duration) in SOURCES.items()
    }
    for key, frames in clips.items():
        save_gif(OUT / f"{key}.gif", frames)

    save_gif(
        OUT / "compare_lane_q_vs_sarsa_teacher.gif",
        side_by_side(clips["lane_q_teacher"], clips["lane_sarsa_teacher"]),
    )
    save_gif(
        OUT / "compare_full_q_vs_sarsa_teacher.gif",
        side_by_side(clips["full_q_teacher"], clips["full_sarsa_teacher"]),
    )
    save_gif(
        OUT / "compare_q_teacher_vs_no_teacher.gif",
        side_by_side(clips["full_q_teacher"], clips["full_q_no_teacher"]),
    )


if __name__ == "__main__":
    main()
