"""Definisi action space MDP.

Duckiebot adalah differential-drive, sehingga action fisiknya adalah
command kecepatan linear dan angular, a_t=(v_cmd, omega_cmd), bukan sudut
steering Ackermann. Untuk Q-learning pasangan kontinu ini dijadikan tujuh
macro-actions diskrit.
"""

from dataclasses import dataclass
from typing import Tuple
import numpy as np


@dataclass(frozen=True)
class ActionConfig:
    v_fast: float = 0.40
    v_slow: float = 0.15
    w0: float = 1.50
    wheel_base: float = 0.102

@dataclass(frozen=True)
class ActionSpec:
    name: str
    v: float
    omega: float

def build_action_table(cfg: ActionConfig) -> Tuple[ActionSpec, ...]:
    """Membangun A={fast/slow x left/straight/right, brake}.

    Pada lane-following, konfigurasi allowed_actions memask brake agar agent
    tidak menemukan solusi palsu berupa diam. Brake diaktifkan kembali pada
    task stop-sign dan pedestrian.
    """
    return (
        ActionSpec("fast_left", cfg.v_fast, cfg.w0),
        ActionSpec("fast_straight", cfg.v_fast, 0.0),
        ActionSpec("fast_right", cfg.v_fast, -cfg.w0),
        ActionSpec("slow_left", cfg.v_slow, cfg.w0),
        ActionSpec("slow_straight", cfg.v_slow, 0.0),
        ActionSpec("slow_right", cfg.v_slow, -cfg.w0),
        ActionSpec("brake", 0.0, 0.0),
    )


ACTION_TABLE = build_action_table(ActionConfig())


def vw_to_wheels(v: float, omega: float, wheel_base: float) -> np.ndarray:
    """Kinematika inverse differential drive.

    u_L = v - L*omega/2 dan u_R = v + L*omega/2. Hasilnya adalah normalized
    wheel commands yang diterima simulator.
    """
    left = v - 0.5 * wheel_base * omega
    right = v + 0.5 * wheel_base * omega
    return np.clip(np.array([left, right], dtype=np.float32), -1.0, 1.0)


def action_to_wheels(action_id: int, cfg: ActionConfig = ActionConfig()) -> np.ndarray:
    table = build_action_table(cfg)
    if not 0 <= int(action_id) < len(table):
        raise ValueError("action_id must be 0..6")
    action = table[int(action_id)]
    return vw_to_wheels(action.v, action.omega, cfg.wheel_base)
