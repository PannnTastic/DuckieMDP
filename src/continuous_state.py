"""Privileged continuous state untuk SAC tanpa mengubah RawState tabular.

Vektor observation:

    x = [d, phi, v, kappa,
         stop_present, d_stop, sigma_stop,
         duck_present, duck_long, duck_lat,
         duck_v_long_rel, duck_v_lat_rel,
         duck_active, duck_crossing_available,
         stop_hold_progress]

Semua nilai yang masuk network dinormalisasi ke bounds tetap.
"""

from dataclasses import asdict, dataclass
from math import atan2, pi
from typing import Any, Dict, Optional, Tuple

import numpy as np
from gym import spaces
from gym_duckietown.simulator import bezier_point, bezier_tangent

from .state import RawState, StateConfig

OBSERVATION_NAMES = (
    "d",
    "phi",
    "v",
    "kappa",
    "stop_present",
    "d_stop",
    "sigma_stop",
    "duck_present",
    "duck_longitudinal",
    "duck_lateral",
    "duck_v_longitudinal_relative",
    "duck_v_lateral_relative",
    "duck_active",
    "duck_crossing_available",
    # Append-only agar checkpoint 14-D dapat dimigrasikan tanpa menggeser
    # arti seluruh fitur lama.
    "stop_hold_progress",
)

@dataclass(frozen=True)
class ContinuousStateConfig:
    max_speed: float = 0.41
    max_abs_curvature: float = 8.0
    max_stop_distance: float = 3.0
    max_duck_distance: float = 2.0
    max_relative_speed: float = 0.50
    curvature_samples: int = 33
    # Gerbang deteksi Duckie opsional. Default None mempertahankan perilaku
    # lama (Duckie selalu terlihat) demi kompatibilitas checkpoint SAC.
    # Diisi agar paritas informasi dengan classify_duck tabular: hanya
    # Duckie di koridor depan dan di dalam rentang yang dilaporkan.
    duck_detection_range: Optional[float] = None
    duck_detection_corridor_width: Optional[float] = None
    duck_detection_forward_only: bool = False

@dataclass(frozen=True)
class DuckRelativeState:
    present: bool = False
    longitudinal: float = 0.0
    lateral: float = 0.0
    v_longitudinal_relative: float = 0.0
    v_lateral_relative: float = 0.0
    active: bool = False
    crossing_available: bool = False

@dataclass(frozen=True)
class ContinuousState:
    d: float
    phi:float
    v: float
    kappa: float
    stop_present: bool
    d_stop: Optional[float]
    sigma_stop: bool
    duck_present: bool
    duck_longitudinal: float
    duck_lateral: float
    duck_v_longitudinal_relative: float
    duck_v_lateral_relative: float
    duck_active: bool
    duck_crossing_available: bool
    stop_hold_progress: float = 0.0

def _base(env: Any) -> Any:
    return getattr(env, "unwrapped", env)

def _kind(value:Any)->str:
    return str(getattr(value, "value", value)).lower()

def _normalize(vector: np.ndarray) -> np.ndarray:
    vector = np.asarray(vector, dtype=float)
    norm = float(np.linalg.norm(vector))
    return vector / norm if norm > 1e-12 else np.zeros_like(vector)

def _lane_frame(env: Any) -> Tuple[np.ndarray, np.ndarray]:
    env = _base(env)
    _, tangent = env.closest_curve_point(env.cur_pos, env.cur_angle)
    if tangent is None:
        forward = np.array(
            [np.cos(env.cur_angle), 0.0, -np.sin(env.cur_angle)], dtype=float
        )
    else:
        forward = _normalize(tangent)
    right = _normalize(np.cross(forward, np.array([0.0, 1.0, 0.0])))
    return forward, right

def _directed_curve(tile: Dict[str, Any], forward: np.ndarray) -> Optional[np.ndarray]:
    curves = tile.get("curves")
    if curves is None or len(curves) == 0:
        return None
    curves = np.asarray(curves, dtype=float)
    headings = curves[:, -1, :] - curves[:, 0, :]
    headings = np.asarray([_normalize(value) for value in headings])
    return curves[int(np.argmax(np.dot(headings, forward)))]


def curve_signed_curvature(
    curve: np.ndarray,
    samples: int = 33,
    straight_angle_threshold: float = 0.05,
) -> float:
    """Rata-rata signed curvature: perubahan heading dibagi arc length."""
    if samples < 3:
        raise ValueError("curvature_samples must be at least 3")
    curve = np.asarray(curve, dtype=float)
    tangent_before = _normalize(bezier_tangent(curve, 0.05))
    tangent_after = _normalize(bezier_tangent(curve, 0.95))
    cross_y = float(np.cross(tangent_before, tangent_after)[1])
    dot = float(np.clip(np.dot(tangent_before, tangent_after), -1.0, 1.0))
    heading_change = atan2(cross_y, dot)
    if abs(heading_change) <= straight_angle_threshold:
        return 0.0

    points = np.asarray(
        [bezier_point(curve, value) for value in np.linspace(0.0, 1.0, samples)]
    )
    arc_length = float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum())
    return heading_change / arc_length if arc_length > 1e-9 else 0.0


def signed_curvature_ahead(
    env: Any,
    state_cfg: StateConfig,
    continuous_cfg: ContinuousStateConfig,
) -> float:
    """Curvature pada directed lane di tile look-ahead ego."""
    env = _base(env)
    forward, _ = _lane_frame(env)
    probe = np.asarray(env.cur_pos, dtype=float) + state_cfg.tile_lookahead * forward
    tile = env._get_tile(*env.get_grid_coords(probe))
    if tile is None or not tile.get("drivable", False):
        tile = env._get_tile(*env.get_grid_coords(env.cur_pos))
    if tile is None or not tile.get("drivable", False):
        return 0.0
    curve = _directed_curve(tile, forward)
    if curve is None:
        return 0.0
    value = curve_signed_curvature(
        curve,
        samples=continuous_cfg.curvature_samples,
        straight_angle_threshold=state_cfg.curvature_threshold,
    )
    return float(
        np.clip(value, -continuous_cfg.max_abs_curvature, continuous_cfg.max_abs_curvature)
    )


def _crossing_available(controller: Any, duck: Any) -> bool:
    if controller is None:
        return True
    index = next(
        (i for i, candidate in enumerate(controller.ducks) if candidate is duck),
        None,
    )
    if index is None:
        return True
    if hasattr(controller, "crossing_available"):
        return bool(controller.crossing_available(index))
    limit = int(controller.cfg.max_crossings_per_episode)
    return limit <= 0 or int(controller.crossings_started[index]) < limit


def duck_relative_state(
    env: Any,
    ego_speed: float,
    controller: Any = None,
) -> DuckRelativeState:
    """Geometri Duckie terdekat dalam frame lane ego."""
    env = _base(env)
    candidates = [
        obj
        for obj in env.objects
        if getattr(obj, "visible", True) and _kind(getattr(obj, "kind", "")) == "duckie"
    ]
    if not candidates:
        return DuckRelativeState()

    ego_position = np.asarray(env.cur_pos, dtype=float)
    duck = min(
        candidates,
        key=lambda obj: float(
            np.linalg.norm(
                np.asarray(getattr(obj, "center", obj.pos), dtype=float) - ego_position
            )
        ),
    )
    forward, right = _lane_frame(env)
    duck_position = np.asarray(getattr(duck, "center", duck.pos), dtype=float)
    relative_position = duck_position - ego_position
    active = bool(getattr(duck, "pedestrian_active", False))

    duck_velocity = np.zeros(3, dtype=float)
    if active:
        duck_velocity = (
            np.asarray(getattr(duck, "heading", np.zeros(3)), dtype=float)
            * float(getattr(duck, "vel", 0.0))
        )
    ego_velocity = forward * float(ego_speed)
    relative_velocity = duck_velocity - ego_velocity

    return DuckRelativeState(
        present=True,
        longitudinal=float(np.dot(relative_position, forward)),
        lateral=float(np.dot(relative_position, right)),
        v_longitudinal_relative=float(np.dot(relative_velocity, forward)),
        v_lateral_relative=float(np.dot(relative_velocity, right)),
        active=active,
        crossing_available=_crossing_available(controller, duck),
    )


def gate_duck_visibility(
    duck: DuckRelativeState,
    cfg: ContinuousStateConfig,
) -> DuckRelativeState:
    """Terapkan gerbang deteksi opsional pada geometri Duckie.

    Tanpa gerbang, Duckie terdekat selalu dilaporkan berapa pun jaraknya,
    sehingga policy kontinu menerima informasi yang tidak tersedia bagi
    solver tabular (classify_duck memetakan Duckie di luar koridor/rentang
    menjadi NONE). Gerbang ini menyamakan semantik deteksinya.
    """

    if not duck.present:
        return duck
    if cfg.duck_detection_forward_only and duck.longitudinal < 0.0:
        return DuckRelativeState()
    if (
        cfg.duck_detection_corridor_width is not None
        and abs(duck.lateral) > cfg.duck_detection_corridor_width
    ):
        return DuckRelativeState()
    if cfg.duck_detection_range is not None:
        distance = float(np.hypot(duck.longitudinal, duck.lateral))
        if distance > cfg.duck_detection_range:
            return DuckRelativeState()
    return duck


def build_continuous_state(
    env: Any,
    raw: RawState,
    state_cfg: StateConfig,
    continuous_cfg: ContinuousStateConfig,
    controller: Any = None,
    stop_hold_progress: float = 0.0,
) -> ContinuousState:
    duck = gate_duck_visibility(
        duck_relative_state(env, raw.v, controller), continuous_cfg
    )
    return ContinuousState(
        d=raw.d,
        phi=raw.phi,
        v=raw.v,
        kappa=signed_curvature_ahead(env, state_cfg, continuous_cfg),
        stop_present=raw.d_stop is not None,
        d_stop=raw.d_stop,
        sigma_stop=raw.sigma_stop,
        duck_present=duck.present,
        duck_longitudinal=duck.longitudinal,
        duck_lateral=duck.lateral,
        duck_v_longitudinal_relative=duck.v_longitudinal_relative,
        duck_v_lateral_relative=duck.v_lateral_relative,
        duck_active=duck.active,
        duck_crossing_available=duck.crossing_available,
        stop_hold_progress=float(np.clip(stop_hold_progress, 0.0, 1.0)),
    )


def encode_continuous_state(
    state: ContinuousState,
    cfg: ContinuousStateConfig,
) -> np.ndarray:
    stop_distance = (
        1.0
        if not state.stop_present or state.d_stop is None
        else np.clip(state.d_stop / cfg.max_stop_distance, 0.0, 1.0)
    )
    if state.duck_present:
        duck_longitudinal = np.clip(
            state.duck_longitudinal / cfg.max_duck_distance, -1.0, 1.0
        )
        duck_lateral = np.clip(
            state.duck_lateral / cfg.max_duck_distance, -1.0, 1.0
        )
        duck_v_longitudinal = np.clip(
            state.duck_v_longitudinal_relative / cfg.max_relative_speed, -1.0, 1.0
        )
        duck_v_lateral = np.clip(
            state.duck_v_lateral_relative / cfg.max_relative_speed, -1.0, 1.0
        )
    else:
        # Sentinel absent = jarak maksimum di depan, velocity nol. Mask tetap
        # menjadi pembeda utama dari Duckie yang benar-benar berada di sana.
        duck_longitudinal, duck_lateral = 1.0, 0.0
        duck_v_longitudinal, duck_v_lateral = 0.0, 0.0

    values = np.array(
        [
            np.clip(state.d / 0.25, -1.0, 1.0),
            np.clip(state.phi / (pi / 2.0), -1.0, 1.0),
            np.clip(state.v / cfg.max_speed, 0.0, 1.0),
            np.clip(state.kappa / cfg.max_abs_curvature, -1.0, 1.0),
            float(state.stop_present),
            float(stop_distance),
            float(state.sigma_stop),
            float(state.duck_present),
            float(duck_longitudinal),
            float(duck_lateral),
            float(duck_v_longitudinal),
            float(duck_v_lateral),
            float(state.duck_active),
            float(state.duck_crossing_available),
            float(np.clip(state.stop_hold_progress, 0.0, 1.0)),
        ],
        dtype=np.float32,
    )
    if not np.all(np.isfinite(values)):
        raise ValueError("Continuous observation contains non-finite values")
    return values


def continuous_observation_space() -> spaces.Box:
    low = np.array(
        [-1, -1, 0, -1, 0, 0, 0, 0, -1, -1, -1, -1, 0, 0, 0],
        dtype=np.float32,
    )
    high = np.ones(len(OBSERVATION_NAMES), dtype=np.float32)
    return spaces.Box(low=low, high=high, dtype=np.float32)


def continuous_state_to_dict(state: ContinuousState) -> Dict[str, Any]:
    return asdict(state)
