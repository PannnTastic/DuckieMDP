"""Definisi state MDP untuk tugas mengemudi Duckietown.

State policy yang dipakai adalah representasi lane-relative:

    s_t = (d_t, phi_t, v_t, kappa_t, d_stop_t, sigma_stop_t, h_duck_t)

Posisi global (x, z, psi) tetap berada di latent state simulator, tetapi tidak
dimasukkan langsung ke Q-table. Representasi relatif terhadap lajur lebih kecil
dan dapat dipakai pada lokasi peta yang berbeda dengan geometri yang sama.
"""

from dataclasses import asdict, dataclass
from enum import IntEnum
from math import cos, copysign, pi, sin
from typing import Any, Dict, Mapping, Optional, Tuple

import numpy as np
from gym_duckietown.simulator import NotInLane, bezier_tangent


class TileType(IntEnum):
    """Curvature lajur di depan, RELATIF terhadap arah gerak ego."""
    STRAIGHT = 0
    CURVE_LEFT = 1
    CURVE_RIGHT = 2


class DuckThreat(IntEnum):
    NONE = 0
    SIDE_FAR = 1
    SIDE_NEAR = 2
    CROSSING_FAR = 3
    CROSSING_NEAR = 4


@dataclass(frozen=True)
class RawState:
    """State kontinu sebelum proses diskritisasi.

    d adalah error lateral (meter), phi error heading (radian), dan v kecepatan
    aktual. tile menyimpan curvature look-ahead. d_stop adalah jarak stop line,
    sigma_stop adalah memori kepatuhan satu bit, dan duck adalah kelas ancaman
    pedestrian. sigma_stop membuat proses stop-sign lebih mendekati Markov.
    """
    d: float
    phi: float
    v: float
    tile: TileType
    d_stop: Optional[float]
    sigma_stop: bool
    duck: DuckThreat


@dataclass(frozen=True)
class StateConfig:
    stop_lateral_limit: float = 0.40
    stop_orientation_cos: float = 0.70710678
    sign_to_line_offset: float = 0.20
    stop_max_distance: float = 3.0
    stop_zone: float = 0.45
    stop_pass_distance: float = 0.55
    stop_speed: float = 0.02
    tile_lookahead: float = 0.30
    curvature_threshold: float = 0.05
    duck_max_distance: float = 2.0
    duck_near_distance: float = 0.60
    duck_corridor_width: float = 0.35


def _base(env: Any) -> Any:
    return getattr(env, "unwrapped", env)


def _kind(value: Any) -> str:
    return str(getattr(value, "value", value)).lower()


def _heading_vector(angle: float) -> np.ndarray:
    return np.array([cos(angle), 0.0, -sin(angle)], dtype=float)


def _normalize(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    return vector / norm if norm > 0.0 else vector


def _lane_frame(env: Any) -> Tuple[np.ndarray, np.ndarray]:
    """Membentuk sumbu forward/right lokal agar objek dinilai ego-relative."""
    point, tangent = env.closest_curve_point(env.cur_pos, env.cur_angle)
    if point is None or tangent is None:
        forward = _heading_vector(float(env.cur_angle))
    else:
        forward = _normalize(np.asarray(tangent, dtype=float))
        if not np.any(forward):
            forward = _heading_vector(float(env.cur_angle))
    right = np.cross(forward, np.array([0.0, 1.0, 0.0]))
    return forward, right


def classify_tile(tile: Mapping[str, Any]) -> TileType:
    if tile is None or not tile.get("drivable", False):
        raise ValueError("Not on a drivable tile")
    kind = str(tile.get("kind", "")).lower()
    if kind == "curve_left":
        return TileType.CURVE_LEFT
    if kind == "curve_right":
        return TileType.CURVE_RIGHT
    if kind == "straight" or kind.startswith("3way") or kind == "4way":
        return TileType.STRAIGHT
    raise ValueError("Unsupported tile kind: %s" % kind)


def _ego_relative_curve(tile: Mapping[str, Any], forward: np.ndarray, threshold: float) -> TileType:
    """Mengubah kurva peta menjadi kappa_t yang benar dari arah masuk ego.

    Satu tile curve_left dapat memiliki dua directed lane dengan belokan
    berlawanan. Tanda cross product tangent Bezier menentukan left/right pada
    directed curve yang sedang diikuti agent.
    """
    curves = tile.get("curves")
    if curves is None or len(curves) == 0:
        return classify_tile(tile)
    headings = np.asarray(curves)[:, -1, :] - np.asarray(curves)[:, 0, :]
    headings = np.asarray([_normalize(heading) for heading in headings])
    curve = np.asarray(curves[int(np.argmax(np.dot(headings, forward)))])
    tangent_before = _normalize(np.asarray(bezier_tangent(curve, 0.10), dtype=float))
    tangent_after = _normalize(np.asarray(bezier_tangent(curve, 0.90), dtype=float))
    signed_turn = float(np.cross(tangent_before, tangent_after)[1])
    if signed_turn > threshold:
        return TileType.CURVE_LEFT
    if signed_turn < -threshold:
        return TileType.CURVE_RIGHT
    return TileType.STRAIGHT


def tile_ahead(env: Any, cfg: StateConfig) -> TileType:
    """Mengukur curvature pada titik look-ahead, bukan hanya tile saat ini."""
    env = _base(env)
    forward, _ = _lane_frame(env)
    probe = np.asarray(env.cur_pos, dtype=float) + cfg.tile_lookahead * forward
    tile = env._get_tile(*env.get_grid_coords(probe))
    if tile is None or not tile.get("drivable", False):
        tile = env._get_tile(*env.get_grid_coords(env.cur_pos))
    if tile is None:
        raise ValueError("No drivable tile for curvature lookup")
    return _ego_relative_curve(tile, forward, cfg.curvature_threshold)


def next_stop_candidate(env: Any, cfg: StateConfig) -> Tuple[Optional[float], Optional[int]]:
    """Memilih stop sign terdekat yang berada di depan dan menghadap ego.

    Filter lateral dan orientasi mencegah sign lajur berlawanan masuk ke state.
    """
    env = _base(env)
    forward, right = _lane_frame(env)
    candidates = []
    for index, obj in enumerate(env.objects):
        if not getattr(obj, "visible", True) or _kind(obj.kind) != "sign_stop":
            continue
        sign_facing = _heading_vector(float(obj.angle))
        if float(np.dot(sign_facing, forward)) > -cfg.stop_orientation_cos:
            continue
        rel = np.asarray(obj.pos, dtype=float) - np.asarray(env.cur_pos, dtype=float)
        ahead = float(np.dot(rel, forward))
        lateral = abs(float(np.dot(rel, right)))
        if ahead <= 0.0 or lateral > cfg.stop_lateral_limit:
            continue
        distance = max(0.0, ahead - cfg.sign_to_line_offset)
        if distance <= cfg.stop_max_distance:
            candidates.append((distance, index))
    return min(candidates, key=lambda item: item[0]) if candidates else (None, None)


def distance_to_next_stop(env: Any, cfg: StateConfig) -> Optional[float]:
    return next_stop_candidate(env, cfg)[0]


def classify_duck(env: Any, cfg: StateConfig) -> DuckThreat:
    """Memetakan pedestrian ke kelas ancaman di koridor tabrakan ego."""
    env = _base(env)
    forward, right = _lane_frame(env)
    result = DuckThreat.NONE
    for obj in env.objects:
        if not getattr(obj, "visible", True) or _kind(obj.kind) != "duckie":
            continue
        pos = np.asarray(getattr(obj, "center", obj.pos), dtype=float)
        rel = pos - np.asarray(env.cur_pos, dtype=float)
        ahead = float(np.dot(rel, forward))
        lateral = abs(float(np.dot(rel, right)))
        distance = float(np.linalg.norm(rel[[0, 2]]))
        in_forward_corridor = ahead >= 0.0 and lateral <= cfg.duck_corridor_width
        if not in_forward_corridor or distance > cfg.duck_max_distance:
            continue
        crossing = bool(getattr(obj, "pedestrian_active", False))
        if crossing:
            threat = DuckThreat.CROSSING_NEAR if distance <= cfg.duck_near_distance else DuckThreat.CROSSING_FAR
        else:
            threat = DuckThreat.SIDE_NEAR if distance <= cfg.duck_near_distance else DuckThreat.SIDE_FAR
        result = max(result, threat)
    return result


def _terminal_lane_fallback(env: Any) -> Tuple[float, float]:
    last_d, last_phi = getattr(env, "_mdp_last_lane_position", (1.0, 1.0))
    d_reference = last_d if abs(last_d) > 1e-9 else last_phi
    phi_reference = last_phi if abs(last_phi) > 1e-9 else last_d
    return copysign(0.25, d_reference or 1.0), copysign(pi / 2, phi_reference or 1.0)


def get_raw_state(env: Any, sigma_stop: Optional[bool] = None,
                  config: StateConfig = StateConfig()) -> RawState:
    """Mengekstrak s_t dari latent state gym-duckietown."""
    env = _base(env)
    try:
        lane = env.get_lane_pos2(env.cur_pos, env.cur_angle)
        env._mdp_last_lane_position = (float(lane.dist), float(lane.angle_rad))
        d = float(np.clip(lane.dist, -0.25, 0.25))
        phi = float(np.clip(lane.angle_rad, -pi / 2, pi / 2))
    except NotInLane:
        d, phi = _terminal_lane_fallback(env)
    speed = max(0.0, float(env.speed))
    try:
        tile = tile_ahead(env, config)
    except ValueError:
        tile = TileType.STRAIGHT
    if not np.all(np.isfinite([d, phi, speed])):
        raise ValueError("Non-finite raw state")
    if sigma_stop is None:
        sigma_stop = bool(getattr(env, "_mdp_sigma_stop", False))
    return RawState(d, phi, speed, tile, distance_to_next_stop(env, config),
                    bool(sigma_stop), classify_duck(env, config))


def raw_state_to_dict(state: RawState) -> Dict[str, Any]:
    data = asdict(state)
    data["tile"], data["duck"] = int(state.tile), int(state.duck)
    return data
