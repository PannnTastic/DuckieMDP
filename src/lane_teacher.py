from dataclasses import dataclass

from .state import DuckThreat, RawState, TileType


@dataclass(frozen=True)
class LaneTeacherConfig:
    enabled: bool = False
    full_control_episodes: int = 0
    decay_episodes: int = 0
    min_probability: float = 0.0
    d_gain: float = 1.0
    error_threshold: float = 0.10
    brake_for_stop: bool = True
    stop_brake_distance: float = 0.45
    brake_for_duck: bool = True


def teacher_probability(episode: int, cfg: LaneTeacherConfig) -> float:
    if not cfg.enabled:
        return 0.0
    if episode < cfg.full_control_episodes:
        return 1.0
    if cfg.decay_episodes <= 0:
        return cfg.min_probability
    elapsed = episode - cfg.full_control_episodes
    if elapsed >= cfg.decay_episodes:
        return cfg.min_probability
    fraction = max(0.0, elapsed / cfg.decay_episodes)
    return 1.0 - fraction * (1.0 - cfg.min_probability)


def select_lane_teacher_action(state: RawState, cfg: LaneTeacherConfig) -> int:
    """Return aksi aman untuk lane, stop sign, dan crossing Duckie."""
    dangerous_duck = state.duck in {
        DuckThreat.CROSSING_FAR,
        DuckThreat.CROSSING_NEAR,
    }
    if cfg.brake_for_duck and dangerous_duck:
        return 6
    must_stop = (
        state.d_stop is not None
        and state.d_stop <= cfg.stop_brake_distance
        and not state.sigma_stop
    )
    if cfg.brake_for_stop and must_stop:
        return 6
    error = state.phi + cfg.d_gain * state.d
    if error > cfg.error_threshold:
        return 3
    if error < -cfg.error_threshold:
        return 5
    if state.tile == TileType.CURVE_LEFT:
        return 0
    if state.tile == TileType.CURVE_RIGHT:
        return 5
    return 1
