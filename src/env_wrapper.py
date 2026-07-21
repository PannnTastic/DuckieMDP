"""Wrapper yang menyatukan komponen MDP (S, A, P, R, terminal).

Formulasi:

    M = (S, A, P, R, gamma, rho_0, T)

S diekstrak oleh state.py, A oleh actions.py, dan R oleh reward.py. Transition
P(s'|s,a) tidak ditulis sebagai matriks: gym-duckietown menjadi generative
transition model melalui simulasi fisika. reset() mendefinisikan initial-state
distribution rho_0; step() menghasilkan transition, reward, dan kondisi akhir.
"""

from dataclasses import asdict, replace
from math import cos, sin
from typing import Any, Dict, Optional, Sequence, Tuple

import gym
import numpy as np
from gym import spaces
from gym_duckietown.collision import generate_norm, intersects_single_obj
from gym_duckietown.envs import DuckietownEnv
from gym_duckietown.simulator import Simulator, get_agent_corners

from .actions import ActionConfig, action_to_wheels
from .duck_controller import DuckController, DuckControllerConfig, make_ducks_dynamic
from .reward import RewardConfig, StopTracker, compute_reward
from .state import RawState, StateConfig, get_raw_state, next_stop_candidate, raw_state_to_dict


def _kind(value: Any) -> str:
    return str(getattr(value, "value", value)).lower()


def _duck_collision(env: Any) -> bool:
    corners = get_agent_corners(env.cur_pos, env.cur_angle)
    norms = generate_norm(corners)
    for obj in env.objects:
        if _kind(obj.kind) != "duckie" or not getattr(obj, "visible", True):
            continue
        if intersects_single_obj(corners, obj.obj_corners.T, norms, obj.obj_norm):
            return True
    return False


def _any_collision(env: Any) -> bool:
    return bool(env._collision(get_agent_corners(env.cur_pos, env.cur_angle)))


def route_circulation_score(
    position: Sequence[float],
    angle: float,
    center: Sequence[float],
) -> float:
    """Return alignment with the clockwise tangent around ``center``.

    Duckietown uses ``[cos(angle), -sin(angle)]`` as the forward vector in the
    world x-z plane.  Positive values mean clockwise travel, negative values
    mean counter-clockwise travel, and the magnitude is the tangent alignment.
    This is used only to constrain rho_0 on loop maps; it does not enter S.
    """
    point = np.asarray(position, dtype=float).reshape(-1)
    if point.size == 3:
        point = point[[0, 2]]
    if point.size != 2:
        raise ValueError("position must contain x,z or x,y,z coordinates")
    center_xz = np.asarray(center, dtype=float).reshape(-1)
    if center_xz.size != 2:
        raise ValueError("route center must contain x,z coordinates")
    radial = point - center_xz
    radial_norm = float(np.linalg.norm(radial))
    if radial_norm <= 1e-9:
        return 0.0
    clockwise_tangent = np.array([radial[1], -radial[0]], dtype=float) / radial_norm
    forward = np.array([cos(float(angle)), -sin(float(angle))], dtype=float)
    return float(np.dot(forward, clockwise_tangent))


def position_in_bounds_xz(
    position: Sequence[float],
    bounds: Sequence[float],
) -> bool:
    """Check an x-z spawn rectangle encoded as [xmin, xmax, zmin, zmax]."""
    point = np.asarray(position, dtype=float).reshape(-1)
    if point.size == 3:
        x, z = float(point[0]), float(point[2])
    elif point.size == 2:
        x, z = float(point[0]), float(point[1])
    else:
        raise ValueError("position must contain x,z or x,y,z coordinates")
    limits = np.asarray(bounds, dtype=float).reshape(-1)
    if limits.size != 4:
        raise ValueError("spawn_position_bounds_xz must contain four values")
    xmin, xmax, zmin, zmax = (float(value) for value in limits)
    if xmin > xmax or zmin > zmax:
        raise ValueError("spawn position bounds must satisfy min <= max")
    return xmin <= x <= xmax and zmin <= z <= zmax


class DuckieMDPEnv(gym.Wrapper):
    """Finite-action MDP interface di atas simulator kontinu Duckietown."""
    def __init__(
        self,
        env: DuckietownEnv,
        action_cfg: ActionConfig,
        state_cfg: StateConfig,
        reward_cfg: RewardConfig,
        duck_cfg: DuckControllerConfig,
        seed: int,
        goal_tile: Optional[Tuple[int, int]] = None,
        render_observations: bool = True,
        spawn_max_abs_d: Optional[float] = None,
        spawn_max_abs_phi: Optional[float] = None,
        spawn_attempts: int = 50,
        spawn_route_direction: Optional[str] = None,
        spawn_route_center: Optional[Tuple[float, float]] = None,
        spawn_min_route_alignment: float = 0.50,
        spawn_position_bounds_xz: Optional[Tuple[float, float, float, float]] = None,
    ) -> None:
        super().__init__(env)
        self.action_cfg = action_cfg
        self.state_cfg = state_cfg
        self.reward_cfg = reward_cfg
        self.goal_tile = goal_tile
        self.render_observations = render_observations
        self.spawn_max_abs_d = spawn_max_abs_d
        self.spawn_max_abs_phi = spawn_max_abs_phi
        self.spawn_attempts = spawn_attempts
        direction = (
            None
            if spawn_route_direction is None
            else str(spawn_route_direction).strip().lower()
        )
        if direction not in {None, "clockwise", "counterclockwise"}:
            raise ValueError(
                "spawn_route_direction must be clockwise, counterclockwise, or null"
            )
        if not 0.0 <= float(spawn_min_route_alignment) <= 1.0:
            raise ValueError("spawn_min_route_alignment must be in [0, 1]")
        self.spawn_route_direction = direction
        self.spawn_route_center = spawn_route_center
        self.spawn_min_route_alignment = float(spawn_min_route_alignment)
        if spawn_position_bounds_xz is not None:
            # Validate once during construction; reset only performs the check.
            position_in_bounds_xz((0.0, 0.0), spawn_position_bounds_xz)
            self.spawn_position_bounds_xz = tuple(
                float(value) for value in spawn_position_bounds_xz
            )
        else:
            self.spawn_position_bounds_xz = None
        self.last_spawn_route_alignment = np.nan
        self.action_space = spaces.Discrete(7)
        self.stop_tracker = StopTracker(
            state_cfg.stop_zone,
            state_cfg.stop_speed,
            state_cfg.stop_pass_distance,
            state_cfg.stop_hold_steps,
        )
        self.duck_controller = DuckController(env, duck_cfg, seed)
        self._last_state: Optional[RawState] = None
        self._last_stop_id: Optional[int] = None

    def seed(self, seed: int = None):
        return self.env.seed(seed)

    def _spawn_is_accepted(self, state: RawState) -> bool:
        d_ok = self.spawn_max_abs_d is None or abs(state.d) <= self.spawn_max_abs_d
        phi_ok = self.spawn_max_abs_phi is None or abs(state.phi) <= self.spawn_max_abs_phi
        position_ok = (
            self.spawn_position_bounds_xz is None
            or position_in_bounds_xz(
                self.env.cur_pos,
                self.spawn_position_bounds_xz,
            )
        )
        route_ok = True
        self.last_spawn_route_alignment = np.nan
        if self.spawn_route_direction is not None:
            center = self.spawn_route_center
            if center is None:
                center = (
                    self.env.grid_width * self.env.road_tile_size / 2.0,
                    self.env.grid_height * self.env.road_tile_size / 2.0,
                )
            score = route_circulation_score(
                self.env.cur_pos,
                self.env.cur_angle,
                center,
            )
            self.last_spawn_route_alignment = score
            if self.spawn_route_direction == "clockwise":
                route_ok = score >= self.spawn_min_route_alignment
            else:
                route_ok = score <= -self.spawn_min_route_alignment
        return d_ok and phi_ok and position_ok and route_ok

    def reset(self, seed: int = None) -> RawState:
        """Sample s_0 dari rho_0 dengan curriculum pada d dan phi."""
        if seed is not None:
            self.env.seed(seed)
        candidate = None
        for _ in range(max(1, self.spawn_attempts)):
            self.env.reset()
            # Simulator.reset() memulihkan atribut object dari map, termasuk
            # Duckie.visible. Reset controller harus dilakukan setelahnya agar
            # state controller dan mode spawn-on-proximity benar-benar menjadi
            # kondisi awal episode yang diamati policy.
            self.duck_controller.reset(seed)
            self.env._mdp_sigma_stop = False
            self.env._mdp_last_lane_position = (1.0, 1.0)
            candidate = get_raw_state(self.env, False, self.state_cfg)
            if self._spawn_is_accepted(candidate):
                break
        else:
            raise RuntimeError(
                "Could not sample a curriculum spawn satisfying "
                f"|d|<={self.spawn_max_abs_d}, |phi|<={self.spawn_max_abs_phi}, "
                f"route={self.spawn_route_direction}, "
                f"bounds_xz={self.spawn_position_bounds_xz}"
            )
        self.stop_tracker.reset()
        self._last_state = candidate
        _, self._last_stop_id = next_stop_candidate(self.env, self.state_cfg)
        return self._last_state

    def _simulator_step(self, wheels: np.ndarray):
        """Menerapkan satu macro-action selama frame_skip physics steps.

        Satu transition MDP adalah:

            (s_t, a_t) --simulator/frame_skip--> s_(t+1)

        Action holding membantu actuator delay, tetapi frame_skip harus sama
        ketika training dan evaluation.
        """
        if self.render_observations:
            _, reward, done, info = Simulator.step(self.env, wheels)
            return reward, done, info
        action = np.asarray(np.clip(wheels, -1.0, 1.0), dtype=float)
        for _ in range(self.env.frame_skip):
            self.env.update_physics(action)
        info = self.env.get_agent_info()
        result = self.env._compute_done_reward()
        info["Simulator"]["msg"] = result.done_why
        return result.reward, result.done, info

    def step(self, action_id: int):
        """Melakukan transition dan mengembalikan (s_next, reward, done, info)."""
        if self._last_state is None:
            raise RuntimeError("Call reset() before step()")
        previous, previous_stop_id = self._last_state, self._last_stop_id
        self.duck_controller.before_step()
        wheels = action_to_wheels(action_id, self.action_cfg)
        simulator_reward, simulator_done, info = self._simulator_step(wheels)
        current = get_raw_state(self.env, self.stop_tracker.sigma_stop, self.state_cfg)
        _, current_stop_id = next_stop_candidate(self.env, self.state_cfg)
        sigma, events = self.stop_tracker.update(
            previous, current, previous_stop_id, current_stop_id
        )
        self.env._mdp_sigma_stop = sigma
        current = replace(current, sigma_stop=sigma)

        duck_collision = _duck_collision(self.env)
        any_collision = _any_collision(self.env) if simulator_done else False
        max_steps = self.env.step_count >= self.env.max_steps
        reached_goal = self.goal_tile is not None and (
            tuple(self.env.get_grid_coords(self.env.cur_pos)) == self.goal_tile
        )
        # Urutan ini memberi satu termination reason eksplisit sehingga collision
        # objek tidak tercampur dengan off-road.
        if simulator_done and duck_collision:
            reason = "duck_collision"
        elif simulator_done and any_collision:
            reason = "other_collision"
        elif simulator_done and max_steps:
            reason = "timeout"
        elif simulator_done:
            reason = "offroad"
        elif reached_goal:
            reason = "goal"
        else:
            reason = "in_progress"

        events.collision_duck = reason == "duck_collision"
        events.other_collision = reason == "other_collision"
        events.offroad = reason == "offroad"
        events.timeout = reason == "timeout"
        events.goal = reason == "goal"
        # Terminal nyata memutus bootstrap TD. Timeout hanya truncation akibat
        # batas horizon eksperimen, bukan absorbing physical state.
        terminated = reason in {"duck_collision", "other_collision", "offroad", "goal"}
        truncated = reason == "timeout"
        done = terminated or truncated
        reward = compute_reward(current, events, self.reward_cfg)
        info = dict(info)
        info.update(
            {
                "raw_state": raw_state_to_dict(current),
                "events": asdict(events),
                "reward_terms": reward.as_dict(),
                "simulator_reward": float(simulator_reward),
                "action_id": int(action_id),
                "wheel_commands": wheels.tolist(),
                "action_units": "normalized_wheel_commands",
                "termination_reason": reason,
                "terminated": terminated,
                "truncated": truncated,
            }
        )
        self._last_state, self._last_stop_id = current, current_stop_id
        return current, reward.total, done, info


def build_env(config: Dict[str, Any], seed: int) -> DuckieMDPEnv:
    """Factory reproducible: seluruh parameter MDP berasal dari satu YAML."""
    env_cfg = config["environment"]
    duck_cfg = DuckControllerConfig(**config["duck_controller"])
    base = DuckietownEnv(
        map_name=env_cfg["map_name"],
        domain_rand=env_cfg["domain_rand"],
        max_steps=env_cfg["max_steps"],
        frame_skip=env_cfg["frame_skip"],
        user_tile_start=env_cfg.get("user_tile_start"),
        accept_start_angle_deg=env_cfg.get("accept_start_angle_deg", 60),
        seed=seed,
    )
    if duck_cfg.make_dynamic:
        make_ducks_dynamic(base, duck_cfg)
    goal = env_cfg.get("goal_tile")
    return DuckieMDPEnv(
        base,
        ActionConfig(**config["actions"]),
        StateConfig(**config["state"]),
        RewardConfig(**config["reward"]),
        duck_cfg,
        seed,
        tuple(goal) if goal is not None else None,
        render_observations=env_cfg.get("render_observations", True),
        spawn_max_abs_d=env_cfg.get("spawn_max_abs_d"),
        spawn_max_abs_phi=env_cfg.get("spawn_max_abs_phi"),
        spawn_attempts=int(env_cfg.get("spawn_attempts", 50)),
        spawn_route_direction=env_cfg.get("spawn_route_direction"),
        spawn_route_center=(
            tuple(env_cfg["spawn_route_center"])
            if env_cfg.get("spawn_route_center") is not None
            else None
        ),
        spawn_min_route_alignment=float(
            env_cfg.get("spawn_min_route_alignment", 0.50)
        ),
        spawn_position_bounds_xz=(
            tuple(env_cfg["spawn_position_bounds_xz"])
            if env_cfg.get("spawn_position_bounds_xz") is not None
            else None
        ),
    )
