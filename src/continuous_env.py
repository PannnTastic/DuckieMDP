"""Continuous-action wrapper dengan API Gym 0.23 untuk SAC lokal."""

from dataclasses import asdict, replace
from typing import Any, Dict

import gym
import numpy as np
from gym import spaces

from .actions import vw_to_wheels
from .continuous_state import (
    ContinuousState,
    ContinuousStateConfig,
    build_continuous_state,
    continuous_observation_space,
    continuous_state_to_dict,
    encode_continuous_state,
)
from .env_wrapper import DuckieMDPEnv, _any_collision, _duck_collision, build_env
from .reward import compute_reward
from .state import get_raw_state, next_stop_candidate, raw_state_to_dict


class ContinuousDuckieMDPEnv(gym.Wrapper):
    """Mengganti Discrete(7) dengan action `[v_cmd, omega_cmd]`."""

    def __init__(
        self,
        env: DuckieMDPEnv,
        continuous_cfg: ContinuousStateConfig,
    ) -> None:
        super().__init__(env)
        self.continuous_cfg = continuous_cfg
        action_cfg = env.action_cfg
        self.action_space = spaces.Box(
            low=np.array([0.0, -action_cfg.w0], dtype=np.float32),
            high=np.array([action_cfg.v_fast, action_cfg.w0], dtype=np.float32),
            dtype=np.float32,
        )
        self.observation_space = continuous_observation_space()
        self.current_state: ContinuousState = None

    @property
    def mdp_env(self) -> DuckieMDPEnv:
        return self.env

    def reset(self, seed: int = None) -> np.ndarray:
        raw = self.mdp_env.reset(seed)
        self.current_state = build_continuous_state(
            self,
            raw,
            self.mdp_env.state_cfg,
            self.continuous_cfg,
            self.mdp_env.duck_controller,
            self.mdp_env.stop_tracker.hold_progress,
        )
        observation = encode_continuous_state(self.current_state, self.continuous_cfg)
        if not self.observation_space.contains(observation):
            raise ValueError("reset observation is outside observation_space")
        return observation

    def step(self, action: np.ndarray):
        if self.mdp_env._last_state is None:
            raise RuntimeError("Call reset() before step()")
        command = np.asarray(action, dtype=np.float32).reshape(-1)
        if command.shape != (2,):
            raise ValueError("Continuous action must have shape (2,)")
        command = np.clip(command, self.action_space.low, self.action_space.high)
        v_cmd, omega_cmd = float(command[0]), float(command[1])

        previous = self.mdp_env._last_state
        previous_stop_id = self.mdp_env._last_stop_id
        self.mdp_env.duck_controller.before_step()
        wheels = vw_to_wheels(
            v_cmd,
            omega_cmd,
            self.mdp_env.action_cfg.wheel_base,
        )
        simulator_reward, simulator_done, info = self.mdp_env._simulator_step(wheels)

        current = get_raw_state(
            self.mdp_env,
            self.mdp_env.stop_tracker.sigma_stop,
            self.mdp_env.state_cfg,
        )
        _, current_stop_id = next_stop_candidate(
            self.mdp_env,
            self.mdp_env.state_cfg,
        )
        sigma, events = self.mdp_env.stop_tracker.update(
            previous,
            current,
            previous_stop_id,
            current_stop_id,
        )
        self.unwrapped._mdp_sigma_stop = sigma
        current = replace(current, sigma_stop=sigma)

        duck_collision = _duck_collision(self.unwrapped)
        any_collision = _any_collision(self.unwrapped) if simulator_done else False
        max_steps = self.unwrapped.step_count >= self.unwrapped.max_steps
        goal = self.mdp_env.goal_tile
        reached_goal = goal is not None and (
            tuple(self.unwrapped.get_grid_coords(self.unwrapped.cur_pos)) == goal
        )

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
        terminated = reason in {
            "duck_collision",
            "other_collision",
            "offroad",
            "goal",
        }
        truncated = reason == "timeout"
        done = terminated or truncated
        reward = compute_reward(
            current,
            events,
            self.mdp_env.reward_cfg,
            action_omega=omega_cmd,
            curvature=self.current_state.kappa,
        )

        self.mdp_env._last_state = current
        self.mdp_env._last_stop_id = current_stop_id
        self.current_state = build_continuous_state(
            self,
            current,
            self.mdp_env.state_cfg,
            self.continuous_cfg,
            self.mdp_env.duck_controller,
            self.mdp_env.stop_tracker.hold_progress,
        )
        observation = encode_continuous_state(self.current_state, self.continuous_cfg)
        if not self.observation_space.contains(observation):
            raise ValueError("step observation is outside observation_space")

        info = dict(info)
        info.update(
            {
                "raw_state": raw_state_to_dict(current),
                "continuous_state": continuous_state_to_dict(self.current_state),
                "events": asdict(events),
                "reward_terms": reward.as_dict(),
                "simulator_reward": float(simulator_reward),
                "command_v_omega": [v_cmd, omega_cmd],
                "wheel_commands": wheels.tolist(),
                "action_units": "simulator_command_magnitudes",
                "termination_reason": reason,
                "terminated": terminated,
                "truncated": truncated,
                "stop_hold_steps": self.mdp_env.stop_tracker.hold_steps,
                "stop_hold_steps_required": (
                    self.mdp_env.stop_tracker.hold_steps_required
                ),
            }
        )
        return observation, reward.total, done, info


def build_continuous_env(config: Dict[str, Any], seed: int) -> ContinuousDuckieMDPEnv:
    base = build_env(config, seed)
    continuous_cfg = ContinuousStateConfig(**config.get("continuous_state", {}))
    return ContinuousDuckieMDPEnv(base, continuous_cfg)
