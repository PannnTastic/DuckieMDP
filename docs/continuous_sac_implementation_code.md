# Kode Implementasi Continuous-State SAC (M7-M10)

Dokumen ini adalah lembar kerja pengetikan. File target sudah dibuat kosong.
Salin setiap blok secara utuh ke path yang tertulis. Jangan mengubah baseline
tabular atau artefak di `artifacts/ablation/`.

Urutan pengerjaan:

1. M7: `src/continuous_state.py` dan dua unit test;
2. M8: dependency, wrapper kontinu, SAC, compatibility smoke test;
3. M9: config, training, evaluasi, pemilihan checkpoint, dan video;
4. M10: DQN ablation hanya setelah M9 stabil.

## Daftar file

| File | Milestone | Status sekarang |
|---|---|---|
| `requirements-sac.txt` | M8 | terisi |
| `src/continuous_state.py` | M7 | terisi |
| `src/continuous_env.py` | M8 | terisi |
| `src/agents/sac.py` | M8 | terisi |
| `src/train_sac.py` | M9 | terisi |
| `src/evaluate_sac.py` | M9 | terisi |
| `src/select_sac_checkpoint.py` | M9 | terisi |
| `src/render_sac_video.py` | M9 | terisi |
| `scripts/check_sac_compatibility.py` | M8 | terisi |
| `configs/sac_lane.yaml` | M8/M9 | terisi |
| `configs/sac_stop.yaml` | M9 | terisi |
| `configs/sac_full.yaml` | M9 | terisi |
| `tests/test_continuous_curvature.py` | M7 | terisi |
| `tests/test_continuous_state.py` | M7 | terisi |
| `tests/test_continuous_env.py` | M8 | terisi |
| `tests/test_sac.py` | M8 | terisi |

`src/train_dqn.py` dan `configs/dqn_continuous_state.yaml` kini berisi pipeline
M10 agar tidak menjadi placeholder. Eksperimen penuhnya tetap ditunda sampai
hasil M9 dibekukan; DQN bukan blocker SAC.

---

## M7 — `src/continuous_state.py`

```python
"""Privileged continuous state untuk SAC tanpa mengubah RawState tabular.

Vektor observation:

    x = [d, phi, v, kappa,
         stop_present, d_stop, sigma_stop,
         duck_present, duck_long, duck_lat,
         duck_v_long_rel, duck_v_lat_rel,
         duck_active, duck_crossing_available]

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
)


@dataclass(frozen=True)
class ContinuousStateConfig:
    max_speed: float = 0.41
    max_abs_curvature: float = 8.0
    max_stop_distance: float = 3.0
    max_duck_distance: float = 2.0
    max_relative_speed: float = 0.50
    curvature_samples: int = 33


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
    phi: float
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


def _base(env: Any) -> Any:
    return getattr(env, "unwrapped", env)


def _kind(value: Any) -> str:
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


def build_continuous_state(
    env: Any,
    raw: RawState,
    state_cfg: StateConfig,
    continuous_cfg: ContinuousStateConfig,
    controller: Any = None,
) -> ContinuousState:
    duck = duck_relative_state(env, raw.v, controller)
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
        ],
        dtype=np.float32,
    )
    if not np.all(np.isfinite(values)):
        raise ValueError("Continuous observation contains non-finite values")
    return values


def continuous_observation_space() -> spaces.Box:
    low = np.array(
        [-1, -1, 0, -1, 0, 0, 0, 0, -1, -1, -1, -1, 0, 0],
        dtype=np.float32,
    )
    high = np.ones(len(OBSERVATION_NAMES), dtype=np.float32)
    return spaces.Box(low=low, high=high, dtype=np.float32)


def continuous_state_to_dict(state: ContinuousState) -> Dict[str, Any]:
    return asdict(state)
```

## M7 — `tests/test_continuous_curvature.py`

```python
import numpy as np
import pytest

from src.continuous_state import curve_signed_curvature


def _left_curve():
    return np.array(
        [
            [0.0, 0.0, 0.0],
            [0.3, 0.0, 0.0],
            [1.0, 0.0, -0.7],
            [1.0, 0.0, -1.0],
        ],
        dtype=float,
    )


def test_opposite_directed_curves_have_opposite_curvature():
    left = curve_signed_curvature(_left_curve())
    right = curve_signed_curvature(_left_curve()[::-1].copy())
    assert left > 0.0
    assert right < 0.0
    assert abs(left) == pytest.approx(abs(right), rel=1e-6)


def test_straight_curve_has_zero_curvature():
    straight = np.array(
        [[0, 0, 0], [0.3, 0, 0], [0.7, 0, 0], [1.0, 0, 0]],
        dtype=float,
    )
    assert curve_signed_curvature(straight) == pytest.approx(0.0)


def test_rotation_does_not_change_curvature():
    curve = _left_curve()
    angle = np.pi / 2.0
    rotation = np.array(
        [
            [np.cos(angle), 0.0, np.sin(angle)],
            [0.0, 1.0, 0.0],
            [-np.sin(angle), 0.0, np.cos(angle)],
        ]
    )
    rotated = curve @ rotation.T
    assert curve_signed_curvature(rotated) == pytest.approx(
        curve_signed_curvature(curve), rel=1e-6
    )
```

## M7 — `tests/test_continuous_state.py`

```python
from types import SimpleNamespace

import numpy as np

from src.continuous_state import (
    ContinuousState,
    ContinuousStateConfig,
    continuous_observation_space,
    duck_relative_state,
    encode_continuous_state,
)


def _state(**overrides):
    values = dict(
        d=0.0,
        phi=0.0,
        v=0.1,
        kappa=0.0,
        stop_present=False,
        d_stop=None,
        sigma_stop=False,
        duck_present=False,
        duck_longitudinal=0.0,
        duck_lateral=0.0,
        duck_v_longitudinal_relative=0.0,
        duck_v_lateral_relative=0.0,
        duck_active=False,
        duck_crossing_available=False,
    )
    values.update(overrides)
    return ContinuousState(**values)


def test_absent_masks_use_safe_sentinels_and_fit_space():
    cfg = ContinuousStateConfig()
    encoded = encode_continuous_state(_state(), cfg)
    space = continuous_observation_space()
    assert encoded.shape == (14,)
    assert encoded.dtype == np.float32
    assert encoded[4] == 0.0
    assert encoded[5] == 1.0
    assert encoded[7] == 0.0
    assert encoded[8] == 1.0
    assert encoded[9] == 0.0
    assert space.contains(encoded)


class FakeEnv:
    def __init__(self, duck):
        self.cur_pos = np.zeros(3)
        self.cur_angle = 0.0
        self.objects = [duck]

    def closest_curve_point(self, _pos, _angle):
        return np.zeros(3), np.array([1.0, 0.0, 0.0])


def test_duck_geometry_and_controller_phase_are_exposed():
    duck = SimpleNamespace(
        kind="duckie",
        visible=True,
        pos=np.array([1.0, 0.0, 0.5]),
        center=np.array([1.0, 0.0, 0.5]),
        heading=np.array([0.0, 0.0, 1.0]),
        vel=0.02,
        pedestrian_active=True,
    )
    controller = SimpleNamespace(
        ducks=[duck],
        crossings_started=[1],
        cfg=SimpleNamespace(max_crossings_per_episode=1),
    )
    result = duck_relative_state(FakeEnv(duck), ego_speed=0.10, controller=controller)
    assert result.present
    assert result.longitudinal == 1.0
    assert result.lateral == 0.5
    assert result.v_longitudinal_relative == -0.10
    assert result.v_lateral_relative == 0.02
    assert result.active
    assert not result.crossing_available
```

### Test M7

Setelah tiga file di atas selesai diketik:

```bash
pytest -q tests/test_continuous_curvature.py tests/test_continuous_state.py
pytest -q
```

Jangan lanjut M8 bila test lama berubah atau shape Q-table tidak lagi
`(5,5,3,3,4,2,5,7)`.

---

## M8 — `requirements-sac.txt`

```text
-r requirements.txt
--extra-index-url https://download.pytorch.org/whl/cu121
torch==2.1.2+cu121
```

Gunakan environment terpisah supaya `.venv` tabular tidak berubah:

```bash
python3.9 -m venv .venv-sac
.venv-sac/bin/pip install --upgrade pip
.venv-sac/bin/pip install -r requirements-sac.txt
```

## M8 — `src/continuous_env.py`

```python
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
        reward = compute_reward(current, events, self.mdp_env.reward_cfg)

        self.mdp_env._last_state = current
        self.mdp_env._last_stop_id = current_stop_id
        self.current_state = build_continuous_state(
            self,
            current,
            self.mdp_env.state_cfg,
            self.continuous_cfg,
            self.mdp_env.duck_controller,
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
            }
        )
        return observation, reward.total, done, info


def build_continuous_env(config: Dict[str, Any], seed: int) -> ContinuousDuckieMDPEnv:
    base = build_env(config, seed)
    continuous_cfg = ContinuousStateConfig(**config.get("continuous_state", {}))
    return ContinuousDuckieMDPEnv(base, continuous_cfg)
```

Catatan: wrapper ini sengaja terpisah. Jangan mengubah `DuckieMDPEnv.step()` atau
`action_space=Discrete(7)` milik baseline.

## M8 — `src/agents/sac.py`

```python
"""Soft Actor-Critic untuk continuous privileged state Duckietown."""

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict

import numpy as np
import torch
from torch import nn


@dataclass(frozen=True)
class SACConfig:
    gamma: float = 0.99
    tau: float = 0.005
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    alpha_lr: float = 3e-4
    initial_alpha: float = 0.2
    batch_size: int = 256
    replay_capacity: int = 300000
    hidden_size: int = 256
    target_entropy: float = -2.0


class ReplayBuffer:
    def __init__(self, capacity: int, obs_dim: int, action_dim: int, seed: int):
        self.capacity = int(capacity)
        self.obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.action = np.zeros((capacity, action_dim), dtype=np.float32)
        self.reward = np.zeros((capacity, 1), dtype=np.float32)
        self.next_obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.terminated = np.zeros((capacity, 1), dtype=np.float32)
        self.position = 0
        self.size = 0
        self.rng = np.random.RandomState(seed)

    def add(self, obs, action, reward, next_obs, terminated) -> None:
        index = self.position
        self.obs[index] = obs
        self.action[index] = action
        self.reward[index, 0] = reward
        self.next_obs[index] = next_obs
        # Timeout tidak dimasukkan sebagai terminal agar critic tetap bootstrap.
        self.terminated[index, 0] = float(terminated)
        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int, device: torch.device):
        indices = self.rng.randint(0, self.size, size=batch_size)
        arrays = (
            self.obs[indices],
            self.action[indices],
            self.reward[indices],
            self.next_obs[indices],
            self.terminated[indices],
        )
        return tuple(torch.as_tensor(value, device=device) for value in arrays)

    def __len__(self) -> int:
        return self.size


def _mlp(input_dim: int, output_dim: int, hidden: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(input_dim, hidden),
        nn.ReLU(),
        nn.Linear(hidden, hidden),
        nn.ReLU(),
        nn.Linear(hidden, output_dim),
    )


class SquashedGaussianActor(nn.Module):
    LOG_STD_MIN = -5.0
    LOG_STD_MAX = 2.0

    def __init__(self, obs_dim, action_low, action_high, hidden):
        super().__init__()
        action_low = torch.as_tensor(action_low, dtype=torch.float32)
        action_high = torch.as_tensor(action_high, dtype=torch.float32)
        action_dim = int(action_low.numel())
        self.backbone = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.mean = nn.Linear(hidden, action_dim)
        self.log_std = nn.Linear(hidden, action_dim)
        self.register_buffer("action_scale", (action_high - action_low) / 2.0)
        self.register_buffer("action_bias", (action_high + action_low) / 2.0)

    def distribution(self, obs):
        features = self.backbone(obs)
        mean = self.mean(features)
        log_std = torch.clamp(
            self.log_std(features), self.LOG_STD_MIN, self.LOG_STD_MAX
        )
        return torch.distributions.Normal(mean, log_std.exp())

    def sample(self, obs):
        distribution = self.distribution(obs)
        pre_tanh = distribution.rsample()
        squashed = torch.tanh(pre_tanh)
        action = squashed * self.action_scale + self.action_bias
        correction = torch.log(
            self.action_scale * (1.0 - squashed.pow(2)) + 1e-6
        )
        log_probability = (
            distribution.log_prob(pre_tanh) - correction
        ).sum(dim=-1, keepdim=True)
        deterministic = torch.tanh(distribution.mean) * self.action_scale + self.action_bias
        return action, log_probability, deterministic


class Critic(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, hidden: int):
        super().__init__()
        self.net = _mlp(obs_dim + action_dim, 1, hidden)

    def forward(self, obs, action):
        return self.net(torch.cat([obs, action], dim=-1))


class SACAgent:
    def __init__(
        self,
        obs_dim: int,
        action_low: np.ndarray,
        action_high: np.ndarray,
        cfg: SACConfig,
        seed: int,
        device: str = "cpu",
    ) -> None:
        np.random.seed(seed)
        torch.manual_seed(seed)
        self.cfg = cfg
        self.device = torch.device(device)
        self.obs_dim = int(obs_dim)
        self.action_low = np.asarray(action_low, dtype=np.float32)
        self.action_high = np.asarray(action_high, dtype=np.float32)
        action_dim = int(self.action_low.size)

        self.actor = SquashedGaussianActor(
            obs_dim, self.action_low, self.action_high, cfg.hidden_size
        ).to(self.device)
        self.critic1 = Critic(obs_dim, action_dim, cfg.hidden_size).to(self.device)
        self.critic2 = Critic(obs_dim, action_dim, cfg.hidden_size).to(self.device)
        self.target1 = Critic(obs_dim, action_dim, cfg.hidden_size).to(self.device)
        self.target2 = Critic(obs_dim, action_dim, cfg.hidden_size).to(self.device)
        self.target1.load_state_dict(self.critic1.state_dict())
        self.target2.load_state_dict(self.critic2.state_dict())
        for target in (self.target1, self.target2):
            target.requires_grad_(False)

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=cfg.actor_lr)
        critic_parameters = list(self.critic1.parameters()) + list(self.critic2.parameters())
        self.critic_optimizer = torch.optim.Adam(critic_parameters, lr=cfg.critic_lr)
        self.log_alpha = torch.tensor(
            np.log(cfg.initial_alpha),
            dtype=torch.float32,
            device=self.device,
            requires_grad=True,
        )
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=cfg.alpha_lr)
        self.replay = ReplayBuffer(
            cfg.replay_capacity, obs_dim, action_dim, seed + 1
        )
        self.updates = 0

    @property
    def alpha(self):
        return self.log_alpha.exp()

    def select_action(self, observation: np.ndarray, deterministic: bool = False):
        obs = torch.as_tensor(
            observation, dtype=torch.float32, device=self.device
        ).unsqueeze(0)
        with torch.no_grad():
            sampled, _, mean = self.actor.sample(obs)
        action = mean if deterministic else sampled
        return action.squeeze(0).cpu().numpy().astype(np.float32)

    def update(self) -> Dict[str, float]:
        if len(self.replay) < self.cfg.batch_size:
            return {}
        obs, action, reward, next_obs, terminated = self.replay.sample(
            self.cfg.batch_size, self.device
        )

        with torch.no_grad():
            next_action, next_log_probability, _ = self.actor.sample(next_obs)
            next_q = torch.minimum(
                self.target1(next_obs, next_action),
                self.target2(next_obs, next_action),
            )
            target = reward + self.cfg.gamma * (1.0 - terminated) * (
                next_q - self.alpha.detach() * next_log_probability
            )

        q1 = self.critic1(obs, action)
        q2 = self.critic2(obs, action)
        critic_loss = nn.functional.mse_loss(q1, target) + nn.functional.mse_loss(q2, target)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        sampled_action, log_probability, _ = self.actor.sample(obs)
        policy_q = torch.minimum(
            self.critic1(obs, sampled_action),
            self.critic2(obs, sampled_action),
        )
        actor_loss = (self.alpha.detach() * log_probability - policy_q).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        alpha_loss = -(
            self.log_alpha * (log_probability + self.cfg.target_entropy).detach()
        ).mean()
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()

        with torch.no_grad():
            for source, target_network in (
                (self.critic1, self.target1),
                (self.critic2, self.target2),
            ):
                for source_parameter, target_parameter in zip(
                    source.parameters(), target_network.parameters()
                ):
                    target_parameter.mul_(1.0 - self.cfg.tau)
                    target_parameter.add_(self.cfg.tau * source_parameter)

        self.updates += 1
        return {
            "critic_loss": float(critic_loss.item()),
            "actor_loss": float(actor_loss.item()),
            "alpha_loss": float(alpha_loss.item()),
            "alpha": float(self.alpha.detach().item()),
            "mean_q": float(torch.minimum(q1, q2).mean().item()),
        }

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        torch.save(
            {
                "config": asdict(self.cfg),
                "obs_dim": self.obs_dim,
                "action_low": self.action_low,
                "action_high": self.action_high,
                "actor": self.actor.state_dict(),
                "critic1": self.critic1.state_dict(),
                "critic2": self.critic2.state_dict(),
                "target1": self.target1.state_dict(),
                "target2": self.target2.state_dict(),
                "actor_optimizer": self.actor_optimizer.state_dict(),
                "critic_optimizer": self.critic_optimizer.state_dict(),
                "alpha_optimizer": self.alpha_optimizer.state_dict(),
                "log_alpha": self.log_alpha.detach().cpu(),
                "updates": self.updates,
            },
            temporary,
        )
        temporary.replace(path)

    def load(self, path: Path) -> None:
        payload = torch.load(path, map_location=self.device)
        if int(payload["obs_dim"]) != self.obs_dim:
            raise ValueError("Checkpoint observation dimension mismatch")
        if not np.allclose(payload["action_low"], self.action_low) or not np.allclose(
            payload["action_high"], self.action_high
        ):
            raise ValueError("Checkpoint action bounds mismatch")
        self.actor.load_state_dict(payload["actor"])
        self.critic1.load_state_dict(payload["critic1"])
        self.critic2.load_state_dict(payload["critic2"])
        self.target1.load_state_dict(payload["target1"])
        self.target2.load_state_dict(payload["target2"])
        if "actor_optimizer" in payload:
            self.actor_optimizer.load_state_dict(payload["actor_optimizer"])
            self.critic_optimizer.load_state_dict(payload["critic_optimizer"])
            self.alpha_optimizer.load_state_dict(payload["alpha_optimizer"])
        with torch.no_grad():
            self.log_alpha.copy_(payload["log_alpha"].to(self.device))
        self.updates = int(payload.get("updates", 0))
```

## M8 — `tests/test_sac.py`

```python
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from src.agents.sac import SACAgent, SACConfig


def _agent(seed=1):
    return SACAgent(
        obs_dim=14,
        action_low=np.array([0.0, -1.5], dtype=np.float32),
        action_high=np.array([0.41, 1.5], dtype=np.float32),
        cfg=SACConfig(batch_size=8, replay_capacity=64, hidden_size=32),
        seed=seed,
    )


def test_action_respects_asymmetric_bounds():
    agent = _agent()
    for _ in range(20):
        action = agent.select_action(np.zeros(14, dtype=np.float32))
        assert 0.0 <= action[0] <= 0.41
        assert -1.5 <= action[1] <= 1.5


def test_update_is_finite_and_checkpoint_round_trips(tmp_path):
    agent = _agent()
    rng = np.random.RandomState(2)
    for _ in range(16):
        obs = rng.uniform(-1, 1, size=14).astype(np.float32)
        next_obs = rng.uniform(-1, 1, size=14).astype(np.float32)
        action = np.array(
            [rng.uniform(0, 0.41), rng.uniform(-1.5, 1.5)], dtype=np.float32
        )
        agent.replay.add(obs, action, rng.randn(), next_obs, False)
    metrics = agent.update()
    assert metrics
    assert all(np.isfinite(value) for value in metrics.values())

    path = tmp_path / "agent.pt"
    before = agent.select_action(np.zeros(14, dtype=np.float32), deterministic=True)
    agent.save(path)
    restored = _agent(seed=99)
    restored.load(path)
    after = restored.select_action(np.zeros(14, dtype=np.float32), deterministic=True)
    assert np.allclose(before, after)
```

## M8 — `tests/test_continuous_env.py`

```python
from pathlib import Path

import numpy as np
import yaml

from src.continuous_env import build_continuous_env


def test_real_environment_reset_and_step_fit_spaces():
    with Path("configs/sac_lane.yaml").open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    config["environment"]["max_steps"] = 24
    env = build_continuous_env(config, seed=7)
    try:
        observation = env.reset(7)
        assert env.observation_space.contains(observation)
        action = np.array([0.10, 0.0], dtype=np.float32)
        observation, reward, done, info = env.step(action)
        assert env.observation_space.contains(observation)
        assert np.isfinite(reward)
        assert np.allclose(info["command_v_omega"], [0.10, 0.0])
        assert info["termination_reason"] in {
            "in_progress",
            "timeout",
            "offroad",
            "other_collision",
            "duck_collision",
            "goal",
        }
    finally:
        env.close()
```

## M8 — `scripts/check_sac_compatibility.py`

```python
"""Smoke test dependency, wrapper, update SAC, dan checkpoint."""

import argparse
import tempfile
from pathlib import Path

import gym
import numpy as np
import torch
import yaml

from src.agents.sac import SACAgent, SACConfig
from src.continuous_env import build_continuous_env


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/sac_lane.yaml"))
    parser.add_argument("--transitions", type=int, default=1000)
    args = parser.parse_args()

    with args.config.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    seed = int(config["seed"])
    device = str(config["training"].get("device", "cpu"))
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("Config meminta CUDA tetapi CUDA tidak tersedia")
    env = build_continuous_env(config, seed)
    env.action_space.seed(seed)
    agent = SACAgent(
        obs_dim=env.observation_space.shape[0],
        action_low=env.action_space.low,
        action_high=env.action_space.high,
        cfg=SACConfig(**config["sac"]),
        seed=seed,
        device=device,
    )

    observation = env.reset(seed)
    last_metrics = {}
    try:
        for step in range(args.transitions):
            if not env.observation_space.contains(observation):
                raise AssertionError(f"invalid observation at step {step}")
            action = env.action_space.sample()
            next_observation, reward, done, info = env.step(action)
            agent.replay.add(
                observation,
                action,
                reward,
                next_observation,
                bool(info["terminated"]),
            )
            last_metrics = agent.update() or last_metrics
            observation = next_observation
            if done:
                observation = env.reset(seed + step + 1)

        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "smoke.pt"
            agent.save(checkpoint)
            restored = SACAgent(
                env.observation_space.shape[0],
                env.action_space.low,
                env.action_space.high,
                SACConfig(**config["sac"]),
                seed + 99,
                device=device,
            )
            restored.load(checkpoint)
            action = restored.select_action(observation, deterministic=True)
            if not env.action_space.contains(action):
                raise AssertionError("loaded policy produced invalid action")
    finally:
        env.close()

    print(f"python_torch={torch.__version__}")
    print(f"device={agent.device}")
    print(f"gym={gym.__version__}")
    print(f"transitions={args.transitions}")
    print(f"updates={agent.updates}")
    print(f"last_metrics={last_metrics}")
    print("compatibility_smoke=passed")


if __name__ == "__main__":
    main()
```

---

## Urutan eksekusi tanpa melompati gate

Semua command dijalankan dari root `duckie-mdp`.

### 1. Gate M7

```bash
.venv/bin/python -m pytest -q tests/test_continuous_curvature.py tests/test_continuous_state.py
.venv/bin/python -m pytest -q
```

### 2. Gate dependency dan M8

```bash
.venv-sac/bin/python -m pytest -q tests/test_sac.py tests/test_continuous_env.py
.venv-sac/bin/python -m scripts.check_sac_compatibility \
  --config configs/sac_lane.yaml \
  --transitions 1000
```

Output wajib berakhir dengan `compatibility_smoke=passed`. Jangan train panjang
jika ada observation di luar bounds, NaN, checkpoint mismatch, atau test tabular
gagal.

### 3. Train dan pilih checkpoint lane

```bash
.venv-sac/bin/python -m src.train_sac --config configs/sac_lane.yaml
.venv-sac/bin/python -m src.select_sac_checkpoint \
  --config configs/sac_lane.yaml \
  --checkpoint-dir runs/sac_lane/checkpoints \
  --output runs/sac_lane/sac_best.pt
.venv-sac/bin/python -m src.evaluate_sac \
  --config configs/sac_lane.yaml \
  --checkpoint runs/sac_lane/sac_best.pt \
  --episodes 100 \
  --output runs/sac_lane/evaluation_final.json
.venv-sac/bin/python -m src.render_sac_video \
  --config configs/sac_lane.yaml \
  --checkpoint runs/sac_lane/sac_best.pt \
  --output runs/sac_lane/lane_camera.mp4 \
  --fps 20 \
  --view camera
```

Lane lulus jika memenuhi bar pada `docs/continuous_sac_plan.md`. Jangan lanjut
curriculum stop hanya karena video terlihat bagus; gunakan JSON evaluasi.

### 4. Train stop-only

Pastikan `configs/sac_stop.yaml` sudah merupakan salinan penuh lane config dengan
replacement block stop dan evaluation brake ratio 0,25.

```bash
.venv-sac/bin/python -m src.train_sac --config configs/sac_stop.yaml
.venv-sac/bin/python -m src.select_sac_checkpoint \
  --config configs/sac_stop.yaml \
  --checkpoint-dir runs/sac_stop/checkpoints \
  --output runs/sac_stop/sac_best.pt
.venv-sac/bin/python -m src.evaluate_sac \
  --config configs/sac_stop.yaml \
  --checkpoint runs/sac_stop/sac_best.pt \
  --episodes 100 \
  --output runs/sac_stop/evaluation_final.json
```

### 5. Train full stop + Duckie

```bash
.venv-sac/bin/python -m src.train_sac --config configs/sac_full.yaml
.venv-sac/bin/python -m src.select_sac_checkpoint \
  --config configs/sac_full.yaml \
  --checkpoint-dir runs/sac_full/checkpoints \
  --output runs/sac_full/sac_best.pt
.venv-sac/bin/python -m src.evaluate_sac \
  --config configs/sac_full.yaml \
  --checkpoint runs/sac_full/sac_best.pt \
  --episodes 100 \
  --output runs/sac_full/evaluation_final.json
.venv-sac/bin/python -m src.render_sac_video \
  --config configs/sac_full.yaml \
  --checkpoint runs/sac_full/sac_best.pt \
  --output runs/sac_full/full_camera.mp4 \
  --fps 20 \
  --view camera
```

### 6. Perbandingan baseline yang sah

- SAC teacher-free dibandingkan dengan Q-learning teacher-free.
- Keduanya memakai `frame_skip=6`, reward kanonis, horizon, map, dan final seeds
  yang sama.
- Q-table lama cukup dievaluasi ulang; jangan dilatih ulang hanya untuk membuat
  hasilnya kalah/menang.
- Teacher-guided Q-learning/SARSA dilaporkan sebagai assisted track terpisah.
- Jika SAC kalah tetapi melewati engineering bar, laporkan sebagai hasil negatif
  yang valid.

## M10 — DQN ablation

Implementasi executable tersedia di `src/train_dqn.py` dan
`configs/dqn_continuous_state.yaml`. M10 menjawab pertanyaan berbeda:
continuous state dengan action tetap diskrit. Ia bukan syarat keberhasilan
continuous-action SAC. Pipeline memakai observation 14-dimensi yang sama,
tujuh macro-action lama (brake dimask pada lane-only), reward/seeds yang sama,
dan replay DQN terpisah. Eksperimen penuhnya baru dijalankan setelah checkpoint
M9 dibekukan.
Keputusan menunda kode ini mencegah perubahan scope sebelum eksperimen utama
selesai.

## Checklist file sebelum menjalankan kode

- [ ] semua file Python baru memiliki isi sesuai blok, bukan masih 0 byte;
- [ ] `sac_stop.yaml` dan `sac_full.yaml` adalah YAML lengkap, bukan hanya blok
  pengganti;
- [ ] `.venv` tabular tidak dipasangi Torch/SB3;
- [ ] `.venv-sac` dibuat terpisah;
- [ ] test M7, M8, dan seluruh test lama hijau;
- [ ] `frame_skip=6` pada ketiga config SAC;
- [ ] teacher tidak pernah dipanggil oleh wrapper/train/evaluator SAC;
- [ ] timeout disimpan ke replay dengan `terminated=False`;
- [ ] checkpoint best dipilih hanya dari development seeds;
- [ ] final seeds hanya dipakai setelah selection dibekukan;
- [ ] video ditulis 20 FPS berdasarkan waktu simulator.

## M9 — `src/evaluate_sac.py`

```python
"""Deterministic teacher-free evaluation untuk checkpoint SAC."""

import argparse
import json
from collections import Counter
from math import cos, hypot
from pathlib import Path
from typing import Dict, Optional, Sequence

import numpy as np
import yaml

from .agents.sac import SACAgent, SACConfig
from .continuous_env import build_continuous_env


def evaluate_policy(
    config_path: Path,
    checkpoint_path: Path,
    episodes: Optional[int] = None,
    seeds: Optional[Sequence[int]] = None,
) -> Dict[str, float]:
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    evaluation = config["evaluation"]
    if episodes is None:
        episodes = int(evaluation["final_episodes"])
    if seeds is None:
        seeds = [int(value) for value in evaluation["final_seeds"]]

    env = build_continuous_env(config, int(seeds[0]))
    agent = SACAgent(
        env.observation_space.shape[0],
        env.action_space.low,
        env.action_space.high,
        SACConfig(**config["sac"]),
        seed=int(config["seed"]),
        device=str(config["training"].get("device", "cpu")),
    )
    agent.load(checkpoint_path)

    decision_dt = env.unwrapped.delta_time * int(config["environment"]["frame_skip"])
    brake_threshold = float(evaluation["brake_command_threshold"])
    move_threshold = float(evaluation["move_command_threshold"])
    spin_threshold = float(evaluation["spin_omega_threshold"])
    resume_window = int(evaluation["resume_window_steps"])
    min_progress = float(evaluation["success_min_progress_m"])
    max_brake_ratio = float(evaluation["success_max_brake_ratio"])

    reasons = Counter()
    returns = []
    progresses = []
    deviations = []
    brake_ratios = []
    task_success = []
    stop_successes = stop_violations = 0
    false_stop_steps = spin_steps = total_steps = 0
    crossing_steps = yield_steps = 0
    resume_opportunities = resume_successes = 0
    minimum_duck_distances = []

    try:
        for episode in range(episodes):
            seed = int(seeds[episode % len(seeds)]) + episode
            observation = env.reset(seed)
            done = False
            total_return = progress = 0.0
            episode_steps = episode_brakes = 0
            pending_resume = 0
            previous_active = env.current_state.duck_active
            info = {"termination_reason": "in_progress"}

            while not done:
                state_before = env.current_state
                action = agent.select_action(observation, deterministic=True)
                is_brake = float(action[0]) < brake_threshold
                unmet_stop = (
                    state_before.stop_present
                    and state_before.d_stop is not None
                    and state_before.d_stop <= env.mdp_env.reward_cfg.stop_exemption_distance
                    and not state_before.sigma_stop
                )
                crossing_before = state_before.duck_active
                if is_brake and not crossing_before and not unmet_stop:
                    false_stop_steps += 1
                if is_brake and abs(float(action[1])) >= spin_threshold:
                    spin_steps += 1

                observation, reward, done, info = env.step(action)
                state_after = env.current_state
                total_return += float(reward)
                episode_steps += 1
                total_steps += 1
                episode_brakes += int(is_brake)
                deviations.append(abs(state_after.d))
                progress += max(0.0, state_after.v * cos(state_after.phi)) * decision_dt

                events = info["events"]
                stop_successes += int(events["full_stop"])
                stop_violations += int(events["stop_violation"])
                crossing_steps += int(state_after.duck_active)
                yield_steps += int(state_after.duck_active and is_brake)
                if state_after.duck_present:
                    minimum_duck_distances.append(
                        hypot(state_after.duck_longitudinal, state_after.duck_lateral)
                    )

                if previous_active and not state_after.duck_active:
                    resume_opportunities += 1
                    pending_resume = resume_window
                if pending_resume > 0:
                    if float(action[0]) >= move_threshold:
                        resume_successes += 1
                        pending_resume = 0
                    else:
                        pending_resume -= 1
                previous_active = state_after.duck_active

            reasons[info["termination_reason"]] += 1
            returns.append(total_return)
            progresses.append(progress)
            brake_ratio = episode_brakes / max(1, episode_steps)
            brake_ratios.append(brake_ratio)
            task_success.append(
                info["termination_reason"] == "timeout"
                and progress >= min_progress
                and brake_ratio <= max_brake_ratio
            )
    finally:
        env.close()

    stop_opportunities = stop_successes + stop_violations
    failures = (
        reasons["duck_collision"]
        + reasons["other_collision"]
        + reasons["offroad"]
    )
    return {
        "stage": config.get("stage", "unknown"),
        "episodes": episodes,
        "mean_return": float(np.mean(returns)),
        "mean_forward_progress_m": float(np.mean(progresses)),
        "mean_abs_d": float(np.mean(deviations)),
        "p95_abs_d": float(np.percentile(deviations, 95)),
        "mean_brake_ratio": float(np.mean(brake_ratios)),
        "task_success_rate": float(np.mean(task_success)),
        "timeout_rate": reasons["timeout"] / episodes,
        "offroad_rate": reasons["offroad"] / episodes,
        "duck_collision_rate": reasons["duck_collision"] / episodes,
        "other_collision_rate": reasons["other_collision"] / episodes,
        "total_failure_rate": failures / episodes,
        "stop_compliance_rate": (
            stop_successes / stop_opportunities if stop_opportunities else 1.0
        ),
        "stop_opportunities": stop_opportunities,
        "false_stop_rate": false_stop_steps / max(1, total_steps),
        "spin_in_place_rate": spin_steps / max(1, total_steps),
        "duck_yield_step_rate": yield_steps / max(1, crossing_steps),
        "resume_after_clear_rate": (
            resume_successes / resume_opportunities if resume_opportunities else 1.0
        ),
        "resume_opportunities": resume_opportunities,
        "minimum_duck_distance_m": (
            float(np.min(minimum_duck_distances))
            if minimum_duck_distances
            else None
        ),
        "termination_counts": dict(reasons),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = evaluate_policy(args.config, args.checkpoint, args.episodes)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
```

## M9 — `src/render_sac_video.py`

```python
"""Render deterministic SAC pada 20 FPS tanpa mempercepat waktu simulasi."""

import argparse
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import yaml

from .agents.sac import SACAgent, SACConfig
from .continuous_env import build_continuous_env


def capture_frame(env, view):
    if view == "camera":
        frame = env.unwrapped.render_obs()
    elif view == "follow":
        frame = env.unwrapped.render(mode="rgb_array")
    else:
        raise ValueError(f"unsupported view: {view}")
    frame = np.asarray(frame)
    if frame.dtype != np.uint8:
        frame = np.clip(frame, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(frame)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=10101)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--view", choices=("camera", "follow"), default="camera")
    args = parser.parse_args()

    with args.config.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    config["environment"]["render_observations"] = True
    env = build_continuous_env(config, args.seed)
    agent = SACAgent(
        env.observation_space.shape[0],
        env.action_space.low,
        env.action_space.high,
        SACConfig(**config["sac"]),
        seed=int(config["seed"]),
        device=str(config["training"].get("device", "cpu")),
    )
    agent.load(args.checkpoint)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    writer = imageio.get_writer(
        str(args.output),
        format="FFMPEG",
        mode="I",
        fps=args.fps,
        codec="libx264",
        quality=8,
        macro_block_size=16,
        ffmpeg_log_level="warning",
    )
    decision_seconds = env.unwrapped.delta_time * int(
        config["environment"]["frame_skip"]
    )
    video_clock = 0.0
    decisions = 0
    final_reason = "in_progress"
    try:
        observation = env.reset(args.seed)
        writer.append_data(capture_frame(env, args.view))
        done = False
        while not done:
            action = agent.select_action(observation, deterministic=True)
            observation, _, done, info = env.step(action)
            decisions += 1
            final_reason = info["termination_reason"]
            frame = capture_frame(env, args.view)
            video_clock += args.fps * decision_seconds
            while video_clock >= 1.0:
                writer.append_data(frame)
                video_clock -= 1.0
    finally:
        writer.close()
        env.close()
    print(f"video={args.output.resolve()}")
    print(f"fps={args.fps} decisions={decisions} reason={final_reason}")


if __name__ == "__main__":
    main()
```

Renderer menahan satu action selama `frame_skip=6` seperti training. Frame yang
sama diulang sesuai waktu simulator sehingga video 20 FPS tidak membuat robot
terlihat empat kali lebih cepat.

## Penyesuaian evaluasi pada config stop/full

Pada `sac_stop.yaml` dan `sac_full.yaml`, ubah nilai ini karena brake memang
dibutuhkan:

```yaml
evaluation:
  development_episodes: 30
  final_episodes: 100
  development_seeds: [1101, 1202, 1303, 1404, 1505]
  final_seeds: [10101, 10202, 10303, 10404, 10505]
  success_min_progress_m: 5.0
  success_max_brake_ratio: 0.25
  brake_command_threshold: 0.04
  move_command_threshold: 0.10
  spin_omega_threshold: 0.75
  resume_window_steps: 20
```


## M9 — `src/select_sac_checkpoint.py`

```python
"""Pilih checkpoint best hanya memakai development seeds."""

import argparse
import json
import shutil
from pathlib import Path

import yaml

from .evaluate_sac import evaluate_policy


def checkpoint_score(report):
    failure = report["total_failure_rate"]
    if report["stage"] == "lane":
        return (
            report["task_success_rate"],
            -failure,
            report["mean_return"],
            -report["p95_abs_d"],
        )
    return (
        report["task_success_rate"],
        -failure,
        report["stop_compliance_rate"],
        -report["false_stop_rate"],
        report["mean_return"],
        -report["p95_abs_d"],
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    with args.config.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    evaluation = config["evaluation"]
    episodes = int(evaluation["development_episodes"])
    seeds = [int(value) for value in evaluation["development_seeds"]]
    candidates = sorted(args.checkpoint_dir.glob("sac_step_*.pt"))
    if not candidates:
        raise FileNotFoundError(f"No checkpoint in {args.checkpoint_dir}")

    records = []
    best = None
    best_score = None
    for checkpoint in candidates:
        report = evaluate_policy(args.config, checkpoint, episodes, seeds)
        score = checkpoint_score(report)
        records.append(
            {"checkpoint": str(checkpoint), "score": list(score), "report": report}
        )
        # sorted path membuat tie terakhir menjadi step terbesar; karena plan
        # memilih step lebih kecil, update hanya untuk score yang benar-benar >.
        if best_score is None or score > best_score:
            best, best_score = checkpoint, score
        print(f"checkpoint={checkpoint.name} score={score}", flush=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best, args.output)
    selection = args.output.with_name("checkpoint_selection.json")
    selection.write_text(
        json.dumps(
            {"selected": str(best), "output": str(args.output), "records": records},
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"best={best}")
    print(f"copied_to={args.output}")


if __name__ == "__main__":
    main()
```


## M8/M9 — `configs/sac_lane.yaml`

```yaml
algorithm: sac
stage: lane
seed: 73
environment:
  map_name: small_loop
  domain_rand: false
  max_steps: 1500
  frame_skip: 6
  render_observations: false
  accept_start_angle_deg: 10
  spawn_max_abs_d: 0.08
  spawn_max_abs_phi: 0.175
  spawn_attempts: 50
  user_tile_start: null
  goal_tile: null
state:
  stop_lateral_limit: 0.40
  stop_orientation_cos: 0.70710678
  sign_to_line_offset: 0.20
  stop_max_distance: 3.0
  stop_zone: 0.45
  stop_pass_distance: 0.55
  stop_speed: 0.02
  tile_lookahead: 0.30
  curvature_threshold: 0.05
  duck_max_distance: 1.20
  duck_near_distance: 0.60
  duck_corridor_width: 0.60
continuous_state:
  max_speed: 0.41
  max_abs_curvature: 8.0
  max_stop_distance: 3.0
  max_duck_distance: 2.0
  max_relative_speed: 0.50
  curvature_samples: 33
actions:
  v_fast: 0.41
  v_slow: 0.17
  w0: 1.50
  wheel_base: 0.102
duck_controller:
  p_cross: 0.0
  make_dynamic: false
  require_duck: false
  inject_if_missing: false
  spawn_pos: [1.62, 0.50]
  spawn_rotate: 0.0
  spawn_height: 0.08
  walk_distance: 0.90
  trigger_min_ego_distance: 0.35
  trigger_max_ego_distance: 0.45
  max_crossings_per_episode: 0
  inject_stop_if_missing: false
  require_stop: false
  stop_spawn_pos: [1.20, 2.10]
  stop_spawn_rotate: 180.0
  stop_spawn_height: 0.18
reward:
  alpha_progress: 1.0
  alpha_lateral: 2.0
  alpha_heading: 0.5
  step_cost: 0.01
  collision_duck: -200.0
  other_collision: -200.0
  offroad: -200.0
  stop_violation: -40.0
  full_stop: 15.0
  duck_yield: 0.0
  duck_unsafe: -5.0
  duck_yield_speed: 0.04
  unnecessary_stop: -2.0
  idle_speed: 0.04
  stop_exemption_distance: 0.45
  goal: 50.0
sac:
  gamma: 0.99
  tau: 0.005
  actor_lr: 0.0003
  critic_lr: 0.0003
  alpha_lr: 0.0003
  initial_alpha: 0.2
  batch_size: 256
  replay_capacity: 300000
  hidden_size: 256
  target_entropy: -2.0
training:
  total_steps: 300000
  random_steps: 5000
  gradient_steps: 1
  checkpoint_interval: 25000
  log_interval: 1000
  output_dir: runs/sac_lane
  initial_checkpoint: null
  device: cuda
evaluation:
  development_episodes: 30
  final_episodes: 100
  development_seeds: [1101, 1202, 1303, 1404, 1505]
  final_seeds: [10101, 10202, 10303, 10404, 10505]
  success_min_progress_m: 5.0
  success_max_brake_ratio: 0.05
  brake_command_threshold: 0.04
  move_command_threshold: 0.10
  spin_omega_threshold: 0.75
  resume_window_steps: 20
```

## M9 — `configs/sac_stop.yaml`

Gunakan isi `sac_lane.yaml` di atas, lalu ubah tepat blok berikut. Bagian lain
harus identik; ini sengaja berupa patch kecil agar tidak muncul dua sumber
kebenaran reward/hyperparameter.

```yaml
stage: stop

duck_controller:
  p_cross: 0.0
  make_dynamic: true
  require_duck: false
  inject_if_missing: false
  spawn_pos: [1.62, 0.50]
  spawn_rotate: 0.0
  spawn_height: 0.08
  walk_distance: 0.90
  trigger_min_ego_distance: 0.35
  trigger_max_ego_distance: 0.45
  max_crossings_per_episode: 0
  inject_stop_if_missing: true
  require_stop: true
  stop_spawn_pos: [1.20, 2.10]
  stop_spawn_rotate: 180.0
  stop_spawn_height: 0.18

training:
  total_steps: 300000
  random_steps: 5000
  gradient_steps: 1
  checkpoint_interval: 50000
  log_interval: 1000
  output_dir: runs/sac_stop
  initial_checkpoint: runs/sac_lane/sac_best.pt
```

Cara aman membuatnya: salin seluruh `sac_lane.yaml` ke `sac_stop.yaml`, lalu
ganti tiga blok di atas. Jangan menaruh literal teks `Gunakan isi...` ke YAML.

## M9 — `configs/sac_full.yaml`

Salin seluruh `sac_stop.yaml`, lalu ganti blok berikut:

```yaml
stage: full

duck_controller:
  p_cross: 1.0
  make_dynamic: true
  require_duck: true
  inject_if_missing: true
  spawn_pos: [1.62, 0.50]
  spawn_rotate: 0.0
  spawn_height: 0.08
  walk_distance: 0.90
  trigger_min_ego_distance: 0.35
  trigger_max_ego_distance: 0.45
  max_crossings_per_episode: 1
  inject_stop_if_missing: true
  require_stop: true
  stop_spawn_pos: [1.20, 2.10]
  stop_spawn_rotate: 180.0
  stop_spawn_height: 0.18

training:
  total_steps: 1000000
  random_steps: 5000
  gradient_steps: 1
  checkpoint_interval: 50000
  log_interval: 1000
  output_dir: runs/sac_full
  initial_checkpoint: runs/sac_stop/sac_best.pt
```

## M9 — `src/train_sac.py`

```python
"""Teacher-free SAC training dengan replay buffer dan checkpoint atomik."""

import argparse
import csv
import shutil
from pathlib import Path

import numpy as np
import yaml

from .agents.sac import SACAgent, SACConfig
from .continuous_env import build_continuous_env


def _write_rows(rows, path: Path):
    if not rows:
        return
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def train(config_path: Path):
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    seed = int(config["seed"])
    training = config["training"]
    output = Path(training["output_dir"])
    checkpoints = output / "checkpoints"
    checkpoints.mkdir(parents=True, exist_ok=True)
    shutil.copy2(config_path, output / "config.yaml")

    env = build_continuous_env(config, seed)
    env.action_space.seed(seed)
    agent = SACAgent(
        obs_dim=env.observation_space.shape[0],
        action_low=env.action_space.low,
        action_high=env.action_space.high,
        cfg=SACConfig(**config["sac"]),
        seed=seed,
        device=str(training.get("device", "cpu")),
    )
    initial = training.get("initial_checkpoint")
    if initial:
        agent.load(Path(initial))
        print(f"initial_checkpoint={initial}", flush=True)

    total_steps = int(training["total_steps"])
    random_steps = int(training["random_steps"])
    gradient_steps = int(training.get("gradient_steps", 1))
    checkpoint_interval = int(training["checkpoint_interval"])
    log_interval = int(training["log_interval"])
    rows = []
    episode = 0
    episode_return = 0.0
    episode_steps = 0
    observation = env.reset(seed)
    last_metrics = {}

    try:
        for step in range(1, total_steps + 1):
            if step <= random_steps:
                action = env.action_space.sample()
            else:
                action = agent.select_action(observation, deterministic=False)

            next_observation, reward, done, info = env.step(action)
            agent.replay.add(
                observation,
                action,
                reward,
                next_observation,
                bool(info["terminated"]),
            )
            observation = next_observation
            episode_return += float(reward)
            episode_steps += 1

            if step > random_steps:
                for _ in range(gradient_steps):
                    metrics = agent.update()
                    if metrics:
                        last_metrics = metrics

            if done:
                rows.append(
                    {
                        "environment_step": step,
                        "physics_step": step * int(config["environment"]["frame_skip"]),
                        "episode": episode,
                        "episode_return": episode_return,
                        "episode_steps": episode_steps,
                        "termination_reason": info["termination_reason"],
                        "terminated": int(info["terminated"]),
                        "truncated": int(info["truncated"]),
                        "alpha": last_metrics.get("alpha", np.nan),
                        "critic_loss": last_metrics.get("critic_loss", np.nan),
                        "actor_loss": last_metrics.get("actor_loss", np.nan),
                    }
                )
                episode += 1
                episode_return = 0.0
                episode_steps = 0
                observation = env.reset(seed + episode)

            if step % log_interval == 0:
                recent = rows[-20:]
                mean_return = (
                    float(np.mean([row["episode_return"] for row in recent]))
                    if recent
                    else np.nan
                )
                print(
                    f"step={step} episode={episode} mean_return_20={mean_return:.3f} "
                    f"alpha={last_metrics.get('alpha', np.nan):.4f} "
                    f"replay={len(agent.replay)}",
                    flush=True,
                )
                _write_rows(rows, output / "training.csv")

            if step % checkpoint_interval == 0:
                agent.save(checkpoints / f"sac_step_{step:09d}.pt")

        agent.save(output / "sac_final.pt")
        _write_rows(rows, output / "training.csv")
    except KeyboardInterrupt:
        agent.save(output / "sac_interrupted.pt")
        _write_rows(rows, output / "training_partial.csv")
        raise
    finally:
        env.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    train(args.config)


if __name__ == "__main__":
    main()
```
