from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Sequence

import numpy as np


@dataclass(frozen=True)
class DuckControllerConfig:
    p_cross: float = 0.02
    make_dynamic: bool = True
    require_duck: bool = True
    inject_if_missing: bool = False
    spawn_pos: Sequence[float] = (1.62, 0.50)
    spawn_rotate: float = 0.0
    spawn_height: float = 0.08
    walk_distance: float = 0.90
    trigger_min_ego_distance: float = 0.55
    trigger_max_ego_distance: float = 1.10
    # Nol mempertahankan perilaku lama (crossing berulang tanpa batas).
    # Full task memakai satu crossing agar Duckie tidak langsung berbalik arah
    # ketika ego masih sedang yield di lokasi yang sama.
    max_crossings_per_episode: int = 0
    inject_stop_if_missing: bool = False
    require_stop: bool = False
    stop_spawn_pos: Sequence[float] = (1.20, 2.10)
    stop_spawn_rotate: float = 180.0
    stop_spawn_height: float = 0.18


def _kind(value: Any) -> str:
    return str(getattr(value, "value", value)).lower()


def _is_duck(obj: Any) -> bool:
    return _kind(getattr(obj, "kind", "")) == "duckie"


def _object_descriptions(map_data: Dict[str, Any]) -> Iterable[Dict[str, Any]]:
    objects = map_data.get("objects", {})
    return objects.values() if isinstance(objects, dict) else objects


def _inject_duck(map_data: Dict[str, Any], cfg: DuckControllerConfig) -> None:
    description = {
        "kind": "duckie",
        "pos": [float(cfg.spawn_pos[0]), float(cfg.spawn_pos[1])],
        "rotate": float(cfg.spawn_rotate),
        "height": float(cfg.spawn_height),
        "optional": False,
        "static": False,
    }
    objects = map_data.setdefault("objects", {})
    if isinstance(objects, dict):
        objects["mdp_duckie"] = description
    else:
        objects.append(description)


def _inject_stop(map_data: Dict[str, Any], cfg: DuckControllerConfig) -> None:
    """Tambahkan stop sign pada ruas bawah, jauh dari crossing Duckie.

    Koordinat object Duckietown menggunakan frame YAML. Default menghasilkan
    world position sekitar (0.702, 1.814) pada small_loop. Rotasi 180 derajat
    membuat papan menghadap kendaraan yang bergerak ke timur.
    """
    description = {
        "kind": "sign_stop",
        "pos": [float(cfg.stop_spawn_pos[0]), float(cfg.stop_spawn_pos[1])],
        "rotate": float(cfg.stop_spawn_rotate),
        "height": float(cfg.stop_spawn_height),
        "optional": False,
        "static": True,
    }
    objects = map_data.setdefault("objects", {})
    if isinstance(objects, dict):
        objects["mdp_stop_sign"] = description
    else:
        objects.append(description)


def prepare_task_map_data(
    map_data: Dict[str, Any], cfg: DuckControllerConfig
) -> tuple[Dict[str, Any], int, int]:
    """Buat salinan map dengan Duckie dinamis dan stop sign deterministik."""
    prepared = deepcopy(map_data)
    ducks = [desc for desc in _object_descriptions(prepared) if desc.get("kind") == "duckie"]
    if not ducks and cfg.inject_if_missing:
        _inject_duck(prepared, cfg)
        ducks = [desc for desc in _object_descriptions(prepared) if desc.get("kind") == "duckie"]
    if cfg.require_duck and not ducks:
        raise ValueError("Map contains no Duckie object")
    for desc in ducks:
        desc["static"] = False
        desc["optional"] = False

    stops = [desc for desc in _object_descriptions(prepared) if desc.get("kind") == "sign_stop"]
    if not stops and cfg.inject_stop_if_missing:
        _inject_stop(prepared, cfg)
        stops = [desc for desc in _object_descriptions(prepared) if desc.get("kind") == "sign_stop"]
    if cfg.require_stop and not stops:
        raise ValueError("Map contains no stop sign")
    return prepared, len(ducks), len(stops)


def make_ducks_dynamic(env: Any, cfg: DuckControllerConfig) -> int:
    map_data, duck_count, stop_count = prepare_task_map_data(env.map_data, cfg)
    if duck_count or stop_count:
        env.map_data = map_data
        env._interpret_map(map_data)
    return duck_count


class DuckController:
    def __init__(self, env: Any, cfg: DuckControllerConfig, seed: int) -> None:
        self.env = env
        self.cfg = cfg
        self.rng = np.random.RandomState(seed)
        self.ducks: List[Any] = [obj for obj in env.objects if _is_duck(obj)]
        if cfg.require_duck and not self.ducks:
            raise ValueError(f"Map {env.map_name!r} contains no usable Duckie")
        if cfg.make_dynamic:
            invalid = [duck for duck in self.ducks if not hasattr(duck, "pedestrian_active")]
            if invalid:
                raise TypeError("Duckies must be converted to dynamic DuckieObj instances")
            for duck in self.ducks:
                duck.walk_distance = float(cfg.walk_distance)
        self.initial = [self._snapshot(duck) for duck in self.ducks]
        self.crossings_started = [0 for _ in self.ducks]

    @staticmethod
    def _snapshot(duck: Any) -> Dict[str, Any]:
        return {
            "pos": np.array(duck.pos, copy=True),
            "center": np.array(getattr(duck, "center", duck.pos), copy=True),
            "angle": float(duck.angle),
            "heading": np.array(getattr(duck, "heading", [0.0, 0.0]), copy=True),
            "corners": np.array(duck.obj_corners, copy=True),
            "norm": np.array(duck.obj_norm, copy=True),
            "vel": float(getattr(duck, "vel", 0.02)),
        }

    def reset(self, seed: int = None) -> None:
        if seed is not None:
            self.rng = np.random.RandomState(seed)
        for index, (duck, saved) in enumerate(zip(self.ducks, self.initial)):
            duck.pos = np.array(saved["pos"], copy=True)
            duck.center = np.array(saved["center"], copy=True)
            duck.start = np.array(saved["center"], copy=True)
            duck.angle = saved["angle"]
            duck.y_rot = np.rad2deg(saved["angle"])
            duck.heading = np.array(saved["heading"], copy=True)
            duck.obj_corners = np.array(saved["corners"], copy=True)
            duck.obj_norm = np.array(saved["norm"], copy=True)
            duck.vel = saved["vel"]
            duck.pedestrian_active = False
            duck.pedestrian_wait_time = float("inf")
            duck.time = 0.0
            self.crossings_started[index] = 0

    def before_step(self) -> None:
        for index, duck in enumerate(self.ducks):
            if not duck.pedestrian_active:
                duck.pedestrian_wait_time = float("inf")
                limit = int(self.cfg.max_crossings_per_episode)
                if limit > 0 and self.crossings_started[index] >= limit:
                    continue
                motion = np.asarray(duck.heading, dtype=float) * np.sign(float(duck.vel) or 1.0)
                crossing = np.asarray(duck.start, dtype=float) + 0.5 * float(duck.walk_distance) * motion
                rel = crossing - np.asarray(self.env.cur_pos, dtype=float)
                distance = float(np.linalg.norm(rel[[0, 2]]))
                _, tangent = self.env.closest_curve_point(self.env.cur_pos, self.env.cur_angle)
                ahead = tangent is not None and float(np.dot(rel, np.asarray(tangent, dtype=float))) > 0.0
                eligible = (
                    ahead
                    and self.cfg.trigger_min_ego_distance <= distance <= self.cfg.trigger_max_ego_distance
                )
                if eligible and self.rng.random_sample() < self.cfg.p_cross:
                    duck.pedestrian_active = True
                    duck.time = 0.0
                    self.crossings_started[index] += 1
