"""Reproducible branch-point manifests for simulator interventions."""

from dataclasses import asdict, dataclass
from enum import Enum
from hashlib import sha256
import json
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence, Tuple

from .primitives import PRIMITIVE_SCHEMA_VERSION
from .schema import PolicyMode, SolverKind


MANIFEST_SCHEMA_VERSION = "1.0.0"


class WorldMode(str, Enum):
    REACTIVE_WORLD = "reactive_world"
    SCRIPTED_WORLD = "scripted_world"


def _hash(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if hasattr(value, "tolist"):
        return _json_safe(value.tolist())
    if hasattr(value, "item"):
        return value.item()
    if isinstance(value, Enum):
        return value.value
    return value


def capture_rng_state(rng: Any) -> Mapping[str, Any]:
    if rng is None:
        return {"kind": "none"}
    if hasattr(rng, "get_state"):
        state = rng.get_state()
        # Gym's compatibility RNG exposes get_state for both the legacy
        # tuple-based RandomState and newer dict-based BitGenerator state.
        if isinstance(state, Mapping):
            return {"kind": "Generator", "state": _json_safe(state)}
        if isinstance(state, (tuple, list)) and len(state) == 5:
            return {
                "kind": "RandomState",
                "algorithm": state[0],
                "keys": _json_safe(state[1]),
                "position": int(state[2]),
                "has_gauss": int(state[3]),
                "cached_gaussian": float(state[4]),
            }
    bit_generator = getattr(rng, "bit_generator", None)
    if bit_generator is not None:
        return {"kind": "Generator", "state": _json_safe(bit_generator.state)}
    raise TypeError("unsupported RNG type: %s" % type(rng).__name__)


@dataclass(frozen=True)
class ScenarioManifest:
    reset_seed: int
    solver: SolverKind
    policy_mode: PolicyMode
    world_mode: WorldMode
    action_prefix: Tuple[Any, ...]
    map_name: str
    branch_physics_step: int
    branch_sim_time_seconds: float
    branch_ego_position: Tuple[float, float, float]
    branch_ego_heading: float
    stop_tracker_state: Mapping[str, Any]
    duck_controller_state: Mapping[str, Any]
    simulator_rng_state: Mapping[str, Any]
    controller_rng_state: Mapping[str, Any]
    config_path: str
    config_sha256: str
    checkpoint_path: str
    checkpoint_sha256: str
    exogenous_trace: Tuple[Mapping[str, Any], ...] = ()
    manifest_schema_version: str = MANIFEST_SCHEMA_VERSION
    primitive_schema_version: str = PRIMITIVE_SCHEMA_VERSION

    def payload(self) -> Mapping[str, Any]:
        return _json_safe(asdict(self))

    @property
    def manifest_id(self) -> str:
        canonical = json.dumps(
            self.payload(), sort_keys=True, separators=(",", ":"), allow_nan=False
        )
        return sha256(canonical.encode("utf-8")).hexdigest()

    def to_json(self, indent: Optional[int] = 2) -> str:
        payload = dict(self.payload())
        payload["manifest_id"] = self.manifest_id
        return json.dumps(payload, sort_keys=True, indent=indent, allow_nan=False)


def _controller(env: Any) -> Any:
    if hasattr(env, "mdp_env"):
        return env.mdp_env.duck_controller
    return getattr(env, "duck_controller", None)


def _controller_state(controller: Any) -> Mapping[str, Any]:
    if controller is None:
        return {}
    ducks = []
    for duck in controller.ducks:
        ducks.append(
            {
                "position": _json_safe(getattr(duck, "pos", ())),
                "heading": _json_safe(getattr(duck, "heading", ())),
                "velocity": float(getattr(duck, "vel", 0.0)),
                "active": bool(getattr(duck, "pedestrian_active", False)),
                "time": float(getattr(duck, "time", 0.0)),
            }
        )
    return {
        "crossings_started": [int(value) for value in controller.crossings_started],
        "crossing_armed": [bool(value) for value in controller.crossing_armed],
        "ducks": ducks,
    }


def capture_manifest(
    env: Any,
    reset_seed: int,
    action_prefix: Sequence[Any],
    solver: SolverKind,
    policy_mode: PolicyMode,
    config_path: Path,
    checkpoint_path: Path,
    world_mode: WorldMode = WorldMode.REACTIVE_WORLD,
    exogenous_trace: Sequence[Mapping[str, Any]] = (),
) -> ScenarioManifest:
    base = getattr(env, "unwrapped", env)
    controller = _controller(env)
    tracker_owner = getattr(env, "mdp_env", env)
    tracker = getattr(tracker_owner, "stop_tracker", None)
    config = Path(config_path)
    checkpoint = Path(checkpoint_path)
    position = tuple(float(value) for value in base.cur_pos)
    return ScenarioManifest(
        reset_seed=int(reset_seed),
        solver=solver,
        policy_mode=policy_mode,
        world_mode=world_mode,
        action_prefix=tuple(_json_safe(action) for action in action_prefix),
        map_name=str(base.map_name),
        branch_physics_step=int(base.step_count),
        branch_sim_time_seconds=float(base.step_count * base.delta_time),
        branch_ego_position=position,
        branch_ego_heading=float(base.cur_angle),
        stop_tracker_state={
            "sigma_stop": bool(getattr(tracker, "sigma_stop", False)),
            "hold_steps": int(getattr(tracker, "hold_steps", 0)),
        },
        duck_controller_state=_controller_state(controller),
        simulator_rng_state=capture_rng_state(getattr(base, "np_random", None)),
        controller_rng_state=capture_rng_state(
            getattr(controller, "rng", None) if controller is not None else None
        ),
        config_path=str(config),
        config_sha256=_hash(config),
        checkpoint_path=str(checkpoint),
        checkpoint_sha256=_hash(checkpoint),
        exogenous_trace=tuple(dict(item) for item in exogenous_trace),
    )
