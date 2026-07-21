"""Deterministic prefix replay gate used before counterfactual branching."""

from dataclasses import dataclass
from typing import Any, Callable, Mapping, Optional, Sequence, Tuple

import numpy as np

from ..discretizer import discretize
from ..state import DuckThreat, RawState, TileType


REPLAY_ATOL = 1e-7


def _raw(info: Mapping[str, Any]) -> RawState:
    value = info["raw_state"]
    return RawState(
        d=float(value["d"]),
        phi=float(value["phi"]),
        v=float(value["v"]),
        tile=TileType(int(value["tile"])),
        d_stop=None if value["d_stop"] is None else float(value["d_stop"]),
        sigma_stop=bool(value["sigma_stop"]),
        duck=DuckThreat(int(value["duck"])),
    )


def _controller(env: Any) -> Any:
    if hasattr(env, "mdp_env"):
        return env.mdp_env.duck_controller
    return getattr(env, "duck_controller", None)


def _controller_phase(env: Any) -> Tuple[Any, ...]:
    controller = _controller(env)
    if controller is None:
        return ()
    result = []
    for index, duck in enumerate(controller.ducks):
        result.append(
            (
                int(controller.crossings_started[index]),
                bool(controller.crossing_armed[index]),
                bool(getattr(duck, "pedestrian_active", False)),
                float(getattr(duck, "time", 0.0)),
                tuple(float(value) for value in np.asarray(duck.pos).reshape(-1)),
            )
        )
    return tuple(result)


@dataclass(frozen=True)
class ReplayStep:
    step_index: int
    action: Any
    raw_state_vector: Tuple[float, ...]
    discrete_state: Tuple[int, ...]
    observation: Optional[Tuple[float, ...]]
    reward: float
    reward_terms: Mapping[str, float]
    events: Mapping[str, bool]
    controller_phase: Tuple[Any, ...]
    termination_reason: str
    terminated: bool
    truncated: bool


@dataclass(frozen=True)
class ReplayTrace:
    reset_seed: int
    steps: Tuple[ReplayStep, ...]

    @property
    def termination_reason(self) -> str:
        return self.steps[-1].termination_reason if self.steps else "in_progress"


def _raw_vector(raw: RawState) -> Tuple[float, ...]:
    return (
        float(raw.d),
        float(raw.phi),
        float(raw.v),
        float(int(raw.tile)),
        -1.0 if raw.d_stop is None else float(raw.d_stop),
        float(raw.sigma_stop),
        float(int(raw.duck)),
    )


def run_action_replay(
    env_factory: Callable[[int], Any],
    reset_seed: int,
    actions: Sequence[Any],
) -> ReplayTrace:
    env = env_factory(int(reset_seed))
    steps = []
    try:
        env.reset(int(reset_seed))
        for index, action in enumerate(actions):
            observation, reward, done, info = env.step(action)
            raw = _raw(info)
            encoded = None
            if isinstance(observation, np.ndarray):
                encoded = tuple(float(value) for value in observation.reshape(-1))
            steps.append(
                ReplayStep(
                    step_index=index,
                    action=(
                        tuple(float(value) for value in np.asarray(action).reshape(-1))
                        if isinstance(action, (list, tuple, np.ndarray))
                        else int(action)
                    ),
                    raw_state_vector=_raw_vector(raw),
                    discrete_state=discretize(raw),
                    observation=encoded,
                    reward=float(reward),
                    reward_terms={
                        str(name): float(value)
                        for name, value in info.get("reward_terms", {}).items()
                    },
                    events={
                        str(name): bool(value)
                        for name, value in info.get("events", {}).items()
                    },
                    controller_phase=_controller_phase(env),
                    termination_reason=str(info["termination_reason"]),
                    terminated=bool(info.get("terminated", False)),
                    truncated=bool(info.get("truncated", False)),
                )
            )
            if done:
                break
    finally:
        env.close()
    return ReplayTrace(reset_seed=int(reset_seed), steps=tuple(steps))


def _allclose(left: Sequence[float], right: Sequence[float], atol: float) -> bool:
    return bool(np.allclose(left, right, rtol=0.0, atol=atol, equal_nan=True))


def assert_replays_identical(
    left: ReplayTrace,
    right: ReplayTrace,
    atol: float = REPLAY_ATOL,
) -> None:
    if left.reset_seed != right.reset_seed:
        raise AssertionError("reset seed mismatch")
    if len(left.steps) != len(right.steps):
        raise AssertionError("termination step mismatch")
    for first, second in zip(left.steps, right.steps):
        prefix = "replay step %d" % first.step_index
        if first.action != second.action:
            raise AssertionError(prefix + ": action mismatch")
        if first.discrete_state != second.discrete_state:
            raise AssertionError(prefix + ": discrete state mismatch")
        if not _allclose(first.raw_state_vector, second.raw_state_vector, atol):
            raise AssertionError(prefix + ": raw state mismatch")
        if (first.observation is None) != (second.observation is None):
            raise AssertionError(prefix + ": observation availability mismatch")
        if first.observation is not None and not _allclose(
            first.observation, second.observation, atol
        ):
            raise AssertionError(prefix + ": continuous observation mismatch")
        if not _allclose((first.reward,), (second.reward,), atol):
            raise AssertionError(prefix + ": reward mismatch")
        if first.reward_terms.keys() != second.reward_terms.keys():
            raise AssertionError(prefix + ": reward-term keys mismatch")
        for name in first.reward_terms:
            if not _allclose(
                (first.reward_terms[name],), (second.reward_terms[name],), atol
            ):
                raise AssertionError(prefix + ": reward term %s mismatch" % name)
        if first.events != second.events:
            raise AssertionError(prefix + ": event mismatch")
        if first.controller_phase != second.controller_phase:
            raise AssertionError(prefix + ": Duckie controller phase mismatch")
        if (
            first.termination_reason != second.termination_reason
            or first.terminated != second.terminated
            or first.truncated != second.truncated
        ):
            raise AssertionError(prefix + ": termination mismatch")
