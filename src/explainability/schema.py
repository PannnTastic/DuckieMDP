"""Versioned, solver-neutral records used by the explanation pipeline."""

from dataclasses import asdict, dataclass, field, is_dataclass
from enum import Enum
import json
from math import isfinite
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple


SCHEMA_VERSION = "1.0.0"


class SolverKind(str, Enum):
    Q_LEARNING = "q_learning"
    SARSA = "sarsa"
    SAC = "sac"


TABULAR_SOLVERS = frozenset({SolverKind.Q_LEARNING, SolverKind.SARSA})


class PolicyMode(str, Enum):
    GREEDY = "greedy"
    DETERMINISTIC_ACTOR_MEAN = "deterministic_actor_mean"


def _optional_finite(name: str, value: Optional[float]) -> None:
    if value is not None and not isfinite(float(value)):
        raise ValueError("%s must be finite or None" % name)


@dataclass(frozen=True)
class CanonicalState:
    """Semantic state shared by tabular and continuous policies.

    Fields unavailable in a solver representation stay ``None``. In
    particular, a tabular DuckThreat does not pretend to contain metric Duckie
    geometry, and its curvature category does not pretend to be continuous
    curvature.
    """

    d: float
    phi: float
    v: float
    curvature: Optional[float]
    curvature_class: str
    stop_present: bool
    stop_distance: Optional[float]
    stop_satisfied: bool
    stop_hold_progress: Optional[float]
    duck_present: bool
    duck_threat: Optional[str]
    duck_longitudinal: Optional[float]
    duck_lateral: Optional[float]
    duck_v_longitudinal_relative: Optional[float]
    duck_v_lateral_relative: Optional[float]
    duck_active: Optional[bool]
    duck_crossing_available: Optional[bool]
    source_representation: str
    source_index: Optional[Tuple[int, ...]] = None

    def __post_init__(self) -> None:
        for name in (
            "d",
            "phi",
            "v",
            "curvature",
            "stop_distance",
            "stop_hold_progress",
            "duck_longitudinal",
            "duck_lateral",
            "duck_v_longitudinal_relative",
            "duck_v_lateral_relative",
        ):
            _optional_finite(name, getattr(self, name))
        if self.v < 0.0:
            raise ValueError("v must be non-negative")
        if not self.curvature_class:
            raise ValueError("curvature_class must not be empty")
        if not self.source_representation:
            raise ValueError("source_representation must not be empty")
        # Semantic consistency is classified by M6, not rejected by the schema.
        if self.stop_distance is not None and self.stop_distance < 0.0:
            raise ValueError("stop_distance must be non-negative")
        if self.stop_hold_progress is not None and not (
            0.0 <= self.stop_hold_progress <= 1.0
        ):
            raise ValueError("stop_hold_progress must be in [0, 1]")


@dataclass(frozen=True)
class CanonicalAction:
    solver: SolverKind
    v_cmd: float
    omega_cmd: float
    action_id: Optional[int] = None
    action_name: Optional[str] = None

    def __post_init__(self) -> None:
        _optional_finite("v_cmd", self.v_cmd)
        _optional_finite("omega_cmd", self.omega_cmd)
        if self.solver in TABULAR_SOLVERS:
            if self.action_id is None or self.action_name is None:
                raise ValueError("tabular action requires id and name")
            if not 0 <= int(self.action_id) <= 6:
                raise ValueError("tabular action_id must be in [0, 6]")


@dataclass(frozen=True)
class PolicyDecision:
    solver: SolverKind
    policy_mode: PolicyMode
    state: CanonicalState
    action: CanonicalAction
    diagnostics: Mapping[str, Any] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.action.solver != self.solver:
            raise ValueError("decision and action solver mismatch")


@dataclass(frozen=True)
class ExplanationRecord:
    """Extensible top-level record; later milestones fill optional sections."""

    decision: PolicyDecision
    selected_primitive: Optional[str] = None
    trigger: Optional[str] = None
    foil: Optional[Mapping[str, Any]] = None
    state_counterfactual: Optional[Mapping[str, Any]] = None
    action_outcome_counterfactual: Optional[Mapping[str, Any]] = None
    metamorphic_results: Sequence[Mapping[str, Any]] = field(default_factory=tuple)
    validity: Mapping[str, Any] = field(default_factory=dict)
    provenance: Mapping[str, Any] = field(default_factory=dict)
    schema_version: str = SCHEMA_VERSION


def _jsonable(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value):
        return {key: _jsonable(item) for key, item in asdict(value).items()}
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if hasattr(value, "tolist"):
        return _jsonable(value.tolist())
    if hasattr(value, "item"):
        return _jsonable(value.item())
    if isinstance(value, float) and not isfinite(value):
        raise ValueError("schema cannot serialize non-finite float")
    return value


def to_dict(record: Any) -> Dict[str, Any]:
    value = _jsonable(record)
    if not isinstance(value, dict):
        raise TypeError("top-level schema value must serialize to a dictionary")
    return value


def to_json(record: Any, indent: Optional[int] = 2) -> str:
    return json.dumps(
        to_dict(record),
        indent=indent,
        sort_keys=True,
        allow_nan=False,
    )
