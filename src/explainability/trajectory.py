"""Solver-neutral trajectory recording and primitive segmentation."""

from dataclasses import asdict, dataclass, field, is_dataclass
from hashlib import sha256
import json
from math import isfinite
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from .primitives import PRIMITIVE_SCHEMA_VERSION, DrivingPrimitive, PrimitiveLabel
from .schema import PolicyDecision, PolicyMode, SolverKind, to_dict


TRAJECTORY_SCHEMA_VERSION = "1.0.0"


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_provenance(
    checkpoint_path: Optional[Path] = None,
    config_path: Optional[Path] = None,
    primitive_freeze_path: Optional[Path] = Path(
        "docs/primitive_lexicon_v1.freeze.json"
    ),
    extra: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Build immutable file-hash provenance without requiring Git state."""
    result: Dict[str, Any] = {
        "trajectory_schema_version": TRAJECTORY_SCHEMA_VERSION,
        "primitive_schema_version": PRIMITIVE_SCHEMA_VERSION,
    }
    for prefix, candidate in (
        ("checkpoint", checkpoint_path),
        ("config", config_path),
        ("primitive_freeze", primitive_freeze_path),
    ):
        if candidate is None:
            continue
        path = Path(candidate)
        if not path.is_file():
            raise FileNotFoundError(path)
        result[prefix + "_path"] = str(path)
        result[prefix + "_sha256"] = _file_sha256(path)
    if extra:
        result.update(dict(extra))
    return result


def _plain_mapping(value: Optional[Any]) -> Dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, Mapping):
        return dict(value)
    if is_dataclass(value):
        return asdict(value)
    raise TypeError("expected mapping or dataclass, got %s" % type(value).__name__)


@dataclass(frozen=True)
class TrajectoryStep:
    episode_id: str
    step_index: int
    physics_step: Optional[int]
    sim_time_seconds: Optional[float]
    decision: PolicyDecision
    primitive: PrimitiveLabel
    reward: float
    reward_terms: Mapping[str, float]
    events: Mapping[str, bool]
    termination_reason: str
    terminated: bool
    truncated: bool
    position_xz: Optional[Tuple[float, float]] = None
    heading_radians: Optional[float] = None
    extra: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.step_index < 0:
            raise ValueError("step_index must be non-negative")
        if self.physics_step is not None and self.physics_step < 0:
            raise ValueError("physics_step must be non-negative")
        for name, value in (
            ("sim_time_seconds", self.sim_time_seconds),
            ("reward", self.reward),
            ("heading_radians", self.heading_radians),
        ):
            if value is not None and not isfinite(float(value)):
                raise ValueError("%s must be finite" % name)
        if self.position_xz is not None:
            if len(self.position_xz) != 2 or not all(
                isfinite(float(value)) for value in self.position_xz
            ):
                raise ValueError("position_xz must contain two finite values")
        if not self.termination_reason:
            raise ValueError("termination_reason must not be empty")
        if self.primitive.schema_version != PRIMITIVE_SCHEMA_VERSION:
            raise ValueError("primitive schema mismatch")

    def as_dict(self) -> Dict[str, Any]:
        return to_dict(self)


@dataclass(frozen=True)
class PrimitiveSegment:
    episode_id: str
    segment_index: int
    primitive: DrivingPrimitive
    start_step: int
    end_step: int
    duration_steps: int
    start_time_seconds: Optional[float]
    end_time_seconds: Optional[float]
    cumulative_reward: float
    triggers: Tuple[str, ...]
    rule_ids: Tuple[str, ...]
    event_counts: Mapping[str, int]
    undesirable: bool

    def as_dict(self) -> Dict[str, Any]:
        return to_dict(self)


@dataclass(frozen=True)
class TrajectoryRecord:
    episode_id: str
    solver: SolverKind
    policy_mode: PolicyMode
    steps: Tuple[TrajectoryStep, ...]
    segments: Tuple[PrimitiveSegment, ...]
    total_reward: float
    termination_reason: str
    provenance: Mapping[str, Any]
    trajectory_schema_version: str = TRAJECTORY_SCHEMA_VERSION
    primitive_schema_version: str = PRIMITIVE_SCHEMA_VERSION

    def as_dict(self) -> Dict[str, Any]:
        return to_dict(self)

    def to_json(self, indent: Optional[int] = 2) -> str:
        return json.dumps(
            self.as_dict(),
            indent=indent,
            sort_keys=True,
            allow_nan=False,
        )

    def save_json(self, path: Path, indent: Optional[int] = 2) -> None:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_suffix(output.suffix + ".tmp")
        temporary.write_text(self.to_json(indent=indent) + "\n", encoding="utf-8")
        temporary.replace(output)

    def save_steps_jsonl(self, path: Path) -> None:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_suffix(output.suffix + ".tmp")
        with temporary.open("w", encoding="utf-8") as stream:
            for step in self.steps:
                stream.write(
                    json.dumps(
                        step.as_dict(),
                        sort_keys=True,
                        allow_nan=False,
                    )
                    + "\n"
                )
        temporary.replace(output)


class TrajectoryRecorder:
    """Append validated decisions, then segment consecutive primitives."""

    def __init__(
        self,
        episode_id: str,
        provenance: Mapping[str, Any],
        decision_dt_seconds: Optional[float] = None,
    ) -> None:
        if not episode_id:
            raise ValueError("episode_id must not be empty")
        if decision_dt_seconds is not None and decision_dt_seconds <= 0.0:
            raise ValueError("decision_dt_seconds must be positive")
        self.episode_id = str(episode_id)
        self.provenance = dict(provenance)
        self.decision_dt_seconds = decision_dt_seconds
        self._steps: List[TrajectoryStep] = []
        self._solver: Optional[SolverKind] = None
        self._policy_mode: Optional[PolicyMode] = None

    @property
    def steps(self) -> Tuple[TrajectoryStep, ...]:
        return tuple(self._steps)

    def append(
        self,
        decision: PolicyDecision,
        primitive: PrimitiveLabel,
        reward: float,
        info: Optional[Mapping[str, Any]] = None,
        physics_step: Optional[int] = None,
        position_xz: Optional[Sequence[float]] = None,
        heading_radians: Optional[float] = None,
        extra: Optional[Mapping[str, Any]] = None,
    ) -> TrajectoryStep:
        if self._solver is None:
            self._solver = decision.solver
            self._policy_mode = decision.policy_mode
        elif (
            decision.solver != self._solver
            or decision.policy_mode != self._policy_mode
        ):
            raise ValueError("one trajectory cannot mix solver or policy mode")

        metadata = dict(info or {})
        events = _plain_mapping(metadata.get("events"))
        reward_terms = _plain_mapping(metadata.get("reward_terms"))
        step_index = len(self._steps)
        sim_time = (
            None
            if self.decision_dt_seconds is None
            else step_index * self.decision_dt_seconds
        )
        position = (
            None
            if position_xz is None
            else (float(position_xz[0]), float(position_xz[1]))
        )
        step = TrajectoryStep(
            episode_id=self.episode_id,
            step_index=step_index,
            physics_step=(
                int(physics_step)
                if physics_step is not None
                else metadata.get("physics_step")
            ),
            sim_time_seconds=sim_time,
            decision=decision,
            primitive=primitive,
            reward=float(reward),
            reward_terms={
                str(name): float(value) for name, value in reward_terms.items()
            },
            events={str(name): bool(value) for name, value in events.items()},
            termination_reason=str(
                metadata.get("termination_reason", "in_progress")
            ),
            terminated=bool(metadata.get("terminated", False)),
            truncated=bool(metadata.get("truncated", False)),
            position_xz=position,
            heading_radians=(
                None if heading_radians is None else float(heading_radians)
            ),
            extra=dict(extra or {}),
        )
        self._steps.append(step)
        return step

    def finalize(self) -> TrajectoryRecord:
        if not self._steps or self._solver is None or self._policy_mode is None:
            raise ValueError("cannot finalize an empty trajectory")
        return TrajectoryRecord(
            episode_id=self.episode_id,
            solver=self._solver,
            policy_mode=self._policy_mode,
            steps=tuple(self._steps),
            segments=tuple(segment_primitives(self._steps)),
            total_reward=float(sum(step.reward for step in self._steps)),
            termination_reason=self._steps[-1].termination_reason,
            provenance=dict(self.provenance),
        )


def _unique(values: Sequence[str]) -> Tuple[str, ...]:
    return tuple(dict.fromkeys(values))


def _segment(
    steps: Sequence[TrajectoryStep],
    segment_index: int,
) -> PrimitiveSegment:
    first, last = steps[0], steps[-1]
    event_names = sorted({name for step in steps for name in step.events})
    event_counts = {
        name: sum(int(step.events.get(name, False)) for step in steps)
        for name in event_names
    }
    return PrimitiveSegment(
        episode_id=first.episode_id,
        segment_index=segment_index,
        primitive=first.primitive.primitive,
        start_step=first.step_index,
        end_step=last.step_index,
        duration_steps=last.step_index - first.step_index + 1,
        start_time_seconds=first.sim_time_seconds,
        end_time_seconds=last.sim_time_seconds,
        cumulative_reward=float(sum(step.reward for step in steps)),
        triggers=_unique([step.primitive.trigger for step in steps]),
        rule_ids=_unique([step.primitive.rule_id for step in steps]),
        event_counts=event_counts,
        undesirable=first.primitive.undesirable,
    )


def segment_primitives(
    steps: Sequence[TrajectoryStep],
) -> List[PrimitiveSegment]:
    if not steps:
        return []
    segments: List[PrimitiveSegment] = []
    current: List[TrajectoryStep] = [steps[0]]
    for previous, step in zip(steps, steps[1:]):
        if step.episode_id != previous.episode_id:
            raise ValueError("segment_primitives cannot mix episodes")
        if step.step_index != previous.step_index + 1:
            raise ValueError("trajectory step indices must be contiguous")
        if step.primitive.primitive == current[-1].primitive.primitive:
            current.append(step)
        else:
            segments.append(_segment(current, len(segments)))
            current = [step]
    segments.append(_segment(current, len(segments)))
    return segments
