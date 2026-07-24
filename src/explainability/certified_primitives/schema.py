"""Versioned contracts for certified explanation-derived primitives.

The central distinction in this module is intentional:

* an explanation *instance* is certified by the existing M1--M13 pipeline;
* a temporal cluster receives a separate *primitive* certificate.

Consequently, a collection of valid local explanations is never silently
promoted to a certified primitive.
"""

from dataclasses import asdict, dataclass, field, is_dataclass
from enum import Enum
from hashlib import sha256
import json
from math import isfinite
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

from ..schema import CanonicalAction, CanonicalState, SolverKind


CEDP_SCHEMA_VERSION = "2.0.0"


class CertificateStatus(str, Enum):
    CERTIFIED = "CERTIFIED"
    ABSTAINED = "ABSTAINED"
    CERTIFIED_PRIMITIVE = "CERTIFIED_PRIMITIVE"
    PRIMITIVE_CANDIDATE = "PRIMITIVE_CANDIDATE"
    SOLVER_SPECIFIC_PRIMITIVE = "SOLVER_SPECIFIC_PRIMITIVE"
    UNKNOWN = "UNKNOWN"


class SourceKind(str, Enum):
    FULL_TRAJECTORY = "full_trajectory"
    LEGACY_SPARSE = "legacy_sparse"


def stable_id(payload: Mapping[str, Any], prefix: str) -> str:
    encoded = json.dumps(
        _jsonable(dict(payload)),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return "%s-%s" % (prefix, sha256(encoded).hexdigest()[:20])


def file_sha256(path: Path) -> str:
    digest = sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _jsonable(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if hasattr(value, "tolist"):
        return _jsonable(value.tolist())
    if hasattr(value, "item"):
        return _jsonable(value.item())
    if isinstance(value, float) and not isfinite(value):
        raise ValueError("C-EDDP records cannot contain non-finite values")
    return value


def _require_nonempty(name: str, value: str) -> None:
    if not str(value):
        raise ValueError("%s must not be empty" % name)


@dataclass(frozen=True)
class FullDecisionAnchor:
    """A real policy decision plus deterministic replay prefix."""

    anchor_id: str
    solver: SolverKind
    seed: int
    episode_id: str
    step_index: int
    state: CanonicalState
    selected_action: CanonicalAction
    action_prefix: Tuple[Any, ...]
    config_path: str
    checkpoint_path: str
    policy_mode: str
    schema_version: str = CEDP_SCHEMA_VERSION

    def as_dict(self) -> Dict[str, Any]:
        return _jsonable(self)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]):
        values = dict(payload)
        values["solver"] = SolverKind(values["solver"])
        values["state"] = CanonicalState(**values["state"])
        action = dict(values["selected_action"])
        action["solver"] = SolverKind(action["solver"])
        values["selected_action"] = CanonicalAction(**action)
        values["action_prefix"] = tuple(values["action_prefix"])
        return cls(**values)


@dataclass(frozen=True)
class CertifiedExplanationInstance:
    """One local explanation and its M1--M13 certificate."""

    instance_id: str
    solver: str
    seed: int
    episode_id: str
    step_index: int
    source_kind: SourceKind
    status: CertificateStatus
    decision_evidence: Mapping[str, Any]
    outcome_evidence: Mapping[str, Any]
    verification_evidence: Mapping[str, Any]
    certificate: Mapping[str, Any]
    provenance: Mapping[str, Any]
    audit_metadata: Mapping[str, Any] = field(default_factory=dict)
    schema_version: str = CEDP_SCHEMA_VERSION

    def __post_init__(self) -> None:
        for name in ("instance_id", "solver", "episode_id"):
            _require_nonempty(name, getattr(self, name))
        if int(self.step_index) < 0:
            raise ValueError("step_index must be non-negative")
        if self.status == CertificateStatus.CERTIFIED and not all(
            bool(self.certificate.get(name, False))
            for name in (
                "counterfactual_valid",
                "branch_invariants_pass",
                "paired_outcome_valid",
                "deterministic_policy_mode",
                "teacher_inactive",
                "supported_or_reachable_state",
            )
        ):
            raise ValueError("CERTIFIED instance does not satisfy binding gates")

    @property
    def eligible_for_main_discovery(self) -> bool:
        return (
            self.status == CertificateStatus.CERTIFIED
            and self.source_kind == SourceKind.FULL_TRAJECTORY
        )

    def as_dict(self) -> Dict[str, Any]:
        return _jsonable(self)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]):
        values = dict(payload)
        values["source_kind"] = SourceKind(values["source_kind"])
        values["status"] = CertificateStatus(values["status"])
        return cls(**values)


@dataclass(frozen=True)
class ExplanationTrajectory:
    trajectory_id: str
    solver: str
    seed: int
    episode_id: str
    source_kind: SourceKind
    instances: Tuple[CertifiedExplanationInstance, ...]
    provenance: Mapping[str, Any]
    schema_version: str = CEDP_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not self.instances:
            raise ValueError("explanation trajectory cannot be empty")
        previous = None
        for instance in self.instances:
            if (
                instance.solver != self.solver
                or instance.seed != self.seed
                or instance.episode_id != self.episode_id
                or instance.source_kind != self.source_kind
            ):
                raise ValueError("trajectory mixes incompatible instances")
            if previous is not None:
                delta = instance.step_index - previous.step_index
                if self.source_kind == SourceKind.FULL_TRAJECTORY and delta != 1:
                    raise ValueError("full trajectory steps must be contiguous")
                if delta <= 0:
                    raise ValueError("trajectory step order must be strictly increasing")
            previous = instance

    def as_dict(self) -> Dict[str, Any]:
        return _jsonable(self)


@dataclass(frozen=True)
class TemporalExplanationSegment:
    segment_id: str
    trajectory_id: str
    solver: str
    seed: int
    episode_id: str
    source_kind: SourceKind
    start_step: int
    end_step: int
    instance_ids: Tuple[str, ...]
    feature_names: Tuple[str, ...]
    feature_values: Tuple[float, ...]
    certificate_coverage: float
    split: str = "unassigned"
    cluster_id: Optional[int] = None
    schema_version: str = CEDP_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.end_step < self.start_step:
            raise ValueError("segment end precedes start")
        if len(self.feature_names) != len(self.feature_values):
            raise ValueError("segment feature name/value mismatch")
        if not (0.0 <= float(self.certificate_coverage) <= 1.0):
            raise ValueError("certificate coverage must be in [0, 1]")
        if not all(isfinite(float(value)) for value in self.feature_values):
            raise ValueError("segment has non-finite features")

    @property
    def duration(self) -> int:
        return self.end_step - self.start_step + 1

    def as_dict(self) -> Dict[str, Any]:
        return _jsonable(self)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]):
        values = dict(payload)
        values["source_kind"] = SourceKind(values["source_kind"])
        values["instance_ids"] = tuple(values["instance_ids"])
        values["feature_names"] = tuple(values["feature_names"])
        values["feature_values"] = tuple(values["feature_values"])
        return cls(**values)


@dataclass(frozen=True)
class PrimitiveDescriptor:
    functional_name: str
    decision_summary: str
    outcome_summary: str
    verification_summary: str
    temporal_summary: str
    evidence_feature_names: Tuple[str, ...]
    member_certificate_ids: Tuple[str, ...]

    def as_dict(self) -> Dict[str, Any]:
        return _jsonable(self)


@dataclass(frozen=True)
class CertifiedPrimitive:
    primitive_id: str
    cluster_id: int
    status: CertificateStatus
    descriptor: PrimitiveDescriptor
    cluster_freeze_hash: str
    member_certificate_rate: float
    support: int
    seed_support: int
    solver_support: Tuple[str, ...]
    heldout_assignment_rate: float
    bootstrap_stability: float
    outcome_coherence_ratio: float
    verified_properties: Mapping[str, Any]
    boundary_cases: Tuple[str, ...]
    representative_certificates: Tuple[str, ...]
    gate_results: Mapping[str, bool]
    schema_version: str = CEDP_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.status == CertificateStatus.CERTIFIED:
            raise ValueError("use a primitive-level status")
        if self.support < 0 or self.seed_support < 0:
            raise ValueError("support counts must be non-negative")
        for name in (
            "member_certificate_rate",
            "heldout_assignment_rate",
            "bootstrap_stability",
        ):
            value = float(getattr(self, name))
            if not (0.0 <= value <= 1.0):
                raise ValueError("%s must be in [0, 1]" % name)

    def as_dict(self) -> Dict[str, Any]:
        return _jsonable(self)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]):
        values = dict(payload)
        values["status"] = CertificateStatus(values["status"])
        values["descriptor"] = PrimitiveDescriptor(**values["descriptor"])
        for name in (
            "solver_support",
            "boundary_cases",
            "representative_certificates",
        ):
            values[name] = tuple(values[name])
        return cls(**values)


def write_jsonl(path: Path, records: Sequence[Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        for record in records:
            value = record.as_dict() if hasattr(record, "as_dict") else _jsonable(record)
            stream.write(json.dumps(value, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(destination)


def read_jsonl(path: Path) -> Tuple[Dict[str, Any], ...]:
    with Path(path).open("r", encoding="utf-8") as stream:
        return tuple(json.loads(line) for line in stream if line.strip())
