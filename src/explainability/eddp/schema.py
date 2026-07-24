"""Machine-readable contracts for explanation-derived primitive discovery."""

from dataclasses import asdict, dataclass, field
from hashlib import sha256
import json
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

from ..schema import CanonicalAction, CanonicalState, SolverKind, to_dict


EDDP_SCHEMA_VERSION = "1.0.0"


def stable_id(payload: Mapping[str, Any], prefix: str) -> str:
    encoded = json.dumps(
        dict(payload), sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return "%s-%s" % (prefix, sha256(encoded).hexdigest()[:20])


@dataclass(frozen=True)
class AnchorRecord:
    """One real policy decision and the exact prefix needed to reconstruct it."""

    anchor_id: str
    solver: SolverKind
    seed: int
    episode_id: str
    decision_step: int
    block_id: str
    block_offset: int
    selection_context: str
    observed_context: str
    state: CanonicalState
    selected_action: CanonicalAction
    action_prefix: Tuple[Any, ...]
    config_path: str
    checkpoint_path: str
    policy_mode: str
    schema_version: str = EDDP_SCHEMA_VERSION

    def as_dict(self) -> Dict[str, Any]:
        return to_dict(self)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "AnchorRecord":
        values = dict(payload)
        values["solver"] = SolverKind(values["solver"])
        values["state"] = CanonicalState(**values["state"])
        action = dict(values["selected_action"])
        action["solver"] = SolverKind(action["solver"])
        values["selected_action"] = CanonicalAction(**action)
        values["action_prefix"] = tuple(values["action_prefix"])
        return cls(**values)


@dataclass(frozen=True)
class ExplanationAtom:
    """Label-free, local explanation record used by the discovery stage."""

    atom_id: str
    anchor_id: str
    solver: SolverKind
    seed: int
    episode_id: str
    decision_step: int
    block_id: str
    block_offset: int
    selection_context: str
    observed_context: str
    counterfactual_profile: Mapping[str, Any]
    physical_profile: Mapping[str, Any]
    reward_profile: Mapping[str, Any]
    verification_profile: Mapping[str, Any]
    validity: Mapping[str, Any]
    paired_report_path: str
    schema_version: str = EDDP_SCHEMA_VERSION

    def as_dict(self) -> Dict[str, Any]:
        return to_dict(self)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ExplanationAtom":
        values = dict(payload)
        values["solver"] = SolverKind(values["solver"])
        return cls(**values)


def write_jsonl(path, records: Sequence[Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        for record in records:
            value = record.as_dict() if hasattr(record, "as_dict") else record
            stream.write(
                json.dumps(value, sort_keys=True, allow_nan=False) + "\n"
            )
    temporary.replace(path)


def read_jsonl(path) -> Tuple[Dict[str, Any], ...]:
    with path.open("r", encoding="utf-8") as stream:
        return tuple(json.loads(line) for line in stream if line.strip())
