"""Build contiguous explanation trajectories without using primitive labels."""

from collections import defaultdict
from typing import Dict, Iterable, List, Sequence, Tuple

from .schema import (
    CertifiedExplanationInstance,
    ExplanationTrajectory,
    SourceKind,
    stable_id,
)


def build_trajectories(
    instances: Iterable[CertifiedExplanationInstance],
) -> Tuple[ExplanationTrajectory, ...]:
    """Group records and split at every gap or abstention.

    A full trajectory is never bridged across an invalid explanation. Legacy
    sparse records are grouped by their original three-step block when that
    audit field exists; this path is a baseline and remains ineligible for the
    main result.
    """

    groups: Dict[Tuple[str, int, str, str], List[CertifiedExplanationInstance]] = defaultdict(list)
    for record in instances:
        legacy_block = str(record.audit_metadata.get("legacy_block_id", ""))
        key = (
            record.solver,
            int(record.seed),
            record.episode_id,
            legacy_block if record.source_kind == SourceKind.LEGACY_SPARSE else "",
        )
        groups[key].append(record)

    trajectories = []
    for key, values in sorted(groups.items()):
        ordered = sorted(values, key=lambda item: item.step_index)
        current: List[CertifiedExplanationInstance] = []
        chunks: List[Sequence[CertifiedExplanationInstance]] = []
        for record in ordered:
            contiguous = (
                not current or record.step_index == current[-1].step_index + 1
            )
            if record.status.value != "CERTIFIED" or not contiguous:
                if current:
                    chunks.append(tuple(current))
                    current = []
                if record.status.value != "CERTIFIED":
                    continue
            current.append(record)
        if current:
            chunks.append(tuple(current))

        for chunk_index, chunk in enumerate(chunks):
            first = chunk[0]
            identity = {
                "solver": first.solver,
                "seed": first.seed,
                "episode_id": first.episode_id,
                "source_kind": first.source_kind.value,
                "first": first.step_index,
                "last": chunk[-1].step_index,
                "chunk": chunk_index,
            }
            trajectories.append(ExplanationTrajectory(
                trajectory_id=stable_id(identity, "cedp-trajectory"),
                solver=first.solver,
                seed=first.seed,
                episode_id=first.episode_id,
                source_kind=first.source_kind,
                instances=tuple(chunk),
                provenance={
                    "first_instance_id": first.instance_id,
                    "last_instance_id": chunk[-1].instance_id,
                    "abstention_breaks_trajectory": True,
                },
            ))
    return tuple(trajectories)


def validate_full_coverage(
    instances: Sequence[CertifiedExplanationInstance],
    expected_steps: int,
) -> dict:
    full = [item for item in instances if item.source_kind == SourceKind.FULL_TRAJECTORY]
    step_ids = [item.step_index for item in full]
    return {
        "expected_steps": int(expected_steps),
        "observed_steps": len(full),
        "unique_steps": len(set(step_ids)),
        "all_steps_recorded": len(full) == int(expected_steps),
        "step_ids_unique": len(step_ids) == len(set(step_ids)),
        "certified_steps": sum(item.status.value == "CERTIFIED" for item in full),
        "abstained_steps": sum(item.status.value == "ABSTAINED" for item in full),
    }
