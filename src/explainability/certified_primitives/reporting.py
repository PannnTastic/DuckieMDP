"""Machine-readable and human-readable C-EDDP reports."""

import csv
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from .schema import CertifiedPrimitive, TemporalExplanationSegment


def atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(destination)


def write_assignments_csv(
    path: Path,
    segments: Sequence[TemporalExplanationSegment],
    labels: Sequence[int],
    split: Sequence[str],
) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=(
            "segment_id", "trajectory_id", "solver", "seed", "episode_id",
            "start_step", "end_step", "duration", "source_kind", "split",
            "cluster_id", "certificate_coverage",
        ))
        writer.writeheader()
        for segment, label, split_value in zip(segments, labels, split):
            writer.writerow({
                "segment_id": segment.segment_id,
                "trajectory_id": segment.trajectory_id,
                "solver": segment.solver,
                "seed": segment.seed,
                "episode_id": segment.episode_id,
                "start_step": segment.start_step,
                "end_step": segment.end_step,
                "duration": segment.duration,
                "source_kind": segment.source_kind.value,
                "split": split_value,
                "cluster_id": int(label),
                "certificate_coverage": segment.certificate_coverage,
            })
    temporary.replace(destination)


def build_summary(
    primitives: Sequence[CertifiedPrimitive],
    segments: Sequence[TemporalExplanationSegment],
    diagnostics: Mapping[str, Any],
) -> Mapping[str, Any]:
    statuses = {}
    for primitive in primitives:
        statuses[primitive.status.value] = statuses.get(primitive.status.value, 0) + 1
    assigned = int(sum(primitive.support for primitive in primitives))
    return {
        "method": "Certified Explanation-Derived Driving Primitives (C-EDDP)",
        "segments": len(segments),
        "assigned_segments": assigned,
        "primitive_status_counts": statuses,
        "certified_primitives": [
            primitive.as_dict() for primitive in primitives
            if primitive.status.value in {
                "CERTIFIED_PRIMITIVE", "SOLVER_SPECIFIC_PRIMITIVE"
            }
        ],
        "candidate_primitives": [
            primitive.as_dict() for primitive in primitives
            if primitive.status.value == "PRIMITIVE_CANDIDATE"
        ],
        "diagnostics": dict(diagnostics),
        "claim_boundary": (
            "Only full_trajectory M1--M13 instances can support main C-EDDP claims; "
            "legacy_sparse records remain a baseline."
        ),
    }


def write_markdown_catalogue(path: Path, primitives: Sequence[CertifiedPrimitive]) -> None:
    lines = [
        "# C-EDDP Primitive Catalogue",
        "",
        "This catalogue reports primitive-level status separately from local explanation validity.",
        "",
    ]
    for primitive in primitives:
        descriptor = primitive.descriptor
        lines.extend([
            "## %s" % descriptor.functional_name,
            "",
            "- Status: `%s`" % primitive.status.value,
            "- Support: %d segments, %d solver-seed pairs" % (
                primitive.support, primitive.seed_support
            ),
            "- Solvers: %s" % ", ".join(primitive.solver_support),
            "- Why: %s" % descriptor.decision_summary,
            "- What-if: %s" % descriptor.outcome_summary,
            "- Verification: %s" % descriptor.verification_summary,
            "- Temporal role: %s" % descriptor.temporal_summary,
            "",
        ])
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text("\n".join(lines) + "\n", encoding="utf-8")
    temporary.replace(destination)
