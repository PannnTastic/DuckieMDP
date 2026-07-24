"""CEDP9: exercise the frozen assigner and explicit UNKNOWN behavior."""

import argparse
import json
from pathlib import Path

import joblib
import yaml

from src.explainability.certified_primitives.reporting import atomic_json
from src.explainability.certified_primitives.runtime import CertifiedPrimitiveAssigner
from src.explainability.certified_primitives.schema import (
    CertifiedPrimitive,
    TemporalExplanationSegment,
    read_jsonl,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path,
                        default=Path("configs/explainability/cedp_v2.yaml"))
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    output = args.output_dir or Path(config["experiment"]["output_dir"])
    discovery = joblib.load(output / "models" / "discovery_result.joblib")
    primitives = tuple(
        CertifiedPrimitive.from_dict(row)
        for row in read_jsonl(output / "primitive_certificates.jsonl")
    )
    segments = tuple(
        TemporalExplanationSegment.from_dict(row)
        for row in read_jsonl(output / "temporal_explanation_segments.jsonl")
    )
    assigner = CertifiedPrimitiveAssigner(
        discovery.model, primitives,
        radius_multiplier=float(config["discovery"]["radius_multiplier"]),
    )
    assignments = [assigner.assign(item) for item in segments]
    rows = [{
        "segment_id": segment.segment_id,
        "status": assignment.status.value,
        "primitive_id": assignment.primitive_id,
        "functional_name": assignment.functional_name,
        "support_distance": assignment.support_distance,
        "support_radius": assignment.support_radius,
        "explanation": dict(assignment.explanation),
    } for segment, assignment in zip(segments, assignments)]
    summary = {
        "stage": "CEDP9",
        "assignments": rows,
        "status_counts": {
            status: sum(item.status.value == status for item in assignments)
            for status in sorted({item.status.value for item in assignments})
        },
        "out_of_support_abstains": True,
    }
    atomic_json(output / "runtime_assignments.json", summary)
    print(json.dumps(summary["status_counts"], sort_keys=True))


if __name__ == "__main__":
    main()
