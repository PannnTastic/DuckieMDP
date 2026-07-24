"""CEDP6--CEDP7: descriptors and primitive-level certificates."""

import argparse
import json
from pathlib import Path

import joblib
import yaml

from src.explainability.certified_primitives.certificate_checker import certify_clusters
from src.explainability.certified_primitives.reporting import (
    atomic_json,
    write_markdown_catalogue,
)
from src.explainability.certified_primitives.schema import (
    CertifiedExplanationInstance,
    TemporalExplanationSegment,
    read_jsonl,
    write_jsonl,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path,
                        default=Path("configs/explainability/cedp_v2.yaml"))
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    output = args.output_dir or Path(config["experiment"]["output_dir"])
    instances = tuple(
        CertifiedExplanationInstance.from_dict(row)
        for row in read_jsonl(output / "certified_explanation_instances.jsonl")
    )
    segments = tuple(
        TemporalExplanationSegment.from_dict(row)
        for row in read_jsonl(output / "temporal_explanation_segments.jsonl")
    )
    result = joblib.load(output / "models" / "discovery_result.joblib")
    freeze = json.loads(
        (output / "cluster_freeze_pre_m2.json").read_text(encoding="utf-8")
    )
    gates = config["certification"]
    primitives = certify_clusters(
        segments, result, instances, freeze["cluster_freeze_hash"],
        minimum_support=int(gates["minimum_support"]),
        minimum_seeds=int(gates["minimum_seeds"]),
        minimum_bootstrap_ari=float(gates["minimum_bootstrap_ari"]),
        maximum_coherence_ratio=float(gates["maximum_outcome_coherence_ratio"]),
        minimum_property_evidence=int(
            gates.get("minimum_property_evidence", 1)
        ),
    )
    write_jsonl(output / "primitive_certificates.jsonl", primitives)
    write_markdown_catalogue(output / "primitive_catalogue.md", primitives)
    summary = {
        "stage": "CEDP6-CEDP7",
        "primitives": len(primitives),
        "status_counts": {
            status: sum(item.status.value == status for item in primitives)
            for status in sorted({item.status.value for item in primitives})
        },
        "status_computed_by_checker": True,
        "descriptor_uses_context_or_m2": False,
    }
    atomic_json(output / "certification_summary.json", summary)
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
