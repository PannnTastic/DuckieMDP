"""CEDP5: development discovery, freeze, and inductive held-out assignment."""

import argparse
from hashlib import sha256
import json
from pathlib import Path

import joblib
import yaml

from src.explainability.certified_primitives.discovery import discover
from src.explainability.certified_primitives.reporting import (
    atomic_json,
    write_assignments_csv,
)
from src.explainability.certified_primitives.schema import (
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
    segments = tuple(
        TemporalExplanationSegment.from_dict(row)
        for row in read_jsonl(output / "temporal_explanation_segments.jsonl")
    )
    result = discover(
        segments,
        development_seed_count=int(config["discovery"]["development_seed_count"]),
        pca_components=int(config["discovery"].get("pca_components", 15)),
    )
    freeze_payload = {
        "selected_parameters": dict(result.model.selected_parameters),
        "development_seeds": {
            key: list(value) for key, value in result.model.development_seeds.items()
        },
        "segment_ids": [item.segment_id for item in segments],
        "labels": result.labels.tolist(),
        "m2_opened": False,
    }
    encoded = json.dumps(freeze_payload, sort_keys=True, separators=(",", ":"))
    freeze_payload["cluster_freeze_hash"] = sha256(encoded.encode("utf-8")).hexdigest()
    model_dir = output / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(result, model_dir / "discovery_result.joblib")
    atomic_json(output / "cluster_freeze_pre_m2.json", freeze_payload)
    write_assignments_csv(
        output / "cluster_assignments.csv",
        segments, result.labels, result.split,
    )
    summary = {
        "stage": "CEDP5",
        "cluster_freeze_hash": freeze_payload["cluster_freeze_hash"],
        "diagnostics": dict(result.diagnostics),
        "search": list(result.search),
        "m2_used_during_fit": False,
    }
    atomic_json(output / "discovery_summary.json", summary)
    print(json.dumps(summary["diagnostics"], sort_keys=True))


if __name__ == "__main__":
    main()
