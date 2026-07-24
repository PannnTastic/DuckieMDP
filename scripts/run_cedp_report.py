"""CEDP10: unified machine-readable and human-readable report."""

import argparse
from pathlib import Path

import joblib
import yaml

from src.explainability.certified_primitives.reporting import atomic_json, build_summary
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
    segments = tuple(
        TemporalExplanationSegment.from_dict(row)
        for row in read_jsonl(output / "temporal_explanation_segments.jsonl")
    )
    primitives = tuple(
        CertifiedPrimitive.from_dict(row)
        for row in read_jsonl(output / "primitive_certificates.jsonl")
    )
    discovery = joblib.load(output / "models" / "discovery_result.joblib")
    summary = build_summary(primitives, segments, discovery.diagnostics)
    atomic_json(output / "cedp_v2_report.json", summary)
    print("C-EDDP report: segments=%d primitives=%d" % (
        len(segments), len(primitives)
    ))


if __name__ == "__main__":
    main()
