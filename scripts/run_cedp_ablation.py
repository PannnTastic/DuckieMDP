"""CEDP10: leakage-safe segmentation and feature-block ablations."""

import argparse
import json
from pathlib import Path

import yaml

from src.explainability.certified_primitives.discovery import discover
from src.explainability.certified_primitives.reporting import atomic_json
from src.explainability.certified_primitives.schema import (
    CertifiedExplanationInstance,
    TemporalExplanationSegment,
    read_jsonl,
)
from src.explainability.certified_primitives.segmentation import (
    fit_change_point_model,
    segment_all,
)
from src.explainability.certified_primitives.signature import build_signature_trajectory
from src.explainability.certified_primitives.trajectory import build_trajectories


def _subset(segment, predicate):
    pairs = [(name, value) for name, value in zip(
        segment.feature_names, segment.feature_values
    ) if predicate(name)]
    return TemporalExplanationSegment(
        segment_id=segment.segment_id, trajectory_id=segment.trajectory_id,
        solver=segment.solver, seed=segment.seed, episode_id=segment.episode_id,
        source_kind=segment.source_kind, start_step=segment.start_step,
        end_step=segment.end_step, instance_ids=segment.instance_ids,
        feature_names=tuple(name for name, _ in pairs),
        feature_values=tuple(value for _, value in pairs),
        certificate_coverage=segment.certificate_coverage,
    )


def _diagnose(name, segments, development_count):
    try:
        result = discover(
            segments, development_seed_count=development_count
        )
        return {"name": name, "segments": len(segments),
                "diagnostics": dict(result.diagnostics), "error": None}
    except Exception as error:
        return {"name": name, "segments": len(segments),
                "diagnostics": None,
                "error": "%s: %s" % (type(error).__name__, error)}


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
    trajectories = build_trajectories(instances)
    signatures = tuple(build_signature_trajectory(item) for item in trajectories)
    segmentation_cfg = config["segmentation"]
    model = fit_change_point_model(
        signatures,
        quantile=float(segmentation_cfg["distance_quantile"]),
        minimum_duration=int(segmentation_cfg["minimum_duration"]),
    )
    variants = {}
    variants["change_point"] = segment_all(signatures, model, method="change_point")
    variants["fixed_window_3"] = segment_all(
        signatures, model, method="fixed_window", fixed_window=3
    )
    variants["fixed_window_5"] = segment_all(
        signatures, model, method="fixed_window", fixed_window=5
    )
    primary = tuple(
        TemporalExplanationSegment.from_dict(row)
        for row in read_jsonl(output / "temporal_explanation_segments.jsonl")
    )
    feature_variants = {
        "full": primary,
        "decision_only": tuple(_subset(item, lambda name:
            "__decision__" in name or name in {"duration", "certificate_coverage"})
            for item in primary),
        "without_verification": tuple(_subset(item, lambda name:
            "__verification__" not in name) for item in primary),
        "without_temporal": tuple(_subset(item, lambda name:
            "__temporal__" not in name) for item in primary),
        "outcome_only": tuple(_subset(item, lambda name:
            "__outcome__" in name or name in {"duration", "certificate_coverage"})
            for item in primary),
    }
    development_count = int(config["discovery"]["development_seed_count"])
    results = []
    for name, segments in variants.items():
        results.append(_diagnose("segmentation_%s" % name, segments, development_count))
    for name, segments in feature_variants.items():
        results.append(_diagnose("features_%s" % name, segments, development_count))
    payload = {
        "stage": "CEDP10",
        "ablations": results,
        "negative_results_retained": True,
        "eddp_v1_is_external_sparse_baseline": True,
    }
    atomic_json(output / "ablation_report.json", payload)
    print(json.dumps({row["name"]: row["error"] for row in results}, sort_keys=True))


if __name__ == "__main__":
    main()
