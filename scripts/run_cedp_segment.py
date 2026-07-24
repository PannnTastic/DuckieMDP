"""CEDP3--CEDP4: explanation-only signatures and temporal segmentation."""

import argparse
import json
from pathlib import Path

import joblib
import yaml

from src.explainability.certified_primitives.reporting import atomic_json
from src.explainability.certified_primitives.schema import (
    CertifiedExplanationInstance,
    read_jsonl,
    write_jsonl,
)
from src.explainability.certified_primitives.segmentation import (
    fit_change_point_model,
    segment_all,
)
from src.explainability.certified_primitives.signature import (
    build_signature_trajectory,
)
from src.explainability.certified_primitives.trajectory import build_trajectories


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path,
                        default=Path("configs/explainability/cedp_v2.yaml"))
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--method", choices=("change_point", "fixed_window"))
    parser.add_argument("--fixed-window", type=int, default=3)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    output = args.output_dir or Path(config["experiment"]["output_dir"])
    instances = tuple(
        CertifiedExplanationInstance.from_dict(row)
        for row in read_jsonl(output / "certified_explanation_instances.jsonl")
    )
    trajectories = build_trajectories(instances)
    signatures = tuple(build_signature_trajectory(item) for item in trajectories)
    segmentation = config["segmentation"]
    development_count = int(segmentation["development_seed_count"])
    dev_seeds = {}
    for solver in sorted({item.trajectory.solver for item in signatures}):
        values = sorted({item.trajectory.seed for item in signatures
                         if item.trajectory.solver == solver})
        dev_seeds[solver] = set(values[:development_count])
    development = tuple(
        item for item in signatures
        if item.trajectory.seed in dev_seeds[item.trajectory.solver]
    )
    # One model per solver: a pooled threshold is dominated by solvers with
    # discrete action jumps and silences change points for smooth policies.
    models = {}
    for solver in sorted(dev_seeds):
        models[solver] = fit_change_point_model(
            tuple(
                item for item in development
                if item.trajectory.solver == solver
            ),
            quantile=float(segmentation["distance_quantile"]),
            minimum_duration=int(segmentation["minimum_duration"]),
        )
    method = args.method or str(segmentation["primary"])
    segments = tuple(
        segment
        for solver in sorted(models)
        for segment in segment_all(
            (
                item for item in signatures
                if item.trajectory.solver == solver
            ),
            models[solver],
            method=method,
            fixed_window=args.fixed_window,
        )
    )
    write_jsonl(output / "temporal_explanation_segments.jsonl", segments)
    model_dir = output / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(models, model_dir / "segmentation_model.joblib")
    summary = {
        "stage": "CEDP3-CEDP4",
        "instances": len(instances),
        "trajectories": len(trajectories),
        "segments": len(segments),
        "method": method,
        "deterministic": True,
        "full_trajectory_only": all(
            item.source_kind.value == "full_trajectory" for item in segments
        ),
        "development_seeds": {
            key: sorted(value) for key, value in dev_seeds.items()
        },
        "per_solver_distance_threshold": {
            key: float(value.distance_threshold)
            for key, value in models.items()
        },
    }
    atomic_json(output / "segmentation_summary.json", summary)
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
