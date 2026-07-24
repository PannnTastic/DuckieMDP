"""CEDP8: open frozen M2 only after C-EDDP cluster freeze."""

import argparse
from collections import Counter
import json
from pathlib import Path

import joblib
import yaml

from src.explainability.action_outcomes import _environment_action
from src.explainability.certified_primitives.reconciliation import reconcile_after_freeze
from src.explainability.certified_primitives.reporting import atomic_json
from src.explainability.certified_primitives.schema import (
    CertifiedExplanationInstance,
    FullDecisionAnchor,
    TemporalExplanationSegment,
    read_jsonl,
)
from src.explainability.eddp.runtime import environment_factory, load_runtime
from src.explainability.primitives import PrimitiveLabeler
from src.explainability.schema import PolicyDecision, PolicyMode


def _m2_labels(config_path, output):
    config, _, shared, policies, _ = load_runtime(config_path)
    anchors = [
        FullDecisionAnchor.from_dict(row)
        for row in read_jsonl(output / "full_decision_anchors.jsonl")
    ]
    grouped = {}
    for anchor in anchors:
        grouped.setdefault((anchor.solver.value, anchor.seed, anchor.episode_id), []).append(anchor)
    labels = {}
    for (solver, seed, episode_id), records in sorted(grouped.items()):
        env = environment_factory(shared, solver)(seed)
        labeler = PrimitiveLabeler()
        try:
            env.reset(seed)
            for anchor in sorted(records, key=lambda item: item.step_index):
                decision = PolicyDecision(
                    solver=anchor.solver,
                    policy_mode=PolicyMode(anchor.policy_mode),
                    state=anchor.state,
                    action=anchor.selected_action,
                    diagnostics={},
                    metadata={"teacher_active": False},
                )
                _, _, done, info = env.step(_environment_action(anchor.selected_action))
                primitive = labeler.label(
                    decision, info.get("events"), info.get("termination_reason", "in_progress")
                )
                labels[(solver, seed, episode_id, anchor.step_index)] = primitive.primitive.value
                if done:
                    break
        finally:
            env.close()
    return labels


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path,
                        default=Path("configs/explainability/cedp_v2.yaml"))
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    output = args.output_dir or Path(config["experiment"]["output_dir"])
    freeze_path = output / "cluster_freeze_pre_m2.json"
    if not freeze_path.is_file():
        raise FileNotFoundError("cluster must be frozen before M2 reconciliation")
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    if freeze.get("m2_opened") is not False:
        raise ValueError("invalid pre-M2 freeze manifest")
    instances = tuple(
        CertifiedExplanationInstance.from_dict(row)
        for row in read_jsonl(output / "certified_explanation_instances.jsonl")
    )
    instance_by_id = {item.instance_id: item for item in instances}
    segments = tuple(
        TemporalExplanationSegment.from_dict(row)
        for row in read_jsonl(output / "temporal_explanation_segments.jsonl")
    )
    result = joblib.load(output / "models" / "discovery_result.joblib")
    label_by_step = _m2_labels(args.config, output)
    segment_labels = []
    for segment in segments:
        values = []
        for instance_id in segment.instance_ids:
            item = instance_by_id[instance_id]
            values.append(label_by_step[
                (item.solver, item.seed, item.episode_id, item.step_index)
            ])
        segment_labels.append(Counter(values).most_common(1)[0][0])
    reconciliation = reconcile_after_freeze(
        result.labels, segment_labels, segments, result.split,
        cluster_frozen=True,
    )
    payload = {
        "stage": "CEDP8",
        "cluster_freeze_hash": freeze["cluster_freeze_hash"],
        "m2_opened_after_freeze": True,
        "cluster_refit_after_m2": False,
        "reconciliation": reconciliation,
    }
    atomic_json(output / "m2_reconciliation_after_freeze.json", payload)
    print(json.dumps(reconciliation["overall"], sort_keys=True))


if __name__ == "__main__":
    main()
