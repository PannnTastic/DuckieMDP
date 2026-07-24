"""Recompute support-aware state profiles without new simulator rollouts.

The expensive factual/foil reports are reused unchanged. Counterfactual and
verification profiles are recomputed at stored full-trajectory anchors using
one frozen support oracle built from the complete anchor set.
"""

import argparse
from collections import Counter
import json
from pathlib import Path
import shutil

import yaml

from src.explainability.certified_primitives.certificate_adapter import (
    adapt_m1_m13_record,
)
from src.explainability.certified_primitives.reporting import atomic_json
from src.explainability.certified_primitives.schema import (
    FullDecisionAnchor,
    read_jsonl,
    write_jsonl,
)
from src.explainability.eddp.counterfactual_profile import counterfactual_profile
from src.explainability.eddp.runtime import load_runtime
from src.explainability.eddp.support import SupportOracle
from src.explainability.eddp.verification import verification_profile


def _reprofiled_payload(
    anchor,
    old_shard,
    policy,
    support_oracle,
):
    state_counterfactual = counterfactual_profile(policy, anchor.state)
    source_support = support_oracle.classify(policy, anchor.state)
    if not source_support.reachable:
        raise RuntimeError("stored anchor is missing from support oracle")
    verification = verification_profile(
        policy, anchor.state, support_oracle=support_oracle
    )
    attempts = int(state_counterfactual.get("attempts", 0))
    valid_attempts = int(state_counterfactual.get("valid_attempts", 0))
    old_certificate = dict(old_shard.get("certificate", {}))
    outcome = dict(old_shard.get("outcome_evidence", {}))
    old_verification = dict(old_shard.get("verification_evidence", {}))
    provenance = dict(old_shard.get("provenance", {}))
    provenance["support_basis"] = source_support.basis
    return {
        "schema_version": "m1-m13-combined-v2-support-aware",
        "counterfactual_profile": state_counterfactual,
        "physical_profile": outcome.get("physical_profile", {}),
        "reward_profile": outcome.get("reward_profile", {}),
        "action_outcome_counterfactual": outcome.get(
            "action_outcome_counterfactual", {}
        ),
        "verification_profile": verification,
        "metamorphic_results": verification,
        "safety_properties": old_verification.get("safety_properties", {}),
        "validity": {
            "counterfactual_valid": attempts > 0 and valid_attempts == attempts,
            "branch_invariants_pass": bool(
                old_certificate.get("branch_invariants_pass", False)
            ),
            "paired_outcome_valid": bool(
                old_certificate.get("paired_outcome_valid", False)
            ),
            "deterministic_policy_mode": bool(
                old_certificate.get("deterministic_policy_mode", False)
            ),
            "teacher_active": not bool(
                old_certificate.get("teacher_inactive", False)
            ),
        },
        "support": source_support.as_dict(),
        "provenance": provenance,
    }


def _record_key(record):
    return (
        str(record["solver"]),
        int(record["seed"]),
        str(record["episode_id"]),
        int(record["step_index"]),
    )


def _anchor_key(anchor):
    return (
        anchor.solver.value,
        int(anchor.seed),
        str(anchor.episode_id),
        int(anchor.step_index),
    )


def _load_old_records(source):
    """Load compact combined records; per-anchor shards are not required."""

    records = {}
    for name in (
        "certified_explanation_instances.jsonl",
        "abstained_explanations.jsonl",
    ):
        path = source / name
        if not path.is_file():
            continue
        for row in read_jsonl(path):
            key = _record_key(row)
            if key in records:
                raise ValueError("duplicate old explanation record: %r" % (key,))
            records[key] = row
    if not records:
        raise FileNotFoundError(
            "source has no combined C-EDDP explanation records"
        )
    return records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/explainability/cedp_v2.yaml"),
    )
    parser.add_argument("--input-dir", type=Path)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("runs/explanations/cedp_v2_support_aware"),
    )
    parser.add_argument(
        "--max-instances",
        type=int,
        default=0,
        help="Optional deterministic smoke-test prefix; zero means all anchors",
    )
    args = parser.parse_args()
    config, _, _, policies, _ = load_runtime(args.config)
    yaml_config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    source = args.input_dir or Path(yaml_config["experiment"]["output_dir"])
    output = args.output_dir
    shard_dir = output / "instance_shards"
    shard_dir.mkdir(parents=True, exist_ok=True)

    all_anchors = tuple(
        FullDecisionAnchor.from_dict(row)
        for row in read_jsonl(source / "full_decision_anchors.jsonl")
    )
    support_config = config.get("verification_support", {})
    support_oracle = SupportOracle.from_anchors(
        all_anchors,
        policies,
        tabular_support_threshold=int(
            support_config.get("tabular_evaluation_count_threshold", 3)
        ),
        continuous_support_quantile=float(
            support_config.get("continuous_knn_quantile", 0.95)
        ),
        continuous_radius_multiplier=float(
            support_config.get("continuous_radius_multiplier", 1.25)
        ),
        continuous_minimum_group_size=int(
            support_config.get("continuous_minimum_group_size", 3)
        ),
    )
    anchors = (
        all_anchors[: int(args.max_instances)]
        if args.max_instances
        else all_anchors
    )
    old_records = _load_old_records(source)
    instances = []
    transitions = Counter()
    for anchor in anchors:
        key = _anchor_key(anchor)
        if key not in old_records:
            raise KeyError("anchor has no old explanation record: %r" % (key,))
        old_shard = old_records[key]
        payload = _reprofiled_payload(
            anchor,
            old_shard,
            policies[anchor.solver.value],
            support_oracle,
        )
        instance = adapt_m1_m13_record(
            payload,
            solver=anchor.solver.value,
            seed=anchor.seed,
            episode_id=anchor.episode_id,
            step_index=anchor.step_index,
        )
        atomic_json(
            shard_dir / (anchor.anchor_id + ".json"),
            instance.as_dict(),
        )
        instances.append(instance)
        transitions[(
            anchor.solver.value,
            old_shard.get("status", "UNKNOWN"),
            instance.status.value,
        )] += 1

    certified = [
        item for item in instances if item.status.value == "CERTIFIED"
    ]
    abstained = [
        item for item in instances if item.status.value == "ABSTAINED"
    ]
    write_jsonl(output / "certified_explanation_instances.jsonl", certified)
    write_jsonl(output / "abstained_explanations.jsonl", abstained)
    write_jsonl(output / "full_decision_anchors.jsonl", anchors)
    for name in ("anchor_summary.json", "provenance_manifest.json"):
        if (source / name).is_file():
            shutil.copyfile(source / name, output / name)
    counts = Counter(item.solver for item in certified)
    summary = {
        "stage": "CEDP2.2-support-aware-reprofile",
        "reprofiled_from": str(source),
        "support_population_anchors": len(all_anchors),
        "requested_anchors": len(anchors),
        "completed_shards": len(instances),
        "certified_instances": len(certified),
        "abstained_instances": len(abstained),
        "certified_by_solver": dict(sorted(counts.items())),
        "status_transitions": {
            "%s:%s->%s" % key: value
            for key, value in sorted(transitions.items())
        },
        "paired_rollouts_reused": True,
        "collection_complete": len(anchors) == len(all_anchors),
        "source_kind": "full_trajectory",
        "eddp_v1_used_as_input": False,
        "support_contract": {
            "tabular_basis": (
                "C-EDP anchor evaluation count; not training visitation"
            ),
            "tabular_threshold": support_oracle.tabular_support_threshold,
            "continuous_basis": (
                "same-semantic-group nearest-neighbour radius"
            ),
            "relation_claim_requires": "both_supported",
        },
    }
    atomic_json(output / "collection_summary.json", summary)
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
