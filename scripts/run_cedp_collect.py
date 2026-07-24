"""CEDP1--CEDP2: compute certified M1--M13 explanations at every decision."""

import argparse
from collections import Counter
import json
import logging
from pathlib import Path
import warnings

import yaml

from src.explainability.certified_primitives.collection import (
    collect_full_decision_anchors,
    explain_anchor,
)
from src.explainability.certified_primitives.reporting import atomic_json
from src.explainability.certified_primitives.schema import (
    CertifiedExplanationInstance,
    write_jsonl,
)
from src.explainability.eddp.runtime import environment_factory, load_runtime
from src.explainability.eddp.support import SupportOracle


def _quiet():
    warnings.filterwarnings("ignore")
    logging.disable(logging.CRITICAL)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=Path,
        default=Path("configs/explainability/cedp_v2.yaml"),
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--seeds", type=int, default=0,
        help="Use the first N configured base seeds",
    )
    parser.add_argument("--episodes-per-seed", type=int, default=0)
    parser.add_argument("--max-decisions", type=int, default=0)
    parser.add_argument("--max-instances", type=int, default=0)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--anchors-only", action="store_true")
    args = parser.parse_args()
    _quiet()
    config, shared_path, shared, policies, gammas = load_runtime(args.config)
    output = args.output_dir or Path(config["experiment"]["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    collection = config["collection"]
    seeds = list(collection["seeds"])
    if args.seeds:
        seeds = seeds[: args.seeds]
    episodes = args.episodes_per_seed or int(collection["episodes_per_seed"])
    max_decisions = args.max_decisions or int(collection["max_decisions"])
    stride = int(collection["episode_seed_stride"])
    manifest_path = output / "provenance_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError("run CEDP0 first: %s" % manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    anchors = []
    for solver, policy in policies.items():
        checkpoint = Path(config["policies"][solver]["checkpoint"])
        for base_seed in seeds:
            for episode_index in range(int(episodes)):
                reset_seed = int(base_seed) + episode_index * stride
                env = environment_factory(shared, solver)(reset_seed)
                try:
                    anchors.extend(collect_full_decision_anchors(
                        env,
                        policy,
                        reset_seed,
                        shared_path,
                        checkpoint,
                        max_decisions=max_decisions,
                    ))
                finally:
                    env.close()
    anchor_path = output / "full_decision_anchors.jsonl"
    write_jsonl(anchor_path, anchors)
    atomic_json(output / "anchor_summary.json", {
        "stage": "CEDP1",
        "anchors": len(anchors),
        "solvers": dict(Counter(item.solver.value for item in anchors)),
        "source": "full policy rollout; no EDDP v1 atoms",
        "all_decisions_recorded": True,
        "teacher_inactive": True,
    })
    if args.anchors_only:
        print("CEDP1 PASS anchors=%d" % len(anchors))
        return

    support_config = config.get("verification_support", {})
    support_oracle = SupportOracle.from_anchors(
        anchors,
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

    selected = anchors[int(args.start_index):]
    if args.max_instances:
        selected = selected[: int(args.max_instances)]
    shard_dir = output / "instance_shards"
    shard_dir.mkdir(parents=True, exist_ok=True)
    instances = []
    failures = []
    counterfactual = config["counterfactual"]
    for index, anchor in enumerate(selected, start=1):
        shard = shard_dir / (anchor.anchor_id + ".json")
        if shard.is_file():
            instances.append(CertifiedExplanationInstance.from_dict(
                json.loads(shard.read_text(encoding="utf-8"))
            ))
            continue
        try:
            instance = explain_anchor(
                anchor,
                shared_environment=shared,
                shared_config_path=shared_path,
                policy=policies[anchor.solver.value],
                gamma=gammas[anchor.solver.value],
                max_horizon=int(counterfactual["max_horizon"]),
                fixed_horizons=counterfactual["fixed_horizons"],
                event_horizon=bool(
                    counterfactual.get("event_horizon", False)
                ),
                provenance_manifest_sha256=manifest["manifest_sha256"],
                paired_report_path=(
                    output / "paired_reports" / (anchor.anchor_id + ".json")
                ),
                support_oracle=support_oracle,
            )
            atomic_json(shard, instance.as_dict())
            instances.append(instance)
        except Exception as error:
            failures.append({
                "anchor_id": anchor.anchor_id,
                "solver": anchor.solver.value,
                "seed": anchor.seed,
                "step_index": anchor.step_index,
                "error_type": type(error).__name__,
                "error": str(error),
            })
        if index % 10 == 0:
            print("explained=%d/%d success=%d failures=%d" % (
                index, len(selected), len(instances), len(failures)
            ), flush=True)

    instances = [
        CertifiedExplanationInstance.from_dict(
            json.loads(path.read_text(encoding="utf-8"))
        )
        for path in sorted(shard_dir.glob("*.json"))
    ]
    certified = [
        item for item in instances if item.status.value == "CERTIFIED"
    ]
    abstained = [
        item for item in instances if item.status.value == "ABSTAINED"
    ]
    write_jsonl(output / "certified_explanation_instances.jsonl", certified)
    write_jsonl(output / "abstained_explanations.jsonl", abstained)
    atomic_json(output / "collection_failures.json", {"failures": failures})
    minimum = int(collection["minimum_certified_instances_per_solver"])
    counts = Counter(item.solver for item in certified)
    complete = len(instances) == len(anchors) and not failures
    summary = {
        "stage": "CEDP1-CEDP2",
        "requested_anchors": len(anchors),
        "selected_in_invocation": len(selected),
        "completed_shards": len(instances),
        "certified_instances": len(certified),
        "abstained_instances": len(abstained),
        "failures_in_invocation": len(failures),
        "certified_by_solver": dict(sorted(counts.items())),
        "collection_complete": complete,
        "main_budget_met": all(counts[name] >= minimum for name in policies),
        "source_kind": "full_trajectory",
        "eddp_v1_used_as_input": False,
        "support_contract": {
            "source": "frozen C-EDP full-decision anchors",
            "tabular_basis": "evaluation count, not training visitation",
            "tabular_threshold": support_oracle.tabular_support_threshold,
            "continuous_basis": (
                "same-semantic-group nearest-neighbour radius"
            ),
        },
    }
    atomic_json(output / "collection_summary.json", summary)
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
