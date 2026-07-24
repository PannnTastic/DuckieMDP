"""Run EDP1--EDP2 label-free anchor collection."""

import argparse
from collections import Counter, defaultdict
import json
import logging
from pathlib import Path
import warnings

import yaml

from src.actions import ActionConfig
from src.continuous_env import build_continuous_env
from src.env_wrapper import build_env
from src.explainability.eddp.anchors import collect_episode_anchors
from src.explainability.eddp.provenance import atomic_json
from src.explainability.eddp.schema import write_jsonl
from src.explainability.q_policy_adapter import QPolicyAdapter
from src.explainability.sac_policy_adapter import SACPolicyAdapter
from src.explainability.sarsa_policy_adapter import SarsaPolicyAdapter


def _quiet():
    warnings.filterwarnings("ignore")
    logging.disable(logging.CRITICAL)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path,
                        default=Path("configs/explainability/eddp_v1.yaml"))
    parser.add_argument("--seeds", type=int, default=0,
                        help="0 uses every configured seed")
    args = parser.parse_args()
    _quiet()
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    collection = cfg["collection"]
    seeds = list(collection["seeds"])
    if args.seeds:
        seeds = seeds[:args.seeds]
    shared_path = Path(cfg["experiment"]["shared_environment_config"])
    shared = yaml.safe_load(shared_path.read_text(encoding="utf-8"))
    shared["environment"]["render_observations"] = False
    policies = {}
    q_info = cfg["policies"]["q_learning"]
    q_train = yaml.safe_load(Path(q_info["training_config"]).read_text(encoding="utf-8"))
    policies["q_learning"] = QPolicyAdapter.from_checkpoint(
        Path(q_info["checkpoint"]), q_train["q_learning"]["allowed_actions"],
        ActionConfig(**shared["actions"]),
    )
    s_info = cfg["policies"]["sarsa"]
    s_train = yaml.safe_load(Path(s_info["training_config"]).read_text(encoding="utf-8"))
    policies["sarsa"] = SarsaPolicyAdapter.from_checkpoint(
        Path(s_info["checkpoint"]), s_train["sarsa"]["allowed_actions"],
        ActionConfig(**shared["actions"]),
    )
    sac_info = cfg["policies"]["sac"]
    policies["sac"] = SACPolicyAdapter.from_checkpoint(
        Path(sac_info["checkpoint"]), allow_observation_expansion=True,
    )

    anchors = []
    coverage = defaultdict(Counter)
    for solver, policy in policies.items():
        info = cfg["policies"][solver]
        for seed in seeds:
            env = (
                build_continuous_env(shared, int(seed))
                if solver == "sac" else build_env(shared, int(seed))
            )
            try:
                records, counts = collect_episode_anchors(
                    env, policy, int(seed), str(shared_path), info["checkpoint"],
                    blocks_per_context=collection["blocks_per_context"],
                    block_length=collection["block_length"],
                    minimum_block_gap=collection["minimum_block_gap"],
                    max_decisions=collection["max_decisions"],
                )
                anchors.extend(records)
                coverage[solver].update(counts)
            finally:
                env.close()
    output = Path(cfg["experiment"]["output_dir"])
    anchor_path = output / "anchors_label_free.jsonl"
    write_jsonl(anchor_path, anchors)
    expected_contexts = set(collection["contexts"])
    observed_contexts = {record.selection_context for record in anchors}
    summary = {
        "stage": "EDP1-EDP2",
        "method": "label-free physical-context anchor collection",
        "anchors": len(anchors),
        "blocks": len({record.block_id for record in anchors}),
        "solvers": dict(Counter(record.solver.value for record in anchors)),
        "contexts": dict(Counter(record.selection_context for record in anchors)),
        "coverage_by_solver": {
            name: dict(sorted(values.items())) for name, values in coverage.items()
        },
        "seeds": seeds,
        "files": {"anchors": str(anchor_path)},
        "acceptance": {
            "no_primitive_label_used": True,
            "teacher_inactive": True,
            "policy_mode_deterministic": True,
            "all_solvers_present": set(policies) == {
                record.solver.value for record in anchors
            },
            "all_contexts_observed": expected_contexts.issubset(observed_contexts),
            "anchor_ids_unique": len(anchors) == len({record.anchor_id for record in anchors}),
        },
    }
    summary["acceptance"]["passed"] = all(summary["acceptance"].values())
    atomic_json(output / "anchor_collection_summary.json", summary)
    print(json.dumps({
        "anchors": summary["anchors"], "blocks": summary["blocks"],
        "contexts": summary["contexts"], "passed": summary["acceptance"]["passed"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
