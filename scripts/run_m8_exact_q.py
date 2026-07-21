"""Generate the M8 exact Q-table characterization and verification audit."""

import argparse
import json
import logging
from pathlib import Path
import warnings

import numpy as np
import yaml

from src.actions import ActionConfig
from src.discretizer import Q_SHAPE, STATE_SHAPE
from src.env_wrapper import build_env
from src.explainability.explain_q import (
    analyze_one_bin_flips,
    collect_evaluation_reach_counts,
    enumerate_q_policy,
    save_flips,
    save_policy_map,
    save_safety_violations,
    save_summary,
    summarize_exact_policy,
    verify_safety_properties,
)
from src.explainability.q_policy_adapter import QPolicyAdapter


def _quiet():
    warnings.filterwarnings("ignore")
    logging.disable(logging.CRITICAL)


def _load_optional_counts(path):
    if path is None:
        return None
    return np.load(str(path), allow_pickle=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--q-checkpoint", type=Path,
        default=Path("artifacts/ablation/full_task/q_learning_teacher/q_table_best.npy"),
    )
    parser.add_argument(
        "--config", type=Path,
        default=Path("configs/small_loop_stop_duck_q.yaml"),
    )
    parser.add_argument("--training-visit-counts", type=Path, default=None)
    parser.add_argument("--episodes", type=int, default=30)
    parser.add_argument("--historical-support-threshold", type=int, default=3)
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("runs/explanations/m8_exact_q"),
    )
    args = parser.parse_args()
    _quiet()

    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    config["environment"]["render_observations"] = False
    policy = QPolicyAdapter.from_checkpoint(
        args.q_checkpoint,
        allowed_actions=config["q_learning"]["allowed_actions"],
        action_config=ActionConfig(**config["actions"]),
    )
    if policy.q_table.shape != Q_SHAPE:
        raise RuntimeError("Q-table shape changed unexpectedly")

    env = build_env(config, int(config["evaluation"]["seeds"][0]))
    try:
        reach_counts, reach_manifest = collect_evaluation_reach_counts(
            env,
            policy,
            episodes=args.episodes,
            seeds=config["evaluation"]["seeds"],
        )
    finally:
        env.close()

    training_counts = _load_optional_counts(args.training_visit_counts)
    records = enumerate_q_policy(
        policy,
        evaluation_reach_counts=reach_counts,
        training_visit_counts=training_counts,
        historical_reach_support_threshold=args.historical_support_threshold,
    )
    flips, flip_summary = analyze_one_bin_flips(records)
    safety = verify_safety_properties(records)
    exact = summarize_exact_policy(records, flip_summary, safety)

    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    reach_path = output / "evaluation_reach_counts.npy"
    reach_tmp = output / "evaluation_reach_counts.tmp.npy"
    np.save(str(reach_tmp), reach_counts, allow_pickle=False)
    reach_tmp.replace(reach_path)
    policy_map = save_policy_map(records, output / "exact_policy_map.csv")
    flip_path = save_flips(flips, output / "one_bin_action_flips.csv")
    violation_path = save_safety_violations(
        records, output / "safety_property_violations.csv"
    )

    property_totals = {
        property_id: result["breakdown"]
        for property_id, result in safety.items()
    }
    summary = {
        "stage": "M8",
        "method": "exact finite Q-policy characterization and verification",
        "checkpoint": {
            "path": str(args.q_checkpoint),
            "sha256": policy.checkpoint_hash,
            "shape": list(policy.q_table.shape),
            "trained_with_teacher_guidance": True,
            "teacher_active_during_m8": False,
        },
        "config": str(args.config),
        "policy_mode": "greedy_teacher_free_lowest_id_tie_break",
        "support_contract": {
            "training_visit_counts_available": training_counts is not None,
            "training_visit_count": (
                "loaded_from_artifact" if training_counts is not None else None
            ),
            "historical_proxy": "evaluation_reach_count >= %d" % args.historical_support_threshold,
            "proxy_is_not_training_visitation": True,
        },
        "evaluation_reach_manifest": reach_manifest,
        "exact_characterization": exact,
        "acceptance": {
            "q_shape_unchanged": list(policy.q_table.shape) == list(Q_SHAPE),
            "enumerated_states": len(records),
            "expected_states": int(np.prod(STATE_SHAPE)),
            "all_states_enumerated_once": len({row.index for row in records}) == int(np.prod(STATE_SHAPE)),
            "safety_properties_checked_exhaustively": property_totals,
            "policy_violations_are_findings_not_pipeline_failures": True,
        },
        "files": {
            "policy_map": str(policy_map),
            "one_bin_flips": str(flip_path),
            "safety_violations": str(violation_path),
            "evaluation_reach_counts": str(reach_path),
        },
    }
    summary_path = save_summary(summary, output / "m8_summary.json")
    print(json.dumps({
        "state_counts": exact["state_counts"],
        "provenance_counts": exact["provenance_counts"],
        "safety": property_totals,
        "one_bin_flips": len(flips),
    }, sort_keys=True))
    print("summary=%s" % summary_path)


if __name__ == "__main__":
    main()
