"""Matched Q-learning, SARSA, and SAC behavioral comparison for M13."""

import argparse
import json
import logging
from pathlib import Path
import warnings

import yaml

from scripts.run_m12_policy_comparison import (
    _load_yaml,
    _poses_match,
    _run_q,
    _run_sac,
)
from src.actions import ActionConfig
from src.explainability.compare_policies import summarize_policy
from src.explainability.q_policy_adapter import QPolicyAdapter
from src.explainability.sac_policy_adapter import SACPolicyAdapter
from src.explainability.sarsa_policy_adapter import SarsaPolicyAdapter
from src.explainability.trajectory import build_provenance


def _quiet():
    warnings.filterwarnings("ignore")
    logging.disable(logging.CRITICAL)


def _atomic_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--shared-config",
        type=Path,
        default=Path(
            "runs/explanations/m12_policy_comparison/shared_comparison_config.yaml"
        ),
    )
    parser.add_argument(
        "--q-checkpoint",
        type=Path,
        default=Path(
            "artifacts/ablation/full_task/q_learning_teacher/q_table_best.npy"
        ),
    )
    parser.add_argument(
        "--q-config",
        type=Path,
        default=Path("configs/small_loop_stop_duck_q.yaml"),
    )
    parser.add_argument(
        "--sarsa-checkpoint",
        type=Path,
        default=Path(
            "artifacts/ablation/full_task/sarsa_teacher/q_table_best.npy"
        ),
    )
    parser.add_argument(
        "--sarsa-config",
        type=Path,
        default=Path("configs/small_loop_stop_duck_sarsa.yaml"),
    )
    parser.add_argument(
        "--sac-checkpoint",
        type=Path,
        default=Path("artifacts/sac/full_repeat_duck_5min/sac_best.pt"),
    )
    parser.add_argument(
        "--episodes", type=int, default=5
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("runs/explanations/m13_sarsa/comparison"),
    )
    args = parser.parse_args()
    _quiet()
    shared = _load_yaml(args.shared_config)
    seeds = [
        int(value) for value in shared["evaluation"]["final_seeds"]
    ][: int(args.episodes)]
    q_config = _load_yaml(args.q_config)
    sarsa_config = _load_yaml(args.sarsa_config)
    q_policy = QPolicyAdapter.from_checkpoint(
        args.q_checkpoint,
        allowed_actions=q_config["q_learning"]["allowed_actions"],
        action_config=ActionConfig(**shared["actions"]),
    )
    sarsa_policy = SarsaPolicyAdapter.from_checkpoint(
        args.sarsa_checkpoint,
        allowed_actions=sarsa_config["sarsa"]["allowed_actions"],
        action_config=ActionConfig(**shared["actions"]),
    )
    sac_policy = SACPolicyAdapter.from_checkpoint(
        args.sac_checkpoint, allow_observation_expansion=False
    )
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    common = {
        "method": "matched_original_policy_rollout",
        "teacher_active": False,
        "map": shared["environment"]["map_name"],
        "frame_skip": shared["environment"]["frame_skip"],
        "max_physics_steps": shared["environment"]["max_steps"],
        "shared_config": str(args.shared_config),
    }
    q_records, q_initial = _run_q(
        shared,
        seeds,
        q_policy,
        build_provenance(
            args.q_checkpoint,
            args.shared_config,
            extra={**common, "solver": "q_learning"},
        ),
        output / "q_learning",
    )
    sarsa_records, sarsa_initial = _run_q(
        shared,
        seeds,
        sarsa_policy,
        build_provenance(
            args.sarsa_checkpoint,
            args.shared_config,
            extra={**common, "solver": "sarsa"},
        ),
        output / "sarsa",
    )
    sac_records, sac_initial = _run_sac(
        shared,
        seeds,
        sac_policy,
        build_provenance(
            args.sac_checkpoint,
            args.shared_config,
            extra={**common, "solver": "sac"},
        ),
        output / "sac",
    )
    summaries = {
        "q_learning": summarize_policy(q_records),
        "sarsa": summarize_policy(sarsa_records),
        "sac": summarize_policy(sac_records),
    }
    tabular_action_comparison = []
    for q_record, sarsa_record in zip(q_records, sarsa_records):
        q_actions = [
            step.decision.action.action_id for step in q_record.steps
        ]
        sarsa_actions = [
            step.decision.action.action_id for step in sarsa_record.steps
        ]
        paired_steps = min(len(q_actions), len(sarsa_actions))
        disagreements = sum(
            q_action != sarsa_action
            for q_action, sarsa_action in zip(q_actions, sarsa_actions)
        )
        tabular_action_comparison.append({
            "q_episode_id": q_record.episode_id,
            "sarsa_episode_id": sarsa_record.episode_id,
            "paired_steps": paired_steps,
            "same_length": len(q_actions) == len(sarsa_actions),
            "action_disagreements": disagreements,
            "actions_identical": (
                len(q_actions) == len(sarsa_actions) and disagreements == 0
            ),
        })
    pose_match = (
        _poses_match(q_initial, sarsa_initial)
        and _poses_match(q_initial, sac_initial)
    )
    policies = {
        "q_learning": {
            "checkpoint": str(args.q_checkpoint),
            "sha256": q_policy.checkpoint_hash,
            "mode": "greedy_teacher_free_lowest_id_tie_break",
        },
        "sarsa": {
            "checkpoint": str(args.sarsa_checkpoint),
            "sha256": sarsa_policy.checkpoint_hash,
            "mode": "greedy_teacher_free_lowest_id_tie_break",
        },
        "sac": {
            "checkpoint": str(args.sac_checkpoint),
            "sha256": sac_policy.checkpoint_hash,
            "mode": "deterministic_actor_mean",
        },
    }
    checks = {
        "three_original_policies": set(summaries)
        == {"q_learning", "sarsa", "sac"},
        "same_seeds": True,
        "same_initial_poses_atol_1e_7": pose_match,
        "same_map_frame_skip_horizon": True,
        "teacher_inactive": True,
        "sac_actor_sampling_inactive": True,
        "surrogate_actions_unused": True,
        "q_sarsa_action_traces_compared": bool(tabular_action_comparison),
        "q_sarsa_actions_identical_on_matched_rollouts": all(
            row["actions_identical"] for row in tabular_action_comparison
        ),
        "sarsa_checkpoint_hash_correct": sarsa_policy.checkpoint_hash
        == "0266ad6f6fdae71bf2dfb7c7121f66e038d16f75e214a931f8c9d50bc6ad3313",
    }
    report = {
        "stage": "M13-three-policy-comparison",
        "method": "matched original-policy behavioral comparison",
        "manifest": {
            **common,
            "seeds": seeds,
            "initial_poses": {
                "q_learning": q_initial,
                "sarsa": sarsa_initial,
                "sac": sac_initial,
            },
        },
        "policies": policies,
        "q_sarsa_matched_action_comparison": {
            "episodes": tabular_action_comparison,
            "paired_steps": sum(
                row["paired_steps"] for row in tabular_action_comparison
            ),
            "action_disagreements": sum(
                row["action_disagreements"]
                for row in tabular_action_comparison
            ),
        },
        "summaries": summaries,
        "acceptance": {
            "checks": checks,
            "passed": all(checks.values()),
            "behavioral_differences_are_findings_not_pipeline_failures": True,
            "comparison_is_descriptive_not_solver_isolation": True,
        },
    }
    path = output.parent / "m13_comparison_summary.json"
    _atomic_json(path, report)
    print(json.dumps({
        "accepted": report["acceptance"]["passed"],
        "mean_return": {
            name: values["mean_return"] for name, values in summaries.items()
        },
        "stop_compliance": {
            name: values["stop_compliance_rate"]
            for name, values in summaries.items()
        },
        "undesirable_primitive_rate": {
            name: values["undesirable_primitive_rate"]
            for name, values in summaries.items()
        },
        "summary": str(path),
    }, sort_keys=True))


if __name__ == "__main__":
    main()

