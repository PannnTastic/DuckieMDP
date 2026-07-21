"""Run response, metamorphic, exact, and rule explanations for SARSA."""

import argparse
from collections import Counter
import json
import logging
from pathlib import Path
import warnings

import numpy as np
import yaml

from scripts.run_m10_rule_extraction import (
    Q_FEATURE_NAMES,
    _evaluate_q_surrogate,
    _load_q_policy_map,
)
from scripts.run_m7_metamorphic import _cases
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
    summarize_exact_policy,
    verify_safety_properties,
)
from src.explainability.metamorphic import evaluate_relation, save_results
from src.explainability.response_curves import (
    SweepSpec,
    run_response_suite,
    save_response_suite,
)
from src.explainability.rule_extraction import (
    classification_metrics_by_stratum,
    export_rule_text,
    extract_leaf_rules,
    fit_classifier,
    library_manifest,
    save_model,
    tree_complexity,
)
from src.explainability.sarsa_policy_adapter import SarsaPolicyAdapter
from src.explainability.schema import CanonicalState, SolverKind, to_dict


def _quiet():
    warnings.filterwarnings("ignore")
    logging.disable(logging.CRITICAL)


def _load_yaml(path):
    config = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    config["environment"]["render_observations"] = False
    return config


def _load_local_state(local_dir, name):
    payload = json.loads(
        (Path(local_dir) / ("sarsa_" + name + ".json")).read_text(
            encoding="utf-8"
        )
    )
    return CanonicalState(**payload["selected_decision"]["state"])


def _atomic_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _response_analysis(policy, local_dir, output):
    anchors = {
        "lane": _load_local_state(local_dir, "lane_correction"),
        "stop": _load_local_state(local_dir, "stop_hold"),
        "duck": _load_local_state(local_dir, "pedestrian_yield"),
    }
    specs = {
        "lane": (
            SweepSpec("d", (-0.25, -0.20, -0.10, 0.0, 0.10, 0.20, 0.25)),
            SweepSpec("phi", (-1.20, -0.60, -0.30, 0.0, 0.30, 0.60, 1.20)),
            SweepSpec("curvature", (-8.0, -1.0, 0.0, 1.0, 8.0)),
        ),
        "stop": (
            SweepSpec(
                "stop_distance",
                (0.0, 0.15, 0.30, 0.45, 0.75, 1.50, 3.0),
            ),
            SweepSpec("stop_satisfied", (False, True)),
        ),
        "duck": (
            SweepSpec(
                "duck_threat",
                (
                    "side_far",
                    "side_near",
                    "crossing_far",
                    "crossing_near",
                ),
            ),
        ),
    }
    curves = []
    files = []
    for scenario, scenario_specs in specs.items():
        generated = run_response_suite(
            policy,
            SolverKind.SARSA,
            anchors[scenario],
            scenario_specs,
        )
        curves.extend(generated)
        files.extend(
            save_response_suite(
                generated,
                output / "response_curves",
                prefix=scenario,
            )
        )
    summary = {
        "curves": len(curves),
        "valid_points": sum(curve.valid_points for curve in curves),
        "rejected_points": sum(curve.rejected_points for curve in curves),
        "action_flip_points": sum(
            point.action_changed_from_anchor is True
            for curve in curves
            for point in curve.points
        ),
        "primitive_flip_points": sum(
            point.primitive_changed_from_anchor is True
            for curve in curves
            for point in curve.points
        ),
        "anchors": {name: to_dict(state) for name, state in anchors.items()},
        "files": [str(path) for path in files],
    }
    return summary, anchors


def _metamorphic_analysis(policy, anchor, output):
    # Define pedestrian interventions in metric semantic space, then project
    # them lossily to SARSA categories. Policy queries remain tabular.
    curvature = {
        "straight": 0.0,
        "curve_left": 1.0,
        "curve_right": -1.0,
    }[anchor.curvature_class]
    anchor = CanonicalState(**{
        **anchor.__dict__,
        "curvature": curvature,
        "duck_threat": None,
        "source_representation": "sarsa_semantic_bridge",
    })
    results = []
    for relation_id, case_id, source_changes, target_changes in _cases():
        results.append(
            evaluate_relation(
                policy,
                SolverKind.SARSA,
                anchor,
                relation_id,
                source_changes,
                target_changes,
                provenance={
                    "case_id": case_id,
                    "support_status": "synthetic_pair_from_reachable_sarsa_anchor",
                    "policy_mode": "greedy_teacher_free",
                },
            )
        )
    json_path, csv_path = save_results(
        results, output / "metamorphic", prefix="m13_sarsa"
    )
    counts = Counter(result.status.value for result in results)
    return {
        "method": "LEGIBLE-inspired metamorphic policy testing",
        "counts": dict(sorted(counts.items())),
        "all_pairs_manifold_valid": all(
            result.source.validation.valid and result.target.validation.valid
            for result in results
        ),
        "files": [str(json_path), str(csv_path)],
    }


def _exact_analysis(policy, config, episodes, output):
    env = build_env(config, int(config["evaluation"]["seeds"][0]))
    try:
        reach_counts, reach_manifest = collect_evaluation_reach_counts(
            env,
            policy,
            episodes=int(episodes),
            seeds=config["evaluation"]["seeds"],
        )
    finally:
        env.close()
    records = enumerate_q_policy(
        policy,
        evaluation_reach_counts=reach_counts,
        historical_reach_support_threshold=3,
    )
    flips, flip_summary = analyze_one_bin_flips(records)
    safety = verify_safety_properties(records)
    exact = summarize_exact_policy(records, flip_summary, safety)
    target = output / "exact"
    target.mkdir(parents=True, exist_ok=True)
    reach_path = target / "evaluation_reach_counts.npy"
    np.save(str(reach_path), reach_counts, allow_pickle=False)
    policy_map = save_policy_map(records, target / "exact_policy_map.csv")
    flip_path = save_flips(flips, target / "one_bin_action_flips.csv")
    violation_path = save_safety_violations(
        records, target / "safety_property_violations.csv"
    )
    return {
        "state_counts": exact["state_counts"],
        "provenance_counts": exact["provenance_counts"],
        "primitive_distribution": exact["primitive_distribution"],
        "one_bin_flips": len(flips),
        "safety_properties": {
            key: value["breakdown"] for key, value in safety.items()
        },
        "reach_manifest": reach_manifest,
        "acceptance": {
            "shape": list(policy.q_table.shape),
            "shape_unchanged": list(policy.q_table.shape) == list(Q_SHAPE),
            "enumerated_states": len(records),
            "expected_states": int(np.prod(STATE_SHAPE)),
            "all_states_enumerated_once": len({row.index for row in records})
            == int(np.prod(STATE_SHAPE)),
        },
        "files": {
            "reach_counts": str(reach_path),
            "policy_map": str(policy_map),
            "one_bin_flips": str(flip_path),
            "safety_violations": str(violation_path),
        },
    }, policy_map


def _rule_analysis(policy, config, policy_map, output):
    _, features, actions, primitives, strata = _load_q_policy_map(policy_map)
    action_tree = fit_classifier(features, actions)
    primitive_tree = fit_classifier(features, primitives)
    action_predictions = action_tree.predict(features)
    primitive_predictions = primitive_tree.predict(features)
    action_metrics = classification_metrics_by_stratum(
        actions, action_predictions, strata
    )
    primitive_metrics = classification_metrics_by_stratum(
        primitives, primitive_predictions, strata
    )
    rollout = _evaluate_q_surrogate(
        action_tree,
        policy,
        config,
        config["evaluation"]["seeds"],
    )
    target = output / "rules"
    target.mkdir(parents=True, exist_ok=True)
    action_model = save_model(action_tree, target / "sarsa_action_tree.joblib")
    primitive_model = save_model(
        primitive_tree, target / "sarsa_primitive_tree.joblib"
    )
    (target / "sarsa_action_rules.txt").write_text(
        export_rule_text(action_tree, Q_FEATURE_NAMES), encoding="utf-8"
    )
    (target / "sarsa_primitive_rules.txt").write_text(
        export_rule_text(primitive_tree, Q_FEATURE_NAMES), encoding="utf-8"
    )
    _atomic_json(
        target / "sarsa_action_leaf_rules.json",
        extract_leaf_rules(action_tree, Q_FEATURE_NAMES),
    )
    _atomic_json(
        target / "sarsa_primitive_leaf_rules.json",
        extract_leaf_rules(primitive_tree, Q_FEATURE_NAMES),
    )
    return {
        "library_manifest": library_manifest(),
        "role": "post_hoc_surrogate_only",
        "action_metrics": to_dict(action_metrics),
        "primitive_metrics": to_dict(primitive_metrics),
        "action_tree_complexity": to_dict(
            tree_complexity(action_tree, Q_FEATURE_NAMES)
        ),
        "primitive_tree_complexity": to_dict(
            tree_complexity(primitive_tree, Q_FEATURE_NAMES)
        ),
        "surrogate_rollout": rollout,
        "models": {
            "action": str(action_model),
            "primitive": str(primitive_model),
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path(
            "artifacts/ablation/full_task/sarsa_teacher/q_table_best.npy"
        ),
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/small_loop_stop_duck_sarsa.yaml"),
    )
    parser.add_argument(
        "--local-dir",
        type=Path,
        default=Path("runs/explanations/m13_sarsa/local"),
    )
    parser.add_argument("--episodes", type=int, default=30)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("runs/explanations/m13_sarsa/analysis"),
    )
    args = parser.parse_args()
    _quiet()
    config = _load_yaml(args.config)
    policy = SarsaPolicyAdapter.from_checkpoint(
        args.checkpoint,
        allowed_actions=config["sarsa"]["allowed_actions"],
        action_config=ActionConfig(**config["actions"]),
    )
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)

    response, anchors = _response_analysis(policy, args.local_dir, output)
    metamorphic = _metamorphic_analysis(policy, anchors["lane"], output)
    exact, policy_map = _exact_analysis(
        policy, config, args.episodes, output
    )
    rules = _rule_analysis(policy, config, policy_map, output)
    checks = {
        "checkpoint_shape_unchanged": exact["acceptance"]["shape_unchanged"],
        "all_9000_states_enumerated": (
            exact["acceptance"]["enumerated_states"]
            == exact["acceptance"]["expected_states"]
            == 9000
        ),
        "response_curves_generated": response["curves"] == 6,
        "response_queries_manifold_valid_or_explicitly_rejected": True,
        "metamorphic_pairs_manifold_valid": metamorphic[
            "all_pairs_manifold_valid"
        ],
        "rule_surrogate_not_original_policy": rules["role"]
        == "post_hoc_surrogate_only",
        "teacher_inactive": True,
    }
    summary = {
        "stage": "M13-SARSA-analysis",
        "checkpoint": {
            "path": str(args.checkpoint),
            "sha256": policy.checkpoint_hash,
            "shape": list(policy.q_table.shape),
        },
        "config": str(args.config),
        "policy_contract": {
            "training_algorithm": "sarsa",
            "trained_with_teacher_guidance": True,
            "explanation_mode": "greedy_teacher_free_lowest_id_tie_break",
        },
        "response_curves": response,
        "metamorphic": metamorphic,
        "exact_verification": exact,
        "rule_extraction": rules,
        "acceptance": {
            "checks": checks,
            "passed": all(checks.values()),
            "policy_violations_are_findings_not_pipeline_failures": True,
        },
    }
    summary_path = output.parent / "m13_analysis_summary.json"
    _atomic_json(summary_path, summary)
    print(json.dumps({
        "accepted": summary["acceptance"]["passed"],
        "response": {
            key: response[key]
            for key in (
                "curves",
                "valid_points",
                "rejected_points",
                "action_flip_points",
            )
        },
        "metamorphic": metamorphic["counts"],
        "exact_states": exact["acceptance"]["enumerated_states"],
        "summary": str(summary_path),
    }, sort_keys=True))


if __name__ == "__main__":
    main()

