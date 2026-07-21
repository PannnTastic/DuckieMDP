"""Run M10 solver-aware rule extraction and held-out validation."""

import argparse
from collections import Counter
import csv
import json
import logging
from math import cos
from pathlib import Path
import warnings

import numpy as np
import yaml

from src.actions import ActionConfig
from src.continuous_env import build_continuous_env
from src.discretizer import discretize
from src.env_wrapper import build_env
from src.explainability.primitives import label_primitive
from src.explainability.q_policy_adapter import QPolicyAdapter
from src.explainability.rule_extraction import (
    classification_metrics_by_stratum,
    export_rule_text,
    extract_leaf_rules,
    file_sha256,
    fit_action_regressor,
    fit_classifier,
    library_manifest,
    regression_metrics_by_stratum,
    rules_by_prediction,
    save_model,
    tree_complexity,
)
from src.explainability.sac_policy_adapter import SACPolicyAdapter
from src.explainability.schema import CanonicalAction, SolverKind, to_dict
from src.explainability.semantic_state import (
    canonical_from_continuous_state,
    encode_canonical_for_sac,
)
from src.continuous_state import OBSERVATION_NAMES
from src.state import DuckThreat


Q_FEATURE_NAMES = (
    "d_bin",
    "tracking_error_bin",
    "speed_bin",
    "curvature_bin",
    "stop_distance_bin",
    "stop_satisfied_bin",
    "duck_threat_bin",
)


def _quiet():
    warnings.filterwarnings("ignore")
    logging.disable(logging.CRITICAL)


def _load_yaml(path):
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def _atomic_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _write_text(path, content):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)


def _load_q_policy_map(path):
    with Path(path).open("r", encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    features = np.asarray([
        [int(row[name]) for name in Q_FEATURE_NAMES]
        for row in rows
    ], dtype=np.float64)
    actions = np.asarray([int(row["action_id"]) for row in rows])
    primitives = np.asarray([row["primitive"] for row in rows])
    strata = {
        "representable": np.asarray([row["representable"] == "True" for row in rows]),
        "valid_manifold": np.asarray([row["valid_manifold"] == "True" for row in rows]),
        "reachable": np.asarray([row["reachable"] == "True" for row in rows]),
        "supported": np.asarray([row["supported"] == "True" for row in rows]),
    }
    return rows, features, actions, primitives, strata


def _collect_sac_dataset(policy, config, seeds, decisions_per_seed):
    features = []
    actions = []
    primitives = []
    states = []
    provenance = []
    terminations = Counter()
    for seed in seeds:
        env = build_continuous_env(config, int(seed))
        try:
            env.reset(int(seed))
            for step in range(int(decisions_per_seed)):
                state = canonical_from_continuous_state(env.current_state)
                decision = policy.decide(state)
                primitive = label_primitive(state, decision.action)
                features.append(decision.diagnostics["observation"])
                actions.append((decision.action.v_cmd, decision.action.omega_cmd))
                primitives.append(primitive.primitive.value)
                states.append(state)
                provenance.append({"seed": int(seed), "decision_step": step})
                command = np.asarray(
                    [decision.action.v_cmd, decision.action.omega_cmd],
                    dtype=np.float32,
                )
                _, _, done, info = env.step(command)
                if done:
                    terminations[str(info.get("termination_reason", "unknown"))] += 1
                    break
        finally:
            env.close()
    return {
        "features": np.asarray(features, dtype=np.float64),
        "actions": np.asarray(actions, dtype=np.float64),
        "primitives": np.asarray(primitives),
        "states": states,
        "provenance": provenance,
        "terminations": dict(terminations),
    }


def _sac_strata(states):
    stop = np.asarray([
        state.stop_present and not state.stop_satisfied for state in states
    ], dtype=bool)
    pedestrian = np.asarray([
        state.duck_present and state.duck_active is True for state in states
    ], dtype=bool)
    lane = ~(stop | pedestrian)
    return {
        "lane_context": lane,
        "stop_context": stop,
        "pedestrian_context": pedestrian,
    }


def _primitive_from_actions(states, actions):
    labels = []
    for state, values in zip(states, actions):
        action = CanonicalAction(
            solver=SolverKind.SAC,
            v_cmd=float(values[0]),
            omega_cmd=float(values[1]),
        )
        labels.append(label_primitive(state, action).primitive.value)
    return np.asarray(labels)


def _rollout_metrics(rows):
    returns = [row["return"] for row in rows]
    progress = [row["forward_progress_m"] for row in rows]
    deviations = [value for row in rows for value in row["abs_d_values"]]
    stops = sum(row["full_stops"] for row in rows)
    violations = sum(row["stop_violations"] for row in rows)
    crossing = sum(row["duck_crossing_steps"] for row in rows)
    yields = sum(row["duck_yield_steps"] for row in rows)
    reasons = Counter(row["termination_reason"] for row in rows)
    episodes = len(rows)
    failures = reasons["offroad"] + reasons["duck_collision"] + reasons["other_collision"]
    return {
        "episodes": episodes,
        "mean_return": float(np.mean(returns)),
        "mean_forward_progress_m": float(np.mean(progress)),
        "mean_abs_d": float(np.mean(deviations)) if deviations else None,
        "p95_abs_d": float(np.percentile(deviations, 95)) if deviations else None,
        "timeout_rate": reasons["timeout"] / episodes,
        "offroad_rate": reasons["offroad"] / episodes,
        "duck_collision_rate": reasons["duck_collision"] / episodes,
        "other_collision_rate": reasons["other_collision"] / episodes,
        "total_failure_rate": failures / episodes,
        "stop_compliance_rate": stops / (stops + violations) if stops + violations else 1.0,
        "stop_opportunities": stops + violations,
        "duck_yield_step_rate": yields / crossing if crossing else 1.0,
        "termination_counts": dict(reasons),
        "per_seed": rows,
    }


def _evaluate_q_surrogate(model, q_policy, config, seeds):
    episodes = []
    total_mismatches = 0
    total_decisions = 0
    for seed in seeds:
        env = build_env(config, int(seed))
        try:
            raw = env.reset(int(seed))
            done = False
            total_return = 0.0
            progress = 0.0
            deviations = []
            full_stops = stop_violations = 0
            duck_crossing_steps = duck_yield_steps = 0
            decision_dt = env.unwrapped.delta_time * env.unwrapped.frame_skip
            info = {"termination_reason": "in_progress"}
            while not done:
                index = tuple(int(value) for value in discretize(raw))
                action = int(model.predict(np.asarray(index).reshape(1, -1))[0])
                reference = q_policy.decide_raw(raw).action.action_id
                total_mismatches += int(action != reference)
                total_decisions += 1
                raw, reward, done, info = env.step(action)
                total_return += float(reward)
                deviations.append(abs(float(raw.d)))
                progress += max(0.0, raw.v * cos(raw.phi)) * decision_dt
                events = info["events"]
                full_stops += int(events["full_stop"])
                stop_violations += int(events["stop_violation"])
                crossing = raw.duck in {
                    DuckThreat.CROSSING_FAR, DuckThreat.CROSSING_NEAR
                }
                duck_crossing_steps += int(crossing)
                duck_yield_steps += int(crossing and raw.v < 0.04)
            episodes.append({
                "seed": int(seed),
                "return": total_return,
                "forward_progress_m": progress,
                "abs_d_values": deviations,
                "full_stops": full_stops,
                "stop_violations": stop_violations,
                "duck_crossing_steps": duck_crossing_steps,
                "duck_yield_steps": duck_yield_steps,
                "termination_reason": str(info["termination_reason"]),
            })
        finally:
            env.close()
    report = _rollout_metrics(episodes)
    report["action_mismatches_vs_frozen_q_policy"] = total_mismatches
    report["decisions_compared"] = total_decisions
    report["behavioral_equivalence_observed"] = total_mismatches == 0
    return report


def _evaluate_sac_policy(config, seeds, action_function):
    episodes = []
    for seed in seeds:
        env = build_continuous_env(config, int(seed))
        try:
            env.reset(int(seed))
            done = False
            total_return = 0.0
            progress = 0.0
            deviations = []
            full_stops = stop_violations = 0
            duck_crossing_steps = duck_yield_steps = 0
            decision_dt = env.unwrapped.delta_time * int(config["environment"]["frame_skip"])
            info = {"termination_reason": "in_progress"}
            while not done:
                state = canonical_from_continuous_state(env.current_state)
                action = np.asarray(action_function(state), dtype=np.float32)
                before_crossing = env.current_state.duck_active
                _, reward, done, info = env.step(action)
                after = env.current_state
                total_return += float(reward)
                deviations.append(abs(float(after.d)))
                progress += max(0.0, after.v * cos(after.phi)) * decision_dt
                events = info["events"]
                full_stops += int(events["full_stop"])
                stop_violations += int(events["stop_violation"])
                duck_crossing_steps += int(before_crossing)
                duck_yield_steps += int(before_crossing and float(action[0]) < 0.04)
            episodes.append({
                "seed": int(seed),
                "return": total_return,
                "forward_progress_m": progress,
                "abs_d_values": deviations,
                "full_stops": full_stops,
                "stop_violations": stop_violations,
                "duck_crossing_steps": duck_crossing_steps,
                "duck_yield_steps": duck_yield_steps,
                "termination_reason": str(info["termination_reason"]),
            })
        finally:
            env.close()
    return _rollout_metrics(episodes)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--q-policy-map", type=Path,
        default=Path("runs/explanations/m8_exact_q/exact_policy_map.csv"),
    )
    parser.add_argument(
        "--q-checkpoint", type=Path,
        default=Path("artifacts/ablation/full_task/q_learning_teacher/q_table_best.npy"),
    )
    parser.add_argument(
        "--q-config", type=Path,
        default=Path("configs/small_loop_stop_duck_q.yaml"),
    )
    parser.add_argument(
        "--sac-checkpoint", type=Path,
        default=Path("artifacts/sac/full_repeat_duck_5min/sac_best.pt"),
    )
    parser.add_argument(
        "--sac-config", type=Path,
        default=Path("artifacts/sac/full_repeat_duck_5min/config.yaml"),
    )
    parser.add_argument("--decisions-per-seed", type=int, default=400)
    parser.add_argument("--sac-rollout-seeds", type=int, default=5)
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("runs/explanations/m10_rule_extraction"),
    )
    args = parser.parse_args()
    _quiet()
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)

    q_config = _load_yaml(args.q_config)
    q_config["environment"]["render_observations"] = False
    rows, q_x, q_actions, q_primitives, q_strata = _load_q_policy_map(args.q_policy_map)
    q_action_tree = fit_classifier(q_x, q_actions)
    q_primitive_tree = fit_classifier(q_x, q_primitives)
    q_action_prediction = q_action_tree.predict(q_x)
    q_primitive_prediction = q_primitive_tree.predict(q_x)
    q_action_metrics = classification_metrics_by_stratum(
        q_actions, q_action_prediction, q_strata
    )
    q_primitive_metrics = classification_metrics_by_stratum(
        q_primitives, q_primitive_prediction, q_strata
    )
    q_policy = QPolicyAdapter.from_checkpoint(
        args.q_checkpoint,
        allowed_actions=q_config["q_learning"]["allowed_actions"],
        action_config=ActionConfig(**q_config["actions"]),
    )
    q_rollout = _evaluate_q_surrogate(
        q_action_tree,
        q_policy,
        q_config,
        q_config["evaluation"]["seeds"],
    )

    sac_config = _load_yaml(args.sac_config)
    sac_config["environment"]["render_observations"] = False
    sac_policy = SACPolicyAdapter.from_checkpoint(
        args.sac_checkpoint, allow_observation_expansion=True
    )
    development_seeds = [int(value) for value in sac_config["evaluation"]["development_seeds"]]
    final_seeds = [int(value) for value in sac_config["evaluation"]["final_seeds"]]
    train = _collect_sac_dataset(
        sac_policy, sac_config, development_seeds, args.decisions_per_seed
    )
    test = _collect_sac_dataset(
        sac_policy, sac_config, final_seeds, args.decisions_per_seed
    )
    sac_primitive_tree = fit_classifier(
        train["features"],
        train["primitives"],
        max_depth=12,
        min_samples_leaf=5,
        class_weight=None,
    )
    sac_action_tree = fit_action_regressor(
        train["features"],
        train["actions"],
        max_depth=14,
        min_samples_leaf=3,
    )
    train_primitive_prediction = sac_primitive_tree.predict(train["features"])
    test_primitive_prediction = sac_primitive_tree.predict(test["features"])
    train_action_prediction = sac_action_tree.predict(train["features"])
    test_action_prediction = sac_action_tree.predict(test["features"])
    train_induced_primitives = _primitive_from_actions(
        train["states"], train_action_prediction
    )
    test_induced_primitives = _primitive_from_actions(
        test["states"], test_action_prediction
    )
    train_strata = _sac_strata(train["states"])
    test_strata = _sac_strata(test["states"])
    sac_metrics = {
        "primitive_classifier_train": classification_metrics_by_stratum(
            train["primitives"], train_primitive_prediction, train_strata
        ),
        "primitive_classifier_heldout": classification_metrics_by_stratum(
            test["primitives"], test_primitive_prediction, test_strata
        ),
        "action_induced_primitive_train": classification_metrics_by_stratum(
            train["primitives"], train_induced_primitives, train_strata
        ),
        "action_induced_primitive_heldout": classification_metrics_by_stratum(
            test["primitives"], test_induced_primitives, test_strata
        ),
        "action_regression_train": regression_metrics_by_stratum(
            train["actions"], train_action_prediction, train_strata
        ),
        "action_regression_heldout": regression_metrics_by_stratum(
            test["actions"], test_action_prediction, test_strata
        ),
    }

    action_low = np.asarray([0.0, -1.5], dtype=np.float64)
    action_high = np.asarray([0.41, 1.5], dtype=np.float64)

    def actor_action(state):
        action = sac_policy.decide(state).action
        return (action.v_cmd, action.omega_cmd)

    def surrogate_action(state):
        observation = np.asarray(
            encode_canonical_for_sac(state), dtype=np.float64
        )
        return np.clip(
            sac_action_tree.predict(observation.reshape(1, -1))[0],
            action_low,
            action_high,
        )

    rollout_seeds = final_seeds[: int(args.sac_rollout_seeds)]
    sac_actor_rollout = _evaluate_sac_policy(
        sac_config, rollout_seeds, actor_action
    )
    sac_surrogate_rollout = _evaluate_sac_policy(
        sac_config, rollout_seeds, surrogate_action
    )

    models = {
        "q_action_tree": save_model(q_action_tree, output / "q_action_tree.joblib"),
        "q_primitive_tree": save_model(q_primitive_tree, output / "q_primitive_tree.joblib"),
        "sac_primitive_tree": save_model(sac_primitive_tree, output / "sac_primitive_tree.joblib"),
        "sac_action_tree": save_model(sac_action_tree, output / "sac_action_tree.joblib"),
    }
    _write_text(
        output / "q_action_rules.txt",
        export_rule_text(q_action_tree, Q_FEATURE_NAMES, decimals=2),
    )
    _write_text(
        output / "q_primitive_rules.txt",
        export_rule_text(q_primitive_tree, Q_FEATURE_NAMES, decimals=2),
    )
    _write_text(
        output / "sac_primitive_rules.txt",
        export_rule_text(sac_primitive_tree, OBSERVATION_NAMES, decimals=4),
    )
    _write_text(
        output / "sac_action_rules.txt",
        export_rule_text(sac_action_tree, OBSERVATION_NAMES, decimals=4),
    )
    rule_payload = {
        "q_action": rules_by_prediction(extract_leaf_rules(q_action_tree, Q_FEATURE_NAMES)),
        "q_primitive": rules_by_prediction(extract_leaf_rules(q_primitive_tree, Q_FEATURE_NAMES)),
        "sac_primitive": rules_by_prediction(extract_leaf_rules(sac_primitive_tree, OBSERVATION_NAMES)),
        "sac_action": extract_leaf_rules(sac_action_tree, OBSERVATION_NAMES),
    }
    _atomic_json(output / "leaf_rules.json", rule_payload)

    heldout_primitive = sac_metrics["primitive_classifier_heldout"]["all"]
    heldout_induced = sac_metrics["action_induced_primitive_heldout"]["all"]
    heldout_action = sac_metrics["action_regression_heldout"]["all"]
    main_eligibility = {
        "q_action_tree_exact_on_representable_domain": (
            q_action_metrics["representable"].fidelity == 1.0
        ),
        "q_primitive_tree_exact_on_representable_domain": (
            q_primitive_metrics["representable"].fidelity == 1.0
        ),
        "q_surrogate_rollout_matches_frozen_policy": (
            q_rollout["action_mismatches_vs_frozen_q_policy"] == 0
        ),
        "sac_primitive_classifier_heldout_fidelity_ge_0_85": (
            heldout_primitive.fidelity is not None
            and heldout_primitive.fidelity >= 0.85
        ),
        "sac_action_induced_primitive_fidelity_ge_0_80": (
            heldout_induced.fidelity is not None
            and heldout_induced.fidelity >= 0.80
        ),
        "sac_heldout_mae_v_le_0_03": (
            heldout_action.mae_v is not None and heldout_action.mae_v <= 0.03
        ),
        "sac_heldout_mae_omega_le_0_20": (
            heldout_action.mae_omega is not None and heldout_action.mae_omega <= 0.20
        ),
        "sac_surrogate_rollout_no_collision": (
            sac_surrogate_rollout["duck_collision_rate"] == 0.0
            and sac_surrogate_rollout["other_collision_rate"] == 0.0
        ),
        "sac_surrogate_rollout_failure_rate_le_0_20": (
            sac_surrogate_rollout["total_failure_rate"] <= 0.20
        ),
    }

    summary = {
        "stage": "M10",
        "method": "solver-aware decision-tree rule extraction",
        "library_manifest": library_manifest(),
        "policy_contract": {
            "q_learning": "frozen greedy teacher-free lowest-id tie break",
            "sac": "frozen deterministic actor mean",
            "policies_retrained": False,
            "surrogates_are_original_policies": False,
        },
        "q_learning": {
            "checkpoint": str(args.q_checkpoint),
            "checkpoint_sha256": q_policy.checkpoint_hash,
            "samples": len(q_x),
            "feature_names": list(Q_FEATURE_NAMES),
            "action_metrics": q_action_metrics,
            "primitive_metrics": q_primitive_metrics,
            "action_tree_complexity": tree_complexity(q_action_tree, Q_FEATURE_NAMES),
            "primitive_tree_complexity": tree_complexity(q_primitive_tree, Q_FEATURE_NAMES),
            "rollout": q_rollout,
            "tracking_feature_semantics": "lane_heading_entangled_phi_plus_d",
        },
        "sac": {
            "checkpoint": str(args.sac_checkpoint),
            "checkpoint_sha256": sac_policy.checkpoint_hash,
            "feature_names": list(OBSERVATION_NAMES),
            "development_dataset": {
                "seeds": development_seeds,
                "samples": len(train["features"]),
                "terminations": train["terminations"],
            },
            "heldout_dataset": {
                "seeds": final_seeds,
                "samples": len(test["features"]),
                "terminations": test["terminations"],
            },
            "primitive_semantics": "static P(s,a); temporal transition primitives remain trajectory-level",
            "surrogate_rollout_actor_invocations": 0,
            "metrics": sac_metrics,
            "primitive_tree_complexity": tree_complexity(sac_primitive_tree, OBSERVATION_NAMES),
            "action_tree_complexity": tree_complexity(sac_action_tree, OBSERVATION_NAMES),
            "actor_rollout_same_heldout_seeds": sac_actor_rollout,
            "surrogate_rollout_same_heldout_seeds": sac_surrogate_rollout,
            "closed_loop_fidelity_findings": {
                "mean_return_ratio_surrogate_to_actor": (
                    sac_surrogate_rollout["mean_return"] / sac_actor_rollout["mean_return"]
                ),
                "stop_compliance_drop": (
                    sac_actor_rollout["stop_compliance_rate"]
                    - sac_surrogate_rollout["stop_compliance_rate"]
                ),
                "duck_yield_step_rate_drop": (
                    sac_actor_rollout["duck_yield_step_rate"]
                    - sac_surrogate_rollout["duck_yield_step_rate"]
                ),
            },
        },
        "acceptance": {
            "thresholds_frozen_before_real_run": {
                "sac_primitive_classifier_fidelity": 0.85,
                "sac_action_induced_primitive_fidelity": 0.80,
                "sac_mae_v": 0.03,
                "sac_mae_omega": 0.20,
                "sac_surrogate_total_failure_rate": 0.20,
            },
            "checks": main_eligibility,
            "main_result_eligible": all(main_eligibility.values()),
            "meaning_of_main_result": "global rule summary, not policy replacement",
            "continuous_action_tree_policy_replacement_claim_allowed": False,
            "policy_replacement_denied_by_design_and_closed_loop_safety_gap": True,
            "failed_checks": [name for name, value in main_eligibility.items() if not value],
            "failed_surrogates_retained_for_audit": True,
        },
        "models": {
            name: {"path": str(path), "sha256": file_sha256(path)}
            for name, path in models.items()
        },
        "files": {
            "leaf_rules": str(output / "leaf_rules.json"),
            "q_action_rules": str(output / "q_action_rules.txt"),
            "q_primitive_rules": str(output / "q_primitive_rules.txt"),
            "sac_primitive_rules": str(output / "sac_primitive_rules.txt"),
            "sac_action_rules": str(output / "sac_action_rules.txt"),
        },
    }
    _atomic_json(output / "m10_summary.json", to_dict(summary))
    print(json.dumps({
        "q_action_fidelity": q_action_metrics["representable"].fidelity,
        "q_action_leaves": int(q_action_tree.tree_.n_leaves),
        "sac_heldout_primitive_fidelity": heldout_primitive.fidelity,
        "sac_heldout_induced_primitive_fidelity": heldout_induced.fidelity,
        "sac_heldout_mae_v": heldout_action.mae_v,
        "sac_heldout_mae_omega": heldout_action.mae_omega,
        "sac_actor_failure_rate": sac_actor_rollout["total_failure_rate"],
        "sac_surrogate_failure_rate": sac_surrogate_rollout["total_failure_rate"],
        "main_result_eligible": summary["acceptance"]["main_result_eligible"],
        "failed_checks": summary["acceptance"]["failed_checks"],
    }, sort_keys=True))
    print("summary=%s" % (output / "m10_summary.json"))


if __name__ == "__main__":
    main()
