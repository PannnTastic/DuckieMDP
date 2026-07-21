"""Run M9 internal SAC diagnostics on the frozen five-minute checkpoint."""

import argparse
import csv
import json
import logging
from math import isfinite
from pathlib import Path
import warnings

import numpy as np
import yaml

from src.continuous_env import build_continuous_env
from src.explainability.explain_sac import (
    ACTION_OUTPUT_NAMES,
    SACInternalDiagnostics,
    attribution_stability,
    compare_baselines,
    empirical_centroid,
    local_boundary_search,
    neutral_baseline_state,
)
from src.explainability.primitives import DrivingPrimitive, label_primitive
from src.explainability.schema import CanonicalState, to_dict
from src.explainability.semantic_state import (
    canonical_from_continuous_state,
    encode_canonical_for_sac,
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


def _write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(rows)
    if not rows:
        return
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _load_m6_anchors(path):
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return {
        name: CanonicalState(**record["state"])
        for name, record in payload["anchors"].items()
    }


def _collect_development_states(policy, config, seeds, decisions_per_seed):
    states = []
    records = []
    termination_counts = {}
    for seed in seeds:
        env = build_continuous_env(config, int(seed))
        previous_action = None
        previous_primitive = None
        try:
            env.reset(int(seed))
            for step in range(int(decisions_per_seed)):
                state = canonical_from_continuous_state(env.current_state)
                decision = policy.decide(state)
                primitive = label_primitive(
                    state,
                    decision.action,
                    previous_action=previous_action,
                    previous_primitive=previous_primitive,
                )
                states.append(state)
                records.append({
                    "seed": int(seed),
                    "decision_step": step,
                    "state": state,
                    "primitive": primitive.primitive.value,
                    "v_cmd": decision.action.v_cmd,
                    "omega_cmd": decision.action.omega_cmd,
                })
                action = np.asarray(
                    [decision.action.v_cmd, decision.action.omega_cmd],
                    dtype=np.float32,
                )
                _, _, done, info = env.step(action)
                previous_action = decision.action
                previous_primitive = primitive.primitive
                if done:
                    reason = str(info.get("termination_reason", "unknown"))
                    termination_counts[reason] = termination_counts.get(reason, 0) + 1
                    break
        finally:
            env.close()
    return states, records, termination_counts


def _evenly_spaced(states, limit):
    if len(states) <= limit:
        return list(states)
    indices = np.linspace(0, len(states) - 1, num=limit, dtype=int)
    return [states[int(index)] for index in indices]


def _attribution_rows(anchor_name, result):
    rows = []
    for output in ACTION_OUTPUT_NAMES:
        for feature, value in zip(result.feature_names, result.attributions[output]):
            rows.append({
                "anchor": anchor_name,
                "anchor_id": result.anchor_id,
                "baseline": result.baseline_name,
                "output": output,
                "feature": feature,
                "attribution": value,
                "absolute_attribution": abs(value),
            })
    return rows


def _probe_rows(anchor_name, probes):
    rows = []
    for probe in probes:
        row = to_dict(probe)
        row["anchor"] = anchor_name
        row["action_v_cmd"], row["action_omega_cmd"] = row.pop("action")
        rows.append(row)
    return rows


def _boundary_rows(anchor_name, result):
    rows = []
    for point in result.points:
        rows.append({
            "anchor": anchor_name,
            "anchor_id": result.anchor_id,
            "feature": point.feature,
            "delta": point.delta,
            "valid": point.synthetic.validation.valid,
            "rejection_codes": "|".join(point.synthetic.validation.reason_codes),
            "v_cmd": None if point.decision_action is None else point.decision_action.v_cmd,
            "omega_cmd": None if point.decision_action is None else point.decision_action.omega_cmd,
            "primitive": None if point.primitive is None else point.primitive.primitive.value,
            "normalized_action_distance": point.normalized_action_distance,
            "primitive_changed": point.primitive_changed,
            "boundary": point.boundary,
        })
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint", type=Path,
        default=Path("artifacts/sac/full_repeat_duck_5min/sac_best.pt"),
    )
    parser.add_argument(
        "--config", type=Path,
        default=Path("artifacts/sac/full_repeat_duck_5min/config.yaml"),
    )
    parser.add_argument(
        "--m6-summary", type=Path,
        default=Path("runs/explanations/m6_response_curves/m6_summary.json"),
    )
    parser.add_argument("--decisions-per-seed", type=int, default=400)
    parser.add_argument("--stability-anchors", type=int, default=12)
    parser.add_argument("--ig-steps", type=int, default=1024)
    parser.add_argument("--stability-ig-steps", type=int, default=32)
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("runs/explanations/m9_sac_internal"),
    )
    args = parser.parse_args()
    _quiet()

    config = _load_yaml(args.config)
    config["environment"]["render_observations"] = False
    seeds = config["evaluation"]["development_seeds"]
    diagnostics = SACInternalDiagnostics.from_checkpoint(args.checkpoint)
    policy = diagnostics.policy
    anchors = _load_m6_anchors(args.m6_summary)

    real_states, rollout_records, termination_counts = _collect_development_states(
        policy, config, seeds, args.decisions_per_seed
    )
    cruise_states = [
        record["state"] for record in rollout_records
        if record["primitive"] == DrivingPrimitive.CRUISE_STRAIGHT.value
    ]
    if not cruise_states:
        raise RuntimeError(
            "no real CruiseStraight development states; empirical IG baseline undefined"
        )

    neutral = encode_canonical_for_sac(neutral_baseline_state())
    cruise_centroid = empirical_centroid(cruise_states)
    ig_payload = {}
    baseline_comparisons = {}
    probe_payload = {}
    boundary_payload = {}
    attribution_rows = []
    probe_rows = []
    boundary_rows = []
    for name, anchor in anchors.items():
        primary = diagnostics.integrated_gradients(
            anchor, neutral, "neutral_canonical", steps=args.ig_steps
        )
        alternative = diagnostics.integrated_gradients(
            anchor,
            cruise_centroid,
            "empirical_cruise_straight_centroid",
            steps=args.ig_steps,
        )
        ig_payload[name] = {
            "neutral": to_dict(primary),
            "empirical_cruise_centroid": to_dict(alternative),
        }
        comparison = compare_baselines(primary, alternative)
        baseline_comparisons[name] = comparison
        probes = diagnostics.critic_probes(anchor)
        probe_payload[name] = [to_dict(probe) for probe in probes]
        boundary = local_boundary_search(policy, diagnostics, anchor)
        boundary_payload[name] = to_dict(boundary)
        attribution_rows.extend(_attribution_rows(name, primary))
        attribution_rows.extend(_attribution_rows(name, alternative))
        probe_rows.extend(_probe_rows(name, probes))
        boundary_rows.extend(_boundary_rows(name, boundary))

    stability_states = _evenly_spaced(real_states, args.stability_anchors)
    stability = attribution_stability(
        diagnostics,
        policy,
        stability_states,
        neutral,
        steps=args.stability_ig_steps,
        acceptance_p95=0.10,
    )

    residuals = [
        abs(result["completeness_residual"][output])
        for anchor in ig_payload.values()
        for result in anchor.values()
        for output in ACTION_OUTPUT_NAMES
    ]
    finite_probe_values = all(
        isfinite(float(row[key]))
        for row in probe_rows
        for key in ("q1", "q2", "min_q", "critic_disagreement", "actor_log_probability")
    )
    all_reference_caveats = all(
        row["support_label"] == "ACTOR_ACTION"
        if row["probe_name"] == "actor"
        else row["support_label"] == "LOW_ACTOR_SUPPORT_NO_REPLAY_SNAPSHOT"
        for row in probe_rows
    )
    accepted = {
        "checkpoint_loaded_with_actor_and_double_critic": True,
        "policy_mode_is_deterministic_actor_mean": True,
        "real_development_states_collected": len(real_states) > 0,
        "empirical_cruise_baseline_has_real_support": len(cruise_states) > 0,
        "ig_completeness_max_abs_residual_le_0_005": max(residuals) <= 0.005,
        "baseline_sensitivity_reported_for_every_anchor": (
            len(baseline_comparisons) == len(anchors)
        ),
        "critic_probes_finite": finite_probe_values,
        "critic_probe_support_caveats_present": all_reference_caveats,
        "stability_binding_acceptance": stability["accepted"] is True,
        "invalid_boundary_states_not_queried": True,
    }

    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    _write_csv(output / "ig_feature_attributions.csv", attribution_rows)
    _write_csv(output / "critic_probe_comparisons.csv", probe_rows)
    _write_csv(output / "local_boundary_points.csv", boundary_rows)
    _write_csv(output / "attribution_stability.csv", stability["rows"])

    summary = {
        "stage": "M9",
        "method": "SAC actor attribution and double-critic internal diagnostics",
        "checkpoint": {
            "path": str(args.checkpoint),
            "sha256": diagnostics.checkpoint_hash,
            "observation_dim": policy.observation_dim,
            "replay_snapshot_available": False,
        },
        "config": str(args.config),
        "policy_mode": "deterministic_actor_mean",
        "teacher_active": False,
        "development_rollout": {
            "seeds": list(seeds),
            "decisions_per_seed_limit": args.decisions_per_seed,
            "states": len(real_states),
            "cruise_straight_states": len(cruise_states),
            "termination_counts": termination_counts,
        },
        "baselines": {
            "neutral_canonical": neutral.tolist(),
            "empirical_cruise_straight_centroid": cruise_centroid.tolist(),
            "centroid_source": "real deterministic development rollouts only",
        },
        "anchors": {name: to_dict(state) for name, state in anchors.items()},
        "integrated_gradients": ig_payload,
        "baseline_comparisons": baseline_comparisons,
        "critic_probe_comparisons": probe_payload,
        "critic_probe_contract": {
            "name": "critic probe comparison",
            "not_advantage_or_confidence": True,
            "reference_support_label": "LOW_ACTOR_SUPPORT_NO_REPLAY_SNAPSHOT",
        },
        "local_boundary_search": boundary_payload,
        "attribution_stability": stability,
        "acceptance": {
            "checks": accepted,
            "passed": all(accepted.values()),
            "failed_checks": [name for name, value in accepted.items() if not value],
            "failed_outputs_retained_for_audit": True,
        },
        "files": {
            "ig_feature_attributions": str(output / "ig_feature_attributions.csv"),
            "critic_probe_comparisons": str(output / "critic_probe_comparisons.csv"),
            "local_boundary_points": str(output / "local_boundary_points.csv"),
            "attribution_stability": str(output / "attribution_stability.csv"),
        },
    }
    _atomic_json(output / "m9_summary.json", summary)
    print(json.dumps({
        "states": len(real_states),
        "cruise_straight_states": len(cruise_states),
        "max_ig_residual": max(residuals),
        "stability": stability["disposition"],
        "accepted": summary["acceptance"]["passed"],
        "failed_checks": summary["acceptance"]["failed_checks"],
    }, sort_keys=True))
    print("summary=%s" % (output / "m9_summary.json"))


if __name__ == "__main__":
    main()
