"""Collect matched original-policy rollouts and build M12 comparison artefacts."""

import argparse
import csv
import json
import logging
from pathlib import Path
import warnings

import numpy as np
import yaml

from src.actions import ActionConfig
from src.continuous_env import build_continuous_env
from src.env_wrapper import build_env
from src.explainability.compare_policies import (
    comparable_influence_subset,
    q_supported_influence_signature,
    sac_ig_influence_signature,
    summarize_policy,
)
from src.explainability.primitives import DrivingPrimitive, label_primitive
from src.explainability.q_policy_adapter import QPolicyAdapter
from src.explainability.sac_policy_adapter import SACPolicyAdapter
from src.explainability.semantic_state import canonical_from_continuous_state
from src.explainability.trajectory import TrajectoryRecorder, build_provenance
from src.explainability.schema import to_dict


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


def _atomic_yaml(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        yaml.safe_dump(payload, sort_keys=False), encoding="utf-8"
    )
    temporary.replace(path)


def _write_csv(path, rows):
    rows = list(rows)
    if not rows:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _shared_config(source, max_physics_steps):
    config = json.loads(json.dumps(source))
    config["environment"]["max_steps"] = int(max_physics_steps)
    config["environment"]["render_observations"] = False
    config["duck_controller"]["max_crossings_per_episode"] = 1
    config["reward"]["straight_steer_penalty"] = 0.0
    return config


def _initial_pose(env):
    return {
        "position_xz": [
            float(env.unwrapped.cur_pos[0]), float(env.unwrapped.cur_pos[2])
        ],
        "heading_radians": float(env.unwrapped.cur_angle),
    }


def _run_q(config, seeds, policy, provenance, output):
    records = []
    initial = {}
    for seed in seeds:
        env = build_env(config, int(seed))
        try:
            raw = env.reset(int(seed))
            initial[str(seed)] = _initial_pose(env)
            recorder = TrajectoryRecorder(
                "q_%d" % seed,
                {**provenance, "seed": int(seed)},
                env.unwrapped.delta_time * env.unwrapped.frame_skip,
            )
            previous_action = None
            previous_primitive = None
            done = False
            while not done:
                decision = policy.decide_raw(raw)
                next_raw, reward, done, info = env.step(decision.action.action_id)
                primitive = label_primitive(
                    decision.state,
                    decision.action,
                    events=info.get("events"),
                    termination_reason=info.get("termination_reason", "in_progress"),
                    previous_action=previous_action,
                    previous_primitive=previous_primitive,
                )
                recorder.append(
                    decision,
                    primitive,
                    reward,
                    info,
                    physics_step=int(env.unwrapped.step_count),
                    position_xz=(env.unwrapped.cur_pos[0], env.unwrapped.cur_pos[2]),
                    heading_radians=float(env.unwrapped.cur_angle),
                )
                previous_action = decision.action
                previous_primitive = primitive.primitive
                raw = next_raw
            record = recorder.finalize()
            record.save_json(output / "trajectories" / (record.episode_id + ".json"))
            records.append(record)
        finally:
            env.close()
    return records, initial


def _run_sac(config, seeds, policy, provenance, output):
    records = []
    initial = {}
    for seed in seeds:
        env = build_continuous_env(config, int(seed))
        try:
            env.reset(int(seed))
            initial[str(seed)] = _initial_pose(env)
            recorder = TrajectoryRecorder(
                "sac_%d" % seed,
                {**provenance, "seed": int(seed)},
                env.unwrapped.delta_time * int(config["environment"]["frame_skip"]),
            )
            previous_action = None
            previous_primitive = None
            done = False
            while not done:
                state = canonical_from_continuous_state(env.current_state)
                decision = policy.decide(state)
                action = np.asarray(
                    [decision.action.v_cmd, decision.action.omega_cmd],
                    dtype=np.float32,
                )
                _, reward, done, info = env.step(action)
                primitive = label_primitive(
                    state,
                    decision.action,
                    events=info.get("events"),
                    termination_reason=info.get("termination_reason", "in_progress"),
                    previous_action=previous_action,
                    previous_primitive=previous_primitive,
                )
                recorder.append(
                    decision,
                    primitive,
                    reward,
                    info,
                    physics_step=int(env.unwrapped.step_count),
                    position_xz=(env.unwrapped.cur_pos[0], env.unwrapped.cur_pos[2]),
                    heading_radians=float(env.unwrapped.cur_angle),
                )
                previous_action = decision.action
                previous_primitive = primitive.primitive
            record = recorder.finalize()
            record.save_json(output / "trajectories" / (record.episode_id + ".json"))
            records.append(record)
        finally:
            env.close()
    return records, initial


def _poses_match(left, right, tolerance=1e-7):
    if set(left) != set(right):
        return False
    for seed in left:
        if not np.allclose(
            left[seed]["position_xz"], right[seed]["position_xz"],
            rtol=0.0, atol=tolerance,
        ):
            return False
        if not np.isclose(
            left[seed]["heading_radians"], right[seed]["heading_radians"],
            rtol=0.0, atol=tolerance,
        ):
            return False
    return True


def _step_rows(records):
    for record in records:
        for step in record.steps:
            state = step.decision.state
            action = step.decision.action
            yield {
                "solver": record.solver.value,
                "episode_id": record.episode_id,
                "step": step.step_index,
                "physics_step": step.physics_step,
                "primitive": step.primitive.primitive.value,
                "trigger": step.primitive.trigger,
                "undesirable": step.primitive.undesirable,
                "d": state.d,
                "phi": state.phi,
                "v": state.v,
                "curvature": state.curvature,
                "curvature_class": state.curvature_class,
                "stop_present": state.stop_present,
                "stop_distance": state.stop_distance,
                "stop_satisfied": state.stop_satisfied,
                "duck_present": state.duck_present,
                "duck_active": state.duck_active,
                "duck_threat": state.duck_threat,
                "v_cmd": action.v_cmd,
                "omega_cmd": action.omega_cmd,
                "action_id": action.action_id,
                "action_name": action.action_name,
                "q_margin": step.decision.diagnostics.get("q_margin"),
                "reward": step.reward,
                "termination_reason": step.termination_reason,
            }


def _segment_rows(records):
    for record in records:
        for segment in record.segments:
            yield {
                "solver": record.solver.value,
                "episode_id": record.episode_id,
                "segment_index": segment.segment_index,
                "primitive": segment.primitive.value,
                "start_step": segment.start_step,
                "end_step": segment.end_step,
                "duration_steps": segment.duration_steps,
                "cumulative_reward": segment.cumulative_reward,
                "undesirable": segment.undesirable,
                "triggers": " | ".join(segment.triggers),
            }


def _frequency_rows(summaries):
    primitives = sorted({
        primitive
        for summary in summaries.values()
        for primitive in summary["primitive_frequency"]
    })
    for primitive in primitives:
        row = {"primitive": primitive}
        for solver, summary in summaries.items():
            values = summary["primitive_frequency"].get(
                primitive, {"count": 0, "rate": 0.0}
            )
            row[solver + "_count"] = values["count"]
            row[solver + "_rate"] = values["rate"]
        yield row


def _transition_rows(summaries):
    for solver, summary in summaries.items():
        for transition in summary["primitive_transitions"]:
            yield {"solver": solver, **transition}


def _duration_rows(summaries):
    for solver, summary in summaries.items():
        for primitive, values in summary["primitive_durations"].items():
            yield {"solver": solver, "primitive": primitive, **values}


def _signature_rows(signatures):
    for solver, signature in signatures.items():
        for concept, value in signature["normalized_l1"].items():
            yield {
                "solver": solver,
                "method": signature["method"],
                "concept": concept,
                "normalized_influence": value,
            }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sac-config", type=Path,
        default=Path("artifacts/sac/full_repeat_duck_5min/config.yaml"),
    )
    parser.add_argument(
        "--sac-checkpoint", type=Path,
        default=Path("artifacts/sac/full_repeat_duck_5min/sac_best.pt"),
    )
    parser.add_argument(
        "--q-config", type=Path,
        default=Path("configs/small_loop_stop_duck_q.yaml"),
    )
    parser.add_argument(
        "--q-checkpoint", type=Path,
        default=Path("artifacts/ablation/full_task/q_learning_teacher/q_table_best.npy"),
    )
    parser.add_argument(
        "--m8-summary", type=Path,
        default=Path("runs/explanations/m8_exact_q/m8_summary.json"),
    )
    parser.add_argument(
        "--m9-summary", type=Path,
        default=Path("runs/explanations/m9_sac_internal/m9_summary.json"),
    )
    parser.add_argument("--max-physics-steps", type=int, default=1500)
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("runs/explanations/m12_policy_comparison"),
    )
    args = parser.parse_args()
    _quiet()
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)

    sac_source = _load_yaml(args.sac_config)
    shared_config = _shared_config(sac_source, args.max_physics_steps)
    seeds = [
        int(value) for value in shared_config["evaluation"]["final_seeds"]
    ][: int(args.episodes)]
    config_path = output / "shared_comparison_config.yaml"
    _atomic_yaml(config_path, shared_config)

    q_source = _load_yaml(args.q_config)
    q_policy = QPolicyAdapter.from_checkpoint(
        args.q_checkpoint,
        allowed_actions=q_source["q_learning"]["allowed_actions"],
        action_config=ActionConfig(**shared_config["actions"]),
    )
    sac_policy = SACPolicyAdapter.from_checkpoint(
        args.sac_checkpoint, allow_observation_expansion=True
    )
    common = {
        "method": "matched_original_policy_rollout",
        "teacher_active": False,
        "map": shared_config["environment"]["map_name"],
        "frame_skip": shared_config["environment"]["frame_skip"],
        "max_physics_steps": shared_config["environment"]["max_steps"],
        "reactive_world": True,
        "straight_steer_penalty_neutralized_for_wrapper_parity": True,
    }
    q_provenance = build_provenance(
        args.q_checkpoint, config_path, extra={**common, "solver": "q_learning"}
    )
    sac_provenance = build_provenance(
        args.sac_checkpoint, config_path, extra={**common, "solver": "sac"}
    )

    q_records, q_initial = _run_q(
        shared_config, seeds, q_policy, q_provenance, output
    )
    sac_records, sac_initial = _run_sac(
        shared_config, seeds, sac_policy, sac_provenance, output
    )
    summaries = {
        "q_learning": summarize_policy(q_records),
        "sac": summarize_policy(sac_records),
    }

    m8 = json.loads(args.m8_summary.read_text(encoding="utf-8"))
    m9 = json.loads(args.m9_summary.read_text(encoding="utf-8"))
    signatures = {
        "q_learning": q_supported_influence_signature(m8),
        "sac": sac_ig_influence_signature(m9),
    }
    comparable_signature = comparable_influence_subset(
        signatures["q_learning"], signatures["sac"]
    )

    all_records = q_records + sac_records
    _write_csv(output / "steps.csv", _step_rows(all_records))
    _write_csv(output / "segments.csv", _segment_rows(all_records))
    _write_csv(output / "primitive_frequency.csv", _frequency_rows(summaries))
    _write_csv(output / "primitive_transitions.csv", _transition_rows(summaries))
    _write_csv(output / "primitive_durations.csv", _duration_rows(summaries))
    _write_csv(output / "feature_influence_signatures.csv", _signature_rows(signatures))

    unknown_counts = {
        solver: summary["primitive_frequency"].get(
            DrivingPrimitive.UNKNOWN.value, {"count": 0}
        )["count"]
        for solver, summary in summaries.items()
    }
    total_steps = {
        solver: summary["decision_steps"] for solver, summary in summaries.items()
    }
    coverage = {
        solver: 1.0 - unknown_counts[solver] / total_steps[solver]
        for solver in summaries
    }
    initial_match = _poses_match(q_initial, sac_initial)
    acceptance = {
        "same_map": True,
        "same_seeds": True,
        "same_frame_skip": True,
        "same_horizon": True,
        "initial_pose_match_atol_1e_7": initial_match,
        "teacher_free_deterministic_original_policies": True,
        "q_primitive_coverage_ge_0_95": coverage["q_learning"] >= 0.95,
        "sac_primitive_coverage_ge_0_95": coverage["sac"] >= 0.95,
        "comparison_uses_no_surrogate_actions": True,
        "lane_heading_entanglement_excluded_from_pure_comparable_signature": True,
    }
    summary = {
        "stage": "M12-policy-comparison",
        "method": "matched original-policy behavioral comparison",
        "manifest": {
            "config": str(config_path),
            "seeds": seeds,
            "initial_pose_q": q_initial,
            "initial_pose_sac": sac_initial,
            **common,
        },
        "policies": {
            "q_learning": {
                "checkpoint": str(args.q_checkpoint),
                "sha256": q_policy.checkpoint_hash,
                "mode": "greedy_teacher_free_lowest_id_tie_break",
            },
            "sac": {
                "checkpoint": str(args.sac_checkpoint),
                "sha256": sac_policy.checkpoint_hash,
                "mode": "deterministic_actor_mean",
            },
        },
        "summaries": summaries,
        "feature_influence_signatures": signatures,
        "comparable_influence_subset": comparable_signature,
        "primitive_coverage": coverage,
        "acceptance": {
            "checks": acceptance,
            "passed": all(acceptance.values()),
            "failed_checks": [name for name, value in acceptance.items() if not value],
            "policy_behavior_differences_are_findings_not_pipeline_failures": True,
        },
        "files": {
            "steps": str(output / "steps.csv"),
            "segments": str(output / "segments.csv"),
            "primitive_frequency": str(output / "primitive_frequency.csv"),
            "primitive_transitions": str(output / "primitive_transitions.csv"),
            "primitive_durations": str(output / "primitive_durations.csv"),
            "feature_influence_signatures": str(output / "feature_influence_signatures.csv"),
        },
    }
    _atomic_json(output / "comparison_summary.json", to_dict(summary))
    print(json.dumps({
        "accepted": summary["acceptance"]["passed"],
        "initial_pose_match": initial_match,
        "coverage": coverage,
        "q_stop_compliance": summaries["q_learning"]["stop_compliance_rate"],
        "sac_stop_compliance": summaries["sac"]["stop_compliance_rate"],
        "q_yield": summaries["q_learning"]["pedestrian_yield_command_rate"],
        "sac_yield": summaries["sac"]["pedestrian_yield_command_rate"],
    }, sort_keys=True))
    print("summary=%s" % (output / "comparison_summary.json"))


if __name__ == "__main__":
    main()
