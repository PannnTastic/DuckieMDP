"""Run the frozen M7 metamorphic suite on Q-learning and SAC checkpoints."""

import argparse
from collections import Counter, defaultdict
import json
import logging
from pathlib import Path
import warnings

import yaml

from src.actions import ActionConfig
from src.explainability.metamorphic import evaluate_relation, save_results
from src.explainability.q_policy_adapter import QPolicyAdapter
from src.explainability.sac_policy_adapter import SACPolicyAdapter
from src.explainability.schema import CanonicalState, SolverKind, to_dict


def _quiet():
    warnings.filterwarnings("ignore")
    logging.disable(logging.CRITICAL)


def _load_anchor(path: Path):
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    record = payload["anchors"]["lane"]
    return CanonicalState(**record["state"]), {
        "anchor_source": str(path),
        "anchor_scenario": "lane",
        "anchor_seed": record["seed"],
        "anchor_decision_step": record["decision_step"],
    }


def _base(**changes):
    values = {
        "d": 0.0,
        "phi": 0.0,
        "curvature": 0.0,
    }
    values.update(changes)
    return values


def _cases():
    cases = []

    for source_distance, targets in (
        (1.50, (0.75, 0.30, 0.15)),
        (0.75, (0.30, 0.15)),
        (0.30, (0.15,)),
    ):
        source = _base(stop_distance=source_distance)
        for target_distance in targets:
            cases.append((
                "MR-STOP",
                "stop_%.2f_to_%.2f" % (source_distance, target_distance),
                source,
                {"stop_distance": target_distance},
            ))

    pedestrian_states = (
        ("side_far", {
            "duck_longitudinal": 1.0, "duck_lateral": 0.5,
            "duck_v_longitudinal_relative": 0.0,
            "duck_v_lateral_relative": 0.0,
            "duck_active": False, "duck_crossing_available": True,
        }),
        ("side_near", {
            "duck_longitudinal": 0.4, "duck_lateral": 0.2,
            "duck_v_longitudinal_relative": 0.0,
            "duck_v_lateral_relative": 0.0,
            "duck_active": False, "duck_crossing_available": True,
        }),
        ("crossing_far", {
            "duck_longitudinal": 0.9, "duck_lateral": 0.1,
            "duck_v_longitudinal_relative": 0.0,
            "duck_v_lateral_relative": 0.0,
            "duck_active": True, "duck_crossing_available": False,
        }),
        ("crossing_near", {
            "duck_longitudinal": 0.2, "duck_lateral": 0.1,
            "duck_v_longitudinal_relative": 0.0,
            "duck_v_lateral_relative": 0.0,
            "duck_active": True, "duck_crossing_available": False,
        }),
    )
    # Pairs follow the frozen risk ordering.  Every target is strictly riskier
    # than its source under the canonical relation metric.
    for source_index, target_index in ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)):
        source_name, source_duck = pedestrian_states[source_index]
        target_name, target_duck = pedestrian_states[target_index]
        source = _base(**source_duck)
        cases.append((
            "MR-PEDESTRIAN",
            "pedestrian_%s_to_%s" % (source_name, target_name),
            source,
            target_duck,
        ))

    for curvature in (-4.0, -2.0, -1.0, 1.0, 2.0, 4.0):
        cases.append((
            "MR-CURVATURE",
            "curvature_0_to_%+.1f" % curvature,
            _base(curvature=0.0),
            {"curvature": curvature},
        ))

    for d_value, phi_value in (
        (0.05, 0.10), (0.10, 0.20), (0.15, 0.30),
        (-0.05, 0.10), (-0.10, 0.20), (-0.15, 0.30),
    ):
        cases.append((
            "MR-LANE-SYMMETRY",
            "lane_mirror_d_%+.2f_phi_%+.2f" % (d_value, phi_value),
            _base(d=d_value, phi=phi_value),
            {"d": -d_value, "phi": -phi_value},
        ))
    return tuple(cases)


def _summarize(results):
    by_solver = defaultdict(Counter)
    by_relation = defaultdict(Counter)
    for result in results:
        by_solver[result.solver.value][result.status.value] += 1
        by_relation[(result.solver.value, result.relation.relation_id)][result.status.value] += 1
    return {
        "total": len(results),
        "by_solver": {solver: dict(counts) for solver, counts in sorted(by_solver.items())},
        "by_solver_relation": {
            "%s/%s" % key: dict(counts)
            for key, counts in sorted(by_relation.items())
        },
        "all_pairs_manifold_valid": all(
            result.source.validation.valid and result.target.validation.valid
            for result in results
        ),
        "not_applicable": sum(result.status.value == "NOT_APPLICABLE" for result in results),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--m6-summary", type=Path,
        default=Path("runs/explanations/m6_response_curves/m6_summary.json"),
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
        "--output-dir", type=Path,
        default=Path("runs/explanations/m7_metamorphic"),
    )
    args = parser.parse_args()
    _quiet()

    anchor, anchor_provenance = _load_anchor(args.m6_summary)
    q_cfg = yaml.safe_load(args.q_config.read_text(encoding="utf-8"))
    policies = (
        (SolverKind.Q_LEARNING, QPolicyAdapter.from_checkpoint(
            args.q_checkpoint,
            allowed_actions=q_cfg["q_learning"]["allowed_actions"],
            action_config=ActionConfig(**q_cfg["actions"]),
        )),
        (SolverKind.SAC, SACPolicyAdapter.from_checkpoint(
            args.sac_checkpoint, allow_observation_expansion=True,
        )),
    )

    results = []
    for solver, policy in policies:
        support = (
            "unknown_no_visit_count_artifact"
            if solver == SolverKind.Q_LEARNING
            else "real_anchor_synthetic_pairs_not_reachability_proven"
        )
        for relation_id, case_id, source_changes, target_changes in _cases():
            provenance = dict(anchor_provenance)
            provenance.update(
                case_id=case_id,
                support_status=support,
                policy_mode=(
                    "greedy_teacher_free"
                    if solver == SolverKind.Q_LEARNING
                    else "deterministic_actor_mean"
                ),
            )
            results.append(evaluate_relation(
                policy, solver, anchor, relation_id,
                source_changes, target_changes, provenance,
            ))

    output = args.output_dir
    result_json, result_csv = save_results(results, output)
    summary = {
        "stage": "M7",
        "method": "LEGIBLE-inspired metamorphic policy testing",
        "relations": sorted({result.relation.relation_id for result in results}),
        "policy_checkpoints": {
            solver.value: {
                "path": policy.checkpoint_path,
                "sha256": policy.checkpoint_hash,
            }
            for solver, policy in policies
        },
        "anchor": {"state": to_dict(anchor), **anchor_provenance},
        "counts": _summarize(results),
        "acceptance": {
            "all_pairs_manifold_valid": all(
                result.source.validation.valid and result.target.validation.valid
                for result in results
            ),
            "all_preconditions_applicable": all(
                result.status.value != "NOT_APPLICABLE" for result in results
            ),
            "policy_failures_are_findings_not_pipeline_failures": True,
        },
        "files": [str(result_json), str(result_csv)],
    }
    output.mkdir(parents=True, exist_ok=True)
    summary_path = output / "m7_summary.json"
    temporary = summary_path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(summary_path)
    print(json.dumps(summary["counts"], sort_keys=True))
    print("summary=%s" % summary_path)


if __name__ == "__main__":
    main()
