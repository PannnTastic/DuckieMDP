"""Build the auditable M13 SARSA explanation extension report."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from ..discretizer import Q_SHAPE
from .explanation_report import _local_case, explanation_index_rows


REPORT_SCHEMA_VERSION = "1.0.0"
SARSA_CHECKPOINT = (
    "artifacts/ablation/full_task/sarsa_teacher/q_table_best.npy"
)
SARSA_SHA256 = (
    "0266ad6f6fdae71bf2dfb7c7121f66e038d16f75e214a931f8c9d50bc6ad3313"
)
Q_CHECKPOINT = (
    "artifacts/ablation/full_task/q_learning_teacher/q_table_best.npy"
)
SCENARIOS = ("lane_correction", "stop_hold", "pedestrian_yield")
SUMMARY_FILES = {
    "local": "runs/explanations/m13_sarsa/m13_local_summary.json",
    "analysis": "runs/explanations/m13_sarsa/m13_analysis_summary.json",
    "comparison": "runs/explanations/m13_sarsa/m13_comparison_summary.json",
}


def _load_json(path: Path):
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def _sha256(path: Path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def compare_tabular_policies(q_table, sarsa_table):
    """Compare values and lowest-id greedy actions over the finite domain."""
    q_table = np.asarray(q_table)
    sarsa_table = np.asarray(sarsa_table)
    if q_table.shape != Q_SHAPE or sarsa_table.shape != Q_SHAPE:
        raise ValueError(
            "both tabular checkpoints must have shape %r" % (Q_SHAPE,)
        )
    q_actions = np.argmax(q_table, axis=-1)
    sarsa_actions = np.argmax(sarsa_table, axis=-1)
    agreement = q_actions == sarsa_actions
    return {
        "tables_numerically_equal": bool(np.array_equal(q_table, sarsa_table)),
        "maximum_absolute_q_value_difference": float(
            np.max(np.abs(q_table - sarsa_table))
        ),
        "representable_states": int(agreement.size),
        "greedy_action_agreement_states": int(np.sum(agreement)),
        "greedy_action_disagreement_states": int(np.sum(~agreement)),
        "greedy_action_agreement_rate": float(np.mean(agreement)),
        "tie_break": "lowest_action_id_via_numpy_argmax",
    }


def build_sarsa_extension_report(repo_root):
    """Assemble SARSA evidence and reject mixed checkpoints or policy modes."""
    root = Path(repo_root).resolve()
    summaries = {
        name: _load_json(root / relative)
        for name, relative in SUMMARY_FILES.items()
    }
    source_paths = [root / relative for relative in SUMMARY_FILES.values()]

    local_cases = []
    for scenario in SCENARIOS:
        relative = (
            "runs/explanations/m13_sarsa/local/"
            f"sarsa_{scenario}.json"
        )
        path = root / relative
        local_cases.append(
            _local_case("sarsa", scenario, _load_json(path), relative)
        )
        source_paths.append(path)

    local = summaries["local"]
    analysis = summaries["analysis"]
    comparison = summaries["comparison"]
    comparison_policy = comparison["policies"]["sarsa"]

    sarsa_path = root / SARSA_CHECKPOINT
    q_path = root / Q_CHECKPOINT
    table_comparison = compare_tabular_policies(
        np.load(str(q_path), allow_pickle=False),
        np.load(str(sarsa_path), allow_pickle=False),
    )
    source_paths.extend((q_path, sarsa_path))

    checks = {
        "three_local_scenarios_loaded": len(local_cases) == 3,
        "all_local_branches_valid": all(
            case["branch_valid"] for case in local_cases
        ),
        "all_local_cases_identify_sarsa": all(
            case["solver"] == "sarsa" for case in local_cases
        ),
        "all_local_cases_use_canonical_checkpoint": all(
            case["checkpoint_path"] == SARSA_CHECKPOINT
            and case["checkpoint_sha256"] == SARSA_SHA256
            for case in local_cases
        ),
        "local_stage_accepted": bool(local["acceptance"]["passed"]),
        "analysis_stage_accepted": bool(analysis["acceptance"]["passed"]),
        "comparison_stage_accepted": bool(
            comparison["acceptance"]["passed"]
        ),
        "local_summary_uses_canonical_checkpoint": (
            local["checkpoint"]["path"] == SARSA_CHECKPOINT
            and local["checkpoint"]["sha256"] == SARSA_SHA256
        ),
        "analysis_uses_canonical_checkpoint": (
            analysis["checkpoint"]["path"] == SARSA_CHECKPOINT
            and analysis["checkpoint"]["sha256"] == SARSA_SHA256
        ),
        "comparison_uses_canonical_checkpoint": (
            comparison_policy["checkpoint"] == SARSA_CHECKPOINT
            and comparison_policy["sha256"] == SARSA_SHA256
        ),
        "checkpoint_file_hash_matches_manifest": (
            _sha256(sarsa_path) == SARSA_SHA256
        ),
        "q_shape_unchanged": tuple(analysis["checkpoint"]["shape"])
        == Q_SHAPE,
        "all_9000_states_enumerated": (
            analysis["exact_verification"]["acceptance"][
                "enumerated_states"
            ]
            == analysis["exact_verification"]["acceptance"][
                "expected_states"
            ]
            == 9000
        ),
        "teacher_inactive_during_explanation": (
            local["policy_contract"]["teacher_active"] is False
            and comparison["manifest"]["teacher_active"] is False
        ),
        "greedy_original_policy_explained": (
            comparison_policy["mode"]
            == "greedy_teacher_free_lowest_id_tie_break"
            and analysis["policy_contract"]["explanation_mode"]
            == "greedy_teacher_free_lowest_id_tie_break"
        ),
        "surrogate_not_used_as_policy": bool(
            analysis["acceptance"]["checks"][
                "rule_surrogate_not_original_policy"
            ]
        ),
        "metamorphic_pairs_manifold_valid": bool(
            analysis["metamorphic"]["all_pairs_manifold_valid"]
        ),
        "matched_rollout_action_trace_checked": bool(
            comparison["acceptance"]["checks"][
                "q_sarsa_action_traces_compared"
            ]
        ),
    }
    failed = sorted(name for name, passed in checks.items() if not passed)

    report = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "stage": "M13-SARSA-explanation-extension",
        "framework": (
            "primitive-grounded decision explanation + paired action-outcome "
            "counterfactual + metamorphic/safety verification"
        ),
        "scope": "frozen_tabular_sarsa_policy_in_fully_observable_mdp",
        "policy": {
            "solver": "sarsa",
            "checkpoint": SARSA_CHECKPOINT,
            "sha256": SARSA_SHA256,
            "training": "teacher_guided_sarsa",
            "explanation_and_evaluation": (
                "greedy_teacher_free_lowest_id_tie_break"
            ),
        },
        "local_explanations": local_cases,
        "global_analysis": analysis,
        "three_policy_behavioral_comparison": comparison,
        "q_learning_sarsa_table_comparison": table_comparison,
        "acceptance": {
            "passed": not failed,
            "checks": checks,
            "failed_checks": failed,
            "binding_validation": True,
        },
        "artefact_manifest": {
            str(path.relative_to(root)): {
                "sha256": _sha256(path),
                "bytes": path.stat().st_size,
            }
            for path in sorted(set(source_paths))
        },
        "epistemic_limits": [
            "SARSA used teacher guidance during training, but the explained policy is teacher-free greedy lookup.",
            "Q-margin is action-value separation, not a probability or calibrated confidence.",
            "Violations in unsupported table cells are not described as learned behavior.",
            "Metamorphic failures are policy findings, not explanation-pipeline failures.",
            "The rule tree is a post-hoc surrogate and never replaces the original SARSA table.",
            "The five-seed comparison validates integration and is not population-level inference.",
        ],
    }
    return report


def sarsa_explanation_index_rows(report):
    return explanation_index_rows({
        "local_explanations": report["local_explanations"]
    })

