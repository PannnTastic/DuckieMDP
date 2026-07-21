"""Build the versioned, machine-readable M12 explanation report."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


REPORT_SCHEMA_VERSION = "1.0.0"
SCENARIOS = ("lane_correction", "stop_hold", "pedestrian_yield")
SOLVERS = ("q", "sac")
STAGE_FILES = {
    "response_curves": "runs/explanations/m6_response_curves/m6_summary.json",
    "metamorphic_verification": "runs/explanations/m7_metamorphic/m7_summary.json",
    "exact_q_verification": "runs/explanations/m8_exact_q/m8_summary.json",
    "sac_internal_diagnostics": "runs/explanations/m9_sac_internal/m9_summary.json",
    "rule_extraction": "runs/explanations/m10_rule_extraction/m10_summary.json",
    "bottom_up_clustering": "runs/explanations/m11_bottom_up_clustering/m11_summary.json",
    "policy_comparison": "runs/explanations/m12_policy_comparison/comparison_summary.json",
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


def _action_label(action):
    if action.get("action_name"):
        return action["action_name"]
    return "(v={:+.4f}, omega={:+.4f})".format(
        float(action["v_cmd"]), float(action["omega_cmd"])
    )


def _final_discounted_return(branch):
    profiles = branch.get("reward_profile", [])
    return None if not profiles else profiles[-1]["discounted_total"]


def _local_case(solver_key, scenario, payload, relative_path):
    selected = payload["selected_decision"]
    factual = payload["factual"]
    counterfactual = payload["counterfactual"]
    diagnostics = selected.get("diagnostics", {})
    metadata = selected.get("metadata", {})
    invariants = payload["branch_invariants"]
    return {
        "case_id": f"{solver_key}_{scenario}",
        "solver": selected["solver"],
        "policy_mode": selected["policy_mode"],
        "scenario": scenario,
        "source_file": relative_path,
        "world_mode": payload["world_mode"],
        "manifest_id": payload["manifest_id"],
        "checkpoint_path": metadata.get("checkpoint_path"),
        "checkpoint_sha256": metadata.get("checkpoint_hash_sha256"),
        "state": selected["state"],
        "selected_action": selected["action"],
        "selected_action_label": _action_label(selected["action"]),
        "selected_primitive": factual["first_primitive"],
        "foil_action": payload["foil_action"],
        "foil_action_label": _action_label(payload["foil_action"]),
        "foil_primitive": counterfactual["first_primitive"],
        "q_margin": diagnostics.get("q_margin"),
        "branch_invariants": invariants,
        "branch_valid": (
            all(bool(invariants[name]) for name in (
                "only_first_action_forced", "same_manifest",
                "same_policy_selected_action_at_branch", "selected_and_foil_differ",
            ))
            and invariants.get("teacher_active") is False
        ),
        "selected_discounted_return": _final_discounted_return(factual),
        "foil_discounted_return": _final_discounted_return(counterfactual),
        "physical_delta_counterfactual_minus_factual": (
            payload["physical_delta_counterfactual_minus_factual"]
        ),
        "reward_delta_counterfactual_minus_factual": (
            payload["reward_delta_counterfactual_minus_factual"]
        ),
        "factual_physical_outcome": factual["physical"],
        "counterfactual_physical_outcome": counterfactual["physical"],
        "explanation_text": payload["explanation"],
        "claim_scope": "simulator_based_interventional_counterfactual",
    }


def build_unified_report(repo_root):
    """Load accepted stage artefacts and assemble one auditable report."""
    root = Path(repo_root).resolve()
    stages = {}
    artefact_paths = []
    for name, relative in STAGE_FILES.items():
        path = root / relative
        stages[name] = _load_json(path)
        artefact_paths.append(path)

    local_cases = []
    for solver_key in SOLVERS:
        for scenario in SCENARIOS:
            relative = f"runs/explanations/{solver_key}_{scenario}.json"
            path = root / relative
            local_cases.append(
                _local_case(solver_key, scenario, _load_json(path), relative)
            )
            artefact_paths.append(path)

    m7 = stages["metamorphic_verification"]
    m6 = stages["response_curves"]
    m8 = stages["exact_q_verification"]
    m9 = stages["sac_internal_diagnostics"]
    m10 = stages["rule_extraction"]
    m11 = stages["bottom_up_clustering"]
    m12 = stages["policy_comparison"]
    expected_checkpoint_hashes = {
        "q_learning": m12["policies"]["q_learning"]["sha256"],
        "sac": m12["policies"]["sac"]["sha256"],
    }
    expected_checkpoint_paths = {
        "q_learning": m12["policies"]["q_learning"]["checkpoint"],
        "sac": m12["policies"]["sac"]["checkpoint"],
    }
    stage_checkpoint_checks = {
        "m6_uses_canonical_policy_checkpoints": all(
            m6["policy_checkpoints"][solver]["path"]
            == expected_checkpoint_paths[solver]
            and m6["policy_checkpoints"][solver]["sha256"]
            == expected_checkpoint_hashes[solver]
            for solver in ("q_learning", "sac")
        ),
        "m7_uses_canonical_policy_checkpoints": all(
            m7["policy_checkpoints"][solver]["path"]
            == expected_checkpoint_paths[solver]
            and m7["policy_checkpoints"][solver]["sha256"]
            == expected_checkpoint_hashes[solver]
            for solver in ("q_learning", "sac")
        ),
        "m8_uses_canonical_q_checkpoint": (
            m8["checkpoint"]["path"] == expected_checkpoint_paths["q_learning"]
            and m8["checkpoint"]["sha256"]
            == expected_checkpoint_hashes["q_learning"]
        ),
        "m9_uses_canonical_sac_checkpoint": (
            m9["checkpoint"]["path"] == expected_checkpoint_paths["sac"]
            and m9["checkpoint"]["sha256"] == expected_checkpoint_hashes["sac"]
        ),
        "m10_uses_canonical_policy_checkpoints": (
            m10["q_learning"]["checkpoint"]
            == expected_checkpoint_paths["q_learning"]
            and m10["q_learning"]["checkpoint_sha256"]
            == expected_checkpoint_hashes["q_learning"]
            and m10["sac"]["checkpoint"] == expected_checkpoint_paths["sac"]
            and m10["sac"]["checkpoint_sha256"]
            == expected_checkpoint_hashes["sac"]
        ),
    }
    checks = {
        "all_six_local_cases_loaded": len(local_cases) == 6,
        "all_local_branches_valid": all(case["branch_valid"] for case in local_cases),
        "all_local_cases_use_canonical_checkpoint_hash": all(
            case["checkpoint_sha256"] == expected_checkpoint_hashes[case["solver"]]
            for case in local_cases
        ),
        "all_local_cases_use_canonical_checkpoint_path": all(
            case["checkpoint_path"] == expected_checkpoint_paths[case["solver"]]
            for case in local_cases
        ),
        **stage_checkpoint_checks,
        "m7_all_pairs_manifold_valid": bool(
            m7["acceptance"]["all_pairs_manifold_valid"]
        ),
        "m8_all_9000_states_enumerated": (
            m8["acceptance"]["enumerated_states"]
            == m8["acceptance"]["expected_states"]
            == 9000
        ),
        "m8_q_shape_unchanged": bool(m8["acceptance"]["q_shape_unchanged"]),
        "m9_internal_diagnostics_accepted": bool(m9["acceptance"]["passed"]),
        "m10_global_rule_summary_eligible": bool(
            m10["acceptance"]["main_result_eligible"]
        ),
        "m10_sac_surrogate_not_used_as_policy": not bool(
            m10["acceptance"]["continuous_action_tree_policy_replacement_claim_allowed"]
        ),
        "m11_bottom_up_validation_accepted": bool(
            m11["acceptance"]["main_result_eligible"]
        ),
        "m12_matched_comparison_accepted": bool(m12["acceptance"]["passed"]),
        "m12_original_policies_only": bool(
            m12["acceptance"]["checks"]["comparison_uses_no_surrogate_actions"]
        ),
    }
    failed = sorted(name for name, passed in checks.items() if not passed)

    artefacts = {
        str(path.relative_to(root)): {
            "sha256": _sha256(path),
            "bytes": path.stat().st_size,
        }
        for path in sorted(artefact_paths)
    }
    report = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "stage": "M12-unified-explanation-report",
        "framework": (
            "primitive-grounded decision explanation + paired action-outcome "
            "counterfactual + metamorphic/safety verification"
        ),
        "scope": "fully_observable_mdp_policies",
        "policies": m12["policies"],
        "policy_modes": {
            "q_learning": "greedy_teacher_free_lowest_id_tie_break",
            "sac": "deterministic_actor_mean",
        },
        "local_explanations": local_cases,
        "global_decision_explanation": {
            "response_curves": stages["response_curves"],
            "exact_q_characterization": stages["exact_q_verification"],
            "sac_internal_diagnostics": stages["sac_internal_diagnostics"],
            "solver_aware_rules": stages["rule_extraction"],
        },
        "verification": {
            "metamorphic": stages["metamorphic_verification"],
            "exact_q_safety": m8["acceptance"][
                "safety_properties_checked_exhaustively"
            ],
        },
        "behavioral_comparison": stages["policy_comparison"],
        "m11_bottom_up_clustering": stages["bottom_up_clustering"],
        "acceptance": {
            "passed": not failed,
            "checks": checks,
            "failed_checks": failed,
            "binding_validation": True,
            "failed_outputs_retained_for_audit": True,
        },
        "artefact_manifest": artefacts,
        "epistemic_limits": [
            "COViz-inspired paired rollouts answer what-if outcomes, not the full reason alone.",
            "SAC Integrated Gradients is baseline-sensitive and complements behavioral evidence.",
            "Q one-bin flips and SAC IG are solver-specific signatures, not identical estimators.",
            "Reactive Duckie behavior is endogenous and remains part of each branch outcome.",
            "Surrogate trees summarize policies; the original policies produce local explanations.",
        ],
    }
    return report


def explanation_index_rows(report):
    """Flatten local cases for the human-readable M12 CSV index."""
    rows = []
    for case in report["local_explanations"]:
        rows.append({
            "case_id": case["case_id"],
            "solver": case["solver"],
            "policy_mode": case["policy_mode"],
            "scenario": case["scenario"],
            "selected_action": case["selected_action_label"],
            "selected_primitive": case["selected_primitive"],
            "foil_action": case["foil_action_label"],
            "foil_primitive": case["foil_primitive"],
            "q_margin": case["q_margin"],
            "selected_discounted_return": case["selected_discounted_return"],
            "foil_discounted_return": case["foil_discounted_return"],
            "world_mode": case["world_mode"],
            "branch_valid": case["branch_valid"],
            "source_file": case["source_file"],
        })
    return rows
