import json

from src.explainability.explanation_report import (
    _action_label,
    _final_discounted_return,
    _local_case,
    explanation_index_rows,
)


def test_action_label_supports_discrete_and_continuous_actions():
    assert _action_label({
        "action_name": "brake", "v_cmd": 0.0, "omega_cmd": 0.0
    }) == "brake"
    label = _action_label({
        "action_name": None, "v_cmd": 0.17, "omega_cmd": -0.25
    })
    assert label == "(v=+0.1700, omega=-0.2500)"


def test_final_discounted_return_uses_longest_profile():
    assert _final_discounted_return({"reward_profile": []}) is None
    assert _final_discounted_return({"reward_profile": [
        {"horizon_steps": 1, "discounted_total": 1.0},
        {"horizon_steps": 5, "discounted_total": 4.0},
    ]}) == 4.0


def test_local_case_treats_teacher_inactive_as_required_invariant():
    action = {
        "action_id": 6, "action_name": "brake", "solver": "q_learning",
        "v_cmd": 0.0, "omega_cmd": 0.0,
    }
    branch = {
        "first_primitive": "StopHold",
        "physical": {},
        "reward_profile": [{"discounted_total": 1.0}],
    }
    payload = {
        "selected_decision": {
            "solver": "q_learning", "policy_mode": "greedy", "state": {},
            "action": action, "diagnostics": {},
            "metadata": {
                "checkpoint_path": "q.npy",
                "checkpoint_hash_sha256": "abc123",
            },
        },
        "factual": branch,
        "counterfactual": dict(branch),
        "foil_action": dict(action),
        "branch_invariants": {
            "only_first_action_forced": True,
            "same_manifest": True,
            "same_policy_selected_action_at_branch": True,
            "selected_and_foil_differ": True,
            "teacher_active": False,
        },
        "world_mode": "reactive",
        "manifest_id": "test",
        "physical_delta_counterfactual_minus_factual": {},
        "reward_delta_counterfactual_minus_factual": {},
        "explanation": "test",
    }
    case = _local_case("q", "stop", payload, "case.json")
    assert case["branch_valid"] is True
    assert case["checkpoint_path"] == "q.npy"
    assert case["checkpoint_sha256"] == "abc123"


def test_explanation_index_preserves_solver_and_contrast_fields():
    report = {"local_explanations": [{
        "case_id": "q_stop",
        "solver": "q_learning",
        "policy_mode": "greedy",
        "scenario": "stop",
        "selected_action_label": "brake",
        "selected_primitive": "StopHold",
        "foil_action_label": "slow_straight",
        "foil_primitive": "DecelerateStop",
        "q_margin": 1.5,
        "selected_discounted_return": 2.0,
        "foil_discounted_return": 1.0,
        "world_mode": "reactive",
        "branch_valid": True,
        "source_file": "case.json",
    }]}
    rows = explanation_index_rows(report)
    assert len(rows) == 1
    assert rows[0]["selected_primitive"] == "StopHold"
    assert rows[0]["foil_primitive"] == "DecelerateStop"
    assert rows[0]["q_margin"] == 1.5
