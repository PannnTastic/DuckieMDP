import copy

import numpy as np

from src.explainability.eddp.anchors import physical_context
from src.explainability.eddp.clustering import discover
from src.explainability.eddp.schema import ExplanationAtom
from src.explainability.eddp.signature import (
    ATOM_FEATURE_NAMES, assert_label_free_feature_contract,
    atom_feature_vector, build_segment_dataset,
)
from src.explainability.schema import SolverKind


def _atom(index=0, solver=SolverKind.Q_LEARNING, block="b0"):
    cf = {"attempts": 4, "valid_attempts": 4, "any_flip": True,
          "minimum_flip_distance": 0.2}
    for concept in ("lateral", "heading", "speed", "curvature",
                    "stop_distance", "stop_satisfied", "duck_risk"):
        cf[concept + "_flip"] = concept == "lateral"
        cf[concept + "_abs_delta"] = 0.2 if concept == "lateral" else 1.0
        cf[concept + "_signed_delta"] = 0.2 if concept == "lateral" else 0.0
    physical = {name.split("physical__", 1)[-1]: 0.0
                for name in ATOM_FEATURE_NAMES if name.startswith("physical__")}
    verification = {name.split("verification__", 1)[-1]: 0.0
                    for name in ATOM_FEATURE_NAMES if name.startswith("verification__")}
    validity = {
        "counterfactual_valid_fraction": 1.0,
        "branch_invariants_pass": True,
        "paired_outcome_valid": True,
    }
    return ExplanationAtom(
        atom_id="a%d" % index, anchor_id="x%d" % index,
        solver=solver, seed=1, episode_id="e", decision_step=index,
        block_id=block, block_offset=index, selection_context="lane",
        observed_context="lane", counterfactual_profile=cf,
        physical_profile=physical, reward_profile={"secret": 999.0},
        verification_profile=verification, validity=validity,
        paired_report_path="not-read-by-signature.json",
    )


def test_eddp_feature_contract_excludes_solver_action_and_primitive():
    assert_label_free_feature_contract(ATOM_FEATURE_NAMES)
    for forbidden in ("solver", "action_name", "primitive", "q_margin", "critic"):
        try:
            assert_label_free_feature_contract((forbidden,))
        except ValueError:
            pass
        else:
            raise AssertionError("leakage guard accepted %s" % forbidden)


def test_atom_signature_ignores_solver_metadata_and_reward_text():
    left = _atom()
    right_payload = left.as_dict()
    right_payload["solver"] = "sac"
    right_payload["reward_profile"] = {"primitive": 12345.0}
    right_payload["paired_report_path"] = "contains-other-labels.json"
    right = ExplanationAtom.from_dict(right_payload)
    assert np.array_equal(atom_feature_vector(left), atom_feature_vector(right))


def test_temporal_segment_uses_fixed_block_not_primitive_boundary():
    atoms = (_atom(0), _atom(1), _atom(2))
    dataset = build_segment_dataset(atoms)
    assert dataset.features.shape == (1, len(dataset.feature_names))
    assert dataset.atom_ids == (("a0", "a1", "a2"),)
    assert dataset.metadata[0]["length"] == 3



def test_temporal_segment_records_quarantined_middle_atom_gap():
    dataset = build_segment_dataset((_atom(0), _atom(2)))
    assert dataset.features.shape[0] == 1
    assert dataset.metadata[0]["length"] == 2
    assert dataset.metadata[0]["expected_length"] == 3
    assert dataset.metadata[0]["missing_offsets"] == "1"
    assert dataset.metadata[0]["complete_window"] is False

def test_discovery_has_disjoint_seed_split_and_inductive_assignment():
    rng = np.random.RandomState(3)
    features, metadata = [], []
    for solver in ("q_learning", "sarsa", "sac"):
        for seed in (1, 2, 3, 4, 5):
            for center in (-3.0, 3.0):
                for _ in range(3):
                    features.append(rng.normal(center, 0.1, size=4))
                    metadata.append({"solver": solver, "seed": seed})
    result, _ = discover(np.asarray(features), metadata)
    assert set(result.split) == {"development", "heldout"}
    assert result.diagnostics["development"]["clusters"] >= 2
    assert result.diagnostics["deterministic_rerun"]
