import copy

import numpy as np
import pytest

from src.explainability.cluster_primitives import (
    deterministic_hdbscan_check,
    fit_hdbscan,
    search_hdbscan,
    split_masks,
)
from src.explainability.reconcile_clusters import reconcile
from src.explainability.signatures import (
    FEATURE_NAMES,
    assert_label_free_feature_contract,
    build_signature_dataset,
    derive_evaluation_labels,
)


def _row(episode, solver, seed, step, primitive="CruiseStraight"):
    turning = step % 4 >= 2
    return {
        "solver": solver,
        "episode_id": "%s_%s" % (episode, seed),
        "step": str(step),
        "physics_step": str((step + 1) * 6),
        "primitive": primitive,
        "trigger": "must never become a feature",
        "undesirable": "False",
        "d": str(0.02 * (step % 3 - 1)),
        "phi": str(0.03 * (step % 2)),
        "v": "0.20",
        "curvature": "1.0" if turning else "0.0",
        "curvature_class": "curve_left" if turning else "straight",
        "stop_present": "False",
        "stop_distance": "",
        "stop_satisfied": "False",
        "duck_present": "False",
        "duck_active": "False",
        "duck_threat": "none",
        "v_cmd": "0.17" if turning else "0.41",
        "omega_cmd": "0.8" if turning else "0.0",
        "action_id": "0",
        "action_name": "ignored",
        "q_margin": "1.0",
        "reward": "999.0",
        "termination_reason": "in_progress",
    }


def test_fixed_window_signatures_are_label_independent():
    rows = [
        _row("q", "q_learning", 1, step, "CruiseStraight")
        for step in range(10)
    ]
    original = build_signature_dataset(rows, window_size=5)
    changed = copy.deepcopy(rows)
    for index, row in enumerate(changed):
        row["primitive"] = "YieldHold" if index % 2 else "StopHold"
        row["trigger"] = "changed label"
        row["undesirable"] = "True"
        row["reward"] = str(-1000 - index)
    relabeled = build_signature_dataset(changed, window_size=5)
    assert original.feature_names == FEATURE_NAMES
    assert_label_free_feature_contract(original.feature_names)
    np.testing.assert_array_equal(original.features, relabeled.features)
    assert len(original.features) == 2
    assert original.metadata[0]["start_step"] == 0
    assert original.metadata[1]["start_step"] == 5


def test_evaluation_labels_are_opened_separately_after_signatures():
    rows = [
        _row(
            "q", "q_learning", 1, step,
            "YieldHold" if step < 4 else "ResumeAfterYield",
        )
        for step in range(5)
    ]
    dataset = build_signature_dataset(rows, window_size=5)
    labels = derive_evaluation_labels(rows, dataset.metadata)
    assert labels[0]["primitive"] == "YieldHold"
    assert labels[0]["label_purity"] == 0.8
    assert labels[0]["primitive_counts"] == {
        "ResumeAfterYield": 1,
        "YieldHold": 4,
    }


def test_hdbscan_search_and_fit_are_deterministic_without_labels():
    rng = np.random.RandomState(0)
    matrix = np.vstack([
        rng.normal(loc=-3.0, scale=0.15, size=(30, 4)),
        rng.normal(loc=0.0, scale=0.15, size=(30, 4)),
        rng.normal(loc=3.0, scale=0.15, size=(30, 4)),
    ])
    selected, candidates = search_hdbscan(
        matrix,
        min_cluster_sizes=(8, 12),
        min_samples_values=(3, 5),
    )
    assert candidates
    assert all(
        item["primitive_labels_used_for_selection"] is False
        for item in candidates
    )
    _, labels, diagnostics = fit_hdbscan(matrix, selected["parameters"])
    assert diagnostics.clusters == 3
    assert diagnostics.coverage == 1.0
    assert deterministic_hdbscan_check(matrix, selected["parameters"])
    assert len(labels) == 90


def test_reconciliation_reports_purity_noise_and_solver_strata():
    clusters = np.asarray([0, 0, 1, 1, -1])
    primitives = ["A", "A", "B", "C", "A"]
    metadata = [
        {"solver": "q_learning", "episode_id": "q_1", "start_step": i, "end_step": i}
        for i in range(5)
    ]
    features = np.arange(10, dtype=float).reshape(5, 2)
    report = reconcile(clusters, primitives, metadata, features)
    assert report["metrics"]["all"]["coverage"] == 0.8
    assert report["metrics"]["all"]["noise_rate"] == pytest.approx(0.2)
    assert report["metrics"]["all"]["purity"] == 0.75
    assert report["metrics"]["solver_q_learning"] == report["metrics"]["all"]
    assert report["confusion_matrix"]["-1"] == {"A": 1}


def test_seed_split_is_disjoint_and_balanced_by_solver():
    metadata = []
    for solver in ("q_learning", "sac"):
        for seed in (1, 2, 3, 4, 5):
            metadata.append({"solver": solver, "seed": seed})
    development, heldout, manifest = split_masks(metadata, development_seed_count=3)
    assert development.sum() == 6
    assert heldout.sum() == 4
    assert not np.any(development & heldout)
    assert manifest["q_learning"]["development"] == [1, 2, 3]
    assert manifest["sac"]["heldout"] == [4, 5]
