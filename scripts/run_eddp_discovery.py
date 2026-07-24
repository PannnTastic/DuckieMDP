"""Run EDP5--EDP10 discovery, freeze, naming, and M2 reconciliation."""

import argparse
from collections import Counter
import csv
import json
from pathlib import Path

import joblib
import numpy as np
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from src.explainability.eddp.clustering import (
    discover, kmeans_sensitivity, split_by_seed,
)
from src.explainability.eddp.provenance import atomic_json, file_sha256
from src.explainability.eddp.reconcile import (
    build_cluster_cards, freeze_hash, majority_segment_labels, reconcile,
)
from src.explainability.eddp.schema import ExplanationAtom, read_jsonl
from src.explainability.eddp.signature import (
    assert_label_free_feature_contract, build_segment_dataset,
)


def _atomic_csv(path, rows, fieldnames=None):
    rows = list(rows)
    if not rows and fieldnames is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    names = fieldnames or list(rows[0])
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=names)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _labels_after_freeze(atoms):
    labels = {}
    rows = []
    for atom in atoms:
        payload = json.loads(Path(atom.paired_report_path).read_text(encoding="utf-8"))
        label = payload["factual"]["first_primitive"]
        labels[atom.atom_id] = label
        rows.append({"atom_id": atom.atom_id, "frozen_m2_primitive": label})
    return labels, rows


def _solver_predictability(features, metadata, split):
    split = np.asarray(split)
    train = split == "development"
    test = split == "heldout"
    y = np.asarray([row["solver"] for row in metadata])
    if len(set(y[train])) < 2 or not test.any():
        return None
    model = LogisticRegression(max_iter=3000, random_state=0).fit(features[train], y[train])
    return {
        "development_accuracy": float(accuracy_score(y[train], model.predict(features[train]))),
        "heldout_accuracy": float(accuracy_score(y[test], model.predict(features[test]))),
        "chance_reference": 1.0 / len(set(y)),
        "interpretation": "diagnostic only; solver is not a clustering feature",
    }


def _outcome_coherence(features, labels, seed=17):
    features = np.asarray(features, dtype=float)
    labels = np.asarray(labels, dtype=int)
    mask = labels >= 0
    if len(set(labels[mask])) < 2:
        return {"observed_within_mse": None, "permuted_mean_mse": None, "ratio": None}

    def within(values):
        total, count = 0.0, 0
        for cluster_id in sorted(set(values[mask])):
            points = features[mask][values[mask] == cluster_id]
            centroid = points.mean(axis=0)
            total += float(((points - centroid) ** 2).sum())
            count += int(points.size)
        return total / max(1, count)

    observed = within(labels)
    rng = np.random.RandomState(seed)
    permuted = []
    for _ in range(100):
        candidate = labels.copy()
        candidate[mask] = rng.permutation(candidate[mask])
        permuted.append(within(candidate))
    baseline = float(np.mean(permuted))
    return {
        "observed_within_mse": observed,
        "permuted_mean_mse": baseline,
        "ratio": None if baseline == 0.0 else observed / baseline,
        "better_than_permuted": observed < baseline,
    }


def _ablation(features, names, metadata, groups):
    results = {}
    for name, prefixes in groups.items():
        keep = np.asarray([
            not any(feature.startswith(prefix) for prefix in prefixes)
            for feature in names
        ])
        if int(keep.sum()) < 2:
            continue
        try:
            result, _ = discover(features[:, keep], metadata)
            results[name] = {
                "retained_features": int(keep.sum()),
                "selected_parameters": dict(result.selected_parameters),
                "diagnostics": dict(result.diagnostics),
            }
        except Exception as error:
            results[name] = {
                "retained_features": int(keep.sum()),
                "error": "%s: %s" % (type(error).__name__, error),
            }
    return results



def _reward_segment_features(atoms, atom_groups):
    """Aggregate reward deltas for the same frozen temporal blocks."""

    by_id = {atom.atom_id: atom for atom in atoms}
    reward_names = tuple(sorted({
        name for atom in atoms for name in atom.reward_profile
    }))
    names = tuple(
        "%s__reward__%s" % (statistic, name)
        for statistic in ("mean", "std", "delta")
        for name in reward_names
    )
    rows = []
    for group in atom_groups:
        matrix = np.asarray([
            [float(by_id[atom_id].reward_profile.get(name, 0.0)) for name in reward_names]
            for atom_id in group
        ], dtype=np.float64)
        rows.append(np.concatenate([
            matrix.mean(axis=0), matrix.std(axis=0), matrix[-1] - matrix[0]
        ]))
    return names, np.vstack(rows)


def _discover_variant(features, names, metadata, development_seed_count=3):
    """Apply the same development-only variance rule and discovery protocol."""

    split, _ = split_by_seed(metadata, development_seed_count)
    development = split == "development"
    keep = np.asarray(features)[development].var(axis=0) > 1.0e-12
    if int(keep.sum()) < 2:
        return {"status": "INSUFFICIENT_VARIANCE", "retained_features": int(keep.sum())}
    try:
        result, _ = discover(
            np.asarray(features)[:, keep], metadata, development_seed_count
        )
        return {
            "status": "COMPLETED",
            "retained_features": int(keep.sum()),
            "feature_names": [
                name for name, selected in zip(names, keep) if selected
            ],
            "selected_parameters": dict(result.selected_parameters),
            "diagnostics": dict(result.diagnostics),
        }
    except Exception as error:
        return {
            "status": "NOT_IDENTIFIABLE",
            "retained_features": int(keep.sum()),
            "error": "%s: %s" % (type(error).__name__, error),
        }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path,
                        default=Path("configs/explainability/eddp_v1.yaml"))
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    output = Path(config["experiment"]["output_dir"])
    atoms = tuple(
        ExplanationAtom.from_dict(row)
        for row in read_jsonl(output / "explanation_atoms_label_free.jsonl")
    )
    dataset = build_segment_dataset(
        atoms, config["discovery"]["minimum_atoms_per_segment"]
    )
    if len(dataset.metadata) < 12:
        raise RuntimeError("not enough temporal explanation segments")
    assert_label_free_feature_contract(dataset.feature_names)

    preliminary_split, _ = split_by_seed(
        dataset.metadata, config["collection"]["development_seed_count"]
    )
    dev = preliminary_split == "development"
    variances = dataset.features[dev].var(axis=0)
    keep = variances > float(config["discovery"]["minimum_development_variance"])
    selected_names = tuple(
        name for name, selected in zip(dataset.feature_names, keep) if selected
    )
    selected_features = dataset.features[:, keep]
    reward_names, reward_features = _reward_segment_features(
        atoms, dataset.atom_ids
    )
    assert_label_free_feature_contract(reward_names)
    physical_mask = np.asarray([
        "__physical__" in name for name in dataset.feature_names
    ])
    physical_features = dataset.features[:, physical_mask]
    physical_names = tuple(
        name for name, selected in zip(dataset.feature_names, physical_mask)
        if selected
    )
    extended_ablations = {
        "physical_only": _discover_variant(
            physical_features, physical_names, dataset.metadata,
            config["collection"]["development_seed_count"],
        ),
        "physical_plus_reward": _discover_variant(
            np.hstack([physical_features, reward_features]),
            physical_names + reward_names,
            dataset.metadata,
            config["collection"]["development_seed_count"],
        ),
        "complete_fixed_window_only": None,
        "explanation_change_point": {
            "status": "NOT_EXECUTED_DATASET_LIMITATION",
            "reason": (
                "pilot stores sparse preselected three-step blocks; joining them "
                "would create false temporal adjacency"
            ),
        },
        "rollout_natural_frequency": {
            "status": "NOT_EXECUTED_DATASET_LIMITATION",
            "reason": (
                "collector intentionally retained stratified anchors, not every "
                "natural rollout decision"
            ),
        },
        "per_solver": {},
    }
    complete = np.asarray([
        bool(row["complete_window"]) for row in dataset.metadata
    ])
    extended_ablations["complete_fixed_window_only"] = _discover_variant(
        selected_features[complete], selected_names,
        tuple(row for row, selected in zip(dataset.metadata, complete) if selected),
        config["collection"]["development_seed_count"],
    )
    for solver in sorted({row["solver"] for row in dataset.metadata}):
        solver_mask = np.asarray([
            row["solver"] == solver for row in dataset.metadata
        ])
        extended_ablations["per_solver"][solver] = _discover_variant(
            selected_features[solver_mask], selected_names,
            tuple(
                row for row, selected in zip(dataset.metadata, solver_mask)
                if selected
            ),
            config["collection"]["development_seed_count"],
        )
    assert_label_free_feature_contract(selected_names)
    discovery, scaled = discover(
        selected_features, dataset.metadata,
        config["collection"]["development_seed_count"],
    )
    kmeans, kmeans_labels, kmeans_search = kmeans_sensitivity(
        scaled, discovery.split
    )
    cards = build_cluster_cards(
        discovery.all_labels, selected_features, scaled,
        selected_names, dataset.metadata,
    )

    assignment_rows = []
    for index, metadata in enumerate(dataset.metadata):
        assignment_rows.append({
            **dict(metadata),
            "segment_index": index,
            "split": discovery.split[index],
            "hdbscan_cluster": int(discovery.all_labels[index]),
            "kmeans_cluster": int(kmeans_labels[index]),
            "atom_ids": "|".join(dataset.atom_ids[index]),
        })
    assignment_path = output / "cluster_assignments_unlabeled.csv"
    _atomic_csv(assignment_path, assignment_rows)
    signature_rows = []
    for index, metadata in enumerate(dataset.metadata):
        row = {"segment_index": index, **dict(metadata)}
        row.update({
            name: float(value)
            for name, value in zip(selected_names, selected_features[index])
        })
        signature_rows.append(row)
    signature_path = output / "signatures_unlabeled.csv"
    _atomic_csv(signature_path, signature_rows)

    model_dir = output / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(discovery.scaler, model_dir / "scaler.joblib")
    joblib.dump(discovery.clusterer, model_dir / "hdbscan_development.joblib")
    joblib.dump(kmeans, model_dir / "kmeans.joblib")
    freeze_payload = {
        "stage": "EDP6-EDP8-pre-M2",
        "feature_names": selected_names,
        "selected_parameters": dict(discovery.selected_parameters),
        "diagnostics": dict(discovery.diagnostics),
        "cluster_cards": cards,
        "assignment_sha256": file_sha256(assignment_path),
        "signature_sha256": file_sha256(signature_path),
        "m2_labels_opened": False,
    }
    freeze_payload["freeze_sha256"] = freeze_hash(freeze_payload)
    freeze_path = output / "cluster_freeze_pre_m2.json"
    atomic_json(freeze_path, freeze_payload)

    # Only after the immutable cluster freeze exists may M2 labels be opened.
    label_by_atom, label_rows = _labels_after_freeze(atoms)
    label_path = output / "m2_labels_after_cluster_freeze.csv"
    _atomic_csv(label_path, label_rows)
    primitive_labels, window_purity = majority_segment_labels(
        dataset.atom_ids, label_by_atom
    )
    reconciliation = reconcile(
        discovery.all_labels, primitive_labels,
        dataset.metadata, discovery.split,
    )
    kmeans_reconciliation = reconcile(
        kmeans_labels, primitive_labels, dataset.metadata, discovery.split
    )
    shared_clusters = sum(
        len(card["solver_counts"]) >= 2 for card in cards
    )
    solver_specific = sum(
        len(card["solver_counts"]) == 1 for card in cards
    )
    summary = {
        "stage": "EDP5-EDP10",
        "method": "explanation-derived driving primitive discovery",
        "segments": len(dataset.metadata),
        "atoms": len(atoms),
        "features_before_variance_filter": len(dataset.feature_names),
        "features_after_variance_filter": len(selected_names),
        "hdbscan": {
            "selected_parameters": dict(discovery.selected_parameters),
            "diagnostics": dict(discovery.diagnostics),
            "search": list(discovery.search),
            "reconciliation_after_freeze": reconciliation,
        },
        "kmeans_sensitivity": {
            "search": list(kmeans_search),
            "reconciliation_after_freeze": kmeans_reconciliation,
        },
        "cluster_cards": cards,
        "shared_clusters": shared_clusters,
        "solver_specific_clusters": solver_specific,
        "mean_m2_window_purity": float(window_purity.mean()),
        "solver_predictability": _solver_predictability(
            scaled, dataset.metadata, discovery.split
        ),
        "outcome_coherence": _outcome_coherence(
            scaled, discovery.all_labels
        ),
        "ablations": _ablation(
            selected_features, selected_names, dataset.metadata,
            {
                "without_state_counterfactual": ("mean__counterfactual__", "std__counterfactual__", "delta__counterfactual__"),
                "without_paired_physical_outcome": ("mean__physical__", "std__physical__", "delta__physical__"),
                "without_verification": ("mean__verification__", "std__verification__", "delta__verification__"),
            },
        ),
        "extended_ablations": extended_ablations,
        "files": {
            "cluster_freeze_pre_m2": str(freeze_path),
            "assignments_unlabeled": str(assignment_path),
            "signatures_unlabeled": str(signature_path),
            "m2_labels_after_freeze": str(label_path),
        },
        "acceptance": {
            "cluster_frozen_before_m2_labels": freeze_path.is_file(),
            "feature_contract_label_free": True,
            "development_and_heldout_present": set(discovery.split) == {"development", "heldout"},
            "deterministic_hdbscan": discovery.diagnostics["deterministic_rerun"],
            "at_least_two_development_clusters": discovery.diagnostics["development"]["clusters"] >= 2,
            "inductive_heldout_assignment": True,
            "all_segments_have_m2_evaluation_label": len(primitive_labels) == len(dataset.metadata),
            "scientific_metrics_are_not_engineering_gates": True,
        },
    }
    summary["acceptance"]["passed"] = all(summary["acceptance"].values())
    summary_path = output / "eddp_discovery_summary.json"
    atomic_json(summary_path, summary)
    print(json.dumps({
        "segments": summary["segments"],
        "clusters": discovery.diagnostics["all"]["clusters"],
        "coverage": discovery.diagnostics["all"]["coverage"],
        "shared_clusters": shared_clusters,
        "passed": summary["acceptance"]["passed"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
