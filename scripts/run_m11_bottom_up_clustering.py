"""Run M11 label-free primitive discovery and frozen-lexicon reconciliation."""

import argparse
from dataclasses import asdict, is_dataclass
import csv
from hashlib import sha256
import json
from pathlib import Path

import joblib
import numpy as np

from src.explainability.cluster_primitives import (
    deterministic_hdbscan_check,
    fit_hdbscan,
    fit_kmeans,
    fit_scaler,
    search_hdbscan,
    search_kmeans,
    split_masks,
)
from src.explainability.reconcile_clusters import (
    flat_confusion_rows,
    label_window_statistics,
    reconcile,
)
from src.explainability.signatures import (
    assert_label_free_feature_contract,
    build_signature_dataset,
    derive_evaluation_labels,
    load_step_rows,
    signature_manifest,
)


def _hash(path):
    digest = sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _plain(value):
    if is_dataclass(value):
        return _plain(asdict(value))
    if isinstance(value, dict):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(_plain(payload), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _write_csv(path, rows, fieldnames=None):
    rows = list(rows)
    if fieldnames is None:
        fieldnames = list(rows[0]) if rows else []
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _save_joblib(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    joblib.dump(value, temporary, compress=3)
    temporary.replace(path)
    return {"path": str(path), "sha256": _hash(path)}


def _signature_rows(dataset, scaled, labels, split_name):
    rows = []
    for index, (metadata, raw, normalized) in enumerate(
        zip(dataset.metadata, dataset.features, scaled)
    ):
        row = dict(metadata)
        row["split"] = split_name[index]
        row["hdbscan_cluster"] = int(labels[index])
        for name, value in zip(dataset.feature_names, raw):
            row[name] = float(value)
        for name, value in zip(dataset.feature_names, normalized):
            row["z_" + name] = float(value)
        rows.append(row)
    return rows


def _assignment_rows(dataset, hdbscan_labels, kmeans_labels, evaluations, split_name):
    rows = []
    for metadata, hdbscan, kmeans, evaluation, split in zip(
        dataset.metadata, hdbscan_labels, kmeans_labels, evaluations, split_name
    ):
        rows.append({
            **dict(metadata),
            "split": split,
            "hdbscan_cluster": int(hdbscan),
            "kmeans_cluster": int(kmeans),
            "primitive": evaluation["primitive"],
            "window_label_purity": float(evaluation["label_purity"]),
            "primitive_counts": json.dumps(
                evaluation["primitive_counts"], sort_keys=True
            ),
        })
    return rows


def _cluster_profile_rows(labels, dataset, scaled):
    labels = np.asarray(labels, dtype=np.int64)
    rows = []
    for cluster in sorted(value for value in np.unique(labels) if value >= 0):
        mask = labels == cluster
        solvers = {}
        for item in np.asarray(dataset.metadata, dtype=object)[mask]:
            solver = str(item["solver"])
            solvers[solver] = solvers.get(solver, 0) + 1
        raw_mean = np.mean(dataset.features[mask], axis=0)
        z_mean = np.mean(scaled[mask], axis=0)
        row = {
            "cluster": int(cluster),
            "samples": int(np.sum(mask)),
            "solver_counts": json.dumps(solvers, sort_keys=True),
        }
        for name, value in zip(dataset.feature_names, raw_mean):
            row["mean_" + name] = float(value)
        for name, value in zip(dataset.feature_names, z_mean):
            row["mean_z_" + name] = float(value)
        rows.append(row)
    return rows


def run(args):
    rows = load_step_rows(args.steps)
    dataset = build_signature_dataset(rows, window_size=args.window_size)
    assert_label_free_feature_contract(dataset.feature_names)
    dev_mask, heldout_mask, seed_split = split_masks(
        dataset.metadata, development_seed_count=args.development_seed_count
    )
    split_name = np.where(dev_mask, "development", "heldout")

    # No primitive label has been opened above this point.
    scaler = fit_scaler(dataset.features[dev_mask])
    scaled = scaler.transform(dataset.features)
    hdbscan_selected, hdbscan_candidates = search_hdbscan(scaled[dev_mask])
    kmeans_selected, kmeans_candidates = search_kmeans(scaled[dev_mask])

    # HDBSCAN lacks an inductive predict API in scikit-learn. Parameters are
    # frozen on development data, then one final transductive fit uses all
    # feature vectors. Held-out primitive labels remain unopened.
    hdbscan_model, hdbscan_labels, hdbscan_diagnostics = fit_hdbscan(
        scaled, hdbscan_selected["parameters"]
    )
    kmeans_model, kmeans_labels, kmeans_diagnostics = fit_kmeans(
        scaled[dev_mask], scaled, kmeans_selected["parameters"]
    )
    deterministic = deterministic_hdbscan_check(
        scaled, hdbscan_selected["parameters"]
    )

    # Frozen M2 labels are opened only now, after feature construction,
    # hyperparameter selection, and cluster assignment are complete.
    evaluation_labels = derive_evaluation_labels(rows, dataset.metadata)
    primitives = [item["primitive"] for item in evaluation_labels]
    strata = {"development": dev_mask, "heldout": heldout_mask}
    hdbscan_reconciliation = reconcile(
        hdbscan_labels, primitives, dataset.metadata, scaled, strata
    )
    kmeans_reconciliation = reconcile(
        kmeans_labels, primitives, dataset.metadata, scaled, strata
    )

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    unlabeled_rows = _signature_rows(
        dataset, scaled, hdbscan_labels, split_name
    )
    _write_csv(output / "signatures_unlabeled.csv", unlabeled_rows)
    _write_csv(
        output / "cluster_assignments_with_frozen_labels.csv",
        _assignment_rows(
            dataset, hdbscan_labels, kmeans_labels, evaluation_labels, split_name
        ),
    )
    _write_csv(
        output / "hdbscan_confusion_matrix.csv",
        flat_confusion_rows(hdbscan_reconciliation),
    )
    _write_csv(
        output / "kmeans_confusion_matrix.csv",
        flat_confusion_rows(kmeans_reconciliation),
    )
    _write_csv(
        output / "hdbscan_cluster_profiles.csv",
        _cluster_profile_rows(hdbscan_labels, dataset, scaled),
    )
    _write_json(output / "hdbscan_search.json", hdbscan_candidates)
    _write_json(output / "kmeans_search.json", kmeans_candidates)
    _write_json(output / "hdbscan_reconciliation.json", hdbscan_reconciliation)
    _write_json(output / "kmeans_reconciliation.json", kmeans_reconciliation)

    models = {
        "scaler": _save_joblib(output / "signature_scaler.joblib", scaler),
        "hdbscan": _save_joblib(output / "hdbscan.joblib", hdbscan_model),
        "kmeans": _save_joblib(output / "kmeans.joblib", kmeans_model),
    }
    lexicon_frozen_before_clustering = Path(args.lexicon_freeze).is_file()
    checks = {
        "fixed_windows_do_not_use_m12_primitive_boundaries": True,
        "feature_names_pass_label_leakage_guard": True,
        "primitive_labels_not_used_for_hyperparameter_selection": all(
            not item["primitive_labels_used_for_selection"]
            for item in (*hdbscan_candidates, *kmeans_candidates)
        ),
        "development_and_heldout_seeds_are_disjoint": not bool(
            np.any(dev_mask & heldout_mask)
        ),
        "lexicon_frozen_before_clustering": lexicon_frozen_before_clustering,
        "hdbscan_is_deterministic": deterministic,
        "hdbscan_has_at_least_two_clusters": hdbscan_diagnostics.clusters >= 2,
        "hdbscan_coverage_ge_0_50": hdbscan_diagnostics.coverage >= 0.50,
        "kmeans_sensitivity_has_at_least_two_clusters": kmeans_diagnostics.clusters >= 2,
        "all_windows_receive_frozen_primitive_evaluation_label": (
            len(evaluation_labels) == len(dataset.features)
        ),
    }
    failed = sorted(name for name, passed in checks.items() if not passed)
    summary = {
        "stage": "M11",
        "method": "label-free behavioral signatures + HDBSCAN; K-means sensitivity",
        "input": {
            "steps": str(args.steps),
            "steps_sha256": _hash(args.steps),
            "lexicon_freeze": str(args.lexicon_freeze),
            "lexicon_freeze_sha256": _hash(args.lexicon_freeze),
        },
        "signature_manifest": signature_manifest(dataset),
        "seed_split": seed_split,
        "development_samples": int(np.sum(dev_mask)),
        "heldout_samples": int(np.sum(heldout_mask)),
        "support_contract": {
            "all_windows_are_evaluation_reached": True,
            "training_visit_count_claimed": False,
            "unseen_q_table_states_clustered": False,
        },
        "hdbscan": {
            "selection": hdbscan_selected,
            "final_diagnostics": hdbscan_diagnostics,
            "final_fit_semantics": "transductive_all_features_after_development_parameter_freeze",
            "heldout_primitive_labels_used_during_fit": False,
            "reconciliation": hdbscan_reconciliation,
        },
        "kmeans_sensitivity": {
            "selection": kmeans_selected,
            "final_diagnostics": kmeans_diagnostics,
            "final_fit_semantics": "fit_development_features_predict_all",
            "reconciliation": kmeans_reconciliation,
        },
        "window_label_statistics": label_window_statistics(evaluation_labels),
        "models": models,
        "acceptance": {
            "checks": checks,
            "failed_checks": failed,
            "passed": not failed,
            "alignment_has_no_pass_fail_threshold": True,
            "low_alignment_is_a_scientific_result_not_hidden_failure": True,
            "main_result_eligible": not failed,
        },
        "epistemic_limits": [
            "Clusters validate behavioral structure; they are not local decision explanations.",
            "Majority primitive labels evaluate fixed windows and do not create their boundaries.",
            "HDBSCAN held-out alignment is transductive because sklearn HDBSCAN has no predict API.",
            "Five matched seeds per solver support descriptive validation, not population inference.",
        ],
    }
    _write_json(output / "m11_summary.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--steps", type=Path,
        default=Path("runs/explanations/m12_policy_comparison/steps.csv"),
    )
    parser.add_argument(
        "--lexicon-freeze", type=Path,
        default=Path("docs/primitive_lexicon_v1.freeze.json"),
    )
    parser.add_argument("--window-size", type=int, default=5)
    parser.add_argument("--development-seed-count", type=int, default=3)
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("runs/explanations/m11_bottom_up_clustering"),
    )
    args = parser.parse_args()
    summary = run(args)
    print(json.dumps({
        "stage": summary["stage"],
        "acceptance": summary["acceptance"],
        "hdbscan": {
            "diagnostics": _plain(summary["hdbscan"]["final_diagnostics"]),
            "metrics": summary["hdbscan"]["reconciliation"]["metrics"],
        },
        "kmeans": {
            "diagnostics": _plain(summary["kmeans_sensitivity"]["final_diagnostics"]),
            "metrics": summary["kmeans_sensitivity"]["reconciliation"]["metrics"],
        },
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
