"""Unsupervised clustering utilities for M11 behavioral signatures."""

from dataclasses import asdict, dataclass
from typing import Any, Iterable, Mapping, Optional, Sequence

import numpy as np
from sklearn.cluster import HDBSCAN, KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


CLUSTER_SCHEMA_VERSION = "1.0.0"


@dataclass(frozen=True)
class ClusterDiagnostics:
    samples: int
    clusters: int
    noise_samples: int
    coverage: float
    noise_rate: float
    silhouette: Optional[float]
    schema_version: str = CLUSTER_SCHEMA_VERSION


def fit_scaler(development_features):
    matrix = np.asarray(development_features, dtype=np.float64)
    if matrix.ndim != 2 or len(matrix) < 2:
        raise ValueError("development features must be non-empty 2-D")
    scaler = StandardScaler()
    scaler.fit(matrix)
    return scaler


def cluster_diagnostics(features, labels) -> ClusterDiagnostics:
    matrix = np.asarray(features, dtype=np.float64)
    assigned = np.asarray(labels, dtype=np.int64)
    if matrix.ndim != 2 or assigned.shape != (len(matrix),):
        raise ValueError("cluster labels do not match feature matrix")
    non_noise = assigned >= 0
    unique = np.unique(assigned[non_noise])
    coverage = float(np.mean(non_noise)) if len(assigned) else 0.0
    silhouette = None
    if len(unique) >= 2 and int(np.sum(non_noise)) > len(unique):
        silhouette = float(silhouette_score(matrix[non_noise], assigned[non_noise]))
    return ClusterDiagnostics(
        samples=int(len(assigned)),
        clusters=int(len(unique)),
        noise_samples=int(np.sum(~non_noise)),
        coverage=coverage,
        noise_rate=1.0 - coverage,
        silhouette=silhouette,
    )


def _candidate_payload(parameters, diagnostics, objective):
    return {
        "parameters": dict(parameters),
        "diagnostics": asdict(diagnostics),
        "unsupervised_objective": float(objective),
        "primitive_labels_used_for_selection": False,
    }


def search_hdbscan(
    development_features,
    min_cluster_sizes: Sequence[int] = (8, 12, 16, 24, 32),
    min_samples_values: Sequence[int] = (3, 5, 8),
):
    """Select HDBSCAN parameters using features and unsupervised metrics only."""
    matrix = np.asarray(development_features, dtype=np.float64)
    candidates = []
    best = None
    for min_cluster_size in min_cluster_sizes:
        if int(min_cluster_size) >= len(matrix):
            continue
        for min_samples in min_samples_values:
            model = HDBSCAN(
                min_cluster_size=int(min_cluster_size),
                min_samples=int(min_samples),
                cluster_selection_method="eom",
                allow_single_cluster=False,
            )
            labels = model.fit_predict(matrix)
            diagnostics = cluster_diagnostics(matrix, labels)
            silhouette = diagnostics.silhouette
            objective = -1.0 if silhouette is None else float(
                silhouette + 0.25 * diagnostics.coverage
            )
            payload = _candidate_payload(
                {
                    "min_cluster_size": int(min_cluster_size),
                    "min_samples": int(min_samples),
                    "cluster_selection_method": "eom",
                },
                diagnostics,
                objective,
            )
            candidates.append(payload)
            rank = (
                diagnostics.clusters >= 2,
                objective,
                diagnostics.coverage,
                -int(min_cluster_size),
                -int(min_samples),
            )
            if best is None or rank > best[0]:
                best = (rank, payload)
    if best is None or not best[0][0]:
        raise RuntimeError("HDBSCAN search found no multi-cluster solution")
    return best[1], tuple(candidates)


def fit_hdbscan(features, parameters: Mapping[str, Any]):
    matrix = np.asarray(features, dtype=np.float64)
    model = HDBSCAN(
        min_cluster_size=int(parameters["min_cluster_size"]),
        min_samples=int(parameters["min_samples"]),
        cluster_selection_method=str(parameters.get("cluster_selection_method", "eom")),
        allow_single_cluster=False,
    )
    labels = model.fit_predict(matrix)
    return model, labels, cluster_diagnostics(matrix, labels)


def search_kmeans(
    development_features,
    k_values: Iterable[int] = range(2, 13),
    random_state: int = 0,
):
    matrix = np.asarray(development_features, dtype=np.float64)
    candidates = []
    best = None
    for k in k_values:
        if int(k) >= len(matrix):
            continue
        model = KMeans(n_clusters=int(k), n_init=20, random_state=int(random_state))
        labels = model.fit_predict(matrix)
        silhouette = float(silhouette_score(matrix, labels))
        payload = {
            "parameters": {"k": int(k), "random_state": int(random_state), "n_init": 20},
            "diagnostics": asdict(cluster_diagnostics(matrix, labels)),
            "unsupervised_objective": silhouette,
            "primitive_labels_used_for_selection": False,
        }
        candidates.append(payload)
        rank = (silhouette, -int(k))
        if best is None or rank > best[0]:
            best = (rank, payload)
    if best is None:
        raise RuntimeError("K-means search produced no candidate")
    return best[1], tuple(candidates)


def fit_kmeans(development_features, all_features, parameters):
    development = np.asarray(development_features, dtype=np.float64)
    matrix = np.asarray(all_features, dtype=np.float64)
    model = KMeans(
        n_clusters=int(parameters["k"]),
        n_init=int(parameters.get("n_init", 20)),
        random_state=int(parameters.get("random_state", 0)),
    )
    model.fit(development)
    labels = model.predict(matrix)
    return model, labels, cluster_diagnostics(matrix, labels)


def deterministic_hdbscan_check(features, parameters) -> bool:
    _, first, _ = fit_hdbscan(features, parameters)
    _, second, _ = fit_hdbscan(features, parameters)
    return bool(np.array_equal(first, second))


def split_masks(metadata, development_seed_count: int = 3):
    """Create solver-balanced seed split without reading primitive labels."""
    seeds_by_solver = {}
    for item in metadata:
        solver = str(item["solver"])
        seeds_by_solver.setdefault(solver, set()).add(int(item["seed"]))
    development = set()
    heldout = set()
    manifest = {}
    for solver, seeds in sorted(seeds_by_solver.items()):
        ordered = sorted(seeds)
        if len(ordered) <= int(development_seed_count):
            raise ValueError("not enough seeds for disjoint M11 held-out split")
        dev = ordered[: int(development_seed_count)]
        test = ordered[int(development_seed_count) :]
        manifest[solver] = {"development": dev, "heldout": test}
        development.update((solver, seed) for seed in dev)
        heldout.update((solver, seed) for seed in test)
    dev_mask = np.asarray([
        (str(item["solver"]), int(item["seed"])) in development
        for item in metadata
    ])
    heldout_mask = np.asarray([
        (str(item["solver"]), int(item["seed"])) in heldout
        for item in metadata
    ])
    if np.any(dev_mask & heldout_mask) or not np.all(dev_mask | heldout_mask):
        raise ValueError("invalid M11 seed split")
    return dev_mask, heldout_mask, manifest
