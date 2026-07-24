"""Development-only discovery with inductive held-out assignment."""

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Sequence, Tuple

import numpy as np
from sklearn.cluster import HDBSCAN, KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.preprocessing import StandardScaler


@dataclass(frozen=True)
class DiscoveryResult:
    scaler: Any
    clusterer: Any
    development_labels: np.ndarray
    heldout_labels: np.ndarray
    all_labels: np.ndarray
    split: np.ndarray
    search: Tuple[Mapping[str, Any], ...]
    selected_parameters: Mapping[str, Any]
    diagnostics: Mapping[str, Any]


def split_by_seed(metadata: Sequence[Mapping[str, Any]], development_count: int = 3):
    dev_seeds = {}
    for solver in sorted({row["solver"] for row in metadata}):
        values = sorted({int(row["seed"]) for row in metadata if row["solver"] == solver})
        dev_seeds[solver] = set(values[: int(development_count)])
    split = np.asarray([
        "development" if int(row["seed"]) in dev_seeds[row["solver"]]
        else "heldout"
        for row in metadata
    ])
    return split, {key: sorted(value) for key, value in dev_seeds.items()}


def _diagnostics(features, labels):
    labels = np.asarray(labels, dtype=int)
    mask = labels >= 0
    clusters = sorted(set(labels[mask]))
    coverage = float(mask.mean()) if labels.size else 0.0
    silhouette = None
    if len(clusters) >= 2 and int(mask.sum()) > len(clusters):
        silhouette = float(silhouette_score(features[mask], labels[mask]))
    return {
        "clusters": len(clusters),
        "coverage": coverage,
        "noise": 1.0 - coverage,
        "silhouette": silhouette,
    }


def _centroid_assignment(development, labels, heldout):
    cluster_ids = sorted(set(int(value) for value in labels if int(value) >= 0))
    if not cluster_ids:
        return np.full(len(heldout), -1, dtype=int), {}
    centroids = {}
    radii = {}
    for cluster_id in cluster_ids:
        points = development[np.asarray(labels) == cluster_id]
        centroid = points.mean(axis=0)
        distances = np.linalg.norm(points - centroid, axis=1)
        centroids[cluster_id] = centroid
        radii[cluster_id] = max(float(np.quantile(distances, 0.95)), 1e-6)
    assigned = []
    for point in heldout:
        distances = {
            cluster_id: float(np.linalg.norm(point - centroid))
            for cluster_id, centroid in centroids.items()
        }
        best = min(distances, key=distances.get)
        assigned.append(best if distances[best] <= radii[best] * 1.25 else -1)
    return np.asarray(assigned, dtype=int), {
        str(key): {"radius_p95": radii[key]} for key in cluster_ids
    }


def discover(features, metadata, development_seed_count: int = 3):
    features = np.asarray(features, dtype=np.float64)
    split, dev_seeds = split_by_seed(metadata, development_seed_count)
    dev_mask = split == "development"
    heldout_mask = split == "heldout"
    if dev_mask.sum() < 10 or heldout_mask.sum() < 2:
        raise ValueError("EDDP needs development and held-out segments")
    scaler = StandardScaler().fit(features[dev_mask])
    scaled = scaler.transform(features)
    dev = scaled[dev_mask]
    search = []
    best = None
    for min_cluster_size in (4, 6, 8, 12):
        for min_samples in (2, 3, 5):
            if min_cluster_size >= len(dev):
                continue
            model = HDBSCAN(
                min_cluster_size=min_cluster_size,
                min_samples=min_samples,
                cluster_selection_method="eom",
                allow_single_cluster=False,
            )
            labels = model.fit_predict(dev)
            diagnostic = _diagnostics(dev, labels)
            silhouette = diagnostic["silhouette"]
            objective = (-1.0 if silhouette is None else silhouette) + 0.25 * diagnostic["coverage"]
            item = {
                "min_cluster_size": min_cluster_size,
                "min_samples": min_samples,
                "objective": float(objective),
                **diagnostic,
            }
            search.append(item)
            if diagnostic["clusters"] >= 2 and (best is None or objective > best[0]):
                best = (objective, item)
    if best is None:
        raise RuntimeError("HDBSCAN did not find two development clusters")
    parameters = {
        "min_cluster_size": best[1]["min_cluster_size"],
        "min_samples": best[1]["min_samples"],
        "cluster_selection_method": "eom",
        "allow_single_cluster": False,
    }
    clusterer = HDBSCAN(**parameters).fit(dev)
    dev_labels = np.asarray(clusterer.labels_, dtype=int)
    heldout_labels, assignment = _centroid_assignment(
        dev, dev_labels, scaled[heldout_mask]
    )
    all_labels = np.full(len(features), -1, dtype=int)
    all_labels[dev_mask] = dev_labels
    all_labels[heldout_mask] = heldout_labels
    deterministic = np.array_equal(
        dev_labels, HDBSCAN(**parameters).fit_predict(dev)
    )
    diagnostics = {
        "development": _diagnostics(dev, dev_labels),
        "heldout": _diagnostics(scaled[heldout_mask], heldout_labels),
        "all": _diagnostics(scaled, all_labels),
        "development_seeds": dev_seeds,
        "inductive_assignment": "nearest development centroid with 1.25*p95 radius",
        "assignment_support": assignment,
        "deterministic_rerun": bool(deterministic),
    }
    return DiscoveryResult(
        scaler=scaler, clusterer=clusterer,
        development_labels=dev_labels, heldout_labels=heldout_labels,
        all_labels=all_labels, split=split, search=tuple(search),
        selected_parameters=parameters, diagnostics=diagnostics,
    ), scaled


def kmeans_sensitivity(features, split, k_values=range(2, 11)):
    features = np.asarray(features, dtype=np.float64)
    dev_mask = np.asarray(split) == "development"
    heldout_mask = np.asarray(split) == "heldout"
    search = []
    best = None
    for k in k_values:
        if k >= int(dev_mask.sum()):
            continue
        model = KMeans(n_clusters=int(k), random_state=0, n_init=20).fit(features[dev_mask])
        score = float(silhouette_score(features[dev_mask], model.labels_))
        search.append({"k": int(k), "development_silhouette": score})
        if best is None or score > best[0]:
            best = (score, model)
    model = best[1]
    labels = np.empty(len(features), dtype=int)
    labels[dev_mask] = model.labels_
    labels[heldout_mask] = model.predict(features[heldout_mask])
    return model, labels, tuple(search)
