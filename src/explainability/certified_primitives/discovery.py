"""Development-only discovery and frozen held-out primitive assignment."""

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Sequence, Tuple

import numpy as np
from sklearn.cluster import HDBSCAN
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.preprocessing import StandardScaler

from .schema import SourceKind, TemporalExplanationSegment


@dataclass(frozen=True)
class FrozenDiscoveryModel:
    feature_names: Tuple[str, ...]
    scaler: Any
    clusterer: Any
    centroids: Mapping[int, np.ndarray]
    radii: Mapping[int, float]
    selected_parameters: Mapping[str, Any]
    development_seeds: Mapping[str, Tuple[int, ...]]
    reducer: Any = None


@dataclass(frozen=True)
class DiscoveryResult:
    model: FrozenDiscoveryModel
    labels: np.ndarray
    split: np.ndarray
    scaled_features: np.ndarray
    diagnostics: Mapping[str, Any]
    search: Tuple[Mapping[str, Any], ...]


def _matrix(segments: Sequence[TemporalExplanationSegment]) -> np.ndarray:
    if not segments:
        raise ValueError("discovery requires segments")
    names = segments[0].feature_names
    if any(item.feature_names != names for item in segments):
        raise ValueError("segment feature schemas differ")
    matrix = np.asarray([item.feature_values for item in segments], dtype=np.float64)
    if not np.isfinite(matrix).all():
        raise ValueError("segment matrix contains non-finite values")
    return matrix


def split_by_seed(
    segments: Sequence[TemporalExplanationSegment],
    development_seed_count: int = 3,
) -> Tuple[np.ndarray, Mapping[str, Tuple[int, ...]]]:
    selected: Dict[str, Tuple[int, ...]] = {}
    for solver in sorted({item.solver for item in segments}):
        seeds = sorted({int(item.seed) for item in segments if item.solver == solver})
        selected[solver] = tuple(seeds[: int(development_seed_count)])
    split = np.asarray([
        "development" if int(item.seed) in selected[item.solver] else "heldout"
        for item in segments
    ])
    return split, selected


def _cluster_stats(features: np.ndarray, labels: np.ndarray) -> Dict[str, Any]:
    labels = np.asarray(labels, dtype=int)
    mask = labels >= 0
    cluster_ids = sorted(set(labels[mask]))
    silhouette = None
    if len(cluster_ids) >= 2 and int(mask.sum()) > len(cluster_ids):
        silhouette = float(silhouette_score(features[mask], labels[mask]))
    return {
        "clusters": len(cluster_ids),
        "coverage": float(mask.mean()) if len(labels) else 0.0,
        "noise_rate": float(1.0 - mask.mean()) if len(labels) else 1.0,
        "silhouette": silhouette,
    }


def _centroids_and_radii(
    features: np.ndarray, labels: np.ndarray, quantile: float = 0.95
) -> Tuple[Dict[int, np.ndarray], Dict[int, float]]:
    centroids, radii = {}, {}
    for cluster_id in sorted(set(int(value) for value in labels if int(value) >= 0)):
        points = features[np.asarray(labels) == cluster_id]
        centroid = points.mean(axis=0)
        distances = np.linalg.norm(points - centroid, axis=1)
        centroids[cluster_id] = centroid
        radii[cluster_id] = max(float(np.quantile(distances, quantile)), 1e-8)
    return centroids, radii


def assign_frozen(
    model: FrozenDiscoveryModel,
    raw_features: np.ndarray,
    radius_multiplier: float = 1.25,
) -> np.ndarray:
    values = np.asarray(raw_features, dtype=np.float64)
    scaled = model.scaler.transform(values)
    reducer = getattr(model, "reducer", None)
    if reducer is not None:
        scaled = reducer.transform(scaled)
    assigned = []
    for row in scaled:
        if not model.centroids:
            assigned.append(-1)
            continue
        distances = {
            cluster_id: float(np.linalg.norm(row - centroid))
            for cluster_id, centroid in model.centroids.items()
        }
        best = min(distances, key=distances.get)
        assigned.append(
            best
            if distances[best] <= model.radii[best] * float(radius_multiplier)
            else -1
        )
    return np.asarray(assigned, dtype=int)


def bootstrap_stability(
    development: np.ndarray,
    reference_labels: np.ndarray,
    parameters: Mapping[str, Any],
    *,
    repeats: int = 24,
    seed: int = 0,
) -> float:
    rng = np.random.RandomState(seed)
    scores = []
    n = len(development)
    if n < 10:
        return 0.0
    for _ in range(int(repeats)):
        indices = np.sort(rng.choice(n, size=max(8, int(0.8 * n)), replace=False))
        try:
            labels = HDBSCAN(**dict(parameters)).fit_predict(development[indices])
        except ValueError:
            scores.append(0.0)
            continue
        if len(set(int(value) for value in labels if int(value) >= 0)) < 2:
            scores.append(0.0)
            continue
        centroids, radii = _centroids_and_radii(development[indices], labels)
        temporary = type("BootstrapModel", (), {
            "scaler": type("Identity", (), {"transform": staticmethod(lambda x: x)})(),
            "centroids": centroids,
            "radii": radii,
        })()
        predicted = assign_frozen(temporary, development)
        mask = (reference_labels >= 0) & (predicted >= 0)
        scores.append(
            float(adjusted_rand_score(reference_labels[mask], predicted[mask]))
            if int(mask.sum()) >= 3 else 0.0
        )
    return float(np.mean(scores)) if scores else 0.0


def outcome_coherence_ratio(
    features: np.ndarray,
    labels: np.ndarray,
    feature_names: Sequence[str],
    *,
    permutations: int = 64,
    seed: int = 0,
) -> float:
    outcome_indices = [
        index for index, name in enumerate(feature_names) if "__outcome__" in name
    ]
    mask = np.asarray(labels) >= 0
    if not outcome_indices or int(mask.sum()) < 3:
        return 1.0e9
    values = np.asarray(features)[mask][:, outcome_indices]
    target = np.asarray(labels)[mask]

    def within(current):
        total, count = 0.0, 0
        for cluster_id in set(current):
            points = values[current == cluster_id]
            if len(points):
                total += float(np.square(points - points.mean(axis=0)).sum())
                count += int(points.size)
        return total / max(1, count)

    observed = within(target)
    rng = np.random.RandomState(seed)
    null = [within(rng.permutation(target)) for _ in range(int(permutations))]
    denominator = float(np.mean(null))
    return observed / denominator if denominator > 1e-12 else 1.0e9


def discover(
    segments: Sequence[TemporalExplanationSegment],
    *,
    development_seed_count: int = 3,
    allow_legacy_baseline: bool = False,
    pca_components: int = 15,
) -> DiscoveryResult:
    if not allow_legacy_baseline and any(
        item.source_kind != SourceKind.FULL_TRAJECTORY for item in segments
    ):
        raise ValueError("main discovery accepts only full_trajectory segments")
    raw = _matrix(segments)
    split, development_seeds = split_by_seed(segments, development_seed_count)
    dev_mask = split == "development"
    heldout_mask = split == "heldout"
    if int(dev_mask.sum()) < 10 or int(heldout_mask.sum()) < 2:
        raise ValueError("C-EDDP needs development and held-out segments")
    scaler = StandardScaler().fit(raw[dev_mask])
    scaled_full = scaler.transform(raw)
    # The ~600-dim aggregated signature makes HDBSCAN unstable under resampling
    # (bootstrap ARI hovers at the gate). Projecting to a compact PCA subspace
    # sharpens the clusters and lifts stability, still entirely within
    # explanation feature space. Coherence is measured on the named full
    # features; clustering, centroids, and held-out assignment use the subspace.
    n_comp = min(int(pca_components), scaled_full.shape[1], int(dev_mask.sum()) - 1)
    reducer = (
        PCA(n_components=int(n_comp), random_state=0).fit(scaled_full[dev_mask])
        if n_comp >= 2 else None
    )
    scaled = reducer.transform(scaled_full) if reducer is not None else scaled_full
    development = scaled[dev_mask]
    search = []
    best = None
    for min_cluster_size in (4, 6, 8, 12, 16):
        for min_samples in (2, 3, 5):
            if min_cluster_size >= len(development):
                continue
            parameters = {
                "min_cluster_size": min_cluster_size,
                "min_samples": min_samples,
                "cluster_selection_method": "eom",
                "allow_single_cluster": False,
            }
            labels = HDBSCAN(**parameters).fit_predict(development)
            stats = _cluster_stats(development, labels)
            objective = (
                (-1.0 if stats["silhouette"] is None else stats["silhouette"])
                + 0.25 * stats["coverage"]
            )
            row = {**parameters, **stats, "objective": float(objective)}
            search.append(row)
            if stats["clusters"] >= 2 and (best is None or objective > best[0]):
                best = (objective, parameters)
    if best is None:
        raise RuntimeError("HDBSCAN did not discover two development clusters")
    parameters = best[1]
    clusterer = HDBSCAN(**parameters).fit(development)
    development_labels = np.asarray(clusterer.labels_, dtype=int)
    centroids, radii = _centroids_and_radii(development, development_labels)
    model = FrozenDiscoveryModel(
        feature_names=segments[0].feature_names,
        scaler=scaler,
        clusterer=clusterer,
        centroids=centroids,
        radii=radii,
        selected_parameters=parameters,
        development_seeds=development_seeds,
        reducer=reducer,
    )
    labels = np.full(len(segments), -1, dtype=int)
    labels[dev_mask] = development_labels
    labels[heldout_mask] = assign_frozen(model, raw[heldout_mask])
    deterministic = np.array_equal(
        development_labels, HDBSCAN(**parameters).fit_predict(development)
    )
    diagnostics = {
        "development": _cluster_stats(development, labels[dev_mask]),
        "heldout": _cluster_stats(scaled[heldout_mask], labels[heldout_mask]),
        "all": _cluster_stats(scaled, labels),
        "bootstrap_stability": bootstrap_stability(
            development, development_labels, parameters
        ),
        "outcome_coherence_ratio": outcome_coherence_ratio(
            scaled_full, labels, segments[0].feature_names
        ),
        "deterministic_rerun": bool(deterministic),
        "development_seeds": {
            key: list(value) for key, value in development_seeds.items()
        },
        "heldout_assignment_without_refit": True,
    }
    return DiscoveryResult(
        model=model,
        labels=labels,
        split=split,
        scaled_features=scaled,
        diagnostics=diagnostics,
        search=tuple(search),
    )
