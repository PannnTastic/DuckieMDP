"""Reconcile label-free M11 clusters with the frozen M2 primitive lexicon."""

from collections import Counter, defaultdict
from typing import Any, Dict, Mapping, Optional, Sequence

import numpy as np
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score


RECONCILIATION_SCHEMA_VERSION = "1.0.0"


def _weighted_cluster_purity(cluster_labels, primitive_labels) -> Optional[float]:
    clusters = np.asarray(cluster_labels, dtype=np.int64)
    primitives = np.asarray(primitive_labels, dtype=object)
    mask = clusters >= 0
    if not np.any(mask):
        return None
    correct = 0
    for cluster in np.unique(clusters[mask]):
        counts = Counter(primitives[clusters == cluster])
        correct += max(counts.values())
    return float(correct / np.sum(mask))


def _metric_block(cluster_labels, primitive_labels):
    clusters = np.asarray(cluster_labels, dtype=np.int64)
    primitives = np.asarray(primitive_labels, dtype=object)
    if clusters.shape != primitives.shape:
        raise ValueError("cluster and primitive labels differ in shape")
    mask = clusters >= 0
    coverage = float(np.mean(mask)) if len(mask) else 0.0
    if not np.any(mask):
        nmi = ari = None
    else:
        nmi = float(normalized_mutual_info_score(primitives[mask], clusters[mask]))
        ari = float(adjusted_rand_score(primitives[mask], clusters[mask]))
    return {
        "samples": int(len(clusters)),
        "clustered_samples": int(np.sum(mask)),
        "coverage": coverage,
        "noise_rate": 1.0 - coverage,
        "purity": _weighted_cluster_purity(clusters, primitives),
        "normalized_mutual_information": nmi,
        "adjusted_rand_index": ari,
    }


def _cluster_records(cluster_labels, primitive_labels, metadata):
    clusters = np.asarray(cluster_labels, dtype=np.int64)
    primitives = np.asarray(primitive_labels, dtype=object)
    result = []
    for cluster in sorted(np.unique(clusters)):
        if int(cluster) < 0:
            continue
        indices = np.flatnonzero(clusters == cluster)
        primitive_counts = Counter(primitives[indices])
        solver_counts = Counter(str(metadata[index]["solver"]) for index in indices)
        dominant, count = sorted(
            primitive_counts.items(), key=lambda pair: (-pair[1], pair[0])
        )[0]
        result.append({
            "cluster": int(cluster),
            "samples": int(len(indices)),
            "dominant_primitive": str(dominant),
            "cluster_purity": float(count / len(indices)),
            "primitive_counts": dict(sorted(primitive_counts.items())),
            "solver_counts": dict(sorted(solver_counts.items())),
        })
    return result


def _representatives(cluster_labels, features, metadata, primitive_labels):
    labels = np.asarray(cluster_labels, dtype=np.int64)
    matrix = np.asarray(features, dtype=np.float64)
    primitives = np.asarray(primitive_labels, dtype=object)
    result = []
    for cluster in sorted(value for value in np.unique(labels) if value >= 0):
        indices = np.flatnonzero(labels == cluster)
        centroid = np.mean(matrix[indices], axis=0)
        distances = np.linalg.norm(matrix[indices] - centroid, axis=1)
        chosen = int(indices[int(np.argmin(distances))])
        item = dict(metadata[chosen])
        item.update({
            "cluster": int(cluster),
            "primitive": str(primitives[chosen]),
            "distance_to_cluster_centroid": float(np.min(distances)),
        })
        result.append(item)
    return result


def reconcile(
    cluster_labels,
    primitive_labels: Sequence[str],
    metadata: Sequence[Mapping[str, Any]],
    features,
    split_masks: Optional[Mapping[str, Sequence[bool]]] = None,
):
    """Open labels after clustering and quantify cluster/lexicon agreement."""
    clusters = np.asarray(cluster_labels, dtype=np.int64)
    primitives = np.asarray(primitive_labels, dtype=object)
    matrix = np.asarray(features, dtype=np.float64)
    if len(clusters) != len(metadata) or len(clusters) != len(matrix):
        raise ValueError("M11 reconciliation inputs differ in length")
    strata = {"all": np.ones(len(clusters), dtype=bool)}
    if split_masks:
        strata.update({name: np.asarray(mask, dtype=bool) for name, mask in split_masks.items()})
    solvers = sorted({str(item["solver"]) for item in metadata})
    for solver in solvers:
        strata["solver_" + solver] = np.asarray([
            str(item["solver"]) == solver for item in metadata
        ])
    metrics = {}
    for name, mask in strata.items():
        if mask.shape != (len(clusters),):
            raise ValueError("invalid reconciliation stratum")
        metrics[name] = _metric_block(clusters[mask], primitives[mask])

    confusion = defaultdict(Counter)
    for cluster, primitive in zip(clusters, primitives):
        confusion[int(cluster)][str(primitive)] += 1
    return {
        "schema_version": RECONCILIATION_SCHEMA_VERSION,
        "metrics": metrics,
        "clusters": _cluster_records(clusters, primitives, metadata),
        "confusion_matrix": {
            str(cluster): dict(sorted(counts.items()))
            for cluster, counts in sorted(confusion.items())
        },
        "representative_windows": _representatives(
            clusters, matrix, metadata, primitives
        ),
        "low_alignment_is_not_implementation_failure": True,
    }


def label_window_statistics(evaluation_labels):
    purities = np.asarray([
        float(item["label_purity"]) for item in evaluation_labels
    ], dtype=np.float64)
    primitives = [str(item["primitive"]) for item in evaluation_labels]
    return {
        "samples": int(len(purities)),
        "mean_majority_label_purity": float(np.mean(purities)),
        "fully_homogeneous_windows": int(np.sum(purities == 1.0)),
        "mixed_windows": int(np.sum(purities < 1.0)),
        "primitive_counts": dict(sorted(Counter(primitives).items())),
    }


def flat_confusion_rows(reconciliation):
    rows = []
    for cluster, counts in reconciliation["confusion_matrix"].items():
        total = sum(int(value) for value in counts.values())
        for primitive, count in counts.items():
            rows.append({
                "cluster": int(cluster),
                "primitive": primitive,
                "count": int(count),
                "cluster_fraction": float(count / total),
            })
    return rows
