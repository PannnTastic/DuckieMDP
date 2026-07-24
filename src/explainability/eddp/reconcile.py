"""Cluster cards, functional naming, and sealed-M2 reconciliation."""

from collections import Counter
from hashlib import sha256
import json
from typing import Any, Dict, Mapping

import numpy as np
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score


def freeze_hash(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        dict(payload), sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return sha256(encoded).hexdigest()


def _feature(row, names, name, default=0.0):
    try:
        return float(row[names.index(name)])
    except ValueError:
        return float(default)


def _functional_name(cluster_id, rows, names, context_counts, solver_counts):
    mean = np.asarray(rows, dtype=float).mean(axis=0)
    stop_delta = _feature(mean, names, "mean__physical__delta_stop_violations")
    clearance_delta = _feature(
        mean, names, "mean__physical__delta_minimum_duck_clearance_m"
    )
    lane_delta = _feature(
        mean, names, "mean__physical__delta_max_abs_lateral_error_m"
    )
    heading_delta = _feature(
        mean, names, "mean__physical__delta_max_abs_heading_error_rad"
    )
    factual_brake = _feature(mean, names, "mean__physical__factual_brake_ratio")
    progress_delta = _feature(
        mean, names, "mean__physical__delta_forward_progress_m"
    )
    factual_progress = _feature(
        mean, names, "mean__physical__factual_forward_progress_m"
    )
    total = float(sum(context_counts.values()))
    duck_fraction = context_counts.get("duck", 0) / total
    stop_fraction = context_counts.get("stop", 0) / total
    lane_fraction = context_counts.get("lane", 0) / total
    if duck_fraction >= 0.75:
        base, reason = (
            "PedestrianRiskModulatedControl",
            "cluster is dominated by Duckie context and a distinct temporal response",
        )
    elif stop_fraction >= 0.75:
        base, reason = (
            "StopDistanceModulatedControl",
            "cluster is dominated by stop context and stop-distance decision boundaries",
        )
    elif len(solver_counts) == 1 and "sac" in solver_counts:
        base, reason = (
            "ContinuousLowSpeedRegulation",
            "continuous-control-only cluster with lower factual progress",
        )
    elif stop_delta > 0.10:
        base, reason = "StopComplianceProtection", "foil causes more stop violations"
    elif clearance_delta < -0.03:
        base, reason = (
            "PedestrianClearanceProtection",
            "factual branch preserves greater Duckie clearance",
        )
    elif lane_delta > 0.015 or heading_delta > 0.05:
        base, reason = "LaneErrorRecovery", "foil produces larger lane or heading error"
    elif progress_delta > 0.03 and factual_brake > 0.50:
        base, reason = (
            "ConservativeHold",
            "foil progresses farther while factual branch mostly holds",
        )
    elif factual_brake > 0.20 and lane_fraction > 0.30:
        base, reason = (
            "ConservativeLaneRegulation",
            "lane-focused cluster combines regulation with frequent braking",
        )
    elif duck_fraction > 0.40 and factual_brake < 0.10:
        base, reason = (
            "ContextAwareProceed",
            "Duckie-aware cluster maintains progress without a dominant hold response",
        )
    elif factual_progress > 0.03:
        base, reason = "ProgressPreservingRegulation", "factual trajectory maintains progress"
    else:
        base, reason = "AdaptiveTrajectoryControl", "mixed local trajectory regulation"
    return "%s_C%02d" % (base, int(cluster_id)), reason


def build_cluster_cards(labels, raw_features, scaled_features, feature_names, metadata):
    labels = np.asarray(labels, dtype=int)
    cards = []
    for cluster_id in sorted(set(labels[labels >= 0])):
        indices = np.flatnonzero(labels == cluster_id)
        raw = raw_features[indices]
        scaled = scaled_features[indices]
        centroid = scaled.mean(axis=0)
        distances = np.linalg.norm(scaled - centroid, axis=1)
        medoid_index = int(indices[int(np.argmin(distances))])
        z_mean = scaled.mean(axis=0)
        top = np.argsort(np.abs(z_mean))[::-1][:12]
        solver_counts = Counter(metadata[index]["solver"] for index in indices)
        context_counts = Counter(
            metadata[index]["selection_context"] for index in indices
        )
        name, rationale = _functional_name(
            cluster_id,
            raw,
            list(feature_names),
            context_counts,
            solver_counts,
        )
        cards.append({
            "cluster_id": int(cluster_id),
            "candidate_name": name,
            "naming_rationale": rationale,
            "status": (
                "PRIMITIVE_CANDIDATE"
                if len(solver_counts) >= 2 else "SOLVER_SPECIFIC_BEHAVIOR"
            ),
            "support": int(len(indices)),
            "seed_support": int(len({metadata[index]["seed"] for index in indices})),
            "solver_counts": dict(sorted(solver_counts.items())),
            "context_counts": dict(sorted(context_counts.items())),
            "representative_segment_index": medoid_index,
            "representative_block_id": metadata[medoid_index]["block_id"],
            "top_standardized_features": [
                {"feature": feature_names[index], "mean_z": float(z_mean[index])}
                for index in top
            ],
        })
    return cards


def majority_segment_labels(atom_groups, label_by_atom):
    labels, purities = [], []
    for group in atom_groups:
        values = [label_by_atom[atom_id] for atom_id in group]
        counts = Counter(values)
        label, count = counts.most_common(1)[0]
        labels.append(label)
        purities.append(float(count) / len(values))
    return np.asarray(labels, dtype=object), np.asarray(purities, dtype=float)


def _metrics(cluster_labels, primitive_labels):
    cluster_labels = np.asarray(cluster_labels, dtype=int)
    primitive_labels = np.asarray(primitive_labels, dtype=object)
    mask = cluster_labels >= 0
    if not mask.any():
        return {"coverage": 0.0, "purity": None, "nmi": None, "ari": None}
    weighted_correct, confusion = 0, []
    for cluster_id in sorted(set(cluster_labels[mask])):
        counts = Counter(primitive_labels[cluster_labels == cluster_id])
        weighted_correct += counts.most_common(1)[0][1]
        for primitive, count in sorted(counts.items()):
            confusion.append({
                "cluster_id": int(cluster_id), "primitive": primitive,
                "count": int(count),
            })
    return {
        "coverage": float(mask.mean()),
        "purity": float(weighted_correct) / int(mask.sum()),
        "nmi": float(normalized_mutual_info_score(
            primitive_labels[mask], cluster_labels[mask]
        )),
        "ari": float(adjusted_rand_score(
            primitive_labels[mask], cluster_labels[mask]
        )),
        "confusion": confusion,
    }


def reconcile(labels, primitive_labels, metadata, split):
    labels = np.asarray(labels, dtype=int)
    primitive_labels = np.asarray(primitive_labels, dtype=object)
    split = np.asarray(split)
    result = {"overall": _metrics(labels, primitive_labels)}
    for value in ("development", "heldout"):
        mask = split == value
        result[value] = _metrics(labels[mask], primitive_labels[mask])
    result["by_solver"] = {}
    for solver in sorted({row["solver"] for row in metadata}):
        mask = np.asarray([row["solver"] == solver for row in metadata])
        result["by_solver"][solver] = _metrics(labels[mask], primitive_labels[mask])
    return result
