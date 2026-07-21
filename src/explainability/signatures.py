"""Label-free behavioral signatures for M11 bottom-up validation.

The clustering input is built from fixed temporal windows.  Window boundaries
must not depend on primitive labels: the M12 ``segments.csv`` boundaries are
therefore intentionally not used here because they were created from the
top-down labeler that M11 is meant to validate independently.
"""

from dataclasses import dataclass
import csv
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence, Tuple

import numpy as np


SIGNATURE_SCHEMA_VERSION = "1.0.0"
THREAT_LEVELS = ("side_far", "side_near", "crossing_far", "crossing_near")
FORBIDDEN_FEATURE_TOKENS = (
    "primitive",
    "trigger",
    "solver",
    "scenario",
    "event",
    "termination",
    "reward",
    "action_id",
    "action_name",
    "undesirable",
)

FEATURE_NAMES = (
    "mean_d",
    "mean_abs_d",
    "max_abs_d",
    "mean_phi",
    "mean_abs_phi",
    "max_abs_phi",
    "mean_tracking_error",
    "mean_abs_tracking_error",
    "mean_speed",
    "std_speed",
    "mean_curvature",
    "mean_abs_curvature",
    "fraction_curve_left",
    "fraction_curve_right",
    "fraction_straight",
    "stop_present_fraction",
    "stop_unsatisfied_fraction",
    "stop_satisfied_fraction",
    "mean_present_stop_distance",
    "duck_present_fraction",
    "duck_active_fraction",
    "duck_side_far_fraction",
    "duck_side_near_fraction",
    "duck_crossing_far_fraction",
    "duck_crossing_near_fraction",
    "mean_v_cmd",
    "std_v_cmd",
    "min_v_cmd",
    "max_v_cmd",
    "mean_omega_cmd",
    "std_omega_cmd",
    "mean_abs_omega_cmd",
    "max_abs_omega_cmd",
    "mean_abs_delta_v_cmd",
    "deceleration_fraction",
    "mean_abs_delta_omega_cmd",
    "steering_reversal_rate",
)


@dataclass(frozen=True)
class SignatureDataset:
    features: np.ndarray
    feature_names: Tuple[str, ...]
    metadata: Tuple[Mapping[str, Any], ...]
    window_size: int
    schema_version: str = SIGNATURE_SCHEMA_VERSION


def load_step_rows(path: Path) -> Tuple[Dict[str, str], ...]:
    with Path(path).open("r", encoding="utf-8", newline="") as stream:
        return tuple(dict(row) for row in csv.DictReader(stream))


def assert_label_free_feature_contract(feature_names: Sequence[str]) -> None:
    lowered = tuple(str(name).lower() for name in feature_names)
    leaks = [
        name
        for name in lowered
        if any(token in name for token in FORBIDDEN_FEATURE_TOKENS)
    ]
    if leaks:
        raise ValueError("M11 feature-label leakage: %s" % sorted(leaks))
    if len(lowered) != len(set(lowered)):
        raise ValueError("M11 feature names must be unique")


def _float(row: Mapping[str, str], key: str, default: float = 0.0) -> float:
    value = row.get(key, "")
    if value in (None, "", "None", "nan"):
        return float(default)
    return float(value)


def _bool(row: Mapping[str, str], key: str) -> float:
    return float(str(row.get(key, "")).strip().lower() in {"1", "true", "yes"})


def _mean(values) -> float:
    return float(np.mean(np.asarray(values, dtype=np.float64))) if values else 0.0


def _std(values) -> float:
    return float(np.std(np.asarray(values, dtype=np.float64))) if values else 0.0


def _curvature(row: Mapping[str, str]) -> float:
    raw = row.get("curvature", "")
    if raw not in (None, "", "None", "nan"):
        return float(raw)
    return {"curve_left": 1.0, "curve_right": -1.0, "straight": 0.0}.get(
        str(row.get("curvature_class", "straight")), 0.0
    )


def _reversal_rate(omega: np.ndarray, deadband: float = 0.15) -> float:
    if len(omega) < 2:
        return 0.0
    valid = (np.abs(omega[:-1]) >= deadband) & (np.abs(omega[1:]) >= deadband)
    reversals = valid & (omega[:-1] * omega[1:] < 0.0)
    return float(np.mean(reversals))


def _window_features(rows: Sequence[Mapping[str, str]]) -> np.ndarray:
    d = np.asarray([_float(row, "d") for row in rows], dtype=np.float64)
    phi = np.asarray([_float(row, "phi") for row in rows], dtype=np.float64)
    speed = np.asarray([_float(row, "v") for row in rows], dtype=np.float64)
    curvature = np.asarray([_curvature(row) for row in rows], dtype=np.float64)
    v_cmd = np.asarray([_float(row, "v_cmd") for row in rows], dtype=np.float64)
    omega = np.asarray([_float(row, "omega_cmd") for row in rows], dtype=np.float64)
    tracking = d + phi
    curve_classes = [str(row.get("curvature_class", "straight")) for row in rows]
    stop_present = np.asarray([_bool(row, "stop_present") for row in rows])
    stop_satisfied = np.asarray([_bool(row, "stop_satisfied") for row in rows])
    stop_distances = [
        _float(row, "stop_distance")
        for row in rows
        if row.get("stop_distance", "") not in (None, "", "None", "nan")
    ]
    duck_present = np.asarray([_bool(row, "duck_present") for row in rows])
    duck_active = np.asarray([_bool(row, "duck_active") for row in rows])
    threats = [str(row.get("duck_threat", "") or "none") for row in rows]
    delta_v = np.diff(v_cmd)
    delta_omega = np.diff(omega)

    values = (
        np.mean(d), np.mean(np.abs(d)), np.max(np.abs(d)),
        np.mean(phi), np.mean(np.abs(phi)), np.max(np.abs(phi)),
        np.mean(tracking), np.mean(np.abs(tracking)),
        np.mean(speed), np.std(speed),
        np.mean(curvature), np.mean(np.abs(curvature)),
        _mean([value == "curve_left" for value in curve_classes]),
        _mean([value == "curve_right" for value in curve_classes]),
        _mean([value == "straight" for value in curve_classes]),
        np.mean(stop_present),
        np.mean(stop_present * (1.0 - stop_satisfied)),
        np.mean(stop_satisfied),
        _mean(stop_distances),
        np.mean(duck_present), np.mean(duck_active),
        *(_mean([value == level for value in threats]) for level in THREAT_LEVELS),
        np.mean(v_cmd), np.std(v_cmd), np.min(v_cmd), np.max(v_cmd),
        np.mean(omega), np.std(omega), np.mean(np.abs(omega)), np.max(np.abs(omega)),
        _mean(np.abs(delta_v).tolist()),
        _mean((delta_v < -1e-9).tolist()),
        _mean(np.abs(delta_omega).tolist()),
        _reversal_rate(omega),
    )
    result = np.asarray(values, dtype=np.float64)
    if result.shape != (len(FEATURE_NAMES),) or not np.all(np.isfinite(result)):
        raise ValueError("invalid M11 signature vector")
    return result


def build_signature_dataset(
    rows: Sequence[Mapping[str, str]],
    window_size: int = 5,
    drop_incomplete: bool = True,
) -> SignatureDataset:
    """Build non-overlapping, label-independent fixed windows."""
    if int(window_size) < 2:
        raise ValueError("window_size must be at least two decisions")
    assert_label_free_feature_contract(FEATURE_NAMES)
    episodes = defaultdict(list)
    for row in rows:
        episodes[str(row["episode_id"])].append(row)
    features = []
    metadata = []
    for episode_id in sorted(episodes):
        ordered = sorted(episodes[episode_id], key=lambda row: int(row["step"]))
        for offset in range(0, len(ordered), int(window_size)):
            window = ordered[offset : offset + int(window_size)]
            if len(window) < int(window_size) and drop_incomplete:
                continue
            steps = [int(row["step"]) for row in window]
            if steps != list(range(steps[0], steps[0] + len(steps))):
                raise ValueError("M11 windows require contiguous decision steps")
            features.append(_window_features(window))
            seed_text = episode_id.rsplit("_", 1)[-1]
            metadata.append({
                "episode_id": episode_id,
                "solver": str(window[0].get("solver", "unknown")),
                "seed": int(seed_text) if seed_text.isdigit() else None,
                "window_index": offset // int(window_size),
                "start_step": steps[0],
                "end_step": steps[-1],
                "duration_steps": len(window),
                "support_basis": "evaluation_reached_only",
            })
    matrix = np.vstack(features) if features else np.empty((0, len(FEATURE_NAMES)))
    return SignatureDataset(
        features=matrix,
        feature_names=FEATURE_NAMES,
        metadata=tuple(metadata),
        window_size=int(window_size),
    )


def derive_evaluation_labels(
    rows: Sequence[Mapping[str, str]],
    metadata: Sequence[Mapping[str, Any]],
) -> Tuple[Dict[str, Any], ...]:
    """Open frozen M2 labels only after signatures/clusters exist."""
    lookup = {
        (str(row["episode_id"]), int(row["step"])): str(row["primitive"])
        for row in rows
    }
    result = []
    for item in metadata:
        labels = [
            lookup[(str(item["episode_id"]), step)]
            for step in range(int(item["start_step"]), int(item["end_step"]) + 1)
        ]
        counts = Counter(labels)
        label, count = sorted(counts.items(), key=lambda pair: (-pair[1], pair[0]))[0]
        result.append({
            "primitive": label,
            "label_purity": float(count / len(labels)),
            "primitive_counts": dict(sorted(counts.items())),
        })
    return tuple(result)


def signature_manifest(dataset: SignatureDataset) -> Dict[str, Any]:
    return {
        "schema_version": dataset.schema_version,
        "segmentation": "non_overlapping_fixed_decision_windows",
        "window_size": dataset.window_size,
        "samples": int(len(dataset.features)),
        "feature_names": list(dataset.feature_names),
        "primitive_used_as_feature": False,
        "trigger_used_as_feature": False,
        "event_name_used_as_feature": False,
        "solver_used_as_feature": False,
        "m12_primitive_segments_used_as_boundaries": False,
    }
