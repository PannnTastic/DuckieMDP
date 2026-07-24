"""Deterministic explanation change-point segmentation and baselines."""

from dataclasses import dataclass
from typing import Iterable, List, Mapping, Sequence, Tuple

import numpy as np

from .schema import SourceKind, TemporalExplanationSegment, stable_id
from .signature import SignatureTrajectory


AGGREGATES = ("mean", "std", "slope", "delta", "min", "max")


@dataclass(frozen=True)
class SegmentationModel:
    method: str
    distance_threshold: float
    minimum_duration: int
    feature_center: Tuple[float, ...]
    feature_scale: Tuple[float, ...]


def fit_change_point_model(
    trajectories: Sequence[SignatureTrajectory],
    *,
    quantile: float = 0.85,
    minimum_duration: int = 3,
) -> SegmentationModel:
    if not trajectories:
        raise ValueError("cannot fit segmentation without trajectories")
    stacked = np.vstack([item.matrix for item in trajectories])
    center = np.median(stacked, axis=0)
    scale = np.median(np.abs(stacked - center), axis=0) * 1.4826
    scale[scale < 1e-9] = 1.0
    distances = []
    for item in trajectories:
        normalized = (item.matrix - center) / scale
        if len(normalized) > 1:
            distances.extend(np.linalg.norm(np.diff(normalized, axis=0), axis=1))
    threshold = float(np.quantile(distances, quantile)) if distances else float("inf")
    return SegmentationModel(
        method="robust_explanation_distance",
        distance_threshold=threshold,
        minimum_duration=max(1, int(minimum_duration)),
        feature_center=tuple(float(value) for value in center),
        feature_scale=tuple(float(value) for value in scale),
    )


def change_points(signature: SignatureTrajectory, model: SegmentationModel) -> Tuple[int, ...]:
    matrix = signature.matrix
    if len(matrix) <= 1:
        return (0, len(matrix))
    center = np.asarray(model.feature_center, dtype=np.float64)
    scale = np.asarray(model.feature_scale, dtype=np.float64)
    normalized = (matrix - center) / scale
    distance = np.linalg.norm(np.diff(normalized, axis=0), axis=1)
    boundaries = [0]
    last = 0
    for index, value in enumerate(distance, start=1):
        if (
            value >= model.distance_threshold
            and index - last >= model.minimum_duration
            and len(matrix) - index >= model.minimum_duration
        ):
            boundaries.append(index)
            last = index
    boundaries.append(len(matrix))
    return tuple(boundaries)


def fixed_window_points(length: int, window: int) -> Tuple[int, ...]:
    if window <= 0:
        raise ValueError("window must be positive")
    points = list(range(0, int(length), int(window)))
    if not points or points[-1] != int(length):
        points.append(int(length))
    return tuple(points)


def _slope(matrix: np.ndarray) -> np.ndarray:
    if len(matrix) <= 1:
        return np.zeros(matrix.shape[1], dtype=np.float64)
    x = np.arange(len(matrix), dtype=np.float64)
    centered = x - x.mean()
    denominator = float(np.dot(centered, centered))
    return np.dot(centered, matrix - matrix.mean(axis=0)) / denominator


def aggregate_segment(
    signature: SignatureTrajectory,
    start: int,
    stop: int,
) -> TemporalExplanationSegment:
    if not 0 <= start < stop <= len(signature.matrix):
        raise ValueError("invalid segment bounds")
    matrix = signature.matrix[start:stop]
    arrays = (
        matrix.mean(axis=0),
        matrix.std(axis=0),
        _slope(matrix),
        matrix[-1] - matrix[0],
        matrix.min(axis=0),
        matrix.max(axis=0),
    )
    names = tuple(
        "%s__%s" % (statistic, name)
        for statistic in AGGREGATES
        for name in signature.feature_names
    ) + ("duration", "certificate_coverage")
    records = signature.trajectory.instances[start:stop]
    coverage = float(np.mean([
        item.status.value == "CERTIFIED" for item in records
    ]))
    values = tuple(float(value) for array in arrays for value in array) + (
        float(len(records)),
        coverage,
    )
    identity = {
        "trajectory_id": signature.trajectory.trajectory_id,
        "start": records[0].step_index,
        "end": records[-1].step_index,
        "instance_ids": [item.instance_id for item in records],
    }
    return TemporalExplanationSegment(
        segment_id=stable_id(identity, "cedp-segment"),
        trajectory_id=signature.trajectory.trajectory_id,
        solver=signature.trajectory.solver,
        seed=signature.trajectory.seed,
        episode_id=signature.trajectory.episode_id,
        source_kind=signature.trajectory.source_kind,
        start_step=records[0].step_index,
        end_step=records[-1].step_index,
        instance_ids=tuple(item.instance_id for item in records),
        feature_names=names,
        feature_values=values,
        certificate_coverage=coverage,
    )


def segment_trajectory(
    signature: SignatureTrajectory,
    model: SegmentationModel,
    *,
    method: str = "change_point",
    fixed_window: int = 3,
) -> Tuple[TemporalExplanationSegment, ...]:
    boundaries = (
        change_points(signature, model)
        if method == "change_point"
        else fixed_window_points(len(signature.matrix), fixed_window)
    )
    return tuple(
        aggregate_segment(signature, start, stop)
        for start, stop in zip(boundaries, boundaries[1:])
        if stop > start
    )


def segment_all(
    signatures: Iterable[SignatureTrajectory],
    model: SegmentationModel,
    *,
    method: str = "change_point",
    fixed_window: int = 3,
) -> Tuple[TemporalExplanationSegment, ...]:
    return tuple(
        segment
        for signature in signatures
        for segment in segment_trajectory(
            signature, model, method=method, fixed_window=fixed_window
        )
    )
