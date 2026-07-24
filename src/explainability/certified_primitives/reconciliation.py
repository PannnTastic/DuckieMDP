"""Post-freeze external reconciliation with the sealed M2 taxonomy."""

from typing import Mapping, Sequence

import numpy as np

from ..eddp.reconcile import reconcile
from .schema import TemporalExplanationSegment


def reconcile_after_freeze(
    cluster_labels: Sequence[int],
    m2_labels: Sequence[str],
    segments: Sequence[TemporalExplanationSegment],
    split: Sequence[str],
    *,
    cluster_frozen: bool,
) -> Mapping[str, object]:
    if not cluster_frozen:
        raise ValueError("M2 reconciliation is forbidden before cluster freeze")
    if not (len(cluster_labels) == len(m2_labels) == len(segments) == len(split)):
        raise ValueError("reconciliation arrays differ in length")
    metadata = [
        {"solver": item.solver, "seed": item.seed} for item in segments
    ]
    return reconcile(
        np.asarray(cluster_labels, dtype=int),
        np.asarray(m2_labels, dtype=object),
        metadata,
        np.asarray(split),
    )
