"""Frozen, support-aware runtime primitive assignment."""

from dataclasses import dataclass
from typing import Any, Mapping, Optional, Sequence

import numpy as np

from .discovery import FrozenDiscoveryModel
from .schema import (
    CertificateStatus,
    CertifiedPrimitive,
    TemporalExplanationSegment,
)


@dataclass(frozen=True)
class RuntimeAssignment:
    status: CertificateStatus
    primitive_id: Optional[str]
    functional_name: str
    cluster_id: Optional[int]
    support_distance: Optional[float]
    support_radius: Optional[float]
    explanation: Mapping[str, Any]


class CertifiedPrimitiveAssigner:
    def __init__(
        self,
        model: FrozenDiscoveryModel,
        primitives: Sequence[CertifiedPrimitive],
        radius_multiplier: float = 1.25,
    ) -> None:
        self.model = model
        self.primitives = {item.cluster_id: item for item in primitives}
        self.radius_multiplier = float(radius_multiplier)

    def assign(self, segment: TemporalExplanationSegment) -> RuntimeAssignment:
        if segment.feature_names != self.model.feature_names:
            return self._unknown("feature_schema_mismatch")
        if segment.source_kind.value != "full_trajectory":
            return self._unknown("non_main_source_kind")
        projected = self.model.scaler.transform(
            np.asarray([segment.feature_values], dtype=np.float64)
        )
        reducer = getattr(self.model, "reducer", None)
        if reducer is not None:
            projected = reducer.transform(projected)
        scaled = projected[0]
        if not self.model.centroids:
            return self._unknown("no_frozen_cluster_support")
        distances = {
            cluster_id: float(np.linalg.norm(scaled - centroid))
            for cluster_id, centroid in self.model.centroids.items()
        }
        best = min(distances, key=distances.get)
        radius = float(self.model.radii[best]) * self.radius_multiplier
        distance = distances[best]
        if distance > radius:
            return self._unknown(
                "outside_support_radius", distance=distance, radius=radius
            )
        primitive = self.primitives.get(best)
        if primitive is None or primitive.status not in {
            CertificateStatus.CERTIFIED_PRIMITIVE,
            CertificateStatus.SOLVER_SPECIFIC_PRIMITIVE,
        }:
            return self._unknown(
                "cluster_not_certified", distance=distance, radius=radius
            )
        descriptor = primitive.descriptor
        return RuntimeAssignment(
            status=primitive.status,
            primitive_id=primitive.primitive_id,
            functional_name=descriptor.functional_name,
            cluster_id=best,
            support_distance=distance,
            support_radius=radius,
            explanation={
                "why": descriptor.decision_summary,
                "what_if": descriptor.outcome_summary,
                "verification": descriptor.verification_summary,
                "temporal_role": descriptor.temporal_summary,
                "evidence_ids": list(primitive.representative_certificates),
            },
        )

    @staticmethod
    def _unknown(
        reason: str,
        distance: Optional[float] = None,
        radius: Optional[float] = None,
    ) -> RuntimeAssignment:
        return RuntimeAssignment(
            status=CertificateStatus.UNKNOWN,
            primitive_id=None,
            functional_name="Unknown",
            cluster_id=None,
            support_distance=distance,
            support_radius=radius,
            explanation={"reason": reason},
        )
