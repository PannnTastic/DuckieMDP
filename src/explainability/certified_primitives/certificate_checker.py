"""Primitive-level certification gates for frozen C-EDP clusters."""

from typing import Any, Dict, Mapping, Sequence, Tuple

import numpy as np

from .descriptor import build_descriptor
from .discovery import DiscoveryResult
from .schema import (
    CertificateStatus,
    CertifiedExplanationInstance,
    CertifiedPrimitive,
    TemporalExplanationSegment,
    stable_id,
)


RELATIONS = ("stop", "pedestrian", "curvature", "lane_symmetry")


def _verified_properties(
    instance_ids: Sequence[str],
    instance_by_id: Mapping[str, CertifiedExplanationInstance],
) -> Dict[str, Any]:
    """Aggregate only empirically eligible source-target relation pairs."""

    result = {}
    records = [
        instance_by_id[value]
        for value in instance_ids
        if value in instance_by_id
    ]
    for relation in RELATIONS:
        applicable = eligible = passed = failed = abstained = 0
        pair_strata: Dict[str, int] = {}
        for record in records:
            profile = record.verification_evidence.get(
                "verification_profile", {}
            )
            if not bool(profile.get("%s_applicable" % relation, False)):
                continue
            applicable += 1
            stratum = str(
                profile.get("%s_pair_stratum" % relation, "unknown")
            )
            pair_strata[stratum] = pair_strata.get(stratum, 0) + 1
            if bool(profile.get("%s_eligible" % relation, False)):
                eligible += 1
                passed += int(
                    bool(profile.get("%s_pass" % relation, False))
                )
                failed += int(
                    bool(profile.get("%s_fail" % relation, False))
                )
            else:
                abstained += 1
        result[relation] = {
            "applicable": applicable,
            "eligible": eligible,
            "abstained": abstained,
            "passed": passed,
            "failed": failed,
            "pass_rate": (
                float(passed) / eligible if eligible else None
            ),
            "claimed": eligible > 0,
            "pair_strata": dict(sorted(pair_strata.items())),
        }
    return result


def _representative_and_boundary(
    cluster_id: int,
    indices: np.ndarray,
    discovery: DiscoveryResult,
    segments: Sequence[TemporalExplanationSegment],
) -> Tuple[Tuple[str, ...], Tuple[str, ...]]:
    centroid = discovery.model.centroids[int(cluster_id)]
    distance = np.linalg.norm(
        discovery.scaled_features[indices] - centroid, axis=1
    )
    ordered = indices[np.argsort(distance)]
    representatives = tuple(
        segments[index].instance_ids[
            len(segments[index].instance_ids) // 2
        ]
        for index in ordered[: min(3, len(ordered))]
    )
    boundaries = tuple(
        segments[index].instance_ids[-1]
        for index in ordered[-min(3, len(ordered)):]
    )
    return representatives, boundaries


def certify_clusters(
    segments: Sequence[TemporalExplanationSegment],
    discovery: DiscoveryResult,
    instances: Sequence[CertifiedExplanationInstance],
    cluster_freeze_hash: str,
    *,
    minimum_support: int = 12,
    minimum_seeds: int = 3,
    minimum_bootstrap_ari: float = 0.70,
    maximum_coherence_ratio: float = 0.80,
    minimum_property_evidence: int = 1,
) -> Tuple[CertifiedPrimitive, ...]:
    if int(minimum_property_evidence) < 1:
        raise ValueError("minimum_property_evidence must be positive")
    instance_by_id = {item.instance_id: item for item in instances}
    primitives = []
    labels = np.asarray(discovery.labels, dtype=int)
    for cluster_id in sorted(
        set(int(value) for value in labels if int(value) >= 0)
    ):
        indices = np.flatnonzero(labels == cluster_id)
        members = [segments[index] for index in indices]
        instance_ids = tuple(dict.fromkeys(
            value for segment in members for value in segment.instance_ids
        ))
        member_records = [
            instance_by_id[value]
            for value in instance_ids
            if value in instance_by_id
        ]
        certificate_rate = (
            float(np.mean([
                record.status.value == "CERTIFIED"
                for record in member_records
            ]))
            if member_records else 0.0
        )
        seed_support = len({
            (item.solver, int(item.seed)) for item in members
        })
        solver_support = tuple(sorted({item.solver for item in members}))
        heldout_count = sum(
            discovery.split[index] == "heldout" for index in indices
        )
        heldout_rate = (
            float(heldout_count) / len(indices) if len(indices) else 0.0
        )
        properties = _verified_properties(instance_ids, instance_by_id)
        claimed = [
            data for data in properties.values() if data["claimed"]
        ]
        eligible_property_evidence = sum(
            int(data["eligible"]) for data in claimed
        )
        applicable_properties_pass = (
            bool(claimed)
            and all(data["pass_rate"] == 1.0 for data in claimed)
        )
        representatives, boundaries = _representative_and_boundary(
            cluster_id, indices, discovery, segments
        )
        descriptor = build_descriptor(
            cluster_id, members, instance_by_id, population=segments
        )
        gates = {
            "all_members_certified": certificate_rate == 1.0,
            "full_trajectory_only": all(
                item.source_kind.value == "full_trajectory"
                for item in members
            ),
            "minimum_support": len(indices) >= int(minimum_support),
            "minimum_seed_support": seed_support >= int(minimum_seeds),
            "heldout_assignment_available": heldout_count > 0,
            "bootstrap_stability": float(
                discovery.diagnostics.get("bootstrap_stability", 0.0)
            ) >= float(minimum_bootstrap_ari),
            "outcome_coherence": float(
                discovery.diagnostics.get(
                    "outcome_coherence_ratio", 1.0e9
                )
            ) <= float(maximum_coherence_ratio),
            "minimum_property_evidence": (
                eligible_property_evidence
                >= int(minimum_property_evidence)
            ),
            "claimed_properties_pass": applicable_properties_pass,
            "traceable_descriptor": bool(
                descriptor.member_certificate_ids
            ),
        }
        passed = all(gates.values())
        status = (
            CertificateStatus.CERTIFIED_PRIMITIVE
            if passed and len(solver_support) >= 2
            else CertificateStatus.SOLVER_SPECIFIC_PRIMITIVE
            if passed
            else CertificateStatus.PRIMITIVE_CANDIDATE
        )
        identity = {
            "cluster_freeze_hash": cluster_freeze_hash,
            "cluster_id": cluster_id,
            "member_segment_ids": [
                item.segment_id for item in members
            ],
        }
        primitives.append(CertifiedPrimitive(
            primitive_id=stable_id(identity, "cedp-primitive"),
            cluster_id=cluster_id,
            status=status,
            descriptor=descriptor,
            cluster_freeze_hash=cluster_freeze_hash,
            member_certificate_rate=certificate_rate,
            support=len(indices),
            seed_support=seed_support,
            solver_support=solver_support,
            heldout_assignment_rate=heldout_rate,
            bootstrap_stability=float(
                discovery.diagnostics.get("bootstrap_stability", 0.0)
            ),
            outcome_coherence_ratio=float(
                discovery.diagnostics.get(
                    "outcome_coherence_ratio", 1.0e9
                )
            ),
            verified_properties=properties,
            boundary_cases=boundaries,
            representative_certificates=representatives,
            gate_results=gates,
        ))
    return tuple(primitives)
