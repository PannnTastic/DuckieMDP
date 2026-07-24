"""Evidence-grounded functional descriptors for discovered primitives."""

from typing import Dict, Mapping, Optional, Sequence, Tuple

import numpy as np

from .schema import (
    CertifiedExplanationInstance,
    PrimitiveDescriptor,
    TemporalExplanationSegment,
)


def _mean_feature(
    segments: Sequence[TemporalExplanationSegment], name: str
) -> float:
    if not segments:
        return 0.0
    try:
        index = segments[0].feature_names.index(name)
    except ValueError:
        return 0.0
    return float(np.mean([item.feature_values[index] for item in segments]))


def _dominant_evidence(
    segments: Sequence[TemporalExplanationSegment], prefix: str, limit: int = 5
) -> Tuple[Tuple[str, float], ...]:
    if not segments:
        return ()
    names = segments[0].feature_names
    values = np.mean(
        np.asarray([item.feature_values for item in segments], dtype=np.float64),
        axis=0,
    )
    candidates = [
        (name, float(values[index]))
        for index, name in enumerate(names)
        if name.startswith(prefix)
    ]
    return tuple(sorted(candidates, key=lambda item: abs(item[1]), reverse=True)[:limit])


MINIMUM_APPLICABILITY = 0.05
AMBIGUITY_MARGIN = 1.2
MINIMUM_POPULATION = 10
Z_NAMING_MARGIN = 0.25


def _gated_score(
    segments: Sequence[TemporalExplanationSegment],
    gate_feature: str,
    components: Sequence[float],
    *,
    gated: bool = True,
) -> float:
    """Mean of evidence components, optionally applicability-gated.

    The mean keeps domains comparable regardless of how many evidence
    features each one has. The verification-relation gate is only meaningful
    for the small-population ratio fallback: relation preconditions do not
    track behaviour phases (a held stop is outside the stop relation's
    precondition), so the population z-score path scores ungated and lets
    the corpus baseline discount ubiquitous evidence instead.
    """

    if not components:
        return 0.0
    if gated:
        gate = max(0.0, min(1.0, _mean_feature(segments, gate_feature)))
        if gate < MINIMUM_APPLICABILITY:
            return 0.0
        return gate * float(np.mean(components))
    return float(np.mean(components))


def _domain_scores(
    segments: Sequence[TemporalExplanationSegment],
    *,
    gated: bool = True,
) -> Dict[str, float]:
    return {
        "StopCompliance": _gated_score(
            segments,
            "mean__verification__stop_applicable",
            (
                abs(_mean_feature(
                    segments, "mean__decision__stop_distance_flip"
                )),
                abs(_mean_feature(
                    segments, "mean__decision__stop_satisfied_flip"
                )),
                max(0.0, _mean_feature(
                    segments, "mean__outcome__delta_stop_violations"
                )),
                min(1.0, max(0.0, _mean_feature(
                    segments, "mean__outcome__factual_full_stops"
                ))),
            ),
            gated=gated,
        ),
        "PedestrianYield": _gated_score(
            segments,
            "mean__verification__pedestrian_applicable",
            (
                abs(_mean_feature(segments, "mean__decision__duck_risk_flip")),
                max(0.0, -_mean_feature(
                    segments, "mean__outcome__delta_minimum_duck_clearance_m"
                )),
            ),
            gated=gated,
        ),
        "CurveFollowing": _gated_score(
            segments,
            "mean__verification__curvature_applicable",
            (
                abs(_mean_feature(segments, "mean__decision__curvature_flip")),
                max(0.0, _mean_feature(
                    segments, "mean__outcome__delta_max_abs_lateral_error_m"
                )),
            ),
            gated=gated,
        ),
        "LaneRecovery": _gated_score(
            segments,
            "mean__verification__lane_symmetry_applicable",
            (
                abs(_mean_feature(segments, "mean__decision__lateral_flip")),
                abs(_mean_feature(segments, "mean__decision__heading_flip")),
                max(0.0, _mean_feature(
                    segments, "mean__outcome__delta_max_abs_heading_error_rad"
                )),
            ),
            gated=gated,
        ),
    }


def _select_by_ratio(scores: Mapping[str, float]) -> str:
    ranked = sorted(scores, key=scores.get, reverse=True)
    if scores[ranked[0]] <= 1e-8:
        return "ProgressRegulation"
    if scores[ranked[1]] > 1e-8 and (
        scores[ranked[0]] < AMBIGUITY_MARGIN * scores[ranked[1]]
    ):
        return "MixedBehavior"
    return ranked[0]


def _select_by_population_z(
    scores: Mapping[str, float],
    population: Sequence[TemporalExplanationSegment],
) -> str:
    """Name by the domain whose evidence is distinctive against the corpus.

    Absolute evidence levels are dominated by environment base rates (for
    example a Duckie exists in almost every state), so a cluster is named
    after the domain where it stands out from the segment population, in
    standard-deviation units.
    """

    per_segment = [
        _domain_scores((segment,), gated=False) for segment in population
    ]
    z_scores: Dict[str, float] = {}
    for domain, value in scores.items():
        column = np.asarray(
            [row[domain] for row in per_segment], dtype=np.float64
        )
        spread = float(column.std())
        if value <= 1e-8 or spread <= 1e-9:
            z_scores[domain] = float("-inf")
            continue
        z_scores[domain] = (value - float(column.mean())) / spread
    ranked = sorted(z_scores, key=z_scores.get, reverse=True)
    top, second = z_scores[ranked[0]], z_scores[ranked[1]]
    if not np.isfinite(top) or top < Z_NAMING_MARGIN:
        return "ProgressRegulation"
    if np.isfinite(second) and top - second < Z_NAMING_MARGIN:
        return "MixedBehavior"
    return ranked[0]


def _explanation_derived_name(
    segments: Sequence[TemporalExplanationSegment],
) -> str:
    """Name from the explanation signature only — verification applicability,
    counterfactual flips, and factual outcome — never raw state, action, or
    context. These are the transparent, representation-aware rules a shallow
    tree recovered from explanation-only features (family recoverable at ~0.95
    balanced accuracy), so the label is genuinely derived from the explanation
    it summarises.
    """
    lane_symmetry = _mean_feature(
        segments, "mean__verification__lane_symmetry_applicable"
    )
    stop_satisfied_flip = _mean_feature(
        segments, "mean__decision__stop_satisfied_flip"
    )
    duck_available = _mean_feature(
        segments, "mean__outcome__factual_minimum_duck_clearance_available"
    )
    duck_clearance = _mean_feature(
        segments, "mean__outcome__factual_minimum_duck_clearance_m"
    )
    pedestrian_applicable = _mean_feature(
        segments, "mean__verification__pedestrian_applicable"
    )
    # Lane symmetry applies only on a straight lane => keeping/correcting the lane.
    if lane_symmetry > 0.5:
        return "LaneKeeping"
    # PedestrianYield is representation-aware. A continuous-action policy exposes a
    # metric near-miss clearance to a Duckie; a tabular policy has only a categorical
    # duck and yields by *stopping*, so its yield surfaces as a stop whose pedestrian
    # metamorphic relation is co-active (the Duckie, not a stop-line, is the trigger).
    # This recovers pedestrian yielding across both policy classes without keying on
    # raw state — the discriminator is the verification relation, not the observation.
    metric_near_yield = duck_available > 0.5 and duck_clearance <= 0.40
    duck_triggered_stop = stop_satisfied_flip > 0.5 and pedestrian_applicable > 0.5
    if metric_near_yield or duck_triggered_stop:
        return "PedestrianYield"
    # A stop with no co-active pedestrian relation => a stop-line obligation.
    if stop_satisfied_flip > 0.5:
        return "StopCompliance"
    # Sustained steering that is neither lane-symmetric, a stop, nor a Duckie yield.
    return "CurveNegotiation"


def build_descriptor(
    cluster_id: int,
    segments: Sequence[TemporalExplanationSegment],
    instance_by_id: Mapping[str, CertifiedExplanationInstance],
    population: Optional[Sequence[TemporalExplanationSegment]] = None,
) -> PrimitiveDescriptor:
    """Translate computed explanation evidence, never context/action labels."""

    functional = _explanation_derived_name(segments)

    decision_top = _dominant_evidence(segments, "mean__decision__")
    outcome_top = _dominant_evidence(segments, "mean__outcome__")
    verification_top = tuple(
        item for item in _dominant_evidence(
            segments, "mean__verification__", limit=12
        ) if item[0].endswith("_pass") and item[1] > 0.0
    )[:4]
    temporal_top = _dominant_evidence(segments, "mean__temporal__", limit=4)
    references = tuple(dict.fromkeys(
        instance_id
        for segment in segments
        for instance_id in segment.instance_ids
        if instance_id in instance_by_id
    ))
    evidence_names = tuple(dict.fromkeys(
        name for name, _ in decision_top + outcome_top + verification_top + temporal_top
    ))

    def render(items):
        return ", ".join("%s=%+.3f" % item for item in items) or "no dominant evidence"

    return PrimitiveDescriptor(
        functional_name="%s_C%02d" % (functional, int(cluster_id)),
        decision_summary="Decision boundary evidence: " + render(decision_top),
        outcome_summary="Factual-versus-foil evidence: " + render(outcome_top),
        verification_summary="Verified relation evidence: " + render(verification_top),
        temporal_summary="Temporal explanation evidence: " + render(temporal_top),
        evidence_feature_names=evidence_names,
        member_certificate_ids=references,
    )
