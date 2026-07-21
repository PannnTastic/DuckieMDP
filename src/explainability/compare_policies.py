"""Unified behavioral comparison of frozen Q-learning and SAC policies."""

from collections import Counter, defaultdict
from statistics import mean, median

import numpy as np


COMPARISON_SCHEMA_VERSION = "1.0.0"


def primitive_frequency(records):
    counts = Counter(
        step.primitive.primitive.value
        for record in records
        for step in record.steps
    )
    total = sum(counts.values())
    return {
        name: {"count": count, "rate": count / total if total else 0.0}
        for name, count in sorted(counts.items())
    }


def primitive_transitions(records):
    counts = Counter()
    for record in records:
        for left, right in zip(record.segments, record.segments[1:]):
            counts[(left.primitive.value, right.primitive.value)] += 1
    return tuple(
        {
            "source": source,
            "target": target,
            "count": count,
        }
        for (source, target), count in sorted(counts.items())
    )


def primitive_durations(records):
    values = defaultdict(list)
    for record in records:
        for segment in record.segments:
            values[segment.primitive.value].append(int(segment.duration_steps))
    result = {}
    for primitive, durations in sorted(values.items()):
        result[primitive] = {
            "segments": len(durations),
            "mean_steps": float(mean(durations)),
            "median_steps": float(median(durations)),
            "min_steps": min(durations),
            "max_steps": max(durations),
        }
    return result


def _safe_correlation(left, right):
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    if len(left) < 2 or np.std(left) <= 1e-12 or np.std(right) <= 1e-12:
        return None
    return float(np.corrcoef(left, right)[0, 1])


def _first_stop_brake_distances(record, hold_threshold):
    distances = []
    obligation_active = False
    brake_recorded = False
    for step in record.steps:
        state = step.decision.state
        action = step.decision.action
        obligation = (
            state.stop_present
            and not state.stop_satisfied
            and state.stop_distance is not None
        )
        if obligation and not obligation_active:
            brake_recorded = False
        if obligation and not brake_recorded and action.v_cmd <= hold_threshold:
            distances.append(float(state.stop_distance))
            brake_recorded = True
        obligation_active = obligation
    return distances


def _is_straight_state(state, curvature_threshold):
    """Use each solver's canonical road representation consistently."""
    if state.curvature is not None:
        return abs(float(state.curvature)) <= curvature_threshold
    return state.curvature_class == "straight"


def summarize_policy(records, hold_threshold=0.04, straight_curvature=0.05):
    if not records:
        raise ValueError("comparison requires at least one trajectory")
    steps = [step for record in records for step in record.steps]
    episodes = len(records)
    full_stops = sum(int(step.events.get("full_stop", False)) for step in steps)
    stop_violations = sum(int(step.events.get("stop_violation", False)) for step in steps)
    active_duck_steps = [
        step for step in steps
        if step.decision.state.duck_present
        and step.decision.state.duck_active is True
    ]
    duck_yield_steps = sum(
        step.decision.action.v_cmd <= hold_threshold for step in active_duck_steps
    )
    unnecessary = sum(
        step.primitive.primitive.value == "UnnecessaryBrake" for step in steps
    )
    unsafe = sum(
        step.primitive.primitive.value == "UnsafeProceed" for step in steps
    )
    undesirable = sum(step.primitive.undesirable for step in steps)
    braking_distances = [
        distance
        for record in records
        for distance in _first_stop_brake_distances(record, hold_threshold)
    ]
    straight_steps = [
        step for step in steps
        if _is_straight_state(step.decision.state, straight_curvature)
    ]
    lane_errors = [
        step.decision.state.phi + step.decision.state.d
        for step in straight_steps
    ]
    omegas = [step.decision.action.omega_cmd for step in straight_steps]
    reasons = Counter(record.termination_reason for record in records)
    stop_opportunities = full_stops + stop_violations
    return {
        "schema_version": COMPARISON_SCHEMA_VERSION,
        "solver": records[0].solver.value,
        "policy_mode": records[0].policy_mode.value,
        "episodes": episodes,
        "decision_steps": len(steps),
        "mean_return": float(mean(record.total_reward for record in records)),
        "termination_counts": dict(sorted(reasons.items())),
        "primitive_frequency": primitive_frequency(records),
        "primitive_transitions": primitive_transitions(records),
        "primitive_durations": primitive_durations(records),
        "undesirable_primitive_rate": undesirable / len(steps),
        "unnecessary_brake_rate": unnecessary / len(steps),
        "unsafe_proceed_rate": unsafe / len(steps),
        "stop_compliance_rate": (
            full_stops / stop_opportunities if stop_opportunities else 1.0
        ),
        "stop_opportunities": stop_opportunities,
        "stop_violations": stop_violations,
        "pedestrian_yield_command_rate": (
            duck_yield_steps / len(active_duck_steps) if active_duck_steps else 1.0
        ),
        "pedestrian_active_steps": len(active_duck_steps),
        "first_stop_brake_distance": {
            "observations": len(braking_distances),
            "mean_m": None if not braking_distances else float(mean(braking_distances)),
            "median_m": None if not braking_distances else float(median(braking_distances)),
            "min_m": None if not braking_distances else float(min(braking_distances)),
            "max_m": None if not braking_distances else float(max(braking_distances)),
        },
        "steering_response": {
            "straight_steps": len(straight_steps),
            "mean_abs_omega": (
                None if not omegas else float(np.mean(np.abs(omegas)))
            ),
            "lane_error_omega_correlation": _safe_correlation(lane_errors, omegas),
        },
    }


def q_supported_influence_signature(m8_summary):
    one_bin = m8_summary["exact_characterization"]["one_bin"]
    dimension_to_concept = {
        "d_bin": "lane",
        "tracking_error_bin": "lane_heading_entangled",
        "speed_bin": "speed",
        "curvature_bin": "road",
        "duck_threat_bin": "pedestrian",
    }
    values = {
        concept: float(one_bin[dimension + "/supported"]["flip_rate"])
        for dimension, concept in dimension_to_concept.items()
    }
    stop_rows = [
        one_bin["stop_distance_bin/supported"],
        one_bin["stop_satisfied_bin/supported"],
    ]
    comparisons = sum(row["comparisons"] for row in stop_rows)
    values["stop"] = (
        sum(row["flips"] for row in stop_rows) / comparisons
        if comparisons else 0.0
    )
    scale = sum(values.values())
    return {
        "method": "supported_one_bin_action_flip_rate",
        "raw": values,
        "normalized_l1": {
            name: value / scale if scale else 0.0
            for name, value in values.items()
        },
        "caveat": "tracking_error_bin is phi+d and remains lane-heading-entangled",
    }


def sac_ig_influence_signature(m9_summary):
    totals = Counter()
    by_anchor = {}
    for anchor, record in m9_summary["integrated_gradients"].items():
        outputs = record["neutral"]["concept_absolute"]
        anchor_totals = Counter()
        for output in outputs.values():
            anchor_totals.update({name: float(value) for name, value in output.items()})
        by_anchor[anchor] = dict(anchor_totals)
        totals.update(anchor_totals)
    scale = sum(totals.values())
    return {
        "method": "neutral_baseline_absolute_integrated_gradients",
        "raw": dict(totals),
        "normalized_l1": {
            name: value / scale if scale else 0.0
            for name, value in sorted(totals.items())
        },
        "by_anchor": by_anchor,
        "caveat": "three critical anchors only; baseline-sensitive claims remain governed by M9",
    }


def comparable_influence_subset(q_signature, sac_signature):
    concepts = ("speed", "road", "stop", "pedestrian")
    q_values = {name: q_signature["raw"].get(name, 0.0) for name in concepts}
    sac_values = {name: sac_signature["raw"].get(name, 0.0) for name in concepts}

    def normalize(values):
        scale = sum(values.values())
        return {name: value / scale if scale else 0.0 for name, value in values.items()}

    return {
        "concepts": list(concepts),
        "q_learning_normalized": normalize(q_values),
        "sac_normalized": normalize(sac_values),
        "excluded": ["lane", "heading", "lane_heading_entangled"],
        "reason": "Q tracking uses phi+d, so pure lane/heading influence is not comparable",
    }
