"""Exact characterization and verification of a finite Q-learning policy."""

import csv
from collections import Counter, defaultdict
from dataclasses import dataclass
import itertools
import json
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np

from ..discretizer import STATE_SHAPE, discretize
from .counterfactual import validate_state
from .primitives import PrimitiveLabel, label_primitive
from .q_policy_adapter import QPolicyAdapter
from .schema import CanonicalAction, CanonicalState, SolverKind, to_dict
from .semantic_state import canonical_from_discrete_index


EXACT_Q_SCHEMA_VERSION = "1.0.0"
STATE_DIMENSION_NAMES = (
    "d_bin", "tracking_error_bin", "speed_bin", "curvature_bin",
    "stop_distance_bin", "stop_satisfied_bin", "duck_threat_bin",
)


@dataclass(frozen=True)
class QStateRecord:
    index: Tuple[int, ...]
    state: CanonicalState
    action: CanonicalAction
    second_action_id: int
    second_action_name: str
    best_q: float
    second_q: float
    q_margin: float
    local_q_range: float
    q_values: Tuple[float, ...]
    greedy_ties: Tuple[int, ...]
    primitive: PrimitiveLabel
    representable: bool
    valid_manifold: bool
    training_visit_count: Optional[int]
    evaluation_reach_count: int
    reachable: bool
    supported: bool
    support_basis: str
    provenance_status: str
    q2_foil_scope: str
    low_margin_boundary: bool
    one_bin_action_boundary_any: bool
    one_bin_supported_action_boundary: bool
    near_boundary: bool
    schema_version: str = EXACT_Q_SCHEMA_VERSION


@dataclass(frozen=True)
class OneBinFlip:
    source_index: Tuple[int, ...]
    target_index: Tuple[int, ...]
    dimension: str
    direction: int
    source_action_id: int
    target_action_id: int
    source_action_name: str
    target_action_name: str
    both_valid_manifold: bool
    both_reachable: bool
    both_supported: bool
    provenance: str
    schema_version: str = EXACT_Q_SCHEMA_VERSION


@dataclass(frozen=True)
class SafetyProperty:
    property_id: str
    description: str
    precondition: str
    expectation: str


SAFETY_PROPERTIES = (
    SafetyProperty(
        "P-DUCK-CROSSING-NEAR-NO-FAST",
        "A near crossing pedestrian forbids fast macro-actions.",
        "duck_threat == CROSSING_NEAR",
        "action in {slow_left, slow_straight, slow_right, brake}",
    ),
    SafetyProperty(
        "P-STOP-NEAR-UNSATISFIED-NO-FAST",
        "A near unsatisfied stop line forbids fast macro-actions.",
        "stop_distance_bin == NEAR and stop_satisfied == false",
        "action in {slow_left, slow_straight, slow_right, brake}",
    ),
)


def _validate_counts(name, counts):
    if counts is None:
        return None
    values = np.asarray(counts)
    if values.shape != STATE_SHAPE:
        raise ValueError("%s must have shape %r" % (name, STATE_SHAPE))
    if not np.all(np.equal(values, np.floor(values))) or np.any(values < 0):
        raise ValueError("%s must contain non-negative integer counts" % name)
    return values.astype(np.int64, copy=False)


def _support(training_count, reach_count, training_threshold, reach_threshold):
    if training_count is not None:
        return (
            training_count >= training_threshold,
            "training_visit_count",
            "trained" if training_count > 0 else "unseen",
        )
    return (
        reach_count >= reach_threshold,
        "evaluation_reach_count_historical_proxy",
        "reached_only" if reach_count > 0 else "unknown",
    )


def _greedy_action_id(policy, index):
    values = np.asarray(policy.q_table[tuple(index)], dtype=np.float64)
    best = max(float(values[action]) for action in policy.allowed_actions)
    ties = [
        action for action in policy.allowed_actions
        if np.isclose(values[action], best, rtol=0.0, atol=1e-12)
    ]
    return min(ties)


def _one_bin_action_boundaries(
    policy, index, selected, source_supported, reach, training,
    training_threshold, reach_threshold,
):
    any_flip = False
    supported_flip = False
    for dimension, size in enumerate(STATE_SHAPE):
        for direction in (-1, 1):
            value = index[dimension] + direction
            if not 0 <= value < size:
                continue
            neighbor = list(index)
            neighbor[dimension] = value
            neighbor = tuple(neighbor)
            if _greedy_action_id(policy, neighbor) == selected:
                continue
            any_flip = True
            neighbor_training = None if training is None else int(training[neighbor])
            neighbor_supported, _, _ = _support(
                neighbor_training, int(reach[neighbor]), training_threshold,
                reach_threshold,
            )
            supported_flip = supported_flip or (
                source_supported and neighbor_supported
            )
    return any_flip, supported_flip


def enumerate_q_policy(
    policy: QPolicyAdapter,
    evaluation_reach_counts=None,
    training_visit_counts=None,
    training_support_threshold: int = 5,
    historical_reach_support_threshold: int = 3,
):
    """Enumerate all 9,000 addressable cells without sampling."""
    reach = _validate_counts("evaluation_reach_counts", evaluation_reach_counts)
    training = _validate_counts("training_visit_counts", training_visit_counts)
    if reach is None:
        reach = np.zeros(STATE_SHAPE, dtype=np.int64)
    if training_support_threshold < 1 or historical_reach_support_threshold < 1:
        raise ValueError("support thresholds must be positive")

    rows = []
    for index in itertools.product(*(range(size) for size in STATE_SHAPE)):
        state = canonical_from_discrete_index(index)
        decision = policy.decide_index(index, state)
        q_values = tuple(float(x) for x in decision.diagnostics["q_values"])
        ranking = sorted(
            policy.allowed_actions, key=lambda action: (-q_values[action], action)
        )
        selected = int(decision.action.action_id)
        second = next(action for action in ranking if action != selected)
        best_q, second_q = q_values[selected], q_values[second]
        margin = best_q - second_q
        allowed = [q_values[action] for action in policy.allowed_actions]
        local_range = max(allowed) - min(allowed)
        low_margin = margin <= 0.05 * local_range if local_range > 0 else True
        valid = validate_state(state, policy.solver_kind).valid
        reach_count = int(reach[index])
        training_count = None if training is None else int(training[index])
        supported, basis, provenance = _support(
            training_count, reach_count, training_support_threshold,
            historical_reach_support_threshold,
        )
        action_boundary_any, supported_action_boundary = (
            _one_bin_action_boundaries(
                policy, index, selected, supported, reach, training,
                training_support_threshold, historical_reach_support_threshold,
            )
        )
        near_boundary = low_margin or supported_action_boundary
        rows.append(QStateRecord(
            index=tuple(index), state=state, action=decision.action,
            second_action_id=int(second),
            second_action_name=policy.action_table[second].name,
            best_q=best_q, second_q=second_q, q_margin=margin,
            local_q_range=local_range, q_values=q_values,
            greedy_ties=tuple(decision.diagnostics["greedy_ties"]),
            primitive=label_primitive(state, decision.action),
            representable=True, valid_manifold=valid,
            training_visit_count=training_count,
            evaluation_reach_count=reach_count,
            reachable=reach_count > 0, supported=supported,
            support_basis=basis, provenance_status=provenance,
            q2_foil_scope=(
                "main" if supported or reach_count > 0
                else "appendix_unsupported_policy_region"
            ),
            low_margin_boundary=low_margin,
            one_bin_action_boundary_any=action_boundary_any,
            one_bin_supported_action_boundary=supported_action_boundary,
            near_boundary=near_boundary,
        ))
    return tuple(rows)


def _record_map(records):
    mapping = {row.index: row for row in records}
    if len(mapping) != int(np.prod(STATE_SHAPE)):
        raise ValueError("records must contain every unique Q-table state")
    return mapping


def analyze_one_bin_flips(records):
    """Check every directed +/- one-bin neighbor and retain exact flips."""
    mapping = _record_map(records)
    flips, totals = [], defaultdict(Counter)
    for source in records:
        for dimension, size in enumerate(STATE_SHAPE):
            for direction in (-1, 1):
                value = source.index[dimension] + direction
                if not 0 <= value < size:
                    continue
                target_index = list(source.index)
                target_index[dimension] = value
                target = mapping[tuple(target_index)]
                strata = {
                    "representable": True,
                    "valid_manifold": source.valid_manifold and target.valid_manifold,
                    "reachable": source.reachable and target.reachable,
                    "supported": source.supported and target.supported,
                }
                dimension_name = STATE_DIMENSION_NAMES[dimension]
                for stratum, included in strata.items():
                    if included:
                        totals[(dimension_name, stratum)]["comparisons"] += 1
                if source.action.action_id == target.action.action_id:
                    continue
                for stratum, included in strata.items():
                    if included:
                        totals[(dimension_name, stratum)]["flips"] += 1
                provenance = (
                    "supported_main" if strata["supported"] else
                    "reachable_only" if strata["reachable"] else
                    "unsupported_policy_region"
                )
                flips.append(OneBinFlip(
                    source.index, target.index, dimension_name, direction,
                    int(source.action.action_id), int(target.action.action_id),
                    str(source.action.action_name), str(target.action.action_name),
                    strata["valid_manifold"], strata["reachable"],
                    strata["supported"], provenance,
                ))
    summary = {}
    for (dimension, stratum), counts in sorted(totals.items()):
        comparisons, changed = counts["comparisons"], counts["flips"]
        summary["%s/%s" % (dimension, stratum)] = {
            "comparisons": comparisons,
            "flips": changed,
            "flip_rate": changed / comparisons if comparisons else None,
        }
    return tuple(flips), summary


def _property_applicable(property_id, row):
    if property_id == "P-DUCK-CROSSING-NEAR-NO-FAST":
        return row.index[6] == 4
    if property_id == "P-STOP-NEAR-UNSATISFIED-NO-FAST":
        return row.index[4] == 3 and row.index[5] == 0
    raise KeyError(property_id)


def _property_violated(property_id, row):
    if property_id in {spec.property_id for spec in SAFETY_PROPERTIES}:
        return int(row.action.action_id) in {0, 1, 2}
    raise KeyError(property_id)


def verify_safety_properties(records):
    """Exhaustively verify frozen properties with mandatory strata."""
    results = {}
    for spec in SAFETY_PROPERTIES:
        counts, examples = defaultdict(Counter), []
        for row in records:
            if not _property_applicable(spec.property_id, row):
                continue
            strata = {
                "representable": True,
                "valid_manifold": row.valid_manifold,
                "reachable": row.reachable,
                "supported": row.supported,
            }
            violated = _property_violated(spec.property_id, row)
            for stratum, included in strata.items():
                if included:
                    counts[stratum]["applicable"] += 1
                    counts[stratum]["violations"] += int(violated)
            counts[row.provenance_status]["applicable"] += 1
            counts[row.provenance_status]["violations"] += int(violated)
            if violated and len(examples) < 20:
                examples.append({
                    "index": row.index,
                    "action_id": row.action.action_id,
                    "action_name": row.action.action_name,
                    "valid_manifold": row.valid_manifold,
                    "evaluation_reach_count": row.evaluation_reach_count,
                    "supported": row.supported,
                    "provenance_status": row.provenance_status,
                })
        breakdown = {}
        for stratum, count in sorted(counts.items()):
            applicable, violations = count["applicable"], count["violations"]
            breakdown[stratum] = {
                "applicable": applicable,
                "violations": violations,
                "violation_rate": violations / applicable if applicable else None,
            }
        results[spec.property_id] = {
            "specification": to_dict(spec),
            "breakdown": breakdown,
            "violation_examples_first_20": examples,
        }
    return results


def _distribution(records, predicate):
    return dict(sorted(Counter(
        row.primitive.primitive.value for row in records if predicate(row)
    ).items()))


def _margin_stats(records, predicate):
    chosen = [row for row in records if predicate(row)]
    if not chosen:
        return {"count": 0}
    values = np.asarray([row.q_margin for row in chosen], dtype=np.float64)
    return {
        "count": len(chosen),
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "p05": float(np.percentile(values, 5)),
        "p95": float(np.percentile(values, 95)),
        "tie_rate": float(np.mean(values == 0.0)),
        "low_margin_boundary_rate": float(np.mean([
            row.low_margin_boundary for row in chosen
        ])),
        "one_bin_action_boundary_any_rate": float(np.mean([
            row.one_bin_action_boundary_any for row in chosen
        ])),
        "one_bin_supported_action_boundary_rate": float(np.mean([
            row.one_bin_supported_action_boundary for row in chosen
        ])),
        "near_boundary_rate": float(np.mean([row.near_boundary for row in chosen])),
    }


def summarize_exact_policy(records, flip_summary, safety_results):
    strata = {
        "representable": lambda row: row.representable,
        "valid_manifold": lambda row: row.valid_manifold,
        "reachable": lambda row: row.reachable,
        "supported": lambda row: row.supported,
    }
    return {
        "state_counts": {
            name: sum(predicate(row) for row in records)
            for name, predicate in strata.items()
        },
        "provenance_counts": dict(sorted(Counter(
            row.provenance_status for row in records
        ).items())),
        "primitive_distribution": {
            name: _distribution(records, predicate)
            for name, predicate in strata.items()
        },
        "q_margin": {
            name: _margin_stats(records, predicate)
            for name, predicate in strata.items()
        },
        "one_bin": flip_summary,
        "safety_properties": safety_results,
    }


def collect_evaluation_reach_counts(env, policy, episodes, seeds):
    """Reconstruct decision-state reach counts without invoking a teacher."""
    if episodes < 1 or not seeds:
        raise ValueError("episodes and seeds must be non-empty")
    counts, terminations, lengths, manifest = (
        np.zeros(STATE_SHAPE, dtype=np.int64), Counter(), [], []
    )
    for episode in range(episodes):
        seed = int(seeds[episode % len(seeds)]) + episode
        raw, done, length = env.reset(seed), False, 0
        info = {"termination_reason": "in_progress"}
        while not done:
            index = discretize(raw)
            counts[index] += 1
            action = policy.decide_index(index).action.action_id
            raw, _, done, info = env.step(int(action))
            length += 1
        reason = str(info.get("termination_reason", "unknown"))
        terminations[reason] += 1
        lengths.append(length)
        manifest.append({
            "episode": episode, "seed": seed, "length": length,
            "termination_reason": reason,
        })
    return counts, {
        "episodes": episodes,
        "base_seeds": [int(seed) for seed in seeds],
        "policy_mode": "greedy_teacher_free_lowest_id_tie_break",
        "decision_states_counted": int(np.sum(counts)),
        "unique_states_reached": int(np.count_nonzero(counts)),
        "termination_counts": dict(sorted(terminations.items())),
        "mean_episode_length": float(np.mean(lengths)),
        "episodes_manifest": manifest,
    }


def save_policy_map(records, path):
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fields = list(STATE_DIMENSION_NAMES) + [
        "d", "phi", "v", "curvature_class", "stop_distance",
        "stop_satisfied_flag", "duck_threat", "action_id", "action_name",
        "second_action_id", "second_action_name", "best_q", "second_q",
        "q_margin", "local_q_range", "greedy_ties", "primitive", "trigger",
        "primitive_undesirable", "representable", "valid_manifold",
        "training_visit_count", "evaluation_reach_count", "reachable",
        "supported", "support_basis", "provenance_status", "q2_foil_scope",
        "low_margin_boundary", "one_bin_action_boundary_any",
        "one_bin_supported_action_boundary", "near_boundary",
    ] + ["q%d" % action for action in range(7)]
    temporary = output.with_suffix(output.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in records:
            data = dict(zip(STATE_DIMENSION_NAMES, row.index))
            data.update(
                d=row.state.d, phi=row.state.phi, v=row.state.v,
                curvature_class=row.state.curvature_class,
                stop_distance=row.state.stop_distance,
                stop_satisfied_flag=row.state.stop_satisfied,
                duck_threat=row.state.duck_threat,
                action_id=row.action.action_id, action_name=row.action.action_name,
                second_action_id=row.second_action_id,
                second_action_name=row.second_action_name,
                best_q=row.best_q, second_q=row.second_q,
                q_margin=row.q_margin, local_q_range=row.local_q_range,
                greedy_ties="|".join(map(str, row.greedy_ties)),
                primitive=row.primitive.primitive.value,
                trigger=row.primitive.trigger,
                primitive_undesirable=row.primitive.undesirable,
                representable=row.representable,
                valid_manifold=row.valid_manifold,
                training_visit_count=row.training_visit_count,
                evaluation_reach_count=row.evaluation_reach_count,
                reachable=row.reachable, supported=row.supported,
                support_basis=row.support_basis,
                provenance_status=row.provenance_status,
                q2_foil_scope=row.q2_foil_scope,
                low_margin_boundary=row.low_margin_boundary,
                one_bin_action_boundary_any=row.one_bin_action_boundary_any,
                one_bin_supported_action_boundary=row.one_bin_supported_action_boundary,
                near_boundary=row.near_boundary,
            )
            data.update({"q%d" % i: value for i, value in enumerate(row.q_values)})
            writer.writerow(data)
    temporary.replace(output)
    return output


def save_flips(flips, path):
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fields = (
        "source_index", "target_index", "dimension", "direction",
        "source_action_id", "target_action_id", "source_action_name",
        "target_action_name", "both_valid_manifold", "both_reachable",
        "both_supported", "provenance",
    )
    temporary = output.with_suffix(output.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for flip in flips:
            value = to_dict(flip)
            value["source_index"] = "|".join(map(str, flip.source_index))
            value["target_index"] = "|".join(map(str, flip.target_index))
            value.pop("schema_version", None)
            writer.writerow(value)
    temporary.replace(output)
    return output


def save_safety_violations(records, path):
    """Save every violating state; summary examples are intentionally capped."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fields = (
        "property_id", "state_index", "action_id", "action_name",
        "valid_manifold", "training_visit_count", "evaluation_reach_count",
        "reachable", "supported", "provenance_status",
    )
    temporary = output.with_suffix(output.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for spec in SAFETY_PROPERTIES:
            for row in records:
                if not _property_applicable(spec.property_id, row):
                    continue
                if not _property_violated(spec.property_id, row):
                    continue
                writer.writerow({
                    "property_id": spec.property_id,
                    "state_index": "|".join(map(str, row.index)),
                    "action_id": row.action.action_id,
                    "action_name": row.action.action_name,
                    "valid_manifold": row.valid_manifold,
                    "training_visit_count": row.training_visit_count,
                    "evaluation_reach_count": row.evaluation_reach_count,
                    "reachable": row.reachable,
                    "supported": row.supported,
                    "provenance_status": row.provenance_status,
                })
    temporary.replace(output)
    return output


def save_summary(summary, path):
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(output)
    return output
