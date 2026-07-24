"""Build solver-neutral explanation signatures and temporal segments."""

from dataclasses import dataclass
import re
from typing import Any, Dict, Mapping, Sequence, Tuple

import numpy as np

from .counterfactual_profile import CONCEPTS
from .schema import ExplanationAtom


FORBIDDEN_TOKENS = (
    "solver", "policy", "checkpoint", "seed", "episode", "scenario",
    "action", "primitive", "trigger", "label", "q_value", "q_margin",
    "integrated_gradient", "critic", "natural_language",
)


COUNTERFACTUAL_FEATURES = tuple(
    name
    for concept in CONCEPTS
    for name in (
        "%s_flip" % concept,
        "%s_abs_delta" % concept,
        "%s_signed_delta" % concept,
    )
) + ("any_flip", "minimum_flip_distance")

PHYSICAL_SCALARS = (
    "steps", "elapsed_seconds", "forward_progress_m",
    "mean_abs_lateral_error_m", "max_abs_lateral_error_m",
    "mean_abs_heading_error_rad", "max_abs_heading_error_rad",
    "minimum_duck_clearance_m", "minimum_duck_clearance_available",
    "stop_violations", "full_stops", "passed_stops", "brake_steps",
    "brake_ratio", "steering_reversals", "cumulative_steering_jerk",
    "lane_departure", "duck_collision", "other_collision",
)

PHYSICAL_FEATURES = tuple(
    "%s_%s" % (role, name)
    for role in ("factual", "foil")
    for name in PHYSICAL_SCALARS
) + tuple(
    "delta_%s" % name
    for name in (
        "forward_progress_m", "mean_abs_lateral_error_m",
        "max_abs_lateral_error_m", "mean_abs_heading_error_rad",
        "max_abs_heading_error_rad", "minimum_duck_clearance_m",
        "stop_violations", "full_stops", "brake_ratio",
        "steering_reversals", "cumulative_steering_jerk",
    )
) + (
    "factual_safe", "foil_safe", "factual_safer_than_foil",
    "termination_changed",
)

VERIFICATION_FEATURES = tuple(
    "%s_%s" % (relation, suffix)
    for relation in ("stop", "pedestrian", "curvature", "lane_symmetry")
    for suffix in ("applicable", "pass", "fail")
)

VALIDITY_FEATURES = (
    "counterfactual_valid_fraction", "branch_invariants_pass",
    "paired_outcome_valid",
)

ATOM_FEATURE_NAMES = tuple(
    "counterfactual__%s" % name for name in COUNTERFACTUAL_FEATURES
) + tuple(
    "physical__%s" % name for name in PHYSICAL_FEATURES
) + tuple(
    "verification__%s" % name for name in VERIFICATION_FEATURES
) + tuple(
    "validity__%s" % name for name in VALIDITY_FEATURES
)


def assert_label_free_feature_contract(names: Sequence[str]) -> None:
    violations = []
    for name in names:
        lowered = str(name).lower()
        tokens = set(re.split(r"[^a-z0-9]+", lowered))
        for token in FORBIDDEN_TOKENS:
            exact_phrase = token in lowered if "_" in token else False
            exact_token = "_" not in token and token in tokens
            if exact_phrase or exact_token:
                violations.append((name, token))
    if violations:
        raise ValueError("EDDP label/solver leakage: %r" % violations)


def _number(value: Any) -> float:
    if value is None:
        return 0.0
    if isinstance(value, bool):
        return float(value)
    number = float(value)
    if not np.isfinite(number):
        raise ValueError("non-finite EDDP feature")
    return number


def atom_feature_vector(atom: ExplanationAtom) -> np.ndarray:
    values = []
    for name in COUNTERFACTUAL_FEATURES:
        values.append(_number(atom.counterfactual_profile.get(name, 0.0)))
    for name in PHYSICAL_FEATURES:
        values.append(_number(atom.physical_profile.get(name, 0.0)))
    for name in VERIFICATION_FEATURES:
        values.append(_number(atom.verification_profile.get(name, 0.0)))
    for name in VALIDITY_FEATURES:
        values.append(_number(atom.validity.get(name, 0.0)))
    result = np.asarray(values, dtype=np.float64)
    if result.shape != (len(ATOM_FEATURE_NAMES),):
        raise AssertionError("wrong atom feature shape")
    return result


@dataclass(frozen=True)
class SegmentDataset:
    feature_names: Tuple[str, ...]
    features: np.ndarray
    metadata: Tuple[Mapping[str, Any], ...]
    atom_ids: Tuple[Tuple[str, ...], ...]


def build_segment_dataset(
    atoms: Sequence[ExplanationAtom], minimum_atoms: int = 2
) -> SegmentDataset:
    """Aggregate real consecutive block atoms without M2 segment boundaries."""

    assert_label_free_feature_contract(ATOM_FEATURE_NAMES)
    grouped: Dict[str, list] = {}
    for atom in atoms:
        grouped.setdefault(atom.block_id, []).append(atom)
    feature_names = tuple(
        "%s__%s" % (statistic, name)
        for statistic in ("mean", "std", "delta")
        for name in ATOM_FEATURE_NAMES
    )
    assert_label_free_feature_contract(feature_names)
    rows = []
    metadata = []
    atom_groups = []
    for block_id, group in sorted(grouped.items()):
        ordered = sorted(group, key=lambda item: item.block_offset)
        if len(ordered) < int(minimum_atoms):
            continue
        offsets = [item.block_offset for item in ordered]
        expected_offsets = list(range(min(offsets), max(offsets) + 1))
        missing_offsets = sorted(set(expected_offsets).difference(offsets))
        matrix = np.vstack([atom_feature_vector(item) for item in ordered])
        row = np.concatenate(
            [matrix.mean(axis=0), matrix.std(axis=0), matrix[-1] - matrix[0]]
        )
        rows.append(row)
        first = ordered[0]
        metadata.append({
            "block_id": block_id,
            "solver": first.solver.value,
            "seed": first.seed,
            "episode_id": first.episode_id,
            "selection_context": first.selection_context,
            "start_step": first.decision_step,
            "end_step": ordered[-1].decision_step,
            "length": len(ordered),
            "expected_length": len(expected_offsets),
            "missing_offsets": "|".join(str(value) for value in missing_offsets),
            "complete_window": not missing_offsets,
        })
        atom_groups.append(tuple(item.atom_id for item in ordered))
    matrix = (
        np.vstack(rows).astype(np.float64)
        if rows else np.empty((0, len(feature_names)), dtype=np.float64)
    )
    return SegmentDataset(
        feature_names=feature_names,
        features=matrix,
        metadata=tuple(metadata),
        atom_ids=tuple(atom_groups),
    )


def physical_profile_from_report(report: Mapping[str, Any]) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    for role, key in (("factual", "factual"), ("foil", "counterfactual")):
        physical = report[key]["physical"]
        for name in PHYSICAL_SCALARS:
            if name == "minimum_duck_clearance_available":
                value = physical.get("minimum_duck_clearance_m") is not None
            elif name == "minimum_duck_clearance_m":
                raw = physical.get(name)
                value = 2.0 if raw is None else raw
            else:
                value = physical.get(name, 0.0)
            result["%s_%s" % (role, name)] = value
    delta = report["physical_delta_counterfactual_minus_factual"]
    for name in PHYSICAL_FEATURES:
        if not name.startswith("delta_"):
            continue
        raw_name = name[len("delta_"):]
        value = delta.get(raw_name)
        result[name] = 0.0 if value is None else value
    factual = report["factual"]["physical"]
    foil = report["counterfactual"]["physical"]
    factual_safe = not any(bool(factual.get(name)) for name in (
        "lane_departure", "duck_collision", "other_collision"
    )) and int(factual.get("stop_violations", 0)) == 0
    foil_safe = not any(bool(foil.get(name)) for name in (
        "lane_departure", "duck_collision", "other_collision"
    )) and int(foil.get("stop_violations", 0)) == 0
    result["factual_safe"] = factual_safe
    result["foil_safe"] = foil_safe
    result["factual_safer_than_foil"] = factual_safe and not foil_safe
    result["termination_changed"] = (
        factual.get("termination_reason") != foil.get("termination_reason")
    )
    return result


def reward_profile_from_report(report: Mapping[str, Any]) -> Dict[str, float]:
    values = report.get("reward_delta_counterfactual_minus_factual", {})
    result = {}
    if values:
        final = values[sorted(values, key=lambda item: int(item))[-1]]
        result["discounted_total"] = _number(final)
    factual = report.get("factual", {}).get("reward_profile", [])
    foil = report.get("counterfactual", {}).get("reward_profile", [])
    if factual and foil:
        factual_terms = factual[-1].get("discounted_terms", {})
        foil_terms = foil[-1].get("discounted_terms", {})
        for name in sorted(set(factual_terms) | set(foil_terms)):
            result["term_%s" % name] = (
                _number(foil_terms.get(name, 0.0))
                - _number(factual_terms.get(name, 0.0))
            )
    return result
