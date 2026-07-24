"""Leakage-free numerical signatures derived only from certified evidence."""

from dataclasses import dataclass
import re
from typing import Any, Mapping, Optional, Sequence, Tuple

import numpy as np

from ..eddp.counterfactual_profile import CONCEPTS
from ..eddp.signature import PHYSICAL_FEATURES, VERIFICATION_FEATURES
from .schema import CertifiedExplanationInstance, ExplanationTrajectory


FORBIDDEN_TOKENS = (
    "solver",
    "policy",
    "checkpoint",
    "seed",
    "episode",
    "scenario",
    "context",
    "action",
    "primitive",
    "trigger",
    "label",
    "q_value",
    "q_margin",
    "integrated_gradient",
    "critic",
    "natural_language",
    "raw_state",
)

DECISION_FEATURES = tuple(
    name
    for concept in CONCEPTS
    for name in (
        "%s_flip" % concept,
        "%s_abs_delta" % concept,
        "%s_signed_delta" % concept,
    )
) + (
    "any_flip",
    "minimum_flip_distance",
    "valid_fraction",
    "boundary_proximity",
)

CERTIFICATE_FEATURES = (
    "counterfactual_valid",
    "branch_invariants_pass",
    "paired_outcome_valid",
    "deterministic_mode",
    "teacher_inactive",
    "support_valid",
)

TEMPORAL_FEATURES = (
    "previous_distance",
    "decision_change",
    "risk_change",
    "verification_transition",
    "evidence_persistence",
    "certificate_continuity",
)

INSTANCE_FEATURE_NAMES = (
    tuple("decision__%s" % name for name in DECISION_FEATURES)
    + tuple("outcome__%s" % name for name in PHYSICAL_FEATURES)
    + tuple("verification__%s" % name for name in VERIFICATION_FEATURES)
    + tuple("certificate__%s" % name for name in CERTIFICATE_FEATURES)
    + tuple("temporal__%s" % name for name in TEMPORAL_FEATURES)
)


def assert_explanation_only_contract(names: Sequence[str]) -> None:
    violations = []
    for name in names:
        lowered = str(name).lower()
        tokens = set(re.split(r"[^a-z0-9]+", lowered))
        for forbidden in FORBIDDEN_TOKENS:
            match = forbidden in lowered if "_" in forbidden else forbidden in tokens
            if match:
                violations.append((name, forbidden))
    if violations:
        raise ValueError("C-EDDP feature leakage: %r" % violations)


def _number(value: Any) -> float:
    if value is None:
        return 0.0
    if isinstance(value, bool):
        return float(value)
    number = float(value)
    if not np.isfinite(number):
        raise ValueError("non-finite C-EDDP feature")
    return number


def _profile(record: CertifiedExplanationInstance, block: str, key: str) -> Mapping[str, Any]:
    container = {
        "decision": record.decision_evidence,
        "outcome": record.outcome_evidence,
        "verification": record.verification_evidence,
    }[block]
    value = container.get(key, {})
    return value if isinstance(value, Mapping) else {}


def _base_vector(record: CertifiedExplanationInstance) -> np.ndarray:
    decision = _profile(record, "decision", "counterfactual_profile")
    outcome = _profile(record, "outcome", "physical_profile")
    verification = _profile(record, "verification", "verification_profile")
    values = []
    for name in DECISION_FEATURES:
        if name == "valid_fraction":
            attempts = max(1.0, _number(decision.get("attempts", 0.0)))
            value = _number(decision.get("valid_attempts", 0.0)) / attempts
        elif name == "boundary_proximity":
            distance = max(0.0, _number(decision.get("minimum_flip_distance", 0.0)))
            value = 1.0 / (1.0 + distance)
        else:
            value = decision.get(name, 0.0)
        values.append(_number(value))
    for name in PHYSICAL_FEATURES:
        values.append(_number(outcome.get(name, 0.0)))
    for name in VERIFICATION_FEATURES:
        values.append(_number(verification.get(name, 0.0)))
    cert = record.certificate
    for source_name in (
        "counterfactual_valid",
        "branch_invariants_pass",
        "paired_outcome_valid",
        "deterministic_policy_mode",
        "teacher_inactive",
        "supported_or_reachable_state",
    ):
        values.append(_number(cert.get(source_name, False)))
    return np.asarray(values, dtype=np.float64)


def _risk_score(record: CertifiedExplanationInstance) -> float:
    profile = _profile(record, "outcome", "physical_profile")
    return float(
        _number(profile.get("foil_lane_departure", 0.0))
        + _number(profile.get("foil_duck_collision", 0.0))
        + _number(profile.get("foil_other_collision", 0.0))
        + _number(profile.get("foil_stop_violations", 0.0))
        - _number(profile.get("factual_lane_departure", 0.0))
        - _number(profile.get("factual_duck_collision", 0.0))
        - _number(profile.get("factual_other_collision", 0.0))
        - _number(profile.get("factual_stop_violations", 0.0))
    )


def _verification_bits(record: CertifiedExplanationInstance) -> np.ndarray:
    profile = _profile(record, "verification", "verification_profile")
    return np.asarray(
        [_number(profile.get(name, 0.0)) for name in VERIFICATION_FEATURES],
        dtype=np.float64,
    )


def instance_feature_vector(
    record: CertifiedExplanationInstance,
    previous: Optional[CertifiedExplanationInstance] = None,
) -> np.ndarray:
    """Build a vector that is invariant to solver/action/context metadata."""

    assert_explanation_only_contract(INSTANCE_FEATURE_NAMES)
    base = _base_vector(record)
    if previous is None:
        temporal = np.asarray((0.0, 0.0, 0.0, 0.0, 1.0, 1.0), dtype=np.float64)
    else:
        previous_base = _base_vector(previous)
        decision_count = len(DECISION_FEATURES)
        decision_change = float(np.linalg.norm(
            base[:decision_count] - previous_base[:decision_count]
        ))
        verification_transition = float(np.count_nonzero(
            _verification_bits(record) != _verification_bits(previous)
        ))
        active_now = base[:decision_count] != 0.0
        active_before = previous_base[:decision_count] != 0.0
        union = int(np.count_nonzero(active_now | active_before))
        persistence = (
            float(np.count_nonzero(active_now & active_before)) / union
            if union else 1.0
        )
        temporal = np.asarray((
            float(np.linalg.norm(base - previous_base)),
            decision_change,
            _risk_score(record) - _risk_score(previous),
            verification_transition,
            persistence,
            float(
                record.certificate.get("binding_gate_pass", False)
                and previous.certificate.get("binding_gate_pass", False)
            ),
        ), dtype=np.float64)
    result = np.concatenate((base, temporal))
    if result.shape != (len(INSTANCE_FEATURE_NAMES),):
        raise AssertionError("wrong C-EDDP instance signature shape")
    return result


@dataclass(frozen=True)
class SignatureTrajectory:
    trajectory: ExplanationTrajectory
    feature_names: Tuple[str, ...]
    matrix: np.ndarray


def build_signature_trajectory(trajectory: ExplanationTrajectory) -> SignatureTrajectory:
    rows = []
    previous = None
    for record in trajectory.instances:
        rows.append(instance_feature_vector(record, previous))
        previous = record
    return SignatureTrajectory(
        trajectory=trajectory,
        feature_names=INSTANCE_FEATURE_NAMES,
        matrix=np.vstack(rows).astype(np.float64),
    )
