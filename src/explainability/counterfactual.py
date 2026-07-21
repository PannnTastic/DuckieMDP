"""Valid, provenance-carrying state interventions for explainable policies.

M6 never queries a policy with an arbitrary hand-written vector.  Every
counterfactual starts from a real rollout anchor, repairs dependent fields, is
projected into the target solver representation, and is then validated.
"""

from dataclasses import dataclass, replace
from hashlib import sha256
import json
from math import hypot, isfinite, pi
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

from ..continuous_state import ContinuousStateConfig, continuous_observation_space
from ..discretizer import discretize
from .schema import CanonicalState, SolverKind, to_dict
from .semantic_state import (
    encode_canonical_for_sac,
    raw_state_from_canonical,
)


COUNTERFACTUAL_SCHEMA_VERSION = "1.0.0"


@dataclass(frozen=True)
class ManifoldBounds:
    max_abs_d: float = 0.25
    max_abs_phi: float = pi / 2.0
    max_speed: float = 0.41
    max_abs_curvature: float = 8.0
    max_stop_distance: float = 3.0
    max_duck_distance: float = 2.0
    max_relative_speed: float = 0.50
    curvature_class_threshold: float = 0.05
    q_duck_max_distance: float = 1.20
    q_duck_near_distance: float = 0.60
    q_duck_corridor_width: float = 0.60


@dataclass(frozen=True)
class ControllerSemantics:
    """Controller rules required to judge flag combinations.

    In the canonical one-crossing and repeat-rearm tasks, an actively crossing
    Duckie is not simultaneously armed for another crossing.  Unlimited
    controllers without re-arm can explicitly opt into the alternate meaning.
    """

    allow_active_and_crossing_available: bool = False
    label: str = "one_crossing_or_repeat_rearm"


@dataclass(frozen=True)
class ValidationIssue:
    code: str
    field: str
    message: str


@dataclass(frozen=True)
class ManifoldValidation:
    valid: bool
    solver: SolverKind
    issues: Tuple[ValidationIssue, ...]
    encoded_observation_valid: Optional[bool]
    schema_version: str = COUNTERFACTUAL_SCHEMA_VERSION

    @property
    def reason_codes(self) -> Tuple[str, ...]:
        return tuple(issue.code for issue in self.issues)


@dataclass(frozen=True)
class SyntheticStateRecord:
    anchor_id: str
    intervention_name: str
    requested_changes: Mapping[str, Any]
    applied_changes: Mapping[str, Any]
    repair_notes: Tuple[str, ...]
    state: CanonicalState
    validation: ManifoldValidation
    schema_version: str = COUNTERFACTUAL_SCHEMA_VERSION


def state_anchor_id(state: CanonicalState) -> str:
    canonical = json.dumps(
        to_dict(state), sort_keys=True, separators=(",", ":"), allow_nan=False
    )
    return sha256(canonical.encode("utf-8")).hexdigest()


def _issue(issues, code: str, field: str, message: str) -> None:
    issues.append(ValidationIssue(code=code, field=field, message=message))


def _finite_values(state: CanonicalState) -> Sequence[Tuple[str, Optional[float]]]:
    return tuple(
        (name, getattr(state, name))
        for name in (
            "d",
            "phi",
            "v",
            "curvature",
            "stop_distance",
            "stop_hold_progress",
            "duck_longitudinal",
            "duck_lateral",
            "duck_v_longitudinal_relative",
            "duck_v_lateral_relative",
        )
    )


def validate_state(
    state: CanonicalState,
    solver: SolverKind,
    bounds: ManifoldBounds = ManifoldBounds(),
    controller: ControllerSemantics = ControllerSemantics(),
    continuous_config: ContinuousStateConfig = ContinuousStateConfig(),
) -> ManifoldValidation:
    """Validate semantic consistency and the target policy input contract."""

    solver = SolverKind(solver)
    issues = []
    for name, value in _finite_values(state):
        if value is not None and not isfinite(float(value)):
            _issue(issues, "NON_FINITE", name, "%s must be finite" % name)

    if abs(state.d) > bounds.max_abs_d:
        _issue(issues, "D_OUT_OF_BOUNDS", "d", "lateral offset exceeds lane bound")
    if abs(state.phi) > bounds.max_abs_phi:
        _issue(issues, "PHI_OUT_OF_BOUNDS", "phi", "heading error exceeds bound")
    if state.v < 0.0 or state.v > bounds.max_speed:
        _issue(issues, "SPEED_OUT_OF_BOUNDS", "v", "ego speed exceeds bound")
    if state.curvature_class not in {"straight", "curve_left", "curve_right"}:
        _issue(
            issues,
            "CURVATURE_CLASS_INVALID",
            "curvature_class",
            "curvature class is not recognized",
        )
    if state.curvature is not None:
        if abs(state.curvature) > bounds.max_abs_curvature:
            _issue(
                issues,
                "CURVATURE_OUT_OF_BOUNDS",
                "curvature",
                "curvature exceeds configured bound",
            )
        expected = _curvature_class(state.curvature, bounds)
        if expected != state.curvature_class:
            _issue(
                issues,
                "CURVATURE_CLASS_MISMATCH",
                "curvature_class",
                "class does not match continuous curvature",
            )

    progress = state.stop_hold_progress
    if not state.stop_present:
        if state.stop_distance is not None:
            _issue(issues, "STOP_ABSENT_DISTANCE", "stop_distance", "absent stop needs None distance")
        if state.stop_satisfied:
            _issue(issues, "STOP_ABSENT_SATISFIED", "stop_satisfied", "absent stop cannot be satisfied")
        if progress not in (None, 0.0):
            _issue(issues, "STOP_ABSENT_PROGRESS", "stop_hold_progress", "absent stop needs zero progress")
    else:
        if state.stop_distance is None:
            _issue(issues, "STOP_PRESENT_NO_DISTANCE", "stop_distance", "present stop requires distance")
        elif not 0.0 <= state.stop_distance <= bounds.max_stop_distance:
            _issue(issues, "STOP_DISTANCE_OUT_OF_BOUNDS", "stop_distance", "stop distance exceeds bounds")
        if progress is None or not 0.0 <= progress <= 1.0:
            _issue(issues, "STOP_PROGRESS_INVALID", "stop_hold_progress", "hold progress must be in [0,1]")
        elif state.stop_satisfied and progress != 1.0:
            _issue(issues, "STOP_SATISFIED_PROGRESS", "stop_hold_progress", "satisfied stop requires full progress")
        elif not state.stop_satisfied and progress >= 1.0:
            _issue(issues, "STOP_UNSATISFIED_PROGRESS", "stop_hold_progress", "full progress requires satisfied flag")

    geometry_names = (
        "duck_longitudinal",
        "duck_lateral",
        "duck_v_longitudinal_relative",
        "duck_v_lateral_relative",
    )
    geometry = tuple(getattr(state, name) for name in geometry_names)
    if not state.duck_present:
        if any(value is not None for value in geometry):
            _issue(issues, "DUCK_ABSENT_GEOMETRY", "duck_present", "absent Duckie needs canonical None geometry")
        if state.duck_active not in (None, False):
            _issue(issues, "DUCK_ABSENT_ACTIVE", "duck_active", "absent Duckie cannot be active")
        if state.duck_crossing_available not in (None, False):
            _issue(issues, "DUCK_ABSENT_AVAILABLE", "duck_crossing_available", "absent Duckie cannot be armed")
    elif solver == SolverKind.SAC:
        if any(value is None for value in geometry):
            _issue(issues, "DUCK_PRESENT_NO_GEOMETRY", "duck_present", "SAC Duckie requires metric geometry")
        else:
            if abs(float(state.duck_longitudinal)) > bounds.max_duck_distance:
                _issue(issues, "DUCK_LONG_OUT_OF_BOUNDS", "duck_longitudinal", "Duckie longitudinal distance exceeds bounds")
            if abs(float(state.duck_lateral)) > bounds.max_duck_distance:
                _issue(issues, "DUCK_LAT_OUT_OF_BOUNDS", "duck_lateral", "Duckie lateral distance exceeds bounds")
            for name in geometry_names[2:]:
                if abs(float(getattr(state, name))) > bounds.max_relative_speed:
                    _issue(issues, "DUCK_SPEED_OUT_OF_BOUNDS", name, "relative Duckie speed exceeds bounds")
        if state.duck_active is None or state.duck_crossing_available is None:
            _issue(issues, "DUCK_FLAGS_MISSING", "duck_active", "SAC Duckie requires controller flags")
        elif (
            state.duck_active
            and state.duck_crossing_available
            and not controller.allow_active_and_crossing_available
        ):
            _issue(issues, "DUCK_PHASE_INCONSISTENT", "duck_crossing_available", "active crossing is not simultaneously armed")
        if state.duck_threat not in (None, "none"):
            _issue(issues, "SAC_CATEGORICAL_DUCK", "duck_threat", "SAC projection must use metric Duckie features")
    else:
        if state.duck_threat not in {
            "side_far", "side_near", "crossing_far", "crossing_near"
        }:
            _issue(issues, "Q_DUCK_CATEGORY_MISSING", "duck_threat", "Q-learning Duckie requires threat category")
        if any(value is not None for value in geometry):
            _issue(issues, "Q_METRIC_DUCK", "duck_present", "Q projection must remove metric Duckie geometry")

    encoded_valid: Optional[bool] = None
    try:
        if solver == SolverKind.SAC:
            encoded = encode_canonical_for_sac(state, continuous_config)
            encoded_valid = bool(continuous_observation_space().contains(encoded))
            if not encoded_valid:
                _issue(issues, "SAC_OBSERVATION_OUT_OF_BOUNDS", "state", "encoded SAC observation is outside Box bounds")
        else:
            discretize(raw_state_from_canonical(state))
            encoded_valid = True
    except (TypeError, ValueError, KeyError) as error:
        encoded_valid = False
        _issue(issues, "SOLVER_ENCODING_FAILED", "state", str(error))

    return ManifoldValidation(
        valid=not issues,
        solver=solver,
        issues=tuple(issues),
        encoded_observation_valid=encoded_valid,
    )


def _curvature_class(value: float, bounds: ManifoldBounds) -> str:
    if value > bounds.curvature_class_threshold:
        return "curve_left"
    if value < -bounds.curvature_class_threshold:
        return "curve_right"
    return "straight"


def _absent_duck(values: Dict[str, Any]) -> None:
    values.update(
        duck_present=False,
        duck_threat="none" if values.get("source_representation", "").startswith("q_") else None,
        duck_longitudinal=None,
        duck_lateral=None,
        duck_v_longitudinal_relative=None,
        duck_v_lateral_relative=None,
        duck_active=None,
        duck_crossing_available=None,
    )


def project_state_for_solver(
    state: CanonicalState,
    solver: SolverKind,
    bounds: ManifoldBounds = ManifoldBounds(),
) -> CanonicalState:
    """Loss-aware projection of one semantic state into a policy input."""

    solver = SolverKind(solver)
    values = dict(state.__dict__)
    values["source_index"] = None
    if solver == SolverKind.SAC:
        values["source_representation"] = "sac_counterfactual_projection"
        values["duck_threat"] = None
        if not values["duck_present"]:
            _absent_duck(values)
            values["duck_threat"] = None
        return CanonicalState(**values)

    values["source_representation"] = "q_counterfactual_projection"
    values["curvature"] = None
    values["stop_hold_progress"] = 1.0 if values["stop_satisfied"] else 0.0
    if not values["duck_present"]:
        _absent_duck(values)
        values["duck_threat"] = "none"
        return CanonicalState(**values)

    long_value = values.get("duck_longitudinal")
    lat_value = values.get("duck_lateral")
    if long_value is not None and lat_value is not None:
        distance = hypot(float(long_value), float(lat_value))
        visible = (
            float(long_value) >= 0.0
            and abs(float(lat_value)) <= bounds.q_duck_corridor_width
            and distance <= bounds.q_duck_max_distance
        )
        if not visible:
            _absent_duck(values)
            values["duck_threat"] = "none"
            return CanonicalState(**values)
        crossing = bool(values.get("duck_active"))
        suffix = "near" if distance <= bounds.q_duck_near_distance else "far"
        values["duck_threat"] = ("crossing_" if crossing else "side_") + suffix
    elif values.get("duck_threat") in (None, "none"):
        # Leave an explicitly invalid projection for the validator to reject.
        values["duck_threat"] = None
    crossing = values.get("duck_threat") in {"crossing_far", "crossing_near"}
    values["duck_active"] = crossing
    values["duck_crossing_available"] = None
    for name in (
        "duck_longitudinal",
        "duck_lateral",
        "duck_v_longitudinal_relative",
        "duck_v_lateral_relative",
    ):
        values[name] = None
    return CanonicalState(**values)


def make_counterfactual(
    anchor: CanonicalState,
    solver: SolverKind,
    intervention_name: str,
    changes: Mapping[str, Any],
    anchor_id: Optional[str] = None,
    bounds: ManifoldBounds = ManifoldBounds(),
    controller: ControllerSemantics = ControllerSemantics(),
) -> SyntheticStateRecord:
    """Apply one intervention, repair dependencies, project, and validate."""

    unknown = sorted(set(changes).difference(anchor.__dict__))
    if unknown:
        raise ValueError("unknown CanonicalState fields: %s" % unknown)
    values = dict(anchor.__dict__)
    values.update(dict(changes))
    values["source_representation"] = "synthetic_counterfactual"
    values["source_index"] = None
    repairs = []

    if "curvature" in changes and changes["curvature"] is not None:
        values["curvature_class"] = _curvature_class(float(changes["curvature"]), bounds)
        repairs.append("curvature_class derived from curvature")

    if changes.get("stop_present") is False:
        values.update(stop_distance=None, stop_satisfied=False, stop_hold_progress=0.0)
        repairs.append("cleared dependent stop fields")
    if "stop_distance" in changes and changes["stop_distance"] is not None:
        values["stop_present"] = True
        if not anchor.stop_present:
            values["stop_satisfied"] = False
            values["stop_hold_progress"] = 0.0
        repairs.append("stop_present enabled for distance intervention")
    if "stop_hold_progress" in changes:
        values["stop_present"] = True
        progress = float(changes["stop_hold_progress"])
        values["stop_satisfied"] = progress >= 1.0
        repairs.append("stop satisfaction derived from hold progress")
    if changes.get("stop_satisfied") is True:
        values["stop_present"] = True
        values["stop_hold_progress"] = 1.0
        repairs.append("full hold progress set for satisfied stop")

    if changes.get("duck_present") is False:
        _absent_duck(values)
        repairs.append("cleared dependent Duckie fields")
    metric_duck_change = any(
        name in changes
        for name in (
            "duck_longitudinal",
            "duck_lateral",
            "duck_v_longitudinal_relative",
            "duck_v_lateral_relative",
        )
    )
    if metric_duck_change:
        values["duck_present"] = True
        repairs.append("duck_present enabled for metric intervention")
    if "duck_active" in changes:
        values["duck_present"] = True
        if bool(changes["duck_active"]) and not controller.allow_active_and_crossing_available:
            values["duck_crossing_available"] = False
            repairs.append("crossing_available cleared for active Duckie")
    if "duck_crossing_available" in changes:
        values["duck_present"] = True
        if bool(changes["duck_crossing_available"]) and not controller.allow_active_and_crossing_available:
            values["duck_active"] = False
            repairs.append("duck_active cleared for armed Duckie")

    semantic = CanonicalState(**values)
    # Validate the physical/semantic intervention before lossy projection.
    # Otherwise Q-learning could silently turn an out-of-range metric value
    # into a valid category (for example curvature=9 -> curve_left).
    semantic_solver = (
        SolverKind.SAC
        if anchor.curvature is not None or anchor.duck_threat is None
        else SolverKind.Q_LEARNING
    )
    semantic_validation = validate_state(
        semantic, semantic_solver, bounds, controller
    )
    projected_anchor = project_state_for_solver(anchor, solver, bounds)
    projected = project_state_for_solver(semantic, solver, bounds)
    projected_validation = validate_state(projected, solver, bounds, controller)
    unique_issues = {}
    for issue in semantic_validation.issues + projected_validation.issues:
        unique_issues[(issue.code, issue.field, issue.message)] = issue
    issues = tuple(unique_issues.values())
    validation = ManifoldValidation(
        valid=not issues,
        solver=SolverKind(solver),
        issues=issues,
        encoded_observation_valid=projected_validation.encoded_observation_valid,
    )
    applied = {
        name: getattr(projected, name)
        for name in projected.__dict__
        if getattr(projected_anchor, name) != getattr(projected, name)
        and name not in {"source_representation", "source_index"}
    }
    return SyntheticStateRecord(
        anchor_id=anchor_id or state_anchor_id(anchor),
        intervention_name=str(intervention_name),
        requested_changes=dict(changes),
        applied_changes=applied,
        repair_notes=tuple(repairs),
        state=projected,
        validation=validation,
    )
