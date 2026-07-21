"""Solver-neutral response curves over valid state counterfactuals."""

import csv
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

from .counterfactual import (
    ControllerSemantics,
    ManifoldBounds,
    SyntheticStateRecord,
    make_counterfactual,
    state_anchor_id,
)
from .primitives import PrimitiveLabel, PrimitiveThresholds, label_primitive
from .schema import (
    CanonicalAction,
    CanonicalState,
    PolicyDecision,
    SolverKind,
    TABULAR_SOLVERS,
    to_dict,
)


RESPONSE_CURVE_SCHEMA_VERSION = "1.0.0"


@dataclass(frozen=True)
class SweepSpec:
    feature: str
    values: Tuple[Any, ...]
    base_changes: Mapping[str, Any] = None
    intervention_name: Optional[str] = None

    def __post_init__(self) -> None:
        if not self.feature:
            raise ValueError("sweep feature must not be empty")
        if not self.values:
            raise ValueError("sweep must contain at least one value")
        if self.base_changes is None:
            object.__setattr__(self, "base_changes", {})


@dataclass(frozen=True)
class ResponsePoint:
    requested_value: Any
    synthetic: SyntheticStateRecord
    decision: Optional[PolicyDecision]
    primitive: Optional[PrimitiveLabel]
    action_changed_from_anchor: Optional[bool]
    primitive_changed_from_anchor: Optional[bool]


@dataclass(frozen=True)
class MinimalStateCounterfactual:
    requested_value: Any
    distance_from_anchor: Optional[float]
    action: CanonicalAction
    primitive: str


@dataclass(frozen=True)
class ResponseCurve:
    anchor_id: str
    solver: SolverKind
    feature: str
    anchor_state: CanonicalState
    anchor_decision: PolicyDecision
    anchor_primitive: PrimitiveLabel
    points: Tuple[ResponsePoint, ...]
    valid_points: int
    rejected_points: int
    minimal_action_counterfactual: Optional[MinimalStateCounterfactual]
    minimal_primitive_counterfactual: Optional[MinimalStateCounterfactual]
    schema_version: str = RESPONSE_CURVE_SCHEMA_VERSION

    def to_json(self, indent: Optional[int] = 2) -> str:
        return json.dumps(
            to_dict(self),
            sort_keys=True,
            indent=indent,
            allow_nan=False,
        )

    def save_json(self, path: Path, indent: Optional[int] = 2) -> None:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_suffix(output.suffix + ".tmp")
        temporary.write_text(self.to_json(indent=indent) + "\n", encoding="utf-8")
        temporary.replace(output)

    def save_csv(self, path: Path) -> None:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_suffix(output.suffix + ".tmp")
        fieldnames = (
            "anchor_id",
            "solver",
            "feature",
            "requested_value",
            "valid",
            "rejection_codes",
            "v_cmd",
            "omega_cmd",
            "action_id",
            "action_name",
            "primitive",
            "action_changed",
            "primitive_changed",
        )
        with temporary.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for point in self.points:
                action = point.decision.action if point.decision else None
                writer.writerow(
                    {
                        "anchor_id": self.anchor_id,
                        "solver": self.solver.value,
                        "feature": self.feature,
                        "requested_value": point.requested_value,
                        "valid": point.synthetic.validation.valid,
                        "rejection_codes": "|".join(
                            point.synthetic.validation.reason_codes
                        ),
                        "v_cmd": None if action is None else action.v_cmd,
                        "omega_cmd": None if action is None else action.omega_cmd,
                        "action_id": None if action is None else action.action_id,
                        "action_name": None if action is None else action.action_name,
                        "primitive": (
                            None if point.primitive is None
                            else point.primitive.primitive.value
                        ),
                        "action_changed": point.action_changed_from_anchor,
                        "primitive_changed": point.primitive_changed_from_anchor,
                    }
                )
        temporary.replace(output)


def _same_action(left: CanonicalAction, right: CanonicalAction) -> bool:
    if left.solver != right.solver:
        return False
    if left.solver in TABULAR_SOLVERS:
        return left.action_id == right.action_id
    return (
        abs(left.v_cmd - right.v_cmd) <= 1e-7
        and abs(left.omega_cmd - right.omega_cmd) <= 1e-7
    )


def _distance(requested: Any, anchor_value: Any) -> Optional[float]:
    if isinstance(requested, bool) or isinstance(anchor_value, bool):
        return None
    try:
        return abs(float(requested) - float(anchor_value))
    except (TypeError, ValueError):
        return None


def _minimal_counterfactual(
    points: Sequence[ResponsePoint],
    anchor_value: Any,
) -> Optional[MinimalStateCounterfactual]:
    if not points:
        return None
    ranked = sorted(
        enumerate(points),
        key=lambda item: (
            _distance(item[1].requested_value, anchor_value) is None,
            _distance(item[1].requested_value, anchor_value)
            if _distance(item[1].requested_value, anchor_value) is not None
            else item[0],
        ),
    )
    chosen = ranked[0][1]
    return MinimalStateCounterfactual(
        requested_value=chosen.requested_value,
        distance_from_anchor=_distance(chosen.requested_value, anchor_value),
        action=chosen.decision.action,
        primitive=chosen.primitive.primitive.value,
    )


def run_response_curve(
    policy: Any,
    solver: SolverKind,
    anchor: CanonicalState,
    spec: SweepSpec,
    bounds: ManifoldBounds = ManifoldBounds(),
    controller: ControllerSemantics = ControllerSemantics(),
    primitive_thresholds: PrimitiveThresholds = PrimitiveThresholds(),
) -> ResponseCurve:
    """Query a policy only at points accepted by the manifold validator."""

    solver = SolverKind(solver)
    anchor_identifier = state_anchor_id(anchor)
    anchor_record = make_counterfactual(
        anchor=anchor,
        solver=solver,
        intervention_name="anchor_projection",
        changes={},
        anchor_id=anchor_identifier,
        bounds=bounds,
        controller=controller,
    )
    if not anchor_record.validation.valid:
        raise ValueError(
            "anchor is invalid for %s: %s"
            % (solver.value, anchor_record.validation.reason_codes)
        )
    anchor_decision = policy.decide(anchor_record.state)
    anchor_primitive = label_primitive(
        anchor_decision.state,
        anchor_decision.action,
        thresholds=primitive_thresholds,
    )

    points = []
    for requested in spec.values:
        changes: Dict[str, Any] = dict(spec.base_changes)
        changes[spec.feature] = requested
        synthetic = make_counterfactual(
            anchor=anchor,
            solver=solver,
            intervention_name=(
                spec.intervention_name or ("sweep_%s" % spec.feature)
            ),
            changes=changes,
            anchor_id=anchor_identifier,
            bounds=bounds,
            controller=controller,
        )
        if not synthetic.validation.valid:
            points.append(
                ResponsePoint(
                    requested_value=requested,
                    synthetic=synthetic,
                    decision=None,
                    primitive=None,
                    action_changed_from_anchor=None,
                    primitive_changed_from_anchor=None,
                )
            )
            continue
        decision = policy.decide(synthetic.state)
        primitive = label_primitive(
            decision.state,
            decision.action,
            thresholds=primitive_thresholds,
        )
        points.append(
            ResponsePoint(
                requested_value=requested,
                synthetic=synthetic,
                decision=decision,
                primitive=primitive,
                action_changed_from_anchor=not _same_action(
                    decision.action, anchor_decision.action
                ),
                primitive_changed_from_anchor=(
                    primitive.primitive != anchor_primitive.primitive
                ),
            )
        )

    valid = tuple(point for point in points if point.synthetic.validation.valid)
    action_flipped = tuple(
        point for point in valid if point.action_changed_from_anchor is True
    )
    primitive_flipped = tuple(
        point for point in valid if point.primitive_changed_from_anchor is True
    )
    anchor_value = getattr(anchor, spec.feature)
    minimal_action = _minimal_counterfactual(action_flipped, anchor_value)
    minimal_primitive = _minimal_counterfactual(
        primitive_flipped, anchor_value
    )
    return ResponseCurve(
        anchor_id=anchor_identifier,
        solver=solver,
        feature=spec.feature,
        anchor_state=anchor_record.state,
        anchor_decision=anchor_decision,
        anchor_primitive=anchor_primitive,
        points=tuple(points),
        valid_points=len(valid),
        rejected_points=len(points) - len(valid),
        minimal_action_counterfactual=minimal_action,
        minimal_primitive_counterfactual=minimal_primitive,
    )


def run_response_suite(
    policy: Any,
    solver: SolverKind,
    anchor: CanonicalState,
    specs: Sequence[SweepSpec],
    **kwargs: Any,
) -> Tuple[ResponseCurve, ...]:
    return tuple(
        run_response_curve(policy, solver, anchor, spec, **kwargs)
        for spec in specs
    )


def save_response_suite(
    curves: Sequence[ResponseCurve],
    output_dir: Path,
    prefix: str,
) -> Tuple[Path, ...]:
    output = Path(output_dir)
    written = []
    for curve in curves:
        stem = "%s_%s_%s" % (prefix, curve.solver.value, curve.feature)
        json_path = output / (stem + ".json")
        csv_path = output / (stem + ".csv")
        curve.save_json(json_path)
        curve.save_csv(csv_path)
        written.extend((json_path, csv_path))
    return tuple(written)
