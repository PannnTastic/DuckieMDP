"""LEGIBLE-inspired metamorphic testing for DuckieMDP policies.

The module tests domain expectations under valid, controlled state
interventions.  It is deliberately solver-neutral at the interface while the
action comparison remains solver-aware: Q-learning uses ordinal/discrete
actions and SAC uses continuous command tolerances.
"""

import csv
from dataclasses import dataclass, field
from enum import Enum
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
from .primitives import DrivingPrimitive, PrimitiveLabel, label_primitive
from .schema import CanonicalState, PolicyDecision, SolverKind, TABULAR_SOLVERS, to_dict


METAMORPHIC_SCHEMA_VERSION = "1.0.0"


class MetamorphicStatus(str, Enum):
    PASS = "PASS"
    FAIL = "FAIL"
    NOT_APPLICABLE = "NOT_APPLICABLE"


@dataclass(frozen=True)
class RelationTolerance:
    sac_speed: float = 0.01
    sac_omega_symmetry: float = 0.15
    numeric: float = 1e-7


@dataclass(frozen=True)
class MetamorphicRelation:
    relation_id: str
    description: str
    precondition: str
    intervention: str
    expected_relation: str
    applicable_solvers: Tuple[SolverKind, ...] = (
        SolverKind.Q_LEARNING,
        SolverKind.SARSA,
        SolverKind.SAC,
    )
    tolerance: RelationTolerance = RelationTolerance()
    schema_version: str = METAMORPHIC_SCHEMA_VERSION


RELATIONS: Mapping[str, MetamorphicRelation] = {
    "MR-STOP": MetamorphicRelation(
        relation_id="MR-STOP",
        description="Approaching an unsatisfied stop must not increase speed.",
        precondition=(
            "stop present and unsatisfied; no Duckie; locally straight; "
            "safe lateral and heading errors; target stop is closer"
        ),
        intervention="decrease stop_distance while preserving context",
        expected_relation="speed(target) <= speed(source) + tolerance",
    ),
    "MR-PEDESTRIAN": MetamorphicRelation(
        relation_id="MR-PEDESTRIAN",
        description="Increasing pedestrian risk must not produce faster proceed.",
        precondition=(
            "no unsatisfied stop obligation; fixed lane context; target "
            "pedestrian risk is greater than source risk"
        ),
        intervention="increase pedestrian threat while preserving lane context",
        expected_relation=(
            "speed(target) <= speed(source) + tolerance and target is not "
            "UnsafeProceed/PrematureResume"
        ),
    ),
    "MR-CURVATURE": MetamorphicRelation(
        relation_id="MR-CURVATURE",
        description="Sharper curvature must not increase commanded speed.",
        precondition=(
            "no stop or Duckie obligation; safe lane errors; target absolute "
            "curvature is greater than source"
        ),
        intervention="increase absolute ego-relative curvature",
        expected_relation="speed(target) <= speed(source) + tolerance",
    ),
    "MR-LANE-SYMMETRY": MetamorphicRelation(
        relation_id="MR-LANE-SYMMETRY",
        description="A mirrored local lane error should mirror steering.",
        precondition=(
            "locally straight symmetric road; no stop or Duckie; target "
            "equals (d, phi)=(-d, -phi)"
        ),
        intervention="mirror d and phi",
        expected_relation="omega(target) ~= -omega(source)",
    ),
}


@dataclass(frozen=True)
class MetamorphicResult:
    relation: MetamorphicRelation
    solver: SolverKind
    anchor_id: str
    status: MetamorphicStatus
    reason: str
    source: SyntheticStateRecord
    target: SyntheticStateRecord
    source_decision: Optional[PolicyDecision] = None
    target_decision: Optional[PolicyDecision] = None
    source_primitive: Optional[PrimitiveLabel] = None
    target_primitive: Optional[PrimitiveLabel] = None
    measurements: Mapping[str, Any] = field(default_factory=dict)
    provenance: Mapping[str, Any] = field(default_factory=dict)
    schema_version: str = METAMORPHIC_SCHEMA_VERSION


def speed_level(decision: PolicyDecision) -> int:
    """Return the frozen Q-learning speed order brake < slow < fast."""

    name = decision.action.action_name
    if decision.solver not in TABULAR_SOLVERS or name is None:
        raise ValueError("speed_level is defined only for named Q actions")
    if name == "brake":
        return 0
    if name.startswith("slow_"):
        return 1
    if name.startswith("fast_"):
        return 2
    raise ValueError("unknown Q-learning action name: %s" % name)


def _pedestrian_risk(state: CanonicalState) -> float:
    categorical = {
        None: 0.0,
        "none": 0.0,
        "side_far": 1.0,
        "side_near": 2.0,
        "crossing_far": 3.0,
        "crossing_near": 4.0,
    }
    if state.duck_threat in categorical and state.duck_threat is not None:
        return categorical[state.duck_threat]
    if not state.duck_present:
        return 0.0
    active = 3.0 if state.duck_active else 1.0
    if state.duck_longitudinal is None or state.duck_lateral is None:
        return active
    distance = (state.duck_longitudinal ** 2 + state.duck_lateral ** 2) ** 0.5
    proximity = max(0.0, 1.0 - min(distance, 2.0) / 2.0)
    return active + proximity


def _precondition(
    relation_id: str,
    source: CanonicalState,
    target: CanonicalState,
    tolerance: RelationTolerance,
) -> Tuple[bool, str]:
    eps = tolerance.numeric
    if relation_id == "MR-STOP":
        valid = (
            source.stop_present
            and target.stop_present
            and not source.stop_satisfied
            and not target.stop_satisfied
            and not source.duck_present
            and not target.duck_present
            and source.stop_distance is not None
            and target.stop_distance is not None
            and target.stop_distance < source.stop_distance - eps
            and source.curvature_class == "straight"
            and target.curvature_class == "straight"
            and abs(source.d) <= 0.10
            and abs(source.phi) <= 0.30
        )
        return valid, "formal stop precondition" if valid else "stop precondition false"

    if relation_id == "MR-PEDESTRIAN":
        no_stop = not (
            (source.stop_present and not source.stop_satisfied)
            or (target.stop_present and not target.stop_satisfied)
        )
        same_lane = (
            abs(source.d - target.d) <= eps
            and abs(source.phi - target.phi) <= eps
            and source.curvature_class == target.curvature_class
        )
        increased = _pedestrian_risk(target) > _pedestrian_risk(source) + eps
        valid = no_stop and same_lane and increased
        return valid, (
            "pedestrian risk increased under fixed lane context"
            if valid else "pedestrian precondition false"
        )

    if relation_id == "MR-CURVATURE":
        source_kappa = 0.0 if source.curvature is None else source.curvature
        target_kappa = 0.0 if target.curvature is None else target.curvature
        # Q-learning loses metric curvature, so class change is the meaningful
        # ordered intervention after projection.
        sharper = (
            abs(target_kappa) > abs(source_kappa) + eps
            if source.curvature is not None and target.curvature is not None
            else source.curvature_class == "straight"
            and target.curvature_class in {"curve_left", "curve_right"}
        )
        valid = (
            not source.stop_present
            and not target.stop_present
            and not source.duck_present
            and not target.duck_present
            and abs(source.d) <= 0.10
            and abs(source.phi) <= 0.30
            and sharper
        )
        return valid, "absolute curvature increased" if valid else "curvature precondition false"

    if relation_id == "MR-LANE-SYMMETRY":
        valid = (
            not source.stop_present
            and not target.stop_present
            and not source.duck_present
            and not target.duck_present
            and source.curvature_class == "straight"
            and target.curvature_class == "straight"
            and abs(target.d + source.d) <= eps
            and abs(target.phi + source.phi) <= eps
            and (abs(source.d) > eps or abs(source.phi) > eps)
        )
        return valid, "valid mirrored local lane context" if valid else "lane symmetry precondition false"

    raise KeyError("unknown metamorphic relation: %s" % relation_id)


def _mirror_q_action(name: str) -> str:
    if name.endswith("_left"):
        return name[:-5] + "_right"
    if name.endswith("_right"):
        return name[:-6] + "_left"
    return name


def _evaluate_expectation(
    relation: MetamorphicRelation,
    source: PolicyDecision,
    target: PolicyDecision,
    source_primitive: PrimitiveLabel,
    target_primitive: PrimitiveLabel,
) -> Tuple[bool, str, Dict[str, Any]]:
    measurements: Dict[str, Any] = {
        "source_v_cmd": source.action.v_cmd,
        "target_v_cmd": target.action.v_cmd,
        "source_omega_cmd": source.action.omega_cmd,
        "target_omega_cmd": target.action.omega_cmd,
        "source_primitive": source_primitive.primitive.value,
        "target_primitive": target_primitive.primitive.value,
    }
    if source.solver in TABULAR_SOLVERS:
        source_speed = speed_level(source)
        target_speed = speed_level(target)
        measurements.update(
            source_speed_level=source_speed,
            target_speed_level=target_speed,
        )
        speed_ok = target_speed <= source_speed
    else:
        delta = target.action.v_cmd - source.action.v_cmd
        measurements["speed_delta"] = delta
        measurements["speed_tolerance"] = relation.tolerance.sac_speed
        speed_ok = delta <= relation.tolerance.sac_speed

    if relation.relation_id in {"MR-STOP", "MR-CURVATURE"}:
        return speed_ok, "speed monotonicity", measurements

    if relation.relation_id == "MR-PEDESTRIAN":
        unsafe = target_primitive.primitive in {
            DrivingPrimitive.UNSAFE_PROCEED,
            DrivingPrimitive.PREMATURE_RESUME,
        }
        measurements["unsafe_target_primitive"] = unsafe
        return speed_ok and not unsafe, "pedestrian risk response", measurements

    if relation.relation_id == "MR-LANE-SYMMETRY":
        if source.solver in TABULAR_SOLVERS:
            expected = _mirror_q_action(source.action.action_name)
            actual = target.action.action_name
            measurements.update(expected_target_action=expected, actual_target_action=actual)
            return actual == expected, "exact mirrored Q action", measurements
        residual = abs(target.action.omega_cmd + source.action.omega_cmd)
        measurements.update(
            omega_symmetry_residual=residual,
            omega_symmetry_tolerance=relation.tolerance.sac_omega_symmetry,
        )
        return residual <= relation.tolerance.sac_omega_symmetry, "continuous steering symmetry", measurements

    raise KeyError("unknown relation: %s" % relation.relation_id)


def evaluate_relation(
    policy: Any,
    solver: SolverKind,
    anchor: CanonicalState,
    relation_id: str,
    source_changes: Mapping[str, Any],
    target_changes: Mapping[str, Any],
    provenance: Optional[Mapping[str, Any]] = None,
    bounds: ManifoldBounds = ManifoldBounds(),
    controller: ControllerSemantics = ControllerSemantics(),
) -> MetamorphicResult:
    """Evaluate one valid state pair without querying rejected states."""

    solver = SolverKind(solver)
    relation = RELATIONS[relation_id]
    anchor_identifier = state_anchor_id(anchor)
    source_request = dict(source_changes)
    target_request = dict(source_changes)
    target_request.update(dict(target_changes))
    source = make_counterfactual(
        anchor, solver, relation_id + ":source", source_request,
        anchor_identifier, bounds, controller,
    )
    target = make_counterfactual(
        anchor, solver, relation_id + ":target", target_request,
        anchor_identifier, bounds, controller,
    )
    metadata = dict(provenance or {})

    if solver not in relation.applicable_solvers:
        return MetamorphicResult(
            relation, solver, anchor_identifier, MetamorphicStatus.NOT_APPLICABLE,
            "solver not covered by relation", source, target, provenance=metadata,
        )
    if not source.validation.valid or not target.validation.valid:
        codes = sorted(set(source.validation.reason_codes + target.validation.reason_codes))
        return MetamorphicResult(
            relation, solver, anchor_identifier, MetamorphicStatus.NOT_APPLICABLE,
            "invalid manifold pair: %s" % ",".join(codes), source, target,
            measurements={"validation_codes": codes}, provenance=metadata,
        )

    applicable, reason = _precondition(
        relation_id, source.state, target.state, relation.tolerance
    )
    if not applicable:
        return MetamorphicResult(
            relation, solver, anchor_identifier, MetamorphicStatus.NOT_APPLICABLE,
            reason, source, target, provenance=metadata,
        )

    source_decision = policy.decide(source.state)
    target_decision = policy.decide(target.state)
    source_primitive = label_primitive(source.state, source_decision.action)
    target_primitive = label_primitive(target.state, target_decision.action)
    passed, expectation, measurements = _evaluate_expectation(
        relation, source_decision, target_decision,
        source_primitive, target_primitive,
    )
    return MetamorphicResult(
        relation=relation,
        solver=solver,
        anchor_id=anchor_identifier,
        status=MetamorphicStatus.PASS if passed else MetamorphicStatus.FAIL,
        reason=("satisfied " if passed else "violated ") + expectation,
        source=source,
        target=target,
        source_decision=source_decision,
        target_decision=target_decision,
        source_primitive=source_primitive,
        target_primitive=target_primitive,
        measurements=measurements,
        provenance=metadata,
    )


def save_results(
    results: Sequence[MetamorphicResult],
    output_dir: Path,
    prefix: str = "m7",
) -> Tuple[Path, Path]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    json_path = output / (prefix + "_metamorphic_results.json")
    csv_path = output / (prefix + "_metamorphic_results.csv")
    json_tmp = json_path.with_suffix(json_path.suffix + ".tmp")
    json_tmp.write_text(
        json.dumps([to_dict(result) for result in results], indent=2,
                   sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    json_tmp.replace(json_path)

    fields = (
        "relation_id", "solver", "status", "reason", "anchor_id",
        "source_action", "target_action", "source_primitive",
        "target_primitive", "support_status",
    )
    csv_tmp = csv_path.with_suffix(csv_path.suffix + ".tmp")
    with csv_tmp.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for result in results:
            writer.writerow({
                "relation_id": result.relation.relation_id,
                "solver": result.solver.value,
                "status": result.status.value,
                "reason": result.reason,
                "anchor_id": result.anchor_id,
                "source_action": None if result.source_decision is None else (
                    result.source_decision.action.action_name
                    or "(%.6f,%.6f)" % (
                        result.source_decision.action.v_cmd,
                        result.source_decision.action.omega_cmd,
                    )
                ),
                "target_action": None if result.target_decision is None else (
                    result.target_decision.action.action_name
                    or "(%.6f,%.6f)" % (
                        result.target_decision.action.v_cmd,
                        result.target_decision.action.omega_cmd,
                    )
                ),
                "source_primitive": None if result.source_primitive is None else result.source_primitive.primitive.value,
                "target_primitive": None if result.target_primitive is None else result.target_primitive.primitive.value,
                "support_status": result.provenance.get("support_status"),
            })
    csv_tmp.replace(csv_path)
    return json_path, csv_path
