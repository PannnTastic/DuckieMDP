"""Label-free state counterfactuals and deterministic foil selection."""

from math import hypot
from typing import Any, Dict, Iterable, Mapping, Sequence, Tuple

import numpy as np

from ..action_outcomes import q_action, sac_action
from ..counterfactual import make_counterfactual
from ..schema import CanonicalAction, SolverKind, TABULAR_SOLVERS


CONCEPTS = (
    "lateral", "heading", "speed", "curvature",
    "stop_distance", "stop_satisfied", "duck_risk",
)


def _action_changed(source, target) -> bool:
    if source.solver in TABULAR_SOLVERS:
        return source.action.action_id != target.action.action_id
    dv = abs(target.action.v_cmd - source.action.v_cmd) / 0.41
    dw = abs(target.action.omega_cmd - source.action.omega_cmd) / 1.5
    return hypot(dv, dw) >= 0.10


def _candidate_changes(state, solver: SolverKind):
    numeric = {
        "lateral": ("d", (-0.20, -0.10, -0.05, 0.0, 0.05, 0.10, 0.20), 0.40),
        "heading": ("phi", (-0.80, -0.40, -0.20, 0.0, 0.20, 0.40, 0.80), 1.60),
        "speed": ("v", (0.0, 0.08, 0.17, 0.28, 0.41), 0.41),
        "curvature": ("curvature", (-4.0, -2.0, 0.0, 2.0, 4.0), 8.0),
    }
    for concept, (field, values, scale) in numeric.items():
        current = getattr(state, field)
        current_number = 0.0 if current is None else float(current)
        for value in values:
            if abs(value - current_number) <= 1e-9:
                continue
            yield concept, {field: value}, (value - current_number) / scale

    if state.stop_present and state.stop_distance is not None:
        for value in (0.10, 0.25, 0.50, 0.90, 1.50, 3.00):
            if abs(value - float(state.stop_distance)) > 1e-9:
                yield (
                    "stop_distance", {"stop_distance": value},
                    (value - float(state.stop_distance)) / 3.0,
                )
        yield "stop_satisfied", {"stop_satisfied": not state.stop_satisfied}, 1.0
    else:
        yield "stop_distance", {"stop_distance": 0.30}, 1.0

    yield "duck_risk", {"duck_present": False}, -1.0
    if solver in TABULAR_SOLVERS:
        for threat in ("side_far", "side_near", "crossing_far", "crossing_near"):
            crossing = threat.startswith("crossing")
            yield "duck_risk", {
                "duck_present": True,
                "duck_threat": threat,
                "duck_active": crossing,
                "duck_crossing_available": None,
            }, 1.0
    else:
        for longitudinal, lateral, active in (
            (1.0, 0.5, False), (0.4, 0.2, False),
            (0.8, 0.1, True), (0.2, 0.1, True),
        ):
            yield "duck_risk", {
                "duck_longitudinal": longitudinal,
                "duck_lateral": lateral,
                "duck_v_longitudinal_relative": 0.0,
                "duck_v_lateral_relative": 0.0,
                "duck_active": active,
                "duck_crossing_available": not active,
            }, 1.0


def counterfactual_profile(policy: Any, state) -> Dict[str, Any]:
    source = policy.decide(state)
    best = {concept: None for concept in CONCEPTS}
    attempts = 0
    valid_attempts = 0
    for concept, changes, signed_distance in _candidate_changes(state, source.solver):
        attempts += 1
        record = make_counterfactual(
            state, source.solver, "eddp_%s" % concept, changes
        )
        if not record.validation.valid:
            continue
        valid_attempts += 1
        target = policy.decide(record.state)
        if not _action_changed(source, target):
            continue
        candidate = {
            "distance": abs(float(signed_distance)),
            "signed_delta": float(signed_distance),
            "applied_changes": dict(record.applied_changes),
        }
        if best[concept] is None or candidate["distance"] < best[concept]["distance"]:
            best[concept] = candidate

    result: Dict[str, Any] = {
        "attempts": attempts,
        "valid_attempts": valid_attempts,
        "any_flip": any(value is not None for value in best.values()),
    }
    global_candidates = []
    for concept in CONCEPTS:
        item = best[concept]
        result["%s_flip" % concept] = item is not None
        result["%s_abs_delta" % concept] = 1.0 if item is None else item["distance"]
        result["%s_signed_delta" % concept] = 0.0 if item is None else item["signed_delta"]
        if item is not None:
            global_candidates.append((item["distance"], concept))
    if global_candidates:
        distance, concept = min(global_candidates)
        result["minimum_flip_distance"] = float(distance)
        result["minimum_flip_concept"] = concept
    else:
        result["minimum_flip_distance"] = 1.0
        result["minimum_flip_concept"] = "none"
    return result


def choose_foil(policy: Any, decision):
    """Choose a primary foil before observing simulator outcomes."""

    if decision.solver in TABULAR_SOLVERS:
        values = np.asarray(decision.diagnostics["q_values"], dtype=float)
        allowed = tuple(int(value) for value in decision.diagnostics["allowed_actions"])
        alternatives = [value for value in allowed if value != decision.action.action_id]
        foil_id = max(alternatives, key=lambda value: (values[value], -value))
        return q_action(policy, foil_id), "second_best_table_action"

    lattice = (
        (0.0, 0.0), (0.17, 0.0), (0.41, 0.0),
        (0.17, -1.5), (0.17, 1.5),
        (0.41, -1.5), (0.41, 1.5),
    )
    source = np.asarray([decision.action.v_cmd / 0.41,
                         decision.action.omega_cmd / 1.5])
    choices = []
    for v_cmd, omega_cmd in lattice:
        target = np.asarray([v_cmd / 0.41, omega_cmd / 1.5])
        distance = float(np.linalg.norm(target - source))
        if distance >= 0.10:
            choices.append((distance, v_cmd, omega_cmd))
    _, v_cmd, omega_cmd = min(choices)
    # Carry the deciding solver so the foil matches the selected action's owner.
    foil = CanonicalAction(
        solver=decision.solver,
        v_cmd=float(v_cmd),
        omega_cmd=float(omega_cmd),
    )
    return foil, "nearest_frozen_action_lattice"
