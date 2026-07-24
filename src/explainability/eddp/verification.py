"""Primitive-label-free, support-aware local metamorphic checks.

An applicable relation is claimable only when both the observed source and
the synthetic target lie inside the frozen empirical support contract. A
semantically valid but unsupported intervention is retained as ``ABSTAIN``;
it is never silently converted into either a pass or a failure.
"""

from typing import Any, Dict, Optional

from ..counterfactual import make_counterfactual
from ..schema import TABULAR_SOLVERS
from ..metamorphic import speed_level
from .support import SupportEvidence, SupportOracle, pair_stratum


RELATIONS = ("stop", "pedestrian", "curvature", "lane_symmetry")


def _speed_not_increased(source, target) -> bool:
    if source.solver in TABULAR_SOLVERS:
        return speed_level(target) <= speed_level(source)
    return target.action.v_cmd <= source.action.v_cmd + 0.01


def _unknown_support(policy: Any) -> SupportEvidence:
    kind = getattr(policy, "solver_kind", "unknown")
    solver = str(getattr(kind, "value", kind))
    return SupportEvidence(
        solver=solver,
        stratum="unknown",
        basis="support_oracle_not_provided",
        reachable=False,
        supported=False,
    )


def _evaluate(
    policy: Any,
    state,
    relation: str,
    support_oracle: Optional[SupportOracle] = None,
) -> Dict[str, Any]:
    source = policy.decide(state)
    changes = None
    if relation == "stop":
        applicable = (
            state.stop_present and not state.stop_satisfied
            and state.stop_distance is not None and state.stop_distance > 0.12
            and not state.duck_present and state.curvature_class == "straight"
            and abs(state.d) <= 0.10 and abs(state.phi) <= 0.30
        )
        if applicable:
            changes = {"stop_distance": max(0.10, state.stop_distance * 0.5)}
    elif relation == "pedestrian":
        applicable = not (state.stop_present and not state.stop_satisfied)
        if applicable:
            if source.solver in TABULAR_SOLVERS:
                ordering = [
                    "side_far", "side_near", "crossing_far", "crossing_near"
                ]
                current = state.duck_threat if state.duck_present else "none"
                index = ordering.index(current) if current in ordering else -1
                if index < len(ordering) - 1:
                    threat = ordering[index + 1]
                    active = threat.startswith("crossing")
                    changes = {
                        "duck_present": True,
                        "duck_threat": threat,
                        "duck_active": active,
                        "duck_crossing_available": None,
                    }
            else:
                changes = {
                    "duck_longitudinal": 0.2,
                    "duck_lateral": 0.1,
                    "duck_v_longitudinal_relative": 0.0,
                    "duck_v_lateral_relative": 0.0,
                    "duck_active": True,
                    "duck_crossing_available": False,
                }
    elif relation == "curvature":
        applicable = (
            not state.stop_present and not state.duck_present
            and state.curvature_class == "straight"
            and abs(state.d) <= 0.10 and abs(state.phi) <= 0.30
        )
        if applicable:
            changes = {"curvature": 2.0}
    elif relation == "lane_symmetry":
        applicable = (
            not state.stop_present and not state.duck_present
            and state.curvature_class == "straight"
            and (abs(state.d) > 1e-7 or abs(state.phi) > 1e-7)
        )
        if applicable:
            changes = {"d": -state.d, "phi": -state.phi}
    else:
        raise KeyError(relation)

    if changes is None:
        return {
            "status": "NOT_APPLICABLE",
            "applicable": False,
            "eligible": False,
            "raw_pass": False,
            "reason": "relation_precondition_false",
        }

    target_record = make_counterfactual(
        state, source.solver, "eddp_mr_%s" % relation, changes
    )
    if not target_record.validation.valid:
        return {
            "status": "ABSTAIN",
            "applicable": True,
            "eligible": False,
            "raw_pass": False,
            "reason": "counterfactual_target_invalid",
        }

    target = policy.decide(target_record.state)
    if relation == "lane_symmetry":
        if source.solver in TABULAR_SOLVERS:
            name = source.action.action_name
            expected = (
                name[:-5] + "_right" if name.endswith("_left") else
                name[:-6] + "_left" if name.endswith("_right") else name
            )
            passed = target.action.action_name == expected
        else:
            passed = (
                abs(target.action.omega_cmd + source.action.omega_cmd) <= 0.15
            )
    else:
        passed = _speed_not_increased(source, target)

    if support_oracle is None:
        source_support = target_support = _unknown_support(policy)
    else:
        source_support = support_oracle.classify(policy, state)
        target_support = support_oracle.classify(
            policy, target_record.state
        )
    pair = pair_stratum(source_support, target_support)
    eligible = pair == "both_supported"
    return {
        "status": (
            "PASS" if passed else "FAIL"
        ) if eligible else "ABSTAIN",
        "applicable": True,
        "eligible": eligible,
        "raw_pass": bool(passed),
        "reason": (
            "both_states_empirically_supported"
            if eligible
            else "pair_outside_empirical_support"
        ),
        "pair_stratum": pair,
        "source_support": source_support.as_dict(),
        "target_support": target_support.as_dict(),
    }


def verification_profile(
    policy: Any,
    state,
    support_oracle: Optional[SupportOracle] = None,
) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    for relation in RELATIONS:
        evidence = _evaluate(policy, state, relation, support_oracle)
        status = str(evidence["status"])
        result["%s_applicable" % relation] = bool(
            evidence["applicable"]
        )
        result["%s_eligible" % relation] = bool(evidence["eligible"])
        result["%s_pass" % relation] = status == "PASS"
        result["%s_fail" % relation] = status == "FAIL"
        result["%s_abstain" % relation] = status == "ABSTAIN"
        result["%s_status" % relation] = status
        result["%s_raw_pass" % relation] = bool(
            evidence.get("raw_pass", False)
        )
        result["%s_reason" % relation] = evidence.get("reason")
        result["%s_pair_stratum" % relation] = evidence.get(
            "pair_stratum"
        )
        result["%s_source_support" % relation] = evidence.get(
            "source_support", {}
        )
        result["%s_target_support" % relation] = evidence.get(
            "target_support", {}
        )
    return result
