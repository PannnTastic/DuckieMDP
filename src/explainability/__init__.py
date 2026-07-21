"""Explainable-policy tools with lazy solver-specific dependencies."""

from typing import TYPE_CHECKING

from .schema import (
    SCHEMA_VERSION,
    CanonicalAction,
    CanonicalState,
    ExplanationRecord,
    PolicyDecision,
    PolicyMode,
    SolverKind,
    to_dict,
    to_json,
)

if TYPE_CHECKING:
    from .primitives import (
        DrivingPrimitive,
        PrimitiveLabel,
        PrimitiveLabeler,
        PrimitiveThresholds,
    )
    from .q_policy_adapter import QPolicyAdapter
    from .sac_policy_adapter import SACPolicyAdapter

__all__ = (
    "SCHEMA_VERSION",
    "CanonicalAction",
    "CanonicalState",
    "ExplanationRecord",
    "PolicyDecision",
    "PolicyMode",
    "DrivingPrimitive",
    "PrimitiveLabel",
    "PrimitiveLabeler",
    "PrimitiveThresholds",
    "QPolicyAdapter",
    "SACPolicyAdapter",
    "SolverKind",
    "to_dict",
    "to_json",
)


def __getattr__(name: str):
    if name in {
        "DrivingPrimitive",
        "PrimitiveLabel",
        "PrimitiveLabeler",
        "PrimitiveThresholds",
    }:
        from . import primitives

        return getattr(primitives, name)
    if name == "QPolicyAdapter":
        from .q_policy_adapter import QPolicyAdapter

        return QPolicyAdapter
    if name == "SACPolicyAdapter":
        from .sac_policy_adapter import SACPolicyAdapter

        return SACPolicyAdapter
    raise AttributeError(name)
