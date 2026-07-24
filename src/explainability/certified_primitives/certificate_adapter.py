"""Adapters from M1--M13 outputs into the C-EDDP instance contract."""

from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple

from ..eddp.schema import ExplanationAtom
from .schema import (
    CertificateStatus,
    CertifiedExplanationInstance,
    SourceKind,
    stable_id,
)


BINDING_GATES = (
    "counterfactual_valid",
    "branch_invariants_pass",
    "paired_outcome_valid",
    "deterministic_policy_mode",
    "teacher_inactive",
    "supported_or_reachable_state",
)


def _all_gates(certificate: Mapping[str, Any]) -> bool:
    return all(bool(certificate.get(name, False)) for name in BINDING_GATES)


def _certificate_from_payload(payload: Mapping[str, Any]) -> Dict[str, Any]:
    validity = dict(payload.get("validity", {}))
    provenance = dict(payload.get("provenance", {}))
    support = dict(payload.get("support", {}))
    result = {
        "counterfactual_valid": bool(
            validity.get("counterfactual_valid", validity.get("valid", False))
        ),
        "branch_invariants_pass": bool(
            validity.get("branch_invariants_pass", False)
        ),
        "paired_outcome_valid": bool(
            validity.get("paired_outcome_valid", False)
        ),
        "deterministic_policy_mode": bool(
            validity.get(
                "deterministic_policy_mode",
                provenance.get("deterministic_policy_mode", False),
            )
        ),
        "teacher_inactive": not bool(
            validity.get(
                "teacher_active",
                provenance.get("teacher_active", True),
            )
        ),
        "supported_or_reachable_state": bool(
            validity.get(
                "supported_or_reachable_state",
                support.get("reachable", False) or support.get("supported", False),
            )
        ),
    }
    result["support_evidence"] = support
    result["binding_gate_pass"] = _all_gates(result)
    return result


def adapt_m1_m13_record(
    payload: Mapping[str, Any],
    *,
    solver: Optional[str] = None,
    seed: Optional[int] = None,
    episode_id: Optional[str] = None,
    step_index: Optional[int] = None,
) -> CertifiedExplanationInstance:
    """Normalize a full M1--M13 record without consuming M2/EDDP labels.

    The adapter accepts the versioned ``ExplanationRecord`` shape as well as
    the combined JSON objects emitted by local and paired explanation scripts.
    Missing binding evidence causes abstention; it is never defaulted to pass.
    """

    decision = dict(payload.get("decision", {}))
    decision_state = dict(decision.get("state", {}))
    provenance = dict(payload.get("provenance", {}))
    resolved_solver = str(
        solver
        if solver is not None
        else decision.get("solver", payload.get("solver", ""))
    )
    resolved_seed = int(
        seed if seed is not None else payload.get("seed", provenance.get("seed", 0))
    )
    resolved_episode = str(
        episode_id
        if episode_id is not None
        else payload.get("episode_id", provenance.get("episode_id", ""))
    )
    resolved_step = int(
        step_index
        if step_index is not None
        else payload.get("step_index", payload.get("decision_step", 0))
    )
    if not resolved_solver or not resolved_episode:
        raise ValueError("M1--M13 record lacks solver or episode provenance")

    certificate = _certificate_from_payload(payload)
    status = (
        CertificateStatus.CERTIFIED
        if certificate["binding_gate_pass"]
        else CertificateStatus.ABSTAINED
    )
    decision_evidence = {
        "state_counterfactual": payload.get("state_counterfactual", {}),
        "counterfactual_profile": payload.get("counterfactual_profile", {}),
        "minimum_intervention": payload.get("minimum_intervention", {}),
    }
    outcome_evidence = {
        "action_outcome_counterfactual": payload.get(
            "action_outcome_counterfactual", payload.get("paired_outcome", {})
        ),
        "physical_profile": payload.get("physical_profile", {}),
        "reward_profile": payload.get("reward_profile", {}),
    }
    verification_evidence = {
        "metamorphic_results": payload.get("metamorphic_results", ()),
        "verification_profile": payload.get("verification_profile", {}),
        "safety_properties": payload.get("safety_properties", {}),
    }
    identity = {
        "solver": resolved_solver,
        "seed": resolved_seed,
        "episode_id": resolved_episode,
        "step_index": resolved_step,
        "certificate_hash_source": provenance.get("manifest_sha256", ""),
    }
    return CertifiedExplanationInstance(
        instance_id=stable_id(identity, "cedp-instance"),
        solver=resolved_solver,
        seed=resolved_seed,
        episode_id=resolved_episode,
        step_index=resolved_step,
        source_kind=SourceKind.FULL_TRAJECTORY,
        status=status,
        decision_evidence=decision_evidence,
        outcome_evidence=outcome_evidence,
        verification_evidence=verification_evidence,
        certificate=certificate,
        provenance=provenance,
        audit_metadata={
            "source_schema_version": payload.get("schema_version"),
            "state_source_representation": decision_state.get(
                "source_representation"
            ),
        },
    )


def adapt_legacy_atom(atom: ExplanationAtom) -> CertifiedExplanationInstance:
    """Adapt EDDP v1 solely for compatibility tests and baseline comparison.

    ``LEGACY_SPARSE`` is deliberately ineligible for main C-EDDP discovery.
    """

    validity = dict(atom.validity)
    certificate = {
        "counterfactual_valid": float(
            validity.get("counterfactual_valid_fraction", 0.0)
        ) > 0.0,
        "branch_invariants_pass": bool(
            validity.get("branch_invariants_pass", False)
        ),
        "paired_outcome_valid": bool(
            validity.get("paired_outcome_valid", False)
        ),
        "deterministic_policy_mode": True,
        "teacher_inactive": True,
        "supported_or_reachable_state": True,
    }
    certificate["binding_gate_pass"] = _all_gates(certificate)
    status = (
        CertificateStatus.CERTIFIED
        if certificate["binding_gate_pass"]
        else CertificateStatus.ABSTAINED
    )
    return CertifiedExplanationInstance(
        instance_id=stable_id({"legacy_atom_id": atom.atom_id}, "cedp-legacy"),
        solver=atom.solver.value,
        seed=int(atom.seed),
        episode_id=str(atom.episode_id),
        step_index=int(atom.decision_step),
        source_kind=SourceKind.LEGACY_SPARSE,
        status=status,
        decision_evidence={"counterfactual_profile": dict(atom.counterfactual_profile)},
        outcome_evidence={
            "physical_profile": dict(atom.physical_profile),
            "reward_profile": dict(atom.reward_profile),
        },
        verification_evidence={
            "verification_profile": dict(atom.verification_profile)
        },
        certificate=certificate,
        provenance={
            "paired_report_path": atom.paired_report_path,
            "source_atom_id": atom.atom_id,
            "source_anchor_id": atom.anchor_id,
        },
        audit_metadata={
            "selection_context": atom.selection_context,
            "observed_context": atom.observed_context,
            "legacy_block_id": atom.block_id,
            "legacy_block_offset": atom.block_offset,
        },
    )


def partition_instances(
    records: Iterable[CertifiedExplanationInstance],
) -> Tuple[Tuple[CertifiedExplanationInstance, ...], Tuple[CertifiedExplanationInstance, ...]]:
    certified, abstained = [], []
    for record in records:
        (certified if record.status == CertificateStatus.CERTIFIED else abstained).append(record)
    return tuple(certified), tuple(abstained)
