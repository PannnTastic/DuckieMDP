"""Full-trajectory collection and M1--M13 explanation evaluation."""

import json
from pathlib import Path
from typing import Any, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from ..action_outcomes import (
    EventHorizonConfig,
    _environment_action,
    prepare_branch,
    run_paired_outcomes,
)
from ..eddp.counterfactual_profile import choose_foil, counterfactual_profile
from ..eddp.runtime import environment_factory
from ..eddp.signature import physical_profile_from_report, reward_profile_from_report
from ..eddp.support import SupportOracle
from ..eddp.verification import verification_profile
from ..q_policy_adapter import QPolicyAdapter
from ..sac_policy_adapter import SACPolicyAdapter
from ..td3_policy_adapter import TD3PolicyAdapter
from ..schema import TABULAR_SOLVERS, to_dict
from ..semantic_state import canonical_from_continuous_state
from .certificate_adapter import adapt_m1_m13_record
from .schema import (
    CertifiedExplanationInstance,
    FullDecisionAnchor,
    stable_id,
)


def _decision(policy: Any, env: Any, raw: Any):
    if isinstance(policy, QPolicyAdapter):
        return policy.decide_raw(raw)
    if isinstance(policy, (SACPolicyAdapter, TD3PolicyAdapter)):
        return policy.decide(canonical_from_continuous_state(env.current_state))
    raise TypeError("unsupported policy %s" % type(policy).__name__)


def _prefix_value(decision):
    if decision.solver in TABULAR_SOLVERS:
        return int(decision.action.action_id)
    return [float(decision.action.v_cmd), float(decision.action.omega_cmd)]


def collect_full_decision_anchors(
    env: Any,
    policy: Any,
    seed: int,
    config_path: Path,
    checkpoint_path: Path,
    *,
    max_decisions: int = 250,
) -> Tuple[FullDecisionAnchor, ...]:
    """Record every real policy decision, without context or primitive sampling."""

    raw = env.reset(int(seed))
    prefix: List[Any] = []
    records = []
    solver_name = (
        policy.solver_kind.value
        if hasattr(policy, "solver_kind")
        else "sac"
    )
    episode_id = "%s_%d" % (solver_name, int(seed))
    for step in range(int(max_decisions)):
        decision = _decision(policy, env, raw)
        identity = {
            "solver": decision.solver.value,
            "seed": int(seed),
            "episode_id": episode_id,
            "step_index": int(step),
        }
        records.append(FullDecisionAnchor(
            anchor_id=stable_id(identity, "cedp-anchor"),
            solver=decision.solver,
            seed=int(seed),
            episode_id=episode_id,
            step_index=int(step),
            state=decision.state,
            selected_action=decision.action,
            action_prefix=tuple(prefix),
            config_path=str(config_path),
            checkpoint_path=str(checkpoint_path),
            policy_mode=decision.policy_mode.value,
        ))
        action_value = _prefix_value(decision)
        next_raw, _, done, _ = env.step(_environment_action(decision.action))
        prefix.append(action_value)
        raw = next_raw
        if done:
            break
    return tuple(records)


def explain_anchor(
    anchor: FullDecisionAnchor,
    *,
    shared_environment: Mapping[str, Any],
    shared_config_path: Path,
    policy: Any,
    gamma: float,
    max_horizon: int,
    fixed_horizons: Sequence[int],
    support_oracle: SupportOracle,
    event_horizon: bool = False,
    provenance_manifest_sha256: str = "",
    paired_report_path: Optional[Path] = None,
) -> CertifiedExplanationInstance:
    """Run the existing three-pillar explanation machinery at one timestep."""

    factory = environment_factory(shared_environment, anchor.solver.value)
    prepared = prepare_branch(
        env_factory=factory,
        reset_seed=anchor.seed,
        action_prefix=anchor.action_prefix,
        policy=policy,
        config_path=shared_config_path,
        checkpoint_path=Path(anchor.checkpoint_path),
    )
    foil, foil_protocol = choose_foil(policy, prepared.selected_decision)
    report = run_paired_outcomes(
        env_factory=factory,
        prepared=prepared,
        policy=policy,
        foil_action=foil,
        max_horizon=int(max_horizon),
        gamma=float(gamma),
        fixed_horizons=tuple(int(value) for value in fixed_horizons),
        event_horizon=EventHorizonConfig(enabled=bool(event_horizon)),
    )
    report_payload = to_dict(report)
    if paired_report_path is not None:
        destination = Path(paired_report_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(destination.suffix + ".tmp")
        temporary.write_text(
            json.dumps(report_payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        temporary.replace(destination)
    state_counterfactual = counterfactual_profile(policy, anchor.state)
    source_support = support_oracle.classify(policy, anchor.state)
    if not source_support.reachable:
        raise RuntimeError("collected anchor is missing from support oracle")
    verification = verification_profile(
        policy, anchor.state, support_oracle=support_oracle
    )
    invariants = dict(report.branch_invariants)
    invariant_pass = all(
        bool(value) for name, value in invariants.items()
        if name != "teacher_active"
    ) and not bool(invariants.get("teacher_active", True))
    attempts = int(state_counterfactual.get("attempts", 0))
    valid_attempts = int(state_counterfactual.get("valid_attempts", 0))
    payload = {
        "schema_version": "m1-m13-combined-v2-support-aware",
        "counterfactual_profile": state_counterfactual,
        "physical_profile": physical_profile_from_report(report_payload),
        "reward_profile": reward_profile_from_report(report_payload),
        "verification_profile": verification,
        "action_outcome_counterfactual": {
            "manifest_id": report_payload.get("manifest_id"),
            "foil_protocol": foil_protocol,
            "physical_delta_counterfactual_minus_factual": report_payload.get(
                "physical_delta_counterfactual_minus_factual", {}
            ),
            "reward_delta_counterfactual_minus_factual": report_payload.get(
                "reward_delta_counterfactual_minus_factual", {}
            ),
        },
        "metamorphic_results": verification,
        "validity": {
            "counterfactual_valid": attempts > 0 and valid_attempts == attempts,
            "branch_invariants_pass": invariant_pass,
            "paired_outcome_valid": invariant_pass,
            "deterministic_policy_mode": True,
            "teacher_active": False,
        },
        "support": source_support.as_dict(),
        "provenance": {
            "manifest_sha256": provenance_manifest_sha256,
            "paired_manifest_id": report_payload.get("manifest_id"),
            "checkpoint_path": anchor.checkpoint_path,
            "config_path": anchor.config_path,
            "policy_mode": anchor.policy_mode,
            "teacher_active": False,
            "deterministic_policy_mode": True,
            "support_basis": source_support.basis,
            "paired_report_path": (
                None if paired_report_path is None else str(paired_report_path)
            ),
        },
    }
    return adapt_m1_m13_record(
        payload,
        solver=anchor.solver.value,
        seed=anchor.seed,
        episode_id=anchor.episode_id,
        step_index=anchor.step_index,
    )
