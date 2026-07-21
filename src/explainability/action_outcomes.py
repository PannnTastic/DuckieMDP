"""COViz-inspired factual-versus-foil paired simulator rollouts."""

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable, Optional, Sequence, Tuple

import numpy as np

from ..actions import build_action_table
from .primitives import DrivingPrimitive, PrimitiveLabeler, PrimitiveThresholds
from .q_policy_adapter import QPolicyAdapter
from .sac_policy_adapter import SACPolicyAdapter
from .scenario_manifest import (
    ScenarioManifest,
    WorldMode,
    capture_manifest,
)
from .schema import CanonicalAction, PolicyDecision, SolverKind, TABULAR_SOLVERS
from .semantic_state import canonical_from_continuous_state
from .temporal_outcomes import (
    DEFAULT_FIXED_HORIZONS,
    BranchOutcome,
    PairedOutcomeReport,
    build_explanation_text,
    compute_physical_outcome,
    compute_reward_profile,
    final_reward_delta,
    physical_delta,
)
from .trajectory import TrajectoryRecorder


@dataclass(frozen=True)
class EventHorizonConfig:
    enabled: bool = False
    minimum_steps: int = 3
    stop_on_termination: bool = True
    stop_on_lane_stable: bool = True
    stop_on_stop_event: bool = True
    stop_on_duck_clear: bool = True
    stop_on_primitive_change: bool = False
    lane_abs_d: float = 0.05
    lane_abs_phi: float = 0.10


@dataclass(frozen=True)
class PreparedBranch:
    manifest: ScenarioManifest
    selected_decision: PolicyDecision


def q_action(policy: QPolicyAdapter, action_id: int) -> CanonicalAction:
    action = int(action_id)
    if not 0 <= action < len(policy.action_table):
        raise ValueError("Q action id must be in [0, 6]")
    spec = policy.action_table[action]
    return CanonicalAction(
        solver=policy.solver_kind,
        action_id=action,
        action_name=spec.name,
        v_cmd=float(spec.v),
        omega_cmd=float(spec.omega),
    )


def sac_action(v_cmd: float, omega_cmd: float) -> CanonicalAction:
    return CanonicalAction(
        solver=SolverKind.SAC,
        v_cmd=float(v_cmd),
        omega_cmd=float(omega_cmd),
    )


def _policy_decision(policy: Any, env: Any) -> PolicyDecision:
    if isinstance(policy, QPolicyAdapter):
        raw = getattr(env, "_last_state", None)
        if raw is None and hasattr(env, "mdp_env"):
            raw = env.mdp_env._last_state
        if raw is None:
            raise RuntimeError("Q environment has no current RawState")
        return policy.decide_raw(raw)
    if isinstance(policy, SACPolicyAdapter):
        if not hasattr(env, "current_state"):
            raise TypeError("SAC policy requires ContinuousDuckieMDPEnv")
        return policy.decide(canonical_from_continuous_state(env.current_state))
    raise TypeError("unsupported policy adapter: %s" % type(policy).__name__)


def _environment_action(action: CanonicalAction) -> Any:
    if action.solver in TABULAR_SOLVERS:
        if action.action_id is None:
            raise ValueError("Q action requires action_id")
        return int(action.action_id)
    return np.asarray([action.v_cmd, action.omega_cmd], dtype=np.float32)


def _canonical_prefix_action(
    raw_action: Any,
    solver: SolverKind,
    env: Any,
) -> CanonicalAction:
    if solver in TABULAR_SOLVERS:
        action_id = int(raw_action)
        owner = getattr(env, "mdp_env", env)
        table = build_action_table(owner.action_cfg)
        spec = table[action_id]
        return CanonicalAction(
            solver=solver,
            action_id=action_id,
            action_name=spec.name,
            v_cmd=float(spec.v),
            omega_cmd=float(spec.omega),
        )
    values = np.asarray(raw_action, dtype=float).reshape(-1)
    if values.size != 2:
        raise ValueError("SAC prefix action must contain v_cmd and omega_cmd")
    return sac_action(values[0], values[1])


def _execute_prefix(
    env: Any,
    policy: Any,
    solver: SolverKind,
    action_prefix: Sequence[Any],
    labeler: Optional[PrimitiveLabeler] = None,
) -> None:
    for index, raw_action in enumerate(action_prefix):
        policy_decision = _policy_decision(policy, env)
        action = _canonical_prefix_action(raw_action, solver, env)
        executed = replace(
            policy_decision,
            action=action,
            metadata={
                **dict(policy_decision.metadata),
                "executed_action_source": "manifest_prefix",
            },
        )
        _, _, done, info = env.step(_environment_action(action))
        if labeler is not None:
            labeler.label(executed, info.get("events"), info["termination_reason"])
        if done:
            raise RuntimeError(
                "action prefix terminated before branch at prefix index %d" % index
            )


def prepare_branch(
    env_factory: Callable[[int], Any],
    reset_seed: int,
    action_prefix: Sequence[Any],
    policy: Any,
    config_path: Path,
    checkpoint_path: Path,
    world_mode: WorldMode = WorldMode.REACTIVE_WORLD,
) -> PreparedBranch:
    env = env_factory(int(reset_seed))
    try:
        env.reset(int(reset_seed))
        selected = _policy_decision(policy, env)
        _execute_prefix(env, policy, selected.solver, action_prefix)
        selected = _policy_decision(policy, env)
        manifest = capture_manifest(
            env=env,
            reset_seed=reset_seed,
            action_prefix=action_prefix,
            solver=selected.solver,
            policy_mode=selected.policy_mode,
            config_path=config_path,
            checkpoint_path=checkpoint_path,
            world_mode=world_mode,
        )
        return PreparedBranch(manifest=manifest, selected_decision=selected)
    finally:
        env.close()


def _same_action(left: CanonicalAction, right: CanonicalAction) -> bool:
    if left.solver != right.solver:
        return False
    if left.solver in TABULAR_SOLVERS:
        return left.action_id == right.action_id
    return bool(
        np.allclose(
            [left.v_cmd, left.omega_cmd],
            [right.v_cmd, right.omega_cmd],
            atol=1e-7,
            rtol=0.0,
        )
    )


def _next_canonical(policy: Any, env: Any):
    return _policy_decision(policy, env).state


def _event_horizon_reason(
    config: EventHorizonConfig,
    step_count: int,
    initial_state: Any,
    current_state: Any,
    first_primitive: DrivingPrimitive,
    current_primitive: DrivingPrimitive,
    info: Any,
) -> Optional[str]:
    if not config.enabled or step_count < max(1, config.minimum_steps):
        return None
    if config.stop_on_termination and (
        info.get("terminated", False) or info.get("truncated", False)
    ):
        return "termination"
    events = info.get("events", {})
    if config.stop_on_stop_event and initial_state.stop_present and any(
        events.get(name, False)
        for name in ("full_stop", "passed_stop", "stop_violation")
    ):
        return "stop_event"
    initial_duck_risk = initial_state.duck_active is True or (
        initial_state.duck_threat in {"crossing_far", "crossing_near"}
    )
    current_duck_risk = current_state.duck_active is True or (
        current_state.duck_threat in {"crossing_far", "crossing_near"}
    )
    if config.stop_on_duck_clear and initial_duck_risk and not current_duck_risk:
        return "duck_clear"
    lane_only = not initial_state.stop_present and not initial_state.duck_present
    if config.stop_on_lane_stable and lane_only and (
        abs(current_state.d) <= config.lane_abs_d
        and abs(current_state.phi) <= config.lane_abs_phi
    ):
        return "lane_stable"
    if config.stop_on_primitive_change and current_primitive != first_primitive:
        return "primitive_change"
    return None


@dataclass(frozen=True)
class _BranchExecution:
    outcome: BranchOutcome
    selected_decision: PolicyDecision
    reconstructed_manifest_id: str


def _run_branch(
    role: str,
    env_factory: Callable[[int], Any],
    manifest: ScenarioManifest,
    policy: Any,
    first_action: CanonicalAction,
    max_horizon: int,
    gamma: float,
    fixed_horizons: Sequence[int],
    event_horizon: EventHorizonConfig,
    primitive_thresholds: PrimitiveThresholds,
) -> _BranchExecution:
    if max_horizon <= 0:
        raise ValueError("max_horizon must be positive")
    env = env_factory(manifest.reset_seed)
    labeler = PrimitiveLabeler(primitive_thresholds)
    try:
        env.reset(manifest.reset_seed)
        _execute_prefix(
            env,
            policy,
            manifest.solver,
            manifest.action_prefix,
            labeler=labeler,
        )
        reconstructed = capture_manifest(
            env=env,
            reset_seed=manifest.reset_seed,
            action_prefix=manifest.action_prefix,
            solver=manifest.solver,
            policy_mode=manifest.policy_mode,
            config_path=Path(manifest.config_path),
            checkpoint_path=Path(manifest.checkpoint_path),
            world_mode=manifest.world_mode,
            exogenous_trace=manifest.exogenous_trace,
        )
        if reconstructed.manifest_id != manifest.manifest_id:
            raise AssertionError("reconstructed branch point does not match manifest")

        selected_at_branch = _policy_decision(policy, env)
        initial_state = selected_at_branch.state
        base = env.unwrapped
        decision_dt = float(base.delta_time * base.frame_skip)
        recorder = TrajectoryRecorder(
            episode_id="%s-%s" % (manifest.manifest_id[:12], role),
            provenance={
                "manifest_id": manifest.manifest_id,
                "checkpoint_sha256": manifest.checkpoint_sha256,
                "config_sha256": manifest.config_sha256,
                "world_mode": manifest.world_mode.value,
                "branch_role": role,
            },
            decision_dt_seconds=decision_dt,
        )
        action_sources = []
        first_primitive = None
        horizon_reason = None
        for index in range(max_horizon):
            policy_decision = _policy_decision(policy, env)
            if index == 0:
                executed_action = first_action
                source = "selected_policy" if role == "factual" else "forced_foil"
            else:
                executed_action = policy_decision.action
                source = "policy"
            executed = replace(
                policy_decision,
                action=executed_action,
                metadata={
                    **dict(policy_decision.metadata),
                    "executed_action_source": source,
                    "policy_proposed_action": to_action_dict(policy_decision.action),
                },
            )
            position = (float(base.cur_pos[0]), float(base.cur_pos[2]))
            heading = float(base.cur_angle)
            _, reward, done, info = env.step(_environment_action(executed_action))
            primitive = labeler.label(
                executed,
                info.get("events"),
                info["termination_reason"],
            )
            if first_primitive is None:
                first_primitive = primitive.primitive
            action_sources.append(source)
            recorder.append(
                decision=executed,
                primitive=primitive,
                reward=reward,
                info=info,
                physics_step=int(base.step_count),
                position_xz=position,
                heading_radians=heading,
                extra={"branch_role": role, "action_source": source},
            )
            if done:
                horizon_reason = "termination"
                break
            current_state = _next_canonical(policy, env)
            horizon_reason = _event_horizon_reason(
                event_horizon,
                index + 1,
                initial_state,
                current_state,
                first_primitive,
                primitive.primitive,
                info,
            )
            if horizon_reason is not None:
                break

        trajectory = recorder.finalize()
        reward_profile = compute_reward_profile(
            trajectory, gamma=gamma, horizons=fixed_horizons
        )
        physical = compute_physical_outcome(
            trajectory,
            decision_dt_seconds=decision_dt,
            brake_command_threshold=primitive_thresholds.hold_command_speed,
        )
        return _BranchExecution(
            outcome=BranchOutcome(
                role=role,
                first_action=first_action,
                first_primitive=first_primitive.value,
                action_source_sequence=tuple(action_sources),
                reward_profile=reward_profile,
                physical=physical,
                trajectory=trajectory,
                event_horizon_reason=horizon_reason,
            ),
            selected_decision=selected_at_branch,
            reconstructed_manifest_id=reconstructed.manifest_id,
        )
    finally:
        env.close()


def to_action_dict(action: CanonicalAction):
    return {
        "solver": action.solver.value,
        "action_id": action.action_id,
        "action_name": action.action_name,
        "v_cmd": action.v_cmd,
        "omega_cmd": action.omega_cmd,
    }


def run_paired_outcomes(
    env_factory: Callable[[int], Any],
    prepared: PreparedBranch,
    policy: Any,
    foil_action: CanonicalAction,
    max_horizon: int = 30,
    gamma: float = 0.99,
    fixed_horizons: Sequence[int] = DEFAULT_FIXED_HORIZONS,
    event_horizon: EventHorizonConfig = EventHorizonConfig(),
    primitive_thresholds: PrimitiveThresholds = PrimitiveThresholds(),
) -> PairedOutcomeReport:
    selected = prepared.selected_decision
    if foil_action.solver != selected.solver:
        raise ValueError("foil and selected action must use the same solver")
    if _same_action(foil_action, selected.action):
        raise ValueError("foil action must differ from selected action")

    factual_execution = _run_branch(
        "factual",
        env_factory,
        prepared.manifest,
        policy,
        selected.action,
        max_horizon,
        gamma,
        fixed_horizons,
        event_horizon,
        primitive_thresholds,
    )
    counter_execution = _run_branch(
        "counterfactual",
        env_factory,
        prepared.manifest,
        policy,
        foil_action,
        max_horizon,
        gamma,
        fixed_horizons,
        event_horizon,
        primitive_thresholds,
    )
    if not _same_action(factual_execution.selected_decision.action, selected.action):
        raise AssertionError("factual reconstruction changed selected policy action")
    if not _same_action(counter_execution.selected_decision.action, selected.action):
        raise AssertionError("counterfactual reconstruction changed selected policy action")

    factual = factual_execution.outcome
    counterfactual = counter_execution.outcome
    invariants = {
        "same_manifest": (
            factual_execution.reconstructed_manifest_id
            == counter_execution.reconstructed_manifest_id
            == prepared.manifest.manifest_id
        ),
        "same_policy_selected_action_at_branch": True,
        "only_first_action_forced": (
            factual.action_source_sequence[0] == "selected_policy"
            and counterfactual.action_source_sequence[0] == "forced_foil"
            and all(value == "policy" for value in factual.action_source_sequence[1:])
            and all(
                value == "policy" for value in counterfactual.action_source_sequence[1:]
            )
        ),
        "selected_and_foil_differ": True,
        "teacher_active": False,
    }
    explanation = build_explanation_text(
        selected.action,
        foil_action,
        factual,
        counterfactual,
    )
    return PairedOutcomeReport(
        manifest_id=prepared.manifest.manifest_id,
        world_mode=prepared.manifest.world_mode.value,
        selected_decision=selected,
        foil_action=foil_action,
        factual=factual,
        counterfactual=counterfactual,
        physical_delta_counterfactual_minus_factual=physical_delta(
            factual.physical, counterfactual.physical
        ),
        reward_delta_counterfactual_minus_factual=final_reward_delta(
            factual.reward_profile, counterfactual.reward_profile
        ),
        branch_invariants=invariants,
        single_rollout_is_probability=False,
        explanation=explanation,
    )
