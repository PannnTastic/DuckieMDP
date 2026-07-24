"""Label-free physical-context anchor collection."""

from collections import Counter
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np

from .schema import AnchorRecord, stable_id
from ..action_outcomes import _environment_action
from ..q_policy_adapter import QPolicyAdapter
from ..sac_policy_adapter import SACPolicyAdapter
from ..schema import SolverKind, TABULAR_SOLVERS, to_dict
from ..semantic_state import canonical_from_continuous_state


PHYSICAL_CONTEXTS = ("duck", "stop", "lane", "nominal")


def physical_context(state) -> str:
    """Classify observable context without consulting a primitive label."""

    if state.duck_present:
        metric_near = (
            state.duck_longitudinal is not None
            and abs(float(state.duck_longitudinal)) <= 1.0
        )
        categorical_risk = state.duck_threat in {
            "side_near", "crossing_far", "crossing_near"
        }
        if state.duck_active or metric_near or categorical_risk:
            return "duck"
    if (
        state.stop_present
        and not state.stop_satisfied
        and state.stop_distance is not None
        and float(state.stop_distance) <= 0.90
    ):
        return "stop"
    if abs(float(state.d)) >= 0.07 or abs(float(state.phi)) >= 0.12:
        return "lane"
    return "nominal"


def _decision(policy, env, raw):
    if isinstance(policy, QPolicyAdapter):
        return policy.decide_raw(raw)
    if isinstance(policy, SACPolicyAdapter):
        return policy.decide(canonical_from_continuous_state(env.current_state))
    raise TypeError("unsupported policy %s" % type(policy).__name__)


def _prefix_value(decision):
    if decision.solver in TABULAR_SOLVERS:
        return int(decision.action.action_id)
    return [float(decision.action.v_cmd), float(decision.action.omega_cmd)]


def collect_episode_anchors(
    env: Any,
    policy: Any,
    seed: int,
    config_path: str,
    checkpoint_path: str,
    blocks_per_context: int = 2,
    block_length: int = 3,
    minimum_block_gap: int = 8,
    max_decisions: int = 250,
) -> Tuple[Tuple[AnchorRecord, ...], Mapping[str, int]]:
    """Collect short real-decision blocks using physical contexts only."""

    raw = env.reset(int(seed))
    prefix: List[Any] = []
    records: List[AnchorRecord] = []
    counts = Counter()
    last_block_start = -10 ** 9
    active_context = None
    active_block = None
    remaining = 0
    block_offset = 0
    episode_id = "%s_%d" % (policy.solver_kind.value if isinstance(policy, QPolicyAdapter)
                              else "sac", int(seed))

    for step in range(int(max_decisions)):
        decision = _decision(policy, env, raw)
        observed = physical_context(decision.state)
        if remaining <= 0:
            eligible = (
                counts[observed] < int(blocks_per_context)
                and step - last_block_start >= int(minimum_block_gap)
            )
            if eligible:
                active_context = observed
                active_block = "%s-%s-%02d" % (
                    episode_id, observed, counts[observed]
                )
                counts[observed] += 1
                last_block_start = step
                remaining = int(block_length)
                block_offset = 0

        if remaining > 0:
            identity = {
                "solver": decision.solver.value,
                "seed": int(seed),
                "episode": episode_id,
                "step": int(step),
                "block": active_block,
            }
            records.append(AnchorRecord(
                anchor_id=stable_id(identity, "anchor"),
                solver=decision.solver,
                seed=int(seed),
                episode_id=episode_id,
                decision_step=int(step),
                block_id=str(active_block),
                block_offset=int(block_offset),
                selection_context=str(active_context),
                observed_context=observed,
                state=decision.state,
                selected_action=decision.action,
                action_prefix=tuple(prefix),
                config_path=str(config_path),
                checkpoint_path=str(checkpoint_path),
                policy_mode=decision.policy_mode.value,
            ))
            remaining -= 1
            block_offset += 1

        action_value = _prefix_value(decision)
        next_raw, _, done, _ = env.step(_environment_action(decision.action))
        prefix.append(action_value)
        raw = next_raw
        if done:
            break
        if all(counts[name] >= int(blocks_per_context) for name in PHYSICAL_CONTEXTS) \
                and remaining <= 0:
            break

    return tuple(records), {
        name: int(counts[name]) for name in PHYSICAL_CONTEXTS
    }
