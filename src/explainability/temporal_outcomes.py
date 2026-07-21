"""Temporal reward and semantic physical-outcome profiles for M5."""

from dataclasses import dataclass
import json
from math import cos, hypot
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np

from .schema import CanonicalAction, PolicyDecision, to_dict
from .trajectory import TrajectoryRecord


OUTCOME_SCHEMA_VERSION = "1.0.0"
DEFAULT_FIXED_HORIZONS = (1, 5, 10, 20, 30)


@dataclass(frozen=True)
class RewardHorizonPoint:
    horizon_steps: int
    discounted_total: float
    undiscounted_total: float
    discounted_terms: Mapping[str, float]
    undiscounted_terms: Mapping[str, float]


@dataclass(frozen=True)
class PhysicalOutcomeProfile:
    steps: int
    elapsed_seconds: Optional[float]
    forward_progress_m: float
    mean_abs_lateral_error_m: float
    max_abs_lateral_error_m: float
    mean_abs_heading_error_rad: float
    max_abs_heading_error_rad: float
    minimum_duck_clearance_m: Optional[float]
    stop_violations: int
    full_stops: int
    passed_stops: int
    brake_steps: int
    brake_ratio: float
    steering_reversals: int
    cumulative_steering_jerk: float
    lane_departure: bool
    duck_collision: bool
    other_collision: bool
    termination_reason: str
    primitive_sequence: Tuple[str, ...]
    primitive_transition_steps: Tuple[int, ...]


@dataclass(frozen=True)
class BranchOutcome:
    role: str
    first_action: CanonicalAction
    first_primitive: str
    action_source_sequence: Tuple[str, ...]
    reward_profile: Tuple[RewardHorizonPoint, ...]
    physical: PhysicalOutcomeProfile
    trajectory: TrajectoryRecord
    event_horizon_reason: Optional[str]


@dataclass(frozen=True)
class PairedOutcomeReport:
    manifest_id: str
    world_mode: str
    selected_decision: PolicyDecision
    foil_action: CanonicalAction
    factual: BranchOutcome
    counterfactual: BranchOutcome
    physical_delta_counterfactual_minus_factual: Mapping[str, Optional[float]]
    reward_delta_counterfactual_minus_factual: Mapping[str, float]
    branch_invariants: Mapping[str, bool]
    single_rollout_is_probability: bool
    explanation: str
    outcome_schema_version: str = OUTCOME_SCHEMA_VERSION

    def as_dict(self) -> Dict[str, Any]:
        return to_dict(self)

    def to_json(self, indent: Optional[int] = 2) -> str:
        return json.dumps(
            self.as_dict(),
            indent=indent,
            sort_keys=True,
            allow_nan=False,
        )

    def save_json(self, path: Path, indent: Optional[int] = 2) -> None:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_suffix(output.suffix + ".tmp")
        temporary.write_text(self.to_json(indent=indent) + "\n", encoding="utf-8")
        temporary.replace(output)

    def save_text(self, path: Path) -> None:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_suffix(output.suffix + ".tmp")
        temporary.write_text(self.explanation.rstrip() + "\n", encoding="utf-8")
        temporary.replace(output)


def compute_reward_profile(
    trajectory: TrajectoryRecord,
    gamma: float = 0.99,
    horizons: Sequence[int] = DEFAULT_FIXED_HORIZONS,
) -> Tuple[RewardHorizonPoint, ...]:
    if not 0.0 <= gamma <= 1.0:
        raise ValueError("gamma must be in [0, 1]")
    requested = sorted({int(value) for value in horizons if int(value) > 0})
    if not requested:
        raise ValueError("at least one positive horizon is required")
    result = []
    for requested_horizon in requested:
        horizon = min(requested_horizon, len(trajectory.steps))
        steps = trajectory.steps[:horizon]
        term_names = sorted({name for step in steps for name in step.reward_terms})
        discounted_terms = {name: 0.0 for name in term_names}
        undiscounted_terms = {name: 0.0 for name in term_names}
        discounted_total = 0.0
        undiscounted_total = 0.0
        for index, step in enumerate(steps):
            discount = gamma ** index
            discounted_total += discount * step.reward
            undiscounted_total += step.reward
            for name in term_names:
                value = float(step.reward_terms.get(name, 0.0))
                discounted_terms[name] += discount * value
                undiscounted_terms[name] += value
        result.append(
            RewardHorizonPoint(
                horizon_steps=horizon,
                discounted_total=float(discounted_total),
                undiscounted_total=float(undiscounted_total),
                discounted_terms=discounted_terms,
                undiscounted_terms=undiscounted_terms,
            )
        )
    # Different requested horizons can collapse to the same short terminal
    # horizon. Keep one point per actual horizon so reports are unambiguous.
    unique = {point.horizon_steps: point for point in result}
    return tuple(unique[key] for key in sorted(unique))


def compute_physical_outcome(
    trajectory: TrajectoryRecord,
    decision_dt_seconds: float,
    brake_command_threshold: float = 0.04,
) -> PhysicalOutcomeProfile:
    if decision_dt_seconds <= 0.0:
        raise ValueError("decision_dt_seconds must be positive")
    steps = trajectory.steps
    if not steps:
        raise ValueError("physical outcome requires at least one step")
    lateral = np.asarray([abs(step.decision.state.d) for step in steps], dtype=float)
    heading = np.asarray([abs(step.decision.state.phi) for step in steps], dtype=float)
    progress = sum(
        max(0.0, step.decision.state.v * cos(step.decision.state.phi))
        * decision_dt_seconds
        for step in steps
    )
    clearances = [
        hypot(
            float(step.decision.state.duck_longitudinal),
            float(step.decision.state.duck_lateral),
        )
        for step in steps
        if step.decision.state.duck_present
        and step.decision.state.duck_longitudinal is not None
        and step.decision.state.duck_lateral is not None
    ]
    commands = [step.decision.action for step in steps]
    brake_steps = sum(
        int(action.v_cmd <= brake_command_threshold) for action in commands
    )
    steering_reversals = sum(
        int(previous.omega_cmd * current.omega_cmd < 0.0)
        for previous, current in zip(commands, commands[1:])
        if abs(previous.omega_cmd) > 1e-6 and abs(current.omega_cmd) > 1e-6
    )
    steering_jerk = sum(
        abs(current.omega_cmd - previous.omega_cmd) / decision_dt_seconds
        for previous, current in zip(commands, commands[1:])
    )
    events = [step.events for step in steps]
    primitive_sequence = tuple(segment.primitive.value for segment in trajectory.segments)
    transition_steps = tuple(segment.start_step for segment in trajectory.segments)
    return PhysicalOutcomeProfile(
        steps=len(steps),
        elapsed_seconds=len(steps) * decision_dt_seconds,
        forward_progress_m=float(progress),
        mean_abs_lateral_error_m=float(lateral.mean()),
        max_abs_lateral_error_m=float(lateral.max()),
        mean_abs_heading_error_rad=float(heading.mean()),
        max_abs_heading_error_rad=float(heading.max()),
        minimum_duck_clearance_m=(
            None if not clearances else float(min(clearances))
        ),
        stop_violations=sum(int(value.get("stop_violation", False)) for value in events),
        full_stops=sum(int(value.get("full_stop", False)) for value in events),
        passed_stops=sum(int(value.get("passed_stop", False)) for value in events),
        brake_steps=brake_steps,
        brake_ratio=brake_steps / len(steps),
        steering_reversals=steering_reversals,
        cumulative_steering_jerk=float(steering_jerk),
        lane_departure=any(value.get("offroad", False) for value in events),
        duck_collision=any(value.get("collision_duck", False) for value in events),
        other_collision=any(value.get("other_collision", False) for value in events),
        termination_reason=trajectory.termination_reason,
        primitive_sequence=primitive_sequence,
        primitive_transition_steps=transition_steps,
    )


def physical_delta(
    factual: PhysicalOutcomeProfile,
    counterfactual: PhysicalOutcomeProfile,
) -> Mapping[str, Optional[float]]:
    names = (
        "forward_progress_m",
        "mean_abs_lateral_error_m",
        "max_abs_lateral_error_m",
        "mean_abs_heading_error_rad",
        "max_abs_heading_error_rad",
        "minimum_duck_clearance_m",
        "stop_violations",
        "full_stops",
        "brake_ratio",
        "steering_reversals",
        "cumulative_steering_jerk",
    )
    result: Dict[str, Optional[float]] = {}
    for name in names:
        left = getattr(factual, name)
        right = getattr(counterfactual, name)
        result[name] = None if left is None or right is None else float(right - left)
    return result


def final_reward_delta(
    factual: Sequence[RewardHorizonPoint],
    counterfactual: Sequence[RewardHorizonPoint],
) -> Mapping[str, float]:
    factual_by_horizon = {point.horizon_steps: point for point in factual}
    counter_by_horizon = {point.horizon_steps: point for point in counterfactual}
    common = sorted(set(factual_by_horizon).intersection(counter_by_horizon))
    return {
        str(horizon): float(
            counter_by_horizon[horizon].discounted_total
            - factual_by_horizon[horizon].discounted_total
        )
        for horizon in common
    }


def action_description(action: CanonicalAction) -> str:
    if action.action_name:
        return action.action_name
    return "(v_cmd={:+.3f}, omega_cmd={:+.3f})".format(
        action.v_cmd, action.omega_cmd
    )


def build_explanation_text(
    selected_action: CanonicalAction,
    foil_action: CanonicalAction,
    factual: BranchOutcome,
    counterfactual: BranchOutcome,
) -> str:
    factual_physical = factual.physical
    counter_physical = counterfactual.physical
    lines = [
        "Selected action: %s" % action_description(selected_action),
        "Selected primitive: %s" % factual.first_primitive,
        "Contrast action: %s" % action_description(foil_action),
        "Contrast primitive: %s" % counterfactual.first_primitive,
        "",
        "Simulator-based interventional outcome (not a probability):",
        "- selected discounted return: {:+.3f}".format(
            factual.reward_profile[-1].discounted_total
        ),
        "- contrast discounted return: {:+.3f}".format(
            counterfactual.reward_profile[-1].discounted_total
        ),
        "- selected max |d|: {:.3f} m".format(
            factual_physical.max_abs_lateral_error_m
        ),
        "- contrast max |d|: {:.3f} m".format(
            counter_physical.max_abs_lateral_error_m
        ),
    ]
    if (
        factual_physical.minimum_duck_clearance_m is not None
        and counter_physical.minimum_duck_clearance_m is not None
    ):
        lines.extend(
            [
                "- selected minimum Duckie clearance: {:.3f} m".format(
                    factual_physical.minimum_duck_clearance_m
                ),
                "- contrast minimum Duckie clearance: {:.3f} m".format(
                    counter_physical.minimum_duck_clearance_m
                ),
            ]
        )
    lines.extend(
        [
            "- selected termination: %s" % factual_physical.termination_reason,
            "- contrast termination: %s" % counter_physical.termination_reason,
            "",
            "Selected primitive sequence: "
            + " -> ".join(factual_physical.primitive_sequence),
            "Contrast primitive sequence: "
            + " -> ".join(counter_physical.primitive_sequence),
        ]
    )
    return "\n".join(lines)
