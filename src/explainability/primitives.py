"""Shared, context-aware driving primitive lexicon for Q-learning and SAC."""

from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any, Mapping, Optional, Union

from .schema import CanonicalAction, CanonicalState, PolicyDecision


PRIMITIVE_SCHEMA_VERSION = "1.0.1"


class DrivingPrimitive(str, Enum):
    CRUISE_STRAIGHT = "CruiseStraight"
    CRUISE_CURVE_LEFT = "CruiseCurveLeft"
    CRUISE_CURVE_RIGHT = "CruiseCurveRight"
    LANE_CORRECT_LEFT = "LaneCorrectLeft"
    LANE_CORRECT_RIGHT = "LaneCorrectRight"
    DECELERATE_LANE = "DecelerateLane"
    EMERGENCY_LANE_RECOVERY = "EmergencyLaneRecovery"

    APPROACH_STOP = "ApproachStop"
    DECELERATE_STOP = "DecelerateStop"
    STOP_HOLD = "StopHold"
    STOP_SATISFIED = "StopSatisfied"
    RESUME_AFTER_STOP = "ResumeAfterStop"

    APPROACH_CROSSING = "ApproachCrossing"
    YIELD_DECELERATE = "YieldDecelerate"
    YIELD_HOLD = "YieldHold"
    WAIT_FOR_CLEARANCE = "WaitForClearance"
    RESUME_AFTER_YIELD = "ResumeAfterYield"

    UNNECESSARY_BRAKE = "UnnecessaryBrake"
    UNSAFE_PROCEED = "UnsafeProceed"
    STOP_VIOLATION = "StopViolation"
    LANE_DEPARTURE = "LaneDeparture"
    OSCILLATORY_STEERING = "OscillatorySteering"
    PREMATURE_RESUME = "PrematureResume"
    UNKNOWN = "Unknown"


UNDESIRABLE_PRIMITIVES = frozenset(
    {
        DrivingPrimitive.UNNECESSARY_BRAKE,
        DrivingPrimitive.UNSAFE_PROCEED,
        DrivingPrimitive.STOP_VIOLATION,
        DrivingPrimitive.LANE_DEPARTURE,
        DrivingPrimitive.OSCILLATORY_STEERING,
        DrivingPrimitive.PREMATURE_RESUME,
    }
)


@dataclass(frozen=True)
class PrimitiveThresholds:
    """Frozen physical/semantic thresholds used by both solvers."""

    hold_command_speed: float = 0.03
    slow_command_speed: float = 0.18
    unsafe_proceed_speed: float = 0.08
    deceleration_delta: float = 0.02
    steering_deadband: float = 0.15
    oscillation_omega: float = 0.50
    lane_tracking_error: float = 0.10
    severe_lateral_error: float = 0.18
    severe_heading_error: float = 0.70
    stop_hold_distance: float = 0.45
    stop_decelerate_distance: float = 1.00
    duck_risk_longitudinal_min: float = -0.15
    duck_risk_longitudinal_max: float = 0.80
    duck_corridor_half_width: float = 0.40
    duck_approach_longitudinal_min: float = -0.30
    duck_approach_radius: float = 1.50

    def __post_init__(self) -> None:
        for name, value in asdict(self).items():
            if float(value) < 0.0 and not name.endswith("longitudinal_min"):
                raise ValueError("%s must be non-negative" % name)
        if self.stop_hold_distance > self.stop_decelerate_distance:
            raise ValueError("stop hold distance cannot exceed deceleration distance")
        if self.duck_risk_longitudinal_min >= self.duck_risk_longitudinal_max:
            raise ValueError("invalid Duckie risk interval")


@dataclass(frozen=True)
class PrimitiveLabel:
    primitive: DrivingPrimitive
    trigger: str
    rule_id: str
    undesirable: bool
    schema_version: str = PRIMITIVE_SCHEMA_VERSION

    def as_dict(self):
        return {
            "primitive": self.primitive.value,
            "trigger": self.trigger,
            "rule_id": self.rule_id,
            "undesirable": self.undesirable,
            "schema_version": self.schema_version,
        }


_STOP_PRIMITIVES = frozenset(
    {
        DrivingPrimitive.APPROACH_STOP,
        DrivingPrimitive.DECELERATE_STOP,
        DrivingPrimitive.STOP_HOLD,
        DrivingPrimitive.STOP_SATISFIED,
    }
)
_YIELD_PRIMITIVES = frozenset(
    {
        DrivingPrimitive.YIELD_DECELERATE,
        DrivingPrimitive.YIELD_HOLD,
        DrivingPrimitive.WAIT_FOR_CLEARANCE,
    }
)


def _primitive(value: Optional[Union[str, DrivingPrimitive]]) -> Optional[DrivingPrimitive]:
    if value is None or isinstance(value, DrivingPrimitive):
        return value
    return DrivingPrimitive(value)


def _event(events: Optional[Any], name: str) -> bool:
    if events is None:
        return False
    if isinstance(events, Mapping):
        return bool(events.get(name, False))
    return bool(getattr(events, name, False))


def _result(
    primitive: DrivingPrimitive,
    trigger: str,
    rule_id: str,
) -> PrimitiveLabel:
    return PrimitiveLabel(
        primitive=primitive,
        trigger=trigger,
        rule_id=rule_id,
        undesirable=primitive in UNDESIRABLE_PRIMITIVES,
    )


def _duck_risk(state: CanonicalState, thresholds: PrimitiveThresholds) -> bool:
    categorical = state.duck_threat in {"crossing_far", "crossing_near"}
    if categorical:
        return True
    if not state.duck_present or state.duck_active is not True:
        return False
    if state.duck_longitudinal is None or state.duck_lateral is None:
        return True
    return (
        thresholds.duck_risk_longitudinal_min
        <= state.duck_longitudinal
        <= thresholds.duck_risk_longitudinal_max
        and abs(state.duck_lateral) <= thresholds.duck_corridor_half_width
    )

def _duck_relevant(state: CanonicalState, thresholds: PrimitiveThresholds) -> bool:
    """Ignore an always-visible Duckie that is far from the ego interaction."""
    if state.duck_threat not in {None, "none"}:
        return True
    if not state.duck_present:
        return False
    if state.duck_longitudinal is None or state.duck_lateral is None:
        return True
    if state.duck_longitudinal < thresholds.duck_approach_longitudinal_min:
        return False
    squared_distance = (
        state.duck_longitudinal * state.duck_longitudinal
        + state.duck_lateral * state.duck_lateral
    )
    return squared_distance <= thresholds.duck_approach_radius ** 2


def label_primitive(
    state: CanonicalState,
    action: CanonicalAction,
    events: Optional[Any] = None,
    termination_reason: str = "in_progress",
    previous_action: Optional[CanonicalAction] = None,
    previous_primitive: Optional[Union[str, DrivingPrimitive]] = None,
    thresholds: PrimitiveThresholds = PrimitiveThresholds(),
) -> PrimitiveLabel:
    """Apply one frozen, solver-neutral precedence-ordered rule set."""

    prior = _primitive(previous_primitive)
    command_hold = action.v_cmd <= thresholds.hold_command_speed
    command_slow = action.v_cmd <= thresholds.slow_command_speed
    decelerating = action.v_cmd <= state.v - thresholds.deceleration_delta
    lane_error = state.phi + state.d
    severe_lane = (
        abs(state.d) >= thresholds.severe_lateral_error
        or abs(state.phi) >= thresholds.severe_heading_error
    )
    correcting = (
        abs(action.omega_cmd) >= thresholds.steering_deadband
        and lane_error * action.omega_cmd > 0.0
    )
    duck_risk = _duck_risk(state, thresholds)
    duck_relevant = _duck_relevant(state, thresholds)

    # Terminal/event rules have highest precedence because they describe what
    # actually happened rather than merely what the command appeared to mean.
    if termination_reason == "offroad" or _event(events, "offroad"):
        return _result(
            DrivingPrimitive.LANE_DEPARTURE,
            "ego left the drivable lane",
            "event.offroad",
        )
    if _event(events, "stop_violation"):
        return _result(
            DrivingPrimitive.STOP_VIOLATION,
            "ego passed the stop line before satisfying the full stop",
            "event.stop_violation",
        )
    if termination_reason == "duck_collision" or _event(events, "collision_duck"):
        return _result(
            DrivingPrimitive.UNSAFE_PROCEED,
            "ego collided with the pedestrian",
            "event.duck_collision",
        )

    # Temporal transition labels must be evaluated before ordinary stop/yield
    # rules, otherwise a resume is swallowed by a generic approach label.
    if prior in _YIELD_PRIMITIVES and duck_risk and (
        action.v_cmd > thresholds.unsafe_proceed_speed
    ):
        return _result(
            DrivingPrimitive.PREMATURE_RESUME,
            "ego resumed while pedestrian risk was still active",
            "pedestrian.premature_resume",
        )
    if prior in _YIELD_PRIMITIVES and not duck_risk and not command_hold:
        return _result(
            DrivingPrimitive.RESUME_AFTER_YIELD,
            "pedestrian risk cleared and forward motion resumed",
            "pedestrian.resume",
        )
    if prior in _STOP_PRIMITIVES and state.stop_satisfied and not command_hold:
        return _result(
            DrivingPrimitive.RESUME_AFTER_STOP,
            "full-stop obligation was satisfied before forward motion resumed",
            "stop.resume",
        )

    # Pedestrian rules outrank stop-sign rules because an active crossing is the
    # more immediate safety constraint when both objects are visible.
    if duck_risk:
        if command_hold:
            return _result(
                DrivingPrimitive.YIELD_HOLD,
                "pedestrian occupied or approached the ego corridor",
                "pedestrian.hold",
            )
        if command_slow or decelerating:
            return _result(
                DrivingPrimitive.YIELD_DECELERATE,
                "ego reduced speed for an active pedestrian risk",
                "pedestrian.decelerate",
            )
        return _result(
            DrivingPrimitive.UNSAFE_PROCEED,
            "forward command exceeded the safe yielding speed",
            "pedestrian.unsafe_proceed",
        )
    # Active corridor risk outranks a stop above, but an inactive/side Duckie
    # must not hide an unsatisfied stop-sign obligation.
    stop_obligation = state.stop_present and not state.stop_satisfied
    if duck_relevant and not stop_obligation:
        if command_hold and prior in _YIELD_PRIMITIVES:
            return _result(
                DrivingPrimitive.WAIT_FOR_CLEARANCE,
                "ego remained stopped until the pedestrian scene cleared",
                "pedestrian.wait_clearance",
            )
        return _result(
            DrivingPrimitive.APPROACH_CROSSING,
            "a pedestrian was present but not yet in the ego corridor",
            "pedestrian.approach",
        )

    # A full-stop event is the exact transition into the satisfied state.
    if _event(events, "full_stop"):
        return _result(
            DrivingPrimitive.STOP_SATISFIED,
            "the required full stop was completed",
            "stop.full_stop_event",
        )
    if state.stop_present:
        distance = state.stop_distance
        if state.stop_satisfied:
            if command_hold:
                return _result(
                    DrivingPrimitive.STOP_SATISFIED,
                    "stop obligation is satisfied while ego remains stationary",
                    "stop.satisfied",
                )
            return _result(
                DrivingPrimitive.RESUME_AFTER_STOP,
                "stop obligation is satisfied and ego proceeds",
                "stop.resume_without_history",
            )
        if (
            distance is not None
            and distance <= thresholds.stop_hold_distance
            and command_hold
        ):
            return _result(
                DrivingPrimitive.STOP_HOLD,
                "ego held zero speed inside the stop zone",
                "stop.hold",
            )
        if (
            distance is not None
            and distance <= thresholds.stop_decelerate_distance
            and (command_slow or decelerating)
        ):
            return _result(
                DrivingPrimitive.DECELERATE_STOP,
                "ego reduced speed while approaching an unsatisfied stop",
                "stop.decelerate",
            )
        return _result(
            DrivingPrimitive.APPROACH_STOP,
            "an unsatisfied stop line was visible ahead",
            "stop.approach",
        )

    if previous_action is not None and (
        abs(previous_action.omega_cmd) >= thresholds.oscillation_omega
        and abs(action.omega_cmd) >= thresholds.oscillation_omega
        and previous_action.omega_cmd * action.omega_cmd < 0.0
    ):
        return _result(
            DrivingPrimitive.OSCILLATORY_STEERING,
            "large steering command reversed sign on consecutive decisions",
            "lane.oscillatory_steering",
        )
    if severe_lane:
        if command_hold or correcting:
            return _result(
                DrivingPrimitive.EMERGENCY_LANE_RECOVERY,
                "severe lane error triggered a conservative recovery command",
                "lane.emergency_recovery",
            )
        return _result(
            DrivingPrimitive.LANE_DEPARTURE,
            "severe lane error continued without a corrective command",
            "lane.departure_risk",
        )
    if command_hold:
        return _result(
            DrivingPrimitive.UNNECESSARY_BRAKE,
            "ego stopped without a stop, pedestrian, or lane emergency",
            "lane.unnecessary_brake",
        )
    if abs(lane_error) >= thresholds.lane_tracking_error and (
        abs(action.omega_cmd) >= thresholds.steering_deadband
    ):
        primitive = (
            DrivingPrimitive.LANE_CORRECT_LEFT
            if action.omega_cmd > 0.0
            else DrivingPrimitive.LANE_CORRECT_RIGHT
        )
        return _result(
            primitive,
            "steering command corrected lane-relative tracking error",
            "lane.correct",
        )
    if command_slow and decelerating:
        return _result(
            DrivingPrimitive.DECELERATE_LANE,
            "ego reduced speed for lane geometry or tracking stability",
            "lane.decelerate",
        )
    if state.curvature_class == "curve_left":
        return _result(
            DrivingPrimitive.CRUISE_CURVE_LEFT,
            "ego cruised on a left-curving lane segment",
            "lane.cruise_curve_left",
        )
    if state.curvature_class == "curve_right":
        return _result(
            DrivingPrimitive.CRUISE_CURVE_RIGHT,
            "ego cruised on a right-curving lane segment",
            "lane.cruise_curve_right",
        )
    if abs(action.omega_cmd) >= thresholds.steering_deadband:
        primitive = (
            DrivingPrimitive.LANE_CORRECT_LEFT
            if action.omega_cmd > 0.0
            else DrivingPrimitive.LANE_CORRECT_RIGHT
        )
        return _result(
            primitive,
            "steering command adjusted a nominally straight lane",
            "lane.correct_without_large_error",
        )
    return _result(
        DrivingPrimitive.CRUISE_STRAIGHT,
        "ego maintained forward motion on a straight lane",
        "lane.cruise_straight",
    )


class PrimitiveLabeler:
    """Stateful convenience wrapper that supplies one-step temporal context."""

    def __init__(
        self,
        thresholds: PrimitiveThresholds = PrimitiveThresholds(),
    ) -> None:
        self.thresholds = thresholds
        self.previous_action: Optional[CanonicalAction] = None
        self.previous_primitive: Optional[DrivingPrimitive] = None

    def reset(self) -> None:
        self.previous_action = None
        self.previous_primitive = None

    def label(
        self,
        decision: PolicyDecision,
        events: Optional[Any] = None,
        termination_reason: str = "in_progress",
    ) -> PrimitiveLabel:
        result = label_primitive(
            state=decision.state,
            action=decision.action,
            events=events,
            termination_reason=termination_reason,
            previous_action=self.previous_action,
            previous_primitive=self.previous_primitive,
            thresholds=self.thresholds,
        )
        self.previous_action = decision.action
        self.previous_primitive = result.primitive
        return result
