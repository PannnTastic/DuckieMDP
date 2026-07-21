"""Faithful per-decision explanation fields for multiview renderers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

from .primitives import DrivingPrimitive, label_primitive
from .schema import CanonicalAction, CanonicalState, SolverKind


@dataclass(frozen=True)
class VideoExplanation:
    primitive: str
    trigger: str
    rule_id: str
    undesirable: bool
    foil_label: str
    separation_label: str
    caveat: str


def _short_trigger(text, limit=78):
    value = " ".join(str(text).split())
    return value if len(value) <= limit else value[: limit - 3] + "..."


def q_video_explanation(
    state: CanonicalState,
    action: CanonicalAction,
    q_values: Sequence[float],
    allowed_actions: Sequence[int],
    action_table,
    previous_action: Optional[CanonicalAction] = None,
    previous_primitive: Optional[str] = None,
):
    values = np.asarray(q_values, dtype=np.float64)
    alternatives = [int(item) for item in allowed_actions if int(item) != action.action_id]
    foil_id = max(alternatives, key=lambda item: (values[item], -item))
    foil = action_table[foil_id]
    label = label_primitive(
        state,
        action,
        previous_action=previous_action,
        previous_primitive=previous_primitive,
    )
    margin = float(values[action.action_id] - values[foil_id])
    return VideoExplanation(
        primitive=label.primitive.value,
        trigger=_short_trigger(label.trigger),
        rule_id=label.rule_id,
        undesirable=label.undesirable,
        foil_label=f"{foil_id}/{foil.name}",
        separation_label=f"Q-margin={margin:+.3f}",
        caveat="exact Q-table lookup at current discrete state",
    )


def sac_video_explanation(
    state: CanonicalState,
    action: CanonicalAction,
    probe_names: Sequence[str],
    probe_actions: np.ndarray,
    selected_q: float,
    probe_q: Sequence[float],
    previous_action: Optional[CanonicalAction] = None,
    previous_primitive: Optional[str] = None,
):
    label = label_primitive(
        state,
        action,
        previous_action=previous_action,
        previous_primitive=previous_primitive,
    )
    values = np.asarray(probe_q, dtype=np.float64)
    actions = np.asarray(probe_actions, dtype=np.float64)
    order = list(np.argsort(values)[::-1])
    foil_index = order[0]
    # Prefer a high-valued canonical probe with a different semantic primitive.
    for index in order:
        probe_action = CanonicalAction(
            solver=SolverKind.SAC,
            v_cmd=float(actions[index, 0]),
            omega_cmd=float(actions[index, 1]),
        )
        probe_label = label_primitive(state, probe_action)
        if probe_label.primitive != label.primitive:
            foil_index = int(index)
            break
    difference = float(selected_q - values[foil_index])
    return VideoExplanation(
        primitive=label.primitive.value,
        trigger=_short_trigger(label.trigger),
        rule_id=label.rule_id,
        undesirable=label.undesirable,
        foil_label=f"{foil_index}/{probe_names[foil_index]}",
        separation_label=f"critic-probe delta={difference:+.3f}",
        caveat="critic probe is supporting OOD-sensitive evidence, not a Q-margin",
    )
