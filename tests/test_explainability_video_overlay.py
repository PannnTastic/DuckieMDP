import numpy as np

from src.actions import ActionConfig, build_action_table
from src.render_multiview_video import explanation_view_panel
from src.explainability.schema import CanonicalAction, CanonicalState, SolverKind
from src.explainability.video_overlay import (
    q_video_explanation,
    VideoExplanation,
    sac_video_explanation,
)


def _state(stop=False, distance=None, duck=False):
    return CanonicalState(
        d=0.0,
        phi=0.0,
        v=0.1,
        curvature=0.0,
        curvature_class="straight",
        stop_present=stop,
        stop_distance=distance,
        stop_satisfied=False,
        stop_hold_progress=0.0,
        duck_present=duck,
        duck_threat=None,
        duck_longitudinal=0.2 if duck else None,
        duck_lateral=0.0 if duck else None,
        duck_v_longitudinal_relative=0.0 if duck else None,
        duck_v_lateral_relative=0.0 if duck else None,
        duck_active=True if duck else None,
        duck_crossing_available=True if duck else None,
        source_representation="test",
    )


def test_q_overlay_uses_second_best_allowed_action_and_exact_margin():
    table = build_action_table(ActionConfig())
    action = CanonicalAction(
        solver=SolverKind.Q_LEARNING,
        action_id=6,
        action_name=table[6].name,
        v_cmd=table[6].v,
        omega_cmd=table[6].omega,
    )
    q_values = np.asarray([0.0, 2.0, 0.0, 0.0, 1.0, 0.0, 5.0])
    explanation = q_video_explanation(
        _state(stop=True, distance=0.2), action, q_values, range(7), table
    )
    assert explanation.primitive == "StopHold"
    assert explanation.foil_label == "1/fast_straight"
    assert explanation.separation_label == "Q-margin=+3.000"


def test_sac_overlay_prefers_semantically_contrasting_probe():
    action = CanonicalAction(
        solver=SolverKind.SAC, v_cmd=0.0, omega_cmd=0.0
    )
    probes = np.asarray([[0.0, 0.0], [0.17, 0.0], [0.41, 0.0]])
    explanation = sac_video_explanation(
        _state(duck=True),
        action,
        ["brake", "slow_straight", "fast_straight"],
        probes,
        selected_q=4.0,
        probe_q=[3.9, 3.8, 3.7],
    )
    assert explanation.primitive == "YieldHold"
    assert explanation.foil_label == "1/slow_straight"
    assert explanation.separation_label == "critic-probe delta=+0.200"
    assert "not a Q-margin" in explanation.caveat


def test_explanation_panel_replaces_image_view_with_readable_contract():
    explanation = VideoExplanation(
        primitive="StopHold",
        trigger="stop line is near and the full-stop obligation is unsatisfied",
        rule_id="P-STOP-HOLD",
        undesirable=False,
        foil_label="4/slow_straight",
        separation_label="Q-margin=+2.500",
        caveat="exact Q-table lookup at current discrete state",
    )
    panel = explanation_view_panel(
        explanation,
        "6/brake (v=+0.00, omega=+0.00)",
        "SOLVER: SARSA / GREEDY EVALUATION",
    )

    assert panel.shape == (480, 640, 3)
    assert panel.dtype == np.uint8
    assert int(np.max(panel)) > 200
    assert len(np.unique(panel.reshape(-1, 3), axis=0)) > 10
