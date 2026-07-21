import json
from pathlib import Path

import numpy as np
import pytest

from src.discretizer import Q_SHAPE
from src.explainability.primitives import DrivingPrimitive, PrimitiveLabel
from src.explainability.q_policy_adapter import QPolicyAdapter
from src.explainability.schema import CanonicalState
from src.explainability.trajectory import (
    TRAJECTORY_SCHEMA_VERSION,
    TrajectoryRecorder,
    build_provenance,
    segment_primitives,
)


def _state():
    return CanonicalState(
        d=0.0,
        phi=0.0,
        v=0.2,
        curvature=None,
        curvature_class="straight",
        stop_present=False,
        stop_distance=None,
        stop_satisfied=False,
        stop_hold_progress=0.0,
        duck_present=False,
        duck_threat="none",
        duck_longitudinal=None,
        duck_lateral=None,
        duck_v_longitudinal_relative=None,
        duck_v_lateral_relative=None,
        duck_active=None,
        duck_crossing_available=None,
        source_representation="unit_test",
    )


def _primitive(value, rule="unit.rule"):
    return PrimitiveLabel(
        primitive=value,
        trigger="unit-test trigger",
        rule_id=rule,
        undesirable=value in {
            DrivingPrimitive.UNNECESSARY_BRAKE,
            DrivingPrimitive.LANE_DEPARTURE,
        },
    )


def _decision():
    return QPolicyAdapter(np.zeros(Q_SHAPE, dtype=np.float32)).decide(_state())


def test_recorder_preserves_step_data_and_segments_consecutive_primitives():
    recorder = TrajectoryRecorder(
        episode_id="q-seed-101",
        provenance={"checkpoint_sha256": "abc"},
        decision_dt_seconds=0.2,
    )
    decision = _decision()
    sequence = [
        DrivingPrimitive.CRUISE_STRAIGHT,
        DrivingPrimitive.CRUISE_STRAIGHT,
        DrivingPrimitive.STOP_HOLD,
        DrivingPrimitive.STOP_HOLD,
        DrivingPrimitive.RESUME_AFTER_STOP,
    ]
    rewards = [0.2, 0.3, -0.1, 1.0, 0.4]
    for index, (primitive, reward) in enumerate(zip(sequence, rewards)):
        recorder.append(
            decision=decision,
            primitive=_primitive(primitive),
            reward=reward,
            info={
                "reward_terms": {"progress": reward, "total": reward},
                "events": {"full_stop": index == 3},
                "termination_reason": "timeout" if index == 4 else "in_progress",
                "terminated": False,
                "truncated": index == 4,
            },
            physics_step=index * 6,
            position_xz=(0.1 * index, 0.2 * index),
            heading_radians=0.01 * index,
        )

    record = recorder.finalize()
    assert record.trajectory_schema_version == TRAJECTORY_SCHEMA_VERSION
    assert len(record.steps) == 5
    assert len(record.segments) == 3
    assert record.total_reward == pytest.approx(sum(rewards))
    assert record.termination_reason == "timeout"
    assert record.steps[2].sim_time_seconds == pytest.approx(0.4)
    assert record.steps[3].events["full_stop"] is True

    cruise, hold, resume = record.segments
    assert (cruise.start_step, cruise.end_step, cruise.duration_steps) == (0, 1, 2)
    assert (hold.start_step, hold.end_step, hold.duration_steps) == (2, 3, 2)
    assert hold.cumulative_reward == pytest.approx(0.9)
    assert hold.event_counts["full_stop"] == 1
    assert resume.primitive == DrivingPrimitive.RESUME_AFTER_STOP


def test_record_serialization_and_atomic_outputs_are_valid_json(tmp_path):
    recorder = TrajectoryRecorder("episode", {"config_sha256": "def"})
    recorder.append(
        _decision(),
        _primitive(DrivingPrimitive.CRUISE_STRAIGHT),
        reward=0.25,
        info={"termination_reason": "in_progress"},
    )
    record = recorder.finalize()
    payload = json.loads(record.to_json())
    assert payload["solver"] == "q_learning"
    assert payload["steps"][0]["primitive"]["primitive"] == "CruiseStraight"

    json_path = tmp_path / "trajectory.json"
    jsonl_path = tmp_path / "steps.jsonl"
    record.save_json(json_path)
    record.save_steps_jsonl(jsonl_path)
    assert json.loads(json_path.read_text())["episode_id"] == "episode"
    lines = jsonl_path.read_text().splitlines()
    assert len(lines) == 1
    assert json.loads(lines[0])["step_index"] == 0
    assert not list(tmp_path.glob("*.tmp"))


def test_provenance_hashes_input_files(tmp_path):
    checkpoint = tmp_path / "checkpoint.bin"
    config = tmp_path / "config.yaml"
    checkpoint.write_bytes(b"policy")
    config.write_text("seed: 1\n", encoding="utf-8")
    provenance = build_provenance(
        checkpoint_path=checkpoint,
        config_path=config,
        primitive_freeze_path=None,
        extra={"seed": 1},
    )
    assert provenance["checkpoint_sha256"] != provenance["config_sha256"]
    assert provenance["seed"] == 1
    assert provenance["primitive_schema_version"] == "1.0.1"


def test_segmenter_rejects_noncontiguous_steps():
    recorder = TrajectoryRecorder("episode", {})
    step = recorder.append(
        _decision(),
        _primitive(DrivingPrimitive.CRUISE_STRAIGHT),
        reward=0.0,
    )
    from dataclasses import replace

    with pytest.raises(ValueError, match="contiguous"):
        segment_primitives([step, replace(step, step_index=2)])
