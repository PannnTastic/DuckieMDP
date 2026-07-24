import copy

import numpy as np
import pytest

from src.explainability.certified_primitives.certificate_adapter import (
    adapt_m1_m13_record,
)
from src.explainability.certified_primitives.certificate_checker import (
    certify_clusters,
)
from src.explainability.certified_primitives.discovery import discover
from src.explainability.certified_primitives.runtime import (
    CertifiedPrimitiveAssigner,
)
from src.explainability.certified_primitives.schema import (
    CertificateStatus,
    CertifiedExplanationInstance,
    ExplanationTrajectory,
    SourceKind,
    TemporalExplanationSegment,
)
from src.explainability.certified_primitives.segmentation import (
    fit_change_point_model,
    segment_trajectory,
)
from src.explainability.certified_primitives.signature import (
    INSTANCE_FEATURE_NAMES,
    assert_explanation_only_contract,
    build_signature_trajectory,
    instance_feature_vector,
)
from src.explainability.certified_primitives.trajectory import build_trajectories


def _payload(valid=True):
    cf = {"attempts": 2, "valid_attempts": 2, "any_flip": True,
          "minimum_flip_distance": 0.2, "lateral_flip": True,
          "lateral_abs_delta": 0.2, "lateral_signed_delta": -0.2}
    validity = {
        "counterfactual_valid": valid,
        "branch_invariants_pass": valid,
        "paired_outcome_valid": valid,
        "deterministic_policy_mode": True,
        "teacher_active": False,
        "supported_or_reachable_state": True,
    }
    return {
        "schema_version": "m1-m13-test",
        "counterfactual_profile": cf,
        "physical_profile": {
            "factual_safe": True, "foil_safe": False,
            "delta_stop_violations": 1.0,
        },
        "verification_profile": {
            "stop_applicable": True, "stop_eligible": True,
            "stop_pass": True, "stop_fail": False,
            "stop_abstain": False, "stop_status": "PASS",
            "stop_pair_stratum": "both_supported",
        },
        "validity": validity,
        "provenance": {
            "manifest_sha256": "abc", "teacher_active": False,
            "deterministic_policy_mode": True,
        },
    }


def _instance(step=0, seed=1, solver="q_learning", valid=True, episode=None):
    return adapt_m1_m13_record(
        _payload(valid), solver=solver, seed=seed,
        episode_id=episode or "%s_%d" % (solver, seed), step_index=step,
    )


def test_adapter_abstains_when_a_binding_gate_fails():
    assert _instance(valid=True).status == CertificateStatus.CERTIFIED
    assert _instance(valid=False).status == CertificateStatus.ABSTAINED


def test_full_trajectory_is_eligible_but_legacy_kind_is_not():
    record = _instance()
    assert record.eligible_for_main_discovery
    payload = record.as_dict()
    payload["source_kind"] = "legacy_sparse"
    legacy = CertifiedExplanationInstance.from_dict(payload)
    assert not legacy.eligible_for_main_discovery


def test_signature_does_not_use_solver_action_context_or_m2_metadata():
    assert_explanation_only_contract(INSTANCE_FEATURE_NAMES)
    left = _instance()
    payload = left.as_dict()
    payload["solver"] = "sac"
    payload["audit_metadata"] = {
        "selection_context": "duck", "m2_primitive": "YieldHold",
        "action_name": "brake", "q_margin": 999.0,
    }
    right = CertifiedExplanationInstance.from_dict(payload)
    assert np.array_equal(instance_feature_vector(left), instance_feature_vector(right))


def test_abstention_and_gaps_break_full_trajectories():
    records = (_instance(0), _instance(1), _instance(2, valid=False),
               _instance(3), _instance(4))
    trajectories = build_trajectories(records)
    assert [len(item.instances) for item in trajectories] == [2, 2]


def test_change_point_segmentation_is_deterministic_and_contiguous():
    records = tuple(_instance(step) for step in range(8))
    trajectory = ExplanationTrajectory(
        trajectory_id="t", solver="q_learning", seed=1,
        episode_id="q_learning_1", source_kind=SourceKind.FULL_TRAJECTORY,
        instances=records, provenance={},
    )
    signature = build_signature_trajectory(trajectory)
    model = fit_change_point_model((signature,), minimum_duration=2)
    first = segment_trajectory(signature, model)
    second = segment_trajectory(signature, model)
    assert [item.as_dict() for item in first] == [item.as_dict() for item in second]
    assert first[0].start_step == 0
    assert first[-1].end_step == 7
    assert sum(item.duration for item in first) == 8


def _synthetic_segments(source=SourceKind.FULL_TRAJECTORY):
    segments = []
    instances = []
    names = (
        "mean__decision__stop_distance_flip",
        "mean__outcome__delta_stop_violations",
        "mean__verification__stop_pass",
        "duration",
        "certificate_coverage",
    )
    rng = np.random.RandomState(7)
    index = 0
    for solver in ("q_learning", "sarsa"):
        for seed in (1, 2, 3, 4, 5):
            for center in (-4.0, 4.0):
                for _ in range(3):
                    record = _instance(index, seed, solver, episode="%s_%d" % (solver, seed))
                    instances.append(record)
                    values = (
                        float(center + rng.normal(0, 0.08)),
                        float(center + rng.normal(0, 0.08)),
                        1.0,
                        3.0,
                        1.0,
                    )
                    segments.append(TemporalExplanationSegment(
                        segment_id="s%d" % index, trajectory_id="t%d" % index,
                        solver=solver, seed=seed, episode_id=record.episode_id,
                        source_kind=source, start_step=index, end_step=index,
                        instance_ids=(record.instance_id,), feature_names=names,
                        feature_values=values, certificate_coverage=1.0,
                    ))
                    index += 1
    return tuple(segments), tuple(instances)


def test_main_discovery_rejects_legacy_sparse_segments():
    segments, _ = _synthetic_segments(SourceKind.LEGACY_SPARSE)
    with pytest.raises(ValueError, match="full_trajectory"):
        discover(segments)


def test_discovery_certificate_and_runtime_abstention():
    segments, instances = _synthetic_segments()
    result = discover(segments)
    primitives = certify_clusters(
        segments, result, instances, "freeze-test",
        minimum_support=4, minimum_seeds=2,
        minimum_bootstrap_ari=0.0, maximum_coherence_ratio=1.0e9,
    )
    assert primitives
    assert all(item.member_certificate_rate == 1.0 for item in primitives)
    assigner = CertifiedPrimitiveAssigner(result.model, primitives)
    supported_index = int(np.flatnonzero(result.labels >= 0)[0])
    assignment = assigner.assign(segments[supported_index])
    assert assignment.status in {
        CertificateStatus.CERTIFIED_PRIMITIVE,
        CertificateStatus.SOLVER_SPECIFIC_PRIMITIVE,
    }
    far_payload = segments[supported_index].as_dict()
    far_payload["feature_values"] = [1.0e6] * len(far_payload["feature_values"])
    far = TemporalExplanationSegment.from_dict(far_payload)
    assert assigner.assign(far).status == CertificateStatus.UNKNOWN


def _descriptor_segment(features):
    names = tuple(features)
    return TemporalExplanationSegment(
        segment_id="seg-descriptor-test",
        trajectory_id="traj-descriptor-test",
        solver="q_learning",
        seed=20101,
        episode_id="q_learning_20101",
        source_kind=SourceKind.FULL_TRAJECTORY,
        start_step=0,
        end_step=2,
        instance_ids=("a", "b", "c"),
        feature_names=names,
        feature_values=tuple(float(features[name]) for name in names),
        certificate_coverage=1.0,
    )


def test_descriptor_naming_is_explanation_derived():
    from src.explainability.certified_primitives.descriptor import build_descriptor

    # A stop whose pedestrian relation is idle -> a stop-line obligation.
    stop_segment = _descriptor_segment({
        "mean__decision__stop_satisfied_flip": 1.0,
        "mean__verification__lane_symmetry_applicable": 0.0,
        "mean__verification__pedestrian_applicable": 0.0,
    })
    assert build_descriptor(0, (stop_segment,), {}).functional_name.startswith(
        "StopCompliance"
    )

    # The lane-symmetry metamorphic relation applies (a straight lane) -> LaneKeeping.
    lane_segment = _descriptor_segment({
        "mean__verification__lane_symmetry_applicable": 1.0,
    })
    assert build_descriptor(1, (lane_segment,), {}).functional_name.startswith(
        "LaneKeeping"
    )

    # A continuous policy measured a near-miss clearance to a Duckie -> PedestrianYield.
    duck_metric_segment = _descriptor_segment({
        "mean__verification__lane_symmetry_applicable": 0.0,
        "mean__decision__stop_satisfied_flip": 0.0,
        "mean__outcome__factual_minimum_duck_clearance_available": 1.0,
        "mean__outcome__factual_minimum_duck_clearance_m": 0.2,
    })
    assert build_descriptor(2, (duck_metric_segment,), {}).functional_name.startswith(
        "PedestrianYield"
    )

    # A tabular policy has no metric clearance and yields by stopping; its pedestrian
    # metamorphic relation is co-active, so the stop is a Duckie yield, not a stop-line.
    duck_stop_segment = _descriptor_segment({
        "mean__verification__lane_symmetry_applicable": 0.0,
        "mean__decision__stop_satisfied_flip": 1.0,
        "mean__verification__pedestrian_applicable": 1.0,
        "mean__outcome__factual_minimum_duck_clearance_available": 0.0,
    })
    assert build_descriptor(3, (duck_stop_segment,), {}).functional_name.startswith(
        "PedestrianYield"
    )

    # None of the above signatures fire -> CurveNegotiation (sustained steering).
    curve_segment = _descriptor_segment({
        "mean__verification__lane_symmetry_applicable": 0.0,
        "mean__outcome__factual_minimum_duck_clearance_available": 0.0,
    })
    assert build_descriptor(4, (curve_segment,), {}).functional_name.startswith(
        "CurveNegotiation"
    )
