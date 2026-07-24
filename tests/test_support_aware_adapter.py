"""The local C-EDDP certificate must use measured source support."""

from src.explainability.certified_primitives.certificate_adapter import (
    adapt_m1_m13_record,
)
from src.explainability.certified_primitives.schema import CertificateStatus


def _payload(reachable):
    return {
        "schema_version": "support-aware-test",
        "counterfactual_profile": {
            "attempts": 1,
            "valid_attempts": 1,
        },
        "verification_profile": {},
        "validity": {
            "counterfactual_valid": True,
            "branch_invariants_pass": True,
            "paired_outcome_valid": True,
            "deterministic_policy_mode": True,
            "teacher_active": False,
        },
        "support": {
            "solver": "q_learning",
            "stratum": "reachable" if reachable else "unseen",
            "basis": "unit_test",
            "reachable": reachable,
            "supported": False,
            "observation_count": 1 if reachable else 0,
        },
        "provenance": {
            "manifest_sha256": "support-test",
            "deterministic_policy_mode": True,
            "teacher_active": False,
        },
    }


def test_source_support_is_not_hardcoded_to_true():
    abstained = adapt_m1_m13_record(
        _payload(False),
        solver="q_learning",
        seed=1,
        episode_id="unseen",
        step_index=0,
    )
    certified = adapt_m1_m13_record(
        _payload(True),
        solver="q_learning",
        seed=1,
        episode_id="reachable",
        step_index=0,
    )

    assert abstained.status == CertificateStatus.ABSTAINED
    assert not abstained.certificate["supported_or_reachable_state"]
    assert certified.status == CertificateStatus.CERTIFIED
    assert certified.certificate["support_evidence"]["basis"] == "unit_test"
