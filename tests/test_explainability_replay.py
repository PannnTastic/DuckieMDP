from dataclasses import replace

import pytest

from src.explainability.scenario_manifest import capture_rng_state
from src.explainability.simulator_branching import (
    ReplayStep,
    ReplayTrace,
    assert_replays_identical,
)


def _trace(reward=1.0):
    step = ReplayStep(
        step_index=0,
        action=(0.2, 0.0),
        raw_state_vector=(0.0, 0.0, 0.2, 0.0, -1.0, 0.0, 0.0),
        discrete_state=(2, 2, 2, 0, 0, 0, 0),
        observation=(0.0,) * 15,
        reward=reward,
        reward_terms={"progress": reward, "total": reward},
        events={"offroad": False},
        controller_phase=((1, False, True, 0.2, (1.0, 0.0, 1.0)),),
        termination_reason="in_progress",
        terminated=False,
        truncated=False,
    )
    return ReplayTrace(reset_seed=7, steps=(step,))


def test_replay_comparison_accepts_numerical_identity_within_contract():
    left = _trace()
    right_step = replace(
        left.steps[0],
        raw_state_vector=(1e-8, 0.0, 0.2, 0.0, -1.0, 0.0, 0.0),
    )
    assert_replays_identical(left, replace(left, steps=(right_step,)))


def test_replay_comparison_reports_reward_and_controller_divergence():
    with pytest.raises(AssertionError, match="reward mismatch"):
        assert_replays_identical(_trace(), _trace(reward=1.1))
    changed = replace(
        _trace().steps[0],
        controller_phase=((2, False, True, 0.2, (1.0, 0.0, 1.0)),),
    )
    with pytest.raises(AssertionError, match="controller phase"):
        assert_replays_identical(_trace(), ReplayTrace(7, (changed,)))


def test_numpy_rng_state_capture_is_json_compatible():
    import json
    import numpy as np

    rng = np.random.RandomState(11)
    rng.random_sample()
    payload = capture_rng_state(rng)
    encoded = json.dumps(payload, sort_keys=True, allow_nan=False)
    assert "RandomState" in encoded
    assert len(payload["keys"]) == 624


def test_dict_based_compat_rng_state_capture_is_json_compatible():
    import json

    class CompatRng:
        def get_state(self):
            return {
                "bit_generator": "PCG64",
                "state": {"state": 123, "inc": 5},
                "has_uint32": 0,
                "uinteger": 0,
            }

