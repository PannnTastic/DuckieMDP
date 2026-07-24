from types import SimpleNamespace

import numpy as np

from src.continuous_state import (
    ContinuousState,
    ContinuousStateConfig,
    continuous_observation_space,
    duck_relative_state,
    encode_continuous_state,
)


def _state(**overrides):
    values = dict(
        d=0.0,
        phi=0.0,
        v=0.1,
        kappa=0.0,
        stop_present=False,
        d_stop=None,
        sigma_stop=False,
        duck_present=False,
        duck_longitudinal=0.0,
        duck_lateral=0.0,
        duck_v_longitudinal_relative=0.0,
        duck_v_lateral_relative=0.0,
        duck_active=False,
        duck_crossing_available=False,
        stop_hold_progress=0.0,
    )
    values.update(overrides)
    return ContinuousState(**values)


def test_absent_masks_use_safe_sentinels_and_fit_space():
    cfg = ContinuousStateConfig()
    encoded = encode_continuous_state(_state(), cfg)
    space = continuous_observation_space()
    assert encoded.shape == (15,)
    assert encoded.dtype == np.float32
    assert encoded[4] == 0.0
    assert encoded[5] == 1.0
    assert encoded[7] == 0.0
    assert encoded[8] == 1.0
    assert encoded[9] == 0.0
    assert encoded[14] == 0.0
    assert space.contains(encoded)


def test_stop_hold_progress_is_the_append_only_final_feature():
    encoded = encode_continuous_state(
        _state(stop_present=True, d_stop=0.2, stop_hold_progress=2.0 / 3.0),
        ContinuousStateConfig(),
    )
    assert encoded.shape == (15,)
    assert encoded[-1] == np.float32(2.0 / 3.0)


class FakeEnv:
    def __init__(self, duck):
        self.cur_pos = np.zeros(3)
        self.cur_angle = 0.0
        self.objects = [duck]

    def closest_curve_point(self, _pos, _angle):
        return np.zeros(3), np.array([1.0, 0.0, 0.0])


def test_duck_geometry_and_controller_phase_are_exposed():
    duck = SimpleNamespace(
        kind="duckie",
        visible=True,
        pos=np.array([1.0, 0.0, 0.5]),
        center=np.array([1.0, 0.0, 0.5]),
        heading=np.array([0.0, 0.0, 1.0]),
        vel=0.02,
        pedestrian_active=True,
    )
    controller = SimpleNamespace(
        ducks=[duck],
        crossings_started=[1],
        cfg=SimpleNamespace(max_crossings_per_episode=1),
    )
    result = duck_relative_state(FakeEnv(duck), ego_speed=0.10, controller=controller)
    assert result.present
    assert result.longitudinal == 1.0
    assert result.lateral == 0.5
    assert result.v_longitudinal_relative == -0.10
    assert result.v_lateral_relative == 0.02
    assert result.active
    assert not result.crossing_available


def test_duck_detection_gate_mirrors_tabular_visibility():
    from src.continuous_state import (
        ContinuousStateConfig,
        DuckRelativeState,
        gate_duck_visibility,
    )

    gated = ContinuousStateConfig(
        duck_detection_range=1.20,
        duck_detection_corridor_width=0.60,
        duck_detection_forward_only=True,
    )
    near_ahead = DuckRelativeState(
        present=True, longitudinal=0.50, lateral=0.10, active=True
    )
    far_ahead = DuckRelativeState(present=True, longitudinal=1.60, lateral=0.10)
    behind = DuckRelativeState(present=True, longitudinal=-0.30, lateral=0.10)
    wide = DuckRelativeState(present=True, longitudinal=0.50, lateral=0.90)

    assert gate_duck_visibility(near_ahead, gated) is near_ahead
    assert gate_duck_visibility(far_ahead, gated).present is False
    assert gate_duck_visibility(behind, gated).present is False
    assert gate_duck_visibility(wide, gated).present is False

    # Default config keeps legacy always-visible behaviour for SAC checkpoints.
    legacy = ContinuousStateConfig()
    assert gate_duck_visibility(far_ahead, legacy) is far_ahead
    assert gate_duck_visibility(behind, legacy) is behind
