"""Loss-aware conversion between solver states and canonical semantics."""

from math import pi
from typing import Dict, Sequence, Tuple

import numpy as np

from ..continuous_state import (
    ContinuousState,
    ContinuousStateConfig,
    encode_continuous_state,
)
from ..discretizer import STATE_SHAPE
from ..state import DuckThreat, RawState, TileType
from .schema import CanonicalState


_TILE_TO_NAME: Dict[TileType, str] = {
    TileType.STRAIGHT: "straight",
    TileType.CURVE_LEFT: "curve_left",
    TileType.CURVE_RIGHT: "curve_right",
}
_NAME_TO_TILE = {value: key for key, value in _TILE_TO_NAME.items()}
_DUCK_TO_NAME: Dict[DuckThreat, str] = {
    DuckThreat.NONE: "none",
    DuckThreat.SIDE_FAR: "side_far",
    DuckThreat.SIDE_NEAR: "side_near",
    DuckThreat.CROSSING_FAR: "crossing_far",
    DuckThreat.CROSSING_NEAR: "crossing_near",
}
_NAME_TO_DUCK = {value: key for key, value in _DUCK_TO_NAME.items()}

_D_REPRESENTATIVES = (-0.20, -0.10, 0.0, 0.10, 0.20)
_TRACKING_REPRESENTATIVES = (-0.75, -0.30, 0.0, 0.30, 0.75)
_V_REPRESENTATIVES = (0.02, 0.10, 0.25)
_STOP_REPRESENTATIVES = (None, 1.50, 0.65, 0.15)


def canonical_from_raw_state(raw: RawState) -> CanonicalState:
    duck_threat = _DUCK_TO_NAME[DuckThreat(raw.duck)]
    duck_present = DuckThreat(raw.duck) != DuckThreat.NONE
    crossing = DuckThreat(raw.duck) in {
        DuckThreat.CROSSING_FAR,
        DuckThreat.CROSSING_NEAR,
    }
    return CanonicalState(
        d=float(raw.d),
        phi=float(raw.phi),
        v=float(raw.v),
        curvature=None,
        curvature_class=_TILE_TO_NAME[TileType(raw.tile)],
        stop_present=raw.d_stop is not None,
        stop_distance=None if raw.d_stop is None else float(raw.d_stop),
        stop_satisfied=bool(raw.sigma_stop),
        stop_hold_progress=1.0 if raw.sigma_stop else 0.0,
        duck_present=duck_present,
        duck_threat=duck_threat,
        duck_longitudinal=None,
        duck_lateral=None,
        duck_v_longitudinal_relative=None,
        duck_v_lateral_relative=None,
        duck_active=crossing if duck_present else None,
        duck_crossing_available=None,
        source_representation="q_raw_state",
    )


def canonical_from_continuous_state(state: ContinuousState) -> CanonicalState:
    if state.kappa > 0.05:
        curvature_class = "curve_left"
    elif state.kappa < -0.05:
        curvature_class = "curve_right"
    else:
        curvature_class = "straight"
    return CanonicalState(
        d=float(state.d),
        phi=float(state.phi),
        v=float(state.v),
        curvature=float(state.kappa),
        curvature_class=curvature_class,
        stop_present=bool(state.stop_present),
        stop_distance=(
            None if not state.stop_present or state.d_stop is None
            else float(state.d_stop)
        ),
        stop_satisfied=bool(state.sigma_stop),
        stop_hold_progress=float(state.stop_hold_progress),
        duck_present=bool(state.duck_present),
        duck_threat=None,
        duck_longitudinal=(
            float(state.duck_longitudinal) if state.duck_present else None
        ),
        duck_lateral=float(state.duck_lateral) if state.duck_present else None,
        duck_v_longitudinal_relative=(
            float(state.duck_v_longitudinal_relative)
            if state.duck_present else None
        ),
        duck_v_lateral_relative=(
            float(state.duck_v_lateral_relative) if state.duck_present else None
        ),
        duck_active=bool(state.duck_active) if state.duck_present else None,
        duck_crossing_available=(
            bool(state.duck_crossing_available) if state.duck_present else None
        ),
        source_representation="sac_continuous_state",
    )


def raw_state_from_canonical(state: CanonicalState) -> RawState:
    try:
        tile = _NAME_TO_TILE[state.curvature_class]
    except KeyError as error:
        raise ValueError("unknown curvature_class: %s" % state.curvature_class) from error
    threat_name = state.duck_threat or ("none" if not state.duck_present else None)
    if threat_name is None or threat_name not in _NAME_TO_DUCK:
        raise ValueError("Q-learning query requires a categorical duck_threat")
    return RawState(
        d=float(state.d),
        phi=float(state.phi),
        v=float(state.v),
        tile=tile,
        d_stop=state.stop_distance if state.stop_present else None,
        sigma_stop=bool(state.stop_satisfied),
        duck=_NAME_TO_DUCK[threat_name],
    )


def continuous_state_from_canonical(state: CanonicalState) -> ContinuousState:
    if state.curvature is None:
        raise ValueError("SAC query requires continuous curvature")
    if state.duck_present:
        geometry = (
            state.duck_longitudinal,
            state.duck_lateral,
            state.duck_v_longitudinal_relative,
            state.duck_v_lateral_relative,
        )
        if any(value is None for value in geometry):
            raise ValueError("SAC query with Duckie present requires metric geometry")
        if state.duck_active is None or state.duck_crossing_available is None:
            raise ValueError("SAC query with Duckie present requires controller flags")
    return ContinuousState(
        d=float(state.d),
        phi=float(state.phi),
        v=float(state.v),
        kappa=float(state.curvature),
        stop_present=bool(state.stop_present),
        d_stop=state.stop_distance if state.stop_present else None,
        sigma_stop=bool(state.stop_satisfied),
        duck_present=bool(state.duck_present),
        duck_longitudinal=float(state.duck_longitudinal or 0.0),
        duck_lateral=float(state.duck_lateral or 0.0),
        duck_v_longitudinal_relative=float(
            state.duck_v_longitudinal_relative or 0.0
        ),
        duck_v_lateral_relative=float(state.duck_v_lateral_relative or 0.0),
        duck_active=bool(state.duck_active),
        duck_crossing_available=bool(state.duck_crossing_available),
        stop_hold_progress=float(state.stop_hold_progress or 0.0),
    )


def encode_canonical_for_sac(
    state: CanonicalState,
    config: ContinuousStateConfig = ContinuousStateConfig(),
) -> np.ndarray:
    return encode_continuous_state(continuous_state_from_canonical(state), config)


def canonical_from_discrete_index(index: Sequence[int]) -> CanonicalState:
    """Return an explicitly approximate representative for one Q-table cell."""
    value = tuple(int(item) for item in index)
    if len(value) != len(STATE_SHAPE) or any(
        item < 0 or item >= size for item, size in zip(value, STATE_SHAPE)
    ):
        raise ValueError("invalid discrete state index: %r" % (value,))
    d = _D_REPRESENTATIVES[value[0]]
    tracking = _TRACKING_REPRESENTATIVES[value[1]]
    phi = float(np.clip(tracking - d, -pi / 2.0, pi / 2.0))
    raw = RawState(
        d=d,
        phi=phi,
        v=_V_REPRESENTATIVES[value[2]],
        tile=TileType(value[3]),
        d_stop=_STOP_REPRESENTATIVES[value[4]],
        sigma_stop=bool(value[5]),
        duck=DuckThreat(value[6]),
    )
    canonical = canonical_from_raw_state(raw)
    return CanonicalState(
        **{
            **canonical.__dict__,
            "source_representation": "q_discrete_representative",
            "source_index": value,
        }
    )
