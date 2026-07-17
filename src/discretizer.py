"""Pemetaan state kontinu menjadi finite state space untuk tabular RL.

Indeks Q-table adalah:

    s_bar = (bin(d), bin(e), bin(v), kappa,
             bin(d_stop), sigma_stop, h_duck)

dengan tracking error e=phi+d. Menggunakan e menghilangkan state aliasing:
dua pose dengan phi sama dapat membutuhkan aksi berbeda jika d-nya berbeda.
"""

from typing import Tuple

import numpy as np

from .state import RawState

D_BINS = np.array([-0.15, -0.05, 0.05, 0.15])
TRACKING_ERROR_BINS = np.array([-0.50, -0.10, 0.10, 0.50])
V_BINS = np.array([0.04, 0.16])
STATE_SHAPE = (5, 5, 3, 3, 4, 2, 5)
Q_SHAPE = STATE_SHAPE + (7,)


def discretize(state: RawState) -> Tuple[int, ...]:
    """Mengembalikan indeks finite-state yang aman dipakai sebagai Q[state]."""
    if state.d_stop is None:
        stop = 0
    elif state.d_stop > 1.0:
        stop = 1
    elif state.d_stop >= 0.3:
        stop = 2
    else:
        stop = 3
    # e=phi+d memakai sinyal koreksi yang sama dengan lane teacher. Binning phi
    # saja dahulu menggabungkan pose aman dan berbahaya ke dalam satu state.
    # Perubahan representasi ini membuat policy lane-following stabil.
    tracking_error = state.phi + state.d
    index = (
        int(np.digitize(state.d, D_BINS)),
        int(np.digitize(tracking_error, TRACKING_ERROR_BINS)),
        int(np.digitize(state.v, V_BINS)),
        int(state.tile),
        stop,
        int(state.sigma_stop),
        int(state.duck),
    )
    if any(i < 0 or i >= n for i, n in zip(index, STATE_SHAPE)):
        raise IndexError(index)
    return index
