import numpy as np

from src.discretizer import Q_SHAPE
from src.train import _broadcast_lane_prior


def test_broadcast_lane_prior_copies_only_lane_context_to_every_context():
    q = np.zeros(Q_SHAPE, dtype=np.float32)
    lane = np.arange(np.prod(Q_SHAPE[:4]) * Q_SHAPE[-1], dtype=np.float32).reshape(
        Q_SHAPE[:4] + (Q_SHAPE[-1],)
    )
    q[:, :, :, :, 0, 0, 0, :] = lane
    q[:, :, :, :, 3, 1, 4, :] = -123.0

    _broadcast_lane_prior(q)

    for stop in range(Q_SHAPE[4]):
        for sigma in range(Q_SHAPE[5]):
            for duck in range(Q_SHAPE[6]):
                np.testing.assert_array_equal(q[:, :, :, :, stop, sigma, duck, :], lane)
