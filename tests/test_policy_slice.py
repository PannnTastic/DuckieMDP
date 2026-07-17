import numpy as np

from src.discretizer import Q_SHAPE
from src.policy_slice import greedy_policy_slice


def test_policy_slice_contains_selected_actions():
    q = np.zeros(Q_SHAPE, dtype=np.float32)
    q[..., 6] = 1.0
    text = greedy_policy_slice(q, 2, 0, 0, 0, 0)
    assert text.count("BR") >= 25
    assert "phi4" in text and "phi0" in text

