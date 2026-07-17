import numpy as np
from src.discretizer import Q_SHAPE, STATE_SHAPE, discretize
from src.state import DuckThreat, RawState, TileType


def make_state(d=0.0, phi=0.0, v=0.0, stop=None):
    return RawState(d, phi, v, TileType.STRAIGHT, stop, False, DuckThreat.NONE)


def test_q_shape():
    assert Q_SHAPE == (5, 5, 3, 3, 4, 2, 5, 7)
    assert int(np.prod(STATE_SHAPE)) == 9000


def test_stop_bins():
    assert discretize(make_state(stop=None))[4] == 0
    assert discretize(make_state(stop=1.1))[4] == 1
    assert discretize(make_state(stop=0.3))[4] == 2
    assert discretize(make_state(stop=0.29))[4] == 3


def test_random_valid_states_never_overflow():
    rng = np.random.RandomState(1)
    for _ in range(5000):
        state = RawState(rng.uniform(-0.25, 0.25), rng.uniform(-1.57, 1.57),
                         rng.uniform(0, 1), TileType(rng.randint(3)),
                         None if rng.rand() < 0.2 else rng.uniform(0, 4),
                         bool(rng.randint(2)), DuckThreat(rng.randint(5)))
        index = discretize(state)
        assert all(0 <= i < n for i, n in zip(index, STATE_SHAPE))
