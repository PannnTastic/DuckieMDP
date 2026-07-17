"""Tabular off-policy Q-learning untuk finite MDP Duckietown.

Update Bellman:

    Q(s,a) <- Q(s,a) + alpha * [target - Q(s,a)]
    target = r                           jika terminal
    target = r + gamma*max_a' Q(s',a')  selain itu
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence, Tuple

import numpy as np

from ..discretizer import Q_SHAPE


@dataclass(frozen=True)
class QLearningConfig:
    gamma: float = 0.99
    alpha_lr: float = 0.10
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_steps: int = 50000
    allowed_actions: Sequence[int] = (0, 1, 2, 3, 4, 5, 6)


class QLearningAgent:
    def __init__(self, cfg: QLearningConfig, seed: int = 0) -> None:
        self.cfg, self.steps = cfg, 0
        self.rng = np.random.RandomState(seed)
        self.q = np.zeros(Q_SHAPE, dtype=np.float32)
        self.allowed_actions = np.asarray(tuple(cfg.allowed_actions), dtype=int)
        if self.allowed_actions.size == 0 or np.any(self.allowed_actions < 0) or np.any(self.allowed_actions >= 7):
            raise ValueError("allowed_actions must contain action ids in [0, 6]")

    @property
    def epsilon(self) -> float:
        ratio = min(1.0, self.steps / max(1, self.cfg.epsilon_decay_steps))
        return self.cfg.epsilon_start + ratio * (self.cfg.epsilon_end - self.cfg.epsilon_start)

    def select_action(self, state: Tuple[int, ...], greedy: bool = False) -> int:
        """Epsilon-greedy saat training; argmax Q saat evaluation."""
        if not greedy and self.rng.random_sample() < self.epsilon:
            action = int(self.rng.choice(self.allowed_actions))
        else:
            values = self.q[state][self.allowed_actions]
            best = self.allowed_actions[np.flatnonzero(values == values.max())]
            action = int(self.rng.choice(best))
        if not greedy:
            self.steps += 1
        return action

    def update(self, state: Tuple[int, ...], action: int, reward: float,
               next_state: Optional[Tuple[int, ...]], done: bool) -> None:
        """Melakukan satu TD update dari sampel transition (s, a, r, s')."""
        if int(action) not in self.allowed_actions:
            raise ValueError(f"Action {action} is masked")
        if done:
            # Terminal sejati: tidak ada nilai masa depan yang di-bootstrap.
            target = reward
        else:
            # Off-policy target memakai action terbaik pada next_state.
            target = reward + self.cfg.gamma * float(self.q[next_state][self.allowed_actions].max())
        key = state + (int(action),)
        self.q[key] += self.cfg.alpha_lr * (target - self.q[key])

    def save(self, path: Path) -> None:
        np.save(str(path), self.q)

    def load(self, path: Path) -> None:
        q = np.load(str(path))
        if q.shape != Q_SHAPE:
            raise ValueError("Wrong Q-table shape: %r" % (q.shape,))
        self.q = q.astype(np.float32, copy=False)
