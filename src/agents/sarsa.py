"""Tabular on-policy SARSA untuk finite MDP Duckietown.

Berbeda dari Q-learning yang memakai ``max_a' Q(s', a')``, SARSA memakai
nilai aksi berikutnya yang benar-benar dipilih behavior policy:

    Q(s,a) <- Q(s,a) + alpha * [target - Q(s,a)]
    target = r                              jika terminal
    target = r + gamma * Q(s', a')          selain itu

Nama SARSA berasal dari tuple pengalaman ``(S, A, R, S', A')``. Agent tetap
model-free: ia memakai satu sampel hasil simulator dan tidak membutuhkan
transition model ``P(s'|s,a)``.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence, Tuple

import numpy as np

from ..discretizer import Q_SHAPE


@dataclass(frozen=True)
class SarsaConfig:
    gamma: float = 0.99
    alpha_lr: float = 0.10
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_steps: int = 50000
    allowed_actions: Sequence[int] = (0, 1, 2, 3, 4, 5, 6)


class SarsaAgent:
    """Epsilon-greedy tabular SARSA dengan random tie-breaking."""

    def __init__(self, cfg: SarsaConfig, seed: int = 0) -> None:
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

    def select_action(
        self,
        state: Tuple[int, ...],
        greedy: bool = False,
        advance_step: bool = True,
    ) -> int:
        """Pilih aksi behavior; ``greedy=True`` dipakai saat evaluasi.

        ``advance_step=False`` berguna untuk menghitung aksi bootstrap pada
        artificial time-limit tanpa menganggapnya sebagai keputusan baru yang
        benar-benar dieksekusi.
        """
        if not greedy and self.rng.random_sample() < self.epsilon:
            action = int(self.rng.choice(self.allowed_actions))
        else:
            values = self.q[state][self.allowed_actions]
            best = self.allowed_actions[np.flatnonzero(values == values.max())]
            action = int(self.rng.choice(best))
        if not greedy and advance_step:
            self.steps += 1
        return action

    def update(
        self,
        state: Tuple[int, ...],
        action: int,
        reward: float,
        next_state: Optional[Tuple[int, ...]],
        next_action: Optional[int],
        done: bool,
    ) -> None:
        """Update dari sampel ``(s, a, r, s', a')``.

        ``done`` hanya berarti terminal sejati. Timeout/truncation mengirim
        ``done=False`` agar nilai kelanjutan tetap di-bootstrap.
        """
        if int(action) not in self.allowed_actions:
            raise ValueError(f"Action {action} is masked")
        if done:
            target = float(reward)
        else:
            if next_state is None or next_action is None:
                raise ValueError("Non-terminal SARSA update requires next_state and next_action")
            if int(next_action) not in self.allowed_actions:
                raise ValueError(f"Next action {next_action} is masked")
            target = float(reward) + self.cfg.gamma * float(self.q[next_state + (int(next_action),)])
        key = state + (int(action),)
        self.q[key] += self.cfg.alpha_lr * (target - self.q[key])

    def save(self, path: Path) -> None:
        np.save(str(path), self.q)

    def load(self, path: Path) -> None:
        q = np.load(str(path))
        if q.shape != Q_SHAPE:
            raise ValueError("Wrong Q-table shape: %r" % (q.shape,))
        self.q = q.astype(np.float32, copy=False)
