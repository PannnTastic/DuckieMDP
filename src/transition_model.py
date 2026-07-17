from collections import defaultdict
from pathlib import Path
from typing import DefaultDict, Dict, Iterable, List, Optional, Tuple

import numpy as np


State = Tuple[int, ...]
Outcome = Tuple[float, Optional[State], float, bool]
Key = Tuple[State, int, Optional[State], bool]


class EmpiricalTransitionModel:
    """Sparse count-based estimate of P(s'|s,a) and E[r|s,a,s']."""

    def __init__(self) -> None:
        self.counts: DefaultDict[Key, int] = defaultdict(int)
        self.reward_sums: DefaultDict[Key, float] = defaultdict(float)

    def observe(
        self,
        state: State,
        action: int,
        reward: float,
        next_state: Optional[State],
        terminal: bool,
    ) -> None:
        key = (tuple(state), int(action), None if terminal else tuple(next_state), bool(terminal))
        self.counts[key] += 1
        self.reward_sums[key] += float(reward)

    @property
    def source_states(self) -> Tuple[State, ...]:
        return tuple(sorted({key[0] for key in self.counts}))

    def observed_actions(self, state: State) -> Tuple[int, ...]:
        return tuple(sorted({key[1] for key in self.counts if key[0] == tuple(state)}))

    def outcomes(self, state: State, action: int) -> List[Outcome]:
        matching = [key for key in self.counts if key[0] == tuple(state) and key[1] == int(action)]
        total = sum(self.counts[key] for key in matching)
        if total == 0:
            return []
        result = []
        for key in matching:
            count = self.counts[key]
            result.append(
                (count / total, key[2], self.reward_sums[key] / count, key[3])
            )
        return result

    def coverage(self, allowed_actions: Iterable[int]) -> Dict[str, float]:
        states = self.source_states
        allowed = tuple(int(a) for a in allowed_actions)
        pairs = sum(len(self.observed_actions(state)) for state in states)
        possible = len(states) * len(allowed)
        return {
            "observed_states": len(states),
            "observed_state_action_pairs": pairs,
            "state_action_coverage": pairs / possible if possible else 0.0,
            "transition_outcomes": len(self.counts),
            "samples": int(sum(self.counts.values())),
        }

    def save(self, path: Path) -> None:
        keys = sorted(self.counts, key=lambda key: (key[0], key[1], key[3], key[2] or ()))
        state_width = len(keys[0][0]) if keys else 0
        states = np.asarray([key[0] for key in keys], dtype=np.int16).reshape(-1, state_width)
        next_states = np.full((len(keys), state_width), -1, dtype=np.int16)
        for index, key in enumerate(keys):
            if key[2] is not None:
                next_states[index] = key[2]
        np.savez_compressed(
            str(path),
            states=states,
            actions=np.asarray([key[1] for key in keys], dtype=np.int8),
            next_states=next_states,
            terminals=np.asarray([key[3] for key in keys], dtype=np.bool_),
            counts=np.asarray([self.counts[key] for key in keys], dtype=np.int64),
            reward_sums=np.asarray([self.reward_sums[key] for key in keys], dtype=np.float64),
        )

    @classmethod
    def load(cls, path: Path) -> "EmpiricalTransitionModel":
        data = np.load(str(path))
        model = cls()
        for state, action, next_state, terminal, count, reward_sum in zip(
            data["states"], data["actions"], data["next_states"], data["terminals"],
            data["counts"], data["reward_sums"]
        ):
            terminal = bool(terminal)
            key = (
                tuple(int(v) for v in state),
                int(action),
                None if terminal else tuple(int(v) for v in next_state),
                terminal,
            )
            model.counts[key] = int(count)
            model.reward_sums[key] = float(reward_sum)
        return model
