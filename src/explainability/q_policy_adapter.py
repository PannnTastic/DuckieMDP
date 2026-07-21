"""Deterministic, teacher-free adapter for tabular Q-learning policies."""

from hashlib import sha256
from pathlib import Path
from typing import Optional, Sequence, Tuple

import numpy as np

from ..actions import ActionConfig, build_action_table
from ..discretizer import Q_SHAPE, STATE_SHAPE, discretize
from ..state import RawState
from .schema import (
    CanonicalAction,
    CanonicalState,
    PolicyDecision,
    PolicyMode,
    SolverKind,
    TABULAR_SOLVERS,
)
from .semantic_state import (
    canonical_from_discrete_index,
    canonical_from_raw_state,
    raw_state_from_canonical,
)


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class QPolicyAdapter:
    """Expose one Q-table through the solver-neutral decision schema.

    Evaluation in the training agent randomly resolves exact argmax ties.  An
    explanation must be reproducible, so this adapter records every tied action
    and deterministically selects the smallest action id.  It never calls a
    teacher and never mutates the table or the agent RNG.
    """

    def __init__(
        self,
        q_table: np.ndarray,
        allowed_actions: Sequence[int] = tuple(range(7)),
        action_config: ActionConfig = ActionConfig(),
        checkpoint_hash: Optional[str] = None,
        checkpoint_path: Optional[str] = None,
        solver_kind: SolverKind = SolverKind.Q_LEARNING,
    ) -> None:
        table = np.asarray(q_table)
        if table.shape != Q_SHAPE:
            raise ValueError("Wrong Q-table shape: %r" % (table.shape,))
        if not np.all(np.isfinite(table)):
            raise ValueError("Q-table contains non-finite values")
        allowed = tuple(int(action) for action in allowed_actions)
        if not allowed or len(set(allowed)) != len(allowed):
            raise ValueError("allowed_actions must be non-empty and unique")
        if any(action < 0 or action >= Q_SHAPE[-1] for action in allowed):
            raise ValueError("allowed_actions must contain ids in [0, 6]")
        solver_kind = SolverKind(solver_kind)
        if solver_kind not in TABULAR_SOLVERS:
            raise ValueError("QPolicyAdapter requires a tabular solver kind")

        # Copy to make the adapter a read-only snapshot even if training keeps
        # modifying its original array in another object.
        self.q_table = table.astype(np.float32, copy=True)
        self.q_table.setflags(write=False)
        self.allowed_actions = allowed
        self.action_table = build_action_table(action_config)
        self.checkpoint_hash = checkpoint_hash
        self.checkpoint_path = checkpoint_path
        self.solver_kind = solver_kind

    @classmethod
    def from_checkpoint(
        cls,
        path: Path,
        allowed_actions: Sequence[int] = tuple(range(7)),
        action_config: ActionConfig = ActionConfig(),
        solver_kind: SolverKind = SolverKind.Q_LEARNING,
    ) -> "QPolicyAdapter":
        checkpoint = Path(path)
        table = np.load(str(checkpoint), allow_pickle=False)
        return cls(
            table,
            allowed_actions=allowed_actions,
            action_config=action_config,
            checkpoint_hash=_file_sha256(checkpoint),
            checkpoint_path=str(checkpoint),
            solver_kind=solver_kind,
        )

    def decide(self, state: CanonicalState) -> PolicyDecision:
        raw = raw_state_from_canonical(state)
        return self._decide_index(discretize(raw), state)

    def decide_raw(self, raw: RawState) -> PolicyDecision:
        return self._decide_index(
            discretize(raw),
            canonical_from_raw_state(raw),
        )

    def decide_index(
        self,
        index: Sequence[int],
        canonical_state: Optional[CanonicalState] = None,
    ) -> PolicyDecision:
        key = self._validate_index(index)
        state = canonical_state or canonical_from_discrete_index(key)
        return self._decide_index(key, state)

    @staticmethod
    def _validate_index(index: Sequence[int]) -> Tuple[int, ...]:
        key = tuple(int(item) for item in index)
        if len(key) != len(STATE_SHAPE) or any(
            item < 0 or item >= size for item, size in zip(key, STATE_SHAPE)
        ):
            raise ValueError("invalid Q-table state index: %r" % (key,))
        return key

    def _decide_index(
        self,
        index: Tuple[int, ...],
        state: CanonicalState,
    ) -> PolicyDecision:
        values = np.asarray(self.q_table[index], dtype=np.float64)
        allowed_values = values[np.asarray(self.allowed_actions, dtype=int)]
        best_value = float(np.max(allowed_values))
        ties = tuple(
            action
            for action in self.allowed_actions
            if bool(np.isclose(values[action], best_value, rtol=0.0, atol=1e-12))
        )
        selected = min(ties)
        spec = self.action_table[selected]

        sorted_values = np.sort(allowed_values)[::-1]
        q_margin = (
            None
            if sorted_values.size < 2
            else float(sorted_values[0] - sorted_values[1])
        )
        action = CanonicalAction(
            solver=self.solver_kind,
            action_id=selected,
            action_name=spec.name,
            v_cmd=float(spec.v),
            omega_cmd=float(spec.omega),
        )
        return PolicyDecision(
            solver=self.solver_kind,
            policy_mode=PolicyMode.GREEDY,
            state=state,
            action=action,
            diagnostics={
                "discrete_state": index,
                "q_values": tuple(float(value) for value in values),
                "allowed_actions": self.allowed_actions,
                "greedy_ties": ties,
                "q_margin": q_margin,
            },
            metadata={
                "teacher_active": False,
                "tie_break": "lowest_action_id",
                "checkpoint_hash_sha256": self.checkpoint_hash,
                "checkpoint_path": self.checkpoint_path,
            },
        )
