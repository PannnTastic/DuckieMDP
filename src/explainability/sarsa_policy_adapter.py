"""Deterministic, teacher-free adapter for a frozen tabular SARSA policy."""

from pathlib import Path
from typing import Sequence

from ..actions import ActionConfig
from .q_policy_adapter import QPolicyAdapter, _file_sha256
from .schema import SolverKind


class SarsaPolicyAdapter(QPolicyAdapter):
    """Expose a SARSA table while preserving its solver identity.

    SARSA and Q-learning share the same finite state/action representation in
    this repository. Their Bellman targets differ during training, but frozen
    evaluation for both is a deterministic greedy lookup.
    """

    @classmethod
    def from_checkpoint(
        cls,
        path: Path,
        allowed_actions: Sequence[int] = tuple(range(7)),
        action_config: ActionConfig = ActionConfig(),
    ) -> "SarsaPolicyAdapter":
        checkpoint = Path(path)
        import numpy as np

        table = np.load(str(checkpoint), allow_pickle=False)
        return cls(
            table,
            allowed_actions=allowed_actions,
            action_config=action_config,
            checkpoint_hash=_file_sha256(checkpoint),
            checkpoint_path=str(checkpoint),
            solver_kind=SolverKind.SARSA,
        )

