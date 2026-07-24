"""Lightweight deterministic adapter for a trained TD3 actor.

TD3 shares SAC's continuous representation (15-dim metric observation, a
(v_cmd, omega_cmd) action), so the explanation pipeline treats it through the
``CONTINUOUS_SOLVERS`` branches. The only difference from the SAC adapter is
that the TD3 actor is a plain deterministic network with no latent
distribution, so the decision carries no latent diagnostics.
"""

from hashlib import sha256
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch

from ..agents.td3 import DeterministicActor
from ..continuous_state import (
    OBSERVATION_NAMES,
    ContinuousState,
    ContinuousStateConfig,
    continuous_observation_space,
)
from .schema import (
    CanonicalAction,
    CanonicalState,
    PolicyDecision,
    PolicyMode,
    SolverKind,
)
from .semantic_state import (
    canonical_from_continuous_state,
    encode_canonical_for_sac,
)


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_trusted_checkpoint(path: Path, device: torch.device) -> Dict:
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


class TD3PolicyAdapter:
    """Load only the deterministic actor for explanation-time inference."""

    def __init__(
        self,
        actor: DeterministicActor,
        observation_config: ContinuousStateConfig,
        checkpoint_obs_dim: int,
        checkpoint_hash: Optional[str] = None,
        checkpoint_path: Optional[str] = None,
        device: str = "cpu",
    ) -> None:
        self.device = torch.device(device)
        self.actor = actor.to(self.device).eval()
        self.observation_config = observation_config
        self.observation_dim = len(OBSERVATION_NAMES)
        self.checkpoint_obs_dim = int(checkpoint_obs_dim)
        self.checkpoint_hash = checkpoint_hash
        self.checkpoint_path = checkpoint_path
        self.solver_kind = SolverKind.TD3
        self.observation_space = continuous_observation_space()

    @classmethod
    def from_checkpoint(
        cls,
        path: Path,
        observation_config: ContinuousStateConfig = ContinuousStateConfig(),
        device: str = "cpu",
    ) -> "TD3PolicyAdapter":
        checkpoint = Path(path)
        torch_device = torch.device(device)
        payload = _load_trusted_checkpoint(checkpoint, torch_device)
        required = {"config", "obs_dim", "action_low", "action_high", "actor"}
        missing = sorted(required.difference(payload))
        if missing:
            raise ValueError("TD3 checkpoint missing keys: %s" % missing)

        checkpoint_obs_dim = int(payload["obs_dim"])
        target_obs_dim = len(OBSERVATION_NAMES)
        if checkpoint_obs_dim != target_obs_dim:
            raise ValueError(
                "Checkpoint observation dimension mismatch: "
                "checkpoint=%d, adapter=%d"
                % (checkpoint_obs_dim, target_obs_dim)
            )

        action_low = np.asarray(payload["action_low"], dtype=np.float32)
        action_high = np.asarray(payload["action_high"], dtype=np.float32)
        if action_low.shape != (2,) or action_high.shape != (2,):
            raise ValueError("TD3 checkpoint must use action (v_cmd, omega_cmd)")
        if np.any(action_high <= action_low):
            raise ValueError("invalid TD3 action bounds")

        hidden = int(payload["config"]["hidden_size"])
        actor = DeterministicActor(
            target_obs_dim, action_low, action_high, hidden
        ).to(torch_device)
        actor.load_state_dict(payload["actor"])
        return cls(
            actor=actor,
            observation_config=observation_config,
            checkpoint_obs_dim=checkpoint_obs_dim,
            checkpoint_hash=_file_sha256(checkpoint),
            checkpoint_path=str(checkpoint),
            device=device,
        )

    def decide(self, state: CanonicalState) -> PolicyDecision:
        observation = encode_canonical_for_sac(state, self.observation_config)
        return self.decide_encoded(observation, state)

    def decide_continuous(self, state: ContinuousState) -> PolicyDecision:
        return self.decide(canonical_from_continuous_state(state))

    def decide_encoded(
        self,
        observation: np.ndarray,
        state: CanonicalState,
    ) -> PolicyDecision:
        vector = np.asarray(observation, dtype=np.float32)
        if vector.shape != (self.observation_dim,):
            raise ValueError(
                "TD3 observation shape must be (%d,), got %r"
                % (self.observation_dim, vector.shape)
            )
        if not np.all(np.isfinite(vector)):
            raise ValueError("TD3 observation contains non-finite values")
        if not self.observation_space.contains(vector):
            raise ValueError("TD3 observation lies outside declared bounds")

        tensor = torch.as_tensor(
            vector, dtype=torch.float32, device=self.device
        ).unsqueeze(0)
        with torch.no_grad():
            action_tensor = self.actor(tensor)
        action_values = action_tensor.squeeze(0).cpu().numpy()
        action = CanonicalAction(
            solver=SolverKind.TD3,
            v_cmd=float(action_values[0]),
            omega_cmd=float(action_values[1]),
        )
        return PolicyDecision(
            solver=SolverKind.TD3,
            policy_mode=PolicyMode.DETERMINISTIC_ACTOR_MEAN,
            state=state,
            action=action,
            diagnostics={
                "observation": tuple(float(value) for value in vector),
                "observation_names": OBSERVATION_NAMES,
            },
            metadata={
                "teacher_active": False,
                "actor_sampling": False,
                "checkpoint_hash_sha256": self.checkpoint_hash,
                "checkpoint_path": self.checkpoint_path,
                "checkpoint_observation_dim": self.checkpoint_obs_dim,
                "adapter_observation_dim": self.observation_dim,
                "device": str(self.device),
            },
        )
