"""Lightweight deterministic adapter for a trained SAC actor."""

from hashlib import sha256
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch

from ..agents.sac import SACConfig, SquashedGaussianActor
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
    """Load our local training artifact across PyTorch 1.x/2.x defaults."""
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:  # PyTorch versions before the weights_only argument.
        return torch.load(path, map_location=device)


class SACPolicyAdapter:
    """Load only the actor, avoiding the training replay-buffer allocation."""

    def __init__(
        self,
        actor: SquashedGaussianActor,
        observation_config: ContinuousStateConfig,
        checkpoint_obs_dim: int,
        observation_expanded: bool,
        checkpoint_hash: Optional[str] = None,
        checkpoint_path: Optional[str] = None,
        device: str = "cpu",
    ) -> None:
        self.device = torch.device(device)
        self.actor = actor.to(self.device).eval()
        self.observation_config = observation_config
        self.observation_dim = len(OBSERVATION_NAMES)
        self.checkpoint_obs_dim = int(checkpoint_obs_dim)
        self.observation_expanded = bool(observation_expanded)
        self.checkpoint_hash = checkpoint_hash
        self.checkpoint_path = checkpoint_path
        self.observation_space = continuous_observation_space()

    @classmethod
    def from_checkpoint(
        cls,
        path: Path,
        observation_config: ContinuousStateConfig = ContinuousStateConfig(),
        device: str = "cpu",
        allow_observation_expansion: bool = False,
    ) -> "SACPolicyAdapter":
        checkpoint = Path(path)
        torch_device = torch.device(device)
        payload = _load_trusted_checkpoint(checkpoint, torch_device)
        required = {"config", "obs_dim", "action_low", "action_high", "actor"}
        missing = sorted(required.difference(payload))
        if missing:
            raise ValueError("SAC checkpoint missing keys: %s" % missing)

        cfg = SACConfig(**payload["config"])
        checkpoint_obs_dim = int(payload["obs_dim"])
        target_obs_dim = len(OBSERVATION_NAMES)
        expanded = checkpoint_obs_dim != target_obs_dim
        if expanded and not (
            allow_observation_expansion
            and checkpoint_obs_dim + 1 == target_obs_dim
        ):
            raise ValueError(
                "Checkpoint observation dimension mismatch: "
                "checkpoint=%d, adapter=%d"
                % (checkpoint_obs_dim, target_obs_dim)
            )

        action_low = np.asarray(payload["action_low"], dtype=np.float32)
        action_high = np.asarray(payload["action_high"], dtype=np.float32)
        if action_low.shape != (2,) or action_high.shape != (2,):
            raise ValueError("SAC checkpoint must use action (v_cmd, omega_cmd)")
        if np.any(action_high <= action_low):
            raise ValueError("invalid SAC action bounds")

        actor = SquashedGaussianActor(
            target_obs_dim,
            action_low,
            action_high,
            cfg.hidden_size,
        ).to(torch_device)
        actor_state = payload["actor"]
        if expanded:
            actor_state = cls._expanded_actor_state(actor, actor_state)
        actor.load_state_dict(actor_state)
        return cls(
            actor=actor,
            observation_config=observation_config,
            checkpoint_obs_dim=checkpoint_obs_dim,
            observation_expanded=expanded,
            checkpoint_hash=_file_sha256(checkpoint),
            checkpoint_path=str(checkpoint),
            device=device,
        )

    @staticmethod
    def _expanded_actor_state(
        actor: SquashedGaussianActor,
        source_state: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """Append a zero-weight feature while preserving the old actor."""
        target_state = actor.state_dict()
        expanded: Dict[str, torch.Tensor] = {}
        for name, target in target_state.items():
            if name not in source_state:
                raise ValueError("checkpoint actor missing tensor %s" % name)
            source = source_state[name]
            if name == "backbone.0.weight":
                if source.shape[0] != target.shape[0] or (
                    source.shape[1] + 1 != target.shape[1]
                ):
                    raise ValueError("unsupported actor observation expansion")
                value = torch.zeros_like(target)
                value[:, : source.shape[1]] = source.to(target.device)
                expanded[name] = value
            else:
                if source.shape != target.shape:
                    raise ValueError("checkpoint actor shape mismatch at %s" % name)
                expanded[name] = source.to(target.device)
        return expanded

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
                "SAC observation shape must be (%d,), got %r"
                % (self.observation_dim, vector.shape)
            )
        if not np.all(np.isfinite(vector)):
            raise ValueError("SAC observation contains non-finite values")
        if not self.observation_space.contains(vector):
            raise ValueError("SAC observation lies outside declared bounds")

        tensor = torch.as_tensor(
            vector,
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(0)
        with torch.no_grad():
            distribution = self.actor.distribution(tensor)
            latent_mean = distribution.mean
            latent_std = distribution.stddev
            squashed = torch.tanh(latent_mean)
            action_tensor = (
                squashed * self.actor.action_scale + self.actor.action_bias
            )
            correction = torch.log(
                self.actor.action_scale * (1.0 - squashed.pow(2)) + 1e-6
            )
            log_probability = (
                distribution.log_prob(latent_mean) - correction
            ).sum(dim=-1)

        action_values = action_tensor.squeeze(0).cpu().numpy()
        action = CanonicalAction(
            solver=SolverKind.SAC,
            v_cmd=float(action_values[0]),
            omega_cmd=float(action_values[1]),
        )
        return PolicyDecision(
            solver=SolverKind.SAC,
            policy_mode=PolicyMode.DETERMINISTIC_ACTOR_MEAN,
            state=state,
            action=action,
            diagnostics={
                "observation": tuple(float(value) for value in vector),
                "observation_names": OBSERVATION_NAMES,
                "latent_mean": tuple(
                    float(value) for value in latent_mean.squeeze(0).cpu().numpy()
                ),
                "latent_std": tuple(
                    float(value) for value in latent_std.squeeze(0).cpu().numpy()
                ),
                "actor_log_probability_at_mean": float(
                    log_probability.squeeze(0).cpu().item()
                ),
            },
            metadata={
                "teacher_active": False,
                "actor_sampling": False,
                "checkpoint_hash_sha256": self.checkpoint_hash,
                "checkpoint_path": self.checkpoint_path,
                "checkpoint_observation_dim": self.checkpoint_obs_dim,
                "adapter_observation_dim": self.observation_dim,
                "observation_expanded": self.observation_expanded,
                "device": str(self.device),
            },
        )
