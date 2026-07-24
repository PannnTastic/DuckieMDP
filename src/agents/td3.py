"""Twin Delayed DDPG untuk continuous privileged state Duckietown.

Struktur checkpoint dan replay mengikuti agents/sac.py agar tooling hilir
(evaluasi, adapter explanation) dapat diperluas dengan pola yang sama.
Perbedaan algoritmik terhadap SAC: actor deterministik dengan noise
eksplorasi Gaussian, target policy smoothing, dan delayed policy update.
"""

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict

import numpy as np
import torch
from torch import nn

from .sac import Critic, ReplayBuffer


@dataclass(frozen=True)
class TD3Config:
    gamma: float = 0.99
    tau: float = 0.005
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    batch_size: int = 256
    replay_capacity: int = 300000
    hidden_size: int = 256
    exploration_noise: float = 0.10
    target_policy_noise: float = 0.20
    target_noise_clip: float = 0.50
    policy_delay: int = 2
    # Freeze the actor for this many critic updates before letting it move.
    # Zero keeps standard TD3. A positive value is a critic warm-up used when
    # fine-tuning a good policy under a changed reward: the stale critic re-fits
    # the new returns while the loaded actor keeps driving, so actor updates do
    # not chase a wrong critic and destroy the policy.
    actor_update_start: int = 0


class DeterministicActor(nn.Module):
    def __init__(self, obs_dim, action_low, action_high, hidden):
        super().__init__()
        action_low = torch.as_tensor(action_low, dtype=torch.float32)
        action_high = torch.as_tensor(action_high, dtype=torch.float32)
        action_dim = int(action_low.numel())
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, action_dim),
        )
        self.register_buffer("action_scale", (action_high - action_low) / 2.0)
        self.register_buffer("action_bias", (action_high + action_low) / 2.0)

    def forward(self, obs):
        return torch.tanh(self.net(obs)) * self.action_scale + self.action_bias


class TD3Agent:
    def __init__(
        self,
        obs_dim: int,
        action_low: np.ndarray,
        action_high: np.ndarray,
        cfg: TD3Config,
        seed: int,
        device: str = "cpu",
    ) -> None:
        np.random.seed(seed)
        torch.manual_seed(seed)
        self.cfg = cfg
        self.device = torch.device(device)
        self.obs_dim = int(obs_dim)
        self.action_low = np.asarray(action_low, dtype=np.float32)
        self.action_high = np.asarray(action_high, dtype=np.float32)
        action_dim = int(self.action_low.size)

        self.actor = DeterministicActor(
            obs_dim, self.action_low, self.action_high, cfg.hidden_size
        ).to(self.device)
        self.actor_target = DeterministicActor(
            obs_dim, self.action_low, self.action_high, cfg.hidden_size
        ).to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_target.requires_grad_(False)
        self.critic1 = Critic(obs_dim, action_dim, cfg.hidden_size).to(self.device)
        self.critic2 = Critic(obs_dim, action_dim, cfg.hidden_size).to(self.device)
        self.target1 = Critic(obs_dim, action_dim, cfg.hidden_size).to(self.device)
        self.target2 = Critic(obs_dim, action_dim, cfg.hidden_size).to(self.device)
        self.target1.load_state_dict(self.critic1.state_dict())
        self.target2.load_state_dict(self.critic2.state_dict())
        for target in (self.target1, self.target2):
            target.requires_grad_(False)

        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=cfg.actor_lr
        )
        critic_parameters = (
            list(self.critic1.parameters()) + list(self.critic2.parameters())
        )
        self.critic_optimizer = torch.optim.Adam(
            critic_parameters, lr=cfg.critic_lr
        )
        self.replay = ReplayBuffer(
            cfg.replay_capacity, obs_dim, action_dim, seed + 1
        )
        self.noise_rng = np.random.RandomState(seed + 2)
        self.updates = 0
        # Baseline for the critic warm-up. Loading a checkpoint restores a large
        # cumulative ``updates`` count, so the warm-up must be measured relative
        # to where this training run begins, not from zero.
        self._warmup_baseline = 0

    def select_action(self, observation: np.ndarray, deterministic: bool = False):
        obs = torch.as_tensor(
            observation, dtype=torch.float32, device=self.device
        ).unsqueeze(0)
        with torch.no_grad():
            action = self.actor(obs).squeeze(0).cpu().numpy().astype(np.float32)
        if not deterministic:
            scale = (self.action_high - self.action_low) / 2.0
            action = action + self.noise_rng.normal(
                0.0, self.cfg.exploration_noise * scale
            ).astype(np.float32)
        return np.clip(action, self.action_low, self.action_high)

    def update(self) -> Dict[str, float]:
        if len(self.replay) < self.cfg.batch_size:
            return {}
        obs, action, reward, next_obs, terminated = self.replay.sample(
            self.cfg.batch_size, self.device
        )

        with torch.no_grad():
            scale = self.actor.action_scale
            noise = torch.clamp(
                torch.randn_like(action) * self.cfg.target_policy_noise * scale,
                -self.cfg.target_noise_clip * scale,
                self.cfg.target_noise_clip * scale,
            )
            low = torch.as_tensor(self.action_low, device=self.device)
            high = torch.as_tensor(self.action_high, device=self.device)
            next_action = torch.clamp(
                self.actor_target(next_obs) + noise, low, high
            )
            next_q = torch.minimum(
                self.target1(next_obs, next_action),
                self.target2(next_obs, next_action),
            )
            target = reward + self.cfg.gamma * (1.0 - terminated) * next_q

        q1 = self.critic1(obs, action)
        q2 = self.critic2(obs, action)
        critic_loss = (
            nn.functional.mse_loss(q1, target)
            + nn.functional.mse_loss(q2, target)
        )
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        metrics = {
            "critic_loss": float(critic_loss.item()),
            "mean_q": float(torch.minimum(q1, q2).mean().item()),
        }

        self.updates += 1
        if self.updates % self.cfg.policy_delay == 0:
            if self.updates - self._warmup_baseline >= self.cfg.actor_update_start:
                actor_loss = -self.critic1(obs, self.actor(obs)).mean()
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()
                metrics["actor_loss"] = float(actor_loss.item())

            with torch.no_grad():
                for source, target_network in (
                    (self.critic1, self.target1),
                    (self.critic2, self.target2),
                    (self.actor, self.actor_target),
                ):
                    for source_parameter, target_parameter in zip(
                        source.parameters(), target_network.parameters()
                    ):
                        target_parameter.mul_(1.0 - self.cfg.tau)
                        target_parameter.add_(self.cfg.tau * source_parameter)
        return metrics

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        torch.save(
            {
                "algorithm": "td3",
                "config": asdict(self.cfg),
                "obs_dim": self.obs_dim,
                "action_low": self.action_low,
                "action_high": self.action_high,
                "actor": self.actor.state_dict(),
                "actor_target": self.actor_target.state_dict(),
                "critic1": self.critic1.state_dict(),
                "critic2": self.critic2.state_dict(),
                "target1": self.target1.state_dict(),
                "target2": self.target2.state_dict(),
                "actor_optimizer": self.actor_optimizer.state_dict(),
                "critic_optimizer": self.critic_optimizer.state_dict(),
                "updates": self.updates,
            },
            temporary,
        )
        temporary.replace(path)

    def load(self, path: Path) -> None:
        payload = torch.load(path, map_location=self.device)
        if int(payload["obs_dim"]) != self.obs_dim:
            raise ValueError(
                "Checkpoint observation dimension mismatch: "
                f"checkpoint={payload['obs_dim']}, environment={self.obs_dim}"
            )
        if not np.allclose(
            payload["action_low"], self.action_low
        ) or not np.allclose(payload["action_high"], self.action_high):
            raise ValueError("Checkpoint action bounds mismatch")
        self.actor.load_state_dict(payload["actor"])
        self.actor_target.load_state_dict(payload["actor_target"])
        self.critic1.load_state_dict(payload["critic1"])
        self.critic2.load_state_dict(payload["critic2"])
        self.target1.load_state_dict(payload["target1"])
        self.target2.load_state_dict(payload["target2"])
        self.actor_optimizer.load_state_dict(payload["actor_optimizer"])
        self.critic_optimizer.load_state_dict(payload["critic_optimizer"])
        self.updates = int(payload.get("updates", 0))
        # Restart the critic warm-up window from the resumed update count.
        self._warmup_baseline = self.updates
