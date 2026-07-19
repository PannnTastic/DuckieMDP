"""Soft Actor-Critic untuk continuous privileged state Duckietown."""

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict

import numpy as np
import torch
from torch import nn


@dataclass(frozen=True)
class SACConfig:
    gamma: float = 0.99
    tau: float = 0.005
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    alpha_lr: float = 3e-4
    initial_alpha: float = 0.2
    batch_size: int = 256
    replay_capacity: int = 300000
    hidden_size: int = 256
    target_entropy: float = -2.0


class ReplayBuffer:
    def __init__(self, capacity: int, obs_dim: int, action_dim: int, seed: int):
        self.capacity = int(capacity)
        self.obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.action = np.zeros((capacity, action_dim), dtype=np.float32)
        self.reward = np.zeros((capacity, 1), dtype=np.float32)
        self.next_obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.terminated = np.zeros((capacity, 1), dtype=np.float32)
        self.position = 0
        self.size = 0
        self.rng = np.random.RandomState(seed)

    def add(self, obs, action, reward, next_obs, terminated) -> None:
        index = self.position
        self.obs[index] = obs
        self.action[index] = action
        self.reward[index, 0] = reward
        self.next_obs[index] = next_obs
        # Timeout tidak dimasukkan sebagai terminal agar critic tetap bootstrap.
        self.terminated[index, 0] = float(terminated)
        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int, device: torch.device):
        indices = self.rng.randint(0, self.size, size=batch_size)
        arrays = (
            self.obs[indices],
            self.action[indices],
            self.reward[indices],
            self.next_obs[indices],
            self.terminated[indices],
        )
        return tuple(torch.as_tensor(value, device=device) for value in arrays)

    def __len__(self) -> int:
        return self.size


def _mlp(input_dim: int, output_dim: int, hidden: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(input_dim, hidden),
        nn.ReLU(),
        nn.Linear(hidden, hidden),
        nn.ReLU(),
        nn.Linear(hidden, output_dim),
    )


class SquashedGaussianActor(nn.Module):
    LOG_STD_MIN = -5.0
    LOG_STD_MAX = 2.0

    def __init__(self, obs_dim, action_low, action_high, hidden):
        super().__init__()
        action_low = torch.as_tensor(action_low, dtype=torch.float32)
        action_high = torch.as_tensor(action_high, dtype=torch.float32)
        action_dim = int(action_low.numel())
        self.backbone = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.mean = nn.Linear(hidden, action_dim)
        self.log_std = nn.Linear(hidden, action_dim)
        self.register_buffer("action_scale", (action_high - action_low) / 2.0)
        self.register_buffer("action_bias", (action_high + action_low) / 2.0)

    def distribution(self, obs):
        features = self.backbone(obs)
        mean = self.mean(features)
        log_std = torch.clamp(
            self.log_std(features), self.LOG_STD_MIN, self.LOG_STD_MAX
        )
        return torch.distributions.Normal(mean, log_std.exp())

    def sample(self, obs):
        distribution = self.distribution(obs)
        pre_tanh = distribution.rsample()
        squashed = torch.tanh(pre_tanh)
        action = squashed * self.action_scale + self.action_bias
        correction = torch.log(
            self.action_scale * (1.0 - squashed.pow(2)) + 1e-6
        )
        log_probability = (
            distribution.log_prob(pre_tanh) - correction
        ).sum(dim=-1, keepdim=True)
        deterministic = torch.tanh(distribution.mean) * self.action_scale + self.action_bias
        return action, log_probability, deterministic


class Critic(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, hidden: int):
        super().__init__()
        self.net = _mlp(obs_dim + action_dim, 1, hidden)

    def forward(self, obs, action):
        return self.net(torch.cat([obs, action], dim=-1))


class SACAgent:
    def __init__(
        self,
        obs_dim: int,
        action_low: np.ndarray,
        action_high: np.ndarray,
        cfg: SACConfig,
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

        self.actor = SquashedGaussianActor(
            obs_dim, self.action_low, self.action_high, cfg.hidden_size
        ).to(self.device)
        self.critic1 = Critic(obs_dim, action_dim, cfg.hidden_size).to(self.device)
        self.critic2 = Critic(obs_dim, action_dim, cfg.hidden_size).to(self.device)
        self.target1 = Critic(obs_dim, action_dim, cfg.hidden_size).to(self.device)
        self.target2 = Critic(obs_dim, action_dim, cfg.hidden_size).to(self.device)
        self.target1.load_state_dict(self.critic1.state_dict())
        self.target2.load_state_dict(self.critic2.state_dict())
        for target in (self.target1, self.target2):
            target.requires_grad_(False)

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=cfg.actor_lr)
        critic_parameters = list(self.critic1.parameters()) + list(self.critic2.parameters())
        self.critic_optimizer = torch.optim.Adam(critic_parameters, lr=cfg.critic_lr)
        self.log_alpha = torch.tensor(
            np.log(cfg.initial_alpha),
            dtype=torch.float32,
            device=self.device,
            requires_grad=True,
        )
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=cfg.alpha_lr)
        self.replay = ReplayBuffer(
            cfg.replay_capacity, obs_dim, action_dim, seed + 1
        )
        self.updates = 0

    @property
    def alpha(self):
        return self.log_alpha.exp()

    def select_action(self, observation: np.ndarray, deterministic: bool = False):
        obs = torch.as_tensor(
            observation, dtype=torch.float32, device=self.device
        ).unsqueeze(0)
        with torch.no_grad():
            sampled, _, mean = self.actor.sample(obs)
        action = mean if deterministic else sampled
        return action.squeeze(0).cpu().numpy().astype(np.float32)

    def update(self) -> Dict[str, float]:
        if len(self.replay) < self.cfg.batch_size:
            return {}
        obs, action, reward, next_obs, terminated = self.replay.sample(
            self.cfg.batch_size, self.device
        )

        with torch.no_grad():
            next_action, next_log_probability, _ = self.actor.sample(next_obs)
            next_q = torch.minimum(
                self.target1(next_obs, next_action),
                self.target2(next_obs, next_action),
            )
            target = reward + self.cfg.gamma * (1.0 - terminated) * (
                next_q - self.alpha.detach() * next_log_probability
            )

        q1 = self.critic1(obs, action)
        q2 = self.critic2(obs, action)
        critic_loss = nn.functional.mse_loss(q1, target) + nn.functional.mse_loss(q2, target)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        sampled_action, log_probability, _ = self.actor.sample(obs)
        policy_q = torch.minimum(
            self.critic1(obs, sampled_action),
            self.critic2(obs, sampled_action),
        )
        actor_loss = (self.alpha.detach() * log_probability - policy_q).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        alpha_loss = -(
            self.log_alpha * (log_probability + self.cfg.target_entropy).detach()
        ).mean()
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()

        with torch.no_grad():
            for source, target_network in (
                (self.critic1, self.target1),
                (self.critic2, self.target2),
            ):
                for source_parameter, target_parameter in zip(
                    source.parameters(), target_network.parameters()
                ):
                    target_parameter.mul_(1.0 - self.cfg.tau)
                    target_parameter.add_(self.cfg.tau * source_parameter)

        self.updates += 1
        return {
            "critic_loss": float(critic_loss.item()),
            "actor_loss": float(actor_loss.item()),
            "alpha_loss": float(alpha_loss.item()),
            "alpha": float(self.alpha.detach().item()),
            "mean_q": float(torch.minimum(q1, q2).mean().item()),
        }

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        torch.save(
            {
                "config": asdict(self.cfg),
                "obs_dim": self.obs_dim,
                "action_low": self.action_low,
                "action_high": self.action_high,
                "actor": self.actor.state_dict(),
                "critic1": self.critic1.state_dict(),
                "critic2": self.critic2.state_dict(),
                "target1": self.target1.state_dict(),
                "target2": self.target2.state_dict(),
                "actor_optimizer": self.actor_optimizer.state_dict(),
                "critic_optimizer": self.critic_optimizer.state_dict(),
                "alpha_optimizer": self.alpha_optimizer.state_dict(),
                "log_alpha": self.log_alpha.detach().cpu(),
                "updates": self.updates,
            },
            temporary,
        )
        temporary.replace(path)

    def _expanded_actor_state(self, source_state: Dict[str, torch.Tensor]):
        """Append one zero-weight observation feature without changing policy."""
        target_state = self.actor.state_dict()
        expanded = {}
        for name, target in target_state.items():
            source = source_state[name]
            if name == "backbone.0.weight":
                if source.shape[1] + 1 != target.shape[1]:
                    raise ValueError("Unsupported actor observation expansion")
                value = torch.zeros_like(target)
                value[:, : source.shape[1]] = source.to(value.device)
                expanded[name] = value
            else:
                if source.shape != target.shape:
                    raise ValueError(f"Checkpoint actor shape mismatch at {name}")
                expanded[name] = source
        return expanded

    @staticmethod
    def _expanded_critic_state(
        model: nn.Module,
        source_state: Dict[str, torch.Tensor],
        old_obs_dim: int,
        new_obs_dim: int,
    ):
        """Move action columns right while inserting a zero observation column."""
        target_state = model.state_dict()
        expanded = {}
        for name, target in target_state.items():
            source = source_state[name]
            if name == "net.0.weight":
                action_dim = source.shape[1] - old_obs_dim
                if (
                    new_obs_dim != old_obs_dim + 1
                    or target.shape[1] != new_obs_dim + action_dim
                ):
                    raise ValueError("Unsupported critic observation expansion")
                value = torch.zeros_like(target)
                value[:, :old_obs_dim] = source[:, :old_obs_dim].to(value.device)
                value[:, new_obs_dim:] = source[:, old_obs_dim:].to(value.device)
                expanded[name] = value
            else:
                if source.shape != target.shape:
                    raise ValueError(f"Checkpoint critic shape mismatch at {name}")
                expanded[name] = source
        return expanded

    def load(self, path: Path, allow_observation_expansion: bool = False) -> None:
        payload = torch.load(path, map_location=self.device)
        checkpoint_obs_dim = int(payload["obs_dim"])
        expanded_observation = checkpoint_obs_dim != self.obs_dim
        if expanded_observation and not (
            allow_observation_expansion
            and checkpoint_obs_dim + 1 == self.obs_dim
        ):
            raise ValueError(
                "Checkpoint observation dimension mismatch: "
                f"checkpoint={checkpoint_obs_dim}, environment={self.obs_dim}"
            )
        if not np.allclose(payload["action_low"], self.action_low) or not np.allclose(
            payload["action_high"], self.action_high
        ):
            raise ValueError("Checkpoint action bounds mismatch")
        if expanded_observation:
            self.actor.load_state_dict(self._expanded_actor_state(payload["actor"]))
            for model, key in (
                (self.critic1, "critic1"),
                (self.critic2, "critic2"),
                (self.target1, "target1"),
                (self.target2, "target2"),
            ):
                model.load_state_dict(
                    self._expanded_critic_state(
                        model,
                        payload[key],
                        checkpoint_obs_dim,
                        self.obs_dim,
                    )
                )
        else:
            self.actor.load_state_dict(payload["actor"])
            self.critic1.load_state_dict(payload["critic1"])
            self.critic2.load_state_dict(payload["critic2"])
            self.target1.load_state_dict(payload["target1"])
            self.target2.load_state_dict(payload["target2"])
        if "actor_optimizer" in payload and not expanded_observation:
            self.actor_optimizer.load_state_dict(payload["actor_optimizer"])
            self.critic_optimizer.load_state_dict(payload["critic_optimizer"])
            self.alpha_optimizer.load_state_dict(payload["alpha_optimizer"])
        elif "alpha_optimizer" in payload:
            # Optimizer actor/critic memuat tensor momentum berukuran 14-D dan
            # tidak aman dimigrasikan. Optimizer temperatur hanya satu skalar.
            self.alpha_optimizer.load_state_dict(payload["alpha_optimizer"])
        with torch.no_grad():
            self.log_alpha.copy_(payload["log_alpha"].to(self.device))
        self.updates = int(payload.get("updates", 0))
