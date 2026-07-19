"""DQN ablation: continuous privileged state with discrete macro-actions."""

from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Deque, Sequence, Tuple

import numpy as np

from ..state import RawState

try:
    import torch
    from torch import nn
except ImportError:
    torch, nn = None, None


def encode_state(state: RawState, max_stop_distance: float = 3.0) -> np.ndarray:
    tile = np.eye(3, dtype=np.float32)[int(state.tile)]
    duck = np.eye(5, dtype=np.float32)[int(state.duck)]
    stop = 1.0 if state.d_stop is None else min(state.d_stop / max_stop_distance, 1.0)
    values = np.array([state.d / 0.25, state.phi / (np.pi / 2), state.v / 0.4], dtype=np.float32)
    return np.concatenate([values, tile, [stop, float(state.sigma_stop)], duck]).astype(np.float32)


@dataclass(frozen=True)
class DQNConfig:
    gamma: float = 0.99
    lr: float = 1e-3
    batch_size: int = 64
    replay_size: int = 100000
    target_update: int = 1000
    hidden_size: int = 256
    allowed_actions: Sequence[int] = (0, 1, 2, 3, 4, 5, 6)


class ReplayBuffer:
    def __init__(self, capacity: int, seed: int = 0) -> None:
        self.data: Deque[Tuple] = deque(maxlen=capacity)
        self.rng = np.random.RandomState(seed)

    def add(self, state, action, reward, next_state, done) -> None:
        self.data.append((state, action, reward, next_state, done))

    def sample(self, size: int):
        indices = self.rng.choice(len(self.data), size=size, replace=False)
        batch = [self.data[int(index)] for index in indices]
        return tuple(np.asarray(items) for items in zip(*batch))

    def __len__(self) -> int:
        return len(self.data)


if nn is not None:
    class QNetwork(nn.Module):
        def __init__(
            self,
            input_size: int = 13,
            actions: int = 7,
            hidden_size: int = 256,
        ) -> None:
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(input_size, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, actions),
            )

        def forward(self, x):
            return self.net(x)
else:
    class QNetwork:
        def __init__(self, *args, **kwargs) -> None:
            raise ImportError("Install torch to use DQN")


class DQNAgent:
    def __init__(
        self,
        cfg: DQNConfig,
        device: str = "cpu",
        obs_dim: int = 13,
        actions: int = 7,
        seed: int = 0,
    ) -> None:
        if torch is None:
            raise ImportError("Install torch to use DQN")
        torch.manual_seed(seed)
        self.cfg, self.device, self.updates = cfg, torch.device(device), 0
        self.obs_dim, self.actions = int(obs_dim), int(actions)
        self.rng = np.random.RandomState(seed)
        self.allowed_actions = np.asarray(tuple(cfg.allowed_actions), dtype=np.int64)
        if (
            self.allowed_actions.size == 0
            or np.any(self.allowed_actions < 0)
            or np.any(self.allowed_actions >= self.actions)
        ):
            raise ValueError("allowed_actions contains an invalid action id")
        self.online = QNetwork(obs_dim, actions, cfg.hidden_size).to(self.device)
        self.target = QNetwork(obs_dim, actions, cfg.hidden_size).to(self.device)
        self.target.load_state_dict(self.online.state_dict())
        self.optimizer = torch.optim.Adam(self.online.parameters(), lr=cfg.lr)
        self.buffer = ReplayBuffer(cfg.replay_size, seed + 1)

    def _encode(self, state) -> np.ndarray:
        if isinstance(state, RawState):
            return encode_state(state)
        value = np.asarray(state, dtype=np.float32).reshape(-1)
        if value.shape != (self.obs_dim,):
            raise ValueError(
                f"Expected observation shape {(self.obs_dim,)}, got {value.shape}"
            )
        return value

    def select_action(self, state) -> int:
        x = torch.as_tensor(
            self._encode(state), dtype=torch.float32, device=self.device
        ).unsqueeze(0)
        with torch.no_grad():
            q_values = self.online(x)[0, self.allowed_actions]
        best = np.flatnonzero(
            q_values.cpu().numpy() == float(q_values.max().item())
        )
        return int(self.allowed_actions[int(self.rng.choice(best))])

    def train_step(self) -> float:
        if len(self.buffer) < self.cfg.batch_size:
            return 0.0
        s, a, r, ns, done = self.buffer.sample(self.cfg.batch_size)
        s = torch.as_tensor(np.stack(s), dtype=torch.float32, device=self.device)
        ns = torch.as_tensor(np.stack(ns), dtype=torch.float32, device=self.device)
        a = torch.as_tensor(a, dtype=torch.long, device=self.device)
        r = torch.as_tensor(r, dtype=torch.float32, device=self.device)
        done = torch.as_tensor(done, dtype=torch.float32, device=self.device)
        prediction = self.online(s).gather(1, a[:, None]).squeeze(1)
        with torch.no_grad():
            next_values = self.target(ns)[:, self.allowed_actions].max(1).values
            target = r + self.cfg.gamma * (1.0 - done) * next_values
        loss = nn.functional.smooth_l1_loss(prediction, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.updates += 1
        if self.updates % self.cfg.target_update == 0:
            self.target.load_state_dict(self.online.state_dict())
        return float(loss.item())

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        torch.save(
            {
                "config": asdict(self.cfg),
                "obs_dim": self.obs_dim,
                "actions": self.actions,
                "online": self.online.state_dict(),
                "target": self.target.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "updates": self.updates,
            },
            temporary,
        )
        temporary.replace(path)

    def load(self, path: Path) -> None:
        payload = torch.load(path, map_location=self.device)
        if int(payload["obs_dim"]) != self.obs_dim:
            raise ValueError("DQN checkpoint observation dimension mismatch")
        if int(payload["actions"]) != self.actions:
            raise ValueError("DQN checkpoint action count mismatch")
        self.online.load_state_dict(payload["online"])
        self.target.load_state_dict(payload["target"])
        if "optimizer" in payload:
            self.optimizer.load_state_dict(payload["optimizer"])
        self.updates = int(payload.get("updates", 0))
