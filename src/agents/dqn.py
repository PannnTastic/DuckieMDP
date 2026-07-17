from collections import deque
from dataclasses import dataclass
import random
from typing import Deque, Tuple
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


class ReplayBuffer:
    def __init__(self, capacity: int) -> None:
        self.data: Deque[Tuple] = deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, done) -> None:
        self.data.append((state, action, reward, next_state, done))

    def sample(self, size: int):
        batch = random.sample(self.data, size)
        return tuple(np.asarray(items) for items in zip(*batch))

    def __len__(self) -> int:
        return len(self.data)


if nn is not None:
    class QNetwork(nn.Module):
        def __init__(self, input_size: int = 13, actions: int = 7) -> None:
            super().__init__()
            self.net = nn.Sequential(nn.Linear(input_size, 128), nn.ReLU(),
                                     nn.Linear(128, 128), nn.ReLU(), nn.Linear(128, actions))

        def forward(self, x):
            return self.net(x)
else:
    class QNetwork:
        def __init__(self, *args, **kwargs) -> None:
            raise ImportError("Install torch to use DQN")


class DQNAgent:
    def __init__(self, cfg: DQNConfig, device: str = "cpu") -> None:
        if torch is None:
            raise ImportError("Install torch to use DQN")
        self.cfg, self.device, self.updates = cfg, torch.device(device), 0
        self.online, self.target = QNetwork().to(self.device), QNetwork().to(self.device)
        self.target.load_state_dict(self.online.state_dict())
        self.optimizer = torch.optim.Adam(self.online.parameters(), lr=cfg.lr)
        self.buffer = ReplayBuffer(cfg.replay_size)

    def select_action(self, state: RawState) -> int:
        x = torch.tensor(encode_state(state), device=self.device).unsqueeze(0)
        with torch.no_grad():
            return int(self.online(x).argmax(1).item())

    def train_step(self) -> float:
        if len(self.buffer) < self.cfg.batch_size:
            return 0.0
        s, a, r, ns, done = self.buffer.sample(self.cfg.batch_size)
        s = torch.tensor(np.stack(s), device=self.device)
        ns = torch.tensor(np.stack(ns), device=self.device)
        a = torch.tensor(a, dtype=torch.long, device=self.device)
        r = torch.tensor(r, dtype=torch.float32, device=self.device)
        done = torch.tensor(done, dtype=torch.float32, device=self.device)
        prediction = self.online(s).gather(1, a[:, None]).squeeze(1)
        with torch.no_grad():
            target = r + self.cfg.gamma * (1.0 - done) * self.target(ns).max(1).values
        loss = nn.functional.smooth_l1_loss(prediction, target)
        self.optimizer.zero_grad(); loss.backward(); self.optimizer.step()
        self.updates += 1
        if self.updates % self.cfg.target_update == 0:
            self.target.load_state_dict(self.online.state_dict())
        return float(loss.item())
