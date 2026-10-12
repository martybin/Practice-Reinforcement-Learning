"""Deep Q-Network implemented from scratch on PyTorch.

Components: online and target Q-networks, uniform experience replay, epsilon-greedy
exploration, Bellman targets (vanilla or Double-DQN), Huber loss, gradient clipping
and hard/soft target-network updates.

Bellman target for a sampled transition ``(s, a, r, s', d)`` where ``d = 1`` only for a
*terminated* transition (truncation by a time limit is not terminal)::

    y = r + gamma * (1 - d) * Q_target(s', a*)          a* = argmax_a Q_online(s', a)   (Double DQN)
    y = r + gamma * (1 - d) * max_a Q_target(s', a)                                      (vanilla DQN)
    loss = Huber(Q_online(s, a), y)
"""

from __future__ import annotations

import copy
import dataclasses
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from ..utils.checkpoints import load_checkpoint, save_checkpoint
from .networks import QNetwork
from .replay_buffer import ReplayBuffer

CHECKPOINT_FORMAT = 1


@dataclass(frozen=True)
class DQNConfig:
    hidden_dims: tuple = (128, 128)
    activation: str = "relu"
    learning_rate: float = 5e-4
    weight_decay: float = 0.0
    gamma: float = 0.99
    batch_size: int = 128
    replay_capacity: int = 200_000
    learning_starts: int = 2_000          # environment steps collected before learning begins
    train_freq: int = 1                   # environment steps between update rounds
    gradient_steps: int = 1               # gradient steps per update round
    target_update_interval: int = 1_000   # gradient steps between hard target copies (used when tau >= 1)
    tau: float = 1.0                      # < 1 enables soft (Polyak) updates after every gradient step
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_steps: int = 50_000     # linear decay horizon in environment steps
    grad_clip_norm: float = 10.0
    huber_delta: float = 1.0
    double_dqn: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "hidden_dims", tuple(int(h) for h in self.hidden_dims))
        if not 0.0 <= self.gamma <= 1.0:
            raise ValueError("gamma must be in [0, 1]")
        if not 0.0 < self.tau <= 1.0:
            raise ValueError("tau must be in (0, 1]")
        for name in ("batch_size", "replay_capacity", "train_freq", "gradient_steps", "target_update_interval"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be >= 1")
        if not (0.0 <= self.epsilon_end <= self.epsilon_start <= 1.0):
            raise ValueError("require 0 <= epsilon_end <= epsilon_start <= 1")

    def to_dict(self) -> dict:
        d = dataclasses.asdict(self)
        d["hidden_dims"] = list(self.hidden_dims)
        return d


def resolve_device(device: str | torch.device = "auto") -> torch.device:
    """``"auto"`` -> CUDA when available, else CPU."""
    if isinstance(device, torch.device):
        return device
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dev = torch.device(device)
    if dev.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested but torch.cuda.is_available() is False. Use --device cpu or auto.")
    return dev


class DQNAgent:
    def __init__(self, obs_dim: int, n_actions: int, config: DQNConfig | None = None,
                 device: str | torch.device = "auto", seed: int = 0):
        self.cfg = config or DQNConfig()
        self.obs_dim, self.n_actions = int(obs_dim), int(n_actions)
        self.device = resolve_device(device)
        self.rng = np.random.default_rng(seed)

        self.q_net = QNetwork(obs_dim, n_actions, self.cfg.hidden_dims, self.cfg.activation).to(self.device)
        self.target_net = copy.deepcopy(self.q_net).requires_grad_(False)
        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=self.cfg.learning_rate,
                                          weight_decay=self.cfg.weight_decay)
        self.buffer = ReplayBuffer(self.cfg.replay_capacity, obs_dim, seed=seed + 1)

        self.steps = 0      # environment steps seen (drives epsilon and update scheduling)
        self.updates = 0    # gradient steps performed
        self._eps_start_step = 0
        self._eps_params = (self.cfg.epsilon_start, self.cfg.epsilon_end, self.cfg.epsilon_decay_steps)

    # ---------------------------------------------------------------- exploration
    @property
    def epsilon(self) -> float:
        start, end, horizon = self._eps_params
        frac = min(1.0, (self.steps - self._eps_start_step) / max(1, horizon))
        return start + frac * (end - start)

    def restart_exploration(self, start: float, end: float | None = None, decay_steps: int | None = None) -> None:
        """Restart the epsilon schedule now (used when the task changes between curriculum stages)."""
        end = self.cfg.epsilon_end if end is None else end
        decay_steps = self.cfg.epsilon_decay_steps if decay_steps is None else decay_steps
        self._eps_params = (start, min(end, start), decay_steps)
        self._eps_start_step = self.steps

    def q_values(self, obs) -> np.ndarray:
        with torch.no_grad():
            t = torch.as_tensor(np.asarray(obs, dtype=np.float32), device=self.device).unsqueeze(0)
            return self.q_net(t)[0].cpu().numpy()

    def act(self, obs, greedy: bool = False) -> int:
        """Epsilon-greedy action (``greedy=True`` disables exploration, e.g. for evaluation)."""
        if not greedy and self.rng.random() < self.epsilon:
            return int(self.rng.integers(self.n_actions))
        return int(np.argmax(self.q_values(obs)))

    # ---------------------------------------------------------------- learning
    def remember(self, obs, action: int, reward: float, next_obs, terminated: bool) -> None:
        self.buffer.add(obs, action, reward, next_obs, terminated)
        self.steps += 1

    def should_update(self) -> bool:
        return (len(self.buffer) >= max(self.cfg.batch_size, self.cfg.learning_starts)
                and self.steps % self.cfg.train_freq == 0)

    def td_targets(self, rewards: torch.Tensor, next_obs: torch.Tensor, dones: torch.Tensor) -> torch.Tensor:
        """Bellman targets ``y``; terminal transitions bootstrap to zero."""
        with torch.no_grad():
            if self.cfg.double_dqn:
                best = self.q_net(next_obs).argmax(dim=1, keepdim=True)
                next_q = self.target_net(next_obs).gather(1, best).squeeze(1)
            else:
                next_q = self.target_net(next_obs).max(dim=1).values
            return rewards + self.cfg.gamma * (1.0 - dones) * next_q

    def update(self) -> dict[str, float] | None:
        """One gradient step on a replay batch; returns training statistics (or ``None`` if not ready)."""
        if len(self.buffer) < self.cfg.batch_size:
            return None
        batch = self.buffer.sample(self.cfg.batch_size, self.device)
        q_all = self.q_net(batch["obs"])
        q_sa = q_all.gather(1, batch["actions"].unsqueeze(1)).squeeze(1)
        targets = self.td_targets(batch["rewards"], batch["next_obs"], batch["dones"])
        loss = F.huber_loss(q_sa, targets, delta=self.cfg.huber_delta)

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), self.cfg.grad_clip_norm)
        self.optimizer.step()
        self.updates += 1
        self._update_target()

        with torch.no_grad():
            return {
                "loss": float(loss.item()),
                "q_mean": float(q_all.mean().item()),
                "q_max": float(q_all.max().item()),
                "q_min": float(q_all.min().item()),
                "q_taken_mean": float(q_sa.mean().item()),
                "td_error_abs": float((q_sa - targets).abs().mean().item()),
                "grad_norm": float(grad_norm),
            }

    def _update_target(self) -> None:
        if self.cfg.tau < 1.0:
            with torch.no_grad():
                for tp, p in zip(self.target_net.parameters(), self.q_net.parameters()):
                    tp.mul_(1.0 - self.cfg.tau).add_(p, alpha=self.cfg.tau)
        elif self.updates % self.cfg.target_update_interval == 0:
            self.sync_target()

    def sync_target(self) -> None:
        self.target_net.load_state_dict(self.q_net.state_dict())

    # ---------------------------------------------------------------- persistence
    def state_payload(self, metadata: dict | None = None) -> dict:
        return {
            "format_version": CHECKPOINT_FORMAT,
            "q_net": self.q_net.state_dict(),
            "target_net": self.target_net.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "config": self.cfg.to_dict(),
            "obs_dim": self.obs_dim,
            "n_actions": self.n_actions,
            "steps": self.steps,
            "updates": self.updates,
            "metadata": metadata or {},
        }

    def save(self, path: str | Path, metadata: dict | None = None, overwrite: bool = False) -> Path:
        return save_checkpoint(path, self.state_payload(metadata), overwrite=overwrite)

    def load_weights(self, payload: dict, load_optimizer: bool = True) -> None:
        self.q_net.load_state_dict(payload["q_net"])
        self.target_net.load_state_dict(payload["target_net"])
        if load_optimizer and "optimizer" in payload:
            self.optimizer.load_state_dict(payload["optimizer"])
        self.steps = int(payload.get("steps", 0))
        self.updates = int(payload.get("updates", 0))

    @classmethod
    def from_checkpoint(cls, path: str | Path, device: str | torch.device = "auto", seed: int = 0,
                        load_optimizer: bool = False) -> tuple["DQNAgent", dict]:
        """Rebuild an agent from a checkpoint; returns ``(agent, metadata)``."""
        dev = resolve_device(device)
        payload = load_checkpoint(path, map_location=dev)
        if payload.get("format_version") != CHECKPOINT_FORMAT:
            raise ValueError(f"Unsupported checkpoint format {payload.get('format_version')!r} in {path}")
        cfg = DQNConfig(**payload["config"])
        # a tiny replay buffer: inference-only agents never need the training memory
        cfg = dataclasses.replace(cfg, replay_capacity=max(cfg.batch_size, 1024))
        agent = cls(payload["obs_dim"], payload["n_actions"], cfg, device=dev, seed=seed)
        agent.load_weights(payload, load_optimizer=load_optimizer)
        return agent, payload.get("metadata", {})
