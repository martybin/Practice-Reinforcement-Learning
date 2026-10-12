"""Uniform experience replay with fixed-size circular storage."""

from __future__ import annotations

import numpy as np
import torch


class ReplayBuffer:
    """Stores ``(s, a, r, s', terminated)`` transitions.

    Only true terminations are stored as ``done``: a time-limit truncation is *not*
    terminal for the Bellman target (the state still has future value), so the
    caller must pass ``terminated`` and not ``terminated or truncated``.
    """

    def __init__(self, capacity: int, obs_dim: int, seed: int | None = None):
        if capacity < 1:
            raise ValueError("capacity must be >= 1")
        self.capacity = int(capacity)
        self.obs_dim = int(obs_dim)
        self.obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.next_obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)
        self._pos = 0
        self._size = 0
        self._rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return self._size

    def add(self, obs, action: int, reward: float, next_obs, terminated: bool) -> None:
        i = self._pos
        self.obs[i] = obs
        self.next_obs[i] = next_obs
        self.actions[i] = action
        self.rewards[i] = reward
        self.dones[i] = float(terminated)
        self._pos = (i + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def sample(self, batch_size: int, device: torch.device | str = "cpu") -> dict[str, torch.Tensor]:
        if self._size == 0:
            raise ValueError("Cannot sample from an empty replay buffer")
        idx = self._rng.integers(0, self._size, size=batch_size)
        return {
            "obs": torch.as_tensor(self.obs[idx], device=device),
            "actions": torch.as_tensor(self.actions[idx], device=device),
            "rewards": torch.as_tensor(self.rewards[idx], device=device),
            "next_obs": torch.as_tensor(self.next_obs[idx], device=device),
            "dones": torch.as_tensor(self.dones[idx], device=device),
        }

    def clear(self) -> None:
        self._pos = 0
        self._size = 0
