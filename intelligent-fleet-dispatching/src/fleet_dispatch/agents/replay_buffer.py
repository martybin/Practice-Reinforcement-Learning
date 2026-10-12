"""Uniform experience replay with preallocated NumPy storage.

Storage is one contiguous array per field, written in place as a ring buffer. This avoids per-transition
Python objects and keeps sampling to a handful of fancy-index operations. Each transition stores the
action mask at both s and s', because the masked Double DQN target needs the valid next actions.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np


class Batch(NamedTuple):
    obs: np.ndarray           # (B, obs_dim) float32
    actions: np.ndarray       # (B,) int64
    rewards: np.ndarray       # (B,) float32
    next_obs: np.ndarray      # (B, obs_dim) float32
    terminated: np.ndarray    # (B,) bool  genuine terminal: do not bootstrap
    truncated: np.ndarray     # (B,) bool  time limit: bootstrap normally
    masks: np.ndarray         # (B, n_actions) bool
    next_masks: np.ndarray    # (B, n_actions) bool


class ReplayBuffer:
    def __init__(self, capacity: int, obs_dim: int, n_actions: int, rng: np.random.Generator) -> None:
        if capacity < 1:
            raise ValueError("replay capacity must be >= 1")
        self.capacity = capacity
        self.obs_dim = obs_dim
        self.n_actions = n_actions
        self.rng = rng
        self.obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.next_obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.terminated = np.zeros(capacity, dtype=np.bool_)
        self.truncated = np.zeros(capacity, dtype=np.bool_)
        self.masks = np.zeros((capacity, n_actions), dtype=np.bool_)
        self.next_masks = np.zeros((capacity, n_actions), dtype=np.bool_)
        self._ptr = 0
        self._size = 0

    def __len__(self) -> int:
        return self._size

    def add(
        self,
        obs: np.ndarray,
        action: int,
        reward: float,
        next_obs: np.ndarray,
        terminated: bool,
        truncated: bool,
        mask: np.ndarray,
        next_mask: np.ndarray,
    ) -> None:
        if obs.shape != (self.obs_dim,) or next_obs.shape != (self.obs_dim,):
            raise ValueError(f"observation must have shape ({self.obs_dim},)")
        if mask.shape != (self.n_actions,) or next_mask.shape != (self.n_actions,):
            raise ValueError(f"masks must have shape ({self.n_actions},)")
        i = self._ptr
        self.obs[i] = obs
        self.next_obs[i] = next_obs
        self.actions[i] = action
        self.rewards[i] = reward
        self.terminated[i] = terminated
        self.truncated[i] = truncated
        self.masks[i] = mask
        self.next_masks[i] = next_mask
        self._ptr = (self._ptr + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def sample(self, batch_size: int) -> Batch:
        """Uniform sampling with replacement, as in the original DQN. Requires at least one transition."""
        if self._size == 0:
            raise ValueError("cannot sample from an empty replay buffer")
        idx = self.rng.integers(0, self._size, size=batch_size)
        return Batch(
            obs=self.obs[idx],
            actions=self.actions[idx],
            rewards=self.rewards[idx],
            next_obs=self.next_obs[idx],
            terminated=self.terminated[idx],
            truncated=self.truncated[idx],
            masks=self.masks[idx],
            next_masks=self.next_masks[idx],
        )

    def state_dict(self) -> dict[str, np.ndarray | int]:
        """Only the filled part of the buffer, so checkpoints stay small."""
        n = self._size
        return {
            "obs": self.obs[:n].copy(),
            "next_obs": self.next_obs[:n].copy(),
            "actions": self.actions[:n].copy(),
            "rewards": self.rewards[:n].copy(),
            "terminated": self.terminated[:n].copy(),
            "truncated": self.truncated[:n].copy(),
            "masks": self.masks[:n].copy(),
            "next_masks": self.next_masks[:n].copy(),
            "ptr": self._ptr,
            "size": self._size,
        }

    def load_state_dict(self, state: dict[str, np.ndarray | int]) -> None:
        n = int(state["size"])
        if n > self.capacity:
            raise ValueError(f"saved buffer holds {n} transitions but capacity is {self.capacity}")
        for name in ("obs", "next_obs", "actions", "rewards", "terminated", "truncated", "masks", "next_masks"):
            getattr(self, name)[:n] = state[name]
        self._ptr = int(state["ptr"])
        self._size = n
