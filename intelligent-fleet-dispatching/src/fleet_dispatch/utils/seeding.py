"""Seeding helpers. Scenario generation, episode randomness, and network init all derive from explicit seeds."""

from __future__ import annotations

import random

import numpy as np
import torch


def seed_everything(seed: int) -> None:
    """Seed the global generators. Environments and agents also keep their own seeded generators."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def training_episode_seed(seed_base: int, episode_index: int) -> int:
    """Reset seed for training episode k. Kept separate from the evaluation seeds."""
    return seed_base + episode_index


def evaluation_seeds(seed_start: int, n_episodes: int) -> list[int]:
    return [seed_start + i for i in range(n_episodes)]
