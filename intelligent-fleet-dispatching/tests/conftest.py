"""Shared fixtures. Tests import the package from src/ so they run without installing it."""

from __future__ import annotations

import copy
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fleet_dispatch.config import ProjectConfig, load_config  # noqa: E402
from fleet_dispatch.environment.mining_env import FleetDispatchEnv  # noqa: E402


@pytest.fixture(scope="session")
def base_config() -> ProjectConfig:
    return load_config()


@pytest.fixture
def config(base_config: ProjectConfig) -> ProjectConfig:
    """A fresh copy per test, so tests can mutate it safely."""
    return copy.deepcopy(base_config)


@pytest.fixture
def env(config: ProjectConfig) -> FleetDispatchEnv:
    return FleetDispatchEnv(config)


def run_random_valid_episode(env: FleetDispatchEnv, seed: int, rng_seed: int = 0) -> list[float]:
    """Play a whole shift choosing uniformly among valid actions. Returns the per-step rewards."""
    _, _ = env.reset(seed=seed)
    rng = np.random.default_rng(rng_seed)
    rewards: list[float] = []
    done = False
    while not done:
        action = int(rng.choice(np.flatnonzero(env.action_masks())))
        _, reward, terminated, truncated, _ = env.step(action)
        rewards.append(reward)
        done = terminated or truncated
    return rewards
