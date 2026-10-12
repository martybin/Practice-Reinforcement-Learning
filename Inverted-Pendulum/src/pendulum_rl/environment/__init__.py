"""Gymnasium environment and reward shaping."""

import gymnasium as gym

from .pendulum_env import (
    PHASES, VIOLATIONS, EnvConfig, PendulumEnv, classify_phase, make_env, make_observation, observation_bounds,
)
from .rewards import RewardConfig, compute_reward

ENV_ID = "PendulumSwingUp-v0"
if ENV_ID not in gym.registry:
    gym.register(id=ENV_ID, entry_point="pendulum_rl.environment.pendulum_env:PendulumEnv")

__all__ = ["ENV_ID", "PHASES", "VIOLATIONS", "EnvConfig", "PendulumEnv", "RewardConfig",
           "classify_phase", "compute_reward", "make_env", "make_observation", "observation_bounds"]
