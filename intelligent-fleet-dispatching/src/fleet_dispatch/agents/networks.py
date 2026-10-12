"""Q-network: a configurable MLP that maps an observation to one Q-value per truck-shovel pair."""

from __future__ import annotations

import math
from typing import Sequence

import torch
from torch import nn


class QNetwork(nn.Module):
    """ReLU MLP. Hidden layers use orthogonal init with gain sqrt(2), which suits ReLU and keeps early
    activations well scaled. The output layer uses gain 1 and zero bias, so initial Q-values are small."""

    def __init__(self, obs_dim: int, n_actions: int, hidden_sizes: Sequence[int] = (256, 256)) -> None:
        super().__init__()
        if obs_dim < 1 or n_actions < 1:
            raise ValueError("obs_dim and n_actions must be positive")
        layers: list[nn.Module] = []
        width = obs_dim
        for size in hidden_sizes:
            layers += [nn.Linear(width, size), nn.ReLU()]
            width = size
        layers.append(nn.Linear(width, n_actions))
        self.net = nn.Sequential(*layers)
        self.obs_dim = obs_dim
        self.n_actions = n_actions
        self._init_weights()

    def _init_weights(self) -> None:
        linear_layers = [m for m in self.net if isinstance(m, nn.Linear)]
        for index, layer in enumerate(linear_layers):
            is_output = index == len(linear_layers) - 1
            gain = 1.0 if is_output else math.sqrt(2.0)
            nn.init.orthogonal_(layer.weight, gain=gain)
            nn.init.zeros_(layer.bias)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.net(observations)
