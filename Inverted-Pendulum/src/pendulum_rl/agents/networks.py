"""Q-network: a small configurable multilayer perceptron."""

from __future__ import annotations

from typing import Sequence

import torch
from torch import nn

_ACTIVATIONS = {"relu": nn.ReLU, "tanh": nn.Tanh, "elu": nn.ELU, "silu": nn.SiLU}


class QNetwork(nn.Module):
    """Maps an observation to one Q-value per discrete action."""

    def __init__(self, obs_dim: int, n_actions: int, hidden_dims: Sequence[int] = (128, 128), activation: str = "relu"):
        super().__init__()
        if activation not in _ACTIVATIONS:
            raise ValueError(f"Unknown activation {activation!r}; choose from {sorted(_ACTIVATIONS)}")
        layers: list[nn.Module] = []
        last = obs_dim
        for width in hidden_dims:
            layers += [nn.Linear(last, int(width)), _ACTIVATIONS[activation]()]
            last = int(width)
        layers.append(nn.Linear(last, n_actions))
        self.net = nn.Sequential(*layers)
        self.obs_dim, self.n_actions = obs_dim, n_actions

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)
