"""Masked action helpers, shared by exploration, greedy action choice, and TD-target computation.

The same mask has to drive all three, or the agent learns values for actions it can never take.
Invalid actions are pushed to a large negative finite value (not -inf), so a max or argmax never picks
them, and gathers never produce NaN. Rows with no valid action are handled explicitly by the caller.
"""

from __future__ import annotations

import numpy as np
import torch

MASKED_Q_VALUE = -1.0e9


def apply_action_mask(q_values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Set Q-values of invalid actions to MASKED_Q_VALUE. Works on (batch, actions) or (actions,) tensors."""
    return q_values.masked_fill(~mask, MASKED_Q_VALUE)


def masked_argmax(q_values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return apply_action_mask(q_values, mask).argmax(dim=-1)


def masked_max(q_values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return apply_action_mask(q_values, mask).max(dim=-1).values


def has_valid_action(mask: torch.Tensor) -> torch.Tensor:
    """(batch,) bool: True where at least one action is valid."""
    return mask.any(dim=-1)


def sample_valid_action(mask: np.ndarray, rng: np.random.Generator) -> int:
    """Uniform random choice among valid actions. Exploration never proposes an invalid action."""
    candidates = np.flatnonzero(mask)
    if candidates.size == 0:
        raise ValueError("cannot sample an action: the mask has no valid entries")
    return int(rng.choice(candidates))


def greedy_valid_action(q_values: np.ndarray, mask: np.ndarray) -> int:
    """Argmax of Q over valid actions only."""
    if not mask.any():
        raise ValueError("cannot choose a greedy action: the mask has no valid entries")
    return int(np.argmax(np.where(mask, q_values, -np.inf)))
