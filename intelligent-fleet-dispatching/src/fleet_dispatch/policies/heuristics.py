"""Baseline dispatch policies.

Every baseline receives the same DecisionView the learned agent is built from, and every baseline
chooses only among actions the mask marks valid. They therefore face exactly the same operational
constraints as DQN, and they read nothing the agent cannot observe.

Ties are broken by FIFO truck rank (the longest-waiting truck goes first), so the rules are deterministic
given the state.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np

from fleet_dispatch.environment.mining_env import DecisionView


class NoValidActionError(RuntimeError):
    """Raised if a policy is asked to act when the mask has no valid action."""


class Policy(Protocol):
    name: str

    def act(self, view: DecisionView) -> int: ...


def _candidates(view: DecisionView) -> np.ndarray:
    candidates = np.flatnonzero(view.action_mask)
    if candidates.size == 0:
        raise NoValidActionError(f"no valid dispatch at t={view.time:.2f}")
    return candidates


def _truck_of(actions: np.ndarray, n_shovels: int) -> np.ndarray:
    return actions // n_shovels


class RandomValidPolicy:
    """Uniform choice over valid truck-shovel pairs. The reference floor for any learned policy."""

    name = "random_valid"

    def __init__(self, seed: int) -> None:
        self.rng = np.random.default_rng(seed)

    def act(self, view: DecisionView) -> int:
        candidates = _candidates(view)
        return int(self.rng.choice(candidates))


class NearestEligibleShovelPolicy:
    """Send the truck to the operational shovel with the smallest nominal travel time.

    Among all valid pairs, this picks the pair with the smallest travel time. It ignores queues.
    """

    name = "nearest_eligible"

    def act(self, view: DecisionView) -> int:
        candidates = _candidates(view)
        n_shovels = view.travel_minutes.shape[1]
        trucks = _truck_of(candidates, n_shovels)
        travel = view.travel_minutes.reshape(-1)[candidates]
        rank = view.truck_fifo_rank[trucks]
        order = np.lexsort((rank, travel))  # primary: travel time, secondary: FIFO rank
        return int(candidates[order[0]])


class ShortestQueuePolicy:
    """Earliest expected availability: minimise the estimated time at which loading would begin.

    The estimate is travel time plus the shovel's committed backlog (queued, loading, and en-route
    loads, plus any repair in progress). It accounts for both travel time and congestion.
    """

    name = "shortest_queue"

    def act(self, view: DecisionView) -> int:
        candidates = _candidates(view)
        n_shovels = view.est_start_minutes.shape[1]
        trucks = _truck_of(candidates, n_shovels)
        est_start = view.est_start_minutes.reshape(-1)[candidates]
        rank = view.truck_fifo_rank[trucks]
        order = np.lexsort((rank, est_start))
        return int(candidates[order[0]])
