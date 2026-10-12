"""Reward engineering: turns the simulator's raw counters into a weighted, interpretable reward.

Each component is a raw quantity times a configured coefficient. A disabled component contributes zero,
which makes ablations a config change rather than a code change. The per-step reward is the sum of the
component contributions over the interval between two consecutive decision epochs.

Only realised outcomes of that interval are used. No future information enters the reward, and the
observation never contains anything the reward depends on that the agent could not know.
"""

from __future__ import annotations

from typing import Mapping

from fleet_dispatch.config import RewardConfig

REWARD_COMPONENTS = (
    "payload",
    "truck_wait",
    "unnecessary_travel",
    "shovel_starvation",
    "truck_breakdown",
    "invalid_action",
)

PAYLOAD_UNIT_TONNES = 100.0


def raw_quantities(delta: Mapping[str, float]) -> dict[str, float]:
    """Map raw counter deltas onto the reward components' units."""
    return {
        "payload": delta.get("payload_t", 0.0) / PAYLOAD_UNIT_TONNES,
        "truck_wait": delta.get("truck_wait_min", 0.0),
        "unnecessary_travel": delta.get("unnecessary_travel_min", 0.0),
        "shovel_starvation": delta.get("shovel_starvation_min", 0.0),
        "truck_breakdown": delta.get("truck_breakdowns", 0.0),
        "invalid_action": delta.get("invalid_actions", 0.0),
    }


def compute_reward(delta: Mapping[str, float], cfg: RewardConfig) -> tuple[float, dict[str, float]]:
    """Return (total reward, weighted contribution of each component)."""
    raw = raw_quantities(delta)
    contributions: dict[str, float] = {}
    for name in REWARD_COMPONENTS:
        component = getattr(cfg, name)
        contributions[name] = component.coefficient * raw[name] if component.enabled else 0.0
    return float(sum(contributions.values())), contributions
