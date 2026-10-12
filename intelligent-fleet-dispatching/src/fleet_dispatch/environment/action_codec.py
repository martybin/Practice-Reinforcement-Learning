"""Action encoding, decoding and masking for truck-to-shovel dispatch.

Action a in [0, T*S) means "dispatch truck a // S to shovel a % S". With 20 trucks and 4 shovels there
are 80 actions. An action is VALID only if the truck is IDLE and the shovel is OPERATIONAL. The mask
marks exactly the valid actions. There is no WAIT or no-op action: an idle truck is always dispatched,
and the only real choice is which operational shovel it goes to. The trade-offs are discussed in
docs/environment.md.
"""

from __future__ import annotations

import numpy as np

from fleet_dispatch.environment.entities import ShovelStatus, TruckStatus
from fleet_dispatch.environment.simulator import FleetSimulator


def encode_action(truck_index: int, shovel_index: int, n_shovels: int) -> int:
    return truck_index * n_shovels + shovel_index


def decode_action(action: int, n_shovels: int) -> tuple[int, int]:
    truck_index, shovel_index = divmod(int(action), n_shovels)
    return truck_index, shovel_index


def build_action_mask(truck_idle: np.ndarray, shovel_operational: np.ndarray) -> np.ndarray:
    """(T*S,) boolean mask: True where the truck is idle and the shovel is operational."""
    return (np.asarray(truck_idle, dtype=bool)[:, None] & np.asarray(shovel_operational, dtype=bool)[None, :]).reshape(-1)


def current_action_mask(sim: FleetSimulator) -> np.ndarray:
    truck_idle = np.array([t.status == TruckStatus.IDLE for t in sim.trucks], dtype=bool)
    shovel_ok = np.array([s.status == ShovelStatus.OPERATIONAL for s in sim.shovels], dtype=bool)
    return build_action_mask(truck_idle, shovel_ok)
