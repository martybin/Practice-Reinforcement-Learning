"""Synthetic mine scenario: fleet composition and nominal travel and loading times.

A scenario is fixed for a whole study (its seed is `scenario.seed`). Every truck gets its own speed,
reliability and capacity drawn from NumPy's generator with that seed, so the fleet is heterogeneous
but reproducible. Episode randomness (breakdowns, jitter) comes from the reset seed instead.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from fleet_dispatch.config import ScenarioConfig


@dataclass(frozen=True, eq=False)
class Scenario:
    seed: int
    horizon_minutes: float
    unload_time_minutes: float
    depot_to_shovel_minutes: np.ndarray   # (S,) nominal empty-trip time for speed factor 1.0
    shovel_to_depot_minutes: np.ndarray   # (S,) nominal loaded-trip time for speed factor 1.0
    shovel_loading_rate_tpm: np.ndarray   # (S,)
    truck_capacity_t: np.ndarray          # (T,)
    truck_speed_factor: np.ndarray        # (T,)
    truck_reliability_factor: np.ndarray  # (T,)

    @property
    def n_trucks(self) -> int:
        return int(self.truck_capacity_t.shape[0])

    @property
    def n_shovels(self) -> int:
        return int(self.shovel_loading_rate_tpm.shape[0])

    def travel_empty_minutes(self) -> np.ndarray:
        """(T, S) nominal depot -> shovel travel time for every truck."""
        return self.truck_speed_factor[:, None] * self.depot_to_shovel_minutes[None, :]

    def travel_loaded_minutes(self) -> np.ndarray:
        """(T, S) nominal shovel -> depot travel time for every truck."""
        return self.truck_speed_factor[:, None] * self.shovel_to_depot_minutes[None, :]

    def nominal_load_minutes(self) -> np.ndarray:
        """(T, S) nominal time for shovel j to load truck i to capacity."""
        return self.truck_capacity_t[:, None] / self.shovel_loading_rate_tpm[None, :]


def build_scenario(cfg: ScenarioConfig) -> Scenario:
    """Generate the fleet from the scenario seed. The draw order is fixed, so results are reproducible."""
    rng = np.random.default_rng(cfg.seed)
    n_trucks = cfg.n_trucks
    speed = rng.uniform(*cfg.truck_speed_factor_range, size=n_trucks)
    reliability = rng.uniform(*cfg.truck_reliability_factor_range, size=n_trucks)
    capacity = rng.choice(np.asarray(cfg.truck_capacity_choices_tonnes, dtype=float), size=n_trucks)
    return Scenario(
        seed=cfg.seed,
        horizon_minutes=float(cfg.horizon_minutes),
        unload_time_minutes=float(cfg.unload_time_minutes),
        depot_to_shovel_minutes=np.asarray(cfg.depot_to_shovel_minutes, dtype=float),
        shovel_to_depot_minutes=np.asarray(cfg.shovel_to_depot_minutes, dtype=float),
        shovel_loading_rate_tpm=np.asarray(cfg.shovel_loading_rate_tpm, dtype=float),
        truck_capacity_t=capacity.astype(float),
        truck_speed_factor=speed.astype(float),
        truck_reliability_factor=reliability.astype(float),
    )
