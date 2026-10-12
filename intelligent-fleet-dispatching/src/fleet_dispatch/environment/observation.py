"""Observation features: what the agent may see at a decision epoch, and how each feature is normalised.

Layout of the flat float32 vector (all values clipped to [0, 1]):

  per truck  (T blocks, 7 + (S+1) + 5 features each)
      status one-hot (7) | destination one-hot (S+1: depot, shovel 0..S-1) | loaded | available |
      capacity / max capacity | remaining expected time / time_scale | time in current status / time_scale
  per shovel (S blocks, 8 features each)
      operational | loading rate / max rate | busy | remaining expected load / time_scale |
      queue length / T | trucks en route / T | backlog estimate / time_scale | busy fraction so far
  per pair   (T*S blocks, 2 features each; pair index = truck * S + shovel, the same as the action index)
      nominal travel / time_scale | estimated start-of-loading time / time_scale
  global     (3 features)
      remaining shift fraction | idle-truck fraction | operational-shovel fraction

Information discipline: every value is computed from the current state. Remaining times use NOMINAL
(expected) durations, so the realised noise (travel and loading jitter, the length of a breakdown
repair, and when breakdowns occur) is never revealed. The backlog estimate for a shovel counts only the
loads that are already queued, loading, or en route to it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from fleet_dispatch.environment.entities import (
    N_TRUCK_STATUSES,
    Truck,
    TruckStatus,
    ShovelStatus,
    is_heading_to_shovel,
)
from fleet_dispatch.environment.simulator import FleetSimulator

SHOVEL_FEATURE_DIM = 8
PAIR_FEATURE_DIM = 2
GLOBAL_FEATURE_DIM = 3
_TRUCK_SCALAR_FEATURES = 5  # loaded, available, capacity, remaining time, time in status


def truck_feature_dim(n_shovels: int) -> int:
    return N_TRUCK_STATUSES + (n_shovels + 1) + _TRUCK_SCALAR_FEATURES


def observation_dim(n_trucks: int, n_shovels: int) -> int:
    return (
        n_trucks * truck_feature_dim(n_shovels)
        + n_shovels * SHOVEL_FEATURE_DIM
        + n_trucks * n_shovels * PAIR_FEATURE_DIM
        + GLOBAL_FEATURE_DIM
    )


def feature_names(n_trucks: int, n_shovels: int) -> list[str]:
    """Human-readable name for every entry of the observation, in order."""
    names: list[str] = []
    for k in range(n_trucks):
        names += [f"truck{k}.status_{s.name.lower()}" for s in TruckStatus]
        names += [f"truck{k}.destination_depot" if n == 0 else f"truck{k}.destination_shovel{n - 1}"
                  for n in range(n_shovels + 1)]
        names += [f"truck{k}.loaded", f"truck{k}.available", f"truck{k}.capacity_norm",
                  f"truck{k}.remaining_expected_norm", f"truck{k}.time_in_status_norm"]
    for j in range(n_shovels):
        names += [f"shovel{j}.operational", f"shovel{j}.rate_norm", f"shovel{j}.busy",
                  f"shovel{j}.remaining_load_norm", f"shovel{j}.queue_norm", f"shovel{j}.en_route_norm",
                  f"shovel{j}.backlog_norm", f"shovel{j}.utilization_so_far"]
    for k in range(n_trucks):
        for j in range(n_shovels):
            names += [f"pair{k}_{j}.travel_norm", f"pair{k}_{j}.est_start_norm"]
    names += ["global.remaining_shift_fraction", "global.idle_truck_fraction", "global.operational_shovel_fraction"]
    return names


@dataclass(frozen=True)
class PairEstimates:
    """Decision-time estimates for every truck-shovel pair. Computed from the same state as the observation."""

    travel_minutes: np.ndarray        # (T, S) nominal depot -> shovel time for each truck
    est_start_minutes: np.ndarray     # (T, S) travel + shovel backlog: when loading would begin
    shovel_backlog_minutes: np.ndarray  # (S,) expected work already committed to each shovel


def shovel_backlog(sim: FleetSimulator) -> np.ndarray:
    """Expected minutes of work already committed to each shovel (repair, loading, queue, en route)."""
    now = sim.time
    load_minutes = sim.scenario.nominal_load_minutes()
    backlog = np.zeros(sim.scenario.n_shovels)
    for shovel in sim.shovels:
        total = 0.0
        if shovel.status == ShovelStatus.MAINTENANCE:
            total += shovel.repair.remaining_expected(now)
        if shovel.loading_truck != -1:
            total += shovel.load.remaining_expected(now)
        for k in shovel.queue:
            total += float(load_minutes[k, shovel.index])
        backlog[shovel.index] = total
    for truck in sim.trucks:
        if is_heading_to_shovel(truck):
            backlog[truck.target_shovel] += float(load_minutes[truck.index, truck.target_shovel])
    return backlog


def pair_estimates(sim: FleetSimulator) -> PairEstimates:
    travel = sim.scenario.travel_empty_minutes()
    backlog = shovel_backlog(sim)
    return PairEstimates(travel, travel + backlog[None, :], backlog)


def truck_remaining_expected(sim: FleetSimulator, truck: Truck, backlog: np.ndarray) -> float:
    """Expected minutes until the truck's current activity finishes, from nominal durations only."""
    now = sim.time
    if truck.status in (TruckStatus.TRAVEL_EMPTY, TruckStatus.TRAVEL_LOADED, TruckStatus.UNLOADING):
        return truck.phase.remaining_expected(now)
    if truck.status == TruckStatus.MAINTENANCE:
        return truck.repair.remaining_expected(now)
    if truck.status == TruckStatus.LOADING:
        return sim.shovels[truck.target_shovel].load.remaining_expected(now)
    if truck.status == TruckStatus.QUEUED:
        return float(backlog[truck.target_shovel])
    return 0.0


def build_observation(sim: FleetSimulator, time_scale: float) -> np.ndarray:
    """Build the normalised observation for the current decision epoch."""
    sc = sim.scenario
    n_t, n_s = sc.n_trucks, sc.n_shovels
    now = sim.time
    horizon = sc.horizon_minutes
    max_capacity = float(sc.truck_capacity_t.max())
    max_rate = float(sc.shovel_loading_rate_tpm.max())
    pairs = pair_estimates(sim)
    backlog = pairs.shovel_backlog_minutes
    obs = np.zeros(observation_dim(n_t, n_s), dtype=np.float32)

    def norm_time(minutes: float) -> float:
        return min(max(minutes / time_scale, 0.0), 1.0)

    offset = 0
    t_dim = truck_feature_dim(n_s)
    en_route = np.zeros(n_s)
    for truck in sim.trucks:
        block = obs[offset:offset + t_dim]
        block[int(truck.status)] = 1.0
        block[N_TRUCK_STATUSES + truck.destination] = 1.0
        tail = N_TRUCK_STATUSES + n_s + 1
        block[tail + 0] = float(truck.loaded)
        block[tail + 1] = float(truck.status != TruckStatus.MAINTENANCE)
        block[tail + 2] = truck.capacity_t / max_capacity
        block[tail + 3] = norm_time(truck_remaining_expected(sim, truck, backlog))
        block[tail + 4] = norm_time(now - truck.status_since)
        if is_heading_to_shovel(truck):
            en_route[truck.target_shovel] += 1
        offset += t_dim

    for shovel in sim.shovels:
        j = shovel.index
        block = obs[offset:offset + SHOVEL_FEATURE_DIM]
        block[0] = float(shovel.operational)
        block[1] = shovel.loading_rate_tpm / max_rate
        block[2] = float(shovel.loading_truck != -1)
        block[3] = norm_time(shovel.load.remaining_expected(now)) if shovel.loading_truck != -1 else 0.0
        block[4] = min(len(shovel.queue) / n_t, 1.0)
        block[5] = min(en_route[j] / n_t, 1.0)
        block[6] = norm_time(backlog[j])
        block[7] = min(sim.shovel_busy_min[j] / now, 1.0) if now > 0 else 0.0
        offset += SHOVEL_FEATURE_DIM

    pair_block = obs[offset:offset + n_t * n_s * PAIR_FEATURE_DIM].reshape(n_t, n_s, PAIR_FEATURE_DIM)
    pair_block[:, :, 0] = np.clip(pairs.travel_minutes / time_scale, 0.0, 1.0)
    pair_block[:, :, 1] = np.clip(pairs.est_start_minutes / time_scale, 0.0, 1.0)
    offset += n_t * n_s * PAIR_FEATURE_DIM

    n_idle = sum(1 for t in sim.trucks if t.status == TruckStatus.IDLE)
    n_operational = len(sim.operational_shovels())
    obs[offset] = min(max((horizon - now) / horizon, 0.0), 1.0)
    obs[offset + 1] = n_idle / n_t
    obs[offset + 2] = n_operational / n_s
    return np.clip(obs, 0.0, 1.0, out=obs)
