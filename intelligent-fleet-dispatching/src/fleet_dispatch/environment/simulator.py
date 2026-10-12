"""Event-driven simulator of one open-pit haul shift.

Design decision: a discrete-event engine (a heap of timestamped events) rather than a fixed-step clock.
Durations are continuous minutes, so queueing and breakdown timing are exact and there is no
discretisation error. The clock jumps from one decision epoch to the next. A decision epoch is an
instant at which at least one truck is IDLE and at least one shovel is operational.

Stale events are not removed from the heap. Each entity carries a token, and an event is applied only
if its token still matches. Any state change that reschedules an activity (a breakdown, a repair, a
new segment) bumps the token, which invalidates whatever was scheduled before.
"""

from __future__ import annotations

import heapq
import math
from collections import defaultdict
from enum import IntEnum
from typing import Callable

import numpy as np

from fleet_dispatch.config import SimulationConfig
from fleet_dispatch.environment.entities import (
    DEPOT_NODE,
    DRIVING_STATUSES,
    N_TRUCK_STATUSES,
    Shovel,
    ShovelStatus,
    Truck,
    TruckStatus,
    is_heading_to_shovel,
    shovel_node,
)
from fleet_dispatch.environment.scenario import Scenario


class InvalidActionError(RuntimeError):
    """Raised when a dispatch violates the operational constraints (truck not idle, shovel unavailable)."""


class EventKind(IntEnum):
    TRUCK_ARRIVE_SHOVEL = 0
    TRUCK_ARRIVE_DEPOT = 1
    TRUCK_UNLOAD_DONE = 2
    TRUCK_BREAKDOWN = 3
    TRUCK_REPAIRED = 4
    SHOVEL_LOAD_DONE = 5
    SHOVEL_BREAKDOWN = 6
    SHOVEL_REPAIRED = 7


# Which event completes each timed truck activity. Breakdowns only strike while driving.
_TRUCK_END_EVENT = {
    TruckStatus.TRAVEL_EMPTY: EventKind.TRUCK_ARRIVE_SHOVEL,
    TruckStatus.TRAVEL_LOADED: EventKind.TRUCK_ARRIVE_DEPOT,
    TruckStatus.UNLOADING: EventKind.TRUCK_UNLOAD_DONE,
}

# Counters the environment turns into per-step rewards by differencing between decisions.
RAW_COUNTERS = (
    "payload_t",
    "loads",
    "dispatches",
    "shovel_arrivals",
    "truck_wait_min",
    "unnecessary_travel_min",
    "shovel_starvation_min",
    "truck_breakdowns",
    "shovel_breakdowns",
    "invalid_actions",
)


class FleetSimulator:
    """Simulates one shift. Decisions are made through `dispatch`; `advance_to_decision` moves the clock."""

    def __init__(self, scenario: Scenario, sim_cfg: SimulationConfig) -> None:
        self.scenario = scenario
        self.cfg = sim_cfg
        self._travel_empty = scenario.travel_empty_minutes()
        self._travel_loaded = scenario.travel_loaded_minutes()
        self._load_minutes = scenario.nominal_load_minutes()
        self._handlers: dict[EventKind, Callable[[int], None]] = {
            EventKind.TRUCK_ARRIVE_SHOVEL: self._on_truck_arrive_shovel,
            EventKind.TRUCK_ARRIVE_DEPOT: self._on_truck_arrive_depot,
            EventKind.TRUCK_UNLOAD_DONE: self._on_truck_unload_done,
            EventKind.TRUCK_BREAKDOWN: self._on_truck_breakdown,
            EventKind.TRUCK_REPAIRED: self._on_truck_repaired,
            EventKind.SHOVEL_LOAD_DONE: self._on_shovel_load_done,
            EventKind.SHOVEL_BREAKDOWN: self._on_shovel_breakdown,
            EventKind.SHOVEL_REPAIRED: self._on_shovel_repaired,
        }
        self.rng = np.random.default_rng(0)
        self.trucks: list[Truck] = []
        self.shovels: list[Shovel] = []
        self.time = 0.0
        self.done = False
        self.totals: dict[str, float] = defaultdict(float)
        self.truck_state_min = np.zeros((scenario.n_trucks, N_TRUCK_STATUSES))
        self.shovel_busy_min = np.zeros(scenario.n_shovels)
        self.shovel_operational_min = np.zeros(scenario.n_shovels)
        self.dispatch_log: list[tuple[float, int, int, float]] = []  # (time, truck, shovel, regret_min)
        self.event_log: list[tuple[float, str, int, float]] = []     # (time, event, entity, duration)
        self._events: list[tuple[float, int, int, int, int]] = []    # (time, seq, kind, entity, token)
        self._seq = 0
        self._last_t = 0.0

    # ------------------------------------------------------------------ lifecycle
    def reset(self, rng: np.random.Generator) -> FleetSimulator:
        """Start a new shift. All trucks are idle at the depot and all shovels are operational."""
        self.rng = rng
        sc = self.scenario
        self.trucks = [
            Truck(
                index=k,
                capacity_t=float(sc.truck_capacity_t[k]),
                speed_factor=float(sc.truck_speed_factor[k]),
                reliability_factor=float(sc.truck_reliability_factor[k]),
            )
            for k in range(sc.n_trucks)
        ]
        self.shovels = [
            Shovel(index=j, loading_rate_tpm=float(sc.shovel_loading_rate_tpm[j])) for j in range(sc.n_shovels)
        ]
        self.time = 0.0
        self._last_t = 0.0
        self.done = False
        self.totals = defaultdict(float)
        self.truck_state_min.fill(0.0)
        self.shovel_busy_min.fill(0.0)
        self.shovel_operational_min.fill(0.0)
        self.dispatch_log.clear()
        self.event_log.clear()
        self._events.clear()
        self._seq = 0
        for shovel in self.shovels:
            self._schedule_shovel_exposure(shovel.index)
        return self

    def advance_to_decision(self) -> None:
        """Process events in time order until a decision is needed or the shift ends.

        A decision is needed when a truck is idle and some shovel is operational. Events that share the
        decision instant are processed first, so the agent sees a consistent snapshot of that instant.

        Once the clock reaches the horizon the shift is over. The state at that instant is still exposed
        (with its action mask), so the learner can bootstrap from it: the horizon is a truncation.
        """
        horizon = self.scenario.horizon_minutes
        while True:
            if self.time >= horizon:
                self._accrue_to(horizon)
                self.time = horizon
                self.done = True
                return
            next_time = self._events[0][0] if self._events else math.inf
            if self.has_decision() and next_time > self.time:
                return
            if next_time > horizon:
                self._accrue_to(horizon)
                self.time = horizon
                self.done = True
                return
            t, _, kind_value, entity, token = heapq.heappop(self._events)
            self._accrue_to(t)
            self.time = t
            kind = EventKind(kind_value)
            if self._is_current(kind, entity, token):
                self._handlers[kind](entity)

    # ------------------------------------------------------------------ decisions
    def has_decision(self) -> bool:
        return any(t.status == TruckStatus.IDLE for t in self.trucks) and any(s.operational for s in self.shovels)

    def idle_trucks(self) -> list[int]:
        """Idle trucks in FIFO order (the one that has been waiting longest first)."""
        idle = [t for t in self.trucks if t.status == TruckStatus.IDLE]
        idle.sort(key=lambda t: (t.status_since, t.index))
        return [t.index for t in idle]

    def operational_shovels(self) -> list[int]:
        return [s.index for s in self.shovels if s.operational]

    def dispatch(self, truck_index: int, shovel_index: int) -> float:
        """Send an idle truck to an operational shovel. Returns the unnecessary-travel regret in minutes."""
        truck = self.trucks[truck_index]
        shovel = self.shovels[shovel_index]
        if truck.status != TruckStatus.IDLE:
            raise InvalidActionError(f"truck {truck_index} is {truck.status.name}, not IDLE")
        if not shovel.operational:
            raise InvalidActionError(f"shovel {shovel_index} is in maintenance")
        travel = float(self._travel_empty[truck_index, shovel_index])
        nearest = float(self._travel_empty[truck_index, self.operational_shovels()].min())
        regret = travel - nearest
        truck.target_shovel = shovel_index
        self.totals["dispatches"] += 1
        self.totals["unnecessary_travel_min"] += regret
        self.dispatch_log.append((self.time, truck_index, shovel_index, regret))
        self._begin_truck_activity(
            truck_index, TruckStatus.TRAVEL_EMPTY, shovel_node(shovel_index), travel, self.cfg.travel_jitter_sigma
        )
        return regret

    # ------------------------------------------------------------------ metrics
    def summary(self) -> dict[str, float]:
        """Operational metrics for the episode so far (the shift, once it has ended)."""
        sc = self.scenario
        horizon = sc.horizon_minutes
        state = self.truck_state_min
        productive = state[:, [TruckStatus.TRAVEL_EMPTY, TruckStatus.TRAVEL_LOADED, TruckStatus.LOADING, TruckStatus.UNLOADING]].sum()
        travel = state[:, [TruckStatus.TRAVEL_EMPTY, TruckStatus.TRAVEL_LOADED]].sum()
        arrivals = self.totals["shovel_arrivals"]
        wait = self.totals["truck_wait_min"]
        operating = float(self.shovel_operational_min.sum())
        busy = float(self.shovel_busy_min.sum())
        return {
            "payload_t": float(self.totals["payload_t"]),
            "loads": float(self.totals["loads"]),
            "dispatches": float(self.totals["dispatches"]),
            "avg_wait_min": wait / arrivals if arrivals > 0 else 0.0,
            "truck_wait_min": float(wait),
            "truck_travel_min": float(travel),
            "truck_utilization": float(productive / (sc.n_trucks * horizon)),
            "truck_idle_min": float(state[:, TruckStatus.IDLE].sum()),
            "truck_maintenance_min": float(state[:, TruckStatus.MAINTENANCE].sum()),
            "shovel_utilization": float(busy / (sc.n_shovels * horizon)),
            "shovel_availability": float(operating / (sc.n_shovels * horizon)),
            "shovel_idle_min": operating - busy,
            "shovel_starvation_min": float(self.totals["shovel_starvation_min"]),
            "truck_breakdowns": float(self.totals["truck_breakdowns"]),
            "shovel_breakdowns": float(self.totals["shovel_breakdowns"]),
            "unnecessary_travel_min": float(self.totals["unnecessary_travel_min"]),
            "invalid_actions": float(self.totals["invalid_actions"]),
        }

    # ------------------------------------------------------------------ internals: time accounting
    def _accrue_to(self, t: float) -> None:
        """Integrate time-based statistics over [last_t, t]. The state is constant between events."""
        dt = t - self._last_t
        if dt <= 0.0:
            return
        en_route = np.zeros(self.scenario.n_shovels, dtype=int)
        for truck in self.trucks:
            self.truck_state_min[truck.index, truck.status] += dt
            if truck.status == TruckStatus.QUEUED:
                self.totals["truck_wait_min"] += dt
            if is_heading_to_shovel(truck):
                en_route[truck.target_shovel] += 1
        for shovel in self.shovels:
            if not shovel.operational:
                continue
            self.shovel_operational_min[shovel.index] += dt
            if shovel.loading_truck != -1:
                self.shovel_busy_min[shovel.index] += dt
            elif not shovel.queue and en_route[shovel.index] == 0:
                self.totals["shovel_starvation_min"] += dt
        self._last_t = t

    # ------------------------------------------------------------------ internals: sampling
    def _jitter(self, sigma: float) -> float:
        """Multiplicative lognormal noise with median 1. Disabled by `disturbances_enabled`."""
        if not self.cfg.disturbances_enabled or sigma <= 0.0:
            return 1.0
        return float(np.exp(self.rng.normal(0.0, sigma)))

    def _breakdown_gap(self, mtbf_minutes: float, reliability: float) -> float:
        """Exponential time until the next breakdown, measured in operating minutes."""
        return float(self.rng.exponential(mtbf_minutes * reliability))

    def _repair_times(self, median: float, sigma: float) -> tuple[float, float]:
        """(expected, realised) repair duration. Lognormal with the given median and shape."""
        expected = median * math.exp(0.5 * sigma * sigma)
        realised = median * float(np.exp(self.rng.normal(0.0, sigma))) if sigma > 0.0 else median
        return expected, realised

    # ------------------------------------------------------------------ internals: scheduling
    def _push(self, time: float, kind: EventKind, entity: int, token: int) -> None:
        heapq.heappush(self._events, (time, self._seq, int(kind), entity, token))
        self._seq += 1

    def _is_current(self, kind: EventKind, entity: int, token: int) -> bool:
        if kind == EventKind.SHOVEL_LOAD_DONE:
            return token == self.shovels[entity].load_token
        if kind in (EventKind.SHOVEL_BREAKDOWN, EventKind.SHOVEL_REPAIRED):
            return token == self.shovels[entity].op_token
        return token == self.trucks[entity].token

    def _set_truck_status(self, truck: Truck, status: TruckStatus) -> None:
        truck.status = status
        truck.status_since = self.time

    def _begin_truck_activity(
        self, k: int, status: TruckStatus, destination: int, nominal: float, sigma: float
    ) -> None:
        truck = self.trucks[k]
        truck.destination = destination
        truck.token += 1
        self._set_truck_status(truck, status)
        truck.phase.start(self.time, nominal, nominal * self._jitter(sigma))
        self._schedule_truck_segment(k)

    def _schedule_truck_segment(self, k: int) -> None:
        """Schedule the end of the current timed truck activity and, while driving, a possible breakdown."""
        truck = self.trucks[k]
        now = self.time
        self._push(truck.phase.completion_time(now), _TRUCK_END_EVENT[truck.status], k, truck.token)
        if truck.status in DRIVING_STATUSES and self.cfg.breakdowns_enabled:
            gap = self._breakdown_gap(self.cfg.truck_mtbf_minutes, truck.reliability_factor)
            if gap < truck.phase.remaining_realized(now):
                self._push(now + gap, EventKind.TRUCK_BREAKDOWN, k, truck.token)

    def _schedule_shovel_exposure(self, j: int) -> None:
        """Start a new operating period for shovel j. A breakdown may strike at any time while it runs."""
        if not self.cfg.breakdowns_enabled:
            return
        shovel = self.shovels[j]
        gap = self._breakdown_gap(self.cfg.shovel_mtbf_minutes, 1.0)
        self._push(self.time + gap, EventKind.SHOVEL_BREAKDOWN, j, shovel.op_token)

    def _try_start_loading(self, j: int) -> None:
        """If shovel j is free and its queue is not empty, start loading the head-of-queue truck."""
        shovel = self.shovels[j]
        if not shovel.operational or shovel.loading_truck != -1 or not shovel.queue:
            return
        k = shovel.queue.popleft()
        nominal = float(self._load_minutes[k, j])
        shovel.loading_truck = k
        self._set_truck_status(self.trucks[k], TruckStatus.LOADING)
        shovel.load_token += 1
        shovel.load.start(self.time, nominal, nominal * self._jitter(self.cfg.loading_jitter_sigma))
        self._push(shovel.load.completion_time(self.time), EventKind.SHOVEL_LOAD_DONE, j, shovel.load_token)

    # ------------------------------------------------------------------ internals: event handlers
    def _on_truck_arrive_shovel(self, k: int) -> None:
        truck = self.trucks[k]
        self._set_truck_status(truck, TruckStatus.QUEUED)
        self.shovels[truck.target_shovel].queue.append(k)
        self.totals["shovel_arrivals"] += 1
        self._try_start_loading(truck.target_shovel)

    def _on_truck_arrive_depot(self, k: int) -> None:
        self._begin_truck_activity(
            k, TruckStatus.UNLOADING, DEPOT_NODE, self.scenario.unload_time_minutes, self.cfg.unload_jitter_sigma
        )

    def _on_truck_unload_done(self, k: int) -> None:
        truck = self.trucks[k]
        self.totals["payload_t"] += truck.payload_t
        self.totals["loads"] += 1
        truck.payload_t = 0.0
        truck.target_shovel = -1
        truck.destination = DEPOT_NODE
        self._set_truck_status(truck, TruckStatus.IDLE)

    def _on_shovel_load_done(self, j: int) -> None:
        shovel = self.shovels[j]
        k = shovel.loading_truck
        shovel.loading_truck = -1
        truck = self.trucks[k]
        truck.payload_t = truck.capacity_t  # target_shovel is kept: it records where the load came from
        self._begin_truck_activity(
            k,
            TruckStatus.TRAVEL_LOADED,
            DEPOT_NODE,
            float(self._travel_loaded[k, j]),
            self.cfg.travel_jitter_sigma,
        )
        self._try_start_loading(j)

    def _on_truck_breakdown(self, k: int) -> None:
        truck = self.trucks[k]
        truck.phase.pause(self.time)
        truck.resume_status = truck.status
        self._set_truck_status(truck, TruckStatus.MAINTENANCE)
        truck.token += 1
        self.totals["truck_breakdowns"] += 1
        expected, realised = self._repair_times(self.cfg.truck_repair_median_minutes, self.cfg.truck_repair_sigma)
        truck.repair.start(self.time, expected, realised)
        self._push(truck.repair.completion_time(self.time), EventKind.TRUCK_REPAIRED, k, truck.token)
        self.event_log.append((self.time, "truck_breakdown", k, realised))

    def _on_truck_repaired(self, k: int) -> None:
        truck = self.trucks[k]
        truck.repair.pause(self.time)
        truck.token += 1
        self._set_truck_status(truck, truck.resume_status)
        truck.phase.resume(self.time)
        self._schedule_truck_segment(k)
        self.event_log.append((self.time, "truck_repaired", k, 0.0))

    def _on_shovel_breakdown(self, j: int) -> None:
        shovel = self.shovels[j]
        shovel.status = ShovelStatus.MAINTENANCE
        shovel.op_token += 1
        self.totals["shovel_breakdowns"] += 1
        if shovel.loading_truck != -1:
            shovel.load.pause(self.time)
            shovel.load_token += 1
        expected, realised = self._repair_times(self.cfg.shovel_repair_median_minutes, self.cfg.shovel_repair_sigma)
        shovel.repair.start(self.time, expected, realised)
        self._push(shovel.repair.completion_time(self.time), EventKind.SHOVEL_REPAIRED, j, shovel.op_token)
        self.event_log.append((self.time, "shovel_breakdown", j, realised))

    def _on_shovel_repaired(self, j: int) -> None:
        shovel = self.shovels[j]
        shovel.repair.pause(self.time)
        shovel.status = ShovelStatus.OPERATIONAL
        shovel.op_token += 1
        self._schedule_shovel_exposure(j)
        self.event_log.append((self.time, "shovel_repaired", j, 0.0))
        if shovel.loading_truck != -1:
            shovel.load.resume(self.time)
            shovel.load_token += 1
            self._push(shovel.load.completion_time(self.time), EventKind.SHOVEL_LOAD_DONE, j, shovel.load_token)
        else:
            self._try_start_loading(j)
