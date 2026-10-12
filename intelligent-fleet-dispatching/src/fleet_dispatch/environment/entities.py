"""Mutable state records for haul trucks and loading shovels, and the pausable timer they share.

Node convention used throughout the package: node 0 is the depot/dump, and shovel j is node j + 1.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from enum import IntEnum

DEPOT_NODE = 0


def shovel_node(shovel_index: int) -> int:
    """Node id of a shovel (node 0 is the depot)."""
    return shovel_index + 1


class TruckStatus(IntEnum):
    """Operational state of a haul truck. Only IDLE trucks are eligible for a dispatch action."""

    IDLE = 0            # at the depot, waiting for a dispatch decision
    TRAVEL_EMPTY = 1    # returning empty from the depot to its assigned shovel
    QUEUED = 2          # waiting in a shovel's queue
    LOADING = 3         # being loaded by a shovel
    TRAVEL_LOADED = 4   # hauling ore back to the depot
    UNLOADING = 5       # tipping ore at the depot
    MAINTENANCE = 6     # broken down; under repair


class ShovelStatus(IntEnum):
    OPERATIONAL = 0
    MAINTENANCE = 1


N_TRUCK_STATUSES = len(TruckStatus)

DRIVING_STATUSES = frozenset({TruckStatus.TRAVEL_EMPTY, TruckStatus.TRAVEL_LOADED})


@dataclass
class Timer:
    """A timed activity that can be paused by a breakdown and resumed later.

    `nominal` is the expected duration. It is what observations report, so the agent never sees the
    realised noise. `realized` is the sampled duration, which only the event scheduler reads. Progress
    accrues only while the timer is running (`resumed_at` is set).
    """

    nominal: float = 0.0
    realized: float = 0.0
    elapsed: float = 0.0
    resumed_at: float = math.nan

    @property
    def running(self) -> bool:
        return not math.isnan(self.resumed_at)

    def start(self, now: float, nominal: float, realized: float) -> None:
        self.nominal = nominal
        self.realized = realized
        self.elapsed = 0.0
        self.resumed_at = now

    def pause(self, now: float) -> None:
        if self.running:
            self.elapsed += now - self.resumed_at
            self.resumed_at = math.nan

    def resume(self, now: float) -> None:
        if not self.running:
            self.resumed_at = now

    def progress(self, now: float) -> float:
        extra = now - self.resumed_at if self.running else 0.0
        return self.elapsed + extra

    def remaining_realized(self, now: float) -> float:
        return max(self.realized - self.progress(now), 0.0)

    def remaining_expected(self, now: float) -> float:
        return max(self.nominal - self.progress(now), 0.0)

    def completion_time(self, now: float) -> float:
        return now + self.remaining_realized(now)


@dataclass
class Truck:
    index: int
    capacity_t: float
    speed_factor: float
    reliability_factor: float
    status: TruckStatus = TruckStatus.IDLE
    status_since: float = 0.0
    destination: int = DEPOT_NODE    # node the truck is at or heading to
    target_shovel: int = -1          # assigned shovel index, -1 when none
    payload_t: float = 0.0
    token: int = 0                   # bumped whenever a scheduled truck event must be invalidated
    phase: Timer = field(default_factory=Timer)     # current driving or unloading activity
    repair: Timer = field(default_factory=Timer)    # current breakdown repair
    resume_status: TruckStatus = TruckStatus.IDLE   # status to restore after a repair

    @property
    def loaded(self) -> bool:
        return self.payload_t > 0.0


def is_heading_to_shovel(truck: Truck) -> bool:
    """True while a truck is assigned to a shovel it has not reached yet, including when broken down en route."""
    if truck.target_shovel < 0:
        return False
    if truck.status == TruckStatus.TRAVEL_EMPTY:
        return True
    return truck.status == TruckStatus.MAINTENANCE and truck.resume_status == TruckStatus.TRAVEL_EMPTY


@dataclass
class Shovel:
    index: int
    loading_rate_tpm: float
    status: ShovelStatus = ShovelStatus.OPERATIONAL
    queue: deque[int] = field(default_factory=deque)
    loading_truck: int = -1
    load: Timer = field(default_factory=Timer)      # loading activity for the truck at the face
    repair: Timer = field(default_factory=Timer)
    load_token: int = 0                              # invalidates scheduled loading-completion events
    op_token: int = 0                                # invalidates scheduled shovel breakdown events

    @property
    def operational(self) -> bool:
        return self.status == ShovelStatus.OPERATIONAL
