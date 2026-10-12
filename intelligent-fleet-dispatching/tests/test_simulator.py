"""Simulator invariants: conservation of time, breakdown pause-and-resume, legal dispatch, and leakage-free observations."""

from __future__ import annotations

import copy

import numpy as np
import pytest

from fleet_dispatch.environment.action_codec import current_action_mask
from fleet_dispatch.environment.entities import ShovelStatus, TruckStatus
from fleet_dispatch.environment.observation import build_observation
from fleet_dispatch.environment.scenario import build_scenario
from fleet_dispatch.environment.simulator import FleetSimulator, InvalidActionError


def _run_shift(sim: FleetSimulator, seed: int) -> None:
    """Dispatch uniformly at random among valid pairs until the shift ends."""
    rng = np.random.default_rng(seed)
    sim.reset(rng)
    sim.advance_to_decision()
    while not sim.done:
        cands = np.flatnonzero(current_action_mask(sim))
        action = int(rng.choice(cands))
        sim.dispatch(*divmod(action, sim.scenario.n_shovels))
        sim.advance_to_decision()


def test_truck_time_is_fully_accounted_for(config) -> None:
    sim = FleetSimulator(build_scenario(config.scenario), config.simulation)
    _run_shift(sim, seed=9)
    expected = config.scenario.n_trucks * config.scenario.horizon_minutes
    assert sim.truck_state_min.sum() == pytest.approx(expected)


def test_shovel_busy_time_never_exceeds_operating_time(config) -> None:
    sim = FleetSimulator(build_scenario(config.scenario), config.simulation)
    _run_shift(sim, seed=10)
    assert np.all(sim.shovel_busy_min <= sim.shovel_operational_min + 1e-9)
    assert np.all(sim.shovel_operational_min <= config.scenario.horizon_minutes + 1e-9)


def test_payload_is_consistent_with_loads_and_capacities(config) -> None:
    sim = FleetSimulator(build_scenario(config.scenario), config.simulation)
    _run_shift(sim, seed=11)
    summary = sim.summary()
    caps = sim.scenario.truck_capacity_t
    assert summary["loads"] > 0
    assert summary["loads"] * caps.min() - 1e-6 <= summary["payload_t"] <= summary["loads"] * caps.max() + 1e-6


def test_forced_breakdowns_pause_and_resume_without_losing_time(config) -> None:
    """Very frequent breakdowns exercise pause and resume everywhere. Time must still add up exactly."""
    sim_cfg = copy.deepcopy(config.simulation)
    sim_cfg.truck_mtbf_minutes = 25.0
    sim_cfg.shovel_mtbf_minutes = 60.0
    sim = FleetSimulator(build_scenario(config.scenario), sim_cfg)
    _run_shift(sim, seed=12)
    summary = sim.summary()
    assert summary["truck_breakdowns"] > 5
    assert summary["shovel_breakdowns"] > 0
    assert sim.truck_state_min.sum() == pytest.approx(config.scenario.n_trucks * config.scenario.horizon_minutes)
    assert np.all(sim.shovel_busy_min <= sim.shovel_operational_min + 1e-9)


def test_disabling_breakdowns_removes_them(config) -> None:
    sim_cfg = copy.deepcopy(config.simulation)
    sim_cfg.breakdowns_enabled = False
    sim = FleetSimulator(build_scenario(config.scenario), sim_cfg)
    _run_shift(sim, seed=13)
    summary = sim.summary()
    assert summary["truck_breakdowns"] == 0 and summary["shovel_breakdowns"] == 0
    assert summary["truck_maintenance_min"] == 0


def test_dispatching_a_busy_truck_or_to_a_broken_shovel_is_rejected(config) -> None:
    sim = FleetSimulator(build_scenario(config.scenario), config.simulation).reset(np.random.default_rng(0))
    sim.advance_to_decision()
    sim.dispatch(0, 0)
    with pytest.raises(InvalidActionError):
        sim.dispatch(0, 1)
    sim.shovels[1].status = ShovelStatus.MAINTENANCE
    with pytest.raises(InvalidActionError):
        sim.dispatch(1, 1)


def test_same_seed_gives_identical_shift(config) -> None:
    scenario = build_scenario(config.scenario)
    a = FleetSimulator(scenario, config.simulation)
    b = FleetSimulator(scenario, config.simulation)
    _run_shift(a, seed=14)
    _run_shift(b, seed=14)
    assert a.summary() == b.summary()


def test_observation_does_not_reveal_unobservable_noise(config) -> None:
    """Right after a dispatch, the observation must not depend on the realised travel jitter or breakdown timing.

    The two shifts differ only in whether disturbances are enabled. Both make the same first dispatch at t=0,
    so only the realised noise differs. The observation shows nominal (expected) times only, so it must match.
    """
    scenario = build_scenario(config.scenario)
    noisy = FleetSimulator(scenario, config.simulation)
    cfg_quiet = copy.deepcopy(config.simulation)
    cfg_quiet.disturbances_enabled = False
    quiet = FleetSimulator(scenario, cfg_quiet)
    for sim in (noisy, quiet):
        sim.reset(np.random.default_rng(15))
        sim.advance_to_decision()
        sim.dispatch(3, 2)
    np.testing.assert_array_equal(build_observation(noisy, 60.0), build_observation(quiet, 60.0))


def test_queue_wait_equals_time_spent_in_the_queued_state(config) -> None:
    """Waiting is charged once, as accrued queue minutes. It must equal the time trucks spent QUEUED."""
    sim = FleetSimulator(build_scenario(config.scenario), config.simulation)
    _run_shift(sim, seed=16)
    assert sim.summary()["truck_wait_min"] == pytest.approx(sim.truck_state_min[:, TruckStatus.QUEUED].sum())
