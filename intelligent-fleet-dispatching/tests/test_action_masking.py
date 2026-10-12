"""Action encoding and masking: the mask must describe exactly the legal dispatches, everywhere it is used."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from fleet_dispatch.agents.action_masking import (
    MASKED_Q_VALUE,
    apply_action_mask,
    greedy_valid_action,
    has_valid_action,
    masked_argmax,
    masked_max,
    sample_valid_action,
)
from fleet_dispatch.environment.action_codec import (
    build_action_mask,
    current_action_mask,
    decode_action,
    encode_action,
)
from fleet_dispatch.environment.entities import ShovelStatus, TruckStatus
from fleet_dispatch.environment.mining_env import FleetDispatchEnv
from fleet_dispatch.environment.scenario import build_scenario
from fleet_dispatch.environment.simulator import FleetSimulator


def test_encode_decode_roundtrip_covers_all_pairs() -> None:
    n_shovels = 4
    seen = set()
    for truck in range(20):
        for shovel in range(n_shovels):
            action = encode_action(truck, shovel, n_shovels)
            assert decode_action(action, n_shovels) == (truck, shovel)
            seen.add(action)
    assert seen == set(range(80))


def test_mask_is_outer_product_of_idle_trucks_and_operational_shovels() -> None:
    truck_idle = np.array([True, False, True])
    shovel_ok = np.array([False, True])
    mask = build_action_mask(truck_idle, shovel_ok)
    expected = np.array([[False, True], [False, False], [False, True]]).reshape(-1)
    np.testing.assert_array_equal(mask, expected)


def test_masked_argmax_ignores_invalid_actions_even_with_higher_q() -> None:
    q = torch.tensor([[10.0, 1.0, 2.0]])
    mask = torch.tensor([[False, True, True]])
    assert masked_argmax(q, mask).item() == 2
    assert apply_action_mask(q, mask)[0, 0].item() == MASKED_Q_VALUE


def test_masked_max_and_has_valid_action_handle_empty_rows() -> None:
    q = torch.tensor([[5.0, 6.0], [7.0, 8.0]])
    mask = torch.tensor([[False, True], [False, False]])
    assert masked_max(q, mask).tolist() == [6.0, MASKED_Q_VALUE]
    assert has_valid_action(mask).tolist() == [True, False]


def test_sampling_and_greedy_choice_only_return_valid_actions() -> None:
    rng = np.random.default_rng(0)
    mask = np.zeros(80, dtype=bool)
    mask[[5, 17, 60]] = True
    q = np.arange(80, dtype=float)[::-1]  # the highest Q is at index 0, which is invalid
    samples = {sample_valid_action(mask, rng) for _ in range(200)}
    assert samples <= {5, 17, 60}
    assert greedy_valid_action(q, mask) == 5


def test_empty_mask_is_an_error() -> None:
    with pytest.raises(ValueError):
        sample_valid_action(np.zeros(4, dtype=bool), np.random.default_rng(0))
    with pytest.raises(ValueError):
        greedy_valid_action(np.zeros(4), np.zeros(4, dtype=bool))


def _fresh_sim(config) -> FleetSimulator:
    sim = FleetSimulator(build_scenario(config.scenario), config.simulation)
    sim.reset(np.random.default_rng(0))
    sim.advance_to_decision()
    return sim


def test_maintenance_shovel_removes_its_column(config) -> None:
    sim = _fresh_sim(config)
    sim.shovels[2].status = ShovelStatus.MAINTENANCE
    mask = current_action_mask(sim).reshape(20, 4)
    assert not mask[:, 2].any()
    assert mask[:, [0, 1, 3]].all()


def test_busy_truck_removes_its_row(config) -> None:
    sim = _fresh_sim(config)
    sim.dispatch(7, 0)
    assert sim.trucks[7].status == TruckStatus.TRAVEL_EMPTY
    mask = current_action_mask(sim).reshape(20, 4)
    assert not mask[7].any()
    assert mask[6].all()


def test_mask_matches_simulator_state_throughout_an_episode(config) -> None:
    """At every decision in a full shift, the mask equals idle-truck x operational-shovel from raw state."""
    env = FleetDispatchEnv(config)
    env.reset(seed=21)
    rng = np.random.default_rng(3)
    truncated = False
    while not truncated:
        mask = env.action_masks()
        idle = np.array([t.status == TruckStatus.IDLE for t in env.simulator.trucks])
        ops = np.array([s.operational for s in env.simulator.shovels])
        np.testing.assert_array_equal(mask, build_action_mask(idle, ops))
        _, _, _, truncated, _ = env.step(int(rng.choice(np.flatnonzero(mask))))
    assert env.simulator.totals["invalid_actions"] == 0
