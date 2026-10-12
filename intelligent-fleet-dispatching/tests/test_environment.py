"""Environment behaviour: reset and step semantics, spaces, invalid actions, truncation, and determinism."""

from __future__ import annotations

import copy

import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from conftest import run_random_valid_episode
from fleet_dispatch.environment.entities import TruckStatus
from fleet_dispatch.environment.mining_env import FleetDispatchEnv
from fleet_dispatch.environment.observation import observation_dim
from fleet_dispatch.environment.simulator import InvalidActionError


def test_reset_returns_valid_observation_and_mask(env: FleetDispatchEnv) -> None:
    obs, info = env.reset(seed=3)
    assert obs.shape == (observation_dim(20, 4),)
    assert obs.dtype == np.float32
    assert np.isfinite(obs).all()
    assert env.observation_space.contains(obs)
    mask = info["action_mask"]
    assert mask.shape == (env.action_space.n,) == (80,)
    assert mask.all(), "at t=0 every truck is idle and every shovel is operational"
    assert info["sim_time"] == 0.0


def test_action_space_matches_truck_shovel_pairs(env: FleetDispatchEnv) -> None:
    assert env.action_space.n == env.n_trucks * env.n_shovels == 80
    assert env.observation_space.shape == (env.obs_dim,)
    assert env.observation_space.low.min() == 0.0 and env.observation_space.high.max() == 1.0


def test_valid_dispatch_changes_truck_state(env: FleetDispatchEnv) -> None:
    env.reset(seed=1)
    action = 2 * 4 + 1  # truck 2 to shovel 1
    _, reward, _, truncated, info = env.step(action)
    truck = env.simulator.trucks[2]
    assert truck.status in (TruckStatus.TRAVEL_EMPTY, TruckStatus.IDLE)
    assert info["dispatched"] == (2, 1)
    assert not info["invalid_action"]
    assert not truncated
    assert np.isfinite(reward)


def test_step_is_observed_only_through_mask(env: FleetDispatchEnv) -> None:
    _, info = env.reset(seed=1)
    env.step(0)  # truck 0 to shovel 0
    mask = env.action_masks()
    assert not mask[0:4].any(), "truck 0 is no longer idle, so none of its pairs are valid"
    assert mask[4:8].all(), "the other trucks are still idle"


def test_invalid_action_is_penalised_without_changing_state(config) -> None:
    env = FleetDispatchEnv(config)
    env.reset(seed=1)
    env.step(0)  # make truck 0 busy
    before = [(t.status, t.target_shovel) for t in env.simulator.trucks]
    invalid_action = 0  # truck 0 is no longer idle
    _, reward, _, _, info = env.step(invalid_action)
    after = [(t.status, t.target_shovel) for t in env.simulator.trucks]
    assert info["invalid_action"] is True
    assert info["dispatched"] is None
    assert after == before, "an invalid action must not change any truck"
    # No simulated time passes at t=0 (other trucks are still idle), so the reward is exactly the penalty.
    assert reward == pytest.approx(config.reward.invalid_action.coefficient)
    assert env.simulator.totals["invalid_actions"] == 1


def test_invalid_action_raises_in_raise_mode(config) -> None:
    config.env.invalid_action_policy = "raise"
    env = FleetDispatchEnv(config)
    env.reset(seed=1)
    env.step(0)
    with pytest.raises(InvalidActionError):
        env.step(0)


def test_repeated_invalid_actions_are_bounded(config) -> None:
    config.env.max_consecutive_invalid_actions = 3
    env = FleetDispatchEnv(config)
    env.reset(seed=1)
    env.step(0)
    for _ in range(3):
        env.step(0)
    with pytest.raises(InvalidActionError):
        env.step(0)


def test_out_of_range_action_is_rejected(env: FleetDispatchEnv) -> None:
    env.reset(seed=1)
    with pytest.raises(ValueError):
        env.step(80)
    with pytest.raises(ValueError):
        env.step(-1)


def test_episode_truncates_at_horizon_not_terminates(env: FleetDispatchEnv, config) -> None:
    env.reset(seed=5)
    rng = np.random.default_rng(0)
    terminated = truncated = False
    info = {}
    while not (terminated or truncated):
        action = int(rng.choice(np.flatnonzero(env.action_masks())))
        _, _, terminated, truncated, info = env.step(action)
        assert terminated is False, "the default MDP has no genuine terminal state"
    assert truncated is True
    assert info["sim_time"] == pytest.approx(config.scenario.horizon_minutes)
    assert "episode" in info and info["episode"]["sim_time"] == pytest.approx(config.scenario.horizon_minutes)


def test_stepping_after_the_end_raises(env: FleetDispatchEnv) -> None:
    run_random_valid_episode(env, seed=2)
    with pytest.raises(RuntimeError):
        env.step(0)


def test_identical_seeds_give_identical_trajectories(config) -> None:
    rewards_a = run_random_valid_episode(FleetDispatchEnv(config), seed=7, rng_seed=11)
    rewards_b = run_random_valid_episode(FleetDispatchEnv(config), seed=7, rng_seed=11)
    assert rewards_a == rewards_b


def test_different_seeds_give_different_outcomes(config) -> None:
    rewards_a = run_random_valid_episode(FleetDispatchEnv(config), seed=7, rng_seed=11)
    rewards_b = run_random_valid_episode(FleetDispatchEnv(config), seed=8, rng_seed=11)
    assert rewards_a != rewards_b


def test_gymnasium_env_checker_passes(config) -> None:
    config = copy.deepcopy(config)
    config.env.max_consecutive_invalid_actions = 10_000  # the checker samples many random, often invalid, actions
    env = FleetDispatchEnv(config)
    check_env(env, skip_render_check=True)


def test_render_text_and_snapshot(env: FleetDispatchEnv) -> None:
    env.reset(seed=1)
    env.step(0)
    text = env.render_text()
    assert "shovel 0" in text and "trucks:" in text
    snap = env.snapshot()
    assert len(snap["trucks"]) == 20 and len(snap["shovels"]) == 4
    assert snap["dispatch_log"], "the dispatch just made must appear in the log"
