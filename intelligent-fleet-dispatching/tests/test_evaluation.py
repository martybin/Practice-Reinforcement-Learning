"""Evaluation: metric calculations, paired statistics, baseline validity, and no contamination of learning."""

from __future__ import annotations

import copy

import numpy as np
import pandas as pd
import pytest

from fleet_dispatch.agents.dqn_agent import DQNAgent
from fleet_dispatch.environment.mining_env import FleetDispatchEnv
from fleet_dispatch.metrics.operational_metrics import (
    bootstrap_ci,
    mean_and_std,
    moving_average,
    paired_comparison,
    summarize,
)
from fleet_dispatch.training.evaluate import BASELINE_POLICIES, evaluate_policies
from fleet_dispatch.utils.seeding import evaluation_seeds


def test_mean_std_and_bootstrap_ci_behave() -> None:
    mean, std = mean_and_std(np.array([2.0, 4.0, 6.0]))
    assert mean == pytest.approx(4.0) and std == pytest.approx(2.0)
    values = np.arange(10, dtype=float)
    low, high = bootstrap_ci(values, seed=3)
    assert low <= values.mean() <= high
    assert bootstrap_ci(values, seed=3) == bootstrap_ci(values, seed=3), "bootstrap must be seeded"


def test_moving_average_uses_trailing_window() -> None:
    np.testing.assert_allclose(moving_average(np.array([1.0, 2.0, 3.0, 4.0]), 2), [1.0, 1.5, 2.5, 3.5])


def test_paired_comparison_detects_a_consistent_improvement() -> None:
    seeds = list(range(12))
    rows = [{"policy": "a", "seed": s, "episode_return": 10.0 + s} for s in seeds]
    rows += [{"policy": "b", "seed": s, "episode_return": 8.0 + s} for s in seeds]
    frame = pd.DataFrame(rows)
    result = paired_comparison(frame, "a", "b", "episode_return", n_perm=5000, seed=0)
    assert result.n_pairs == 12
    assert result.mean_diff == pytest.approx(2.0)
    assert result.a_better_fraction == 1.0
    assert result.p_value < 0.01


def test_paired_comparison_respects_metric_direction() -> None:
    """For waiting time, lower is better. So a lower value must count as 'a better'."""
    rows = [{"policy": "a", "seed": s, "avg_wait_min": 1.0} for s in range(6)]
    rows += [{"policy": "b", "seed": s, "avg_wait_min": 3.0} for s in range(6)]
    result = paired_comparison(pd.DataFrame(rows), "a", "b", "avg_wait_min", n_perm=500)
    assert result.a_better_fraction == 1.0


def test_paired_comparison_requires_shared_seeds() -> None:
    frame = pd.DataFrame([{"policy": "a", "seed": 1, "x": 1.0}, {"policy": "b", "seed": 2, "x": 1.0}])
    with pytest.raises(ValueError):
        paired_comparison(frame, "a", "b", "x")


def test_summary_reports_mean_std_and_counts() -> None:
    frame = pd.DataFrame({"policy": ["p"] * 4, "payload_t": [10.0, 20.0, 30.0, 40.0]})
    row = summarize(frame, metrics=("payload_t",)).iloc[0]
    assert row["mean"] == pytest.approx(25.0)
    assert row["n"] == 4


@pytest.fixture
def strict_config(config):
    """Invalid actions raise, so any baseline that chose a masked action would fail the test."""
    cfg = copy.deepcopy(config)
    cfg.env.invalid_action_policy = "raise"
    return cfg


def test_baselines_obey_the_action_mask_and_metrics_are_sane(strict_config) -> None:
    seeds = evaluation_seeds(strict_config.evaluation.seed_start, 2)
    frame = evaluate_policies(strict_config, BASELINE_POLICIES, seeds)
    assert len(frame) == len(BASELINE_POLICIES) * 2
    assert (frame["invalid_actions"] == 0).all()
    assert frame["truck_utilization"].between(0, 1).all()
    assert frame["shovel_utilization"].between(0, 1).all()
    assert (frame["shovel_utilization"] <= frame["shovel_availability"] + 1e-9).all()
    caps = strict_config.scenario.truck_capacity_choices_tonnes
    assert (frame["payload_t"] >= frame["loads"] * min(caps) - 1e-6).all()
    assert (frame["payload_t"] <= frame["loads"] * max(caps) + 1e-6).all()


def test_evaluation_is_reproducible_for_fixed_seeds(strict_config) -> None:
    seeds = evaluation_seeds(strict_config.evaluation.seed_start, 2)
    a = evaluate_policies(strict_config, ["shortest_queue", "random_valid"], seeds)
    b = evaluate_policies(strict_config, ["shortest_queue", "random_valid"], seeds)
    pd.testing.assert_frame_equal(a.drop(columns=["runtime_s"]), b.drop(columns=["runtime_s"]))


def test_evaluation_never_trains_the_agent(strict_config) -> None:
    """Evaluation must not store transitions, take gradient steps, or change weights."""
    env = FleetDispatchEnv(strict_config)
    agent = DQNAgent(env.obs_dim, env.n_actions, strict_config.agent, device="cpu")
    weights_before = [p.detach().clone() for p in agent.online.parameters()]
    seeds = evaluation_seeds(strict_config.evaluation.seed_start, 1)
    evaluate_policies(strict_config, ["dqn"], seeds, agent=agent)
    assert len(agent.buffer) == 0
    assert agent.grad_steps == 0 and agent.env_steps == 0
    for before, after in zip(weights_before, agent.online.parameters()):
        assert np.array_equal(before.numpy(), after.detach().numpy())


def test_dqn_policy_requires_an_agent(strict_config) -> None:
    with pytest.raises(ValueError):
        evaluate_policies(strict_config, ["dqn"], [strict_config.evaluation.seed_start])
