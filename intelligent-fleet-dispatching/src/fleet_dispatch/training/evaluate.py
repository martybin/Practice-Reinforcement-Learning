"""Held-out evaluation of dispatch policies.

Protocol:
  * Every policy runs on the same evaluation seeds (evaluation.seed_start, ...). Each seed is one full shift
    in the same fixed scenario, so comparisons are paired.
  * Policies only read the DecisionView, which is built from observable state. Their actions are always
    drawn from the mask.
  * Nothing is tuned on these seeds. The agent is loaded from a checkpoint and never trained here: no
    observations are stored, no gradients run, and no weights change.
  * One row per (policy, seed) is written to CSV. Aggregation and statistics are in compare_baselines.
"""

from __future__ import annotations

import csv
import time
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from fleet_dispatch.agents.dqn_agent import DQNAgent, DQNPolicy, load_policy_agent
from fleet_dispatch.config import ProjectConfig
from fleet_dispatch.environment.mining_env import FleetDispatchEnv
from fleet_dispatch.environment.scenario import Scenario, build_scenario
from fleet_dispatch.metrics.operational_metrics import EPISODE_METRICS
from fleet_dispatch.policies import NearestEligibleShovelPolicy, RandomValidPolicy, ShortestQueuePolicy
from fleet_dispatch.utils.logging import get_logger
from fleet_dispatch.utils.seeding import evaluation_seeds

_log = get_logger(__name__)

BASELINE_POLICIES = ("random_valid", "nearest_eligible", "shortest_queue")
ALL_POLICIES = ("dqn", *BASELINE_POLICIES)
EVAL_COLUMNS = ("policy", "seed", *EPISODE_METRICS)


def make_policy(name: str, seed: int, agent: DQNAgent | None):
    if name == "dqn":
        if agent is None:
            raise ValueError("policy 'dqn' requires a trained agent (pass --checkpoint)")
        return DQNPolicy(agent)
    if name == "random_valid":
        return RandomValidPolicy(seed=seed)
    if name == "nearest_eligible":
        return NearestEligibleShovelPolicy()
    if name == "shortest_queue":
        return ShortestQueuePolicy()
    raise ValueError(f"unknown policy {name!r}; choose from {ALL_POLICIES}")


def run_episode(env: FleetDispatchEnv, policy, seed: int) -> dict:
    """Play one full shift with `policy`. Returns the episode's operational metrics."""
    _, info = env.reset(seed=seed)
    done = False
    started = time.perf_counter()
    while not done:
        view = env.decision_view()
        action = policy.act(view)
        _, _, terminated, truncated, info = env.step(action)
        done = terminated or truncated
    runtime = time.perf_counter() - started
    episode = info["episode"]
    row = {"policy": policy.name, "seed": seed}
    for metric in EPISODE_METRICS:
        if metric == "runtime_s":
            row[metric] = runtime
        elif metric in episode:
            row[metric] = episode[metric]
        else:
            row[metric] = float(episode["return"]) if metric == "episode_return" else np.nan
    return row


def evaluate_policies(
    config: ProjectConfig,
    policy_names: Iterable[str],
    seeds: list[int],
    *,
    agent: DQNAgent | None = None,
    scenario: Scenario | None = None,
    output_csv: Path | None = None,
) -> pd.DataFrame:
    """Run every named policy on every seed and return a tidy DataFrame (optionally written to CSV)."""
    scenario = scenario or build_scenario(config.scenario)
    env = FleetDispatchEnv(config, scenario=scenario)
    if agent is not None and agent.obs_dim != env.obs_dim:
        raise ValueError(f"agent expects obs_dim={agent.obs_dim}, environment has {env.obs_dim}")
    rows: list[dict] = []
    names = list(policy_names)
    for name in names:
        for seed in seeds:
            policy = make_policy(name, seed, agent)
            rows.append(run_episode(env, policy, seed))
        _log.info("evaluated %-16s on %d seeds (mean return %.2f)", name, len(seeds),
                  np.mean([r["episode_return"] for r in rows if r["policy"] == name]))
    frame = pd.DataFrame(rows, columns=list(EVAL_COLUMNS))
    if output_csv is not None:
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(output_csv, index=False, quoting=csv.QUOTE_MINIMAL)
    return frame


def evaluate_checkpoint(
    config: ProjectConfig,
    checkpoint: Path | None,
    policy_names: Iterable[str],
    n_episodes: int,
    output_csv: Path,
    device: str = "cpu",
) -> pd.DataFrame:
    """Evaluate a trained checkpoint together with the baselines on the held-out seed block."""
    agent = None
    if checkpoint is not None:
        agent, _ = load_policy_agent(checkpoint, device=device)
    seeds = evaluation_seeds(config.evaluation.seed_start, n_episodes)
    return evaluate_policies(config, policy_names, seeds, agent=agent, output_csv=output_csv)
