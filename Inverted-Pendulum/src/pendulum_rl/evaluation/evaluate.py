"""Policy evaluation: fixed-seed episodes, per-episode metrics and summaries.

All controllers (greedy DQN, model-based baseline, random, a swing-up -> stabilizer
hand-off of two DQN policies) share one interface so they are scored by exactly the same
code on exactly the same initial conditions (same seeds).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Callable, Protocol

import numpy as np

from ..environment import EnvConfig, PendulumEnv, RewardConfig
from ..physics import PhysicsParams
from ..physics.dynamics import wrap_angle

# scenario -> EnvConfig overrides. Evaluation episodes never end early on success so that
# *sustained* balance is measured, and sensor noise / disturbances are off unless asked for.
SCENARIOS: dict[str, dict] = {
    "stabilize": dict(start_mode="upright", success_mode="balance", terminate_on_success=False,
                      max_episode_seconds=8.0, physics_randomization=0.0, disturbance_probability=0.0),
    "swingup": dict(start_mode="hanging", success_mode="balance", terminate_on_success=False,
                    max_episode_seconds=10.0, physics_randomization=0.0, disturbance_probability=0.0),
    "combined": dict(start_mode="mixed", success_mode="balance", terminate_on_success=False,
                     max_episode_seconds=10.0, physics_randomization=0.0, disturbance_probability=0.0),
}
PRIMARY_METRIC = {"stabilize": "balance_rate", "swingup": "swing_up_rate", "combined": "balance_rate"}
TRAINED_THRESHOLD = 0.8   # primary success rate at/above which a policy is labelled "trained"


def classify_status(env_steps_trained: int, primary_rate: float, threshold: float = TRAINED_THRESHOLD) -> str:
    """``untrained`` (no learning happened), ``partially trained`` or ``trained`` (primary success >= threshold)."""
    if env_steps_trained <= 0:
        return "untrained"
    return "trained" if primary_rate >= threshold else "partially trained"


# ----------------------------------------------------------------------------- policies
class Policy(Protocol):
    name: str
    action_mode: str  # "discrete" or "continuous"

    def reset(self, env: PendulumEnv) -> None: ...
    def act(self, env: PendulumEnv, obs: np.ndarray): ...


class DQNPolicy:
    action_mode = "discrete"

    def __init__(self, agent, name: str = "dqn"):
        self.agent, self.name = agent, name

    def reset(self, env) -> None: ...

    def act(self, env, obs):
        return self.agent.act(obs, greedy=True)


class BaselinePolicy:
    action_mode = "continuous"
    name = "baseline"

    def __init__(self, controller_factory: Callable[[PendulumEnv], object], name: str = "baseline"):
        self._factory, self.name = controller_factory, name
        self._ctl = None

    def reset(self, env) -> None:
        self._ctl = self._factory(env)
        self._ctl.reset()

    def act(self, env, obs):
        return np.array([self._ctl.command(env.physical_state)], dtype=np.float32)


class RandomPolicy:
    action_mode = "discrete"

    def __init__(self, seed: int = 0, name: str = "random"):
        self.name = name
        self._rng = np.random.default_rng(seed)

    def reset(self, env) -> None: ...

    def act(self, env, obs):
        return int(self._rng.integers(env.n_actions))


class SwitchingPolicy:
    """Swing-up specialist until the pendulum enters the catch region, then a stabilization specialist."""

    action_mode = "discrete"

    def __init__(self, swingup_agent, stabilizer_agent, catch_angle: float = 0.35, catch_speed: float = 4.0,
                 release_angle: float = 0.8, name: str = "dqn_swingup+stabilize"):
        self.swingup, self.stabilizer, self.name = swingup_agent, stabilizer_agent, name
        self.catch_angle, self.catch_speed, self.release_angle = catch_angle, catch_speed, release_angle
        self.stabilizing = False

    def reset(self, env) -> None:
        self.stabilizing = False

    def act(self, env, obs):
        theta, omega = float(wrap_angle(env.physical_state[2])), float(env.physical_state[3])
        if not self.stabilizing and abs(theta) <= self.catch_angle and abs(omega) <= self.catch_speed:
            self.stabilizing = True
        elif self.stabilizing and abs(theta) > self.release_angle:
            self.stabilizing = False
        return (self.stabilizer if self.stabilizing else self.swingup).act(obs, greedy=True)


# ----------------------------------------------------------------------------- running
def make_eval_env(scenario: str, env_cfg: EnvConfig, physics: PhysicsParams, reward: RewardConfig,
                  action_mode: str = "discrete", **overrides) -> PendulumEnv:
    if scenario not in SCENARIOS:
        raise ValueError(f"Unknown scenario {scenario!r}; choose from {sorted(SCENARIOS)}")
    cfg = replace(env_cfg, **{**SCENARIOS[scenario], "action_mode": action_mode, **overrides})
    return PendulumEnv(cfg, physics, reward)


def run_episode(env: PendulumEnv, policy: Policy, seed: int, record: bool = False) -> tuple[dict, dict | None]:
    """Run one episode; returns ``(metrics, trajectory_or_None)``."""
    obs, info = env.reset(seed=seed)
    policy.reset(env)
    traj = {k: [] for k in ("t", "x", "theta", "theta_dot", "command", "energy", "reward")} if record else None
    if record:
        _append(traj, info, 0.0)
    while True:
        action = policy.act(env, obs)
        obs, reward, terminated, truncated, info = env.step(action)
        if record:
            _append(traj, info, reward)
        if terminated or truncated:
            break
    metrics = dict(info["episode_metrics"])
    metrics["seed"] = seed
    return metrics, ({k: np.asarray(v) for k, v in traj.items()} if record else None)


def _append(traj: dict, info: dict, reward: float) -> None:
    traj["t"].append(info["time"]); traj["x"].append(info["x"]); traj["theta"].append(info["theta"])
    traj["theta_dot"].append(info["theta_dot"]); traj["command"].append(info["command"])
    traj["energy"].append(info["swing_energy"]); traj["reward"].append(reward)


def evaluate_policy(policy: Policy, env: PendulumEnv, episodes: int, seed: int = 10_000) -> list[dict]:
    """Run ``episodes`` episodes with seeds ``seed, seed+1, ...`` (identical across policies)."""
    if env.cfg.action_mode != policy.action_mode:
        raise ValueError(f"Policy '{policy.name}' needs a {policy.action_mode} action environment, "
                         f"got {env.cfg.action_mode}")
    rows = []
    for i in range(episodes):
        metrics, _ = run_episode(env, policy, seed + i)
        counts = metrics.pop("action_counts")
        for j, c in enumerate(counts):
            metrics[f"action_count_{j}"] = c
        rows.append(metrics)
    return rows


def summarize(rows: list[dict], scenario: str) -> dict:
    """Aggregate per-episode metrics into the benchmark summary."""
    n = len(rows)
    if n == 0:
        raise ValueError("No episodes to summarize")

    def mean(key):
        vals = [r[key] for r in rows]
        return float(np.mean(vals))

    t_up = [r["time_to_upright"] for r in rows if not math.isnan(r["time_to_upright"])]
    reasons: dict[str, int] = {}
    for r in rows:
        if r["violation"]:
            reasons[r["violation_reason"]] = reasons.get(r["violation_reason"], 0) + 1
    summary = {
        "scenario": scenario,
        "episodes": n,
        "swing_up_rate": float(np.mean([r["swing_up_success"] for r in rows])),
        "balance_rate": float(np.mean([r["balance_success"] for r in rows])),
        "violation_rate": float(np.mean([r["violation"] for r in rows])),
        "mean_abs_angle_error": mean("mean_abs_angle_error"),
        "max_abs_angle_error": float(np.max([r["max_abs_angle_error"] for r in rows])),
        "mean_time_to_upright": float(np.mean(t_up)) if t_up else math.nan,
        "median_time_to_upright": float(np.median(t_up)) if t_up else math.nan,
        "mean_balanced_time": mean("balanced_time"),
        "mean_longest_balance": mean("longest_balance"),
        "mean_effort_abs": mean("effort_abs"),
        "mean_effort_sq": mean("effort_sq"),
        "mean_return": mean("return"),
        "violations_by_reason": reasons,
    }
    summary["primary_metric"] = PRIMARY_METRIC[scenario]
    summary["primary_rate"] = summary[PRIMARY_METRIC[scenario]]
    return summary


def evaluate_agent_quick(agent, env_cfg: EnvConfig, physics: PhysicsParams, reward: RewardConfig, scenario: str,
                         episodes: int, seed: int) -> dict:
    """Greedy evaluation of a DQN agent (used for periodic evaluation during training)."""
    env = make_eval_env(scenario, env_cfg, physics, reward, "discrete")
    try:
        return summarize(evaluate_policy(DQNPolicy(agent), env, episodes, seed), scenario)
    finally:
        env.close()
