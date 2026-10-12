"""Operational metrics, summary statistics, and paired policy comparisons.

Statistics are NumPy only, with no SciPy dependency:
  * Confidence intervals use the percentile bootstrap over episodes (seeded, so results are repeatable).
  * Paired comparisons use identical evaluation seeds for both policies. The test is a sign-flip
    permutation test on the per-seed differences, two-sided.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

EPISODE_METRICS = (
    "episode_return",
    "payload_t",
    "loads",
    "avg_wait_min",
    "truck_wait_min",
    "truck_travel_min",
    "truck_utilization",
    "truck_idle_min",
    "truck_maintenance_min",
    "shovel_utilization",
    "shovel_availability",
    "shovel_idle_min",
    "shovel_starvation_min",
    "truck_breakdowns",
    "shovel_breakdowns",
    "unnecessary_travel_min",
    "invalid_actions",
    "dispatches",
    "runtime_s",
)

# Metrics where a larger value is better. Used only to label comparisons. Nothing is ranked automatically.
HIGHER_IS_BETTER = frozenset({"episode_return", "payload_t", "loads", "truck_utilization", "shovel_utilization"})


def mean_and_std(values: np.ndarray) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return float("nan"), float("nan")
    std = float(values.std(ddof=1)) if values.size > 1 else 0.0
    return float(values.mean()), std


def bootstrap_ci(values: np.ndarray, n_boot: int = 5000, alpha: float = 0.05, seed: int = 0) -> tuple[float, float]:
    """Percentile bootstrap confidence interval for the mean."""
    values = np.asarray(values, dtype=float)
    if values.size < 2:
        mean = float(values.mean()) if values.size else float("nan")
        return mean, mean
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, values.size, size=(n_boot, values.size))
    means = values[idx].mean(axis=1)
    low, high = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return float(low), float(high)


def summarize(episodes: pd.DataFrame, metrics: tuple[str, ...] = EPISODE_METRICS) -> pd.DataFrame:
    """One row per (policy, metric) with mean, standard deviation, bootstrap 95% CI, and episode count."""
    rows = []
    for policy, group in episodes.groupby("policy", sort=False):
        for metric in metrics:
            if metric not in group:
                continue
            values = group[metric].to_numpy(dtype=float)
            mean, std = mean_and_std(values)
            low, high = bootstrap_ci(values)
            rows.append(
                {"policy": policy, "metric": metric, "mean": mean, "std": std,
                 "ci95_low": low, "ci95_high": high, "n": int(values.size)}
            )
    return pd.DataFrame(rows)


@dataclass(frozen=True)
class PairedResult:
    policy_a: str
    policy_b: str
    metric: str
    n_pairs: int
    mean_diff: float          # mean of (a - b) over shared seeds
    ci95_low: float
    ci95_high: float
    p_value: float            # two-sided sign-flip permutation test
    a_better_fraction: float  # share of seeds where a is strictly better than b (direction-aware)


def paired_comparison(
    episodes: pd.DataFrame,
    policy_a: str,
    policy_b: str,
    metric: str,
    n_perm: int = 10000,
    seed: int = 0,
) -> PairedResult:
    """Compare two policies on identical evaluation seeds. Only seeds present for both are used."""
    a = episodes[episodes["policy"] == policy_a].set_index("seed")[metric]
    b = episodes[episodes["policy"] == policy_b].set_index("seed")[metric]
    shared = a.index.intersection(b.index)
    if shared.size == 0:
        raise ValueError(f"no shared evaluation seeds between {policy_a} and {policy_b}")
    diffs = (a.loc[shared] - b.loc[shared]).to_numpy(dtype=float)
    observed = abs(diffs.mean())
    rng = np.random.default_rng(seed)
    signs = rng.choice(np.array([-1.0, 1.0]), size=(n_perm, diffs.size))
    perm_means = np.abs((signs * diffs).mean(axis=1))
    p_value = float((np.sum(perm_means >= observed) + 1) / (n_perm + 1))
    low, high = bootstrap_ci(diffs, seed=seed)
    if metric in HIGHER_IS_BETTER:
        better = float(np.mean(diffs > 0))
    else:
        better = float(np.mean(diffs < 0))
    return PairedResult(
        policy_a=policy_a,
        policy_b=policy_b,
        metric=metric,
        n_pairs=int(diffs.size),
        mean_diff=float(diffs.mean()),
        ci95_low=low,
        ci95_high=high,
        p_value=p_value,
        a_better_fraction=better,
    )


def moving_average(values: np.ndarray, window: int) -> np.ndarray:
    """Trailing moving average. The first window-1 points average over the points available so far."""
    values = np.asarray(values, dtype=float)
    if window <= 1 or values.size == 0:
        return values.copy()
    csum = np.cumsum(np.insert(values, 0, 0.0))
    out = np.empty_like(values)
    for i in range(values.size):
        lo = max(0, i - window + 1)
        out[i] = (csum[i + 1] - csum[lo]) / (i + 1 - lo)
    return out
