"""Publication-style figures generated from the raw CSV outputs of real runs.

Training curves plot the raw per-episode values in light colour, with a trailing moving average on top.
The raw data is never replaced by the smoothed version.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # figures are written to disk; no display is needed
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from fleet_dispatch.metrics.operational_metrics import moving_average  # noqa: E402

PALETTE = {
    "dqn": "#1f77b4",
    "random_valid": "#7f7f7f",
    "nearest_eligible": "#ff7f0e",
    "shortest_queue": "#2ca02c",
}
_POLICY_LABELS = {
    "dqn": "DQN (masked Double)",
    "random_valid": "Random valid",
    "nearest_eligible": "Nearest eligible",
    "shortest_queue": "Shortest queue / earliest availability",
}


def _style() -> None:
    plt.rcParams.update({
        "figure.dpi": 110,
        "savefig.dpi": 160,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "font.size": 9,
    })


def plot_training_curves(episodes_csv: Path, out_dir: Path, window: int = 20) -> list[Path]:
    """Return, loss, epsilon, payload, waiting, utilisation, breakdowns, and reward components against episode."""
    _style()
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(episodes_csv)
    if df.empty:
        raise ValueError(f"{episodes_csv} has no episode rows")
    episodes = df["episode"].to_numpy()

    panels = [
        ("episode_return", "Episode return", True),
        ("mean_loss", "Training loss (Huber)", False),
        ("epsilon", "Exploration epsilon", False),
        ("payload_t", "Payload delivered per shift (t)", True),
        ("avg_wait_min", "Average truck waiting time (min)", True),
        ("shovel_utilization", "Shovel utilisation", True),
        ("truck_breakdowns", "Truck breakdowns per shift", True),
    ]
    fig, axes = plt.subplots(3, 3, figsize=(12, 9))
    flat = axes.ravel()
    written: list[Path] = []
    for ax, (column, title, smooth) in zip(flat, panels):
        values = df[column].to_numpy(dtype=float)
        mask = ~np.isnan(values)
        ax.plot(episodes[mask], values[mask], color="#9ecae1", linewidth=0.8, label="raw episode")
        if smooth and mask.sum() > 1:
            ax.plot(episodes[mask], moving_average(values[mask], window), color="#08519c", linewidth=1.8,
                    label=f"moving average ({window})")
            ax.legend(frameon=False, fontsize=8)
        ax.set_title(title)
        ax.set_xlabel("training episode")
    ax = flat[len(panels)]
    components = [c for c in df.columns if c.startswith("r_")]
    for component in components:
        ax.plot(episodes, moving_average(df[component].to_numpy(dtype=float), window), label=component[2:])
    ax.set_title("Reward components (moving average)")
    ax.set_xlabel("training episode")
    ax.legend(frameon=False, fontsize=7)
    flat[len(panels) + 1].axis("off")
    fig.tight_layout()
    path = out_dir / "training_curves.png"
    fig.savefig(path)
    plt.close(fig)
    written.append(path)
    return written


def plot_policy_comparison(episodes: pd.DataFrame, out_dir: Path) -> list[Path]:
    """Box plots of per-seed outcomes for each policy, using identical evaluation seeds."""
    _style()
    out_dir.mkdir(parents=True, exist_ok=True)
    policies = list(dict.fromkeys(episodes["policy"]))
    metrics = [
        ("episode_return", "Episode return"),
        ("payload_t", "Payload delivered (t)"),
        ("avg_wait_min", "Average truck wait (min)"),
        ("shovel_starvation_min", "Shovel starvation (min)"),
        ("unnecessary_travel_min", "Unnecessary travel (min)"),
        ("truck_breakdowns", "Truck breakdowns"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(13, 7.5))
    for ax, (metric, title) in zip(axes.ravel(), metrics):
        data = [episodes.loc[episodes["policy"] == p, metric].to_numpy(dtype=float) for p in policies]
        box = ax.boxplot(data, patch_artist=True, widths=0.6, showfliers=False)
        for patch, policy in zip(box["boxes"], policies):
            patch.set_facecolor(PALETTE.get(policy, "#cccccc"))
            patch.set_alpha(0.6)
        for i, values in enumerate(data, start=1):
            jitter = np.random.default_rng(i).uniform(-0.12, 0.12, size=values.size)
            ax.scatter(np.full(values.size, i) + jitter, values, s=8, color="black", alpha=0.5, zorder=3)
        ax.set_xticks(range(1, len(policies) + 1))
        ax.set_xticklabels([p.replace("_", "\n") for p in policies], fontsize=8)
        ax.set_title(title)
    fig.suptitle("Per-seed outcomes on held-out evaluation shifts (identical seeds for every policy)")
    fig.tight_layout()
    path = out_dir / "policy_comparison_boxplots.png"
    fig.savefig(path)
    plt.close(fig)
    return [path]


def plot_summary_bars(summary: pd.DataFrame, out_dir: Path) -> list[Path]:
    """Mean with 95% bootstrap confidence interval for the headline metrics."""
    _style()
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics = [
        ("episode_return", "Episode return"),
        ("payload_t", "Payload delivered per shift (t)"),
        ("avg_wait_min", "Average truck wait (min)"),
        ("shovel_utilization", "Shovel utilisation"),
    ]
    fig, axes = plt.subplots(1, len(metrics), figsize=(14, 3.8))
    for ax, (metric, title) in zip(axes, metrics):
        rows = summary[summary["metric"] == metric]
        policies = rows["policy"].tolist()
        means = rows["mean"].to_numpy()
        low = means - rows["ci95_low"].to_numpy()
        high = rows["ci95_high"].to_numpy() - means
        colors = [PALETTE.get(p, "#cccccc") for p in policies]
        ax.bar(range(len(policies)), means, yerr=[low, high], color=colors, capsize=4, alpha=0.85)
        ax.set_xticks(range(len(policies)))
        ax.set_xticklabels([p.replace("_", "\n") for p in policies], fontsize=8)
        ax.set_title(title)
    fig.tight_layout()
    path = out_dir / "policy_summary_bars.png"
    fig.savefig(path)
    plt.close(fig)
    return [path]
