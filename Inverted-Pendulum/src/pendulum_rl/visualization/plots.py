"""Matplotlib figures for training logs and evaluation benchmarks (headless-safe).

Colour follows the entity, not its rank: every controller keeps the same categorical slot in every
figure (DQN = blue, baseline = orange, random = aqua, hand-off pair = yellow). Palette values are the
dark-surface steps of the documented, validated categorical palette; grid/axes are hairlines.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg", force=False)
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

SURFACE = "#1a1a19"
INK = "#ffffff"
INK_2 = "#c3c2b7"
MUTED = "#898781"
GRID = "#2c2c2a"
AXIS = "#383835"
SERIES = ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181"]    # blue, orange, aqua, yellow, magenta
CONTROLLER_COLORS = {"dqn": SERIES[0], "baseline": SERIES[1], "random": SERIES[2], "dqn_swingup+stabilize": SERIES[3]}
STATUS_BAD = "#d03b3b"
STATUS_GOOD = "#0ca30c"


def _rc():
    return matplotlib.rc_context({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": AXIS, "axes.labelcolor": INK_2, "axes.titlecolor": INK, "axes.titlesize": 11,
        "axes.titleweight": "bold", "axes.titlelocation": "left", "axes.labelsize": 9,
        "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelsize": 8, "ytick.labelsize": 8,
        "grid.color": GRID, "grid.linewidth": 0.6, "axes.grid": True, "axes.axisbelow": True,
        "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
        "legend.labelcolor": INK_2, "legend.fontsize": 8, "text.color": INK, "font.size": 9,
        "font.family": ["DejaVu Sans"], "lines.linewidth": 1.8,
    })


def controller_color(name: str) -> str:
    return CONTROLLER_COLORS.get(name, SERIES[4])


@dataclass
class RunLogs:
    run_dir: Path
    episodes: pd.DataFrame | None
    evals: pd.DataFrame | None
    summary: dict | None
    config: str | None


def load_run(run_dir: str | Path) -> RunLogs:
    """Read the CSV/JSON logs written by the trainer. Missing files give ``None`` fields (never fake data)."""
    d = Path(run_dir)

    def read(name):
        p = d / name
        return pd.read_csv(p) if p.is_file() and p.stat().st_size > 0 else None

    summary_p = d / "summary.json"
    cfg_p = d / "run_config.yaml"
    return RunLogs(d, read("episodes.csv"), read("eval.csv"),
                   json.loads(summary_p.read_text(encoding="utf-8")) if summary_p.is_file() else None,
                   cfg_p.read_text(encoding="utf-8") if cfg_p.is_file() else None)


def list_runs(log_root: str | Path) -> list[Path]:
    root = Path(log_root)
    return sorted(p for p in root.iterdir() if (p / "episodes.csv").is_file()) if root.is_dir() else []


def moving_average(values, window: int) -> np.ndarray:
    v = np.asarray(values, dtype=float)
    if len(v) == 0:
        return v
    w = max(1, min(window, len(v)))
    kernel = np.ones(w)
    num = np.convolve(np.nan_to_num(v), kernel, mode="full")[: len(v)]
    den = np.convolve(~np.isnan(v), kernel, mode="full")[: len(v)]
    return np.where(den > 0, num / np.maximum(den, 1), np.nan)


def _line(ax, x, y, color, label=None, raw=None, alpha_raw=0.28):
    if raw is not None:
        ax.plot(x, raw, color=color, alpha=alpha_raw, linewidth=0.8)
    ax.plot(x, y, color=color, label=label)


def training_figure(episodes: pd.DataFrame, evals: pd.DataFrame | None = None, window: int = 20,
                    title: str = "Training progress") -> plt.Figure:
    """Twelve panels covering every metric the project logs."""
    ep = episodes
    x = ep["episode"].to_numpy()
    with _rc():
        fig, axes = plt.subplots(3, 4, figsize=(17, 9.5), constrained_layout=True)
        ax = axes.ravel()
        blue, orange, aqua = SERIES[0], SERIES[1], SERIES[2]

        _line(ax[0], x, moving_average(ep["return"], window), blue, raw=ep["return"])
        ax[0].set_title("Episode reward"); ax[0].set_xlabel("episode"); ax[0].set_ylabel("return (thin = raw)")

        _line(ax[1], x, moving_average(ep["length"], window), blue, raw=ep["length"])
        ax[1].set_title("Episode length"); ax[1].set_xlabel("episode"); ax[1].set_ylabel("control steps")

        ax[2].plot(x, moving_average(ep["success"], window), color=blue)
        ax[2].set_ylim(-0.02, 1.02); ax[2].set_title(f"Success rate (moving avg, {window} ep)")
        ax[2].set_xlabel("episode"); ax[2].set_ylabel("fraction of episodes")

        ax[3].plot(x, moving_average(ep["swing_up_success"], window), color=orange)
        ax[3].set_ylim(-0.02, 1.02); ax[3].set_title("Swing-up success rate")
        ax[3].set_xlabel("episode"); ax[3].set_ylabel("reached upright region")

        _line(ax[4], x, moving_average(ep["longest_balance"], window), aqua, raw=ep["longest_balance"])
        ax[4].set_title("Balance duration"); ax[4].set_xlabel("episode"); ax[4].set_ylabel("longest continuous balance [s]")

        err_deg = np.degrees(ep["mean_abs_angle_error"].to_numpy())
        _line(ax[5], x, moving_average(err_deg, window), orange, raw=err_deg)
        ax[5].set_title("Average angular error"); ax[5].set_xlabel("episode"); ax[5].set_ylabel("mean |theta| [deg]")

        ax[6].plot(x, ep["epsilon"], color=blue)
        ax[6].set_ylim(0, 1.02); ax[6].set_title("Exploration epsilon"); ax[6].set_xlabel("episode")

        loss = ep["loss"].to_numpy(dtype=float)
        if np.isfinite(loss).any():
            ax[7].plot(x, moving_average(loss, window), color=orange)
            ax[7].set_yscale("log")
        ax[7].set_title("TD loss (Huber)"); ax[7].set_xlabel("episode")

        for col, color, lab in (("q_max", SERIES[1], "max"), ("q_mean", SERIES[0], "mean"), ("q_min", SERIES[2], "min")):
            ax[8].plot(x, moving_average(ep[col], window), color=color, label=lab)
        ax[8].set_title("Q-value statistics (batch)"); ax[8].set_xlabel("episode"); ax[8].legend(loc="best")

        frac_cols = sorted([c for c in ep.columns if c.startswith("action_frac_")], key=lambda c: int(c.rsplit("_", 1)[1]))
        if frac_cols:
            ys = [moving_average(ep[c], window) for c in frac_cols]
            ax[9].stackplot(x, ys, colors=SERIES[: len(ys)], labels=[f"action {i}" for i in range(len(ys))],
                            edgecolor=SURFACE, linewidth=1.0)
            ax[9].set_ylim(0, 1); ax[9].legend(loc="lower center", ncol=min(3, len(ys)), frameon=True, facecolor=SURFACE, framealpha=0.9, edgecolor=AXIS)
        ax[9].set_title("Action distribution"); ax[9].set_xlabel("episode"); ax[9].set_ylabel("share of steps")

        if evals is not None and len(evals):
            ax[10].plot(evals["env_steps"], evals["swing_up_rate"], color=orange, marker="o", ms=3, label="swing-up")
            ax[10].plot(evals["env_steps"], evals["balance_rate"], color=blue, marker="o", ms=3, label="balanced")
            ax[10].plot(evals["env_steps"], evals["violation_rate"], color=STATUS_BAD, marker="o", ms=3, label="violation")
            ax[10].set_ylim(-0.02, 1.02); ax[10].legend(loc="best")
            ax[10].set_xlabel("environment steps")
        else:
            ax[10].text(0.5, 0.5, "no greedy evaluations logged yet", ha="center", va="center", color=MUTED, transform=ax[10].transAxes)
        ax[10].set_title("Greedy evaluation (fixed seeds)")

        reasons = ep["termination_reason"].fillna("none").value_counts()
        ax[11].barh(reasons.index[::-1], reasons.to_numpy()[::-1], color=blue, edgecolor=SURFACE, linewidth=1.5)
        ax[11].set_title("How episodes ended"); ax[11].set_xlabel("episodes")
        ax[11].grid(axis="y", visible=False)

        fig.suptitle(title, x=0.01, ha="left", fontsize=14, fontweight="bold", color=INK)
    return fig


def comparison_figure(summaries: dict[str, dict], scenario: str) -> plt.Figure:
    """Controller comparison: success rates, violation rate, speed and effort."""
    names = list(summaries)
    with _rc():
        fig, axes = plt.subplots(1, 4, figsize=(16, 4.9), constrained_layout=True)
        # (a) rates, grouped bars
        metrics = [("swing_up_rate", "swing-up"), ("balance_rate", "balanced"), ("violation_rate", "violation")]
        width = 0.8 / max(1, len(names))
        for i, name in enumerate(names):
            vals = [summaries[name][m] for m, _ in metrics]
            pos = np.arange(len(metrics)) + i * width - 0.4 + width / 2
            axes[0].bar(pos, vals, width=width * 0.88, color=controller_color(name), label=name, edgecolor=SURFACE, linewidth=1.5)
            for p, v in zip(pos, vals):
                axes[0].text(p, v + 0.02, f"{v * 100:.0f}%", ha="center", va="bottom", fontsize=7.5, color=INK_2)
        axes[0].set_xticks(np.arange(len(metrics)), [l for _, l in metrics]); axes[0].set_ylim(0, 1.18)
        axes[0].set_title(f"Episode outcomes ({scenario})"); axes[0].set_ylabel("fraction of episodes")
        axes[0].grid(axis="x", visible=False)
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="outside upper right", ncol=len(labels))

        def simple_bar(ax, key, title, ylabel, scale=1.0):
            vals = [summaries[n][key] * scale for n in names]
            vals = [0 if (isinstance(v, float) and math.isnan(v)) else v for v in vals]
            ax.bar(names, vals, color=[controller_color(n) for n in names], edgecolor=SURFACE, linewidth=1.5, width=0.6)
            for i, v in enumerate(vals):
                label = "n/a" if (isinstance(summaries[names[i]][key], float) and math.isnan(summaries[names[i]][key])) else f"{v:.2f}"
                ax.text(i, v, label, ha="center", va="bottom", fontsize=8, color=INK_2)
            ax.set_title(title); ax.set_ylabel(ylabel); ax.grid(axis="x", visible=False)
            ax.tick_params(axis="x", labelsize=8, labelrotation=20)

        simple_bar(axes[1], "mean_time_to_upright", "Time to reach upright", "seconds (episodes that got there)")
        simple_bar(axes[2], "mean_abs_angle_error", "Mean angular error", "degrees", scale=180 / math.pi)
        simple_bar(axes[3], "mean_effort_abs", "Actuator effort", "integral |u| dt  [s]")
    return fig


def trajectory_figure(trajs: dict[str, dict], title: str = "Example episode") -> plt.Figure:
    """Overlay one trajectory per controller (same initial condition)."""
    with _rc():
        fig, axes = plt.subplots(2, 2, figsize=(12, 7), constrained_layout=True, sharex=True)
        panels = [("theta", "Pendulum angle [deg]", 180 / math.pi), ("x", "Support position [m]", 1.0),
                  ("command", "Drive command u (0.2 s mean)", 1.0), ("energy", "Normalized swing energy (1 = upright)", 1.0)]
        for ax, (key, label, k) in zip(axes.ravel(), panels):
            for name, tr in trajs.items():
                y = tr[key] * k
                if key == "command":    # discrete policies chatter between levels at the control rate: show the average drive
                    y = moving_average(y, 10)
                ax.plot(tr["t"], y, color=controller_color(name), label=name)
            ax.set_ylabel(label)
        axes[0, 0].axhline(0, color=MUTED, linewidth=0.8, linestyle=(0, (4, 3)))
        axes[0, 0].text(0.99, 0.04, "upright = 0", transform=axes[0, 0].transAxes, ha="right", color=MUTED, fontsize=8)
        axes[0, 1].legend(loc="best")
        axes[1, 0].set_xlabel("time [s]"); axes[1, 1].set_xlabel("time [s]")
        fig.suptitle(title, x=0.01, ha="left", fontsize=13, fontweight="bold", color=INK)
    return fig


def save_figure(fig: plt.Figure, path: str | Path, dpi: int = 130) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    return path


def main(argv=None) -> int:
    """``python -m pendulum_rl.visualization.plots --run logs/combined`` -> training.png next to the logs."""
    import argparse
    parser = argparse.ArgumentParser(description="Plot training metrics of a finished or running training run")
    parser.add_argument("--run", required=True, help="log directory, e.g. logs/combined")
    parser.add_argument("--out", default=None, help="output PNG (default: <run>/training.png)")
    parser.add_argument("--window", type=int, default=20)
    args = parser.parse_args(argv)
    logs = load_run(args.run)
    if logs.episodes is None or logs.episodes.empty:
        print(f"No training data in '{args.run}' (episodes.csv missing or empty). Run scripts/train.py first.")
        return 1
    out = Path(args.out) if args.out else Path(args.run) / "training.png"
    save_figure(training_figure(logs.episodes, logs.evals, args.window, title=f"Training progress - {Path(args.run).name}"), out)
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
