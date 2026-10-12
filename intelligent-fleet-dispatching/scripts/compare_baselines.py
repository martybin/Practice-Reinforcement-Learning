"""Turn evaluation episodes into summary tables, paired statistics, and figures.

Run from the repository root after scripts/evaluate.py:

    python scripts/compare_baselines.py
    python scripts/compare_baselines.py --episodes results/evaluation/episodes.csv --figures docs/figures

Outputs
  results/evaluation/summary.csv               mean, std, and bootstrap 95% CI per (policy, metric)
  results/evaluation/paired_comparisons.csv    DQN vs each baseline on identical seeds (permutation p-value)
  results/evaluation/summary.md                markdown tables, ready to paste into the README
  <figures dir>/*.png                          box plots and bar charts generated from these episodes
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd  # noqa: E402

from fleet_dispatch.config import load_config  # noqa: E402
from fleet_dispatch.metrics.operational_metrics import paired_comparison, summarize  # noqa: E402
from fleet_dispatch.metrics.visualization import (  # noqa: E402
    plot_policy_comparison,
    plot_summary_bars,
)
from fleet_dispatch.utils.logging import configure_logging, get_logger  # noqa: E402

_log = get_logger("scripts.compare_baselines")

HEADLINE_METRICS = (
    "payload_t",
    "avg_wait_min",
    "truck_travel_min",
    "truck_utilization",
    "shovel_utilization",
    "shovel_idle_min",
    "shovel_starvation_min",
    "unnecessary_travel_min",
    "truck_breakdowns",
    "invalid_actions",
    "episode_return",
    "runtime_s",
)
PAIRED_METRICS = ("episode_return", "payload_t", "avg_wait_min", "shovel_starvation_min", "truck_breakdowns")


def _markdown_table(summary: pd.DataFrame, policies: list[str]) -> str:
    lines = ["| Metric | " + " | ".join(policies) + " |", "|---|" + "---|" * len(policies)]
    for metric in HEADLINE_METRICS:
        cells = []
        for policy in policies:
            row = summary[(summary["policy"] == policy) & (summary["metric"] == metric)]
            if row.empty:
                cells.append("n/a")
                continue
            r = row.iloc[0]
            cells.append(f"{r['mean']:.2f} ± {r['std']:.2f}")
        lines.append(f"| {metric} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _paired_markdown(pairs: pd.DataFrame) -> str:
    lines = [
        "| Metric | Comparison | Mean diff (A - B) | 95% CI | p (perm.) | A better on |",
        "|---|---|---|---|---|---|",
    ]
    for _, r in pairs.iterrows():
        lines.append(
            f"| {r['metric']} | {r['policy_a']} vs {r['policy_b']} | {r['mean_diff']:.2f} | "
            f"[{r['ci95_low']:.2f}, {r['ci95_high']:.2f}] | {r['p_value']:.4f} | {r['a_better_fraction']:.0%} |"
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--episodes", type=Path, default=None, help="episodes CSV from scripts/evaluate.py")
    parser.add_argument("--figures", type=Path, default=None, help="figure output directory (default: results/figures)")
    parser.add_argument("--reference", default="dqn", help="policy used as A in the paired comparisons")
    parser.add_argument("--n-perm", type=int, default=10000)
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)
    configure_logging(args.log_level)

    config = load_config(args.config)
    episodes_path = args.episodes or config.resolve(config.paths.results_dir) / "evaluation" / "episodes.csv"
    if not episodes_path.exists():
        print(f"episodes CSV not found: {episodes_path}. Run scripts/evaluate.py first.", file=sys.stderr)
        return 2
    episodes = pd.read_csv(episodes_path)
    out_dir = episodes_path.parent
    figures_dir = args.figures or config.resolve(config.paths.results_dir) / "figures"

    policies = list(dict.fromkeys(episodes["policy"]))
    summary = summarize(episodes)
    summary.to_csv(out_dir / "summary.csv", index=False)

    baselines = [p for p in policies if p != args.reference]
    pair_rows = []
    if args.reference in policies:
        for baseline in baselines:
            for metric in PAIRED_METRICS:
                result = paired_comparison(episodes, args.reference, baseline, metric, n_perm=args.n_perm)
                pair_rows.append(result.__dict__)
    pairs = pd.DataFrame(pair_rows)
    pairs.to_csv(out_dir / "paired_comparisons.csv", index=False)

    markdown = [
        f"## Held-out evaluation ({episodes['seed'].nunique()} shared seeds, mean ± std)",
        "",
        _markdown_table(summary, policies),
    ]
    if not pairs.empty:
        markdown += ["", "## Paired comparisons (identical seeds)", "", _paired_markdown(pairs)]
    (out_dir / "summary.md").write_text("\n".join(markdown) + "\n", encoding="utf-8")

    figures = plot_policy_comparison(episodes, figures_dir) + plot_summary_bars(summary, figures_dir)
    print("\n".join(markdown))
    print()
    print(f"summary      : {out_dir / 'summary.csv'}")
    print(f"paired stats : {out_dir / 'paired_comparisons.csv'}")
    for path in figures:
        print(f"figure       : {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
