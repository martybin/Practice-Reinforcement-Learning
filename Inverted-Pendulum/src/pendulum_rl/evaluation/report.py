"""CSV, plot and Markdown report generation for evaluation results (only real, measured numbers)."""

from __future__ import annotations

import csv
import math
from pathlib import Path

from .evaluate import PRIMARY_METRIC, TRAINED_THRESHOLD


def _pct(x: float) -> str:
    return f"{100 * x:.0f}%"


def _num(x: float, fmt: str = "{:.2f}") -> str:
    return "n/a" if x is None or (isinstance(x, float) and math.isnan(x)) else fmt.format(x)


def write_episode_csv(path: Path, rows: list[dict]) -> None:
    keys = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def write_summary_csv(path: Path, results: dict[str, dict]) -> None:
    cols = ["controller", "status", "scenario", "episodes", "swing_up_rate", "balance_rate", "violation_rate",
            "mean_abs_angle_error_deg", "max_abs_angle_error_deg", "mean_time_to_upright_s", "median_time_to_upright_s",
            "mean_balanced_time_s", "mean_longest_balance_s", "mean_effort_abs", "mean_effort_sq", "mean_return"]
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(cols)
        for name, res in results.items():
            s = res["summary"]
            writer.writerow([name, res.get("status", ""), s["scenario"], s["episodes"], s["swing_up_rate"],
                             s["balance_rate"], s["violation_rate"], math.degrees(s["mean_abs_angle_error"]),
                             math.degrees(s["max_abs_angle_error"]), s["mean_time_to_upright"],
                             s["median_time_to_upright"], s["mean_balanced_time"], s["mean_longest_balance"],
                             s["mean_effort_abs"], s["mean_effort_sq"], s["mean_return"]])


def build_report_markdown(results: dict[str, dict], scenario: str, seed: int, meta: dict, figures: list[str]) -> str:
    primary = PRIMARY_METRIC[scenario]
    lines = [
        f"# Evaluation report - scenario `{scenario}`", "",
        f"* Episodes per controller: **{next(iter(results.values()))['summary']['episodes']}** "
        f"(seeds {seed} .. {seed + next(iter(results.values()))['summary']['episodes'] - 1}, identical for every controller)",
        f"* Primary metric for this scenario: **{primary}** (label thresholds: *trained* needs >= {_pct(TRAINED_THRESHOLD)})",
    ]
    for key, value in meta.items():
        lines.append(f"* {key}: {value}")
    lines += ["", "Success definitions: *swing-up* = pendulum entered the upright region (angle error <= 0.35 rad); "
              "*balanced* = held angle error <= 0.1 rad and angular velocity <= 1 rad/s continuously for >= 2 s without a "
              "constraint violation; *violation* = rail end, speed limit or numerical limit reached. "
              "*Actuator effort* = time integral of the absolute normalized command [s].", "",
              "## Results", "",
              "| Controller | Status label | Swing-up | Balanced | Violations | Mean angle error | Time to upright (mean) | Actuator effort |",
              "|---|---|---|---|---|---|---|---|"]
    for name, res in results.items():
        s = res["summary"]
        lines.append(f"| {name} | {res.get('status', '-')} | {_pct(s['swing_up_rate'])} | {_pct(s['balance_rate'])} | "
                     f"{_pct(s['violation_rate'])} | {_num(math.degrees(s['mean_abs_angle_error']), '{:.1f}')} deg | "
                     f"{_num(s['mean_time_to_upright'])} s | {_num(s['mean_effort_abs'])} |")
    lines += ["", "More statistics (maximum error, balance durations, violation reasons) are in `summary.csv`; "
              "per-episode data in `episodes_<controller>.csv`.", "", "## Reading the results", ""]
    for name, res in results.items():
        s = res["summary"]
        rate = s[primary]
        status = res.get("status")
        text = f"* **{name}**: {_pct(rate)} of episodes met the primary criterion ({primary})"
        if status:
            text += f" -> labelled **{status}**"
        if s["violations_by_reason"]:
            text += f"; violations: {s['violations_by_reason']}"
        lines.append(text + ".")
    dqn_like = [n for n in results if n.startswith("dqn")]
    base = results.get("baseline")
    if dqn_like and base:
        n_eps = base["summary"]["episodes"]
        for n in dqn_like:
            d, b = results[n]["summary"][primary], base["summary"][primary]
            se = math.sqrt(d * (1 - d) / n_eps + b * (1 - b) / n_eps)      # binomial standard error of the difference
            if abs(d - b) <= 2 * se or abs(d - b) < 1e-9:
                verdict = "is statistically indistinguishable from"
            else:
                verdict = "is better than" if d > b else "is worse than"
            lines.append(f"* The learned policy `{n}` {verdict} the model-based baseline on `{primary}` "
                         f"({_pct(d)} vs {_pct(b)}; difference {100 * (d - b):+.0f} points, 2 standard errors = {200 * se:.0f} points).")
    lines.append(f"* Sampling uncertainty: with {next(iter(results.values()))['summary']['episodes']} episodes a success rate has a binomial "
                 f"standard error of up to {100 * 0.5 / math.sqrt(next(iter(results.values()))['summary']['episodes']):.0f} percentage points; "
                 f"this is one training run and one set of initial conditions, so small differences are not conclusive.")
    if figures:
        lines += ["", "## Figures", ""] + [f"![{Path(f).stem}]({f})" for f in figures]
    lines += ["", "_Generated by `scripts/evaluate.py`; every number above was measured in this run._", ""]
    return "\n".join(lines)
