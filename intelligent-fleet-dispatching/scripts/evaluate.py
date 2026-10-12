"""Evaluate the trained DQN checkpoint and the baseline policies on held-out evaluation seeds.

Run from the repository root:

    python scripts/evaluate.py                                       # uses checkpoints/dqn_default/final.pt
    python scripts/evaluate.py --checkpoint checkpoints/smoke/final.pt --quick
    python scripts/evaluate.py --policies random_valid,shortest_queue  # baselines only, no checkpoint needed

Writes one row per (policy, seed) to results/evaluation/episodes.csv (or --output).
Evaluation never trains: no transitions are stored and no weights change.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fleet_dispatch.config import ConfigError, load_config  # noqa: E402
from fleet_dispatch.training.evaluate import ALL_POLICIES, evaluate_checkpoint  # noqa: E402
from fleet_dispatch.utils.logging import configure_logging, get_logger  # noqa: E402

_log = get_logger("scripts.evaluate")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--reward", type=Path, default=None)
    parser.add_argument("--quick", action="store_true", help="apply configs/quick.yaml overrides")
    parser.add_argument("--checkpoint", type=Path, default=None,
                        help="DQN checkpoint (default: checkpoints/<training.run_name>/final.pt when 'dqn' is requested)")
    parser.add_argument("--policies", default=",".join(ALL_POLICIES),
                        help=f"comma-separated subset of {ALL_POLICIES}")
    parser.add_argument("--episodes", type=int, default=None, help="number of held-out seeds (default: evaluation.n_episodes)")
    parser.add_argument("--output", type=Path, default=None, help="CSV path (default: results/evaluation/episodes.csv)")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--log-level", default="INFO")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    configure_logging(args.log_level)
    try:
        config = load_config(args.config, args.reward, quick=args.quick)
    except ConfigError as error:
        print(f"configuration error: {error}", file=sys.stderr)
        return 2

    policies = [name.strip() for name in args.policies.split(",") if name.strip()]
    unknown = sorted(set(policies) - set(ALL_POLICIES))
    if unknown:
        print(f"unknown policies: {unknown}; choose from {ALL_POLICIES}", file=sys.stderr)
        return 2

    checkpoint = args.checkpoint
    if "dqn" in policies and checkpoint is None:
        checkpoint = config.resolve(config.paths.checkpoint_dir) / config.training.run_name / "final.pt"
    if "dqn" in policies and not checkpoint.exists():
        print(f"checkpoint not found: {checkpoint}. Train first or pass --checkpoint.", file=sys.stderr)
        return 2
    if "dqn" not in policies:
        checkpoint = None

    n_episodes = args.episodes or config.evaluation.n_episodes
    output = args.output or config.resolve(config.paths.results_dir) / "evaluation" / "episodes.csv"
    frame = evaluate_checkpoint(config, checkpoint, policies, n_episodes, output, device=args.device)
    _log.info("wrote %d rows to %s", len(frame), output)
    print(f"episodes CSV : {output}")
    print(f"rows         : {len(frame)} ({len(policies)} policies x {n_episodes} seeds)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
