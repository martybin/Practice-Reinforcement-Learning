"""Train the masked Double DQN agent.

Run from the repository root:

    python scripts/train.py                          # full configuration (configs/default.yaml)
    python scripts/train.py --quick                  # smoke test: 3 episodes, small network and buffer
    python scripts/train.py --run-name my_run --episodes 500
    python scripts/train.py --run-name my_run --resume
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fleet_dispatch.config import ConfigError, load_config  # noqa: E402
from fleet_dispatch.metrics.visualization import plot_training_curves  # noqa: E402
from fleet_dispatch.training.train import run_training  # noqa: E402
from fleet_dispatch.utils.logging import configure_logging  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=None, help="main YAML (default: configs/default.yaml)")
    parser.add_argument("--reward", type=Path, default=None, help="reward YAML (default: configs/reward.yaml)")
    parser.add_argument("--quick", action="store_true", help="apply configs/quick.yaml overrides")
    parser.add_argument("--run-name", default=None, help="run identifier for checkpoints and logs")
    parser.add_argument("--episodes", type=int, default=None, help="override training.episodes")
    parser.add_argument("--resume", action="store_true", help="continue from the run's latest checkpoint")
    parser.add_argument("--device", default=None, choices=["auto", "cpu", "cuda"], help="override agent.device")
    parser.add_argument("--log-level", default="INFO")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    configure_logging(args.log_level)
    try:
        config = load_config(args.config, args.reward, quick=args.quick)
        if args.episodes is not None:
            config.training.episodes = args.episodes
            config.validate()
    except ConfigError as error:
        print(f"configuration error: {error}", file=sys.stderr)
        return 2

    result = run_training(config, run_name=args.run_name, resume=args.resume, device=args.device)
    figures = plot_training_curves(result.episodes_csv, result.run_dir / "figures")
    print(f"run          : {result.run_name}")
    print(f"episodes     : {result.episodes_completed}")
    print(f"episode log  : {result.episodes_csv}")
    print(f"checkpoints  : {result.checkpoint_dir}")
    for path in figures:
        print(f"figure       : {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
