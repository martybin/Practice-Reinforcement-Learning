#!/usr/bin/env python
"""Train a DQN policy.

Examples
--------
    python scripts/train.py --mode stabilize --episodes 300      # Stage A: balance near upright
    python scripts/train.py --mode swingup   --episodes 1500     # Stage B: swing up from hanging
    python scripts/train.py --mode combined  --episodes 3000     # Stage C: both, randomized physics
    python scripts/train.py --mode curriculum                    # A -> B -> C with promotion

Logs go to logs/<run-name>/, checkpoints to checkpoints/<run-name>/ (best.pt, final.pt).
Existing runs are never overwritten unless --overwrite is given.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pendulum_rl.config import ConfigError, add_config_arguments, load_config  # noqa: E402
from pendulum_rl.training import MODES, trainer_from_config  # noqa: E402


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_config_arguments(p)
    p.add_argument("--mode", choices=MODES, help="stabilize | swingup | combined | curriculum (default from config)")
    p.add_argument("--episodes", type=int, help="training episodes (per stage in curriculum mode)")
    p.add_argument("--seed", type=int, help="random seed")
    p.add_argument("--device", help="auto | cpu | cuda | cuda:0 (default: auto = CUDA when available)")
    p.add_argument("--dt", type=float, help="physics integration timestep in seconds (must divide env.control_dt)")
    p.add_argument("--run-name", help="name of the run directory (default: the mode)")
    p.add_argument("--init-from", metavar="CKPT", help="warm-start network weights from a checkpoint")
    p.add_argument("--overwrite", action="store_true", help="replace an existing run with the same name")
    p.add_argument("--tensorboard", action="store_true", help="also write TensorBoard event files")
    p.add_argument("--render", action="store_true", help="show training episodes in a window (slow: real-time)")
    p.add_argument("--eval-every", type=int, help="episodes between greedy evaluations")
    p.add_argument("--eval-episodes", type=int, help="episodes per greedy evaluation")
    p.add_argument("--print-every", type=int, help="print a progress line every N episodes")
    p.add_argument("--snapshot-every", type=int, help="save the network every N episodes (0 = off); the dashboard's "
                                                      "learning arena replays these snapshots")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    overrides = list(args.overrides)
    if args.dt is not None:
        overrides.append(f"physics.dt={args.dt}")
    try:
        cfg = load_config(args.config, overrides)
        train_overrides = {k: v for k, v in {
            "mode": args.mode, "episodes": args.episodes, "seed": args.seed, "device": args.device,
            "run_name": args.run_name, "init_from": args.init_from, "eval_every": args.eval_every,
            "eval_episodes": args.eval_episodes, "print_every": args.print_every,
            "snapshot_every": args.snapshot_every}.items() if v is not None}
        if args.overwrite:
            train_overrides["overwrite"] = True
        if args.tensorboard:
            train_overrides["tensorboard"] = True
        if args.render:
            train_overrides["render"] = True
        trainer = trainer_from_config(cfg, **train_overrides)
    except (ConfigError, FileExistsError, FileNotFoundError, ValueError, KeyError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    print(f"run '{trainer.cfg.name}' | mode={trainer.cfg.mode} | device={trainer.agent.device} | seed={trainer.cfg.seed}")
    print(f"logs:        {trainer.log_dir}\ncheckpoints: {trainer.ckpt_dir}")
    summary = trainer.train()
    print("\n=== training finished ===")
    print(f"episodes {summary['episodes']}, env steps {summary['env_steps']}, updates {summary['updates']}, "
          f"{summary['wall_time_s']:.0f}s, interrupted={summary['interrupted']}")
    fe = summary["final_eval"]
    print(f"final greedy evaluation ({fe['scenario']}, {fe['episodes']} fixed-seed episodes): "
          f"swing-up {fe['swing_up_rate']:.0%}, balanced {fe['balance_rate']:.0%}, violations {fe['violation_rate']:.0%}")
    print(f"status label: {summary['status'].upper()}  (primary metric '{fe['primary_metric']}' = {fe['primary_rate']:.0%}; "
          f"'trained' requires >= 80%)")
    print(f"\nnext: python scripts/evaluate.py --checkpoint {trainer.ckpt_dir / 'best.pt'} --scenario {fe['scenario']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
