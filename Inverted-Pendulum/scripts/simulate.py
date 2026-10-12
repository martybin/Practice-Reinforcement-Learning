#!/usr/bin/env python
"""Launch the real-time Pygame simulator.

Examples
--------
    python scripts/simulate.py                                   # manual keyboard control
    python scripts/simulate.py --mode baseline                   # energy-shaping + LQR controller
    python scripts/simulate.py --mode dqn --checkpoint checkpoints/combined/best.pt
    python scripts/simulate.py --mode dqn --checkpoint checkpoints/swingup/best.pt \\
                               --stabilizer-checkpoint checkpoints/stabilize/best.pt
    python scripts/simulate.py --headless --mode baseline --seconds 6 --screenshot outputs/baseline.png

Keys: LEFT/RIGHT drive (manual) | SPACE zero drive / pause | 1-4 or TAB switch mode | R reset | P pause
      N next start type | D kick the pendulum | T trail | -/= speed | S screenshot | H help | ESC exit
"""

import argparse
import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pendulum_rl.config import ConfigError, add_config_arguments, build_sim_config, load_config  # noqa: E402
from pendulum_rl.utils import CheckpointError  # noqa: E402


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_config_arguments(p)
    p.add_argument("--mode", choices=("manual", "random", "dqn", "baseline"), help="initial control mode")
    p.add_argument("--checkpoint", help="DQN checkpoint for mode 'dqn'")
    p.add_argument("--stabilizer-checkpoint", help="swing-up checkpoint hands over to this stabilizer near the top")
    p.add_argument("--start", choices=("hanging", "upright", "random"), help="initial pendulum state after reset")
    p.add_argument("--dt", type=float, help="physics timestep (s)")
    p.add_argument("--seed", type=int)
    p.add_argument("--no-render", "--headless", dest="headless", action="store_true",
                   help="no window: simulate offscreen for --seconds (optionally save a --screenshot)")
    p.add_argument("--seconds", type=float, default=8.0, help="headless run length in simulated seconds")
    p.add_argument("--screenshot", help="headless: save the last frame to this PNG path")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    try:
        overrides = list(args.overrides) + ([f"physics.dt={args.dt}"] if args.dt else [])
        cfg = load_config(args.config, overrides)
        sim = build_sim_config(cfg)
        changes = {k: v for k, v in {"mode": args.mode, "checkpoint": args.checkpoint, "start": args.start,
                                      "stabilizer_checkpoint": args.stabilizer_checkpoint, "seed": args.seed}.items()
                   if v is not None}
        sim = replace(sim, **changes)
        if sim.mode == "dqn" and not sim.checkpoint:
            raise ConfigError("mode 'dqn' needs --checkpoint <file.pt>. Train one first, e.g. "
                              "`python scripts/train.py --mode stabilize --episodes 300`, or use --mode baseline.")
        from pendulum_rl.visualization.pygame_simulator import SimulatorApp
        app = SimulatorApp.from_config(cfg, sim, headless=args.headless)
    except (ConfigError, CheckpointError, ValueError, KeyError, ImportError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    if args.headless:
        app.simulate_for(args.seconds)
        app._draw(interpolate=False)
        info = app.info
        print(f"headless run: mode={app.mode_label()} | t={info['time']:.2f}s | phase={'failed' if app.failed else app.phase} "
              f"| theta={info['theta']:+.3f} rad | x={info['x']:+.3f} m | return={app.total_reward:.1f}")
        if args.screenshot:
            Path(args.screenshot).parent.mkdir(parents=True, exist_ok=True)
            app.view.save(args.screenshot)
            print(f"saved {args.screenshot}")
        app.close()
        return 0

    app.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
