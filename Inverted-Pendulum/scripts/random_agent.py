#!/usr/bin/env python
"""Run a uniformly random policy through the Gymnasium environment (API demo and sanity check).

    python scripts/random_agent.py --episodes 5
    python scripts/random_agent.py --episodes 2 --render        # watch it in a window (real time)
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402

from pendulum_rl.config import ConfigError, add_config_arguments, build_env_config, build_physics, build_reward, load_config  # noqa: E402
from pendulum_rl.environment import PendulumEnv  # noqa: E402


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_config_arguments(p)
    p.add_argument("--episodes", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--dt", type=float, help="physics timestep (s)")
    p.add_argument("--render", action="store_true", help="open a window (default: no rendering)")
    args = p.parse_args(argv)
    try:
        cfg = load_config(args.config, list(args.overrides) + ([f"physics.dt={args.dt}"] if args.dt else []))
        env = PendulumEnv(build_env_config(cfg), build_physics(cfg), build_reward(cfg),
                          render_mode="human" if args.render else None)
    except (ConfigError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    returns = []
    for ep in range(args.episodes):
        obs, info = env.reset(seed=args.seed + ep)
        total, steps = 0.0, 0
        while True:
            obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
            total += reward
            steps += 1
            if terminated or truncated:
                break
        m = info["episode_metrics"]
        returns.append(total)
        print(f"episode {ep}: steps {steps:4d} | return {total:8.2f} | ended: {m['termination_reason']:<18s} | "
              f"reached upright: {m['swing_up_success']} | mean |theta| {np.degrees(m['mean_abs_angle_error']):.0f} deg")
    print(f"mean return over {args.episodes} episodes: {np.mean(returns):.2f}")
    env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
