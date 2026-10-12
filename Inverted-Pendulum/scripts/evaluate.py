#!/usr/bin/env python
"""Evaluate controllers on fixed-seed episodes and write CSV logs, plots and a Markdown report.

Examples
--------
    python scripts/evaluate.py --checkpoint checkpoints/combined/best.pt --scenario combined --episodes 50
    python scripts/evaluate.py --checkpoint checkpoints/swingup/best.pt \\
           --stabilizer-checkpoint checkpoints/stabilize/best.pt --scenario swingup
    python scripts/evaluate.py --scenario swingup --untrained-dqn      # baseline + random + untrained network

The model-based baseline (energy shaping + LQR) and a random policy are evaluated on the very same
seeds (hence the same initial states) as the DQN policy so the numbers are directly comparable.
"""

import argparse
import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pendulum_rl.agents import DQNAgent  # noqa: E402
from pendulum_rl.config import (  # noqa: E402
    PROJECT_ROOT, ConfigError, add_config_arguments, build_baseline_config, build_dqn_config, build_env_config,
    build_physics, build_reward, load_config,
)
from pendulum_rl.control import EnergyShapingLQRController  # noqa: E402
from pendulum_rl.environment import EnvConfig, RewardConfig  # noqa: E402
from pendulum_rl.evaluation import (  # noqa: E402
    SCENARIOS, BaselinePolicy, DQNPolicy, RandomPolicy, SwitchingPolicy, classify_status, evaluate_policy,
    make_eval_env, run_episode, summarize,
)
from pendulum_rl.evaluation.report import (  # noqa: E402
    build_report_markdown, write_episode_csv, write_summary_csv,
)
from pendulum_rl.physics import PhysicsParams  # noqa: E402
from pendulum_rl.utils import CheckpointError  # noqa: E402
from pendulum_rl.visualization import plots  # noqa: E402


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_config_arguments(p)
    p.add_argument("--checkpoint", help="DQN checkpoint (.pt) to evaluate")
    p.add_argument("--stabilizer-checkpoint", help="if given, also evaluate --checkpoint (a swing-up policy) handing over to this stabilizer near the top")
    p.add_argument("--scenario", choices=sorted(SCENARIOS), help="stabilize | swingup | combined (default: from checkpoint)")
    p.add_argument("--episodes", type=int, default=50, help="episodes per controller (default 50)")
    p.add_argument("--seed", type=int, default=10_000, help="first episode seed (default 10000)")
    p.add_argument("--device", default="cpu", help="cpu | cuda | auto (default cpu)")
    p.add_argument("--dt", type=float, help="physics timestep override")
    p.add_argument("--no-baseline", action="store_true", help="skip the model-based baseline")
    p.add_argument("--no-random", action="store_true", help="skip the random policy")
    p.add_argument("--untrained-dqn", action="store_true", help="also evaluate a randomly initialised network (label: untrained)")
    p.add_argument("--randomize", type=float, default=0.0, help="physics randomization (+- fraction) for a robustness test")
    p.add_argument("--disturbance-prob", type=float, default=0.0, help="per-step probability of a random velocity kick")
    p.add_argument("--out", help="output directory (default: outputs/eval_<scenario>)")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    try:
        overrides = list(args.overrides) + ([f"physics.dt={args.dt}"] if args.dt else [])
        cfg = load_config(args.config, overrides)
        physics, env_cfg, reward = build_physics(cfg), build_env_config(cfg), build_reward(cfg)
        baseline_cfg = build_baseline_config(cfg)
        meta, train_steps, scenario = {}, 0, args.scenario
        agent = stabilizer = None
        if args.checkpoint:
            agent, meta = DQNAgent.from_checkpoint(_path(args.checkpoint), device=args.device)
            if meta.get("physics"):     # evaluate in the world the policy was trained in
                physics, reward = PhysicsParams(**meta["physics"]), RewardConfig(**meta["reward"])
                env_cfg = EnvConfig(**meta["base_env"])
            train_steps = int(meta.get("training", {}).get("env_steps", agent.steps))
            scenario = scenario or {"stabilize": "stabilize", "swingup": "swingup"}.get(
                meta.get("training", {}).get("mode", ""), "combined")
            if args.stabilizer_checkpoint:
                stabilizer, _ = DQNAgent.from_checkpoint(_path(args.stabilizer_checkpoint), device=args.device)
        scenario = scenario or "combined"
    except (ConfigError, CheckpointError, ValueError, KeyError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    try:
        out_dir = Path(args.out) if args.out else PROJECT_ROOT / "outputs" / f"eval_{scenario}"
        out_dir.mkdir(parents=True, exist_ok=True)
        over = dict(physics_randomization=args.randomize, disturbance_probability=args.disturbance_prob)

        entries = []   # (name, policy, status_label_or_None)
        if agent is not None:
            entries.append(("dqn", DQNPolicy(agent), None))
            if stabilizer is not None:      # also score the two-specialist pipeline: swing-up policy -> stabilizer
                entries.append(("dqn_swingup+stabilize", SwitchingPolicy(agent, stabilizer), None))
        if args.untrained_dqn:
            fresh = DQNAgent(agent.obs_dim if agent else make_eval_env(scenario, env_cfg, physics, reward).obs_dim,
                             len(env_cfg.action_levels), build_dqn_config(cfg), device=args.device, seed=args.seed)
            entries.append(("dqn_untrained", DQNPolicy(fresh, name="dqn_untrained"), "untrained"))
        if not args.no_baseline:
            ctl_cfg = replace(baseline_cfg, control_dt=env_cfg.control_dt)
            entries.append(("baseline", BaselinePolicy(lambda env: EnergyShapingLQRController(env.model, ctl_cfg)), "reference"))
        if not args.no_random:
            entries.append(("random", RandomPolicy(args.seed), "reference"))
        if not entries:
            print("error: nothing to evaluate (pass --checkpoint, or drop --no-baseline / --no-random)", file=sys.stderr)
            return 2

        results, trajectories = {}, {}
        for name, policy, label in entries:
            env = make_eval_env(scenario, env_cfg, physics, reward, policy.action_mode, **over)
            rows = evaluate_policy(policy, env, args.episodes, args.seed)
            summary = summarize(rows, scenario)
            if label is None:   # a trained-checkpoint entry: label from the measured primary metric
                label = classify_status(train_steps, summary["primary_rate"])
            results[name] = {"rows": rows, "summary": summary, "status": label}
            _, trajectories[name] = run_episode(env, policy, args.seed, record=True)
            env.close()
            write_episode_csv(out_dir / f"episodes_{name}.csv", rows)
            print(f"{name:24s} [{label}] swing-up {summary['swing_up_rate']:.0%} | balanced {summary['balance_rate']:.0%} | "
                  f"violations {summary['violation_rate']:.0%} | mean err {summary['mean_abs_angle_error'] * 57.2958:.1f} deg | "
                  f"t_up {summary['mean_time_to_upright']:.2f}s")

    except ValueError as exc:      # e.g. --dt that does not divide env.control_dt
        print(f"error: {exc}", file=sys.stderr)
        return 2

    write_summary_csv(out_dir / "summary.csv", results)
    figures = [
        str(plots.save_figure(plots.comparison_figure({n: r["summary"] for n, r in results.items()}, scenario),
                              out_dir / "comparison.png").name),
        str(plots.save_figure(plots.trajectory_figure(trajectories, f"Example episode (seed {args.seed}, scenario {scenario})"),
                              out_dir / "trajectories.png").name),
    ]
    info = {"checkpoint": args.checkpoint or "none",
            "training": (f"{meta['training']['mode']} mode, {meta['training']['env_steps']} environment steps, "
                         f"{meta['training']['episodes']} episodes") if meta.get("training") else "n/a",
            "robustness test": f"physics randomization +-{args.randomize:.0%}, disturbance prob {args.disturbance_prob}"
            if (args.randomize or args.disturbance_prob) else "none (nominal physics, no disturbances)"}
    (out_dir / "report.md").write_text(build_report_markdown(results, scenario, args.seed, info, figures), encoding="utf-8")
    print(f"\nWrote report, CSVs and plots to {out_dir}")
    return 0


def _path(p: str) -> Path:
    q = Path(p)
    return q if q.is_absolute() or q.exists() else PROJECT_ROOT / q


if __name__ == "__main__":
    raise SystemExit(main())
