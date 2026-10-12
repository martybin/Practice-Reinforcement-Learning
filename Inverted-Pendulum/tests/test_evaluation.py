"""Evaluation metrics, benchmark ordering, report generation and dashboard helpers."""

import math

import numpy as np
import pandas as pd
import pytest

from pendulum_rl.control import EnergyShapingLQRController
from pendulum_rl.environment import EnvConfig, RewardConfig
from pendulum_rl.evaluation import (
    BaselinePolicy, RandomPolicy, evaluate_policy, make_eval_env, run_episode, summarize,
)
from pendulum_rl.evaluation.report import build_report_markdown, write_episode_csv, write_summary_csv
from pendulum_rl.physics import PhysicsParams
from pendulum_rl.visualization import dashboard, plots

PHYS, ENV, REW = PhysicsParams(), EnvConfig(), RewardConfig()


def baseline():
    return BaselinePolicy(lambda env: EnergyShapingLQRController(env.model))


def test_baseline_beats_random_on_the_same_seeds_and_metrics_are_consistent():
    results = {}
    for name, policy in (("baseline", baseline()), ("random", RandomPolicy(0))):
        env = make_eval_env("swingup", ENV, PHYS, REW, policy.action_mode)
        rows = evaluate_policy(policy, env, 6, seed=100)
        results[name] = summarize(rows, "swingup")
        assert [r["seed"] for r in rows] == list(range(100, 106))
    b, r = results["baseline"], results["random"]
    assert b["swing_up_rate"] >= 0.8 and b["balance_rate"] >= 0.8
    assert r["balance_rate"] == 0.0 and r["swing_up_rate"] < b["swing_up_rate"]
    assert 0 < b["mean_time_to_upright"] < 5.0
    assert b["primary_metric"] == "swing_up_rate"
    assert b["mean_effort_abs"] > 0


def test_same_seed_gives_identical_initial_states_across_action_modes():
    discrete = make_eval_env("swingup", ENV, PHYS, REW, "discrete")
    continuous = make_eval_env("swingup", ENV, PHYS, REW, "continuous")
    for seed in (1, 2, 3):
        discrete.reset(seed=seed)
        continuous.reset(seed=seed)
        assert np.array_equal(discrete.state, continuous.state)


def test_policy_and_environment_action_mode_mismatch_is_reported():
    env = make_eval_env("swingup", ENV, PHYS, REW, "discrete")
    with pytest.raises(ValueError, match="continuous"):
        evaluate_policy(baseline(), env, 1)


def test_summarize_on_synthetic_rows():
    def row(**kw):
        base = dict(swing_up_success=True, balance_success=True, violation=False, violation_reason=None,
                    mean_abs_angle_error=0.1, max_abs_angle_error=0.5, time_to_upright=2.0, balanced_time=3.0,
                    longest_balance=3.0, effort_abs=1.0, effort_sq=0.5, **{"return": 10.0})
        base.update(kw)
        return base

    rows = [row(), row(time_to_upright=4.0, max_abs_angle_error=3.0),
            row(swing_up_success=False, balance_success=False, violation=True, violation_reason="rail_limit",
                time_to_upright=math.nan)]
    s = summarize(rows, "combined")
    assert s["swing_up_rate"] == pytest.approx(2 / 3) and s["balance_rate"] == pytest.approx(2 / 3)
    assert s["violation_rate"] == pytest.approx(1 / 3) and s["violations_by_reason"] == {"rail_limit": 1}
    assert s["mean_time_to_upright"] == pytest.approx(3.0) and s["max_abs_angle_error"] == 3.0
    assert s["primary_rate"] == s["balance_rate"]
    with pytest.raises(ValueError):
        summarize([], "combined")


def test_trajectory_recording_and_report_files(tmp_path):
    policy = baseline()
    env = make_eval_env("swingup", ENV, PHYS, REW, "continuous")
    metrics, traj = run_episode(env, policy, seed=5, record=True)
    assert set(traj) == {"t", "x", "theta", "theta_dot", "command", "energy", "reward"}
    assert len(traj["t"]) == metrics["steps"] + 1 and np.all(np.diff(traj["t"]) > 0)
    assert abs(traj["theta"][0]) > 2.9               # starts hanging

    rows = evaluate_policy(policy, env, 2, seed=5)
    summary = summarize(rows, "swingup")
    results = {"baseline": {"rows": rows, "summary": summary, "status": "reference"}}
    write_episode_csv(tmp_path / "episodes.csv", rows)
    write_summary_csv(tmp_path / "summary.csv", results)
    assert len(pd.read_csv(tmp_path / "episodes.csv")) == 2
    assert pd.read_csv(tmp_path / "summary.csv").loc[0, "controller"] == "baseline"
    md = build_report_markdown(results, "swingup", 5, {"checkpoint": "none"}, ["comparison.png"])
    assert "baseline" in md and "swing-up" in md.lower() and "measured in this run" in md

    fig = plots.comparison_figure({"baseline": summary, "random": summary}, "swingup")
    plots.save_figure(fig, tmp_path / "c.png")
    plots.save_figure(plots.trajectory_figure({"baseline": traj}), tmp_path / "t.png")
    assert (tmp_path / "c.png").stat().st_size > 5000 and (tmp_path / "t.png").stat().st_size > 5000


def test_loading_missing_logs_returns_none_not_fake_data(tmp_path):
    logs = plots.load_run(tmp_path / "nothing")
    assert logs.episodes is None and logs.evals is None and logs.summary is None
    assert plots.list_runs(tmp_path / "nothing") == []
    assert dashboard.find_runs(tmp_path / "nothing") == [] and dashboard.find_checkpoints(tmp_path / "x") == []


def test_dashboard_metric_frames_from_real_log_schema():
    n = 30
    ep = pd.DataFrame({
        "episode": range(1, n + 1), "return": np.linspace(0, 50, n), "length": 100, "success": [0, 1] * (n // 2),
        "swing_up_success": 1, "longest_balance": 1.0, "mean_abs_angle_error": 0.3, "epsilon": np.linspace(1, 0.1, n),
        "loss": np.linspace(1, 0.1, n), "q_max": 1.0, "q_mean": 0.5, "q_min": 0.0, "action_frac_0": 0.3,
        "action_frac_1": 0.3, "action_frac_2": 0.4})
    frames = dashboard.metric_frames(ep, window=5)
    assert len(frames) == 10 and list(frames["Action distribution"].columns) == ["action 0", "action 1", "action 2"]
    assert frames["Success rate"]["success"].iloc[-1] == pytest.approx(0.4, abs=0.21)
    fig = plots.training_figure(ep.assign(termination_reason="time_limit"), None, window=5)
    assert len(fig.axes) == 12


def test_moving_average_handles_nan():
    out = plots.moving_average([1.0, np.nan, 3.0, 5.0], 2)
    assert out[0] == 1.0 and out[1] == 1.0 and out[2] == 3.0 and out[3] == 4.0


def test_switching_policy_hands_over_inside_the_catch_region_with_hysteresis():
    from pendulum_rl.evaluation import SwitchingPolicy

    class Stub:
        def __init__(self, action):
            self.action = action

        def act(self, obs, greedy=True):
            return self.action

    env = make_eval_env("swingup", ENV, PHYS, REW, "discrete")
    env.reset(seed=0)
    policy = SwitchingPolicy(Stub(0), Stub(2), catch_angle=0.35, catch_speed=4.0, release_angle=0.8)
    obs = env._observation()
    policy.reset(env)
    assert policy.act(env, obs) == 0                                   # hanging: swing-up specialist
    env.reset(seed=0, options={"state": [0, 0, 0.2, 1.0]})
    assert policy.act(env, obs) == 2 and policy.stabilizing            # inside the catch region: stabilizer
    env.reset(seed=0, options={"state": [0, 0, 0.6, 1.0]})
    assert policy.act(env, obs) == 2                                   # hysteresis: still stabilizing at 0.6 rad
    env.reset(seed=0, options={"state": [0, 0, 1.2, 1.0]})
    assert policy.act(env, obs) == 0 and not policy.stabilizing        # fell out: back to swing-up
    env.reset(seed=0, options={"state": [0, 0, 0.1, 9.0]})
    assert policy.act(env, obs) == 0                                   # too fast to catch: keep swinging


def test_report_verdict_accounts_for_sampling_uncertainty():
    def results(dqn_rate, base_rate, n=100):
        def summary(rate):
            return {"scenario": "combined", "episodes": n, "swing_up_rate": 1.0, "balance_rate": rate, "violation_rate": 1 - rate,
                    "mean_abs_angle_error": 0.1, "max_abs_angle_error": 1.0, "mean_time_to_upright": 1.0,
                    "median_time_to_upright": 1.0, "mean_balanced_time": 1.0, "mean_longest_balance": 1.0,
                    "mean_effort_abs": 1.0, "mean_effort_sq": 1.0, "mean_return": 1.0, "violations_by_reason": {},
                    "primary_metric": "balance_rate", "primary_rate": rate}
        return {"dqn": {"summary": summary(dqn_rate), "status": "trained"}, "baseline": {"summary": summary(base_rate), "status": "reference"}}

    same = build_report_markdown(results(0.65, 0.76), "combined", 0, {}, [])
    assert "statistically indistinguishable" in same and "Sampling uncertainty" in same       # 11 points < 2 standard errors
    better = build_report_markdown(results(0.94, 0.70), "combined", 0, {}, [])
    assert "is better than" in better
    worse = build_report_markdown(results(0.50, 0.90), "combined", 0, {}, [])
    assert "is worse than" in worse
