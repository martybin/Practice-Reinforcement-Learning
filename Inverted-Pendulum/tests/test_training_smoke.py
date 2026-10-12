"""End-to-end data flow: environment -> agent -> optimizer -> logger -> checkpoint -> evaluation."""

import csv
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pendulum_rl.agents import DQNAgent
from pendulum_rl.config import (
    ConfigError, apply_overrides, dataclass_from_dict, deep_merge, load_config, load_yaml, build_env_config,
)
from pendulum_rl.environment import EnvConfig
from pendulum_rl.evaluation import SCENARIOS, classify_status, evaluate_policy, make_eval_env, summarize, DQNPolicy
from pendulum_rl.training import Curriculum, StageConfig, TrainConfig, trainer_from_config, stages_from_config
from pendulum_rl.visualization import plots

ROOT = Path(__file__).resolve().parents[1]


def tiny_config(tmp_path, **train):
    cfg = load_config()
    cfg = apply_overrides(cfg, [
        "dqn.hidden_dims=[32,32]", "dqn.batch_size=32", "dqn.learning_starts=200", "dqn.replay_capacity=5000",
        "dqn.epsilon_decay_steps=1000", "dqn.target_update_interval=50",
        "stages.stabilize.env.max_episode_seconds=1.5", "stages.swingup.env.max_episode_seconds=1.5",
        "stages.combined.env.max_episode_seconds=1.5",
    ])
    kw = dict(mode="stabilize", episodes=6, eval_every=3, eval_episodes=2, seed=0, device="cpu", print_every=1000,
              checkpoint_dir=str(tmp_path / "ckpt"), log_dir=str(tmp_path / "logs"))
    kw.update(train)
    return cfg, kw


@pytest.mark.slow
def test_short_training_run_writes_logs_and_checkpoints(tmp_path, capsys):
    cfg, kw = tiny_config(tmp_path)
    trainer = trainer_from_config(cfg, **kw)
    summary = trainer.train()

    assert summary["episodes"] == 6 and summary["env_steps"] > 200 and summary["updates"] > 0
    assert summary["status"] in ("untrained", "partially trained", "trained")
    assert not summary["interrupted"]

    log_dir, ckpt_dir = tmp_path / "logs" / "stabilize", tmp_path / "ckpt" / "stabilize"
    episodes = pd.read_csv(log_dir / "episodes.csv")
    assert len(episodes) == 6
    for col in ("return", "length", "success", "epsilon", "loss", "q_mean", "q_max", "q_min", "longest_balance",
                "mean_abs_angle_error", "action_frac_0", "action_frac_2", "termination_reason"):
        assert col in episodes.columns
    assert np.isfinite(episodes["return"]).all()
    assert episodes["epsilon"].is_monotonic_decreasing
    assert np.allclose(episodes[["action_frac_0", "action_frac_1", "action_frac_2"]].sum(axis=1), 1.0)
    assert episodes["loss"].notna().any()                       # gradient steps happened and were logged
    assert len(pd.read_csv(log_dir / "eval.csv")) >= 2
    assert json.loads((log_dir / "summary.json").read_text())["run"] == "stabilize"
    assert (log_dir / "run_config.yaml").is_file()
    for name in ("final.pt", "best.pt", "best_stabilize.pt"):
        assert (ckpt_dir / name).is_file()

    agent, meta = DQNAgent.from_checkpoint(ckpt_dir / "final.pt", device="cpu")
    assert meta["training"]["mode"] == "stabilize" and meta["physics"]["max_force"] > 0
    assert meta["env"]["action_levels"] == [-1.0, 0.0, 1.0]

    # evaluation of the saved policy runs on the same machinery
    env = make_eval_env("stabilize", EnvConfig(), trainer.physics, trainer.reward, "discrete")
    rows = evaluate_policy(DQNPolicy(agent), env, 2, seed=1)
    result = summarize(rows, "stabilize")
    assert 0.0 <= result["balance_rate"] <= 1.0 and result["episodes"] == 2

    # plots can be produced from the real logs
    logs = plots.load_run(log_dir)
    fig = plots.training_figure(logs.episodes, logs.evals)
    assert len(fig.axes) == 12
    plots.save_figure(fig, tmp_path / "training.png")
    assert (tmp_path / "training.png").stat().st_size > 5000


@pytest.mark.slow
def test_training_is_reproducible_for_a_fixed_seed(tmp_path):
    def run(sub):
        cfg, kw = tiny_config(tmp_path / sub, episodes=4, eval_every=100)
        trainer_from_config(cfg, **kw).train()
        return pd.read_csv(tmp_path / sub / "logs" / "stabilize" / "episodes.csv")

    a, b = run("a"), run("b")
    pd.testing.assert_frame_equal(a.drop(columns=["wall_time"]), b.drop(columns=["wall_time"]))


def test_existing_run_is_not_overwritten_without_flag(tmp_path):
    cfg, kw = tiny_config(tmp_path, episodes=1, eval_every=100)
    trainer_from_config(cfg, **kw).train()
    with pytest.raises(FileExistsError, match="never overwritten"):
        trainer_from_config(cfg, **kw)
    kw2 = {**kw, "overwrite": True}
    trainer_from_config(cfg, **kw2)                 # explicit intent works


def test_swingup_stage_uses_reach_success_and_hanging_start(tmp_path):
    cfg, kw = tiny_config(tmp_path, mode="swingup", episodes=1, eval_every=100)
    trainer = trainer_from_config(cfg, **kw)
    stage_cfg = trainer._stage_env_cfg(trainer.stages[0])
    assert stage_cfg.start_mode == "hanging" and stage_cfg.success_mode == "reach" and stage_cfg.action_mode == "discrete"
    # the swing-up stage ends on success, so its bonus must beat the discounted value of staying alive: 2 / (1 - gamma)
    gamma = trainer.dqn_cfg.gamma
    assert trainer._stage_reward(trainer.stages[0]).success_bonus > 2.0 / (1.0 - gamma)
    assert trainer.reward.success_bonus != trainer._stage_reward(trainer.stages[0]).success_bonus
    comb = trainer_from_config(cfg, **{**kw, "mode": "combined", "run_name": "c"})
    c_cfg = comb._stage_env_cfg(comb.stages[0])
    assert c_cfg.start_mode == "mixed" and c_cfg.physics_randomization > 0


def test_curriculum_promotion_logic():
    stages = [StageConfig("a", max_episodes=50, promote_success_rate=0.8, promote_window=10, min_episodes=10),
              StageConfig("b", max_episodes=5)]
    cur = Curriculum(stages, promote=True)
    for _ in range(9):
        cur.record(True)
    assert not cur.should_advance()                     # below min_episodes
    cur.record(True)
    assert cur.should_advance() and cur.rolling_success == 1.0
    assert cur.advance().name == "b" and cur.episodes_in_stage == 0 and cur.rolling_success == 0.0
    for _ in range(5):
        cur.record(False)
    assert cur.should_advance()                         # budget exhausted despite no success
    assert cur.advance() is None and cur.finished


def test_single_stage_runs_use_full_budget_even_if_successful():
    cur = Curriculum([StageConfig("a", max_episodes=20, promote_window=5, min_episodes=5)], promote=False)
    for _ in range(19):
        cur.record(True)
    assert not cur.should_advance()
    cur.record(True)
    assert cur.should_advance()


def test_stage_table_validation():
    with pytest.raises(KeyError, match="not defined"):
        stages_from_config({}, ["nope"])
    stages = stages_from_config({"x": {"max_episodes": 9}}, ["x"], max_episodes=3)
    assert stages[0].max_episodes == 3


def test_classify_status_labels():
    assert classify_status(0, 0.95) == "untrained"
    assert classify_status(10_000, 0.2) == "partially trained"
    assert classify_status(10_000, 0.8) == "trained"


def test_train_config_validation():
    with pytest.raises(ValueError):
        TrainConfig(mode="everything")
    assert set(SCENARIOS) == {"stabilize", "swingup", "combined"}


# ------------------------------------------------------------------ configuration handling
def test_default_yaml_files_load_and_build_typed_configs():
    cfg = load_config()
    assert {"physics", "env", "reward", "dqn", "training", "stages", "simulation", "baseline"} <= set(cfg)
    env_cfg = build_env_config(cfg)
    assert env_cfg.control_dt == 0.02 and env_cfg.action_levels == (-1.0, 0.0, 1.0)


def test_config_overrides_and_merge():
    cfg = apply_overrides({"a": {"b": 1}}, ["a.b=2.5", "a.c.d=[1,2]", "x=true"])
    assert cfg == {"a": {"b": 2.5, "c": {"d": [1, 2]}}, "x": True}
    assert deep_merge({"a": {"b": 1, "c": 2}}, {"a": {"b": 9}}) == {"a": {"b": 9, "c": 2}}
    with pytest.raises(ConfigError):
        apply_overrides({}, ["no_equals_sign"])


def test_unknown_config_key_gives_suggestion():
    with pytest.raises(ConfigError, match="did you mean 'max_force'"):
        dataclass_from_dict(__import__("pendulum_rl.physics", fromlist=["PhysicsParams"]).PhysicsParams,
                            {"max_forse": 3}, "physics")


def test_missing_config_file_message(tmp_path):
    with pytest.raises(ConfigError, match="does not exist"):
        load_yaml(tmp_path / "nope.yaml")
    bad = tmp_path / "bad.yaml"
    bad.write_text("a: [1, 2")
    with pytest.raises(ConfigError, match="not valid YAML"):
        load_yaml(bad)


# ------------------------------------------------------------------ command-line scripts
def run_script(*args):
    return subprocess.run([sys.executable, str(ROOT / "scripts" / args[0]), *args[1:]], capture_output=True, text=True,
                          cwd=ROOT, timeout=300)


def test_train_script_reports_missing_config_and_bad_mode():
    r = run_script("train.py", "--config", "does_not_exist.yaml")
    assert r.returncode == 2 and "does not exist" in r.stderr
    r = run_script("train.py", "--mode", "bogus")
    assert r.returncode == 2


def test_evaluate_script_reports_missing_checkpoint():
    r = run_script("evaluate.py", "--checkpoint", "checkpoints/nothing/here.pt")
    assert r.returncode == 2 and "does not exist" in r.stderr


def test_simulate_dqn_mode_without_checkpoint_explains_itself():
    r = run_script("simulate.py", "--mode", "dqn", "--headless")
    assert r.returncode == 2 and "--checkpoint" in r.stderr


@pytest.mark.slow
def test_random_agent_script_runs():
    r = run_script("random_agent.py", "--episodes", "2")
    assert r.returncode == 0 and "mean return" in r.stdout


@pytest.mark.slow
def test_tensorboard_logging_is_optional_and_writes_events(tmp_path):
    pytest.importorskip("tensorboard")
    cfg, kw = tiny_config(tmp_path, episodes=3, eval_every=100, tensorboard=True)
    trainer_from_config(cfg, **kw).train()
    events = list((tmp_path / "logs" / "stabilize" / "tb").glob("events.out.tfevents.*"))
    assert events and events[0].stat().st_size > 0


def test_evaluate_script_rejects_incompatible_timestep_cleanly(tmp_path):
    r = run_script("evaluate.py", "--scenario", "swingup", "--episodes", "1", "--dt", "0.0075", "--no-random",
                   "--out", str(tmp_path))
    assert r.returncode == 2 and "multiple" in r.stderr and "Traceback" not in r.stderr
