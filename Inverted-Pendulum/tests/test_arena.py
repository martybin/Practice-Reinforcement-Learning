"""Learning arena: periodic training snapshots -> replayed trajectories -> self-contained HTML page."""

import json
import shutil
import sys
from pathlib import Path

import pytest

from pendulum_rl.config import apply_overrides, load_config
from pendulum_rl.environment import EnvConfig
from pendulum_rl.training import trainer_from_config
from pendulum_rl.visualization import arena
from pendulum_rl.visualization import dashboard

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def trained_run(tmp_path_factory):
    """A real (tiny) training run that wrote snapshots every 3 episodes."""
    base = tmp_path_factory.mktemp("arena_run")
    cfg = apply_overrides(load_config(), [
        "dqn.hidden_dims=[32,32]", "dqn.batch_size=32", "dqn.learning_starts=200", "dqn.replay_capacity=5000",
        "dqn.epsilon_decay_steps=1000", "dqn.target_update_interval=50", "stages.swingup.env.max_episode_seconds=1.5"])
    trainer = trainer_from_config(cfg, mode="swingup", episodes=6, eval_every=3, eval_episodes=2, seed=0, device="cpu",
                                  print_every=1000, snapshot_every=3, checkpoint_dir=str(base / "checkpoints"),
                                  log_dir=str(base / "logs"))
    trainer.train()
    return base


def test_training_writes_untrained_and_periodic_snapshots(trained_run):
    snaps = arena.list_snapshots(trained_run / "checkpoints" / "swingup")
    assert [s.episode for s in snaps] == [0, 3, 6]
    assert all(s.path.stat().st_size > 1000 for s in snaps)


def test_snapshots_can_be_switched_off(tmp_path):
    cfg = apply_overrides(load_config(), ["dqn.hidden_dims=[8]", "stages.stabilize.env.max_episode_seconds=0.5"])
    trainer = trainer_from_config(cfg, mode="stabilize", episodes=2, eval_every=2, eval_episodes=1, device="cpu",
                                  print_every=1000, snapshot_every=0, checkpoint_dir=str(tmp_path / "c"), log_dir=str(tmp_path / "l"))
    trainer.train()
    assert not (tmp_path / "c" / "stabilize" / "snapshots").exists()
    assert (tmp_path / "c" / "stabilize" / "final.pt").is_file()


def test_record_rollout_is_finite_json_deterministic_and_comparable_across_snapshots(trained_run):
    snaps = arena.list_snapshots(trained_run / "checkpoints" / "swingup")
    a = arena.record_rollout(snaps[0].path, start="hanging", seed=7, seconds=1.0)
    b = arena.record_rollout(snaps[0].path, start="hanging", seed=7, seconds=1.0)
    assert a == b                                                   # same checkpoint + seed -> identical frames
    json.dumps(a, allow_nan=False)                                  # no NaN / inf may reach the page
    n = a["n"]
    for key in ("t", "x", "xd", "th", "thd", "u", "act", "e", "ret", "ph"):
        assert len(a[key]) == n
    assert a["t"][0] == 0.0 and a["dt"] == EnvConfig().control_dt
    assert a["verdict"]["kind"] in ("good", "warn", "bad", "fail")
    other = arena.record_rollout(snaps[-1].path, start="hanging", seed=7, seconds=1.0)
    assert other["x"][0] == a["x"][0] and other["th"][0] == a["th"][0]     # identical start for every snapshot


def test_kick_changes_the_trajectory_and_is_recorded(trained_run):
    path = arena.list_snapshots(trained_run / "checkpoints" / "swingup")[-1].path
    plain = arena.record_rollout(path, start="upright", seed=3, seconds=1.5)
    kicked = arena.record_rollout(path, start="upright", seed=3, seconds=1.5, kick_time=0.5, kick=4.0)
    assert plain["kick_t"] is None and kicked["kick_t"] == pytest.approx(0.5, abs=0.021)
    assert plain["thd"] != kicked["thd"]


def test_record_rollout_rejects_bad_arguments(trained_run):
    path = arena.list_snapshots(trained_run / "checkpoints" / "swingup")[0].path
    with pytest.raises(ValueError, match="start"):
        arena.record_rollout(path, start="sideways")
    with pytest.raises(ValueError, match="seconds"):
        arena.record_rollout(path, seconds=0)


def test_thin_out_keeps_first_and_last_and_is_sorted():
    snaps = [arena.Snapshot(Path(f"ep_{i:05d}.pt"), i * 10) for i in range(101)]
    kept = arena.thin_out(snaps, 20)
    assert len(kept) <= 20 and kept[0].episode == 0 and kept[-1].episode == 1000
    assert [s.episode for s in kept] == sorted(s.episode for s in kept)
    assert arena.thin_out(snaps[:5], 20) == snaps[:5]


@pytest.mark.parametrize("metrics, kind", [
    (dict(violation=True, violation_reason="rail_limit", swing_up_success=False, time_to_upright=float("nan"),
          duration=0.4, balance_success=False, termination_reason="rail_limit", final_abs_angle_error=2.0,
          balanced_time=0.0), "fail"),
    (dict(violation=False, violation_reason=None, swing_up_success=True, time_to_upright=1.0, duration=10.0,
          balance_success=True, termination_reason="time_limit", final_abs_angle_error=0.01, balanced_time=8.0), "good"),
    (dict(violation=False, violation_reason=None, swing_up_success=True, time_to_upright=1.0, duration=10.0,
          balance_success=False, termination_reason="time_limit", final_abs_angle_error=1.5, balanced_time=0.5), "warn"),
    (dict(violation=False, violation_reason=None, swing_up_success=False, time_to_upright=float("nan"), duration=10.0,
          balance_success=False, termination_reason="time_limit", final_abs_angle_error=3.1, balanced_time=0.0), "bad"),
])
def test_verdict_describes_how_the_episode_ended(metrics, kind):
    v = arena.verdict(metrics, EnvConfig())
    assert v["kind"] == kind and v["title"] and v["detail"]


def test_learning_curve_reads_the_eval_log_and_tolerates_a_missing_one(trained_run, tmp_path):
    curve = arena.learning_curve(trained_run / "logs" / "swingup")
    assert curve["episode"] and len(curve["episode"]) == len(curve["balance"]) == len(curve["swing"])
    assert curve["scenario"] == "swingup" and curve["eval_episodes"] == 2
    assert arena.learning_curve(tmp_path / "nothing")["episode"] == []


def test_arena_html_is_self_contained_and_escapes_markup(trained_run):
    path = arena.list_snapshots(trained_run / "checkpoints" / "swingup")[0].path
    r = arena.record_rollout(path, seconds=1.0)
    r["episode"] = 0
    html = arena.build_arena_html("</script><b>x", [r], {"episode": [], "balance": [], "swing": [], "scenario": None},
                                  start="hanging", seed=1, seconds=1.0)
    assert "/*__ARENA_DATA__*/" not in html and "<canvas" in html
    assert "</script><b>" not in html and "\\u003c/script>" in html     # run names cannot break out of the script tag
    with pytest.raises(ValueError):
        arena.build_arena_html("run", [], {}, start="hanging", seed=1, seconds=1.0)


def test_runs_without_snapshots_fall_back_to_best_and_final(trained_run, tmp_path):
    run = tmp_path / "checkpoints" / "old_run"
    run.mkdir(parents=True)
    src = trained_run / "checkpoints" / "swingup"
    shutil.copy(src / "best.pt", run / "best.pt")
    shutil.copy(src / "final.pt", run / "final.pt")
    assert arena.list_snapshots(run) == []
    fallback = arena.fallback_checkpoints(run)
    assert {s.kind for s in fallback} == {"best", "final"}
    assert dashboard.find_arena_runs(tmp_path / "checkpoints") == {"old_run": -2}      # negative = best/final only


def test_dashboard_helpers(trained_run):
    taken = {"combined_1", "combined_2", "swingup_1"}
    assert dashboard.default_run_name("combined", taken) == "combined_3"
    assert dashboard.default_run_name("stabilize", taken) == "stabilize_1"
    cmd = dashboard.build_train_command("combined", 300, "demo", 10, 4)
    assert cmd[0] == sys.executable and cmd[2].endswith("train.py")
    assert cmd[cmd.index("--episodes") + 1] == "300" and cmd[cmd.index("--run-name") + 1] == "demo"
    assert cmd[cmd.index("--snapshot-every") + 1] == "10" and "--overwrite" not in cmd       # never overwrites a run
    runs = dashboard.find_arena_runs(trained_run / "checkpoints")
    assert runs == {"swingup": 3}
    assert all("snapshots" not in p.parts for p in dashboard.find_checkpoints(trained_run / "checkpoints"))


def test_dashboard_arena_tab_renders_real_snapshots(trained_run, monkeypatch):
    pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest
    monkeypatch.setenv("PENDULUM_RL_ROOT", str(trained_run))
    at = AppTest.from_file(str(ROOT / "src" / "pendulum_rl" / "visualization" / "dashboard.py"), default_timeout=120)
    at.run()
    assert not at.exception
    labels = [s.label for s in at.selectbox]
    assert "Training run" in labels and "Pendulum starts" in labels
    assert "Start training" in [b.label for b in at.button]
