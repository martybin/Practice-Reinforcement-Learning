"""Optional Streamlit dashboard: a *learning arena* that replays the network after N training episodes, a launcher
for new training runs, and the usual log / evaluation / checkpoint views (real data only).

    pip install streamlit
    streamlit run src/pendulum_rl/visualization/dashboard.py

The core simulator, trainer and evaluator never import this module. Nothing here is simulated:
if there are no logs or checkpoints yet, the dashboard says so instead of drawing placeholder data.
"""

from __future__ import annotations

import functools
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(os.environ.get("PENDULUM_RL_ROOT") or Path(__file__).resolve().parents[3])   # override for tests
_SRC = Path(__file__).resolve().parents[2]
ARENA_HEIGHT = 1060         # px of the embedded arena page


def find_runs(log_root: Path) -> list[Path]:
    """Run directories (``logs/<run>``) that contain an ``episodes.csv``."""
    return sorted(p for p in log_root.iterdir() if (p / "episodes.csv").is_file()) if log_root.is_dir() else []


def find_checkpoints(ckpt_root: Path) -> list[Path]:
    """Named checkpoints (best/final/...); the many periodic ``snapshots/ep_*.pt`` are listed per run instead."""
    return sorted(p for p in ckpt_root.rglob("*.pt") if "snapshots" not in p.parts) if ckpt_root.is_dir() else []


def find_arena_runs(ckpt_root: Path) -> dict[str, int]:
    """``{run name: number of replayable networks}``; runs with periodic snapshots come first."""
    if not ckpt_root.is_dir():
        return {}
    out = {}
    for run in sorted(p for p in ckpt_root.iterdir() if p.is_dir()):
        snaps = len(list((run / "snapshots").glob("ep_*.pt"))) if (run / "snapshots").is_dir() else 0
        named = sum((run / f"{n}.pt").is_file() for n in ("best", "final"))
        if snaps or named:
            out[run.name] = snaps if snaps else -named        # negative: only best/final exist
    return dict(sorted(out.items(), key=lambda kv: (kv[1] <= 0, kv[0])))


def find_reports(out_root: Path) -> list[Path]:
    return sorted(out_root.glob("eval_*/summary.csv")) if out_root.is_dir() else []


def smooth(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window, min_periods=1).mean()


def metric_frames(episodes: pd.DataFrame, window: int) -> dict[str, pd.DataFrame]:
    """Chart-ready frames (index = episode) for every dashboard panel."""
    ep = episodes.set_index("episode")
    action_cols = sorted([c for c in ep.columns if c.startswith("action_frac_")], key=lambda c: int(c.rsplit("_", 1)[1]))
    frames = {
        "Episode reward": pd.DataFrame({"reward": ep["return"], f"moving avg ({window})": smooth(ep["return"], window)}),
        "Episode length (steps)": pd.DataFrame({"length": ep["length"], f"moving avg ({window})": smooth(ep["length"], window)}),
        "Success rate": pd.DataFrame({"success": smooth(ep["success"].astype(float), window)}),
        "Swing-up success rate": pd.DataFrame({"swing-up": smooth(ep["swing_up_success"].astype(float), window)}),
        "Balance duration (s)": pd.DataFrame({"longest balance": smooth(ep["longest_balance"], window)}),
        "Average angular error (deg)": pd.DataFrame({"mean |theta|": smooth(np.degrees(ep["mean_abs_angle_error"]), window)}),
        "Epsilon": pd.DataFrame({"epsilon": ep["epsilon"]}),
        "TD loss": pd.DataFrame({"loss": smooth(ep["loss"], window)}),
        "Q-value statistics": pd.DataFrame({k: smooth(ep[c], window) for k, c in
                                            (("max", "q_max"), ("mean", "q_mean"), ("min", "q_min"))}),
        "Action distribution": pd.DataFrame({f"action {i}": smooth(ep[c], window) for i, c in enumerate(action_cols)}),
    }
    return frames


def default_run_name(mode: str, taken: set[str]) -> str:
    """``<mode>_1``, ``<mode>_2``, ... - the first name not used by an existing run (runs are never overwritten)."""
    k = 1
    while f"{mode}_{k}" in taken:
        k += 1
    return f"{mode}_{k}"


def build_train_command(mode: str, episodes: int, run_name: str, snapshot_every: int, seed: int) -> list[str]:
    """Command line that the dashboard's *Train* tab launches (the same ``scripts/train.py`` you can run yourself)."""
    return [sys.executable, "-u", str(PROJECT_ROOT / "scripts" / "train.py"), "--mode", mode, "--episodes", str(episodes),
            "--run-name", run_name, "--seed", str(seed), "--snapshot-every", str(snapshot_every),
            "--eval-every", str(max(1, snapshot_every)), "--print-every", "10"]


@functools.lru_cache(maxsize=1)
def _cached_rollout():
    """Streamlit-cached ``record_rollout``: each (snapshot file, start condition) is simulated once per file version."""
    import streamlit as st

    @st.cache_data(show_spinner=False, max_entries=1024)
    def rollout(path: str, mtime_ns: int, start: str, seed: int, seconds: float, kick_time: float | None) -> dict:
        from pendulum_rl.visualization.arena import record_rollout
        return record_rollout(Path(path), start=start, seed=seed, seconds=seconds, kick_time=kick_time)

    return rollout


def arena_tab(st) -> None:  # pragma: no cover - requires a running Streamlit session
    """Replay the network after N training episodes in an animated copy of the simulator scene."""
    import streamlit.components.v1 as components

    runs = find_arena_runs(PROJECT_ROOT / "checkpoints")
    if not runs:
        st.info("No checkpoints in `checkpoints/` yet, so there is nothing to replay. Train a model (the **Train** tab "
                "does it from this page) or run\n\n"
                "`python scripts/train.py --mode combined --episodes 500 --snapshot-every 10 --run-name demo`")
        return
    from pendulum_rl.visualization import arena

    labels = {name: f"{name}  ({n} snapshots)" if n > 0 else f"{name}  (best / final only)" for name, n in runs.items()}
    c1, c2, c3, c4, c5 = st.columns([1.6, 1.8, 0.8, 1.2, 1.4])
    run = c1.selectbox("Training run", list(runs), format_func=labels.get, key="arena_run")
    start = c2.selectbox("Pendulum starts", list(arena.START_MODES), format_func=arena.START_MODES.get, key="arena_start")
    seed = int(c3.number_input("Start seed", 0, 9999, 1, key="arena_seed", help="Same seed = same start for every snapshot."))
    seconds = float(c4.slider("Episode length [s]", 4, 20, 10, key="arena_seconds"))
    kick_on = c5.toggle("Kick it mid-run", False, key="arena_kick", help="One-off angular-velocity kick, to test recovery.")
    kick_time = float(st.slider("Kick at t [s]", 1.0, seconds - 1.0, min(5.0, seconds - 1.0), 0.5, key="arena_kick_t")) if kick_on else None

    ckpt_dir = PROJECT_ROOT / "checkpoints" / run
    snaps = arena.list_snapshots(ckpt_dir)
    fallback = not snaps
    if fallback:
        snaps = arena.fallback_checkpoints(ckpt_dir)
        st.warning("This run has no periodic snapshots (it was trained before they existed), so only its best / final "
                   "network can be replayed. Train again with `--snapshot-every 10` to get the full progression.")
    if not snaps:
        st.error("No readable checkpoint in this run.")
        return
    snaps = arena.thin_out(snaps, 80)

    rollout, rollouts = _cached_rollout(), []
    bar = st.progress(0.0, text="Replaying the saved networks ...")
    for i, snap in enumerate(snaps):
        path, mtime = snap.cache_key
        r = dict(rollout(path, mtime, start, seed, seconds, kick_time))
        r["episode"] = snap.episode
        rollouts.append(r)
        bar.progress((i + 1) / len(snaps), text=f"Replaying the saved networks ... {i + 1}/{len(snaps)}")
    bar.empty()

    meta_physics = None
    try:
        from pendulum_rl.utils.checkpoints import load_checkpoint
        meta_physics = load_checkpoint(snaps[-1].path).get("metadata", {}).get("physics")
    except Exception:       # the arena still works with default geometry
        pass
    html = arena.build_arena_html(run, rollouts, arena.learning_curve(PROJECT_ROOT / "logs" / run), start=start, seed=seed,
                                  seconds=seconds, physics=meta_physics)
    components.html(html, height=ARENA_HEIGHT, scrolling=True)
    st.caption("Every frame is produced by the real simulator driven by the saved network (greedy actions, no exploration). "
               "Keys inside the arena: Space = pause, R = replay, ← / → = previous / next snapshot.")


def train_tab(st) -> None:  # pragma: no cover - requires a running Streamlit session
    """Start ``scripts/train.py`` from the page and follow its progress; new snapshots appear in the arena."""
    st.markdown("Start a new training run here. It runs the same `scripts/train.py` you can run in a terminal and writes "
                "periodic snapshots, so the **Learning arena** can show what the network had learned after each stretch of "
                "episodes. Existing runs are never overwritten.")
    taken = {p.name for p in (PROJECT_ROOT / "checkpoints").glob("*")} | {p.name for p in (PROJECT_ROOT / "logs").glob("*")}
    c1, c2, c3, c4, c5 = st.columns([1.3, 1, 1.5, 1, 0.8])
    mode = c1.selectbox("Mode", ["combined", "swingup", "stabilize", "curriculum"], key="train_mode",
                        help="combined: swing-up and balance in one network. curriculum: stabilize -> swing-up -> combined "
                             "(episodes counts per stage).")
    episodes = int(c2.number_input("Episodes", 20, 20000, 500, 50, key="train_episodes"))
    name = c3.text_input("Run name", default_run_name(mode, taken), key=f"train_name_{mode}")
    snap_every = int(c4.number_input("Snapshot every", 1, 1000, 10, key="train_snap"))
    seed = int(c5.number_input("Seed", 0, 9999, 0, key="train_seed"))

    state = st.session_state.get("train_job")
    running = state is not None and state["proc"].poll() is None
    clash = name in taken
    if clash:
        st.warning(f"A run named `{name}` already exists. Pick another name (existing results are never overwritten).")
    if st.button("Start training", type="primary", disabled=running or clash or not name.strip()):
        script = PROJECT_ROOT / "scripts" / "train.py"
        if not script.is_file():
            st.error(f"Cannot find {script}.")
        else:
            log_dir = PROJECT_ROOT / "logs" / ".launcher"
            log_dir.mkdir(parents=True, exist_ok=True)
            log_path = log_dir / f"{name}.log"
            proc = subprocess.Popen(build_train_command(mode, episodes, name.strip(), snap_every, seed), cwd=PROJECT_ROOT,
                                    stdout=open(log_path, "w", encoding="utf-8"), stderr=subprocess.STDOUT,
                                    env={**os.environ, "PYTHONUNBUFFERED": "1"})
            st.session_state["train_job"] = {"proc": proc, "name": name.strip(), "log": log_path,
                                             "target": episodes * (3 if mode == "curriculum" else 1)}
            st.rerun()

    @st.fragment(run_every=2)
    def progress() -> None:
        job = st.session_state.get("train_job")
        if not job:
            return
        proc = job["proc"]
        rc = proc.poll()
        csv_path = PROJECT_ROOT / "logs" / job["name"] / "episodes.csv"
        done, rolling = 0, None
        if csv_path.is_file():
            try:
                eps = pd.read_csv(csv_path)
                done = len(eps)
                rolling = float(eps["success"].tail(50).mean()) if done else None
            except Exception:        # file is being written; try again on the next refresh
                pass
        target = job["target"]
        text = f"`{job['name']}`: {done} / {target if rc is None else max(done, 1)} episodes" + \
               (f"  |  success over the last 50 episodes: {rolling:.0%}" if rolling is not None else "")
        st.progress(min(1.0, done / target) if rc is None else 1.0, text=text.replace("`", ""))
        if rc is None:
            if st.button("Stop training", key="stop_train"):
                proc.terminate()
        elif rc == 0:
            st.success("Training finished. Open the **Learning arena** tab and select this run.")
        else:
            st.error(f"Training stopped with exit code {rc}. See the log below.")
        try:
            tail = job["log"].read_text(encoding="utf-8", errors="replace").splitlines()[-12:]
        except OSError:
            tail = []
        st.code("\n".join(tail) or "(no output yet)", language="text")

    progress()


def main() -> None:  # pragma: no cover - requires a running Streamlit session
    import streamlit as st

    if str(_SRC) not in sys.path:       # `streamlit run <this file>` does not put src/ on the path
        sys.path.insert(0, str(_SRC))
    st.set_page_config(page_title="Pendulum RL dashboard", layout="wide")
    st.title("Moving-support pendulum - learning dashboard")
    st.caption("Watch what the network learned after N training episodes, start new runs, and inspect logs, reports and "
               "checkpoints. Everything comes from real files in `logs/`, `checkpoints/` and `outputs/`; nothing is "
               "simulated or filled in.")

    log_root, ckpt_root, out_root = PROJECT_ROOT / "logs", PROJECT_ROOT / "checkpoints", PROJECT_ROOT / "outputs"
    runs, ckpts = find_runs(log_root), find_checkpoints(ckpt_root)

    tab_arena, tab_launch, tab_train, tab_eval, tab_ckpt = st.tabs(
        ["Learning arena", "Train", "Training logs", "Evaluation", "Checkpoints"])
    with tab_arena:
        arena_tab(st)
    with tab_launch:
        train_tab(st)

    with tab_train:
        if not runs:
            st.info("No training runs found in `logs/`. Start one, e.g.\n\n"
                    "`python scripts/train.py --mode stabilize --episodes 300`\n\n"
                    "then refresh this page (logs are written while training runs).")
        else:
            run = st.selectbox("Run", runs, format_func=lambda p: p.name)
            window = st.slider("Moving-average window (episodes)", 1, 100, 20)
            episodes = pd.read_csv(run / "episodes.csv")
            if episodes.empty:
                st.warning("This run has no completed episodes yet.")
            else:
                last = episodes.iloc[-1]
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Episodes", int(last["episode"]))
                c2.metric("Environment steps", int(last["env_steps"]))
                c3.metric(f"Success (last {min(window, len(episodes))})", f"{episodes['success'].tail(window).mean():.0%}")
                c4.metric("Epsilon", f"{last['epsilon']:.2f}")
                frames = metric_frames(episodes, window)
                names = list(frames)
                for row in range(0, len(names), 2):
                    cols = st.columns(2)
                    for col, name in zip(cols, names[row:row + 2]):
                        col.subheader(name)
                        col.line_chart(frames[name])
                eval_p = run / "eval.csv"
                if eval_p.is_file():
                    st.subheader("Greedy evaluations (fixed seeds)")
                    ev = pd.read_csv(eval_p)
                    st.line_chart(ev.set_index("env_steps")[["swing_up_rate", "balance_rate", "violation_rate"]])
                    st.dataframe(ev.tail(20))
                summary_p = run / "summary.json"
                if summary_p.is_file():
                    st.subheader("Final summary")
                    st.json(json.loads(summary_p.read_text(encoding="utf-8")))
                else:
                    st.caption("Run still in progress (or interrupted before writing summary.json).")

    with tab_eval:
        reports = find_reports(out_root)
        if not reports:
            st.info("No evaluation reports in `outputs/`. Create one with\n\n"
                    "`python scripts/evaluate.py --checkpoint checkpoints/<run>/best.pt --scenario combined`")
        for summary_p in reports:
            st.subheader(summary_p.parent.name)
            st.dataframe(pd.read_csv(summary_p))
            for png in sorted(summary_p.parent.glob("*.png")):
                st.image(str(png))
            report_md = summary_p.parent / "report.md"
            if report_md.is_file():
                with st.expander("report.md"):
                    st.markdown(report_md.read_text(encoding="utf-8"))

    with tab_ckpt:
        if not ckpts:
            st.info("No checkpoints in `checkpoints/` yet.")
        else:
            st.dataframe(pd.DataFrame({"checkpoint": [str(p.relative_to(PROJECT_ROOT)) for p in ckpts],
                                       "size [kB]": [round(p.stat().st_size / 1024, 1) for p in ckpts]}))


if __name__ == "__main__":  # pragma: no cover
    main()
