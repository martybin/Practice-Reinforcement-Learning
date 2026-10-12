"""Streamlit dashboard for the fleet-dispatching study.

Every number shown here is produced by the real simulator, the trained checkpoints, or the CSV files the
pipeline wrote. Nothing is hardcoded. Run from the repository root:

    streamlit run app/streamlit_app.py

Pages
  Overview    simulation configuration, checkpoint status, and held-out evaluation summary
  Simulation  step through one shift decision by decision, with the DQN or a baseline choosing
  Analytics   training curves, reward components, and evaluation tables from the saved CSVs
  Compare     run every policy live on chosen seeds with the same simulator
"""

from __future__ import annotations

import copy
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from fleet_dispatch.agents.dqn_agent import DQNPolicy, load_policy_agent  # noqa: E402
from fleet_dispatch.config import ConfigError, ProjectConfig, load_config  # noqa: E402
from fleet_dispatch.environment.mining_env import FleetDispatchEnv  # noqa: E402
from fleet_dispatch.environment.reward import REWARD_COMPONENTS  # noqa: E402
from fleet_dispatch.metrics.operational_metrics import moving_average  # noqa: E402
from fleet_dispatch.policies import NearestEligibleShovelPolicy, RandomValidPolicy, ShortestQueuePolicy  # noqa: E402
from fleet_dispatch.training.evaluate import ALL_POLICIES, evaluate_policies  # noqa: E402
from fleet_dispatch.utils.seeding import evaluation_seeds  # noqa: E402

st.set_page_config(page_title="Fleet dispatching (DQN)", layout="wide")

STATUS_COLORS = {
    "IDLE": "#9e9e9e",
    "TRAVEL_EMPTY": "#fb8c00",
    "QUEUED": "#8e24aa",
    "LOADING": "#1e88e5",
    "TRAVEL_LOADED": "#43a047",
    "UNLOADING": "#00897b",
    "MAINTENANCE": "#e53935",
}


@st.cache_resource(show_spinner=False)
def get_config() -> ProjectConfig:
    return load_config()


def find_checkpoints(config: ProjectConfig) -> list[Path]:
    root = config.resolve(config.paths.checkpoint_dir)
    if not root.exists():
        return []
    return sorted(root.glob("*/final.pt")) + sorted(root.glob("*/latest.pt"))


def read_csv_if_exists(path: Path) -> pd.DataFrame | None:
    return pd.read_csv(path) if path.exists() else None


def make_policy(name: str, seed: int, agent):
    if name == "dqn":
        return DQNPolicy(agent)
    if name == "random_valid":
        return RandomValidPolicy(seed=seed)
    if name == "nearest_eligible":
        return NearestEligibleShovelPolicy()
    return ShortestQueuePolicy()


def draw_schematic(snap: dict) -> plt.Figure:
    """State map of the pit. Positions show state, not time: trucks on a road are drawn at its midpoint."""
    fig, ax = plt.subplots(figsize=(10, 4.2))
    n_shovels = len(snap["shovels"])
    shovel_y = np.linspace(0.85, 0.15, n_shovels)
    depot_x, shovel_x = 0.14, 0.82
    for j in range(n_shovels):
        ax.plot([depot_x, shovel_x], [0.5, shovel_y[j]], color="#bdbdbd", linewidth=1, zorder=0)
    ax.add_patch(plt.Rectangle((0.02, 0.35), 0.12, 0.3, color="#cfd8dc", zorder=1))
    ax.text(0.08, 0.5, "DEPOT\n(dump)", ha="center", va="center", fontsize=9)
    for j, shovel in enumerate(snap["shovels"]):
        color = "#c8e6c9" if shovel["status"] == "OPERATIONAL" else "#ffcdd2"
        ax.add_patch(plt.Rectangle((0.82, shovel_y[j] - 0.07), 0.16, 0.14, color=color, zorder=1))
        ax.text(0.9, shovel_y[j], f"Shovel {j}\n{shovel['rate_tpm']:.0f} t/min\nqueue {len(shovel['queue'])}",
                ha="center", va="center", fontsize=8)

    depot_slot = 0
    queue_slot: dict[int, int] = defaultdict(int)
    for truck in snap["trucks"]:
        status, j, k = truck["status"], truck["target_shovel"], truck["index"]
        if status in ("IDLE", "UNLOADING") or (status == "MAINTENANCE" and j < 0):
            x, y = 0.03 + 0.025 * (depot_slot % 5), 0.08 + 0.045 * (depot_slot // 5)
            depot_slot += 1
        elif status in ("TRAVEL_EMPTY", "TRAVEL_LOADED", "MAINTENANCE"):
            frac = 0.5 + 0.03 * ((k % 5) - 2)
            x = depot_x + frac * (shovel_x - depot_x)
            y = 0.5 + frac * (shovel_y[j] - 0.5)
        else:  # QUEUED or LOADING: stack trucks behind the shovel
            slot = queue_slot[j]
            queue_slot[j] += 1
            x, y = shovel_x - 0.05 - 0.035 * slot, shovel_y[j]
        ax.scatter([x], [y], s=60, color=STATUS_COLORS[status], edgecolor="black", zorder=3)
        ax.annotate(str(k), (x, y), xytext=(4, 4), textcoords="offset points", fontsize=6)

    handles = [plt.Line2D([0], [0], marker="o", linestyle="", color=c, label=s.lower().replace("_", " "))
               for s, c in STATUS_COLORS.items()]
    ax.legend(handles=handles, loc="lower center", ncol=4, fontsize=7, frameon=False, bbox_to_anchor=(0.5, -0.12))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    fig.tight_layout()
    return fig


def page_overview(config: ProjectConfig) -> None:
    sc, sim = config.scenario, config.simulation
    st.subheader("Simulation configuration")
    st.caption("A SIMULATION on synthetic data. It is not a validated real-world dispatching system.")
    cols = st.columns(4)
    cols[0].metric("Haul trucks", sc.n_trucks)
    cols[1].metric("Loading shovels", sc.n_shovels)
    cols[2].metric("Shift horizon (min)", f"{sc.horizon_minutes:.0f}")
    cols[3].metric("Truck-shovel actions", sc.n_trucks * sc.n_shovels)
    st.write(
        f"Truck capacities (t): {sc.truck_capacity_choices_tonnes} · Shovel rates (t/min): "
        f"{sc.shovel_loading_rate_tpm} · Depot→shovel (min): {sc.depot_to_shovel_minutes} · "
        f"Shovel→depot (min): {sc.shovel_to_depot_minutes}"
    )
    st.write(
        f"Breakdowns: {'on' if sim.breakdowns_enabled else 'off'} (truck MTBF {sim.truck_mtbf_minutes:.0f} min, "
        f"shovel MTBF {sim.shovel_mtbf_minutes:.0f} min) · Disturbances: {'on' if sim.disturbances_enabled else 'off'}"
    )

    st.subheader("Trained model status")
    checkpoints = find_checkpoints(config)
    if not checkpoints:
        st.info("No checkpoint found yet. Run `python scripts/train.py` (or with `--quick`) first.")
    else:
        status = [{"checkpoint": str(p.relative_to(ROOT)), "file": p.name,
                   "size_mb": round(p.stat().st_size / 1e6, 2)} for p in checkpoints]
        st.dataframe(pd.DataFrame(status), use_container_width=True, hide_index=True)

    st.subheader("Held-out evaluation (from results/evaluation)")
    summary = read_csv_if_exists(config.resolve(config.paths.results_dir) / "evaluation" / "summary.csv")
    if summary is None:
        st.info("No evaluation summary yet. Run `scripts/evaluate.py`, then `scripts/compare_baselines.py`.")
    else:
        headline = summary[summary["metric"].isin(
            ["episode_return", "payload_t", "avg_wait_min", "shovel_starvation_min"])]
        st.dataframe(headline.pivot_table(index="metric", columns="policy", values="mean").round(2),
                     use_container_width=True)


def page_simulation(config: ProjectConfig) -> None:
    st.subheader("Step through one shift")
    st.caption("Each step is one dispatch decision. The selected policy picks a truck-to-shovel pair "
               "from the valid actions.")
    checkpoints = find_checkpoints(config)
    left, right = st.columns(2)
    policy_name = left.selectbox("Policy", list(ALL_POLICIES), index=0 if checkpoints else 1)
    checkpoint = None
    if policy_name == "dqn":
        if not checkpoints:
            st.warning("No trained checkpoint is available. Choose a baseline policy.")
            return
        labels = [str(p.relative_to(ROOT)) for p in checkpoints]
        checkpoint = checkpoints[labels.index(right.selectbox("Checkpoint", labels))]
    seed = int(right.number_input("Evaluation seed", min_value=0, value=config.evaluation.seed_start, step=1))

    with st.expander("Simulation parameters (changing these moves away from the training distribution)"):
        breakdowns = st.checkbox("Breakdowns enabled", value=config.simulation.breakdowns_enabled)
        disturbances = st.checkbox("Travel and loading disturbances enabled", value=config.simulation.disturbances_enabled)
        horizon = st.slider("Shift horizon (min)", 60, 720, int(config.scenario.horizon_minutes), step=30)
    sim_config = copy.deepcopy(config)
    sim_config.simulation.breakdowns_enabled = breakdowns
    sim_config.simulation.disturbances_enabled = disturbances
    sim_config.scenario.horizon_minutes = float(horizon)

    key = (policy_name, str(checkpoint), seed, breakdowns, disturbances, horizon)
    if st.session_state.get("sim_key") != key:
        st.session_state.sim_key = key
        st.session_state.pop("sim", None)

    if st.button("Start / restart shift"):
        env = FleetDispatchEnv(sim_config)
        env.reset(seed=seed)
        agent = load_policy_agent(checkpoint)[0] if checkpoint is not None else None
        st.session_state.sim = {"env": env, "policy": make_policy(policy_name, seed, agent),
                                "log": [], "reward": 0.0, "last": None, "final": None}
    sim = st.session_state.get("sim")
    if sim is None:
        st.info("Press **Start / restart shift** to begin.")
        return

    env = sim["env"]
    b1, b2, b3 = st.columns(3)
    step_clicked = b1.button("Step one decision", disabled=env.simulator.done)
    per_play = b2.number_input("Decisions per play", min_value=1, max_value=200, value=10, step=1)
    play_clicked = b3.button("Play decisions", disabled=env.simulator.done)
    n_steps = 1 if step_clicked else (int(per_play) if play_clicked else 0)
    for _ in range(n_steps):
        if env.simulator.done:
            break
        action = sim["policy"].act(env.decision_view())
        _, reward, _, truncated, info = env.step(action)
        sim["reward"] += reward
        truck, shovel = info["dispatched"] if info["dispatched"] else (None, None)
        sim["log"].append({"decision": info["decision_index"], "time_min": round(info["sim_time"], 2),
                           "truck": truck, "shovel": shovel, "reward": round(reward, 3)})
        sim["last"] = info["reward_components"]
        if truncated:
            sim["final"] = info["episode"]

    snap = env.snapshot()
    m = st.columns(4)
    m[0].metric("Simulated time (min)", f"{snap['time']:.1f}")
    m[1].metric("Remaining shift (min)", f"{snap['remaining']:.1f}")
    m[2].metric("Decisions made", snap["steps"])
    m[3].metric("Cumulative reward", f"{sim['reward']:.2f}")
    st.pyplot(draw_schematic(snap), clear_figure=True)

    left, right = st.columns(2)
    left.write("**Trucks**")
    left.dataframe(pd.DataFrame(snap["trucks"])[["index", "status", "destination", "payload_t",
                                                 "remaining_expected_min"]].round(1),
                   use_container_width=True, hide_index=True, height=280)
    right.write("**Shovels**")
    right.dataframe(pd.DataFrame(snap["shovels"])[["index", "status", "loading_truck", "queue", "backlog_min"]]
                    .round(1), use_container_width=True, hide_index=True)
    right.write("**Last decision: reward components**")
    if sim["last"]:
        right.dataframe(pd.Series(sim["last"], name="contribution").round(3), use_container_width=True)

    st.write("**Recent dispatch decisions**")
    st.dataframe(pd.DataFrame(sim["log"][-20:]), use_container_width=True, hide_index=True)
    st.write("**Breakdown and repair events (latest 25)**")
    events = pd.DataFrame(snap["event_log"], columns=["time_min", "event", "entity", "duration_min"])
    st.dataframe(events.round(2), use_container_width=True, hide_index=True)
    if snap["done"] and sim["final"]:
        st.success(f"Shift finished. Episode return {sim['final']['return']:.2f}; payload "
                   f"{sim['final']['payload_t']:.0f} t; average truck wait {sim['final']['avg_wait_min']:.2f} min.")


def page_analytics(config: ProjectConfig) -> None:
    results = config.resolve(config.paths.results_dir)
    training_dir = results / "training"
    runs = sorted(p.parent.name for p in training_dir.glob("*/episodes.csv")) if training_dir.exists() else []
    st.subheader("Training curves")
    if not runs:
        st.info("No training logs found in results/training.")
    else:
        run = st.selectbox("Training run", runs)
        df = pd.read_csv(training_dir / run / "episodes.csv")
        window = st.slider("Moving-average window (episodes)", 1, 100, 20)
        chart = pd.DataFrame({
            "raw return": df["episode_return"],
            "moving average": moving_average(df["episode_return"].to_numpy(), window),
        }, index=df["episode"])
        st.line_chart(chart)
        cols = st.columns(2)
        cols[0].line_chart(pd.DataFrame({"epsilon": df["epsilon"]}, index=df["episode"]))
        cols[1].line_chart(pd.DataFrame({"mean loss": df["mean_loss"]}, index=df["episode"]))
        components = pd.DataFrame({f"r_{c}": moving_average(df[f"r_{c}"].to_numpy(), window)
                                   for c in REWARD_COMPONENTS}, index=df["episode"])
        st.write("**Reward components per episode (moving average)**")
        st.line_chart(components)

    st.subheader("Evaluation episodes")
    episodes = read_csv_if_exists(results / "evaluation" / "episodes.csv")
    if episodes is None:
        st.info("No evaluation episodes found.")
    else:
        st.dataframe(episodes.round(2), use_container_width=True, height=280)
        paired = read_csv_if_exists(results / "evaluation" / "paired_comparisons.csv")
        if paired is not None:
            st.write("**Paired comparisons (identical seeds)**")
            st.dataframe(paired.round(4), use_container_width=True, hide_index=True)
    figures_dir = results / "figures"
    for path in sorted(figures_dir.glob("*.png")) if figures_dir.exists() else []:
        st.image(str(path), caption=path.name, use_container_width=True)


def page_compare(config: ProjectConfig) -> None:
    st.subheader("Run every policy live on the same seeds")
    checkpoints = find_checkpoints(config)
    checkpoint = None
    if checkpoints:
        labels = [str(p.relative_to(ROOT)) for p in checkpoints]
        checkpoint = checkpoints[labels.index(st.selectbox("DQN checkpoint", labels))]
    policies = st.multiselect("Policies", list(ALL_POLICIES),
                              default=[p for p in ALL_POLICIES if p != "dqn" or checkpoint is not None])
    n_seeds = st.slider("Number of held-out seeds", 1, 20, 5)
    if st.button("Run comparison", disabled=not policies):
        agent = load_policy_agent(checkpoint)[0] if (checkpoint is not None and "dqn" in policies) else None
        runnable = [p for p in policies if p != "dqn" or agent is not None]
        st.session_state.compare = evaluate_policies(
            config, runnable, evaluation_seeds(config.evaluation.seed_start, n_seeds), agent=agent)
    frame = st.session_state.get("compare")
    if frame is not None:
        st.dataframe(frame.groupby("policy")[["episode_return", "payload_t", "avg_wait_min",
                                               "shovel_starvation_min", "truck_breakdowns", "runtime_s"]]
                     .agg(["mean", "std"]).round(2), use_container_width=True)
        st.bar_chart(frame.groupby("policy")["payload_t"].mean())


def main() -> None:
    st.title("Intelligent fleet dispatching with masked Double DQN")
    try:
        config = get_config()
    except ConfigError as error:
        st.error(f"Configuration error: {error}")
        return
    overview, simulation, analytics, compare = st.tabs(["Overview", "Simulation", "Analytics", "Compare"])
    with overview:
        page_overview(config)
    with simulation:
        page_simulation(config)
    with analytics:
        page_analytics(config)
    with compare:
        page_compare(config)


main()
