"""Data side of the dashboard's *learning arena* (no Streamlit, no pygame: importable and testable anywhere).

The arena answers one question visually: **what had the network learned after N training episodes?**
It replays the very same start condition with the greedy policy stored in each training snapshot
(``checkpoints/<run>/snapshots/ep_<N>.pt``, written by ``scripts/train.py``) and hands the recorded
trajectories to a self-contained HTML/canvas page (``arena_template.html``) that draws the mechanism the same way
the Pygame simulator does.

Nothing is invented: every frame comes from :class:`~pendulum_rl.environment.PendulumEnv` driven by the saved
network. A snapshot of an untrained network is simply played back as the (bad) policy it is.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd

from ..agents import DQNAgent
from ..environment import EnvConfig, PendulumEnv, RewardConfig
from ..environment.pendulum_env import PHASES
from ..physics import PhysicsParams

START_MODES = {"hanging": "Hanging at the bottom (swing-up)", "upright": "Near the top (stabilize)",
               "random": "Random angle"}
VIOLATION_TEXT = {
    "rail_limit": "hit the end of the rail", "support_overspeed": "support exceeded its speed limit",
    "pendulum_overspeed": "pendulum spun out of range", "non_finite_state": "numerical failure",
    "numerical_instability": "numerical failure",
}
TEMPLATE = Path(__file__).with_name("arena_template.html")
_SNAPSHOT_FILE = re.compile(r"^ep_(\d+)\.pt$")


@dataclass(frozen=True)
class Snapshot:
    """A saved network and the number of training episodes it had seen."""

    path: Path
    episode: int
    kind: str = "snapshot"            # "snapshot" | "best" | "final"

    @property
    def cache_key(self) -> tuple[str, int]:
        """Changes when the file is rewritten (so cached rollouts of a replaced checkpoint are not reused)."""
        return str(self.path), self.path.stat().st_mtime_ns


# ----------------------------------------------------------------------------- discovering what to show
def list_snapshots(ckpt_run_dir: Path) -> list[Snapshot]:
    """Periodic snapshots of a run, ordered by training episode."""
    folder = Path(ckpt_run_dir) / "snapshots"
    found = []
    for p in folder.glob("ep_*.pt") if folder.is_dir() else []:
        m = _SNAPSHOT_FILE.match(p.name)
        if m:
            found.append(Snapshot(p, int(m.group(1))))
    return sorted(found, key=lambda s: s.episode)


def fallback_checkpoints(ckpt_run_dir: Path) -> list[Snapshot]:
    """``best.pt`` / ``final.pt`` of runs that were trained before snapshots existed (episode read from metadata)."""
    out = []
    for name in ("best", "final"):
        p = Path(ckpt_run_dir) / f"{name}.pt"
        if p.is_file():
            try:
                from ..utils.checkpoints import load_checkpoint
                episodes = int(load_checkpoint(p).get("metadata", {}).get("training", {}).get("episodes", 0))
            except Exception:       # unreadable file: skip it rather than crash the page
                continue
            out.append(Snapshot(p, episodes, name))
    return sorted(out, key=lambda s: (s.episode, s.kind))


def thin_out(snapshots: list[Snapshot], limit: int) -> list[Snapshot]:
    """At most ``limit`` snapshots, evenly spread, always keeping the first and the last."""
    if limit < 2 or len(snapshots) <= limit:
        return list(snapshots)
    idx = sorted({round(i * (len(snapshots) - 1) / (limit - 1)) for i in range(limit)})
    return [snapshots[i] for i in idx]


# ----------------------------------------------------------------------------- recording one rollout
def _num(values, digits: int) -> list:
    return [None if not math.isfinite(v) else round(float(v), digits) for v in values]


def verdict(metrics: dict, env_cfg: EnvConfig) -> dict:
    """One-line, honest summary of how the replayed episode ended."""
    t_up = metrics["time_to_upright"]
    reached = bool(metrics["swing_up_success"])
    reason = metrics["termination_reason"]
    if metrics["violation"]:
        what = VIOLATION_TEXT.get(metrics["violation_reason"], str(metrics["violation_reason"]))
        extra = f" (it had reached the top at {t_up:.1f} s)" if reached and math.isfinite(t_up) and t_up > 0 else ""
        return {"kind": "fail", "title": "FAILED", "detail": f"{what} at t = {metrics['duration']:.1f} s{extra}"}
    ended_up = metrics["final_abs_angle_error"] <= env_cfg.upright_region
    if metrics["balance_success"] and ended_up and reason == "time_limit":
        swing = "started near the top" if not math.isfinite(t_up) or t_up < 0.1 else f"swung up in {t_up:.1f} s"
        return {"kind": "good", "title": "BALANCED",
                "detail": f"{swing}, then held the top ({metrics['balanced_time']:.1f} s balanced)"}
    if reached:
        return {"kind": "warn", "title": "LOST THE TOP",
                "detail": f"reached the top at {t_up:.1f} s but ended {math.degrees(metrics['final_abs_angle_error']):.0f} deg away"}
    if metrics["final_abs_angle_error"] >= 2.8:         # within ~20 deg of straight down
        return {"kind": "bad", "title": "NEVER REACHED THE TOP", "detail": "stayed hanging near the bottom"}
    return {"kind": "bad", "title": "NEVER REACHED THE TOP",
            "detail": f"ended {math.degrees(metrics['final_abs_angle_error']):.0f} deg away from upright"}


def record_rollout(checkpoint: Path, *, start: str = "hanging", seed: int = 1, seconds: float = 10.0,
                   kick_time: float | None = None, kick: float = 4.0, device: str = "cpu") -> dict:
    """Run one greedy episode with the network in ``checkpoint`` and return the trajectory + outcome.

    ``start`` is ``hanging`` / ``upright`` / ``random``. The same ``seed`` gives the same start for every
    checkpoint, so snapshots are directly comparable. ``kick_time`` applies a one-off angular-velocity kick
    of ``kick`` rad/s to test recovery. Physics randomization, disturbances and sensor noise are off.
    """
    if start not in START_MODES:
        raise ValueError(f"start must be one of {sorted(START_MODES)}, got {start!r}")
    if seconds <= 0:
        raise ValueError("seconds must be positive")
    agent, meta = DQNAgent.from_checkpoint(checkpoint, device=device)
    physics = PhysicsParams(**meta["physics"]) if meta.get("physics") else PhysicsParams()
    reward = RewardConfig(**meta["reward"]) if meta.get("reward") else RewardConfig()
    base = EnvConfig(**meta["base_env"]) if meta.get("base_env") else EnvConfig()
    cfg = replace(base, start_mode=start, success_mode="balance", terminate_on_success=False,
                  max_episode_seconds=float(seconds), physics_randomization=0.0, disturbance_probability=0.0,
                  noise_position=0.0, noise_velocity=0.0, noise_angle=0.0, noise_angular_velocity=0.0,
                  action_mode="discrete")
    env = PendulumEnv(cfg, physics, reward)
    cols: dict[str, list] = {k: [] for k in ("t", "x", "xd", "th", "thd", "u", "act", "e", "ret", "ph")}
    ret, kicked_at = 0.0, None

    def add(info: dict) -> None:
        for key, value in (("t", info["time"]), ("x", info["x"]), ("xd", info["x_dot"]), ("th", info["theta"]),
                           ("thd", info["theta_dot"]), ("u", info["command"]), ("act", info["actuator"]),
                           ("e", info["swing_energy"]), ("ret", ret), ("ph", PHASES.index(info["phase"]))):
            cols[key].append(value)

    try:
        obs, info = env.reset(seed=int(seed))
        add(info)
        while True:
            if kick_time is not None and kicked_at is None and env.time >= kick_time - 1e-9:
                env.apply_disturbance(float(kick))
                kicked_at = env.time
            obs, r, terminated, truncated, info = env.step(agent.act(obs, greedy=True))
            ret += r
            add(info)
            if terminated or truncated:
                break
        metrics = info["episode_metrics"]
    finally:
        env.close()

    train = meta.get("training", {})
    return {
        "dt": cfg.control_dt, "n": len(cols["t"]), "kick_t": kicked_at,
        "t": _num(cols["t"], 3), "x": _num(cols["x"], 4), "xd": _num(cols["xd"], 3), "th": _num(cols["th"], 4),
        "thd": _num(cols["thd"], 3), "u": _num(cols["u"], 2), "act": _num(cols["act"], 3),
        "e": _num(cols["e"], 3), "ret": _num(cols["ret"], 2), "ph": [int(p) for p in cols["ph"]],
        "stage": train.get("stage"), "env_steps": int(train.get("env_steps", 0)),
        "verdict": verdict(metrics, cfg),
        "time_to_upright": None if not math.isfinite(metrics["time_to_upright"]) else round(metrics["time_to_upright"], 3),
        "balanced_time": round(metrics["balanced_time"], 3), "duration": round(metrics["duration"], 3),
        "termination": metrics["termination_reason"],
    }


# ----------------------------------------------------------------------------- learning curve
def learning_curve(log_dir: Path, max_points: int = 160) -> dict:
    """Greedy-evaluation success rates logged during training (``eval.csv``), for the arena's curve."""
    path = Path(log_dir) / "eval.csv"
    if not path.is_file():
        return {"episode": [], "balance": [], "swing": [], "scenario": None, "eval_episodes": None}
    ev = pd.read_csv(path)
    if ev.empty or "episode" not in ev:
        return {"episode": [], "balance": [], "swing": [], "scenario": None, "eval_episodes": None}
    if len(ev) > max_points:
        ev = ev.iloc[np.unique(np.linspace(0, len(ev) - 1, max_points).round().astype(int))]
    scenario = ev["scenario"].iloc[-1] if "scenario" in ev else None
    n_eval = int(ev["episodes"].iloc[-1]) if "episodes" in ev and pd.notna(ev["episodes"].iloc[-1]) else None
    return {"episode": [int(v) for v in ev["episode"]], "balance": _num(ev["balance_rate"], 3),
            "swing": _num(ev["swing_up_rate"], 3), "scenario": None if scenario is None else str(scenario),
            "eval_episodes": n_eval}


# ----------------------------------------------------------------------------- HTML
def _geometry(physics: dict | None) -> dict:
    p = PhysicsParams(**physics) if physics else PhysicsParams()
    return {"rail_half": p.rail_half_length, "length": p.pendulum_length, "max_force": p.max_force,
            "max_speed": p.max_support_speed}


def build_arena_html(run: str, rollouts: list[dict], curve: dict, *, start: str, seed: int, seconds: float,
                     physics: dict | None = None, autoplay: bool = True) -> str:
    """Self-contained HTML page for ``streamlit.components.v1.html`` (or to save as a file and open anywhere).

    ``rollouts`` are dicts from :func:`record_rollout` plus an ``episode`` key, ordered by episode.
    """
    if not rollouts:
        raise ValueError("No rollouts to show.")
    payload = {
        "run": run, "start": START_MODES.get(start, start), "seed": int(seed), "seconds": float(seconds),
        "geometry": _geometry(physics), "curve": curve, "autoplay": bool(autoplay),
        "snapshots": sorted(rollouts, key=lambda r: r["episode"]),
    }
    data = json.dumps(payload, allow_nan=False, separators=(",", ":")).replace("<", "\\u003c")
    return TEMPLATE.read_text(encoding="utf-8").replace("/*__ARENA_DATA__*/null", data)
