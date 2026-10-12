"""Training loop for the masked Double DQN agent.

Each training episode is one 8-hour shift, started from reset seed `seed_base + episode`. Transitions go
into the replay buffer, gradient steps run on the configured schedule, and one raw row per episode is
appended to a CSV. The CSV is the source of truth. Smoothed plots and TensorBoard are derived from it.

Evaluation seeds are held out (checked by ProjectConfig.validate), and training never touches them.
"""

from __future__ import annotations

import csv
import json
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from fleet_dispatch.agents.dqn_agent import DQNAgent, resolve_device
from fleet_dispatch.config import ProjectConfig
from fleet_dispatch.environment.mining_env import FleetDispatchEnv
from fleet_dispatch.environment.reward import REWARD_COMPONENTS
from fleet_dispatch.utils.logging import get_logger
from fleet_dispatch.utils.seeding import seed_everything, training_episode_seed

_log = get_logger(__name__)

TRAIN_COLUMNS = (
    "episode",
    "train_seed",
    "env_steps",
    "epsilon",
    "episode_return",
    "decisions",
    "sim_time",
    "payload_t",
    "loads",
    "avg_wait_min",
    "truck_utilization",
    "shovel_utilization",
    "shovel_starvation_min",
    "truck_breakdowns",
    "shovel_breakdowns",
    "invalid_actions",
    "mean_loss",
    "mean_q",
    "grad_steps",
    *(f"r_{name}" for name in REWARD_COMPONENTS),
    "wall_time_s",
)


@dataclass(frozen=True)
class TrainResult:
    run_name: str
    run_dir: Path
    checkpoint_dir: Path
    episodes_csv: Path
    episodes_completed: int


def _try_summary_writer(log_dir: Path):
    try:
        from torch.utils.tensorboard import SummaryWriter
    except ImportError:  # tensorboard is optional; the CSV is always written
        _log.warning("tensorboard is not installed; logging to CSV only")
        return None
    return SummaryWriter(log_dir=str(log_dir))


def run_training(
    config: ProjectConfig,
    *,
    run_name: str | None = None,
    resume: bool = False,
    device: str | None = None,
) -> TrainResult:
    run_name = run_name or config.training.run_name
    results_root = config.resolve(config.paths.results_dir) / "training" / run_name
    checkpoint_dir = config.resolve(config.paths.checkpoint_dir) / run_name
    latest_path = checkpoint_dir / "latest.pt"
    episodes_csv = results_root / "episodes.csv"
    results_root.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    if not resume and latest_path.exists():
        raise FileExistsError(
            f"run '{run_name}' already has checkpoints in {checkpoint_dir}. Choose another --run-name, "
            "or pass --resume to continue it."
        )
    if resume and not latest_path.exists():
        raise FileNotFoundError(f"cannot resume: {latest_path} does not exist")

    seed_everything(config.agent.seed)
    dev = resolve_device(device or config.agent.device)
    env = FleetDispatchEnv(config)
    agent = DQNAgent(env.obs_dim, env.n_actions, config.agent, dev)

    start_episode = 0
    if resume:
        payload = agent.load_training_state(latest_path)
        start_episode = int(payload["episodes_completed"])
        if payload.get("buffer") is None:
            _log.warning("checkpoint has no replay buffer; resuming with an empty buffer")
        _log.info("resumed %s at episode %d (env_steps=%d)", run_name, start_episode, agent.env_steps)

    (results_root / "run_config.json").write_text(
        json.dumps(config.to_dict(), indent=2, default=str), encoding="utf-8"
    )
    write_header = not (resume and episodes_csv.exists())
    csv_file = episodes_csv.open("a" if resume else "w", newline="", encoding="utf-8")
    writer = csv.DictWriter(csv_file, fieldnames=TRAIN_COLUMNS)
    if write_header:
        writer.writeheader()
    tb = _try_summary_writer(results_root / "tensorboard")
    config_dict = config.to_dict()
    include_buffer = config.training.save_replay_buffer

    try:
        for episode in range(start_episode, config.training.episodes):
            train_seed = training_episode_seed(config.training.seed_base, episode)
            row = _run_episode(env, agent, train_seed)
            row["episode"] = episode
            writer.writerow(row)
            csv_file.flush()
            if tb is not None:
                for key in ("episode_return", "epsilon", "mean_loss", "payload_t", "avg_wait_min",
                            "shovel_utilization", "truck_breakdowns"):
                    if row[key] is not None:
                        tb.add_scalar(f"train/{key}", float(row[key]), episode)
                tb.add_scalar("train/shovel_starvation_min", row["shovel_starvation_min"], episode)
                tb.add_scalar("train/truck_utilization", row["truck_utilization"], episode)
            if episode % 10 == 0 or episode == config.training.episodes - 1:
                _log.info(
                    "episode %4d | return %9.2f | payload %8.0f t | wait %5.2f min | eps %.3f | loss %s",
                    episode, row["episode_return"], row["payload_t"], row["avg_wait_min"], row["epsilon"],
                    "n/a" if row["mean_loss"] is None else f"{row['mean_loss']:.4f}",
                )
            completed = episode + 1
            if completed % config.training.checkpoint_every_episodes == 0 or completed == config.training.episodes:
                agent.save(latest_path, config=config_dict, episodes_completed=completed,
                           include_buffer=include_buffer)
    finally:
        csv_file.close()
        if tb is not None:
            tb.close()

    final_path = checkpoint_dir / "final.pt"
    agent.save(final_path, config=config_dict, episodes_completed=config.training.episodes,
               include_buffer=include_buffer)
    _log.info("training finished: %s", final_path)
    return TrainResult(
        run_name=run_name,
        run_dir=results_root,
        checkpoint_dir=checkpoint_dir,
        episodes_csv=episodes_csv,
        episodes_completed=config.training.episodes,
    )


def _run_episode(env: FleetDispatchEnv, agent: DQNAgent, seed: int) -> dict:
    """One shift of interaction and learning. Returns the raw episode row."""
    started = time.perf_counter()
    obs, info = env.reset(seed=seed)
    mask = info["action_mask"]
    done = False
    losses: list[float] = []
    q_means: list[float] = []
    components: dict[str, float] = defaultdict(float)
    decisions = 0
    episode_info: dict = {}
    while not done:
        action = agent.select_action(obs, mask, explore=True)
        next_obs, reward, terminated, truncated, info = env.step(action)
        next_mask = info["action_mask"]
        agent.observe(obs, action, reward, next_obs, terminated, truncated, mask, next_mask)
        learned = agent.maybe_learn()
        if learned is not None:
            losses.append(learned["loss"])
            q_means.append(learned["q_taken_mean"])
        for name, value in info["reward_components"].items():
            components[name] += value
        obs, mask = next_obs, next_mask
        done = terminated or truncated
        decisions += 1
        if done:
            episode_info = info["episode"]

    row = {
        "train_seed": seed,
        "env_steps": agent.env_steps,
        "epsilon": agent.epsilon,
        "episode_return": episode_info["return"],
        "decisions": decisions,
        "sim_time": episode_info["sim_time"],
        "payload_t": episode_info["payload_t"],
        "loads": episode_info["loads"],
        "avg_wait_min": episode_info["avg_wait_min"],
        "truck_utilization": episode_info["truck_utilization"],
        "shovel_utilization": episode_info["shovel_utilization"],
        "shovel_starvation_min": episode_info["shovel_starvation_min"],
        "truck_breakdowns": episode_info["truck_breakdowns"],
        "shovel_breakdowns": episode_info["shovel_breakdowns"],
        "invalid_actions": episode_info["invalid_actions"],
        "mean_loss": (sum(losses) / len(losses)) if losses else None,
        "mean_q": (sum(q_means) / len(q_means)) if q_means else None,
        "grad_steps": agent.grad_steps,
        "wall_time_s": time.perf_counter() - started,
    }
    for name in REWARD_COMPONENTS:
        row[f"r_{name}"] = components.get(name, 0.0)
    return row
