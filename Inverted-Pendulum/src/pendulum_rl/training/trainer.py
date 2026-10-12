"""DQN training loop with staged curriculum, periodic greedy evaluation and logging."""

from __future__ import annotations

import dataclasses
import math
import shutil
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Mapping

import numpy as np
import torch
import yaml

from ..agents import DQNAgent, DQNConfig
from ..config import PROJECT_ROOT, build_dqn_config, build_env_config, build_physics, build_reward, dataclass_from_dict
from ..environment import EnvConfig, PendulumEnv, RewardConfig
from ..evaluation.evaluate import classify_status, evaluate_agent_quick
from ..physics import PhysicsParams
from ..utils.checkpoints import load_checkpoint
from ..utils.seeding import derive_seed, seed_everything
from .callbacks import Callback, CheckpointCallback, ConsoleProgress, CSVLogger, SnapshotCallback, TensorBoardLogger
from .curriculum import STAGE_SCENARIO, Curriculum, StageConfig, stages_from_config

MODES = ("stabilize", "swingup", "combined", "curriculum")


@dataclass(frozen=True)
class TrainConfig:
    mode: str = "combined"                  # stabilize | swingup | combined | curriculum
    episodes: int | None = None             # if set, overrides every stage's max_episodes
    seed: int = 0
    device: str = "auto"
    torch_threads: int = 1                  # CPU threads for PyTorch (tiny MLPs train fastest with 1)
    run_name: str | None = None             # default: the mode name
    checkpoint_dir: str = "checkpoints"
    log_dir: str = "logs"
    eval_every: int = 25                    # training episodes between greedy evaluations
    eval_episodes: int = 10
    eval_seed: int = 50_000
    snapshot_every: int = 25                # save the network every N episodes (dashboard "learning arena"); 0 = off
    print_every: int = 10
    tensorboard: bool = False
    overwrite: bool = False
    render: bool = False                    # show training episodes in a window (real-time speed, for debugging)
    init_from: str | None = None            # warm-start network weights from this checkpoint
    curriculum_order: tuple = ("stabilize", "swingup", "combined")

    def __post_init__(self) -> None:
        if self.mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}, got {self.mode!r}")
        object.__setattr__(self, "curriculum_order", tuple(self.curriculum_order))

    @property
    def name(self) -> str:
        return self.run_name or self.mode


def _resolve(path: str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else PROJECT_ROOT / p


class Trainer:
    def __init__(self, physics: PhysicsParams, env_cfg: EnvConfig, reward: RewardConfig, dqn_cfg: DQNConfig,
                 train_cfg: TrainConfig, stage_table: Mapping[str, Mapping], callbacks: list[Callback] | None = None):
        self.physics, self.base_env_cfg, self.reward, self.dqn_cfg, self.cfg = physics, env_cfg, reward, dqn_cfg, train_cfg
        order = list(train_cfg.curriculum_order) if train_cfg.mode == "curriculum" else [train_cfg.mode]
        self.stages: list[StageConfig] = stages_from_config(stage_table, order, train_cfg.episodes)
        self.curriculum = Curriculum(self.stages, promote=train_cfg.mode == "curriculum")

        self.ckpt_dir = _resolve(train_cfg.checkpoint_dir) / train_cfg.name
        self.log_dir = _resolve(train_cfg.log_dir) / train_cfg.name
        self._prepare_dirs()

        self.callbacks: list[Callback] = callbacks if callbacks is not None else self.default_callbacks()
        seed_everything(train_cfg.seed)
        torch.set_num_threads(max(1, train_cfg.torch_threads))
        probe = self._stage_env_cfg(self.stages[0])
        self.n_actions = len(probe.action_levels)
        with PendulumEnv(probe, physics, self._stage_reward(self.stages[0])) as env:
            self.obs_dim = env.obs_dim
        self.agent = DQNAgent(self.obs_dim, self.n_actions, dqn_cfg, device=train_cfg.device, seed=train_cfg.seed)
        if train_cfg.init_from:
            payload = load_checkpoint(train_cfg.init_from, map_location=self.agent.device)
            if payload["obs_dim"] != self.obs_dim or payload["n_actions"] != self.n_actions:
                raise ValueError(f"--init-from checkpoint expects obs_dim={payload['obs_dim']}, n_actions="
                                 f"{payload['n_actions']} but this run uses {self.obs_dim}/{self.n_actions}")
            self.agent.load_weights(payload, load_optimizer=False)
            self.agent.steps = self.agent.updates = 0
        self.global_episode = 0
        self.stage_name = self.stages[0].name
        self.last_eval: dict | None = None
        self._t0 = time.time()

    # ------------------------------------------------------------------ setup helpers
    def _prepare_dirs(self) -> None:
        existing = [p for p in (self.ckpt_dir, self.log_dir) if p.exists() and any(p.iterdir())]
        if existing and not self.cfg.overwrite:
            raise FileExistsError(
                f"Run '{self.cfg.name}' already has results in: {', '.join(map(str, existing))}. "
                f"Existing trained checkpoints are never overwritten implicitly: choose another --run-name, "
                f"or pass --overwrite if you really want to replace them.")
        for p in existing:
            shutil.rmtree(p)
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)

    def default_callbacks(self) -> list[Callback]:
        cbs: list[Callback] = [CheckpointCallback(self.ckpt_dir), CSVLogger(self.log_dir),
                               SnapshotCallback(self.ckpt_dir, self.cfg.snapshot_every)]
        if self.cfg.tensorboard:
            cbs.append(TensorBoardLogger(self.log_dir))
        cbs.append(ConsoleProgress(self.cfg.print_every))
        return cbs

    def _stage_env_cfg(self, stage: StageConfig) -> EnvConfig:
        cfg = replace(self.base_env_cfg, **stage.env, action_mode="discrete")
        if len(cfg.action_levels) != len(self.base_env_cfg.action_levels) or \
                cfg.include_prev_action != self.base_env_cfg.include_prev_action:
            raise ValueError(f"Stage '{stage.name}' may not change action_levels / include_prev_action "
                             f"(the network's input/output sizes are fixed)")
        return cfg

    def _stage_reward(self, stage: StageConfig) -> RewardConfig:
        return replace(self.reward, **stage.reward)

    def metadata(self, last_eval: dict | None = None) -> dict:
        """Everything needed to rebuild the matching environment and to label the checkpoint."""
        stage = self.curriculum.stage if not self.curriculum.finished else self.stages[-1]
        env_dict = dataclasses.asdict(self._stage_env_cfg(stage))
        env_dict["action_levels"] = list(env_dict["action_levels"])
        env_dict["action_mode"] = "discrete"
        base_dict = dataclasses.asdict(self.base_env_cfg)
        base_dict["action_levels"] = list(base_dict["action_levels"])
        primary = last_eval["primary_rate"] if last_eval else None
        return {
            "physics": self.physics.to_dict(),
            "env": env_dict,
            "base_env": base_dict,
            "reward": dataclasses.asdict(self._stage_reward(stage)),
            "training": {
                "mode": self.cfg.mode, "stage": self.stage_name, "seed": self.cfg.seed,
                "episodes": self.global_episode, "env_steps": self.agent.steps, "updates": self.agent.updates,
                "wall_time_s": time.time() - self._t0,
            },
            "last_eval": _plain(last_eval) if last_eval else None,
            "status": classify_status(self.agent.steps, primary) if primary is not None else "unevaluated",
        }

    # ------------------------------------------------------------------ main loop
    def train(self) -> dict:
        self._emit("on_train_start")
        (self.log_dir / "run_config.yaml").write_text(yaml.safe_dump(
            {"train": _plain(dataclasses.asdict(self.cfg)), "dqn": self.dqn_cfg.to_dict(),
             "stages": [_plain(dataclasses.asdict(s)) for s in self.stages], **{
                 k: v for k, v in self.metadata().items() if k in ("physics", "base_env", "reward")}},
            sort_keys=False), encoding="utf-8")
        interrupted = False
        try:
            while not self.curriculum.finished:
                reason = self._run_stage(self.curriculum.stage)
                self._emit("on_stage_end", self.curriculum.stage, reason)
                self.curriculum.advance()
        except KeyboardInterrupt:
            interrupted = True
            print("\nInterrupted: saving the current network as final.pt", flush=True)
        stage = self.stages[min(self.curriculum.index, len(self.stages) - 1)]
        final_eval = self._evaluate(stage, final=True)
        summary = {
            "run": self.cfg.name, "mode": self.cfg.mode, "interrupted": interrupted,
            "episodes": self.global_episode, "env_steps": self.agent.steps, "updates": self.agent.updates,
            "wall_time_s": time.time() - self._t0, "final_eval": _plain(final_eval),
            "status": classify_status(self.agent.steps, final_eval["primary_rate"]),
            "device": str(self.agent.device),
        }
        self._emit("on_train_end", summary)
        return summary

    def _emit(self, hook: str, *args) -> None:
        for cb in self.callbacks:
            getattr(cb, hook)(self, *args)

    def _run_stage(self, stage: StageConfig) -> str:
        self.stage_name = stage.name
        env = PendulumEnv(self._stage_env_cfg(stage), self.physics, self._stage_reward(stage),
                          render_mode="human" if self.cfg.render else None)
        if self.curriculum.index > 0:
            self.agent.buffer.clear()   # rewards/dynamics of the new task differ from the old ones
            self.agent.restart_exploration(stage.exploration_restart, decay_steps=stage.exploration_decay_steps)
        self._emit("on_stage_start", stage)
        reason = "budget exhausted"
        try:
            while True:
                self._run_episode(env, stage)
                k = self.curriculum.episodes_in_stage
                if k % self.cfg.eval_every == 0:
                    self._evaluate(stage)
                if self.curriculum.should_advance():
                    if k < stage.max_episodes:
                        reason = f"promoted (rolling success {self.curriculum.rolling_success:.2f} >= {stage.promote_success_rate})"
                    break
            if k % self.cfg.eval_every != 0:
                self._evaluate(stage)
        finally:
            env.close()
        return reason

    def _run_episode(self, env: PendulumEnv, stage: StageConfig) -> None:
        agent = self.agent
        obs, _ = env.reset(seed=derive_seed(self.cfg.seed, self.curriculum.index, self.curriculum.episodes_in_stage))
        stats: dict[str, list] = {}
        while True:
            action = agent.act(obs)
            next_obs, reward, terminated, truncated, info = env.step(action)
            agent.remember(obs, action, reward, next_obs, terminated)   # truncation is NOT terminal
            if agent.should_update():
                for _ in range(agent.cfg.gradient_steps):
                    s = agent.update()
                    if s is not None:
                        if not math.isfinite(s["loss"]):
                            raise RuntimeError("Training diverged: non-finite TD loss. Lower dqn.learning_rate "
                                               "or check the reward scale.")
                        for k, v in s.items():
                            stats.setdefault(k, []).append(v)
            obs = next_obs
            if terminated or truncated:
                break

        m = info["episode_metrics"]
        self.curriculum.record(m["success"])
        self.global_episode += 1
        n = max(1, m["steps"])
        mean = lambda key: float(np.mean(stats[key])) if key in stats else math.nan  # noqa: E731
        record = {
            "stage": stage.name, "episode": self.global_episode, "stage_episode": self.curriculum.episodes_in_stage,
            "env_steps": agent.steps, "updates": agent.updates,
            "return": m["return"], "length": m["steps"], "success": m["success"],
            "swing_up_success": m["swing_up_success"], "balance_success": m["balance_success"],
            "balanced_time": m["balanced_time"], "longest_balance": m["longest_balance"],
            "time_to_upright": m["time_to_upright"], "mean_abs_angle_error": m["mean_abs_angle_error"],
            "final_abs_angle_error": m["final_abs_angle_error"], "effort_abs": m["effort_abs"],
            "termination_reason": m["termination_reason"], "epsilon": agent.epsilon,
            "loss": mean("loss"), "q_mean": mean("q_mean"), "q_max": mean("q_max"), "q_min": mean("q_min"),
            "td_error_abs": mean("td_error_abs"), "grad_norm": mean("grad_norm"),
            **{f"action_frac_{i}": c / n for i, c in enumerate(m["action_counts"])},
            "rolling_success": self.curriculum.rolling_success, "wall_time": time.time() - self._t0,
        }
        self._emit("on_episode_end", record)

    def _evaluate(self, stage: StageConfig, final: bool = False) -> dict:
        scenario = STAGE_SCENARIO.get(stage.name, "combined")
        episodes = self.cfg.eval_episodes * (3 if final else 1)
        result = evaluate_agent_quick(self.agent, self._stage_env_cfg(stage), self.physics, self._stage_reward(stage),
                                      scenario, episodes, self.cfg.eval_seed)
        self.last_eval = result
        if not final:
            flat = {k: v for k, v in result.items() if not isinstance(v, (dict, str))}
            record = {"stage": stage.name, "scenario": scenario, "episode": self.global_episode,
                      "env_steps": self.agent.steps, **flat}
            self._emit("on_eval", record)
        return result


def _plain(obj):
    """Make nested config/result structures YAML/JSON/torch-safe (tuples -> lists, numpy -> python)."""
    if isinstance(obj, dict):
        return {k: _plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_plain(v) for v in obj]
    if isinstance(obj, np.generic):
        return obj.item()
    return obj


def trainer_from_config(cfg: Mapping, callbacks: list[Callback] | None = None, **train_overrides) -> Trainer:
    """Build a :class:`Trainer` from a merged config dict; ``train_overrides`` replace ``training`` entries."""
    train_cfg = dataclass_from_dict(TrainConfig, {**(cfg.get("training") or {}), **train_overrides}, "training")
    return Trainer(build_physics(cfg), build_env_config(cfg), build_reward(cfg), build_dqn_config(cfg),
                   train_cfg, cfg.get("stages") or {}, callbacks)
