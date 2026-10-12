"""Masked Double DQN implemented from scratch with PyTorch.

Learning rule (for a transition (s, a, r, s', terminated), with valid next actions M(s')):

    a*      = argmax_{a' in M(s')} Q_online(s', a')      # selection: online network
    y       = r + gamma * (1 - terminated) * Q_target(s', a*)   # evaluation: target network
    loss    = Huber(Q_online(s, a), y)                    # targets are not differentiated

Notes on the decisions made here:
  * `terminated` zeroes the bootstrap. `truncated` (the shift horizon) does not. The time limit is not
    part of the environment dynamics, so the value of the state where it was hit is still the value of
    what would follow.
  * If s' has no valid action, the bootstrap is 0. There is no action-value to bootstrap from. This
    only happens when the horizon leaves no pending decision.
  * Exploration samples only valid actions, so an epsilon-greedy step can never be an invalid action.
  * Only the target network's weights are synchronised hard, every `target_update_interval` gradient
    steps. Soft (Polyak) updates would replace that with theta_target <- tau * theta + (1 - tau) * theta_target
    every step, which gives smoother targets at the cost of more hyperparameters.
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from fleet_dispatch.agents.action_masking import (
    greedy_valid_action,
    has_valid_action,
    masked_argmax,
    masked_max,
    sample_valid_action,
)
from fleet_dispatch.agents.networks import QNetwork
from fleet_dispatch.agents.replay_buffer import Batch, ReplayBuffer
from fleet_dispatch.config import AgentConfig, ProjectConfig, config_from_dict
from fleet_dispatch.utils.logging import get_logger

CHECKPOINT_FORMAT_VERSION = 1
_log = get_logger(__name__)


def resolve_device(requested: str) -> torch.device:
    """'auto' picks CUDA when available. An explicit 'cuda' raises if CUDA is unavailable, rather than silently using the CPU."""
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("device 'cuda' was requested but torch.cuda.is_available() is False")
    return torch.device(requested)


class DQNAgent:
    def __init__(
        self,
        obs_dim: int,
        n_actions: int,
        cfg: AgentConfig,
        device: torch.device | str = "cpu",
    ) -> None:
        self.cfg = cfg
        self.obs_dim = obs_dim
        self.n_actions = n_actions
        self.device = torch.device(device)
        torch.manual_seed(cfg.seed)
        self.rng = np.random.default_rng(cfg.seed)

        self.online = QNetwork(obs_dim, n_actions, cfg.hidden_sizes).to(self.device)
        self.target = QNetwork(obs_dim, n_actions, cfg.hidden_sizes).to(self.device)
        self.target.load_state_dict(self.online.state_dict())
        self.target.eval()
        self.optimizer = torch.optim.Adam(self.online.parameters(), lr=cfg.learning_rate)
        self.buffer = ReplayBuffer(cfg.replay_capacity, obs_dim, n_actions, self.rng)
        self.env_steps = 0
        self.grad_steps = 0

    # ------------------------------------------------------------------ exploration and acting
    @property
    def epsilon(self) -> float:
        """Linear decay from epsilon_start to epsilon_end over epsilon_decay_steps environment steps."""
        fraction = min(self.env_steps / max(self.cfg.epsilon_decay_steps, 1), 1.0)
        return self.cfg.epsilon_start + fraction * (self.cfg.epsilon_end - self.cfg.epsilon_start)

    @torch.no_grad()
    def q_values(self, obs: np.ndarray) -> np.ndarray:
        tensor = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        return self.online(tensor)[0].cpu().numpy()

    def select_action(self, obs: np.ndarray, mask: np.ndarray, explore: bool = False) -> int:
        """Epsilon-greedy over valid actions when exploring. Otherwise greedy over valid actions."""
        if explore and self.rng.random() < self.epsilon:
            return sample_valid_action(mask, self.rng)
        return greedy_valid_action(self.q_values(obs), mask)

    # ------------------------------------------------------------------ learning
    def observe(
        self,
        obs: np.ndarray,
        action: int,
        reward: float,
        next_obs: np.ndarray,
        terminated: bool,
        truncated: bool,
        mask: np.ndarray,
        next_mask: np.ndarray,
    ) -> None:
        self.buffer.add(obs, action, reward, next_obs, terminated, truncated, mask, next_mask)
        self.env_steps += 1

    def maybe_learn(self) -> dict[str, float] | None:
        """Run one gradient step if enough data is available and the schedule says to. Returns metrics or None."""
        cfg = self.cfg
        if len(self.buffer) < max(cfg.learning_starts, cfg.batch_size):
            return None
        if self.env_steps % cfg.train_every_steps != 0:
            return None
        return self.update(self.buffer.sample(cfg.batch_size))

    def _to_tensors(self, batch: Batch) -> dict[str, torch.Tensor]:
        d = self.device
        return {
            "obs": torch.as_tensor(batch.obs, device=d),
            "actions": torch.as_tensor(batch.actions, device=d),
            "rewards": torch.as_tensor(batch.rewards, device=d),
            "next_obs": torch.as_tensor(batch.next_obs, device=d),
            "terminated": torch.as_tensor(batch.terminated, device=d),
            "next_masks": torch.as_tensor(batch.next_masks, device=d),
        }

    @torch.no_grad()
    def compute_targets(
        self,
        rewards: torch.Tensor,
        next_obs: torch.Tensor,
        next_masks: torch.Tensor,
        terminated: torch.Tensor,
    ) -> torch.Tensor:
        """Masked (Double) DQN Bellman targets. Targets carry no gradient."""
        gamma = self.cfg.gamma
        valid_next = has_valid_action(next_masks)
        if self.cfg.double_dqn:
            best_next = masked_argmax(self.online(next_obs), next_masks)
            evaluated = self.target(next_obs).gather(1, best_next.unsqueeze(1)).squeeze(1)
        else:
            evaluated = masked_max(self.target(next_obs), next_masks)
        bootstrap = torch.where(valid_next, evaluated, torch.zeros_like(evaluated))
        not_terminal = (~terminated).to(rewards.dtype)
        return rewards + gamma * not_terminal * bootstrap

    def update(self, batch: Batch) -> dict[str, float]:
        t = self._to_tensors(batch)
        targets = self.compute_targets(t["rewards"], t["next_obs"], t["next_masks"], t["terminated"])
        q_taken = self.online(t["obs"]).gather(1, t["actions"].unsqueeze(1)).squeeze(1)
        loss = F.huber_loss(q_taken, targets, delta=self.cfg.huber_delta)

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = nn.utils.clip_grad_norm_(self.online.parameters(), self.cfg.grad_clip_norm)
        self.optimizer.step()
        self.grad_steps += 1
        if self.grad_steps % self.cfg.target_update_interval == 0:
            self.sync_target()
        return {
            "loss": float(loss.item()),
            "q_taken_mean": float(q_taken.mean().item()),
            "target_mean": float(targets.mean().item()),
            "grad_norm": float(grad_norm),
        }

    def sync_target(self) -> None:
        self.target.load_state_dict(self.online.state_dict())

    # ------------------------------------------------------------------ checkpointing
    def save(self, path: str | Path, *, config: dict[str, Any], episodes_completed: int,
             include_buffer: bool = False) -> None:
        """Atomically write a checkpoint. The full project configuration is embedded in it."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "format_version": CHECKPOINT_FORMAT_VERSION,
            "obs_dim": self.obs_dim,
            "n_actions": self.n_actions,
            "online": self.online.state_dict(),
            "target": self.target.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "env_steps": self.env_steps,
            "grad_steps": self.grad_steps,
            "episodes_completed": episodes_completed,
            "numpy_rng_state": self.rng.bit_generator.state,
            "torch_rng_state": torch.get_rng_state(),
            "buffer": self.buffer.state_dict() if include_buffer else None,
            "config": config,
            "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        tmp = path.with_suffix(path.suffix + ".tmp")
        torch.save(payload, tmp)
        os.replace(tmp, path)
        _log.info("saved checkpoint %s (episodes=%d, env_steps=%d)", path, episodes_completed, self.env_steps)

    def load_training_state(self, path: str | Path) -> dict[str, Any]:
        """Restore everything needed to resume training. The replay buffer is restored only if it was saved."""
        payload = _read_checkpoint(path, self.device)
        _check_shapes(payload, self.obs_dim, self.n_actions, path)
        self.online.load_state_dict(payload["online"])
        self.target.load_state_dict(payload["target"])
        self.optimizer.load_state_dict(payload["optimizer"])
        self.env_steps = int(payload["env_steps"])
        self.grad_steps = int(payload["grad_steps"])
        self.rng.bit_generator.state = payload["numpy_rng_state"]
        torch.set_rng_state(payload["torch_rng_state"].cpu())  # RNG state is always a CPU tensor
        if payload.get("buffer") is not None:
            self.buffer.load_state_dict(payload["buffer"])
        return payload

    def load_weights(self, path: str | Path) -> dict[str, Any]:
        """Load network weights only, for evaluation. Optimiser and buffer are not needed."""
        payload = _read_checkpoint(path, self.device)
        _check_shapes(payload, self.obs_dim, self.n_actions, path)
        self.online.load_state_dict(payload["online"])
        self.target.load_state_dict(payload["target"])
        self.online.eval()
        return payload


class DQNPolicy:
    """Greedy policy adapter that lets a trained agent be evaluated next to the baselines."""

    name = "dqn"

    def __init__(self, agent: DQNAgent) -> None:
        self.agent = agent

    def act(self, view) -> int:  # view: fleet_dispatch.environment.mining_env.DecisionView
        return self.agent.select_action(view.observation, view.action_mask, explore=False)


def _read_checkpoint(path: str | Path, device: torch.device) -> dict[str, Any]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"checkpoint not found: {path}")
    payload = torch.load(path, map_location=device, weights_only=False)
    if payload.get("format_version") != CHECKPOINT_FORMAT_VERSION:
        raise ValueError(f"{path} has checkpoint format {payload.get('format_version')}, "
                         f"expected {CHECKPOINT_FORMAT_VERSION}")
    return payload


def _check_shapes(payload: dict[str, Any], obs_dim: int, n_actions: int, path: str | Path) -> None:
    if payload["obs_dim"] != obs_dim or payload["n_actions"] != n_actions:
        raise ValueError(
            f"{path} was trained for obs_dim={payload['obs_dim']}, n_actions={payload['n_actions']}, "
            f"but the environment has obs_dim={obs_dim}, n_actions={n_actions}"
        )


def load_policy_agent(path: str | Path, device: str = "cpu") -> tuple[DQNAgent, ProjectConfig]:
    """Rebuild an agent from a checkpoint's embedded configuration, for inference."""
    dev = resolve_device(device)
    payload = _read_checkpoint(path, dev)
    config = config_from_dict(payload["config"])
    agent = DQNAgent(payload["obs_dim"], payload["n_actions"], config.agent, dev)
    agent.load_weights(path)
    agent.env_steps = int(payload["env_steps"])
    return agent, config
