"""Masked Double DQN: Bellman targets, terminal and truncated handling, exploration, learning, and checkpoints."""

from __future__ import annotations

import copy

import numpy as np
import pytest
import torch
from torch import nn

from fleet_dispatch.agents.dqn_agent import DQNAgent, load_policy_agent
from fleet_dispatch.agents.replay_buffer import Batch
from fleet_dispatch.config import AgentConfig

OBS_DIM, N_ACTIONS = 6, 8


class FixedQ(nn.Module):
    """A network whose Q-values do not depend on the input. It makes Bellman targets exact."""

    def __init__(self, q_row: list[float]) -> None:
        super().__init__()
        self.register_buffer("q", torch.tensor(q_row, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.q.expand(x.shape[0], -1)


def _agent_cfg(config, **overrides) -> AgentConfig:
    cfg = copy.deepcopy(config.agent)
    cfg.hidden_sizes = [32, 32]
    cfg.batch_size = 16
    cfg.replay_capacity = 500
    cfg.learning_starts = 16
    cfg.epsilon_start, cfg.epsilon_end, cfg.epsilon_decay_steps = 1.0, 0.1, 100
    for key, value in overrides.items():
        setattr(cfg, key, value)
    return cfg


def _fixed_agent(config, online_row, target_row, **overrides) -> DQNAgent:
    agent = DQNAgent(OBS_DIM, N_ACTIONS, _agent_cfg(config, **overrides), device="cpu")
    agent.online = FixedQ(online_row)
    agent.target = FixedQ(target_row)
    return agent


def _targets(agent: DQNAgent, rewards, mask_rows, terminated) -> np.ndarray:
    n = len(rewards)
    next_obs = torch.zeros(n, OBS_DIM)
    out = agent.compute_targets(
        torch.tensor(rewards, dtype=torch.float32),
        next_obs,
        torch.tensor(mask_rows, dtype=torch.bool),
        torch.tensor(terminated, dtype=torch.bool),
    )
    return out.numpy()


ONLINE = [9.0, 2.0, 5.0, 1.0, 0.0, 0.0, 0.0, 0.0]   # the unmasked argmax is action 0
TARGET = [100.0, 3.0, 2.0, 8.0, 0.0, 0.0, 0.0, 0.0]
MASK = [False, True, True, True, False, False, False, False]


def test_double_dqn_selects_with_online_and_evaluates_with_target(config) -> None:
    agent = _fixed_agent(config, ONLINE, TARGET, gamma=0.9, double_dqn=True)
    y = _targets(agent, rewards=[1.0], mask_rows=[MASK], terminated=[False])
    # online argmax over valid actions is action 2 (value 5). The target network evaluates action 2 -> 2.
    assert y[0] == pytest.approx(1.0 + 0.9 * 2.0)


def test_vanilla_dqn_uses_max_over_valid_target_values(config) -> None:
    agent = _fixed_agent(config, ONLINE, TARGET, gamma=0.9, double_dqn=False)
    y = _targets(agent, rewards=[1.0], mask_rows=[MASK], terminated=[False])
    # max over valid actions of the target network is 8 (action 3). Invalid action 0 (100) is ignored.
    assert y[0] == pytest.approx(1.0 + 0.9 * 8.0)


def test_genuine_terminal_transition_does_not_bootstrap(config) -> None:
    agent = _fixed_agent(config, ONLINE, TARGET, gamma=0.9)
    y = _targets(agent, rewards=[2.5], mask_rows=[MASK], terminated=[True])
    assert y[0] == pytest.approx(2.5)


def test_truncated_transition_still_bootstraps(config) -> None:
    """Time-limit truncation is not a terminal state, so the target includes the discounted next value."""
    agent = _fixed_agent(config, ONLINE, TARGET, gamma=0.9)
    y = _targets(agent, rewards=[2.5], mask_rows=[MASK], terminated=[False])
    assert y[0] == pytest.approx(2.5 + 0.9 * 2.0)


def test_no_valid_next_action_gives_zero_bootstrap(config) -> None:
    agent = _fixed_agent(config, ONLINE, TARGET, gamma=0.9)
    y = _targets(agent, rewards=[1.0], mask_rows=[[False] * N_ACTIONS], terminated=[False])
    assert y[0] == pytest.approx(1.0)


def test_batch_targets_are_computed_row_by_row(config) -> None:
    agent = _fixed_agent(config, ONLINE, TARGET, gamma=0.5)
    y = _targets(agent, rewards=[1.0, 1.0, 1.0],
                 mask_rows=[MASK, MASK, [False] * N_ACTIONS], terminated=[False, True, False])
    np.testing.assert_allclose(y, [1.0 + 0.5 * 2.0, 1.0, 1.0])


def test_greedy_action_never_picks_a_masked_action(config) -> None:
    agent = _fixed_agent(config, ONLINE, TARGET)
    obs = np.zeros(OBS_DIM, dtype=np.float32)
    assert agent.select_action(obs, np.array(MASK), explore=False) == 2


def test_exploration_samples_only_valid_actions(config) -> None:
    agent = _fixed_agent(config, ONLINE, TARGET, epsilon_start=1.0, epsilon_end=1.0)
    obs = np.zeros(OBS_DIM, dtype=np.float32)
    picks = {agent.select_action(obs, np.array(MASK), explore=True) for _ in range(300)}
    assert picks <= {1, 2, 3}
    assert len(picks) > 1, "with epsilon=1 the agent should explore several valid actions"


def test_epsilon_decays_linearly_and_then_holds(config) -> None:
    agent = DQNAgent(OBS_DIM, N_ACTIONS, _agent_cfg(config, epsilon_start=1.0, epsilon_end=0.1,
                                                    epsilon_decay_steps=100), device="cpu")
    assert agent.epsilon == pytest.approx(1.0)
    agent.env_steps = 50
    assert agent.epsilon == pytest.approx(0.55)
    agent.env_steps = 500
    assert agent.epsilon == pytest.approx(0.1)


def test_gradient_steps_reduce_loss_on_a_fixed_batch(config) -> None:
    torch.manual_seed(0)
    agent = DQNAgent(OBS_DIM, N_ACTIONS, _agent_cfg(config, learning_rate=1e-2, target_update_interval=10_000),
                     device="cpu")
    rng = np.random.default_rng(0)
    n = 64
    masks = rng.random((n, N_ACTIONS)) < 0.6
    masks[:, 0] = True
    batch = Batch(
        obs=rng.random((n, OBS_DIM)).astype(np.float32),
        actions=np.zeros(n, dtype=np.int64),
        rewards=rng.random(n).astype(np.float32),
        next_obs=rng.random((n, OBS_DIM)).astype(np.float32),
        terminated=rng.random(n) < 0.1,
        truncated=np.zeros(n, dtype=bool),
        masks=masks,
        next_masks=masks,
    )
    first = agent.update(batch)["loss"]
    for _ in range(60):
        last = agent.update(batch)["loss"]
    assert last < first


def test_target_network_changes_only_at_hard_sync(config) -> None:
    agent = DQNAgent(OBS_DIM, N_ACTIONS, _agent_cfg(config, target_update_interval=3), device="cpu")
    rng = np.random.default_rng(0)
    for _ in range(20):
        agent.observe(rng.random(OBS_DIM).astype(np.float32), 0, 1.0, rng.random(OBS_DIM).astype(np.float32),
                      False, False, np.ones(N_ACTIONS, bool), np.ones(N_ACTIONS, bool))

    def same(a, b):
        return all(torch.equal(p, q) for p, q in zip(a.parameters(), b.parameters()))

    agent.update(agent.buffer.sample(16))
    agent.update(agent.buffer.sample(16))
    assert not same(agent.online, agent.target), "target must not follow online between syncs"
    agent.update(agent.buffer.sample(16))
    assert same(agent.online, agent.target), "hard sync copies the online weights"


def test_checkpoint_round_trip_restores_training_state(config, tmp_path) -> None:
    cfg = _agent_cfg(config)
    agent = DQNAgent(OBS_DIM, N_ACTIONS, cfg, device="cpu")
    rng = np.random.default_rng(1)
    for _ in range(40):
        agent.observe(rng.random(OBS_DIM).astype(np.float32), 1, 0.5, rng.random(OBS_DIM).astype(np.float32),
                      False, True, np.ones(N_ACTIONS, bool), np.ones(N_ACTIONS, bool))
    for _ in range(3):
        agent.maybe_learn()
    path = tmp_path / "latest.pt"
    agent.save(path, config=config.to_dict(), episodes_completed=7, include_buffer=True)

    restored = DQNAgent(OBS_DIM, N_ACTIONS, cfg, device="cpu")
    payload = restored.load_training_state(path)
    assert payload["episodes_completed"] == 7
    assert restored.env_steps == agent.env_steps and restored.grad_steps == agent.grad_steps
    for p, q in zip(agent.online.parameters(), restored.online.parameters()):
        assert torch.equal(p, q)
    assert len(restored.buffer) == len(agent.buffer)

    obs = rng.random(OBS_DIM).astype(np.float32)
    mask = np.ones(N_ACTIONS, bool)
    assert restored.select_action(obs, mask) == agent.select_action(obs, mask)


def test_policy_checkpoint_rebuilds_agent_and_config(config, tmp_path) -> None:
    cfg = _agent_cfg(config)
    agent = DQNAgent(OBS_DIM, N_ACTIONS, cfg, device="cpu")
    path = tmp_path / "final.pt"
    # The embedded configuration must describe the network that was actually built (same hidden sizes).
    saved = copy.deepcopy(config)
    saved.agent = cfg
    agent.save(path, config=saved.to_dict(), episodes_completed=1)
    loaded, loaded_config = load_policy_agent(path)
    assert loaded_config.agent.gamma == cfg.gamma
    assert loaded_config.scenario.n_trucks == config.scenario.n_trucks
    obs = np.linspace(0, 1, OBS_DIM, dtype=np.float32)
    np.testing.assert_allclose(loaded.q_values(obs), agent.q_values(obs), rtol=1e-6)


def test_checkpoint_with_wrong_shape_is_rejected(config, tmp_path) -> None:
    agent = DQNAgent(OBS_DIM, N_ACTIONS, _agent_cfg(config), device="cpu")
    path = tmp_path / "latest.pt"
    agent.save(path, config=config.to_dict(), episodes_completed=1)
    other = DQNAgent(OBS_DIM, N_ACTIONS + 1, _agent_cfg(config), device="cpu")
    with pytest.raises(ValueError):
        other.load_training_state(path)
