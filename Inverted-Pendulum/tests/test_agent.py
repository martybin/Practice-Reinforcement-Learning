"""DQN components: replay buffer, Q-network, Bellman targets, updates, target sync and checkpoints."""

import dataclasses

import numpy as np
import pytest
import torch

from pendulum_rl.agents import DQNAgent, DQNConfig, QNetwork, ReplayBuffer
from pendulum_rl.utils import CheckpointExistsError, CheckpointNotFoundError, load_checkpoint, save_checkpoint

OBS, ACTIONS = 6, 3


def small_cfg(**kw):
    base = dict(hidden_dims=(32, 32), batch_size=16, replay_capacity=500, learning_starts=16, target_update_interval=10,
                epsilon_decay_steps=100, double_dqn=False)
    base.update(kw)
    return DQNConfig(**base)


def fill(agent, n=64, seed=0):
    rng = np.random.default_rng(seed)
    for _ in range(n):
        agent.remember(rng.normal(size=OBS), int(rng.integers(ACTIONS)), float(rng.normal()), rng.normal(size=OBS),
                       bool(rng.random() < 0.1))


def test_replay_buffer_sampling_shapes_dtypes_and_wraparound():
    buf = ReplayBuffer(capacity=10, obs_dim=OBS, seed=0)
    with pytest.raises(ValueError):
        buf.sample(4)
    for i in range(25):
        buf.add(np.full(OBS, i), i % ACTIONS, float(i), np.full(OBS, i + 1), i % 7 == 0)
    assert len(buf) == 10                              # capacity respected
    batch = buf.sample(8)
    assert batch["obs"].shape == (8, OBS) and batch["obs"].dtype == torch.float32
    assert batch["actions"].dtype == torch.int64 and batch["actions"].shape == (8,)
    assert batch["rewards"].shape == (8,) and batch["dones"].shape == (8,)
    assert batch["obs"].min() >= 15                    # only the 10 newest transitions (15..24) survive
    assert torch.allclose(batch["next_obs"], batch["obs"] + 1)


def test_replay_buffer_sampling_is_seeded():
    def draw(seed):
        b = ReplayBuffer(50, OBS, seed=seed)
        for i in range(50):
            b.add(np.full(OBS, i), 0, 0.0, np.full(OBS, i), False)
        return b.sample(10)["obs"]
    assert torch.equal(draw(1), draw(1)) and not torch.equal(draw(1), draw(2))


def test_q_network_output_dimensions():
    net = QNetwork(OBS, ACTIONS, hidden_dims=(64, 32, 16))
    assert net(torch.zeros(5, OBS)).shape == (5, ACTIONS)
    assert net(torch.zeros(OBS)).shape == (ACTIONS,)
    with pytest.raises(ValueError):
        QNetwork(OBS, ACTIONS, activation="swish9000")


def test_epsilon_schedule_is_linear_and_restartable():
    agent = DQNAgent(OBS, ACTIONS, small_cfg(epsilon_start=1.0, epsilon_end=0.1, epsilon_decay_steps=100), device="cpu")
    assert agent.epsilon == 1.0
    agent.steps = 50
    assert agent.epsilon == pytest.approx(0.55)
    agent.steps = 1000
    assert agent.epsilon == pytest.approx(0.1)
    agent.restart_exploration(0.4, decay_steps=10)
    assert agent.epsilon == pytest.approx(0.4)
    agent.steps += 10
    assert agent.epsilon == pytest.approx(0.1)


def test_greedy_action_is_argmax_and_deterministic():
    agent = DQNAgent(OBS, ACTIONS, small_cfg(), device="cpu")
    obs = np.random.default_rng(0).normal(size=OBS).astype(np.float32)
    q = agent.q_values(obs)
    assert q.shape == (ACTIONS,)
    assert all(agent.act(obs, greedy=True) == int(np.argmax(q)) for _ in range(10))


def test_exploration_samples_all_actions_when_epsilon_one():
    agent = DQNAgent(OBS, ACTIONS, small_cfg(), device="cpu")
    obs = np.zeros(OBS, dtype=np.float32)
    assert {agent.act(obs) for _ in range(200)} == set(range(ACTIONS))


def test_bellman_target_terminal_vs_nonterminal_vanilla():
    agent = DQNAgent(OBS, ACTIONS, small_cfg(gamma=0.9, double_dqn=False), device="cpu")
    next_obs = torch.randn(4, OBS)
    rewards = torch.tensor([1.0, 2.0, 3.0, 4.0])
    dones = torch.tensor([0.0, 1.0, 0.0, 1.0])
    y = agent.td_targets(rewards, next_obs, dones)
    next_q = agent.target_net(next_obs).max(dim=1).values
    assert torch.allclose(y[dones == 1], rewards[dones == 1])               # no bootstrapping at termination
    assert torch.allclose(y[dones == 0], rewards[dones == 0] + 0.9 * next_q[dones == 0], atol=1e-6)


def test_double_dqn_target_uses_online_argmax_and_target_value():
    agent = DQNAgent(OBS, ACTIONS, small_cfg(gamma=0.9, double_dqn=True), device="cpu")
    for p in agent.target_net.parameters():          # make target differ from online
        p.data += 0.3 * torch.randn_like(p)
    next_obs = torch.randn(8, OBS)
    y = agent.td_targets(torch.zeros(8), next_obs, torch.zeros(8))
    a_star = agent.q_net(next_obs).argmax(1)
    expected = 0.9 * agent.target_net(next_obs).gather(1, a_star[:, None]).squeeze(1)
    assert torch.allclose(y, expected, atol=1e-6)


def test_update_requires_enough_data_then_changes_weights():
    agent = DQNAgent(OBS, ACTIONS, small_cfg(), device="cpu")
    assert agent.update() is None and not agent.should_update()
    fill(agent, 64)
    assert agent.should_update()
    before = [p.detach().clone() for p in agent.q_net.parameters()]
    stats = agent.update()
    assert set(stats) >= {"loss", "q_mean", "q_max", "q_min", "td_error_abs", "grad_norm"}
    assert np.isfinite(stats["loss"]) and stats["q_min"] <= stats["q_mean"] <= stats["q_max"]
    assert any(not torch.equal(b, p) for b, p in zip(before, agent.q_net.parameters()))
    assert agent.updates == 1


def test_dqn_learns_a_fixed_regression_problem():
    """Terminal transitions with constant reward: Q(s, a) must converge to the reward."""
    torch.manual_seed(0)
    agent = DQNAgent(OBS, ACTIONS, small_cfg(learning_rate=3e-3, batch_size=32, replay_capacity=256), device="cpu", seed=0)
    rng = np.random.default_rng(0)
    for _ in range(256):
        a = int(rng.integers(ACTIONS))
        agent.remember(rng.normal(size=OBS), a, [1.0, -1.0, 0.5][a], rng.normal(size=OBS), True)
    first = agent.update()["loss"]
    for _ in range(400):
        last = agent.update()["loss"]
    assert last < 0.1 * first
    q = agent.q_values(rng.normal(size=OBS).astype(np.float32))
    assert np.allclose(q, [1.0, -1.0, 0.5], atol=0.3)


def test_hard_target_update_interval():
    agent = DQNAgent(OBS, ACTIONS, small_cfg(target_update_interval=5), device="cpu")
    fill(agent, 64)
    for _ in range(4):
        agent.update()
    assert not all(torch.equal(a, b) for a, b in zip(agent.q_net.parameters(), agent.target_net.parameters()))
    agent.update()                                    # 5th gradient step triggers the copy
    assert all(torch.equal(a, b) for a, b in zip(agent.q_net.parameters(), agent.target_net.parameters()))


def test_soft_target_update_moves_towards_online_network():
    agent = DQNAgent(OBS, ACTIONS, small_cfg(tau=0.5), device="cpu")
    fill(agent, 64)
    old = [p.detach().clone() for p in agent.target_net.parameters()]
    agent.update()
    for o, t, q in zip(old, agent.target_net.parameters(), agent.q_net.parameters()):
        assert torch.allclose(t, 0.5 * o + 0.5 * q, atol=1e-6)


def test_gradient_clipping_bounds_update_norm():
    agent = DQNAgent(OBS, ACTIONS, small_cfg(grad_clip_norm=0.5), device="cpu")
    rng = np.random.default_rng(0)
    for _ in range(64):
        agent.remember(rng.normal(size=OBS), 0, 1e4, rng.normal(size=OBS), True)     # huge rewards -> huge gradients
    agent.update()
    total = torch.sqrt(sum((p.grad ** 2).sum() for p in agent.q_net.parameters()))
    assert total <= 0.5 + 1e-4


def test_checkpoint_roundtrip_reproduces_q_values(tmp_path):
    agent = DQNAgent(OBS, ACTIONS, small_cfg(), device="cpu", seed=3)
    fill(agent, 64)
    for _ in range(5):
        agent.update()
    path = agent.save(tmp_path / "ckpt.pt", metadata={"note": "hello", "nested": {"a": [1, 2]}})
    loaded, meta = DQNAgent.from_checkpoint(path, device="cpu")
    obs = np.random.default_rng(0).normal(size=OBS).astype(np.float32)
    assert np.allclose(agent.q_values(obs), loaded.q_values(obs), atol=1e-7)
    assert meta == {"note": "hello", "nested": {"a": [1, 2]}}
    assert loaded.steps == agent.steps and loaded.updates == agent.updates
    assert loaded.cfg.hidden_dims == agent.cfg.hidden_dims and dataclasses.asdict(loaded.cfg)["gamma"] == agent.cfg.gamma


def test_checkpoint_overwrite_protection_and_missing_file(tmp_path):
    save_checkpoint(tmp_path / "a.pt", {"x": 1})
    with pytest.raises(CheckpointExistsError):
        save_checkpoint(tmp_path / "a.pt", {"x": 2})
    save_checkpoint(tmp_path / "a.pt", {"x": 3}, overwrite=True)
    assert load_checkpoint(tmp_path / "a.pt")["x"] == 3
    with pytest.raises(CheckpointNotFoundError, match="does not exist"):
        load_checkpoint(tmp_path / "missing.pt")


def test_invalid_dqn_config():
    with pytest.raises(ValueError):
        DQNConfig(gamma=1.5)
    with pytest.raises(ValueError):
        DQNConfig(epsilon_start=0.1, epsilon_end=0.5)
    with pytest.raises(ValueError):
        DQNConfig(batch_size=0)


def test_cpu_device_and_auto_resolution():
    agent = DQNAgent(OBS, ACTIONS, small_cfg(), device="auto")
    assert agent.device.type in ("cpu", "cuda")
    assert next(agent.q_net.parameters()).device.type == agent.device.type
    if not torch.cuda.is_available():
        with pytest.raises(RuntimeError):
            DQNAgent(OBS, ACTIONS, small_cfg(), device="cuda")


def test_seeding_helpers_are_deterministic():
    import random

    from pendulum_rl.utils import derive_seed, seed_everything

    def draw():
        return random.random(), float(np.random.rand()), float(torch.rand(1))

    seed_everything(123)
    a = draw()
    seed_everything(123)
    assert a == draw()
    seed_everything(124)
    assert a != draw()
    assert derive_seed(1, 2, 3) == derive_seed(1, 2, 3) and derive_seed(1, 2, 3) != derive_seed(1, 2, 4)
