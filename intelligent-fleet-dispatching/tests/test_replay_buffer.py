"""Replay buffer: ring-buffer capacity, correct pairing of sampled fields, validation, and checkpoint round-trip."""

from __future__ import annotations

import numpy as np
import pytest

from fleet_dispatch.agents.replay_buffer import ReplayBuffer

OBS_DIM, N_ACTIONS = 3, 4


def _buffer(capacity: int = 10) -> ReplayBuffer:
    return ReplayBuffer(capacity, OBS_DIM, N_ACTIONS, np.random.default_rng(0))


def _add(buffer: ReplayBuffer, i: int) -> None:
    """Store transition i with every field tagged by i, so sampled rows can be checked for consistency."""
    obs = np.full(OBS_DIM, i, dtype=np.float32)
    next_obs = np.full(OBS_DIM, i + 0.5, dtype=np.float32)
    mask = np.zeros(N_ACTIONS, dtype=bool)
    mask[i % N_ACTIONS] = True
    buffer.add(obs, action=i % N_ACTIONS, reward=float(i), next_obs=next_obs,
               terminated=bool(i % 2), truncated=bool(i % 3 == 0), mask=mask, next_mask=mask)


def test_fills_then_wraps_around_at_capacity() -> None:
    buf = _buffer(capacity=10)
    for i in range(14):
        _add(buf, i)
    assert len(buf) == 10
    stored_rewards = sorted(buf.rewards.tolist())
    assert stored_rewards == [float(i) for i in range(4, 14)], "the four oldest transitions were overwritten"


def test_sampled_fields_stay_consistent_within_each_transition() -> None:
    buf = _buffer(capacity=50)
    for i in range(50):
        _add(buf, i)
    batch = buf.sample(32)
    assert batch.obs.shape == (32, OBS_DIM) and batch.obs.dtype == np.float32
    assert batch.masks.shape == (32, N_ACTIONS) and batch.masks.dtype == np.bool_
    tag = batch.rewards.astype(int)
    np.testing.assert_array_equal(batch.obs[:, 0], tag)
    np.testing.assert_allclose(batch.next_obs[:, 0], tag + 0.5)
    np.testing.assert_array_equal(batch.actions, tag % N_ACTIONS)
    np.testing.assert_array_equal(batch.terminated, tag % 2 == 1)
    np.testing.assert_array_equal(batch.truncated, tag % 3 == 0)
    for row, t in enumerate(tag):
        assert batch.masks[row, t % N_ACTIONS] and batch.masks[row].sum() == 1


def test_sampling_is_uniform_over_stored_transitions() -> None:
    buf = _buffer(capacity=4)
    for i in range(4):
        _add(buf, i)
    counts = np.zeros(4)
    for _ in range(200):
        for r in buf.sample(16).rewards.astype(int):
            counts[r] += 1
    frequencies = counts / counts.sum()
    assert np.all(np.abs(frequencies - 0.25) < 0.05)


def test_invalid_inputs_are_rejected() -> None:
    buf = _buffer()
    with pytest.raises(ValueError):
        buf.sample(4)  # empty buffer
    with pytest.raises(ValueError):
        buf.add(np.zeros(OBS_DIM + 1, np.float32), 0, 0.0, np.zeros(OBS_DIM, np.float32),
                False, False, np.zeros(N_ACTIONS, bool), np.zeros(N_ACTIONS, bool))
    with pytest.raises(ValueError):
        ReplayBuffer(0, OBS_DIM, N_ACTIONS, np.random.default_rng(0))


def test_state_dict_round_trip_preserves_contents() -> None:
    source = _buffer(capacity=8)
    for i in range(11):
        _add(source, i)
    state = source.state_dict()
    target = _buffer(capacity=8)
    target.load_state_dict(state)
    assert len(target) == len(source) == 8
    np.testing.assert_array_equal(target.rewards, source.rewards)
    np.testing.assert_array_equal(target.masks, source.masks)
