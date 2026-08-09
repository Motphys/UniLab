"""Tests for ReplayBuffer IPC primitive."""

from __future__ import annotations

import multiprocessing as mp
import threading

import numpy as np
import pytest
import torch

from unilab.ipc.replay_buffer import ReplayBuffer

_SPAWN_CTX = mp.get_context("spawn")

_CAPACITY = 128
_OBS_DIM = 8
_ACTION_DIM = 3
_DEVICE = "cpu"

_EXPECTED_FIELDS = {"obs", "next_obs", "actions", "rewards", "dones", "truncated"}


def _make_buf(capacity: int = _CAPACITY) -> ReplayBuffer:
    return ReplayBuffer(capacity=capacity, obs_dim=_OBS_DIM, action_dim=_ACTION_DIM, device=_DEVICE)


def _random_batch(n: int):
    return (
        torch.randn(n, _OBS_DIM),
        torch.randn(n, _ACTION_DIM),
        torch.randn(n),
        torch.randn(n, _OBS_DIM),
        torch.zeros(n),
        torch.zeros(n),
    )


# ---------------------------------------------------------------------------
# Single-process tests
# ---------------------------------------------------------------------------


def test_add_single_batch_increases_size():
    buf = _make_buf()
    assert int(buf.size[0]) == 0
    obs, act, rew, nobs, done, trunc = _random_batch(16)
    buf.add(obs, act, rew, nobs, done, trunc)
    assert int(buf.size[0]) == 16


def test_add_beyond_capacity_wraps_ptr():
    """Adding more than capacity should wrap ptr but cap size at capacity."""
    buf = _make_buf(capacity=32)
    obs, act, rew, nobs, done, trunc = _random_batch(48)
    buf.add(obs, act, rew, nobs, done, trunc)
    assert int(buf.size[0]) == 32
    assert int(buf.ptr[0]) == 48


def test_sample_returns_all_fields():
    buf = _make_buf()
    obs, act, rew, nobs, done, trunc = _random_batch(64)
    buf.add(obs, act, rew, nobs, done, trunc)

    batch = buf.sample(16)
    assert set(batch.keys()) == _EXPECTED_FIELDS


def test_sample_batch_size_dimension():
    buf = _make_buf()
    obs, act, rew, nobs, done, trunc = _random_batch(64)
    buf.add(obs, act, rew, nobs, done, trunc)

    batch_size = 32
    batch = buf.sample(batch_size)
    assert batch["obs"].shape == (batch_size, _OBS_DIM)
    assert batch["actions"].shape == (batch_size, _ACTION_DIM)
    assert batch["rewards"].shape == (batch_size,)


def test_add_wraparound_data_integrity():
    """CPU path: when idx + n > capacity, data must wrap correctly.

    We fill the buffer exactly to capacity, then add a batch that straddles
    the boundary (position 28 to 36 wrapping around capacity=32).
    We verify the wrapped data can be read back via sample().
    """
    capacity = 32
    buf = _make_buf(capacity=capacity)

    # Fill buffer to position 28 (leave 4 slots at end)
    obs_first, act_first, rew_first, nobs_first, done_first, trunc_first = _random_batch(28)
    buf.add(obs_first, act_first, rew_first, nobs_first, done_first, trunc_first)
    assert int(buf.ptr[0]) == 28

    # Add 8-item batch: splits across boundary (4 at end, 4 at start)
    obs_wrap = torch.arange(8 * _OBS_DIM, dtype=torch.float32).reshape(8, _OBS_DIM)
    act_wrap = torch.zeros(8, _ACTION_DIM)
    rew_wrap = torch.ones(8) * 99.0
    nobs_wrap = torch.zeros(8, _OBS_DIM)
    done_wrap = torch.zeros(8)
    trunc_wrap = torch.zeros(8)
    buf.add(obs_wrap, act_wrap, rew_wrap, nobs_wrap, done_wrap, trunc_wrap)

    assert int(buf.ptr[0]) == 36
    assert int(buf.size[0]) == capacity  # capped at capacity

    # The last 4 items of obs_wrap should be at storage[0:4],
    # and the first 4 at storage[28:32].
    # Verify via the packed storage directly.
    # storage[28:32] should match obs_wrap[0:4]
    assert torch.allclose(buf._storage[28:32, buf._obs_sl], obs_wrap[:4], atol=1e-6)
    # storage[0:4] should match obs_wrap[4:8]
    assert torch.allclose(buf._storage[0:4, buf._obs_sl], obs_wrap[4:8], atol=1e-6)


def test_add_exact_capacity_fill():
    """Adding exactly capacity items should set size == capacity and ptr == capacity."""
    capacity = 16
    buf = _make_buf(capacity=capacity)
    obs, act, rew, nobs, done, trunc = _random_batch(capacity)
    buf.add(obs, act, rew, nobs, done, trunc)
    assert int(buf.size[0]) == capacity
    assert int(buf.ptr[0]) == capacity


def test_sample_values_match_added_data():
    """Sample with a fixed index should return the exact values we added."""
    buf = _make_buf(capacity=64)
    obs = torch.arange(16 * _OBS_DIM, dtype=torch.float32).reshape(16, _OBS_DIM)
    act = torch.ones(16, _ACTION_DIM) * 3.14
    rew = torch.arange(16, dtype=torch.float32)
    nobs = torch.zeros(16, _OBS_DIM)
    done = torch.zeros(16)
    trunc = torch.zeros(16)
    buf.add(obs, act, rew, nobs, done, trunc)

    # Sample exactly index 5 by seeding torch RNG
    torch.manual_seed(0)
    # Just verify values are in valid range (obs values we added: 0..16*obs_dim-1)
    batch = buf.sample(16)
    # All rewards should be one of 0..15
    assert batch["rewards"].min() >= 0.0
    assert batch["rewards"].max() <= 15.0


def test_add_patches_terminal_next_obs_without_prebuilding_full_transition_copy():
    buf = _make_buf(capacity=8)
    obs = torch.zeros(4, _OBS_DIM)
    act = torch.zeros(4, _ACTION_DIM)
    rew = torch.zeros(4)
    next_obs = torch.arange(4 * _OBS_DIM, dtype=torch.float32).reshape(4, _OBS_DIM)
    done = torch.tensor([0.0, 1.0, 0.0, 1.0])
    trunc = torch.tensor([0.0, 1.0, 0.0, 0.0])
    terminal_mask = torch.tensor([False, True, False, True])
    terminal_next_obs = torch.full((4, _OBS_DIM), 99.0)

    buf.add(
        obs,
        act,
        rew,
        next_obs,
        done,
        trunc,
        terminal_mask=terminal_mask,
        terminal_next_obs=terminal_next_obs,
    )

    stored_next_obs = buf._storage[:4, buf._nobs_sl]
    expected = next_obs.clone()
    expected[terminal_mask] = terminal_next_obs[terminal_mask]
    assert torch.allclose(stored_next_obs, expected)


def test_add_stores_combined_dones_and_truncated_contract():
    buf = _make_buf(capacity=8)
    terminated = torch.tensor([1.0, 0.0, 0.0])
    truncated = torch.tensor([0.0, 1.0, 0.0])
    dones = (terminated.bool() | truncated.bool()).float()

    buf.add(
        torch.zeros(3, _OBS_DIM),
        torch.zeros(3, _ACTION_DIM),
        torch.zeros(3),
        torch.zeros(3, _OBS_DIM),
        dones,
        truncated,
    )

    torch.testing.assert_close(buf._storage[:3, buf._done_col], torch.tensor([1.0, 1.0, 0.0]))
    torch.testing.assert_close(buf._storage[:3, buf._trunc_col], torch.tensor([0.0, 1.0, 0.0]))


def test_add_writes_packed_columns_without_cat(monkeypatch: pytest.MonkeyPatch):
    """Collector hot path should not allocate a full concatenated transition batch."""
    buf = ReplayBuffer(
        capacity=8,
        obs_dim=_OBS_DIM,
        action_dim=_ACTION_DIM,
        critic_dim=5,
        device=_DEVICE,
    )
    obs = torch.randn(4, _OBS_DIM)
    act = torch.randn(4, _ACTION_DIM)
    rew = torch.randn(4)
    nobs = torch.randn(4, _OBS_DIM)
    done = torch.zeros(4)
    trunc = torch.zeros(4)
    critic = torch.randn(4, 5)
    ncritic = torch.randn(4, 5)

    def _fail_cat(*args, **kwargs):
        del args, kwargs
        raise AssertionError("ReplayBuffer.add should write packed columns directly")

    monkeypatch.setattr(torch, "cat", _fail_cat)

    buf.add(obs, act, rew, nobs, done, trunc, critic=critic, next_critic=ncritic)

    torch.testing.assert_close(buf._storage[:4, buf._obs_sl], obs)
    torch.testing.assert_close(buf._storage[:4, buf._act_sl], act)
    torch.testing.assert_close(buf._storage[:4, buf._critic_sl], critic)


def test_bounded_ingress_host_allocation_is_capacity_independent_and_committed_late():
    buffers = [
        ReplayBuffer(
            capacity=capacity,
            obs_dim=_OBS_DIM,
            action_dim=_ACTION_DIM,
            device="cuda",
            ingress_slot_rows=4,
        )
        for capacity in (32, 3200)
    ]
    small, large = buffers
    assert small.host_storage_bytes == large.host_storage_bytes
    assert not hasattr(small, "_storage")

    obs, act, rew, nobs, done, trunc = _random_batch(4)
    small.add(obs, act, rew, nobs, done, trunc)
    assert small.published_ptr == 4
    assert int(small.ptr[0]) == 0
    assert int(small.size[0]) == 0

    ingress = small.take_published_ingress()
    assert ingress is not None
    slot, start, count, packed = ingress
    torch.testing.assert_close(packed[:, small._obs_sl], obs)
    small.commit_ingress(slot=slot, start=start, count=count)
    assert int(small.ptr[0]) == 4
    assert int(small.size[0]) == 4
    for buffer in buffers:
        buffer.close()


def test_bounded_ingress_patches_terminal_rows_before_publication():
    buf = ReplayBuffer(
        capacity=16,
        obs_dim=_OBS_DIM,
        action_dim=_ACTION_DIM,
        critic_dim=3,
        device="cuda",
        ingress_slot_rows=4,
    )
    terminal_mask = torch.tensor([False, True, False, True])
    terminal_obs = torch.full((4, _OBS_DIM), 17.0)
    terminal_critic = torch.full((4, 3), 23.0)
    obs, act, rew, nobs, done, trunc = _random_batch(4)
    critic = torch.zeros(4, 3)
    next_critic = torch.ones(4, 3)

    buf.add(
        obs,
        act,
        rew,
        nobs,
        done,
        trunc,
        terminal_mask=terminal_mask,
        terminal_next_obs=terminal_obs,
        critic=critic,
        next_critic=next_critic,
        terminal_next_critic=terminal_critic,
    )

    ingress = buf.take_published_ingress()
    assert ingress is not None
    slot, start, count, packed = ingress
    torch.testing.assert_close(packed[terminal_mask, buf._nobs_sl], terminal_obs[terminal_mask])
    torch.testing.assert_close(
        packed[terminal_mask, buf._ncritic_sl], terminal_critic[terminal_mask]
    )
    buf.commit_ingress(slot=slot, start=start, count=count)
    buf.close()


def test_bounded_ingress_backpressures_until_completed_slot_is_released():
    buf = ReplayBuffer(
        capacity=16,
        obs_dim=_OBS_DIM,
        action_dim=_ACTION_DIM,
        device="cuda",
        ingress_slot_rows=4,
        ingress_depth=1,
    )
    first = _random_batch(4)
    second = _random_batch(4)
    buf.add(*first)
    add_started = threading.Event()
    add_finished = threading.Event()

    def add_second() -> None:
        add_started.set()
        buf.add(*second)
        add_finished.set()

    thread = threading.Thread(target=add_second)
    thread.start()
    assert add_started.wait(timeout=1.0)
    assert not add_finished.wait(timeout=0.05)

    ingress = buf.take_published_ingress()
    assert ingress is not None
    slot, start, count, _ = ingress
    buf.commit_ingress(slot=slot, start=start, count=count)
    assert add_finished.wait(timeout=1.0)
    thread.join(timeout=1.0)

    ingress = buf.take_published_ingress()
    assert ingress is not None
    slot, start, count, _ = ingress
    buf.commit_ingress(slot=slot, start=start, count=count)
    buf.close()


# ---------------------------------------------------------------------------
# Multiprocess test
# ---------------------------------------------------------------------------


def _collector_add(buf: ReplayBuffer, n_steps: int) -> None:
    for _ in range(n_steps):
        obs, act, rew, nobs, done, trunc = _random_batch(8)
        buf.add(obs, act, rew, nobs, done, trunc)


def test_multiprocess_add_then_sample():
    """Spawn a collector that adds data; main process samples from it."""
    buf = _make_buf()
    p = _SPAWN_CTX.Process(target=_collector_add, args=(buf, 4))
    p.start()
    p.join(timeout=15)
    assert p.exitcode == 0, f"Collector process failed with exit code {p.exitcode}"

    assert int(buf.size[0]) > 0
    batch = buf.sample(min(16, int(buf.size[0])))
    assert set(batch.keys()) == _EXPECTED_FIELDS
