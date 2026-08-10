"""Tests for the bounded off-policy replay ingress."""

from __future__ import annotations

import multiprocessing as mp
import threading
import time

import pytest
import torch

from unilab.ipc.replay_buffer import ReplayBuffer

_SPAWN_CTX = mp.get_context("spawn")
_OBS_DIM = 8
_ACTION_DIM = 3


def _make_buf(*, capacity: int = 128, slot_rows: int = 8, depth: int = 2) -> ReplayBuffer:
    return ReplayBuffer(
        capacity=capacity,
        obs_dim=_OBS_DIM,
        action_dim=_ACTION_DIM,
        device="cpu",
        ingress_slot_rows=slot_rows,
        ingress_depth=depth,
    )


def _random_batch(n: int):
    return (
        torch.randn(n, _OBS_DIM),
        torch.randn(n, _ACTION_DIM),
        torch.randn(n),
        torch.randn(n, _OBS_DIM),
        torch.zeros(n),
        torch.zeros(n),
    )


def _take_and_commit(buf: ReplayBuffer) -> tuple[int, int, torch.Tensor]:
    ingress = buf.take_published_ingress()
    assert ingress is not None
    slot, start, count, packed = ingress
    buf.commit_ingress(slot=slot, start=start, count=count)
    return start, count, packed


def test_host_allocation_is_capacity_independent_and_commit_is_device_owned():
    small = _make_buf(capacity=32, slot_rows=4)
    large = _make_buf(capacity=3200, slot_rows=4)
    assert small.host_storage_bytes == large.host_storage_bytes
    assert not hasattr(small, "_storage")

    obs, act, rew, nobs, done, trunc = _random_batch(4)
    small.add(obs, act, rew, nobs, done, trunc)
    assert small.published_ptr == 4
    assert int(small.ptr[0]) == 0
    assert int(small.size[0]) == 0

    start, count, packed = _take_and_commit(small)
    assert (start, count) == (0, 4)
    torch.testing.assert_close(packed[:, small._obs_sl], obs)
    torch.testing.assert_close(packed[:, small._act_sl], act)
    assert int(small.ptr[0]) == 4
    assert int(small.size[0]) == 4
    small.close()
    large.close()


def test_ingress_patches_terminal_rows_before_publication():
    buf = ReplayBuffer(
        capacity=16,
        obs_dim=_OBS_DIM,
        action_dim=_ACTION_DIM,
        device="cpu",
        ingress_slot_rows=4,
        critic_dim=3,
    )
    terminal_mask = torch.tensor([False, True, False, True])
    terminal_obs = torch.full((4, _OBS_DIM), 17.0)
    terminal_critic = torch.full((4, 3), 23.0)
    obs, act, rew, nobs, done, trunc = _random_batch(4)

    buf.add(
        obs,
        act,
        rew,
        nobs,
        done,
        trunc,
        terminal_mask=terminal_mask,
        terminal_next_obs=terminal_obs,
        critic=torch.zeros(4, 3),
        next_critic=torch.ones(4, 3),
        terminal_next_critic=terminal_critic,
    )

    _, _, packed = _take_and_commit(buf)
    torch.testing.assert_close(packed[terminal_mask, buf._nobs_sl], terminal_obs[terminal_mask])
    torch.testing.assert_close(
        packed[terminal_mask, buf._ncritic_sl],
        terminal_critic[terminal_mask],
    )
    buf.close()


def test_ingress_stores_done_and_truncated_contract():
    buf = _make_buf(slot_rows=3)
    truncated = torch.tensor([0.0, 1.0, 0.0])
    dones = torch.tensor([1.0, 1.0, 0.0])
    buf.add(
        torch.zeros(3, _OBS_DIM),
        torch.zeros(3, _ACTION_DIM),
        torch.zeros(3),
        torch.zeros(3, _OBS_DIM),
        dones,
        truncated,
    )

    _, _, packed = _take_and_commit(buf)
    torch.testing.assert_close(packed[:, buf._done_col], dones)
    torch.testing.assert_close(packed[:, buf._trunc_col], truncated)
    buf.close()


def test_ingress_writes_packed_columns_without_cat(monkeypatch: pytest.MonkeyPatch):
    buf = ReplayBuffer(
        capacity=8,
        obs_dim=_OBS_DIM,
        action_dim=_ACTION_DIM,
        device="cpu",
        ingress_slot_rows=4,
        critic_dim=5,
    )
    obs = torch.randn(4, _OBS_DIM)
    actions = torch.randn(4, _ACTION_DIM)
    critic = torch.randn(4, 5)

    def _fail_cat(*args, **kwargs):
        del args, kwargs
        raise AssertionError("ReplayBuffer.add must write packed columns directly")

    monkeypatch.setattr(torch, "cat", _fail_cat)
    buf.add(
        obs,
        actions,
        torch.randn(4),
        torch.randn(4, _OBS_DIM),
        torch.zeros(4),
        torch.zeros(4),
        critic=critic,
        next_critic=torch.randn(4, 5),
    )

    _, _, packed = _take_and_commit(buf)
    torch.testing.assert_close(packed[:, buf._obs_sl], obs)
    torch.testing.assert_close(packed[:, buf._act_sl], actions)
    torch.testing.assert_close(packed[:, buf._critic_sl], critic)
    buf.close()


def test_ingress_rejects_collection_chunk_larger_than_slot():
    buf = _make_buf(slot_rows=4)
    with pytest.raises(ValueError, match="slots hold 4"):
        buf.add(*_random_batch(5))
    buf.close()


def test_ingress_backpressures_until_committed_slot_is_released():
    buf = _make_buf(capacity=16, slot_rows=4, depth=1)
    buf.add(*_random_batch(4))
    add_started = threading.Event()
    add_finished = threading.Event()

    def add_second() -> None:
        add_started.set()
        buf.add(*_random_batch(4))
        add_finished.set()

    thread = threading.Thread(target=add_second)
    thread.start()
    assert add_started.wait(timeout=1.0)
    assert not add_finished.wait(timeout=0.05)
    _take_and_commit(buf)
    assert add_finished.wait(timeout=1.0)
    thread.join(timeout=1.0)
    _take_and_commit(buf)
    buf.close()


def _collector_add(buf: ReplayBuffer, chunks: int) -> None:
    for _ in range(chunks):
        buf.add(*_random_batch(8))


def test_spawned_collector_publishes_bounded_chunks():
    buf = _make_buf(capacity=128, slot_rows=8)
    process = _SPAWN_CTX.Process(target=_collector_add, args=(buf, 4))
    process.start()

    committed = 0
    deadline = time.monotonic() + 15.0
    while committed < 32 and time.monotonic() < deadline:
        ingress = buf.take_published_ingress()
        if ingress is None:
            time.sleep(0.001)
            continue
        slot, start, count, _ = ingress
        buf.commit_ingress(slot=slot, start=start, count=count)
        committed += count

    process.join(timeout=15)
    assert process.exitcode == 0
    assert committed == 32
    assert int(buf.ptr[0]) == 32
    assert int(buf.size[0]) == 32
    buf.close()
