from __future__ import annotations

import shutil

import pytest

from unilab.ipc.memory_budget import (
    estimate_offpolicy_bytes,
    raise_if_shared_memory_over_budget,
)


def test_offpolicy_memory_budget_notes_native_exclusions() -> None:
    estimate = estimate_offpolicy_bytes(
        num_envs=5120,
        replay_buffer_n=1024,
        obs_dim=98,
        action_dim=29,
        critic_dim=101,
        batch_size=8192,
        updates_per_step=8,
    )

    breakdown = str(estimate["breakdown"])
    assert "MuJoCo BatchEnvPool" in breakdown
    assert "CUDA pinned/shared" in breakdown
    assert "driver memory" in breakdown


def test_shared_memory_budget_unknown_available_is_noop(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(shutil, "disk_usage", lambda path: (_ for _ in ()).throw(OSError()))
    estimate = {"total": 1024, "breakdown": "test"}

    raise_if_shared_memory_over_budget(estimate, label="test", path="/missing-shm")


def test_shared_memory_budget_allows_within_threshold(monkeypatch: pytest.MonkeyPatch) -> None:
    class _Usage:
        free = 100 * 1024

    monkeypatch.setattr(shutil, "disk_usage", lambda path: _Usage())
    estimate = {"total": 80 * 1024, "breakdown": "test"}

    raise_if_shared_memory_over_budget(estimate, label="test", threshold=0.8)


def test_shared_memory_budget_raises_before_over_allocating(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _Usage:
        free = 100 * 1024

    monkeypatch.setattr(shutil, "disk_usage", lambda path: _Usage())
    estimate = {"total": 81 * 1024, "breakdown": "test"}

    with pytest.raises(MemoryError) as excinfo:
        raise_if_shared_memory_over_budget(estimate, label="Off-policy (td3)", threshold=0.8)

    message = str(excinfo.value)
    assert "Off-policy (td3)" in message
    assert "/dev/shm" in message
    assert "estimated" in message
    assert "available" in message
    assert "algo.num_envs" in message
    assert "algo.replay_buffer_n" in message
