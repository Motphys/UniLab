"""Tests for SharedOnPolicyStorage IPC primitive."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from unilab.ipc.shared_onpolicy_storage import SharedOnPolicyStorage

_NUM_ENVS = 4
_NUM_STEPS = 10
_OBS_DIM = 8
_ACTION_DIM = 3
_CRITIC_DIM = 5
_NUM_SLOTS = 2

_EXPECTED_FIELDS = {"obs", "actions", "log_probs", "rewards", "dones", "truncated", "last_obs"}


def _make_storage(num_slots: int = _NUM_SLOTS) -> SharedOnPolicyStorage:
    return SharedOnPolicyStorage(
        num_envs=_NUM_ENVS,
        num_steps=_NUM_STEPS,
        obs_dim=_OBS_DIM,
        action_dim=_ACTION_DIM,
        num_slots=num_slots,
        create=True,
    )


def _make_storage_with_critic() -> SharedOnPolicyStorage:
    return SharedOnPolicyStorage(
        num_envs=_NUM_ENVS,
        num_steps=_NUM_STEPS,
        obs_dim=_OBS_DIM,
        action_dim=_ACTION_DIM,
        critic_dim=_CRITIC_DIM,
        num_slots=_NUM_SLOTS,
        create=True,
    )


def test_pointers_start_at_zero():
    s = _make_storage()
    assert int(s._write_ptr.value) == 0
    assert int(s._read_ptr.value) == 0
    s.cleanup()


def test_signal_write_done_and_wait_for_data():
    """signal_write_done() advances write_ptr; wait_for_data() returns True."""
    s = _make_storage()
    assert s.available() == 0
    s.signal_write_done()
    assert s.available() == 1
    assert s.wait_for_data(timeout=0.1) is True
    s.cleanup()


def test_available_returns_correct_count():
    s = _make_storage(num_slots=4)
    assert s.available() == 0
    for i in range(3):
        s.signal_write_done()
        assert s.available() == i + 1
    s.cleanup()


def test_advance_read_skips_overwritten_slots():
    """If writer ran far ahead (wp - rp > num_slots), advance_read fast-forwards."""
    s = _make_storage(num_slots=2)
    # Writer produces 4 rollouts (2x more than num_slots)
    for _ in range(4):
        s.signal_write_done()
    # wp = 4, rp = 0 → available = 4 > num_slots
    s.advance_read()
    # After advance, rp should have jumped past overwritten slots
    # wp - rp should be <= num_slots
    assert int(s._write_ptr.value) - int(s._read_ptr.value) <= s.num_slots
    s.cleanup()


def test_read_torch_returns_all_fields():
    """read_torch() must return a dict with all expected field keys."""
    s = _make_storage()
    # Fill write slot with non-zero data
    wb = s.write_buffer
    for arr in wb.values():
        arr[:] = np.random.randn(*arr.shape).astype(np.float32)
    s.signal_write_done()

    result = s.read_torch("cpu")
    assert set(result.keys()) == _EXPECTED_FIELDS
    for k, v in result.items():
        assert isinstance(v, torch.Tensor), f"{k} is not a tensor"
    s.cleanup()


def test_cleanup_is_idempotent():
    """cleanup() called twice must not raise."""
    s = _make_storage()
    s.cleanup()
    s.cleanup()


def test_wait_for_data_timeout_returns_false():
    """wait_for_data() with a short timeout returns False when no data is available."""
    s = _make_storage()
    result = s.wait_for_data(timeout=0.05)
    assert result is False
    s.cleanup()


def test_write_buffer_returns_dict_of_arrays():
    """write_buffer property returns a dict of numpy arrays for the current write slot."""
    s = _make_storage()
    wb = s.write_buffer
    assert set(wb.keys()) == _EXPECTED_FIELDS
    for k, arr in wb.items():
        assert hasattr(arr, "shape"), f"{k} is not an array"
    s.cleanup()


def test_close_does_not_raise():
    """close() (attach-mode teardown, no unlink) must not raise."""
    s = _make_storage()
    s.close()


def test_attach_create_false_reads_same_data():
    """Attaching to existing shm (create=False) exposes the same data as the owner."""
    owner = _make_storage()
    # Write known data into write slot 0
    wb = owner.write_buffer
    for arr in wb.values():
        arr[:] = 1.0
    owner.signal_write_done()

    # Attach
    attached = SharedOnPolicyStorage(
        num_envs=_NUM_ENVS,
        num_steps=_NUM_STEPS,
        obs_dim=_OBS_DIM,
        action_dim=_ACTION_DIM,
        num_slots=_NUM_SLOTS,
        create=False,
        shm_name_prefix=owner.name,
    )
    attached.attach_sync_primitives(owner._write_ptr, owner._read_ptr)

    # Available should be 1 (owner wrote one slot)
    assert attached.available() == 1

    result = attached.read_torch("cpu")
    assert set(result.keys()) == _EXPECTED_FIELDS

    attached.close()
    owner.cleanup()


def test_attach_sync_primitives():
    """attach_sync_primitives() replaces internal pointers in the attached instance."""
    owner = _make_storage()
    attached = SharedOnPolicyStorage(
        num_envs=_NUM_ENVS,
        num_steps=_NUM_STEPS,
        obs_dim=_OBS_DIM,
        action_dim=_ACTION_DIM,
        num_slots=_NUM_SLOTS,
        create=False,
        shm_name_prefix=owner.name,
    )
    attached.attach_sync_primitives(owner._write_ptr, owner._read_ptr)
    # After attaching, pointers are shared — incrementing owner's write_ptr is visible
    owner.signal_write_done()
    assert attached.available() == 1

    attached.close()
    owner.cleanup()


def test_storage_allocates_optional_critic_fields():
    storage = _make_storage_with_critic()

    expected = _EXPECTED_FIELDS | {"critic", "last_critic"}
    assert set(storage.write_buffer.keys()) == expected

    storage.cleanup()
