"""Tests for SharedWeightSync IPC primitive."""

from __future__ import annotations

import multiprocessing as mp
from copy import deepcopy

import numpy as np
import pytest
import torch

from unilab.ipc.weight_sync import SharedWeightSync

_SPAWN_CTX = mp.get_context("spawn")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_state_dict(shapes: dict) -> dict:
    return {name: torch.randn(shape) for name, shape in shapes.items()}


# ---------------------------------------------------------------------------
# Single-process tests
# ---------------------------------------------------------------------------


def test_write_read_roundtrip(tiny_weight_shapes):
    """Write weights then read them back; values must match within float32 eps."""
    state_dict = _make_state_dict(tiny_weight_shapes)
    ws = SharedWeightSync(tiny_weight_shapes, create=True)
    ws.write_weights(state_dict)

    # Build a zeroed copy to read into
    read_sd = {name: torch.zeros(shape) for name, shape in tiny_weight_shapes.items()}
    ws.read_weights_into(read_sd)

    for name in tiny_weight_shapes:
        assert torch.allclose(state_dict[name].float(), read_sd[name].float(), atol=1e-6), (
            f"Mismatch for {name}"
        )

    ws.cleanup()


def test_version_monotonically_increases(tiny_weight_shapes):
    """Each write_weights call must increment version by 1."""
    state_dict = _make_state_dict(tiny_weight_shapes)

    ws = SharedWeightSync(tiny_weight_shapes, create=True)
    assert ws.version == 0  # raw ctor starts at 0
    ws.write_weights(state_dict)
    assert ws.version == 1
    ws.write_weights(state_dict)
    assert ws.version == 2

    ws.cleanup()


def test_from_state_dict_classmethod(tiny_weight_shapes):
    """from_state_dict() should return a valid object with version >= 1."""
    state_dict = _make_state_dict(tiny_weight_shapes)
    ws = SharedWeightSync.from_state_dict(state_dict, create=True)
    assert ws.version >= 1
    assert ws.name  # non-empty shm name

    read_sd = {name: torch.zeros(shape) for name, shape in tiny_weight_shapes.items()}
    version = ws.read_weights_into(read_sd)
    assert version >= 1

    for name in tiny_weight_shapes:
        assert torch.allclose(state_dict[name].float(), read_sd[name].float(), atol=1e-6)

    ws.cleanup()


def test_cleanup_is_idempotent(tiny_weight_shapes):
    """cleanup() called twice should not raise."""
    state_dict = _make_state_dict(tiny_weight_shapes)
    ws = SharedWeightSync.from_state_dict(state_dict, create=True)
    ws.cleanup()
    ws.cleanup()  # must not raise


def test_close_without_unlink(tiny_weight_shapes):
    """close() closes the handle without unlinking — safe to call from attached processes."""
    state_dict = _make_state_dict(tiny_weight_shapes)
    ws = SharedWeightSync.from_state_dict(state_dict, create=True)
    ws.close()  # must not raise
    ws.cleanup()  # owner still unlinks


def test_attach_create_false_roundtrip(tiny_weight_shapes):
    """create=False attaches to existing shm; read back values from owner."""
    owner = SharedWeightSync.from_state_dict(_make_state_dict(tiny_weight_shapes), create=True)
    # Attach without a lock (lock=None → no-lock path)
    attached = SharedWeightSync(tiny_weight_shapes, create=False, shm_name=owner.name, lock=None)
    assert attached.version == owner.version

    read_sd = {name: torch.zeros(shape) for name, shape in tiny_weight_shapes.items()}
    attached.read_weights_into(read_sd)

    owner_sd = {name: torch.zeros(shape) for name, shape in tiny_weight_shapes.items()}
    owner.read_weights_into(owner_sd)

    for name in tiny_weight_shapes:
        assert torch.allclose(read_sd[name], owner_sd[name], atol=1e-6)

    attached.close()
    owner.cleanup()


def test_write_read_without_lock(tiny_weight_shapes):
    """write_weights and read_weights_into with lock=None use the lockless path."""
    state_dict = _make_state_dict(tiny_weight_shapes)
    owner = SharedWeightSync.from_state_dict(state_dict, create=True)

    # Attach with no lock — exercises else-branch in write_weights / read_weights_into
    ws_nolock = SharedWeightSync(tiny_weight_shapes, create=False, shm_name=owner.name, lock=None)
    new_sd = _make_state_dict(tiny_weight_shapes)
    ws_nolock.write_weights(new_sd)

    read_sd = {name: torch.zeros(shape) for name, shape in tiny_weight_shapes.items()}
    version = ws_nolock.read_weights_into(read_sd)

    assert version >= 1
    for name in tiny_weight_shapes:
        assert torch.allclose(new_sd[name].float(), read_sd[name].float(), atol=1e-6)

    ws_nolock.close()
    owner.cleanup()


# ---------------------------------------------------------------------------
# Multiprocess test
# ---------------------------------------------------------------------------


def _writer_fn(shm_name: str, lock, shapes: dict, out_queue):
    """Subprocess: write random weights and report the version."""
    ws = SharedWeightSync(shapes, create=False, shm_name=shm_name, lock=lock)
    sd = {name: torch.randn(shape) for name, shape in shapes.items()}
    ws.write_weights(sd)
    out_queue.put({"version": ws.version, "sd": {k: v.numpy() for k, v in sd.items()}})
    ws.close()


def test_multiprocess_write_then_read(tiny_weight_shapes):
    """Spawn a writer process; main process reads and verifies."""
    # Create on main side
    ws = SharedWeightSync(tiny_weight_shapes, create=True)
    initial_version = ws.version  # 0

    out_queue = _SPAWN_CTX.Queue()
    p = _SPAWN_CTX.Process(
        target=_writer_fn,
        args=(ws.name, ws._lock, tiny_weight_shapes, out_queue),
    )
    p.start()
    p.join(timeout=15)
    assert p.exitcode == 0, f"Writer process failed with exit code {p.exitcode}"

    result = out_queue.get_nowait()
    written_version = result["version"]
    written_sd = {k: torch.from_numpy(v) for k, v in result["sd"].items()}

    read_sd = {name: torch.zeros(shape) for name, shape in tiny_weight_shapes.items()}
    read_version = ws.read_weights_into(read_sd)

    assert read_version > initial_version
    assert read_version == written_version

    for name in tiny_weight_shapes:
        assert torch.allclose(written_sd[name].float(), read_sd[name].float(), atol=1e-6)

    ws.cleanup()
