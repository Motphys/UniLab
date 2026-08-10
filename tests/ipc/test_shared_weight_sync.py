"""Tests for SharedWeightSync IPC primitive."""

from __future__ import annotations

import multiprocessing as mp
from copy import deepcopy

import numpy as np
import pytest
import torch

from unilab.ipc.weight_sync import SharedWeightSync

_SPAWN_CTX = mp.get_context("spawn")


class _Actor31(torch.nn.Module):
    """Small actor-shaped module: 20 parameters + 11 persistent buffers = 31 state entries."""

    def __init__(self) -> None:
        super().__init__()
        self.fc1 = torch.nn.Linear(8, 16)
        self.ln1 = torch.nn.LayerNorm(16)
        self.fc2 = torch.nn.Linear(16, 16)
        self.ln2 = torch.nn.LayerNorm(16)
        self.fc3 = torch.nn.Linear(16, 16)
        self.ln3 = torch.nn.LayerNorm(16)
        self.fc4 = torch.nn.Linear(16, 16)
        self.ln4 = torch.nn.LayerNorm(16)
        self.fc5 = torch.nn.Linear(16, 3)
        self.ln5 = torch.nn.LayerNorm(3)
        for i in range(11):
            self.register_buffer(f"buf{i}", torch.zeros(4), persistent=True)


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


# ---------------------------------------------------------------------------
# DeviceWeightApplier tests
# ---------------------------------------------------------------------------


def _make_31entry_state_dict(seed: int) -> dict:
    module = _Actor31()
    torch.manual_seed(seed)
    return {name: torch.randn_like(t) for name, t in module.state_dict().items()}


def _assert_state_dict_matches(target_sd: dict, expected_sd: dict) -> None:
    for name, expected in expected_sd.items():
        actual = target_sd[name].detach().cpu()
        assert torch.allclose(actual, expected.float().cpu(), atol=1e-6), f"Mismatch for {name}"


def test_device_applier_cpu_roundtrip(tiny_weight_shapes):
    """Applier on CPU device copies the snapshot into the bound tensors."""
    state_dict = _make_state_dict(tiny_weight_shapes)
    ws = SharedWeightSync.from_state_dict(state_dict, create=True)

    target = {name: torch.zeros(shape) for name, shape in tiny_weight_shapes.items()}
    applier = ws.create_device_applier(target, torch.device("cpu"))
    version = applier.apply()

    assert version == ws.version
    for name in tiny_weight_shapes:
        assert torch.allclose(state_dict[name].float(), target[name].float(), atol=1e-6)

    ws.cleanup()


def test_device_applier_consecutive_versions(tiny_weight_shapes):
    """Back-to-back version publishes each apply completely and return the new version."""
    ws = SharedWeightSync(tiny_weight_shapes, create=True)
    target = {name: torch.zeros(shape) for name, shape in tiny_weight_shapes.items()}
    applier = ws.create_device_applier(target, torch.device("cpu"))

    versions = []
    for seed in range(3):
        sd = {
            name: torch.full(shape, float(seed + 1)) for name, shape in tiny_weight_shapes.items()
        }
        ws.write_weights(sd)
        versions.append(applier.apply())
        for name in tiny_weight_shapes:
            assert torch.allclose(target[name], sd[name], atol=1e-6)

    assert versions == [1, 2, 3]

    ws.cleanup()


def test_device_applier_shape_mismatch_raises(tiny_weight_shapes):
    """Applier construction fails closed on shape/layout mismatch."""
    ws = SharedWeightSync(tiny_weight_shapes, create=True)
    bad = {name: torch.zeros(tuple(shape) + (1,)) for name, shape in tiny_weight_shapes.items()}
    with pytest.raises(ValueError, match="shape mismatch"):
        ws.create_device_applier(bad, torch.device("cpu"))
    ws.cleanup()


def test_device_applier_no_version_bump_without_write(tiny_weight_shapes):
    """Without a new publish, apply() keeps returning the same version (no-op upstream)."""
    state_dict = _make_state_dict(tiny_weight_shapes)
    ws = SharedWeightSync.from_state_dict(state_dict, create=True)
    target = {name: torch.zeros(shape) for name, shape in tiny_weight_shapes.items()}
    applier = ws.create_device_applier(target, torch.device("cpu"))

    v1 = applier.apply()
    v2 = applier.apply()
    assert v1 == v2 == ws.version

    ws.cleanup()


def test_device_applier_31entry_param_buffer_completeness():
    """31-entry (20 params + 11 persistent buffers) snapshot applies every entry."""
    source_sd = _make_31entry_state_dict(seed=0)
    assert len(source_sd) == 31

    ws = SharedWeightSync.from_state_dict(source_sd, create=True)

    target_module = _Actor31()
    with torch.no_grad():
        for t in target_module.state_dict().values():
            t.zero_()
    target_sd = dict(target_module.state_dict())

    applier = ws.create_device_applier(target_module.state_dict(), torch.device("cpu"))
    version = applier.apply()

    assert version == ws.version
    # The bound state_dict storage must have been updated in place.
    _assert_state_dict_matches(target_sd, source_sd)
    # And the module itself observes the new values (same storage).
    _assert_state_dict_matches(dict(target_module.state_dict()), source_sd)

    ws.cleanup()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_device_applier_cuda_roundtrip():
    """CUDA applier: 31-entry snapshot lands on the on-device actor tensors."""
    source_sd = _make_31entry_state_dict(seed=1)
    ws = SharedWeightSync.from_state_dict(source_sd, create=True)

    target_module = _Actor31().cuda()
    target_sd = dict(target_module.state_dict())

    applier = ws.create_device_applier(target_module.state_dict(), torch.device("cuda"))
    version = applier.apply()
    torch.cuda.synchronize()

    assert version == ws.version
    _assert_state_dict_matches(target_sd, source_sd)

    # A second publish must also apply cleanly over the reused staging buffers.
    source_sd2 = _make_31entry_state_dict(seed=2)
    ws.write_weights(source_sd2)
    version2 = applier.apply()
    torch.cuda.synchronize()

    assert version2 == version + 1
    _assert_state_dict_matches(dict(target_module.state_dict()), source_sd2)

    ws.cleanup()


def _constant_writer_fn(shm_name: str, lock, shapes: dict, iterations: int) -> None:
    """Subprocess: alternate publishing all-ones and all-twos snapshots."""
    ws = SharedWeightSync(shapes, create=False, shm_name=shm_name, lock=lock)
    sd_one = {name: torch.ones(shape) for name, shape in shapes.items()}
    sd_two = {name: torch.full(shape, 2.0) for name, shape in shapes.items()}
    for i in range(iterations):
        ws.write_weights(sd_one if i % 2 == 0 else sd_two)
    ws.close()


def test_device_applier_concurrent_publish_no_torn_snapshot(tiny_weight_shapes):
    """Concurrent publish/apply must never surface a partially published state."""
    ws = SharedWeightSync(tiny_weight_shapes, create=True)
    target = {name: torch.zeros(shape) for name, shape in tiny_weight_shapes.items()}
    applier = ws.create_device_applier(target, torch.device("cpu"))

    p = _SPAWN_CTX.Process(
        target=_constant_writer_fn,
        args=(ws.name, ws._lock, tiny_weight_shapes, 50),
    )
    p.start()
    while p.is_alive():
        version = applier.apply()
        if version == 0:
            continue  # initial all-zero snapshot, writer has not published yet
        for name in tiny_weight_shapes:
            values = target[name]
            assert torch.allclose(values, torch.ones_like(values), atol=1e-6) or torch.allclose(
                values, torch.full_like(values, 2.0), atol=1e-6
            ), f"Torn snapshot for {name}"
    p.join(timeout=30)
    assert p.exitcode == 0

    ws.cleanup()
