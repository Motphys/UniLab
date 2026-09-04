"""Owner-level tests for MJWarp's explicit pinned host-cache barrier."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import numpy as np
from unisim.backend.mjwarp.backend import MjwarpBackend


class _FakeStorage:
    def __init__(self, array: np.ndarray) -> None:
        self.array = array

    def numpy(self) -> np.ndarray:
        return self.array


class _FakeWarp:
    def __init__(self) -> None:
        self.empty_calls: list[dict[str, Any]] = []
        self.events: list[tuple[Any, ...]] = []

    def empty(self, shape: tuple[int, ...], **kwargs: Any) -> _FakeStorage:
        self.empty_calls.append({"shape": shape, **kwargs})
        return _FakeStorage(np.empty(shape, dtype=np.float32))

    def copy(self, destination: Any, source: Any) -> None:
        self.events.append(("copy", destination, source))


def _backend(warp: _FakeWarp) -> MjwarpBackend:
    backend = object.__new__(MjwarpBackend)
    backend._warp = warp
    return backend


def test_allocate_host_cache_uses_pinned_cpu_storage() -> None:
    warp = _FakeWarp()
    backend = _backend(warp)
    device_array = SimpleNamespace(shape=(4, 7), dtype="float32")

    storage, cache = backend._allocate_pinned_host_cache(device_array)

    assert warp.empty_calls == [
        {"shape": (4, 7), "dtype": "float32", "device": "cpu", "pinned": True}
    ]
    assert cache is storage.array


def test_refresh_host_cache_batches_all_downloads_before_sync() -> None:
    warp = _FakeWarp()
    backend = _backend(warp)
    backend._device_data = SimpleNamespace(
        qpos="device-qpos", qvel="device-qvel", sensordata="device-sensor"
    )
    backend._qpos_cache_storage = "host-qpos"
    backend._qvel_cache_storage = "host-qvel"
    backend._sensor_cache_storage = "host-sensor"
    backend._synchronize = lambda: warp.events.append(("sync",))  # type: ignore[method-assign]

    backend._refresh_host_cache()

    assert warp.events == [
        ("copy", "host-qpos", "device-qpos"),
        ("copy", "host-qvel", "device-qvel"),
        ("copy", "host-sensor", "device-sensor"),
        ("sync",),
    ]


def test_refresh_reset_scratch_cache_only_scatters_selected_rows() -> None:
    warp = _FakeWarp()
    backend = _backend(warp)
    scratch_values = np.arange(12, dtype=np.float32).reshape(4, 3)
    scratch_storage = _FakeStorage(np.empty_like(scratch_values))
    backend._reset_scratch_data = SimpleNamespace(sensordata=scratch_values)
    backend._reset_scratch_sensor_storage = scratch_storage
    backend._reset_scratch_sensor_cache = scratch_storage.array
    backend._sensor_cache = np.full((6, 3), -1.0, dtype=np.float32)
    backend._download = lambda source, destination: np.copyto(  # type: ignore[method-assign]
        destination.array, source
    )
    backend._synchronize = lambda: warp.events.append(("sync",))  # type: ignore[method-assign]

    backend._refresh_reset_scratch_cache(np.asarray([4, 1], dtype=np.int32))

    np.testing.assert_array_equal(backend._sensor_cache[4], scratch_values[0])
    np.testing.assert_array_equal(backend._sensor_cache[1], scratch_values[1])
    np.testing.assert_array_equal(
        backend._sensor_cache[[0, 2, 3, 5]],
        np.full((4, 3), -1.0, dtype=np.float32),
    )
    assert warp.events == [("sync",)]


def test_host_reset_routes_small_row_sets_through_scratch() -> None:
    warp = _FakeWarp()
    backend = _backend(warp)
    backend._reset_mask_host = np.zeros(8, dtype=np.bool_)
    backend._reset_mask_device = "main-mask"
    backend._device_data = SimpleNamespace(qpos="main-qpos", qvel="main-qvel")
    backend._time_cache = np.ones(8, dtype=np.float32)
    events: list[tuple[Any, ...]] = []
    backend._can_use_reset_scratch = lambda _count: True  # type: ignore[method-assign]
    backend._upload = lambda target, source: events.append(  # type: ignore[method-assign]
        ("upload", target, np.asarray(source).copy())
    )
    backend._execute_device_reset = lambda: events.append(  # type: ignore[method-assign]
        ("main-reset",)
    )
    backend._execute_reset_scratch_forward = (  # type: ignore[method-assign]
        lambda qpos, qvel: events.append(("scratch-forward", qpos.copy(), qvel.copy()))
    )
    backend._execute_device_forward = lambda: events.append(  # type: ignore[method-assign]
        ("main-forward",)
    )
    backend._synchronize = lambda: events.append(("sync",))  # type: ignore[method-assign]
    backend._refresh_reset_scratch_cache = (  # type: ignore[method-assign]
        lambda rows: events.append(("scratch-refresh", rows.copy()))
    )
    backend._refresh_host_cache = lambda: events.append(  # type: ignore[method-assign]
        ("main-refresh",)
    )
    rows = np.asarray([6, 2], dtype=np.int32)
    full_qpos = np.zeros((8, 3), dtype=np.float32)
    full_qvel = np.zeros((8, 2), dtype=np.float32)
    reset_qpos = np.ones((2, 3), dtype=np.float32)
    reset_qvel = np.ones((2, 2), dtype=np.float32)

    backend._execute_host_reset(rows, full_qpos, full_qvel, reset_qpos, reset_qvel)

    assert np.flatnonzero(backend._reset_mask_host).tolist() == [2, 6]
    assert backend._time_cache[rows].tolist() == [0.0, 0.0]
    assert [event[0] for event in events] == [
        "upload",
        "main-reset",
        "upload",
        "upload",
        "scratch-forward",
        "sync",
        "scratch-refresh",
    ]
    assert not any(event[0] in {"main-forward", "main-refresh"} for event in events)


def test_row_body_getters_gather_selected_rows_without_full_batch_reads() -> None:
    """MJWarp's partial-reset getters must not materialize the full env batch."""
    backend = object.__new__(MjwarpBackend)
    backend._body_id_to_tracked_idx = np.asarray([2, 0, 1], dtype=np.intp)
    backend._tracked_pos_w_all = np.arange(4 * 3 * 3, dtype=np.float32).reshape(4, 3, 3)
    backend._tracked_quat_w_all = np.arange(4 * 3 * 4, dtype=np.float32).reshape(4, 3, 4)
    backend._tracked_linvel_w_all = np.arange(4 * 3 * 3, dtype=np.float32).reshape(4, 3, 3) + 1000
    backend._tracked_angvel_w_all = np.arange(4 * 3 * 3, dtype=np.float32).reshape(4, 3, 3) + 2000

    def fail_full_getter(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("row getter must not call a full-batch getter")

    backend.get_body_pos_w = fail_full_getter  # type: ignore[method-assign]
    backend.get_body_quat_w = fail_full_getter  # type: ignore[method-assign]
    backend.get_body_lin_vel_w = fail_full_getter  # type: ignore[method-assign]
    backend.get_body_ang_vel_w = fail_full_getter  # type: ignore[method-assign]

    rows = np.asarray([3, 1, 3], dtype=np.int32)
    body_ids = np.asarray([1, 2], dtype=np.int32)
    mapped = np.asarray([0, 1], dtype=np.intp)
    expected_index = (rows[:, None], mapped)

    pose_pos, pose_quat = backend.get_body_pose_w_rows(rows, body_ids)
    np.testing.assert_array_equal(pose_pos, backend._tracked_pos_w_all[expected_index])
    np.testing.assert_array_equal(pose_quat, backend._tracked_quat_w_all[expected_index])
    np.testing.assert_array_equal(
        backend.get_body_lin_vel_w_rows(rows, body_ids),
        backend._tracked_linvel_w_all[expected_index],
    )
    np.testing.assert_array_equal(
        backend.get_body_ang_vel_w_rows(rows, body_ids),
        backend._tracked_angvel_w_all[expected_index],
    )
