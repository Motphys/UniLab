"""Owner-level tests for MJWarp's explicit pinned host-cache barrier."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import numpy as np

from unilab.base.backend.mjwarp.backend import MjwarpBackend


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
