"""Cold-path CUDA graph eligibility, capture, replay, and fallback tests."""

from __future__ import annotations

from typing import Any

import pytest
from unisim.backend.mjwarp.backend import (
    MjwarpBackend,
    _cuda_graph_eligibility,
    _reset_scratch_capacity_for_batch,
)


class _FakeDevice:
    def __init__(self, *, is_cuda: bool = True) -> None:
        self.is_cuda = is_cuda


class _FakeContext:
    def __enter__(self) -> "_FakeContext":
        return self

    def __exit__(self, *_args: Any) -> None:
        return None


class _FakeCapture(_FakeContext):
    def __init__(self, graph: str, *, failure: Exception | None = None) -> None:
        self.graph = graph
        self._failure = failure

    def __enter__(self) -> "_FakeCapture":
        if self._failure is not None:
            raise self._failure
        return self


class _FakeWarp:
    def __init__(
        self,
        *,
        driver: tuple[int, int] | None = (13, 0),
        mempool: bool = True,
        fail_capture_index: int | None = None,
    ) -> None:
        self.driver = driver
        self.mempool = mempool
        self.fail_capture_index = fail_capture_index
        self.capture_count = 0
        self.launches: list[str] = []

    def get_cuda_driver_version(self) -> tuple[int, int] | None:
        return self.driver

    def is_mempool_enabled(self, _device: Any) -> bool:
        return self.mempool

    def ScopedDevice(self, _device: Any) -> _FakeContext:  # noqa: N802
        return _FakeContext()

    def ScopedCapture(self) -> _FakeCapture:  # noqa: N802
        capture_index = self.capture_count
        self.capture_count += 1
        failure = (
            RuntimeError("synthetic capture failure")
            if capture_index == self.fail_capture_index
            else None
        )
        return _FakeCapture(f"graph-{capture_index}", failure=failure)

    def capture_launch(self, graph: str) -> None:
        self.launches.append(graph)


class _FakeMujocoWarp:
    def __init__(self) -> None:
        self.step_calls = 0
        self.forward_calls = 0
        self.reset_calls = 0

    def step(self, _model: Any, _data: Any) -> None:
        self.step_calls += 1

    def forward(self, _model: Any, _data: Any) -> None:
        self.forward_calls += 1

    def reset_data(self, _model: Any, _data: Any, *, reset: Any) -> None:
        del reset
        self.reset_calls += 1


def _backend(warp: _FakeWarp, mujoco_warp: _FakeMujocoWarp) -> MjwarpBackend:
    backend = MjwarpBackend.__new__(MjwarpBackend)
    backend._warp = warp
    backend._mujoco_warp = mujoco_warp
    backend._device_model = object()
    backend._device_data = object()
    backend._reset_mask_device = object()
    backend._reset_scratch_capacity = 0
    backend._reset_scratch_data = None
    return backend


@pytest.mark.parametrize(
    ("device", "driver", "mempool", "expected_reason"),
    [
        (_FakeDevice(is_cuda=False), (13, 0), True, "not CUDA"),
        (_FakeDevice(), None, True, "unavailable"),
        (_FakeDevice(), (12, 3), True, "older than 12.4"),
        (_FakeDevice(), (12, 4), False, "mempool is disabled"),
    ],
)
def test_cuda_graph_eligibility_fails_closed(
    device: _FakeDevice,
    driver: tuple[int, int] | None,
    mempool: bool,
    expected_reason: str,
) -> None:
    eligible, reason = _cuda_graph_eligibility(
        _FakeWarp(driver=driver, mempool=mempool),
        device,
    )

    assert eligible is False
    assert reason is not None and expected_reason in reason


def test_cuda_graph_capture_replays_fixed_address_operations() -> None:
    warp = _FakeWarp(driver=(12, 4), mempool=True)
    mujoco_warp = _FakeMujocoWarp()
    backend = _backend(warp, mujoco_warp)

    backend._initialize_cuda_graphs(_FakeDevice())

    assert backend._cuda_graph_enabled is True
    assert backend._cuda_graph_disable_reason is None
    assert (backend._step_graph, backend._forward_graph, backend._reset_graph) == (
        "graph-0",
        "graph-1",
        "graph-2",
    )
    capture_calls = (
        mujoco_warp.step_calls,
        mujoco_warp.forward_calls,
        mujoco_warp.reset_calls,
    )

    backend._execute_device_steps(3)
    backend._execute_device_reset()
    backend._execute_device_forward()

    assert warp.launches == ["graph-0", "graph-0", "graph-0", "graph-2", "graph-1"]
    assert (
        mujoco_warp.step_calls,
        mujoco_warp.forward_calls,
        mujoco_warp.reset_calls,
    ) == capture_calls


def test_cuda_graph_capture_includes_materialized_reset_scratch() -> None:
    warp = _FakeWarp(driver=(12, 4), mempool=True)
    mujoco_warp = _FakeMujocoWarp()
    backend = _backend(warp, mujoco_warp)
    backend._reset_scratch_capacity = 4
    backend._reset_scratch_data = object()
    backend._reset_scratch_mask_device = object()

    backend._initialize_cuda_graphs(_FakeDevice())

    assert backend._cuda_graph_enabled is True
    assert backend._reset_scratch_reset_graph == "graph-3"
    assert backend._reset_scratch_forward_graph == "graph-4"
    assert mujoco_warp.reset_calls == 2
    assert mujoco_warp.forward_calls == 2
    assert backend._can_use_reset_scratch(4) is True
    assert backend._can_use_reset_scratch(5) is False


@pytest.mark.parametrize(
    ("num_envs", "expected_capacity"),
    [
        (512, 0),
        (1024, 128),
        (2048, 128),
        (4096, 256),
        (8192, 512),
        (16384, 512),
    ],
)
def test_reset_scratch_capacity_scales_with_batch_and_stays_bounded(
    num_envs: int, expected_capacity: int
) -> None:
    assert _reset_scratch_capacity_for_batch(num_envs) == expected_capacity


def test_ineligible_cuda_graph_warns_and_uses_eager_operations() -> None:
    warp = _FakeWarp(driver=(12, 3), mempool=True)
    mujoco_warp = _FakeMujocoWarp()
    backend = _backend(warp, mujoco_warp)

    with pytest.warns(RuntimeWarning, match="older than 12.4"):
        backend._initialize_cuda_graphs(_FakeDevice())

    backend._execute_device_steps(2)
    backend._execute_device_reset()
    backend._execute_device_forward()

    assert backend._cuda_graph_enabled is False
    assert warp.capture_count == 0
    assert warp.launches == []
    assert mujoco_warp.step_calls == 2
    assert mujoco_warp.reset_calls == 1
    assert mujoco_warp.forward_calls == 1


def test_cuda_graph_capture_failure_atomically_falls_back_to_eager() -> None:
    warp = _FakeWarp(fail_capture_index=1)
    mujoco_warp = _FakeMujocoWarp()
    backend = _backend(warp, mujoco_warp)

    with pytest.warns(RuntimeWarning, match="synthetic capture failure"):
        backend._initialize_cuda_graphs(_FakeDevice())

    assert backend._cuda_graph_enabled is False
    assert backend._step_graph is None
    assert backend._forward_graph is None
    assert backend._reset_graph is None
    assert "synthetic capture failure" in backend._cuda_graph_disable_reason
    captured_calls = (
        mujoco_warp.step_calls,
        mujoco_warp.forward_calls,
        mujoco_warp.reset_calls,
    )

    backend._execute_device_steps(2)
    backend._execute_device_reset()
    backend._execute_device_forward()

    assert warp.launches == []
    assert mujoco_warp.step_calls == captured_calls[0] + 2
    assert mujoco_warp.forward_calls == captured_calls[1] + 1
    assert mujoco_warp.reset_calls == captured_calls[2] + 1
