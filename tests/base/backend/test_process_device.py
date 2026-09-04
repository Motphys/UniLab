"""Cold-path backend process-device routing tests."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from unisim.backend.mjwarp import runtime as mjwarp_runtime

from unilab.base.process_device import (
    configure_backend_process_device,
    resolve_backend_process_device,
)


class _FakeWarpDevice:
    def __init__(self, name: str, *, is_cuda: bool = True) -> None:
        self.name = name
        self.is_cuda = is_cuda

    def __str__(self) -> str:
        return self.name


class _FakeWarp:
    def __init__(self, *, is_cuda: bool = True) -> None:
        self.is_cuda = is_cuda
        self.set_calls: list[str] = []
        self.selected = _FakeWarpDevice("cpu", is_cuda=False)

    def set_device(self, device: str) -> None:
        self.set_calls.append(device)
        self.selected = _FakeWarpDevice(device, is_cuda=self.is_cuda)

    def get_device(self) -> _FakeWarpDevice:
        return self.selected


def test_mjwarp_process_device_follows_rank_learner_device(monkeypatch: pytest.MonkeyPatch) -> None:
    warp = _FakeWarp()
    monkeypatch.setattr(
        mjwarp_runtime,
        "load_mjwarp_dependencies",
        lambda: SimpleNamespace(warp=warp),
    )

    assert configure_backend_process_device("mjwarp", "cuda:3") == "cuda:3"
    assert warp.set_calls == ["cuda:3"]


@pytest.mark.parametrize("backend_type", ["mujoco", "motrix", "drake"])
def test_host_or_backend_owned_devices_do_not_receive_runner_binding(backend_type: str) -> None:
    assert resolve_backend_process_device(backend_type, "cuda:2") is None


@pytest.mark.parametrize("device", [None, "cpu", "mps", "xpu:1"])
def test_mjwarp_process_device_fails_closed_without_cuda(device: str | None) -> None:
    with pytest.raises(ValueError, match="CUDA process device"):
        resolve_backend_process_device("mjwarp", device)


def test_mjwarp_binding_rejects_non_cuda_warp_resolution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    warp = _FakeWarp(is_cuda=False)
    monkeypatch.setattr(
        mjwarp_runtime,
        "load_mjwarp_dependencies",
        lambda: SimpleNamespace(warp=warp),
    )

    with pytest.raises(RuntimeError, match="active CUDA Warp device"):
        mjwarp_runtime.bind_mjwarp_process_device("cuda:1")
