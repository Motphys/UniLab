"""Host-free platform and affinity checks for CUDA-only tensor backends."""

from __future__ import annotations

import sys

import pytest
import torch

import unilab.base.backend_factory as backend_factory
from unilab.base.process_device import (
    configure_backend_process_device,
    resolve_backend_process_device,
)


class _Scene:
    def __init__(self) -> None:
        self.model_file = "robot.xml"
        self.visual_model_file = None
        self.fragment_files: list[str] = []
        self.entity_assets: list[object] = []
        self.entity_variant = None


@pytest.fixture
def fake_cuda(monkeypatch: pytest.MonkeyPatch):
    set_calls: list[int] = []
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(torch.cuda, "set_device", set_calls.append)
    monkeypatch.setattr(torch.version, "hip", None, raising=False)
    return set_calls


@pytest.fixture
def backend_calls(monkeypatch: pytest.MonkeyPatch):
    calls: list[dict[str, object]] = []
    monkeypatch.setattr(
        backend_factory,
        "ensure_robot_assets_for_paths",
        lambda paths: None,
    )
    monkeypatch.setattr(
        backend_factory.unisim,
        "create_backend",
        lambda *args, **kwargs: calls.append({"args": args, "kwargs": kwargs}) or object(),
    )
    return calls


def test_host_bridge_backends_do_not_require_torch_cuda(
    monkeypatch: pytest.MonkeyPatch,
    backend_calls: list[dict[str, object]],
) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    for backend in ("mujoco", "motrix", "drake", "superdex"):
        backend_factory.create_backend(backend, _Scene(), 1, 0.005)

    assert len(backend_calls) == 4


@pytest.mark.parametrize(
    ("backend_type", "kwargs", "runtime_case", "reason"),
    [
        ("newton", {"newton_device": "cpu"}, "valid", "not supported"),
        ("newton", {"newton_device": "mps"}, "valid", "not supported"),
        ("mjwarp", {}, "darwin", "macOS"),
        ("mjwarp", {}, "hip", "ROCm/HIP"),
        ("newton", {"newton_device": "cuda:0"}, "hip", "ROCm/HIP"),
        ("isaacgym", {"isaacgym_device_id": 0}, "hip", "ROCm/HIP"),
        ("isaacsim", {"isaacsim_device_id": 0}, "hip", "ROCm/HIP"),
        ("mjwarp", {}, "unavailable", "unavailable"),
        ("newton", {"newton_device": "cuda:2"}, "valid", "out of range"),
        ("newton", {"newton_device": "cuda:not-an-index"}, "valid", "not supported"),
        ("isaacgym", {"isaacgym_device_id": 1}, "valid", "does not match the learner's"),
        ("isaacsim", {"isaacsim_device_id": 1}, "valid", "does not match the learner's"),
    ],
)
def test_cuda_only_failures_happen_before_construction(
    fake_cuda: list[int],
    backend_calls: list[dict[str, object]],
    monkeypatch: pytest.MonkeyPatch,
    backend_type: str,
    kwargs: dict[str, object],
    runtime_case: str,
    reason: str,
) -> None:
    if runtime_case == "darwin":
        monkeypatch.setattr(sys, "platform", "darwin")
    elif runtime_case == "hip":
        monkeypatch.setattr(torch.version, "hip", "6.3", raising=False)
    elif runtime_case == "unavailable":
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    with pytest.raises(RuntimeError, match=reason) as exc_info:
        backend_factory.create_backend(backend_type, _Scene(), 1, 0.005, **kwargs)

    message = str(exc_info.value)
    assert backend_type in message
    assert "visible_cuda_devices=" in message
    assert "torch=" in message
    assert "platform=" in message
    assert backend_calls == []
    assert fake_cuda == []


def test_cuda_only_backend_can_construct_with_valid_cuda_device(
    fake_cuda: list[int],
    backend_calls: list[dict[str, object]],
) -> None:
    backend_factory.create_backend(
        "newton",
        _Scene(),
        1,
        0.005,
        newton_device="cuda:0",
    )

    assert len(backend_calls) == 1
    assert fake_cuda == []


def test_genesis_backend_can_construct_with_valid_rocm_device(
    fake_cuda: list[int],
    backend_calls: list[dict[str, object]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(torch.version, "hip", "7.2", raising=False)

    backend_factory.create_backend(
        "genesis",
        _Scene(),
        1,
        0.005,
        genesis_device_id=0,
    )

    assert len(backend_calls) == 1
    assert fake_cuda == [0]


@pytest.mark.parametrize("learner_device", ["cpu", "mps"])
def test_cuda_only_process_devices_reject_cpu_learner(learner_device: str) -> None:
    with pytest.raises(ValueError, match="requires a CUDA process device"):
        configure_backend_process_device("genesis", learner_device)


@pytest.mark.parametrize("learner_device", [None, "cpu", "cuda:0"])
def test_host_bridge_process_devices_do_not_receive_external_binding(
    learner_device: str | None,
) -> None:
    for backend in ("mujoco", "motrix", "drake", "superdex"):
        assert resolve_backend_process_device(backend, learner_device) is None
