"""Cold-path backend process-device routing tests."""

from __future__ import annotations

import os
import warnings
from types import SimpleNamespace

import pytest
import torch
from unisim.backend.mjwarp import runtime as mjwarp_runtime

import unilab.base.process_device as process_device
from unilab.base.process_device import (
    apply_backend_env_device_override,
    bind_genesis_process_device,
    configure_backend_process_device,
    resolve_backend_env_device_id,
    resolve_backend_process_device,
    warn_if_backend_device_collision,
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


def test_newton_process_device_follows_rank_learner_device(monkeypatch: pytest.MonkeyPatch) -> None:
    newton_runtime = pytest.importorskip("unisim.backend.newton.runtime")
    warp = _FakeWarp()
    monkeypatch.setattr(
        newton_runtime,
        "load_newton_dependencies",
        lambda: SimpleNamespace(warp=warp),
    )

    assert configure_backend_process_device("newton", "cuda:3") == "cuda:3"
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


@pytest.mark.parametrize(
    "backend_type",
    ["isaacgym", "isaacsim", "genesis", "newton"],
)
def test_offpolicy_rank_routes_host_visible_backend_device(backend_type: str) -> None:
    assert (
        resolve_backend_env_device_id(
            backend_type,
            devices=(0, 1),
            rank=1,
            world_size=1,
            learner_device="cuda:1",
        )
        == 1
    )


@pytest.mark.parametrize(
    "backend_type",
    ["isaacgym", "isaacsim", "genesis", "newton"],
)
def test_torchrun_rank_routes_local_backend_device(backend_type: str) -> None:
    # torchrun remaps CUDA_VISIBLE_DEVICES to [4, 5], so rank 1 must send
    # local index 1 to the worker rather than host-visible index 5.
    assert (
        resolve_backend_env_device_id(
            backend_type,
            devices=(4, 5),
            rank=1,
            local_rank=1,
            world_size=2,
        )
        == 1
    )


def test_backend_env_device_override_does_not_mutate_owner_mapping() -> None:
    owner_override = {"isaacgym_device_id": 0, "nested": {"keep": True}}
    routed = apply_backend_env_device_override(
        owner_override,
        "isaacgym",
        devices=(0, 1),
        rank=1,
        world_size=1,
    )

    assert routed["isaacgym_device_id"] == 1
    assert owner_override["isaacgym_device_id"] == 0
    assert routed["nested"] is owner_override["nested"]


def test_nonzero_rank_device_zero_emits_collision_warning() -> None:
    with pytest.warns(RuntimeWarning, match=r"training\.devices=\[0, 1\]"):
        warn_if_backend_device_collision(
            "genesis",
            devices=(0, 1),
            rank=1,
            device_id=0,
        )


def test_non_gpu_backend_is_left_untouched() -> None:
    owner_override = {"isaacgym_device_id": 0}
    assert (
        apply_backend_env_device_override(
            owner_override,
            "mujoco",
            devices=(0, 1),
            rank=1,
            world_size=1,
        )
        == owner_override
    )


def test_newton_override_carries_cuda_device_string() -> None:
    # uni_rl's collector binder gate only knows mjwarp, so newton's rank-local
    # device must reach spawn collectors as a ``cuda:N`` override string.
    owner_override: dict[str, object] = {"newton_device": None, "nested": {"keep": True}}
    routed = apply_backend_env_device_override(
        owner_override,
        "newton",
        devices=(0, 1),
        rank=1,
        world_size=1,
    )

    assert routed["newton_device"] == "cuda:1"
    assert owner_override["newton_device"] is None
    assert routed["nested"] is owner_override["nested"]


def test_newton_override_falls_back_to_learner_device() -> None:
    routed = apply_backend_env_device_override(
        None,
        "newton",
        devices=None,
        rank=0,
        world_size=1,
        learner_device="cuda:0",
    )

    assert routed["newton_device"] == "cuda:0"


def test_newton_nonzero_rank_device_zero_emits_collision_warning() -> None:
    with pytest.warns(RuntimeWarning, match=r"training\.devices=\[0, 1\]"):
        warn_if_backend_device_collision(
            "newton",
            devices=(0, 1),
            rank=1,
            device_id=0,
        )


@pytest.fixture
def genesis_pin_state(monkeypatch: pytest.MonkeyPatch):
    """Host-free Genesis pin lane: stub torch CUDA state and the pin latch."""

    # monkeypatch.delitem on a missing key records no undo, so the pin's
    # direct os.environ write would leak into later tests; restore manually.
    saved_cvd = os.environ.pop("CUDA_VISIBLE_DEVICES", None)
    set_calls: list[int] = []
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: False)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    monkeypatch.setattr(torch.cuda, "set_device", set_calls.append)
    process_device._reset_genesis_device_pin_for_tests()
    try:
        yield set_calls
    finally:
        process_device._reset_genesis_device_pin_for_tests()
        if saved_cvd is None:
            os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = saved_cvd


def test_genesis_nonzero_device_pins_visible_devices(genesis_pin_state: list[int]) -> None:
    # Quadrants only honors the first visible device (issue #1508), so a
    # non-zero request shrinks CUDA_VISIBLE_DEVICES and reports the
    # in-process index 0 to the rest of the process.
    assert bind_genesis_process_device("cuda:1") == "cuda:0"
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "1"
    assert genesis_pin_state == [0]


def test_genesis_pin_translates_existing_visible_devices(
    genesis_pin_state: list[int],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(os.environ, "CUDA_VISIBLE_DEVICES", "4,5")

    assert bind_genesis_process_device("cuda:1") == "cuda:0"
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "5"


def test_genesis_device_zero_binds_without_pin(genesis_pin_state: list[int]) -> None:
    assert bind_genesis_process_device("cuda:0") == "cuda:0"
    assert "CUDA_VISIBLE_DEVICES" not in os.environ
    assert genesis_pin_state == [0]


def test_genesis_pin_fails_closed_after_cuda_init(
    genesis_pin_state: list[int],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: True)

    with pytest.raises(RuntimeError, match="before any CUDA context"):
        bind_genesis_process_device("cuda:1")
    assert "CUDA_VISIBLE_DEVICES" not in os.environ


def test_genesis_pin_is_idempotent_for_repeated_rank_binding(
    genesis_pin_state: list[int],
) -> None:
    # Entrypoints bind twice (main + defensive runner rebind); a stale pre-pin
    # index from the same rank maps onto the pinned in-process device.
    assert bind_genesis_process_device("cuda:1") == "cuda:0"
    assert bind_genesis_process_device("cuda:1") == "cuda:0"
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "1"
    assert genesis_pin_state == [0, 0]


def test_genesis_resolution_uses_pinned_namespace(genesis_pin_state: list[int]) -> None:
    bind_genesis_process_device("cuda:1")

    # After the pin every in-process consumer resolves to index 0, and the
    # transition collision guard stays quiet because each rank owns its GPU.
    assert (
        resolve_backend_env_device_id(
            "genesis",
            devices=(0, 1),
            rank=1,
            world_size=1,
            learner_device="cuda:1",
        )
        == 0
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        warn_if_backend_device_collision("genesis", devices=(0, 1), rank=1, device_id=0)


def test_genesis_pin_rejects_index_beyond_visible_devices(
    genesis_pin_state: list[int],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(os.environ, "CUDA_VISIBLE_DEVICES", "3")

    with pytest.raises(ValueError, match="CUDA_VISIBLE_DEVICES"):
        bind_genesis_process_device("cuda:1")
