"""Cold-path backend process-device routing tests."""

from __future__ import annotations

import dataclasses
import os
from types import SimpleNamespace

import pytest
import torch
from unisim.backend.mjwarp import runtime as mjwarp_runtime

import unilab.base.process_device as process_device
from unilab.base.process_device import (
    apply_backend_env_device_override,
    apply_collector_tensor_device_override,
    bind_genesis_process_device,
    configure_backend_process_device,
    resolve_backend_env_device_id,
    resolve_backend_process_device,
)
from unilab.rl.runtime.distributed.context import visible_cuda_entries


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


@pytest.mark.parametrize("backend_type", ["mjwarp", "newton", "isaacgym", "isaacsim"])
@pytest.mark.parametrize("device", [None, "cpu", "mps", "xpu:1"])
def test_bound_process_device_fails_closed_without_cuda(
    backend_type: str, device: str | None
) -> None:
    with pytest.raises(ValueError, match="CUDA process device"):
        resolve_backend_process_device(backend_type, device)


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


@pytest.mark.parametrize("backend_type", ["genesis", "newton", "isaacgym", "isaacsim"])
def test_explicit_single_device_fallback_routes_backend_payload(
    backend_type: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)

    assert resolve_backend_env_device_id(backend_type, learner_device="cuda:1") == 1


@pytest.mark.parametrize("backend_type", ["isaacgym", "isaacsim"])
def test_external_cuda_ipc_binding_matches_payload_device(
    backend_type: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    set_devices: list[int] = []
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 1)
    monkeypatch.setattr(torch.cuda, "set_device", set_devices.append)

    assert (
        configure_backend_process_device(
            backend_type,
            "cuda:1",
            backend_device_id=1,
        )
        == "cuda:1"
    )
    assert set_devices == [1]

    with pytest.raises(ValueError, match="does not match its backend payload device"):
        configure_backend_process_device(
            backend_type,
            "cuda:1",
            backend_device_id=0,
        )


def test_single_gpu_mjwarp_process_binding_needs_no_visibility_mask(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    warp = _FakeWarp()
    monkeypatch.setattr(
        mjwarp_runtime,
        "load_mjwarp_dependencies",
        lambda: SimpleNamespace(warp=warp),
    )

    resolved = configure_backend_process_device("mjwarp", "cuda:0")

    assert resolved == "cuda:0"
    assert warp.set_calls == ["cuda:0"]


@pytest.mark.parametrize("backend_type", ["genesis", "newton"])
def test_multi_visible_rank_fails_closed(
    backend_type: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-a,GPU-b")

    with pytest.raises(ValueError, match="exactly one CUDA_VISIBLE_DEVICES"):
        resolve_backend_env_device_id(backend_type)


def test_rank_local_visibility_entries_are_opaque(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    assert visible_cuda_entries(None) == ()
    assert visible_cuda_entries("GPU-a,MIG-b") == ("GPU-a", "MIG-b")


def test_backend_env_device_override_routes_rank_local_zero(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-rank-local")
    owner_override = {"genesis_device_id": 7, "nested": {"keep": True}}

    routed = apply_backend_env_device_override(owner_override, "genesis")

    assert routed["genesis_device_id"] == 0
    assert owner_override["genesis_device_id"] == 7
    assert routed["nested"] is owner_override["nested"]


def test_collector_tensor_device_override_is_explicit_and_rank_local() -> None:
    owner_override = {"collector_tensor_device": "owner-cpu", "nested": {"keep": True}}

    routed = apply_collector_tensor_device_override(
        owner_override,
        "mujoco",
        collector_tensor_device="cuda",
        learner_device="cuda:3",
    )
    assert routed["collector_tensor_device"] == "cuda"
    assert routed["nested"] is owner_override["nested"]
    assert owner_override["collector_tensor_device"] == "owner-cpu"

    with pytest.raises(ValueError, match="training.collector_tensor_device"):
        apply_collector_tensor_device_override(
            owner_override, "mujoco", collector_tensor_device="cuda:3"
        )

    for backend_type in ("motrix", "drake", "superdex"):
        routed = apply_collector_tensor_device_override(
            owner_override,
            backend_type,
            collector_tensor_device="cuda",
            learner_device="cuda:3",
        )
        assert routed["collector_tensor_device"] == "cuda"
        assert owner_override["collector_tensor_device"] == "owner-cpu"

    with pytest.raises(ValueError, match="requires a HOST_BRIDGE"):
        apply_collector_tensor_device_override(
            owner_override, "mjwarp", collector_tensor_device="cuda"
        )


def test_explicit_cpu_carrier_with_cuda_learner_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Issue #2483: a CUDA learner may not pin the CPU carrier by default."""
    monkeypatch.delenv("UNILAB_ALLOW_CUDA_LEARNER_CPU_CARRIER", raising=False)
    with pytest.raises(ValueError, match="collector_tensor_device=cpu with CUDA learner"):
        apply_collector_tensor_device_override(
            {},
            "mujoco",
            collector_tensor_device="cpu",
            learner_device="cuda:3",
        )


def test_explicit_cpu_carrier_with_cuda_learner_honors_opt_in_and_rocm(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    owner_override = {"nested": {"keep": True}}

    monkeypatch.setenv("UNILAB_ALLOW_CUDA_LEARNER_CPU_CARRIER", "1")
    routed = apply_collector_tensor_device_override(
        owner_override,
        "mujoco",
        collector_tensor_device="cpu",
        learner_device="cuda:3",
    )
    assert routed["collector_tensor_device"] == "cpu"

    # ROCm/HIP builds keep the CPU carrier as their only sharing mitigation.
    monkeypatch.delenv("UNILAB_ALLOW_CUDA_LEARNER_CPU_CARRIER")
    monkeypatch.setattr(torch.version, "hip", "6.2")
    routed = apply_collector_tensor_device_override(
        owner_override,
        "mujoco",
        collector_tensor_device="cpu",
        learner_device="cuda:3",
    )
    assert routed["collector_tensor_device"] == "cpu"


def test_explicit_cpu_carrier_stays_allowed_for_non_cuda_or_unknown_learners() -> None:
    for learner_device in (None, "cpu", "mps"):
        routed = apply_collector_tensor_device_override(
            {},
            "mujoco",
            collector_tensor_device="cpu",
            learner_device=learner_device,
        )
        assert routed["collector_tensor_device"] == "cpu"


def test_collector_tensor_device_auto_follows_the_learner_device() -> None:
    owner_override = {"collector_tensor_device": "owner-cpu", "nested": {"keep": True}}

    routed = apply_collector_tensor_device_override(
        owner_override,
        "mujoco",
        collector_tensor_device="auto",
        learner_device="cuda:3",
    )
    assert routed["collector_tensor_device"] == "cuda:3"
    assert routed["nested"] is owner_override["nested"]
    assert owner_override["collector_tensor_device"] == "owner-cpu"

    for learner_device in ("cpu", "mps"):
        routed = apply_collector_tensor_device_override(
            owner_override,
            "mujoco",
            collector_tensor_device="auto",
            learner_device=learner_device,
        )
        expected = "cpu" if learner_device == "cpu" else owner_override["collector_tensor_device"]
        assert routed["collector_tensor_device"] == expected

    routed = apply_collector_tensor_device_override(
        owner_override,
        "mujoco",
        collector_tensor_device="auto",
        learner_device=None,
    )
    assert routed == owner_override

    for backend_type in ("motrix", "drake", "superdex"):
        routed = apply_collector_tensor_device_override(
            owner_override,
            backend_type,
            collector_tensor_device="auto",
            learner_device="cuda:3",
        )
        assert routed["collector_tensor_device"] == "cuda:3"

    # Backends outside the HOST_BRIDGE scope ignore the auto request.
    routed = apply_collector_tensor_device_override(
        owner_override,
        "mjwarp",
        collector_tensor_device="auto",
        learner_device="cuda:3",
    )
    assert routed == owner_override


def test_collector_tensor_device_fails_closed_on_capability_inventory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    profile = process_device._host_bridge_profile("motrix")
    assert profile is not None
    cpu_only = dataclasses.replace(profile, torch_devices=("cpu",))
    monkeypatch.setattr(
        process_device,
        "_host_bridge_profile",
        lambda backend_type: cpu_only if backend_type == "motrix" else None,
    )

    with pytest.raises(ValueError, match="accepted Torch devices are 'cpu'"):
        apply_collector_tensor_device_override({}, "motrix", collector_tensor_device="cuda")

    with pytest.raises(ValueError, match="backend capability declaration"):
        apply_collector_tensor_device_override(
            {}, "motrix", collector_tensor_device="auto", learner_device="cuda:3"
        )

    assert (
        apply_collector_tensor_device_override({}, "motrix", collector_tensor_device="cpu")[
            "collector_tensor_device"
        ]
        == "cpu"
    )


def test_playback_device_uses_collector_tensor_device_without_training_request() -> None:
    owner_override = {"collector_tensor_device": "owner-cpu", "nested": {"keep": True}}

    routed = apply_collector_tensor_device_override(owner_override, "mujoco", learner_device=None)
    assert routed is not owner_override
    assert routed == owner_override

    routed = apply_collector_tensor_device_override(owner_override, "mujoco", learner_device="cpu")
    assert routed["collector_tensor_device"] == "cpu"

    routed = apply_collector_tensor_device_override(
        owner_override, "mujoco", learner_device="cuda:3"
    )
    assert routed["collector_tensor_device"] == "cuda:3"

    routed = apply_collector_tensor_device_override(owner_override, "mujoco", learner_device="mps")
    assert routed["collector_tensor_device"] == "owner-cpu"

    routed = apply_collector_tensor_device_override(
        owner_override, "mjwarp", learner_device="cuda:3"
    )
    assert routed["collector_tensor_device"] == "owner-cpu"


def test_removed_device_keys_fail_closed_with_migration_hint() -> None:
    with pytest.raises(ValueError, match="env.manager_torch_device was removed"):
        apply_collector_tensor_device_override(
            {"manager_torch_device": "cpu"}, "mujoco", collector_tensor_device="auto"
        )
    with pytest.raises(ValueError, match="env.sim_carrier_device was removed"):
        apply_collector_tensor_device_override(
            {"sim_carrier_device": "cpu"}, "mujoco", collector_tensor_device="auto"
        )


def test_collector_tensor_device_is_canonicalized_for_manifests(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert process_device.canonicalize_collector_tensor_device(None) == "cpu"
    assert process_device.canonicalize_collector_tensor_device("cpu") == "cpu"

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 1)
    assert process_device.canonicalize_collector_tensor_device("cuda") == "cuda:1"
    assert process_device.canonicalize_collector_tensor_device("cuda:0") == "cuda:0"


def test_non_gpu_backend_is_left_untouched() -> None:
    owner_override = {"genesis_device_id": 0}
    assert apply_backend_env_device_override(owner_override, "mujoco") == owner_override


@pytest.mark.parametrize("backend", ["motrix_gpu", "newton"])
def test_backend_override_carries_cuda_device_string(
    monkeypatch: pytest.MonkeyPatch,
    backend: str,
) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-rank-local")
    field = f"{backend}_device"
    owner_override: dict[str, object] = {field: None, "nested": {"keep": True}}

    routed = apply_backend_env_device_override(owner_override, backend)

    assert routed[field] == "cuda:0"
    assert owner_override[field] is None
    assert routed["nested"] is owner_override["nested"]


def test_newton_override_falls_back_to_learner_device() -> None:
    routed = apply_backend_env_device_override(
        None,
        "newton",
        learner_device="cuda:0",
    )

    assert routed["newton_device"] == "cuda:0"


@pytest.fixture()
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


def test_genesis_pin_rejects_index_beyond_visible_devices(
    genesis_pin_state: list[int],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(os.environ, "CUDA_VISIBLE_DEVICES", "3")

    with pytest.raises(ValueError, match="CUDA_VISIBLE_DEVICES"):
        bind_genesis_process_device("cuda:1")
