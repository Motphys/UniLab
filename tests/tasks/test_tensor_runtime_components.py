from __future__ import annotations

import numpy as np
import pytest
import torch
from unisim.backend.base import (
    HostBridgeTransferPlan,
    TensorDataPlane,
    TensorExecution,
    TensorLifecycleCapabilities,
    TensorProcessTopology,
)

from unilab.managers._noise.noise_cfg import UniformNoiseCfg
from unilab.tasks.motion_tracking.common import tensor_state_store as tensor_state_store_module
from unilab.tasks.motion_tracking.common.tensor_runtime import (
    TensorEpisodeMetrics,
    TensorObservationNoise,
    TensorResetPlan,
    semantic_fingerprint,
)
from unilab.tasks.motion_tracking.common.tensor_state_store import TensorDeviceStateStore


class _SyncCountingTensor(torch.Tensor):
    synchronization_count = 0

    def tolist(self):
        type(self).synchronization_count += 1
        return super().tolist()

    def item(self):
        type(self).synchronization_count += 1
        return super().item()


class _FakeTorch:
    int64 = torch.int64

    @staticmethod
    def stack(tensors, *, out=None):
        result = torch.stack(tensors, out=out)
        if out is None:
            result = result.as_subclass(_SyncCountingTensor)
        return result


def test_semantic_fingerprint_rejects_semantic_mutation() -> None:
    base = {"std": 0.2, "enabled": True}
    mutated = dict(base)
    mutated["std"] = 0.25

    assert semantic_fingerprint("unilab.test.owner.v1", base) != semantic_fingerprint(
        "unilab.test.owner.v1", mutated
    )
    assert semantic_fingerprint("unilab.test.owner.v1", base) != semantic_fingerprint(
        "unilab.test.other.v1", base
    )


def test_tensor_observation_noise_uses_declared_uniform_bounds() -> None:
    noise = TensorObservationNoise.from_uniform_terms(
        (
            UniformNoiseCfg(n_min=-0.1, n_max=0.2),
            UniformNoiseCfg(n_min=-2.0, n_max=-1.0),
        ),
        (2, 3),
        device=torch.device("cpu"),
    )
    observations = torch.zeros((4, 6))
    generator = torch.Generator().manual_seed(17)

    corrupted = noise.apply(observations, cursor=1, generator=generator)

    assert corrupted is observations
    assert bool((corrupted[:, 1:3] >= -0.1).all()) and bool((corrupted[:, 1:3] <= 0.2).all())
    assert bool((corrupted[:, 3:6] >= -2.0).all()) and bool((corrupted[:, 3:6] <= -1.0).all())
    torch.testing.assert_close(corrupted[:, 0], torch.zeros(4))


def test_tensor_reset_plan_caches_selected_rows() -> None:
    plan = TensorResetPlan(
        terminated=torch.tensor([False, True, False, True]),
        truncated=torch.tensor([True, False, False, False]),
    )

    rows = plan.rows

    torch.testing.assert_close(rows, torch.tensor([0, 1, 3]))
    assert plan.rows is rows


def test_tensor_episode_metrics_include_terminal_then_reset() -> None:
    metrics = TensorEpisodeMetrics.create(3, torch.device("cpu"))

    metrics.update(torch.tensor([1.0, 2.0, 3.0]), torch.tensor([False, True, False]))
    finished = metrics.finished_values(torch.tensor([1]))
    metrics.reset(torch.tensor([1]))
    metrics.update(torch.tensor([4.0, 5.0, 6.0]), torch.tensor([False, False, True]))

    torch.testing.assert_close(finished, torch.tensor([[2.0, 1.0]], dtype=torch.float64))
    torch.testing.assert_close(metrics.rewards, torch.tensor([5.0, 5.0, 9.0]))
    torch.testing.assert_close(metrics.lengths, torch.tensor([2, 1, 2]))
    metrics.reset(torch.tensor([1, 2]))
    torch.testing.assert_close(metrics.rewards, torch.tensor([5.0, 0.0, 0.0]))
    torch.testing.assert_close(metrics.lengths, torch.tensor([2, 0, 0]))


class _HostBridgeBackend:
    backend_type = "fake-host"

    def __init__(self):
        self.selected_read_calls = 0

    def get_tensor_capabilities(self):
        return TensorLifecycleCapabilities(
            execution=TensorExecution.HOST_BRIDGE,
            packed_host_bridge=True,
            process_topology=TensorProcessTopology.IN_PROCESS,
            data_plane=TensorDataPlane.HOST_BRIDGE,
            stream_event_ownership="caller stream; fake synchronizes at host boundary",
            torch_devices=("cpu", "cuda"),
        )

    def compile_host_bridge_io(self, spec):
        backend = self

        class _Plan(HostBridgeTransferPlan):
            def __init__(self, plan_spec):
                self._spec = plan_spec
                self.last_timing = {}
                self._reset: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None

            @property
            def spec(self):
                return self._spec

            @property
            def transfer_stats(self):
                return {}

            def read_state_sensors(self):
                return self._views()

            def write_control(self, ctrl):
                return None

            def step(self, nsteps=1):
                return None

            def apply_reset(self, env_indices, qpos, qvel, randomization=None):
                self._reset = (env_indices, qpos.clone(), qvel.clone())
                return None

            def read_selected_state_sensors(self):
                backend.selected_read_calls += 1
                views = self._views()
                if self._reset is not None:
                    rows, qpos, qvel = self._reset
                    views["qpos"] = views["qpos"].index_copy(0, rows, qpos)
                    views["qvel"] = views["qvel"].index_copy(0, rows, qvel)
                return views

            def close(self):
                return None

            def _views(self):
                state = backend.get_state_views(("qpos", "qvel"), device="cpu")
                state["pelvis_local_linvel"] = backend.get_sensor_view(
                    "pelvis_local_linvel", device="cpu"
                )
                state["torso_gyro"] = backend.get_sensor_view("torso_gyro", device="cpu")
                state["track_pos_w_pelvis"] = backend.get_sensor_view(
                    "track_pos_w_pelvis", device="cpu"
                )
                state["track_quat_w_pelvis"] = backend.get_sensor_view(
                    "track_quat_w_pelvis", device="cpu"
                )
                state["track_linvel_w_pelvis"] = backend.get_sensor_view(
                    "track_linvel_w_pelvis", device="cpu"
                )
                state["track_angvel_w_pelvis"] = backend.get_sensor_view(
                    "track_angvel_w_pelvis", device="cpu"
                )
                return state

        return _Plan(spec)

    def get_state_views(self, names, device=None):
        assert names == ("qpos", "qvel")
        return {
            "qpos": torch.tensor(
                [[0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.25]],
                device=device,
            ),
            "qvel": torch.zeros((1, 7), device=device),
        }

    def get_sensor_view(self, name, device=None):
        values = {"pelvis_local_linvel": (1.0, 2.0, 3.0), "torso_gyro": (-1.0, -2.0, -3.0)}
        if name == "track_pos_w_pelvis":
            return torch.tensor([[0.1, 0.2, 0.3]], device=device)
        if name == "track_quat_w_pelvis":
            return torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=device)
        if name in {"track_linvel_w_pelvis", "track_angvel_w_pelvis"}:
            return torch.zeros((1, 3), device=device)
        return torch.tensor(values[name], device=device).unsqueeze(0)

    def get_body_pos_w(self, body_ids):
        return np.array([[[0.1, 0.2, 0.3]]], dtype=np.float64)

    def get_body_quat_w(self, body_ids):
        return np.array([[[1.0, 0.0, 0.0, 0.0]]], dtype=np.float64)

    def get_body_lin_vel_w(self, body_ids):
        return np.zeros((1, 1, 3), dtype=np.float64)


class _SelectedResetReadinessBackend:
    backend_type = "fake-device-reset"

    def __init__(self):
        self.stale = False
        self.step_calls: list[tuple[torch.Tensor, int]] = []

    def get_tensor_capabilities(self):
        return TensorLifecycleCapabilities(
            execution=TensorExecution.DEVICE_RESIDENT,
            state_views=True,
            state_fields=frozenset({"qpos", "qvel"}),
            sensor_views=True,
            stepping=True,
            selected_reset=True,
            process_topology=TensorProcessTopology.EXTERNAL_WORKER,
            data_plane=TensorDataPlane.CUDA_IPC,
            stream_event_ownership="backend invalidates derived views until readiness step",
            torch_devices=("cpu",),
        )

    def set_state_tensor(self, env_indices, qpos, qvel, randomization=None):
        assert randomization is None
        self.stale = True
        return {"timing": {"selected_reset_ms": 1.0}}

    def step_tensor(self, ctrl, nsteps=1):
        # The readiness call is the only public way to clear this fake's stale
        # derived-view lifecycle, matching the IsaacGym CUDA IPC contract.
        self.stale = False
        self.step_calls.append((ctrl.detach().clone(), int(nsteps)))
        return {"timing": {"readiness_ms": 2.0}}

    def get_state_views(self, names, device=None):
        assert names == ("qpos", "qvel")
        return {
            "qpos": torch.tensor([[0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.25]], device=device),
            "qvel": torch.zeros((1, 7), device=device),
        }

    def get_sensor_view(self, name, device=None):
        if self.stale:
            raise RuntimeError("derived sensor views are stale until the readiness step")
        if name == "pelvis_local_linvel":
            return torch.tensor([[1.0, 2.0, 3.0]], device=device)
        if name == "torso_gyro":
            return torch.tensor([[-1.0, -2.0, -3.0]], device=device)
        if name == "track_pos_w_pelvis":
            return torch.tensor([[0.1, 0.2, 0.3]], device=device)
        if name == "track_quat_w_pelvis":
            return torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=device)
        return torch.zeros((1, 3), device=device)

    def get_body_ang_vel_w(self, body_ids):
        return np.zeros((1, 1, 3), dtype=np.float64)


def test_tensor_state_store_full_read_validates_layout_and_finite_state() -> None:
    store = TensorDeviceStateStore(
        backend=_HostBridgeBackend(),  # pyright: ignore[reportArgumentType]
        device=torch.device("cpu"),
        num_envs=1,
        joint_qpos_ids=np.array([7], dtype=np.int64),
        joint_qvel_ids=np.array([6], dtype=np.int64),
        body_names=("pelvis",),
        body_ids=np.array([0], dtype=np.intp),
    )

    store.read()
    store.validate_finite()

    torch.testing.assert_close(store.joint_pos, torch.tensor([[0.25]]))
    torch.testing.assert_close(store.linvel, torch.tensor([[1.0, 2.0, 3.0]]))
    torch.testing.assert_close(store.robot_body_pos, torch.tensor([[[0.1, 0.2, 0.3]]]))


def test_tensor_state_store_runs_backend_readiness_after_selected_reset() -> None:
    backend = _SelectedResetReadinessBackend()
    store = TensorDeviceStateStore(
        backend=backend,  # pyright: ignore[reportArgumentType]
        device=torch.device("cpu"),
        num_envs=1,
        joint_qpos_ids=np.array([7], dtype=np.int64),
        joint_qvel_ids=np.array([6], dtype=np.int64),
        body_names=("pelvis",),
        body_ids=np.array([0], dtype=np.intp),
    )
    ctrl = torch.tensor([[0.25]], dtype=torch.float32)

    reset_result = store.apply_reset(
        torch.tensor([0], dtype=torch.int64),
        torch.tensor([[0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.25]], dtype=torch.float32),
        torch.zeros((1, 7), dtype=torch.float32),
    )
    readiness_result = store.refresh_after_selected_reset(ctrl, nsteps=3)

    store.read()

    assert reset_result == {"timing": {"selected_reset_ms": 1.0}}
    assert readiness_result == {"timing": {"readiness_ms": 2.0}}
    assert backend.step_calls == [(ctrl, 3)]
    assert backend.stale is False
    torch.testing.assert_close(store.robot_body_pos, torch.tensor([[[0.1, 0.2, 0.3]]]))

    # The barrier is idempotent after the backend has published fresh views.
    assert store.refresh_after_selected_reset(ctrl, nsteps=3) is None
    assert len(backend.step_calls) == 1


def test_tensor_state_store_selected_reset_reads_authoritative_selected_rows() -> None:
    store = TensorDeviceStateStore(
        backend=_HostBridgeBackend(),  # pyright: ignore[reportArgumentType]
        device=torch.device("cpu"),
        num_envs=1,
        joint_qpos_ids=np.array([7], dtype=np.int64),
        joint_qvel_ids=np.array([6], dtype=np.int64),
        body_names=("pelvis",),
        body_ids=np.array([0], dtype=np.intp),
    )
    store.read()
    qpos_before, qvel_before = store.qviews()
    rows = torch.tensor([0], dtype=torch.int64)
    qpos = torch.full_like(qpos_before, -2.0)
    qvel = torch.full_like(qvel_before, -3.0)

    store.apply_reset(rows, qpos, qvel)
    store.read(rows)

    torch.testing.assert_close(store.qpos, qpos)
    torch.testing.assert_close(store.qvel, qvel)
    torch.testing.assert_close(store.joint_pos, qpos[:, 7:])
    torch.testing.assert_close(store.joint_vel, qvel[:, 6:])


def test_tensor_state_store_row_validation_uses_one_bounded_sync(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = TensorDeviceStateStore(
        backend=_HostBridgeBackend(),  # pyright: ignore[reportArgumentType]
        device=torch.device("cpu"),
        num_envs=3,
        joint_qpos_ids=np.array([7], dtype=np.int64),
        joint_qvel_ids=np.array([6], dtype=np.int64),
        body_names=("pelvis",),
        body_ids=np.array([0], dtype=np.intp),
    )
    rows = torch.tensor([0, 2], dtype=torch.int64)
    monkeypatch.setattr(tensor_state_store_module, "torch", _FakeTorch)
    _SyncCountingTensor.synchronization_count = 0

    assert store._validate_rows(rows) is rows
    assert _SyncCountingTensor.synchronization_count == 1


def test_tensor_state_store_empty_rows_validate_and_read_without_sync_or_backend_read(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = _HostBridgeBackend()
    store = TensorDeviceStateStore(
        backend=backend,  # pyright: ignore[reportArgumentType]
        device=torch.device("cpu"),
        num_envs=1,
        joint_qpos_ids=np.array([7], dtype=np.int64),
        joint_qvel_ids=np.array([6], dtype=np.int64),
        body_names=("pelvis",),
        body_ids=np.array([0], dtype=np.intp),
    )
    store.read()
    qpos = store.qpos
    rows = torch.empty((0,), dtype=torch.int64)
    monkeypatch.setattr(tensor_state_store_module, "torch", _FakeTorch)
    _SyncCountingTensor.synchronization_count = 0

    store.read(rows)
    assert store._validate_rows(rows) is rows

    assert store.qpos is qpos
    assert _SyncCountingTensor.synchronization_count == 0
    assert backend.selected_read_calls == 0


@pytest.mark.parametrize("rows", [(-1,), (3,), (-1, 3)])
def test_tensor_state_store_rejects_row_range_with_one_sync(
    monkeypatch: pytest.MonkeyPatch,
    rows: tuple[int, ...],
) -> None:
    store = TensorDeviceStateStore(
        backend=_HostBridgeBackend(),  # pyright: ignore[reportArgumentType]
        device=torch.device("cpu"),
        num_envs=3,
        joint_qpos_ids=np.array([7], dtype=np.int64),
        joint_qvel_ids=np.array([6], dtype=np.int64),
        body_names=("pelvis",),
        body_ids=np.array([0], dtype=np.intp),
    )
    monkeypatch.setattr(tensor_state_store_module, "torch", _FakeTorch)
    _SyncCountingTensor.synchronization_count = 0

    with pytest.raises(IndexError, match=r"outside the environment range \[0, 3\)"):
        store._validate_rows(torch.tensor(rows, dtype=torch.int64))

    assert _SyncCountingTensor.synchronization_count == 1


def test_tensor_state_store_rejects_unsupported_backend_before_read() -> None:
    class UnsupportedBackend:
        backend_type = "fake-unsupported"

        def get_tensor_capabilities(self):
            return TensorLifecycleCapabilities(execution=TensorExecution.UNSUPPORTED)

    with pytest.raises(RuntimeError, match="DEVICE_RESIDENT or HOST_BRIDGE"):
        TensorDeviceStateStore(
            backend=UnsupportedBackend(),  # pyright: ignore[reportArgumentType]
            device=torch.device("cpu"),
            num_envs=1,
            joint_qpos_ids=np.array([0], dtype=np.int64),
            joint_qvel_ids=np.array([0], dtype=np.int64),
            body_names=("pelvis",),
            body_ids=np.array([0], dtype=np.intp),
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
@pytest.mark.parametrize("exact_device", [False, True])
def test_tensor_state_store_accepts_cuda_family_and_current_exact_device(exact_device) -> None:
    class Backend(_HostBridgeBackend):
        def get_tensor_capabilities(self):
            torch_devices = ("cuda",)
            if exact_device:
                torch_devices = (f"cuda:{torch.cuda.current_device()}",)
            return TensorLifecycleCapabilities(
                execution=TensorExecution.HOST_BRIDGE,
                packed_host_bridge=True,
                process_topology=TensorProcessTopology.IN_PROCESS,
                data_plane=TensorDataPlane.HOST_BRIDGE,
                stream_event_ownership="caller stream; fake synchronizes at host boundary",
                torch_devices=torch_devices,
            )

    current = torch.device("cuda", index=torch.cuda.current_device())
    store = TensorDeviceStateStore(
        backend=Backend(),  # pyright: ignore[reportArgumentType]
        device=torch.device("cuda"),
        num_envs=1,
        joint_qpos_ids=np.array([7], dtype=np.int64),
        joint_qvel_ids=np.array([6], dtype=np.int64),
        body_names=("pelvis",),
        body_ids=np.array([0], dtype=np.intp),
    )

    assert store.device == current


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_tensor_state_store_rejects_wrong_exact_cuda_device_before_allocation() -> None:
    class Backend:
        backend_type = "fake-wrong-cuda"

        def get_tensor_capabilities(self):
            wrong_index = (torch.cuda.current_device() + 1) % max(2, torch.cuda.device_count() + 1)
            return TensorLifecycleCapabilities(
                execution=TensorExecution.DEVICE_RESIDENT,
                process_topology=TensorProcessTopology.IN_PROCESS,
                data_plane=TensorDataPlane.DIRECT,
                stream_event_ownership="backend completes fake lifecycle",
                torch_devices=(f"cuda:{wrong_index}",),
            )

    with pytest.raises(RuntimeError, match="did not accept Tensor device"):
        TensorDeviceStateStore(
            backend=Backend(),  # pyright: ignore[reportArgumentType]
            device=torch.device("cuda"),
            num_envs=1,
            joint_qpos_ids=np.array([7], dtype=np.int64),
            joint_qvel_ids=np.array([6], dtype=np.int64),
            body_names=("pelvis",),
            body_ids=np.array([0], dtype=np.intp),
        )
