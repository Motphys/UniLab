"""Focused tests for the scene-owned packed tensor read boundary."""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MethodType, SimpleNamespace
from typing import cast

import numpy as np
import pytest
import torch
from unisim.backend.base import (
    HostBridgeTransferPlan,
    SimBackend,
    TensorDataPlane,
    TensorExecution,
    TensorIOSpec,
    TensorLifecycleCapabilities,
    TensorProcessTopology,
    TrackedBodyStateViews,
)

from unilab.base.entity import EntityCfg, EntityScene, SceneTensorReadSpec


class _SceneTensorBackend:
    """Public-contract fake with a minimal packed host-bridge plan."""

    backend_type = "fake-scene-tensor"
    num_envs = 2
    num_actuators = 1

    def __init__(self, *, device_resident: bool = False) -> None:
        self.device_resident = device_resident
        self.full_reads = 0
        self.selected_reads = 0
        self.sensor_view_calls = 0
        self.aggregate_body_view_calls = 0
        self.closes = 0
        self.io_specs: list[TensorIOSpec] = []
        self.qpos = torch.tensor([[0.1], [0.3]], dtype=torch.float32)
        self.qvel = torch.tensor([[10.0], [20.0]], dtype=torch.float32)
        self.sensors = {
            "imu_gyro": torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=torch.float32),
            "track_pos_w_hip": torch.tensor(
                [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], dtype=torch.float32
            ),
            "track_quat_w_hip": torch.tensor(
                [[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]], dtype=torch.float32
            ),
            "track_linvel_w_hip": torch.zeros((2, 3), dtype=torch.float32),
            "track_angvel_w_hip": torch.ones((2, 3), dtype=torch.float32),
        }
        self.body_blocks = {
            "body_pos_w": self.sensors["track_pos_w_hip"].reshape(2, 3).contiguous(),
            "body_quat_w": self.sensors["track_quat_w_hip"].reshape(2, 4).contiguous(),
            "body_lin_vel_w": self.sensors["track_linvel_w_hip"].reshape(2, 3).contiguous(),
            "body_ang_vel_w": self.sensors["track_angvel_w_hip"].reshape(2, 3).contiguous(),
        }
        self._selected_rows = torch.empty((0,), dtype=torch.int64)

    def get_tensor_capabilities(self) -> TensorLifecycleCapabilities:
        if self.device_resident:
            return TensorLifecycleCapabilities(
                execution=TensorExecution.DEVICE_RESIDENT,
                state_views=True,
                state_fields=frozenset(("qpos", "qvel")),
                sensor_views=True,
                selected_reset=True,
                tracked_body_views=True,
                process_topology=TensorProcessTopology.IN_PROCESS,
                data_plane=TensorDataPlane.DIRECT,
                stream_event_ownership="caller owns the Torch stream",
                torch_devices=("cpu",),
            )
        return TensorLifecycleCapabilities(
            execution=TensorExecution.HOST_BRIDGE,
            state_views=True,
            state_fields=frozenset(("qpos", "qvel")),
            sensor_views=True,
            packed_host_bridge=True,
            process_topology=TensorProcessTopology.IN_PROCESS,
            data_plane=TensorDataPlane.HOST_BRIDGE,
            stream_event_ownership="backend synchronizes at the packed host boundary",
            torch_devices=("cpu",),
        )

    def get_tensor_execution(self) -> TensorExecution:
        return (
            TensorExecution.DEVICE_RESIDENT if self.device_resident else TensorExecution.HOST_BRIDGE
        )

    def get_state_views(self, names, device=None) -> dict[str, torch.Tensor]:
        assert set(names) <= {"qpos", "qvel"}
        target = torch.device(device) if device is not None else torch.device("cpu")
        return {
            "qpos": self.qpos.to(target),
            "qvel": self.qvel.to(target),
        }

    def get_sensor_view(self, name: str, device: str | torch.device = "cpu") -> torch.Tensor:
        self.sensor_view_calls += 1
        return self.sensors[name].to(torch.device(device))

    def get_tracked_body_views(self, body_names=None, device=None) -> TrackedBodyStateViews:
        self.aggregate_body_view_calls += 1
        names = ("hip",) if body_names is None else tuple(body_names)
        names = tuple(name.rsplit("/", 1)[-1] for name in names)
        if names != ("hip",):
            raise ValueError(f"unknown tracked bodies: {names}")
        target = torch.device(device) if device is not None else torch.device("cpu")
        return TrackedBodyStateViews(
            body_names=names,
            pos_w=self.sensors["track_pos_w_hip"].unsqueeze(1).to(target),
            quat_w=self.sensors["track_quat_w_hip"].unsqueeze(1).to(target),
            lin_vel_w=self.sensors["track_linvel_w_hip"].unsqueeze(1).to(target),
            ang_vel_w=self.sensors["track_angvel_w_hip"].unsqueeze(1).to(target),
        )

    def get_actuator_names(self) -> tuple[str, ...]:
        return ("hip_motor", "knee_motor")[: self.num_actuators]

    def get_actuator_joint_names(self) -> tuple[str, ...]:
        return ("hip", "knee")[: self.num_actuators]

    def get_actuator_ctrl_range(self):
        ctrl_range = torch.tensor([[-1.0, 1.0]], dtype=torch.float32)
        return torch.tile(ctrl_range, (self.num_actuators, 1)).numpy()

    def get_joint_range(self, names=None):
        count = len(names) if names is not None else 1
        return torch.tile(torch.tensor([[-1.0, 1.0]], dtype=torch.float32), (count, 1)).numpy()

    def get_default_dof_pos(self, names=None):
        return torch.zeros(len(names) if names is not None else 1, dtype=torch.float32).numpy()

    def get_dof_pos(self):
        return self.qpos.numpy()

    def get_dof_vel(self):
        return self.qvel.numpy()

    def get_body_pos_w(self, body_ids):
        return torch.tile(
            self.sensors["track_pos_w_hip"].unsqueeze(1), (1, len(body_ids), 1)
        ).numpy()

    def get_body_quat_w(self, body_ids):
        return torch.tile(
            self.sensors["track_quat_w_hip"].unsqueeze(1), (1, len(body_ids), 1)
        ).numpy()

    def get_body_lin_vel_w(self, body_ids):
        return torch.tile(
            self.sensors["track_linvel_w_hip"].unsqueeze(1), (1, len(body_ids), 1)
        ).numpy()

    def get_body_ang_vel_w(self, body_ids):
        return torch.tile(
            self.sensors["track_angvel_w_hip"].unsqueeze(1), (1, len(body_ids), 1)
        ).numpy()

    def get_body_lin_vel_b(self, body_ids):
        return self.get_body_lin_vel_w(body_ids)

    def get_body_ang_vel_b(self, body_ids):
        return self.get_body_ang_vel_w(body_ids)

    def get_joint_state_qpos_indices(self, names) -> np.ndarray:
        import numpy as np

        return np.zeros(len(names), dtype=np.int32)

    def get_joint_state_qvel_indices(self, names) -> np.ndarray:
        import numpy as np

        return np.zeros(len(names), dtype=np.int32)

    def get_body_ids(self, names) -> np.ndarray:
        import numpy as np

        return np.arange(len(names), dtype=np.int32)

    def get_joint_dof_pos_indices(self, names) -> np.ndarray:
        import numpy as np

        return np.zeros(len(names), dtype=np.int32)

    def get_joint_dof_vel_indices(self, names) -> np.ndarray:
        import numpy as np

        return np.zeros(len(names), dtype=np.int32)

    def bind_host_bridge_io(self, spec: TensorIOSpec):
        backend = self
        if getattr(spec, "body_names", ()):
            backend.io_specs.append(spec)

        class _Plan(HostBridgeTransferPlan):
            def __init__(self, plan_spec: TensorIOSpec) -> None:
                self._spec = plan_spec
                self.last_timing: dict[str, float] = {}
                self._published_packet: dict[str, torch.Tensor] | None = None

            @property
            def spec(self) -> TensorIOSpec:
                return self._spec

            @property
            def transfer_stats(self) -> dict[str, int]:
                return {
                    "state_sensor_h2d": backend.full_reads + backend.selected_reads,
                    "selected_post_reset_h2d": backend.selected_reads,
                }

            def read_state_sensors(self):
                backend.full_reads += 1
                self._published_packet = backend._packet()
                return self._published_packet

            def read_selected_state_sensors(self):
                backend.selected_reads += 1
                source = backend._packet()
                if self._published_packet is None:
                    self._published_packet = source
                else:
                    packet = {name: value.clone() for name, value in self._published_packet.items()}
                    for name, value in source.items():
                        packet[name][backend._selected_rows] = value[backend._selected_rows]
                    self._published_packet = packet
                return self._published_packet

            def get_body_pose_views(self):
                return SimpleNamespace(
                    body_names=("robot/hip",),
                    pos_w=backend.body_blocks["body_pos_w"],
                    quat_w=backend.body_blocks["body_quat_w"],
                )

            def write_control(self, ctrl) -> None:
                return None

            def step(self, nsteps: int = 1):
                return None

            def apply_reset(self, env_indices, qpos, qvel, randomization=None):
                backend._selected_rows = env_indices
                return None

            def close(self) -> None:
                backend.closes += 1

        return _Plan(spec)

    def _packet(self) -> dict[str, torch.Tensor]:
        packet = self.get_state_views(("qpos", "qvel"), device="cpu")
        packet.update(self.sensors)
        if self.device_resident:
            return packet
        return packet


def _scene(backend: _SceneTensorBackend) -> EntityScene:
    return EntityScene(
        {
            "robot": EntityCfg(
                joint_names=("hip",),
                body_names=("hip",),
                actuator_names=("hip_motor",),
            )
        },
        cast(SimBackend, backend),
    )


def _use_body_capable_tensor_spec(monkeypatch: pytest.MonkeyPatch) -> None:
    """Expose a UniSim-2.2-shaped public TensorIOSpec for packed-block tests."""

    @dataclass(frozen=True)
    class _BodyCapableTensorIOSpec(TensorIOSpec):
        body_names: tuple[str, ...] = ()
        body_fields: tuple[str, ...] = ()
        device: object | None = field(default=None)

    monkeypatch.setattr("unilab.base.entity.TensorIOSpec", _BodyCapableTensorIOSpec)


def _use_legacy_tensor_spec(monkeypatch: pytest.MonkeyPatch) -> None:
    """Expose a pre-2.2 TensorIOSpec without packed body-block parameters."""

    @dataclass(frozen=True)
    class _LegacyTensorIOSpec:
        state_fields: tuple[str, ...]
        sensor_names: tuple[str, ...]
        device: object | None = None

    monkeypatch.setattr("unilab.base.entity.TensorIOSpec", _LegacyTensorIOSpec)


def _specs() -> tuple[SceneTensorReadSpec, ...]:
    return (
        SceneTensorReadSpec(entity="robot", sensor_names=("imu_gyro",)),
        SceneTensorReadSpec(entity="robot", body_names=("hip",)),
    )


def test_host_bridge_scene_plan_packs_all_reads_into_one_phase_transfer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = _SceneTensorBackend()
    _use_body_capable_tensor_spec(monkeypatch)
    scene = _scene(backend)
    plan = scene.compile_tensor_reads("cpu", _specs())

    assert backend.full_reads == 0
    plan.refresh()
    assert backend.full_reads == 1

    joints = plan.joint_tensor_view("robot")
    sensors = plan.sensor_tensor_views("robot", ("imu_gyro",))
    body = plan.body_tensor_view("robot")
    assert plan.transfer_stats["state_sensor_h2d"] == 1
    assert joints.joint_pos.shape == (2, 1)
    torch.testing.assert_close(joints.joint_pos[:, 0], backend.qpos[:, 0])
    torch.testing.assert_close(sensors.values["imu_gyro"], backend.sensors["imu_gyro"])
    torch.testing.assert_close(body.pos_w[:, 0], backend.sensors["track_pos_w_hip"])
    assert body.body_names == ("hip",)

    plan.refresh_selected()
    assert backend.selected_reads == 1
    assert plan.transfer_stats == {
        "state_sensor_h2d": 2,
        "selected_post_reset_h2d": 1,
    }

    plan.close()


def test_scene_plan_packs_actuator_force_sensors_in_request_order() -> None:
    backend = _SceneTensorBackend()
    backend.num_actuators = 2
    force_names = ("force_knee", "force_hip")
    backend.sensors.update(
        {
            "force_hip": torch.tensor([[1.0], [2.0]], dtype=torch.float32),
            "force_knee": torch.tensor([[10.0], [20.0]], dtype=torch.float32),
        }
    )
    plan = _scene(backend).compile_tensor_reads(
        "cpu", (SceneTensorReadSpec(entity="robot", sensor_names=force_names),)
    )
    assert plan.host_plan is not None
    assert plan.host_plan.spec.sensor_names == force_names

    def force_tensor() -> torch.Tensor:
        views = plan.sensor_tensor_views("robot", force_names).values
        return torch.cat([views[name] for name in force_names], dim=1)

    plan.refresh()
    initial_force = force_tensor()
    assert initial_force.shape == (backend.num_envs, backend.num_actuators)
    torch.testing.assert_close(initial_force, torch.tensor([[10.0, 1.0], [20.0, 2.0]]))
    assert backend.sensor_view_calls == 0

    backend.sensors["force_hip"] = torch.tensor([[101.0], [102.0]])
    backend.sensors["force_knee"] = torch.tensor([[110.0], [120.0]])
    rows = torch.tensor([1], dtype=torch.int64)
    plan.host_plan.apply_reset(
        rows,
        torch.zeros((1, backend.qpos.shape[1]), dtype=torch.float32),
        torch.zeros((1, backend.qvel.shape[1]), dtype=torch.float32),
    )
    plan.refresh_selected()
    selected_force = force_tensor()

    torch.testing.assert_close(selected_force[0], initial_force[0])
    torch.testing.assert_close(selected_force[1], torch.tensor([120.0, 102.0]))
    assert backend.full_reads == 1
    assert backend.selected_reads == 1
    assert backend.sensor_view_calls == 0
    assert plan.transfer_stats == {
        "state_sensor_h2d": 2,
        "selected_post_reset_h2d": 1,
    }
    plan.close()


def test_scene_plan_pose_reads_fall_back_to_canonical_sensors_on_older_host_contracts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = _SceneTensorBackend()
    scene = _scene(backend)
    _use_legacy_tensor_spec(monkeypatch)
    plan = scene.compile_tensor_reads(
        "cpu",
        (
            SceneTensorReadSpec(
                entity="robot", body_names=("hip",), body_fields=("pos_w", "quat_w")
            ),
        ),
    )

    assert plan.packet_body_blocks is False
    assert plan.packet_names == ("track_pos_w_hip", "track_quat_w_hip")
    assert backend.io_specs == []
    plan.refresh()
    pose = plan.ray_pose_view("robot")
    assert pose.body_names == ("hip",)
    torch.testing.assert_close(pose.pos_w[:, 0], backend.sensors["track_pos_w_hip"])
    torch.testing.assert_close(pose.quat_w[:, 0], backend.sensors["track_quat_w_hip"])
    plan.close()
    assert backend.closes == 1


def test_device_resident_scene_plan_refreshes_stable_public_views() -> None:
    backend = _SceneTensorBackend(device_resident=True)
    scene = _scene(backend)
    plan = scene.compile_tensor_reads("cpu", _specs())

    assert plan.host_plan is None
    plan.refresh()
    body = plan.body_tensor_view("robot")
    torch.testing.assert_close(body.ang_vel_w[:, 0], backend.sensors["track_angvel_w_hip"])


def test_device_resident_scene_plan_reuses_aggregate_body_views() -> None:
    backend = _SceneTensorBackend(device_resident=True)
    scene = _scene(backend)
    plan = scene.compile_tensor_reads("cpu", _specs())

    plan.refresh()
    first = plan.body_tensor_view("robot")
    second = plan.body_tensor_view("robot")

    assert first is second
    assert backend.aggregate_body_view_calls == 1
    assert first.body_names == ("hip",)
    torch.testing.assert_close(first.pos_w[:, 0], backend.sensors["track_pos_w_hip"])

    plan.refresh()
    third = plan.body_tensor_view("robot")

    assert third is not first
    assert backend.aggregate_body_view_calls == 2


def test_device_resident_scene_plan_refreshes_every_sensor_boundary() -> None:
    backend = _SceneTensorBackend(device_resident=True)
    scene = _scene(backend)
    plan = scene.compile_tensor_reads("cpu", _specs())

    plan.refresh()
    calls_after_first_refresh = backend.sensor_view_calls
    first = plan.sensor_tensor_views("robot", ("imu_gyro",)).values["imu_gyro"]

    # Model adapter-owned derived-sensor semantics: a later public boundary call
    # returns a newly materialized projection rather than mutating the old one.
    backend.sensors["imu_gyro"] = torch.full((2, 3), 7.0, dtype=torch.float32)
    plan.refresh()
    second = plan.sensor_tensor_views("robot", ("imu_gyro",)).values["imu_gyro"]

    assert first is not second
    torch.testing.assert_close(first, torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]))
    torch.testing.assert_close(second, torch.full((2, 3), 7.0))
    # Every compiled sensor crosses its public backend boundary on every phase.
    # Genesis/Newton derived sensors are stable outputs, not native-live arrays.
    assert backend.sensor_view_calls == calls_after_first_refresh + len(plan.sensor_names["robot"])
    assert plan.transfer_stats == {}
    assert plan.last_timing == {}
    plan.close()


def test_device_resident_selected_reset_refreshes_body_sensor_boundary() -> None:
    backend = _SceneTensorBackend(device_resident=True)
    scene = _scene(backend)
    plan = scene.compile_tensor_reads("cpu", _specs())

    plan.refresh_selected()
    calls_after_selected_refresh = backend.sensor_view_calls

    backend.sensors["track_pos_w_hip"] = torch.full((2, 3), 9.0, dtype=torch.float32)
    plan.refresh_selected()
    body = plan.body_tensor_view("robot")

    torch.testing.assert_close(body.pos_w[:, 0], backend.sensors["track_pos_w_hip"])
    assert backend.sensor_view_calls == calls_after_selected_refresh + len(
        plan.sensor_names["robot"]
    )
    plan.close()


def test_scene_plan_packed_joint_layout_is_cached_and_finite_checks_are_deferred() -> None:
    backend = _SceneTensorBackend()
    backend.layout_reads = 0
    original_qpos_indices = backend.get_joint_state_qpos_indices

    def counting_qpos_indices(names):
        backend.layout_reads += 1
        return original_qpos_indices(names)

    backend.get_joint_state_qpos_indices = counting_qpos_indices
    scene = _scene(backend)
    plan = scene.compile_tensor_reads("cpu", _specs())
    plan.refresh()

    first = plan.joint_tensor_view("robot")
    second = plan.joint_tensor_view("robot")

    assert first.joint_pos.shape == (2, 1)
    torch.testing.assert_close(first.joint_pos, second.joint_pos)
    assert backend.layout_reads == 1
    # Packed packets still enforce carrier shape/dtype/device. Finiteness is
    # deferred to the Manager term/result boundary instead of every packet read.
    packet = dict(plan._packet)
    packet["qpos"] = torch.tensor([[torch.nan], [torch.nan]], dtype=torch.float32)
    plan._packet = type(plan._packet)(packet)
    nonfinite = plan.joint_tensor_view("robot")
    assert bool(torch.isnan(nonfinite.joint_pos).all())

    plan.close()


def test_scene_plan_pose_reads_use_packed_pose_blocks_without_velocities(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = _SceneTensorBackend()
    _use_body_capable_tensor_spec(monkeypatch)
    scene = _scene(backend)
    plan = scene.compile_tensor_reads(
        "cpu",
        (
            SceneTensorReadSpec(
                entity="robot", body_names=("hip",), body_fields=("pos_w", "quat_w")
            ),
        ),
    )

    assert plan.body_fields == {"robot": ("pos_w", "quat_w")}
    assert plan.packet_names == ("body_pos_w", "body_quat_w")
    if backend.io_specs:
        assert backend.io_specs[0].body_names == ("hip",)
        assert backend.io_specs[0].body_fields == ("pos_w", "quat_w")
    assert plan.packet_body_axis("body_pos_w") == ("hip",)
    plan.refresh()

    pose = plan.ray_pose_view("robot")
    assert pose.body_names == ("hip",)
    assert pose.pos_w.shape == (2, 1, 3)
    assert pose.quat_w.shape == (2, 1, 4)
    torch.testing.assert_close(pose.pos_w[:, 0], backend.sensors["track_pos_w_hip"])
    torch.testing.assert_close(pose.quat_w[:, 0], backend.sensors["track_quat_w_hip"])
    body = plan.body_tensor_view("robot")
    assert body.pos_w is not None and body.quat_w is not None
    assert body.lin_vel_w is None and body.ang_vel_w is None
    plan.close()


def test_scene_plan_ray_pose_view_preserves_row_strided_packet_slices(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = _SceneTensorBackend()
    _use_body_capable_tensor_spec(monkeypatch)
    scene = _scene(backend)
    plan = scene.compile_tensor_reads(
        "cpu",
        (
            SceneTensorReadSpec(
                entity="robot", body_names=("hip",), body_fields=("pos_w", "quat_w")
            ),
        ),
    )

    storage = torch.zeros((2, 12), dtype=torch.float32)
    pos = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    quat = torch.tensor(((1.0, 0.0, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0)), dtype=torch.float32)
    storage[:, 3:6] = pos
    storage[:, 7:11] = quat
    backend.body_blocks["body_pos_w"] = storage[:, 3:6]
    backend.body_blocks["body_quat_w"] = storage[:, 7:11]
    plan.refresh()
    pose = plan.ray_pose_view("robot")

    assert not pose.pos_w.is_contiguous()
    assert not pose.quat_w.is_contiguous()
    assert pose.pos_w.data_ptr() == storage[:, 3:6].data_ptr()
    assert pose.quat_w.data_ptr() == storage[:, 7:11].data_ptr()
    assert pose.pos_w.stride() == (12, 3, 1)
    assert pose.quat_w.stride() == (12, 4, 1)
    storage[0, 3] = 42.0
    storage[1, 7] = -1.0
    assert pose.pos_w[0, 0, 0].item() == 42.0
    assert pose.quat_w[1, 0, 0].item() == -1.0
    plan.close()


def test_scene_plan_validates_body_field_requests_and_pose_reads(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = _SceneTensorBackend()
    _use_body_capable_tensor_spec(monkeypatch)
    scene = _scene(backend)

    with pytest.raises(ValueError, match="body_fields are unknown"):
        SceneTensorReadSpec(entity="robot", body_names=("hip",), body_fields=("pos_w", "bogus"))
    with pytest.raises(ValueError, match="body_fields must be unique"):
        SceneTensorReadSpec(entity="robot", body_names=("hip",), body_fields=("pos_w", "pos_w"))
    with pytest.raises(ValueError, match="body_fields cannot be empty"):
        SceneTensorReadSpec(entity="robot", body_names=("hip",), body_fields=())

    plan = scene.compile_tensor_reads(
        "cpu",
        (SceneTensorReadSpec(entity="robot", body_names=("hip",), body_fields=("pos_w",)),),
    )
    with pytest.raises(RuntimeError, match="must be refreshed"):
        plan.ray_pose_view("robot")
    plan.refresh()
    with pytest.raises(ValueError, match=r"fields \['quat_w'\] were not compiled"):
        plan.ray_pose_view("robot")
    plan.close()


def test_scene_plan_rejects_reads_before_refresh_and_after_close() -> None:
    backend = _SceneTensorBackend()
    plan = _scene(backend).compile_tensor_reads("cpu", _specs())

    with pytest.raises(RuntimeError, match="must be refreshed"):
        plan.joint_tensor_view("robot")
    with pytest.raises(RuntimeError, match="must be refreshed"):
        plan.transfer_stats

    plan.refresh()
    plan.invalidate()
    with pytest.raises(RuntimeError, match="must be refreshed"):
        plan.sensor_tensor_views("robot", ("imu_gyro",))

    plan.refresh()
    plan.close()
    with pytest.raises(RuntimeError, match="closed"):
        plan.joint_tensor_view("robot")
    with pytest.raises(RuntimeError, match="closed"):
        plan.refresh()


def test_scene_plan_fails_closed_on_missing_entities_names_and_capabilities() -> None:
    backend = _SceneTensorBackend()
    scene = _scene(backend)

    with pytest.raises(TypeError, match="sequence of SceneTensorReadSpec"):
        scene.compile_tensor_reads("cpu", _specs()[0])  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="request is empty"):
        scene.compile_tensor_reads("cpu", ())
    with pytest.raises(KeyError, match="'missing' not found"):
        scene.compile_tensor_reads(
            "cpu", (SceneTensorReadSpec(entity="missing", sensor_names=("imu_gyro",)),)
        )

    plan = scene.compile_tensor_reads("cpu", _specs())
    plan.refresh()
    with pytest.raises(ValueError, match="were not compiled"):
        plan.sensor_tensor_views("robot", ("missing",))
    with pytest.raises(KeyError, match="was not compiled"):
        plan.joint_tensor_view("other")
    plan.close()

    unsupported = _SceneTensorBackend()
    unsupported.get_tensor_capabilities = MethodType(  # type: ignore[method-assign]
        lambda self: TensorLifecycleCapabilities(execution=TensorExecution.UNSUPPORTED),
        unsupported,
    )
    with pytest.raises(NotImplementedError, match="tensor execution is TensorExecution"):
        _scene(unsupported).compile_tensor_reads("cpu", _specs())

    scattered = _SceneTensorBackend()
    original = scattered.get_tensor_capabilities()
    scattered.get_tensor_capabilities = MethodType(  # type: ignore[method-assign]
        lambda self: TensorLifecycleCapabilities(
            execution=original.execution,
            state_views=True,
            state_fields=original.state_fields,
            sensor_views=True,
            process_topology=original.process_topology,
            data_plane=original.data_plane,
            stream_event_ownership=original.stream_event_ownership,
            torch_devices=original.torch_devices,
        ),
        scattered,
    )
    with pytest.raises(NotImplementedError, match="in-process packed host bridge"):
        _scene(scattered).compile_tensor_reads("cpu", _specs())
