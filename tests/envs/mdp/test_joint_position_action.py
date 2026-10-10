"""Upstream-derived NumPy tests for the manager joint-position action."""

from __future__ import annotations

import ast
import dataclasses
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import numpy as np
import pytest
import torch
from unisim.backend.base import (
    SimBackend,
    TensorDataPlane,
    TensorExecution,
    TensorLifecycleCapabilities,
    TensorProcessTopology,
)

from unilab.assets import ASSETS_ROOT_PATH
from unilab.base.backend_factory import create_backend
from unilab.base.entity import EntityCfg, EntityScene
from unilab.base.scene import SceneCfg
from unilab.envs.mdp import (
    JointEffortAction,
    JointEffortActionCfg,
    JointPositionAction,
    JointPositionActionCfg,
    JointVelocityAction,
    JointVelocityActionCfg,
    RelativeJointPositionAction,
    RelativeJointPositionActionCfg,
)
from unilab.envs.mdp.actions import (
    JointPositionAction as ExportedJointPositionAction,
)
from unilab.envs.mdp.actions import (
    JointPositionActionCfg as ExportedJointPositionActionCfg,
)
from unilab.managers._types import ManagerBasedRlEnv


class _Backend:
    backend_type = "fake"
    num_envs = 2
    num_actuators = 3

    def __init__(self) -> None:
        self.actuator_names = ("knee_motor", "hip_motor", "ankle_motor")
        self.target_joint_names = ("knee", "hip", "ankle")
        self.joint_index = {"hip": 0, "knee": 1, "ankle": 2}
        self.dof_pos = np.zeros((self.num_envs, 3), dtype=np.float32)

    def get_actuator_names(self) -> tuple[str, ...]:
        return self.actuator_names

    def get_actuator_joint_names(self) -> tuple[str, ...]:
        return self.target_joint_names

    def get_actuator_ctrl_range(self) -> np.ndarray:
        return np.tile(np.asarray([[-10.0, 10.0]], dtype=np.float32), (3, 1))

    def get_joint_dof_pos_indices(self, names) -> np.ndarray:
        return np.asarray([self.joint_index[name] for name in names], dtype=np.int32)

    def get_joint_dof_vel_indices(self, names) -> np.ndarray:
        return self.get_joint_dof_pos_indices(names)

    def get_dof_pos(self) -> np.ndarray:
        return self.dof_pos

    def get_dof_vel(self) -> np.ndarray:
        return np.zeros_like(self.dof_pos)

    def get_default_dof_pos(self) -> np.ndarray:
        return np.asarray([0.1, 0.2, 0.3], dtype=np.float32)

    def get_joint_range(self) -> np.ndarray:
        return np.tile(np.asarray([[-1.0, 1.0]], dtype=np.float32), (3, 1))

    def get_joint_state_qpos_indices(self, names) -> np.ndarray:
        return np.asarray([self.joint_index[name] for name in names], dtype=np.int32)

    def get_joint_state_qvel_indices(self, names) -> np.ndarray:
        return self.get_joint_state_qpos_indices(names)

    def get_tensor_capabilities(self) -> TensorLifecycleCapabilities:
        return TensorLifecycleCapabilities(
            execution=TensorExecution.HOST_BRIDGE,
            state_views=True,
            state_fields=frozenset(("qpos", "qvel")),
            stepping=True,
            selected_reset=True,
            host_pre_step_control=True,
            packed_host_bridge=True,
            process_topology=TensorProcessTopology.IN_PROCESS,
            data_plane=TensorDataPlane.HOST_BRIDGE,
            stream_event_ownership="caller owns Torch stream",
            torch_devices=("cpu",),
        )

    def get_state_views(self, fields, device=None) -> dict[str, torch.Tensor]:
        assert set(fields) == {"qpos", "qvel"}
        target = torch.device(device) if device is not None else torch.device("cpu")
        return {
            "qpos": torch.from_numpy(self.dof_pos.copy()).to(target),
            "qvel": torch.zeros(self.dof_pos.shape, dtype=torch.float32, device=target),
        }


def _action(
    **overrides,
) -> tuple[JointPositionAction, np.ndarray, EntityScene]:
    return _build_action(JointPositionActionCfg, **overrides)


def _build_action(action_cfg_type, **overrides):
    backend = _Backend()
    control = torch.zeros((backend.num_envs, backend.num_actuators), dtype=torch.float32)
    scene = EntityScene(
        {
            "robot": EntityCfg(
                joint_names=("hip", "knee", "ankle"),
                actuator_names=backend.actuator_names,
            )
        },
        cast(SimBackend, backend),
        control,
    )
    cfg_values = {
        "entity_name": "robot",
        "actuator_names": ("hip|knee",),
        **overrides,
    }
    cfg = action_cfg_type(**cfg_values)
    env = cast(ManagerBasedRlEnv, SimpleNamespace(num_envs=backend.num_envs, scene=scene))
    return cfg.build(env), control, scene


def test_public_exports_are_canonical_objects() -> None:
    assert JointPositionAction is ExportedJointPositionAction
    assert JointPositionActionCfg is ExportedJointPositionActionCfg


def test_default_offset_encoder_bias_and_control_order() -> None:
    action, control, scene = _action(scale=2.0)
    raw = np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    scene["robot"].data.encoder_bias[:, 0] = np.asarray([0.05, 0.1])

    assert isinstance(action.raw_action, torch.Tensor)
    assert isinstance(action.processed_action, torch.Tensor)
    assert action.raw_action.dtype == torch.float32
    assert action.processed_action.dtype == torch.float32
    action.process_actions(torch.from_numpy(raw))
    action.apply_actions()

    assert action.target_names == ["hip", "knee"]
    np.testing.assert_array_equal(action.target_ids, [0, 1])
    torch.testing.assert_close(
        action.processed_action, torch.from_numpy(raw) * 2.0 + torch.tensor([0.1, 0.2])
    )
    np.testing.assert_allclose(
        control[:, 1],
        (action.processed_action[:, 0] - torch.tensor([0.05, 0.1])).numpy(),
    )
    np.testing.assert_allclose(control[:, 0], action.processed_action[:, 1].numpy())
    np.testing.assert_array_equal(control[:, 2], 0.0)


def test_tensor_control_writes_stay_on_control_plane() -> None:
    action, control, scene = _action()
    raw = np.asarray([[0.25, -0.5], [1.0, 1.5]], dtype=np.float32)
    assert isinstance(control, torch.Tensor)
    scene["robot"].data.encoder_bias[:, 0] = np.asarray([0.1, -0.1])

    action.process_actions(torch.from_numpy(raw))
    action.apply_actions()

    expected_processed = np.asarray(
        [[0.25 + 0.1, -0.5 + 0.2], [1.0 + 0.1, 1.5 + 0.2]], dtype=np.float32
    )
    np.testing.assert_allclose(control[:, 1].cpu().numpy(), expected_processed[:, 0] - [0.1, -0.1])
    np.testing.assert_allclose(control[:, 0].cpu().numpy(), expected_processed[:, 1])
    np.testing.assert_array_equal(control[:, 2].cpu().numpy(), 0.0)


def test_regex_scale_offset_clip_and_local_reset() -> None:
    action, _, _ = _action(
        scale={"hip": 2.0, "knee": 3.0},
        offset={"hip": 0.5, "knee": -0.5},
        clip={"hip": (-1.0, 1.0)},
        use_default_offset=False,
    )
    raw = np.asarray([[2.0, 2.0], [-2.0, -2.0]], dtype=np.float32)

    action.process_actions(torch.from_numpy(raw))

    np.testing.assert_allclose(action.processed_action, [[1.0, 5.5], [-1.0, -6.5]])
    np.testing.assert_array_equal(action.raw_action, raw)
    action.reset(torch.tensor([1], dtype=torch.int64))
    np.testing.assert_array_equal(action.raw_action[0], raw[0])
    np.testing.assert_array_equal(action.raw_action[1], 0.0)


def test_base_action_accepts_noncontiguous_term_slice() -> None:
    action, _, _ = _action(scale=2.0)
    sliced_action = torch.zeros((2, 4), dtype=torch.float32)[:, ::2]
    assert not sliced_action.is_contiguous()

    action.process_actions(sliced_action)

    torch.testing.assert_close(action.raw_action, sliced_action)
    torch.testing.assert_close(
        action.processed_action,
        sliced_action * 2.0 + torch.tensor([0.1, 0.2]),
    )


@pytest.mark.parametrize(
    ("overrides", "error", "message"),
    [
        ({"scale": {"missing": 1.0}}, ValueError, "match no targets"),
        (
            {"scale": {"hip|knee": 1.0, ".*": 2.0}},
            ValueError,
            "both match target",
        ),
        ({"clip": {"hip": (1.0, -1.0)}}, ValueError, "exceeds upper"),
        ({"offset": float("nan")}, ValueError, "must be finite"),
        ({"use_default_offset": 1}, TypeError, "must be bool"),
    ],
)
def test_invalid_action_config_fails_at_construction(overrides, error, message) -> None:
    with pytest.raises(error, match=message):
        _action(**overrides)


def test_non_finite_and_wrong_shape_actions_fail_before_control_write() -> None:
    action, control, _ = _action()
    with pytest.raises(ValueError, match="expected action shape"):
        action.process_actions(torch.zeros((2, 1), dtype=torch.float32))
    with pytest.raises(ValueError, match="NaN or Inf"):
        action.process_actions(torch.full((2, 2), torch.nan, dtype=torch.float32))
    np.testing.assert_array_equal(control, 0.0)


@pytest.mark.parametrize(
    ("action_cfg_type", "action_type"),
    [
        (JointVelocityActionCfg, JointVelocityAction),
        (JointEffortActionCfg, JointEffortAction),
    ],
)
def test_velocity_and_effort_actions_use_the_shared_joint_control_mapping(
    action_cfg_type, action_type
) -> None:
    action, control, _ = _build_action(action_cfg_type, scale=2.0)
    assert isinstance(action, action_type)
    raw = np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)

    action.process_actions(torch.from_numpy(raw))
    action.apply_actions()

    np.testing.assert_allclose(control[:, 1], raw[:, 0] * 2.0)
    np.testing.assert_allclose(control[:, 0], raw[:, 1] * 2.0)
    np.testing.assert_array_equal(control[:, 2], 0.0)


def test_tensor_motion_control_stays_on_control_plane_with_default_bias() -> None:
    from types import SimpleNamespace

    from unilab.tasks.motion_tracking.common.manager_terms import (
        MotionCommand,
        MotionJointPositionAction,
        MotionJointPositionActionCfg,
    )

    backend = _Backend()
    control = torch.zeros((backend.num_envs, backend.num_actuators), dtype=torch.float32)
    scene = EntityScene(
        {
            "robot": EntityCfg(
                joint_names=("hip", "knee", "ankle"),
                actuator_names=backend.actuator_names,
            )
        },
        cast(SimBackend, backend),
        control,
    )
    command = MotionCommand.__new__(MotionCommand)
    command.joint_default_bias = torch.tensor(
        [[0.2, -0.1, 0.0], [-0.2, 0.1, 0.0]], dtype=torch.float32
    )
    env = cast(
        ManagerBasedRlEnv,
        SimpleNamespace(
            num_envs=backend.num_envs,
            scene=scene,
            command_manager=SimpleNamespace(get_term=lambda name: command),
        ),
    )
    action = MotionJointPositionActionCfg(
        entity_name="robot",
        actuator_names=("hip|knee",),
        command_name="motion",
        scale=1.0,
        use_default_offset=False,
    ).build(env)
    scene["robot"].data.encoder_bias[:, 0] = np.asarray([0.05, -0.05])

    assert isinstance(action, MotionJointPositionAction)
    action.process_actions(torch.tensor([[0.25, -0.5], [1.0, 1.5]], dtype=torch.float32))
    action.apply_actions()

    expected_hip = torch.tensor([0.25 + 0.2 - 0.05, 1.0 - 0.2 + 0.05])
    expected_knee = torch.tensor([-0.5 - 0.1, 1.5 + 0.1])
    torch.testing.assert_close(control[:, 1], expected_hip)
    torch.testing.assert_close(control[:, 0], expected_knee)
    torch.testing.assert_close(control[:, 2], torch.zeros_like(control[:, 2]))


def test_relative_joint_position_action_reads_current_position_at_apply_time() -> None:
    action, control, scene = _build_action(RelativeJointPositionActionCfg, scale=0.25)
    assert isinstance(action, RelativeJointPositionAction)
    scene["robot"]._backend.dof_pos[:] = np.asarray(
        [[0.4, 0.5, 0.6], [0.7, 0.8, 0.9]], dtype=np.float32
    )
    raw = np.asarray([[0.4, -0.8], [0.8, -0.4]], dtype=np.float32)

    action.process_actions(torch.from_numpy(raw))
    action.apply_actions()

    # Entity joint order is hip, knee, ankle while backend actuator order is
    # knee, hip, ankle.
    np.testing.assert_allclose(control[:, 1], [0.4 + 0.4 * 0.25, 0.7 + 0.8 * 0.25])
    np.testing.assert_allclose(control[:, 0], [0.5 - 0.8 * 0.25, 0.8 - 0.4 * 0.25])


def test_relative_joint_position_action_rejects_nonzero_offsets() -> None:
    with pytest.raises(ValueError, match="does not support a non-zero offset"):
        _build_action(RelativeJointPositionActionCfg, offset={"hip": 0.1})


def test_entity_joint_tensor_view_fails_closed_on_missing_tensor_capability() -> None:
    class UnsupportedBackend(_Backend):
        def get_tensor_capabilities(self) -> TensorLifecycleCapabilities:
            return TensorLifecycleCapabilities(execution=TensorExecution.UNSUPPORTED)

    backend = UnsupportedBackend()
    scene = EntityScene(
        {"robot": EntityCfg(joint_names=("hip",), actuator_names=("hip_motor",))},
        cast(SimBackend, backend),
    )

    with pytest.raises(NotImplementedError, match="tensor execution is TensorExecution"):
        scene["robot"].joint_tensor_view(torch.device("cpu"))


def test_entity_joint_tensor_view_validates_layout_dtype_device_and_finite_values() -> None:
    class InvalidLayoutBackend(_Backend):
        def get_joint_state_qpos_indices(self, names) -> np.ndarray:
            return np.asarray([99 for _ in names], dtype=np.int32)

    backend = InvalidLayoutBackend()
    scene = EntityScene(
        {"robot": EntityCfg(joint_names=("hip",), actuator_names=("hip_motor",))},
        cast(SimBackend, backend),
    )

    with pytest.raises(ValueError, match="columns exceed backend width"):
        scene["robot"].joint_tensor_view(torch.device("cpu"))

    backend.get_state_views = lambda fields, device=None: {
        "qpos": torch.full((2, 1), torch.nan),
        "qvel": torch.zeros((2, 1)),
    }
    backend.get_joint_state_qpos_indices = super(
        InvalidLayoutBackend, backend
    ).get_joint_state_qpos_indices
    with pytest.raises(ValueError, match="tensor joint_pos has NaN or Inf"):
        scene["robot"].joint_tensor_view(torch.device("cpu"))

    backend.get_state_views = lambda fields, device=None: {
        "qpos": torch.zeros((2, 1), dtype=torch.float64),
        "qvel": torch.zeros((2, 1)),
    }
    with pytest.raises(TypeError, match="must be float32"):
        scene["robot"].joint_tensor_view(torch.device("cpu"))


class _SensorTensorBackend(_Backend):
    root_body_name = None

    def __init__(self) -> None:
        super().__init__()
        self.sensor_requests: list[tuple[str, torch.device]] = []
        self._sensors = {
            "imu_gyro": torch.randn((self.num_envs, 3), dtype=torch.float32),
            "track_pos_w_hip": torch.randn((self.num_envs, 3), dtype=torch.float32),
            "track_quat_w_hip": torch.randn((self.num_envs, 4), dtype=torch.float32),
            "track_linvel_w_hip": torch.randn((self.num_envs, 3), dtype=torch.float32),
            "track_angvel_w_hip": torch.randn((self.num_envs, 3), dtype=torch.float32),
        }

    def get_tensor_capabilities(self) -> TensorLifecycleCapabilities:
        capabilities = super().get_tensor_capabilities()
        if self._device_resident:
            capabilities = dataclasses.replace(
                capabilities,
                execution=TensorExecution.DEVICE_RESIDENT,
                data_plane=TensorDataPlane.DIRECT,
                packed_host_bridge=False,
            )
        return dataclasses.replace(capabilities, sensor_views=True)

    def as_device_resident(self) -> None:
        self._device_resident = True

    _device_resident = False

    def tensor_execution(self) -> TensorExecution:
        if self._device_resident:
            return TensorExecution.DEVICE_RESIDENT
        return super().tensor_execution()

    def get_sensor_view(self, name: str, device: str | torch.device = "cpu") -> torch.Tensor:
        self.sensor_requests.append((name, torch.device(device)))
        try:
            return self._sensors[name]
        except KeyError as exc:
            raise KeyError(f"unknown sensor {name!r}") from exc

    def get_body_ids(self, names) -> np.ndarray:
        assert tuple(names) == ("hip",)
        return np.asarray([0], dtype=np.int32)

    def get_body_pos_w(self, body_ids) -> np.ndarray:
        return np.zeros((self.num_envs, len(body_ids), 3), dtype=np.float32)

    def get_body_quat_w(self, body_ids) -> np.ndarray:
        quaternion = np.zeros((len(body_ids), 4), dtype=np.float32)
        quaternion[:, 0] = 1.0
        return np.broadcast_to(quaternion, (self.num_envs, len(body_ids), 4)).copy()

    def get_body_lin_vel_w(self, body_ids) -> np.ndarray:
        return np.zeros((self.num_envs, len(body_ids), 3), dtype=np.float32)

    def get_body_ang_vel_w(self, body_ids) -> np.ndarray:
        return np.zeros((self.num_envs, len(body_ids), 3), dtype=np.float32)

    @classmethod
    def create(cls) -> "_SensorTensorBackend":
        return cls()


def test_entity_sensor_and_body_tensor_views_validate_contract() -> None:
    backend = _SensorTensorBackend.create()
    backend.as_device_resident()
    scene = EntityScene(
        {
            "robot": EntityCfg(
                joint_names=("hip",), body_names=("hip",), actuator_names=("hip_motor",)
            )
        },
        cast(SimBackend, backend),
    )

    sensors = scene["robot"].sensor_tensor_views(torch.device("cpu"), ("imu_gyro",))
    assert sensors.names == ("imu_gyro",)
    torch.testing.assert_close(sensors.values["imu_gyro"], backend._sensors["imu_gyro"])

    body = scene["robot"].body_tensor_view(torch.device("cpu"))
    assert body.body_names == ("hip",)
    assert body.pos_w.shape == (backend.num_envs, 1, 3)
    assert body.quat_w.shape == (backend.num_envs, 1, 4)
    assert body.lin_vel_w.shape == (backend.num_envs, 1, 3)
    assert body.ang_vel_w.shape == (backend.num_envs, 1, 3)
    torch.testing.assert_close(body.pos_w[:, 0], backend._sensors["track_pos_w_hip"])
    torch.testing.assert_close(body.quat_w[:, 0], backend._sensors["track_quat_w_hip"])
    torch.testing.assert_close(body.lin_vel_w[:, 0], backend._sensors["track_linvel_w_hip"])
    torch.testing.assert_close(body.ang_vel_w[:, 0], backend._sensors["track_angvel_w_hip"])
    assert all(device.type == "cpu" for _, device in backend.sensor_requests)


def test_entity_body_tensor_view_fails_closed_on_host_bridge() -> None:
    backend = _SensorTensorBackend.create()
    scene = EntityScene(
        {
            "robot": EntityCfg(
                joint_names=("hip",), body_names=("hip",), actuator_names=("hip_motor",)
            )
        },
        cast(SimBackend, backend),
    )

    with pytest.raises(NotImplementedError, match="scene-owned packed host-bridge plan"):
        scene["robot"].body_tensor_view("cpu")


def test_entity_sensor_tensor_view_fails_closed_on_multi_name_host_bridge() -> None:
    backend = _SensorTensorBackend.create()
    scene = EntityScene(
        {"robot": EntityCfg(joint_names=("hip",), actuator_names=("hip_motor",))},
        cast(SimBackend, backend),
    )

    with pytest.raises(NotImplementedError, match="scene-owned packed host-bridge plan"):
        scene["robot"].sensor_tensor_views("cpu", ("imu_gyro", "imu_accel"))


def test_entity_sensor_and_body_tensor_views_fail_closed() -> None:
    backend = _SensorTensorBackend.create()
    scene = EntityScene(
        {
            "robot": EntityCfg(
                joint_names=("hip",), body_names=("hip",), actuator_names=("hip_motor",)
            )
        },
        cast(SimBackend, backend),
    )

    with pytest.raises(TypeError, match="must be a sequence of strings"):
        scene["robot"].sensor_tensor_views(torch.device("cpu"), "imu_gyro")
    with pytest.raises(ValueError, match="must be unique"):
        scene["robot"].sensor_tensor_views(torch.device("cpu"), ("imu_gyro", "imu_gyro"))
    with pytest.raises(KeyError, match="tensor sensor"):
        scene["robot"].sensor_tensor_views(torch.device("cpu"), ("missing",))
    with pytest.raises(ValueError, match="are not declared"):
        scene["robot"].body_tensor_view(torch.device("cpu"), ("missing",))

    backend.get_tensor_capabilities = lambda: TensorLifecycleCapabilities(
        execution=TensorExecution.UNSUPPORTED
    )
    with pytest.raises(
        NotImplementedError, match="named sensor views are unavailable|tensor execution"
    ):
        scene["robot"].sensor_tensor_views(torch.device("cpu"), ("imu_gyro",))


def test_entity_body_tensor_view_validates_width_dtype_device_and_finite_values() -> None:
    backend = _SensorTensorBackend.create()
    backend.as_device_resident()
    scene = EntityScene(
        {
            "robot": EntityCfg(
                joint_names=("hip",), body_names=("hip",), actuator_names=("hip_motor",)
            )
        },
        cast(SimBackend, backend),
    )
    original = backend.get_sensor_view
    backend.get_sensor_view = lambda name, device=None: torch.zeros(
        (backend.num_envs, 2), dtype=torch.float32
    )
    with pytest.raises(ValueError, match="expected \\(2, 3\\)"):
        scene["robot"].body_tensor_view(torch.device("cpu"))
    backend.get_sensor_view = lambda name, device=None: torch.full(
        (backend.num_envs, 3), torch.nan, dtype=torch.float32
    )
    with pytest.raises(ValueError, match="has NaN or Inf"):
        scene["robot"].body_tensor_view(torch.device("cpu"))
    backend.get_sensor_view = lambda name, device=None: torch.zeros(
        (backend.num_envs, 3), dtype=torch.float64
    )
    with pytest.raises(TypeError, match="must be float32"):
        scene["robot"].body_tensor_view(torch.device("cpu"))
    backend.get_sensor_view = original


@pytest.mark.parametrize("backend_type", ["mujoco", "motrix"])
def test_go2_joint_targets_are_mapped_to_backend_control_order(backend_type: str) -> None:
    if backend_type == "motrix":
        pytest.importorskip("motrixsim")
    else:
        pytest.importorskip(
            "unisim.backend.mujoco.backend",
            reason="unisim-core MuJoCo adapter (mjbatch build) not available",
        )
    joint_names = (
        "FL_hip_joint",
        "FL_thigh_joint",
        "FL_calf_joint",
        "FR_hip_joint",
        "FR_thigh_joint",
        "FR_calf_joint",
        "RL_hip_joint",
        "RL_thigh_joint",
        "RL_calf_joint",
        "RR_hip_joint",
        "RR_thigh_joint",
        "RR_calf_joint",
    )
    scene_cfg = SceneCfg(
        model_file=str(ASSETS_ROOT_PATH / "robots" / "go2" / "scene_flat.xml"),
        entities={
            "robot": EntityCfg(
                joint_names=joint_names,
                actuator_names=(
                    "FR_hip",
                    "FR_thigh",
                    "FR_calf",
                    "FL_hip",
                    "FL_thigh",
                    "FL_calf",
                    "RR_hip",
                    "RR_thigh",
                    "RR_calf",
                    "RL_hip",
                    "RL_thigh",
                    "RL_calf",
                ),
            )
        },
    )
    backend = create_backend(
        backend_type,
        scene_cfg,
        2,
        0.01,
        base_name="base",
    )
    control = torch.zeros((2, backend.num_actuators), dtype=torch.float32)
    scene = EntityScene.from_scene_cfg(scene_cfg, backend, control)
    env = cast(ManagerBasedRlEnv, SimpleNamespace(num_envs=2, scene=scene))
    action = JointPositionActionCfg(
        entity_name="robot",
        actuator_names=(".*",),
        scale=0.25,
        offset={".*_hip_joint": 0.1, ".*_thigh_joint": 0.2, ".*_calf_joint": -0.3},
        use_default_offset=False,
    ).build(env)
    raw = np.arange(24, dtype=np.float32).reshape(2, 12) / 10.0

    action.process_actions(torch.from_numpy(raw))
    action.apply_actions()

    target_index = {name: index for index, name in enumerate(joint_names)}
    expected = np.column_stack(
        [
            action.processed_action[:, target_index[name]]
            for name in backend.get_actuator_joint_names()
        ]
    )
    torch.testing.assert_close(control, torch.from_numpy(expected))


def test_action_module_has_no_training_or_backend_private_dependencies() -> None:
    path = (
        Path(__file__).resolve().parents[3]
        / "src"
        / "unilab"
        / "envs"
        / "mdp"
        / "actions"
        / "actions.py"
    )
    tree = ast.parse(path.read_text(encoding="utf-8"))
    forbidden = ("uni_rl", "unilab.training", "unilab.base.backend")
    imports = [node.module or "" for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)] + [
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    ]
    assert not [name for name in imports if name.startswith(forbidden)]
