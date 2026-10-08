"""Tensor-native motion anchor terms consume the declared scene body phase."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest
import torch

from unilab.base.entity import EntityTensorBodyStateView
from unilab.managers import ManagerTermBaseCfg
from unilab.managers._types import ManagerBasedRlEnv
from unilab.tasks.motion_tracking.common import manager_terms as mt
from unilab.tasks.motion_tracking.common import obs_terms
from unilab.tasks.motion_tracking.common.manager_terms import MotionCommand
from unilab.tasks.motion_tracking.common.tensor_rotation import quat_to_rot6
from unilab.utils.rotation import np_matrix_first_two_cols_from_quat, np_matrix_from_quat


class _MotionCommand(MotionCommand):
    """MotionCommand-shaped fake with only this term's cold/hot dependencies."""

    cfg = None

    def __init__(self, **values: Any) -> None:
        self.__dict__.update(values)

    @property
    def tensor_carrier(self) -> bool:
        return bool(self.__dict__.get("tensor_carrier", False))


class _FakeCommandManager:
    def __init__(self, command: Any) -> None:
        self._command = command

    def get_term(self, name: str) -> Any:
        if name != "motion":
            raise KeyError(name)
        return self._command


class _MissingCommandManager:
    def get_term(self, name: str) -> Any:
        raise KeyError(name)


def _env(command: Any, view: EntityTensorBodyStateView) -> Any:
    class _Scene(dict):
        pass

    scene = _Scene()
    scene._tensor_read_plan = SimpleNamespace(
        body_tensor_view=lambda *_: view,
        sensor_tensor_views=lambda *_: SimpleNamespace(
            values={
                "pelvis_local_linvel": torch.tensor(
                    [[0.2, -0.3, 0.4], [-0.2, 0.3, -0.4]], dtype=torch.float32
                ),
                "torso_gyro": torch.tensor(
                    [[1.0, -2.0, 3.0], [-1.0, 2.0, -3.0]], dtype=torch.float32
                ),
            }
        ),
    )
    return SimpleNamespace(
        num_envs=2,
        device=torch.device("cpu"),
        scene=scene,
        command_manager=_FakeCommandManager(command),
    )


def _command(
    *,
    body_pos_w: np.ndarray,
    body_quat_w: np.ndarray,
) -> Any:
    return _MotionCommand(
        cfg=_MotionCommand(
            entity_name="robot",
            body_names=("torso", "pelvis"),
        ),
        anchor_body_idx=1,
        _body_pos_w=body_pos_w,
        _motion_data=SimpleNamespace(body_quat_w=body_quat_w),
    )


def _view() -> EntityTensorBodyStateView:
    pos = torch.zeros((2, 2, 3), dtype=torch.float32)
    pos[:, 1] = torch.tensor([[0.2, -0.1, 0.4], [-0.2, 0.1, -0.4]], dtype=torch.float32)
    quat = torch.zeros((2, 2, 4), dtype=torch.float32)
    quat[:, 0] = 1.0
    return EntityTensorBodyStateView(
        body_names=("torso", "pelvis"),
        pos_w=pos,
        quat_w=quat,
        lin_vel_w=torch.zeros((2, 2, 3)),
        ang_vel_w=torch.zeros((2, 2, 3)),
    )


def test_quat_to_rot6_matches_numpy_first_two_columns() -> None:
    rng = np.random.default_rng(1818)
    quat = _unit_quat(rng.standard_normal((17, 4), dtype=np.float32))
    expected = np_matrix_first_two_cols_from_quat(quat)

    actual = quat_to_rot6(torch.as_tensor(quat))

    torch.testing.assert_close(actual, torch.as_tensor(expected))


def test_mimiclite_rot6_helper_matches_orientation_rows() -> None:
    rng = np.random.default_rng(1820)
    quat = _unit_quat(rng.standard_normal((32, 4), dtype=np.float32))
    matrix = np_matrix_from_quat(quat)

    actual = obs_terms._quat_to_rot6_rows(torch.as_tensor(quat)).numpy()

    np.testing.assert_allclose(actual[:, :3], matrix[:, 0, :], atol=1e-6)
    np.testing.assert_allclose(actual[:, 3:], matrix[:, 1, :], atol=1e-6)


def test_quat_to_rot6_rejects_invalid_width() -> None:
    with pytest.raises(ValueError, match="quaternion must have final dimension 4"):
        quat_to_rot6(torch.zeros((2, 3)))


def test_obs_motion_feature_layout_interleaves_per_body_state() -> None:
    frames, bodies = 7, 3
    rng = np.random.default_rng(1821)
    loader = SimpleNamespace(
        num_frames=frames,
        num_bodies=bodies,
        body_pos_w=rng.normal(size=(frames, bodies, 3)).astype(np.float32),
        body_quat_w=_unit_quat(rng.normal(size=(frames, bodies, 4)).astype(np.float32)),
        body_lin_vel_w=rng.normal(size=(frames, bodies, 3)).astype(np.float32),
        body_ang_vel_w=rng.normal(size=(frames, bodies, 3)).astype(np.float32),
    )

    table = mt.TensorMotionCommand._make_obs_motion_features(loader, torch.device("cpu"))
    row = table[4].view(bodies, 13)

    assert table.shape == (frames, bodies * 13)
    np.testing.assert_array_equal(row[:, :3].numpy(), loader.body_pos_w[4])
    np.testing.assert_array_equal(row[:, 3:7].numpy(), loader.body_quat_w[4])
    np.testing.assert_array_equal(row[:, 7:10].numpy(), loader.body_lin_vel_w[4])
    np.testing.assert_array_equal(row[:, 10:].numpy(), loader.body_ang_vel_w[4])


def _cfg(term_class: type) -> ManagerTermBaseCfg:
    return ManagerTermBaseCfg(func=term_class, params={"command_name": "motion"})


def _reference_state() -> tuple[np.ndarray, np.ndarray]:
    motion_yaw = np.asarray([1.1, -0.3], dtype=np.float32)
    body_quat = np.zeros((2, 2, 4), dtype=np.float32)
    body_quat[:, :, 0] = np.cos(0.5 * motion_yaw)[:, None]
    body_quat[:, :, 3] = np.sin(0.5 * motion_yaw)[:, None]
    body_pos = np.asarray(
        [
            [[0.1, 0.2, 0.3], [0.9, -0.4, 1.1]],
            [[-0.1, -0.2, -0.3], [-0.6, 0.8, -1.2]],
        ],
        dtype=np.float32,
    )
    return body_pos, body_quat


def _unit_quat(value: np.ndarray) -> np.ndarray:
    value /= np.linalg.norm(value, axis=-1, keepdims=True)
    return value


def test_anchor_position_uses_reference_and_aggregate_body_view() -> None:
    body_pos_w, _ = _reference_state()
    command = _command(body_pos_w=body_pos_w.copy(), body_quat_w=np.zeros((2, 2, 4)))
    env = _env(command, _view())
    view = env.scene._tensor_read_plan.body_tensor_view(None, None)
    term = mt.MotionAnchorPositionObservation(
        _cfg(mt.MotionAnchorPositionObservation), cast(ManagerBasedRlEnv, env)
    )

    value = term(cast(ManagerBasedRlEnv, env), command_name="motion")

    robot_quat = torch.zeros((2, 4), dtype=torch.float32)
    robot_quat[:, 0] = 1.0
    expected_position = torch.as_tensor(body_pos_w[:, 1]) - view.pos_w[:, 1]
    expected = mt.quat_apply_inverse(robot_quat, expected_position)
    assert isinstance(value, torch.Tensor)
    torch.testing.assert_close(value, expected)
    np.testing.assert_array_equal(command._body_pos_w, body_pos_w)


def test_anchor_orientation_uses_reference_and_aggregate_body_view() -> None:
    _, body_quat_w = _reference_state()
    command = _command(body_pos_w=np.zeros((2, 2, 3)), body_quat_w=body_quat_w.copy())
    env = _env(command, _view())
    view = env.scene._tensor_read_plan.body_tensor_view(None, None)
    term = mt.MotionAnchorOrientationObservation(_cfg(mt.MotionAnchorOrientationObservation), env)

    value = term(cast(ManagerBasedRlEnv, env), command_name="motion")

    robot_quat = view.quat_w[:, 1]
    expected_quat = mt.quat_mul(
        mt.quat_conjugate(robot_quat),
        torch.as_tensor(body_quat_w[:, 1]),
    )
    expected = quat_to_rot6(expected_quat)
    assert isinstance(value, torch.Tensor)
    torch.testing.assert_close(value, expected)
    np.testing.assert_array_equal(command._motion_data.body_quat_w, body_quat_w)


def test_anchor_terms_return_default_shape_probe_without_read_plan() -> None:
    command = _command(
        body_pos_w=np.zeros((2, 2, 3), dtype=np.float32),
        body_quat_w=np.zeros((2, 2, 4), dtype=np.float32),
    )
    env = _env(command, _view())
    env.scene._tensor_read_plan = None
    position = mt.MotionAnchorPositionObservation(_cfg(mt.MotionAnchorPositionObservation), env)
    orientation = mt.MotionAnchorOrientationObservation(
        _cfg(mt.MotionAnchorOrientationObservation), env
    )

    assert position(cast(ManagerBasedRlEnv, env), command_name="motion").shape == (2, 3)
    assert orientation(cast(ManagerBasedRlEnv, env), command_name="motion").shape == (2, 6)


def test_anchor_terms_validate_command_binding_and_type() -> None:
    body_pos_w, body_quat_w = _reference_state()
    command = _command(body_pos_w=body_pos_w, body_quat_w=body_quat_w)
    env = _env(command, _view())
    cfg = _cfg(mt.MotionAnchorPositionObservation)
    term = mt.MotionAnchorPositionObservation(cfg, env)

    with pytest.raises(ValueError, match="was bound to 'motion'"):
        term(cast(ManagerBasedRlEnv, env), command_name="other")

    broken = _env(object(), _view())
    with pytest.raises(TypeError, match="expected MotionCommand"):
        mt.MotionAnchorPositionObservation(cfg, cast(ManagerBasedRlEnv, broken))

    missing = _env(command, _view())
    missing.command_manager = _MissingCommandManager()
    with pytest.raises(KeyError, match="not found"):
        mt.MotionAnchorPositionObservation(cfg, cast(ManagerBasedRlEnv, missing))


def test_anchor_terms_declare_command_body_namespace() -> None:
    body_pos_w, body_quat_w = _reference_state()
    command = _command(body_pos_w=body_pos_w, body_quat_w=body_quat_w)
    env = _env(command, _view())
    term = mt.MotionAnchorPositionObservation(_cfg(mt.MotionAnchorPositionObservation), env)

    assert term.tensor_body_names == ("torso", "pelvis")
    assert term.entity_name == "robot"


def test_tensor_command_observation_accessors_return_carrier_views() -> None:
    command = _MotionCommand(tensor_carrier=True)
    command.motion_anchor_pos_b = torch.zeros((2, 3), dtype=torch.float32)
    command.motion_anchor_ori_b = torch.zeros((2, 6), dtype=torch.float32)
    command.robot_body_pos_b = torch.zeros((2, 2, 3), dtype=torch.float32)
    command.robot_body_ori_b = torch.zeros((2, 2, 6), dtype=torch.float32)
    env = _env(command, _view())

    torch.testing.assert_close(
        mt.motion_anchor_pos_b(cast(ManagerBasedRlEnv, env), "motion"),
        command.motion_anchor_pos_b,
    )
    torch.testing.assert_close(
        mt.motion_anchor_ori_b(cast(ManagerBasedRlEnv, env), "motion"),
        command.motion_anchor_ori_b,
    )
    torch.testing.assert_close(
        mt.robot_body_pos_b(cast(ManagerBasedRlEnv, env), "motion"),
        command.robot_body_pos_b.reshape(2, -1),
    )
    torch.testing.assert_close(
        mt.robot_body_ori_b(cast(ManagerBasedRlEnv, env), "motion"),
        command.robot_body_ori_b.reshape(2, -1),
    )


def test_tensor_command_joint_observation_uses_cached_default() -> None:
    robot = SimpleNamespace(
        data=SimpleNamespace(
            default_joint_pos_torch=lambda device: torch.full((2, 29), 0.25),
        )
    )
    command = _MotionCommand(
        tensor_carrier=True,
        robot=robot,
        device_robot_joint_pos=torch.full((2, 29), 1.0),
        joint_default_bias=torch.full((2, 29), 0.125),
    )
    env = _env(command, _view())

    value = mt.motion_joint_pos_rel(cast(ManagerBasedRlEnv, env), "motion")

    assert isinstance(value, torch.Tensor)
    torch.testing.assert_close(value, torch.full((2, 29), 0.625))
