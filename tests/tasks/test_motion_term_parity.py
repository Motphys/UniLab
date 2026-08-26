"""Numerical-contract tests for optimized motion-tracking manager terms."""

from __future__ import annotations

import os
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from numba import config, get_num_threads, threading_layer

from unilab.managers import RewardTermCfg, TerminationTermCfg
from unilab.tasks.motion_tracking.common import kernels
from unilab.tasks.motion_tracking.common import manager_terms as mt
from unilab.utils.rotation import (
    np_matrix_first_two_cols_from_quat,
    np_quat_apply_batched,
    np_quat_apply_inverse_batched,
    np_quat_error_magnitude_squared_batched,
    np_quat_inv,
    np_quat_mul_batched,
    np_yaw_quat,
)


def _make_env(command: Any) -> SimpleNamespace:
    return SimpleNamespace(num_envs=command.num_envs)


def _unit_quat(value: np.ndarray) -> np.ndarray:
    value /= np.linalg.norm(value, axis=-1, keepdims=True)
    return value


@pytest.fixture
def body_setup(monkeypatch: pytest.MonkeyPatch):
    rng = np.random.default_rng(42)
    num_envs, num_bodies = 257, 12
    anchor_body_idx = 4

    body_pos_relative_w = rng.standard_normal((num_envs, num_bodies, 3), dtype=np.float32)
    body_pos_w = body_pos_relative_w.copy()
    robot_body_pos_w = body_pos_relative_w + 0.1 * rng.standard_normal(
        body_pos_relative_w.shape, dtype=np.float32
    )
    body_quat_relative_w = _unit_quat(
        rng.standard_normal((num_envs, num_bodies, 4), dtype=np.float32)
    )
    robot_body_quat_w = _unit_quat(
        body_quat_relative_w
        + 0.05 * rng.standard_normal(body_quat_relative_w.shape, dtype=np.float32)
    )
    body_lin_vel_w = rng.standard_normal((num_envs, num_bodies, 3), dtype=np.float32)
    robot_body_lin_vel_w = body_lin_vel_w + 0.2 * rng.standard_normal(
        body_lin_vel_w.shape, dtype=np.float32
    )
    body_ang_vel_w = rng.standard_normal((num_envs, num_bodies, 3), dtype=np.float32)
    robot_body_ang_vel_w = body_ang_vel_w + 0.5 * rng.standard_normal(
        body_ang_vel_w.shape, dtype=np.float32
    )

    command = SimpleNamespace(
        num_envs=num_envs,
        cfg=SimpleNamespace(body_names=tuple(f"b{i}" for i in range(num_bodies))),
        anchor_body_idx=anchor_body_idx,
        body_pos_w=body_pos_w,
        robot_body_pos_w=robot_body_pos_w,
        anchor_pos_w=body_pos_w[:, anchor_body_idx],
        robot_anchor_pos_w=robot_body_pos_w[:, anchor_body_idx],
        joint_pos=rng.standard_normal((num_envs, 29), dtype=np.float32),
        robot_joint_pos=rng.standard_normal((num_envs, 29), dtype=np.float32),
        body_pos_relative_w=body_pos_relative_w,
        body_quat_relative_w=body_quat_relative_w,
        robot_body_quat_w=robot_body_quat_w,
        body_lin_vel_w=body_lin_vel_w,
        robot_body_lin_vel_w=robot_body_lin_vel_w,
        body_ang_vel_w=body_ang_vel_w,
        robot_body_ang_vel_w=robot_body_ang_vel_w,
    )

    snapshots = {
        key: value.copy() for key, value in vars(command).items() if isinstance(value, np.ndarray)
    }
    monkeypatch.setattr(mt, "_command", lambda env, name: command)
    return command, _make_env(command), snapshots


def _reward_cfg(*, body_names: tuple[str, ...] | None = None) -> RewardTermCfg:
    params: dict[str, Any] = {"command_name": "motion"}
    if body_names is not None:
        params["body_names"] = body_names
    return RewardTermCfg(func=None, weight=1.0, params=params)


def _expected_body_reward(
    reference: np.ndarray,
    actual: np.ndarray,
    body_ids: slice | list[int],
    std: float,
    *,
    orientation: bool,
) -> np.ndarray:
    reference = reference[:, body_ids]
    actual = actual[:, body_ids]
    if orientation:
        error = np_quat_error_magnitude_squared_batched(reference, actual)
    else:
        error = np.square(reference - actual).sum(axis=-1)
    return np.exp(-error.mean(axis=-1) / std**2)


def test_anchor_position_error_exp_bit_parity(body_setup) -> None:
    command, env, snapshots = body_setup
    out = mt.motion_global_anchor_position_error_exp(env, "motion", std=0.3)
    expected = np.exp(
        -np.sum(
            np.square(snapshots["anchor_pos_w"] - snapshots["robot_anchor_pos_w"]),
            axis=-1,
        )
        / 0.3**2
    )
    np.testing.assert_array_equal(out, expected)
    np.testing.assert_array_equal(command.anchor_pos_w, snapshots["anchor_pos_w"])


def test_joint_position_error_exp_bit_parity(body_setup) -> None:
    command, env, snapshots = body_setup
    out = mt.motion_joint_position_error_exp(env, "motion", std=0.2)
    expected = np.exp(
        -np.square(snapshots["joint_pos"] - snapshots["robot_joint_pos"]).mean(axis=-1) / 0.2**2
    )
    np.testing.assert_array_equal(out, expected)


def test_anchor_pos_termination_numba_parity_and_output_reuse(body_setup) -> None:
    command, env, snapshots = body_setup
    cfg = TerminationTermCfg(
        func=mt.bad_anchor_pos_z_only,
        params={"command_name": "motion", "threshold": 0.15},
    )
    term = mt.bad_anchor_pos_z_only(cfg, env)

    out = term(env, "motion", threshold=0.15)
    expected = (
        np.abs(
            snapshots["body_pos_w"][:, command.anchor_body_idx, 2]
            - snapshots["robot_anchor_pos_w"][:, 2]
        )
        > 0.15
    )
    np.testing.assert_array_equal(out, expected)

    out2 = term(env, "motion", threshold=0.3)
    assert out2 is out
    expected2 = (
        np.abs(
            snapshots["body_pos_w"][:, command.anchor_body_idx, 2]
            - snapshots["robot_anchor_pos_w"][:, 2]
        )
        > 0.3
    )
    np.testing.assert_array_equal(out2, expected2)
    np.testing.assert_array_equal(command.body_pos_w, snapshots["body_pos_w"])


@pytest.mark.parametrize(
    ("term_type", "reference_name", "actual_name", "std", "orientation"),
    [
        (
            mt.motion_relative_body_position_error_exp,
            "body_pos_relative_w",
            "robot_body_pos_w",
            0.3,
            False,
        ),
        (
            mt.motion_relative_body_orientation_error_exp,
            "body_quat_relative_w",
            "robot_body_quat_w",
            0.4,
            True,
        ),
        (
            mt.motion_global_body_linear_velocity_error_exp,
            "body_lin_vel_w",
            "robot_body_lin_vel_w",
            1.0,
            False,
        ),
        (
            mt.motion_global_body_angular_velocity_error_exp,
            "body_ang_vel_w",
            "robot_body_ang_vel_w",
            3.14,
            False,
        ),
    ],
)
def test_numba_body_rewards_match_numpy_and_reuse_output(
    body_setup,
    term_type,
    reference_name: str,
    actual_name: str,
    std: float,
    orientation: bool,
) -> None:
    command, env, snapshots = body_setup
    term = term_type(_reward_cfg(), env)

    out = term(env, "motion", std=std)
    expected = _expected_body_reward(
        snapshots[reference_name],
        snapshots[actual_name],
        slice(None),
        std,
        orientation=orientation,
    )
    np.testing.assert_allclose(out, expected, rtol=2e-6, atol=2e-7)
    assert out.dtype == snapshots[reference_name].dtype
    first_result = out.copy()

    second_std = std * 1.5
    out2 = term(env, "motion", std=second_std)
    assert out2 is out
    expected2 = _expected_body_reward(
        snapshots[reference_name],
        snapshots[actual_name],
        slice(None),
        second_std,
        orientation=orientation,
    )
    np.testing.assert_allclose(out2, expected2, rtol=2e-6, atol=2e-7)
    assert np.any(first_result != out2)
    np.testing.assert_array_equal(getattr(command, reference_name), snapshots[reference_name])
    np.testing.assert_array_equal(getattr(command, actual_name), snapshots[actual_name])


@pytest.mark.parametrize(
    ("term_type", "reference_name", "actual_name", "std", "orientation"),
    [
        (
            mt.motion_relative_body_position_error_exp,
            "body_pos_relative_w",
            "robot_body_pos_w",
            0.3,
            False,
        ),
        (
            mt.motion_relative_body_orientation_error_exp,
            "body_quat_relative_w",
            "robot_body_quat_w",
            0.4,
            True,
        ),
        (
            mt.motion_global_body_linear_velocity_error_exp,
            "body_lin_vel_w",
            "robot_body_lin_vel_w",
            1.0,
            False,
        ),
        (
            mt.motion_global_body_angular_velocity_error_exp,
            "body_ang_vel_w",
            "robot_body_ang_vel_w",
            3.14,
            False,
        ),
    ],
)
def test_numba_body_rewards_preserve_body_subset_contract(
    body_setup,
    term_type,
    reference_name: str,
    actual_name: str,
    std: float,
    orientation: bool,
) -> None:
    command, env, snapshots = body_setup
    body_names = ("b0", "b3", "b11")
    term = term_type(_reward_cfg(body_names=body_names), env)

    out = term(env, "motion", std=std, body_names=body_names)
    expected = _expected_body_reward(
        snapshots[reference_name],
        snapshots[actual_name],
        [0, 3, 11],
        std,
        orientation=orientation,
    )
    np.testing.assert_allclose(out, expected, rtol=2e-6, atol=2e-7)
    assert command.cfg.body_names == tuple(f"b{i}" for i in range(12))


def test_motion_hot_kernels_compile_parallel_on_term_construction(body_setup) -> None:
    _, env, _ = body_setup
    mt.bad_anchor_pos_z_only(
        TerminationTermCfg(
            func=mt.bad_anchor_pos_z_only,
            params={"command_name": "motion", "threshold": 0.15},
        ),
        env,
    )
    for term_type in (
        mt.motion_relative_body_position_error_exp,
        mt.motion_relative_body_orientation_error_exp,
        mt.motion_global_body_linear_velocity_error_exp,
        mt.motion_global_body_angular_velocity_error_exp,
    ):
        term_type(_reward_cfg(), env)

    dispatchers = (
        kernels.termination_anchor_pos_kernel,
        kernels.reward_motion_body_pos_kernel,
        kernels.reward_motion_body_ori_kernel,
        kernels.reward_motion_body_lin_vel_kernel,
        kernels.reward_motion_body_ang_vel_kernel,
    )
    for dispatcher in dispatchers:
        assert dispatcher.targetoptions["nopython"] is True
        assert dispatcher.targetoptions["nogil"] is True
        assert dispatcher.targetoptions["parallel"] is True
        assert dispatcher.signatures
    if "NUMBA_THREADING_LAYER" not in os.environ:
        assert threading_layer() == "workqueue"
    if "NUMBA_NUM_THREADS" not in os.environ:
        assert get_num_threads() == min(8, config.NUMBA_DEFAULT_NUM_THREADS)


def test_motion_metrics_kernel_matches_numpy_and_scopes_rows() -> None:
    rng = np.random.default_rng(1701)
    num_envs, num_bodies, num_joints = 257, 12, 29
    anchor_body_idx = 4
    motion_pos = rng.standard_normal((num_envs, num_bodies, 3), dtype=np.float32)
    robot_pos = rng.standard_normal((num_envs, num_bodies, 3), dtype=np.float32)
    relative_pos = rng.standard_normal((num_envs, num_bodies, 3), dtype=np.float32)
    motion_quat = _unit_quat(rng.standard_normal((num_envs, num_bodies, 4), dtype=np.float32))
    robot_quat = _unit_quat(rng.standard_normal((num_envs, num_bodies, 4), dtype=np.float32))
    relative_quat = _unit_quat(rng.standard_normal((num_envs, num_bodies, 4), dtype=np.float32))
    motion_lin = rng.standard_normal((num_envs, num_bodies, 3), dtype=np.float32)
    robot_lin = rng.standard_normal((num_envs, num_bodies, 3), dtype=np.float32)
    motion_ang = rng.standard_normal((num_envs, num_bodies, 3), dtype=np.float32)
    robot_ang = rng.standard_normal((num_envs, num_bodies, 3), dtype=np.float32)
    motion_joint_pos = rng.standard_normal((num_envs, num_joints), dtype=np.float32)
    robot_joint_pos = rng.standard_normal((num_envs, num_joints), dtype=np.float32)
    motion_joint_vel = rng.standard_normal((num_envs, num_joints), dtype=np.float32)
    robot_joint_vel = rng.standard_normal((num_envs, num_joints), dtype=np.float32)
    inputs = (
        motion_pos,
        robot_pos,
        motion_quat,
        robot_quat,
        motion_lin,
        robot_lin,
        motion_ang,
        robot_ang,
        relative_pos,
        relative_quat,
        motion_joint_pos,
        robot_joint_pos,
        motion_joint_vel,
        robot_joint_vel,
    )
    snapshots = tuple(value.copy() for value in inputs)
    expected = (
        np.linalg.norm(motion_pos[:, anchor_body_idx] - robot_pos[:, anchor_body_idx], axis=-1),
        np.sqrt(
            np_quat_error_magnitude_squared_batched(
                motion_quat[:, anchor_body_idx], robot_quat[:, anchor_body_idx]
            )
        ),
        np.linalg.norm(motion_lin[:, anchor_body_idx] - robot_lin[:, anchor_body_idx], axis=-1),
        np.linalg.norm(motion_ang[:, anchor_body_idx] - robot_ang[:, anchor_body_idx], axis=-1),
        np.linalg.norm(relative_pos - robot_pos, axis=-1).mean(axis=-1),
        np.sqrt(np_quat_error_magnitude_squared_batched(relative_quat, robot_quat)).mean(axis=-1),
        np.linalg.norm(motion_lin - robot_lin, axis=-1).mean(axis=-1),
        np.linalg.norm(motion_ang - robot_ang, axis=-1).mean(axis=-1),
        np.linalg.norm(motion_joint_pos - robot_joint_pos, axis=-1),
        np.linalg.norm(motion_joint_vel - robot_joint_vel, axis=-1),
    )
    outputs = tuple(np.full(num_envs, -123.0, dtype=np.float32) for _ in expected)

    def run(rows: np.ndarray) -> None:
        kernels.update_motion_metrics_kernel(
            rows,
            anchor_body_idx,
            motion_pos,
            robot_pos,
            motion_quat,
            robot_quat,
            motion_lin,
            robot_lin,
            motion_ang,
            robot_ang,
            relative_pos,
            relative_quat,
            motion_joint_pos,
            robot_joint_pos,
            motion_joint_vel,
            robot_joint_vel,
            *outputs,
        )

    selected = np.asarray([0, 3, 128, 256], dtype=np.int32)
    run(selected)
    untouched = np.ones(num_envs, dtype=bool)
    untouched[selected] = False
    for actual, reference in zip(outputs, expected, strict=True):
        np.testing.assert_allclose(actual[selected], reference[selected], rtol=2e-6, atol=1e-5)
        np.testing.assert_array_equal(actual[untouched], -123.0)

    run(np.arange(num_envs, dtype=np.int32))
    for actual, reference in zip(outputs, expected, strict=True):
        np.testing.assert_allclose(actual, reference, rtol=2e-6, atol=1e-5)
    for actual, snapshot in zip(inputs, snapshots, strict=True):
        np.testing.assert_array_equal(actual, snapshot)
    assert kernels.update_motion_metrics_kernel.targetoptions["nopython"] is True
    assert kernels.update_motion_metrics_kernel.targetoptions["nogil"] is True
    assert kernels.update_motion_metrics_kernel.targetoptions["parallel"] is True
    assert kernels.update_motion_metrics_kernel.signatures


def test_motion_relative_state_kernel_matches_numpy_and_scopes_rows() -> None:
    rng = np.random.default_rng(1818)
    num_envs, num_bodies = 257, 12
    anchor_body_idx = 4
    motion_pos_local = rng.standard_normal((num_envs, num_bodies, 3), dtype=np.float32)
    env_origins = rng.standard_normal((num_envs, 1, 3), dtype=np.float32)
    motion_pos_world = motion_pos_local + env_origins
    motion_quat = _unit_quat(rng.standard_normal((num_envs, num_bodies, 4), dtype=np.float32))
    robot_pos = rng.standard_normal((num_envs, num_bodies, 3), dtype=np.float32)
    robot_quat = _unit_quat(rng.standard_normal((num_envs, num_bodies, 4), dtype=np.float32))
    inputs = (motion_pos_local, motion_pos_world, motion_quat, robot_pos, robot_quat)
    snapshots = tuple(value.copy() for value in inputs)

    motion_anchor_pos_local = motion_pos_local[:, anchor_body_idx]
    motion_anchor_quat = motion_quat[:, anchor_body_idx]
    robot_anchor_pos = robot_pos[:, anchor_body_idx]
    robot_anchor_quat = robot_quat[:, anchor_body_idx]
    delta_pos = robot_anchor_pos.copy()
    delta_pos[:, 2] = motion_anchor_pos_local[:, 2]
    delta_quat = np_yaw_quat(
        np_quat_mul_batched(robot_anchor_quat, np_quat_inv(motion_anchor_quat))
    )
    expected_body_pos_relative = np_quat_apply_batched(
        delta_quat[:, None],
        motion_pos_local - motion_anchor_pos_local[:, None],
    )
    expected_body_pos_relative += delta_pos[:, None]
    expected_body_quat_relative = np_quat_mul_batched(delta_quat[:, None], motion_quat)
    expected_motion_anchor_pos = np_quat_apply_inverse_batched(
        robot_anchor_quat,
        motion_pos_world[:, anchor_body_idx] - robot_anchor_pos,
    )
    expected_motion_anchor_ori = np_matrix_first_two_cols_from_quat(
        np_quat_mul_batched(np_quat_inv(robot_anchor_quat), motion_anchor_quat)
    )
    expected_robot_body_pos = np_quat_apply_inverse_batched(
        robot_anchor_quat[:, None],
        robot_pos - robot_anchor_pos[:, None],
    )
    expected_robot_body_ori = np_matrix_first_two_cols_from_quat(
        np_quat_mul_batched(np_quat_inv(robot_anchor_quat)[:, None], robot_quat)
    )
    expected = (
        expected_body_pos_relative,
        expected_body_quat_relative,
        expected_motion_anchor_pos,
        expected_motion_anchor_ori,
        expected_robot_body_pos,
        expected_robot_body_ori,
    )
    outputs = tuple(np.full(value.shape, -123.0, dtype=np.float32) for value in expected)
    output_addresses = tuple(value.ctypes.data for value in outputs)

    def run(rows: np.ndarray) -> None:
        kernels.update_motion_relative_state_kernel(
            rows,
            anchor_body_idx,
            motion_pos_local,
            motion_pos_world,
            motion_quat,
            robot_pos,
            robot_quat,
            *outputs,
        )

    selected = np.asarray([0, 3, 128, 256], dtype=np.int32)
    run(selected)
    untouched = np.ones(num_envs, dtype=bool)
    untouched[selected] = False
    for actual, reference in zip(outputs, expected, strict=True):
        np.testing.assert_allclose(actual[selected], reference[selected], rtol=3e-6, atol=2e-6)
        np.testing.assert_array_equal(actual[untouched], -123.0)

    run(np.arange(num_envs, dtype=np.int32))
    for actual, reference in zip(outputs, expected, strict=True):
        np.testing.assert_allclose(actual, reference, rtol=3e-6, atol=2e-6)
    for actual, snapshot in zip(inputs, snapshots, strict=True):
        np.testing.assert_array_equal(actual, snapshot)
    assert tuple(value.ctypes.data for value in outputs) == output_addresses
    assert kernels.update_motion_relative_state_kernel.targetoptions["nopython"] is True
    assert kernels.update_motion_relative_state_kernel.targetoptions["nogil"] is True
    assert kernels.update_motion_relative_state_kernel.targetoptions["parallel"] is True
    assert kernels.update_motion_relative_state_kernel.signatures


def test_object_relative_state_kernel_matches_numpy_and_scopes_rows() -> None:
    rng = np.random.default_rng(1819)
    num_envs = 257
    anchor_pos = rng.standard_normal((num_envs, 3), dtype=np.float32)
    anchor_quat = _unit_quat(rng.standard_normal((num_envs, 4), dtype=np.float32))
    object_pos = rng.standard_normal((num_envs, 3), dtype=np.float32)
    object_quat = _unit_quat(rng.standard_normal((num_envs, 4), dtype=np.float32))
    object_lin_vel = rng.standard_normal((num_envs, 3), dtype=np.float32)
    inputs = (anchor_pos, anchor_quat, object_pos, object_quat, object_lin_vel)
    snapshots = tuple(value.copy() for value in inputs)
    expected = np.concatenate(
        (
            np_quat_apply_inverse_batched(anchor_quat, object_pos - anchor_pos),
            np_matrix_first_two_cols_from_quat(
                np_quat_mul_batched(np_quat_inv(anchor_quat), object_quat)
            ),
            np_quat_apply_inverse_batched(anchor_quat, object_lin_vel),
        ),
        axis=-1,
    )
    output = np.full(expected.shape, -123.0, dtype=np.float32)
    output_address = output.ctypes.data

    selected = np.asarray([0, 3, 128, 256], dtype=np.int32)
    kernels.update_object_relative_state_kernel(
        selected,
        anchor_pos,
        anchor_quat,
        object_pos,
        object_quat,
        object_lin_vel,
        output,
    )
    untouched = np.ones(num_envs, dtype=bool)
    untouched[selected] = False
    np.testing.assert_allclose(output[selected], expected[selected], rtol=3e-6, atol=2e-6)
    np.testing.assert_array_equal(output[untouched], -123.0)

    kernels.update_object_relative_state_kernel(
        np.arange(num_envs, dtype=np.int32),
        anchor_pos,
        anchor_quat,
        object_pos,
        object_quat,
        object_lin_vel,
        output,
    )
    np.testing.assert_allclose(output, expected, rtol=3e-6, atol=2e-6)
    for actual, snapshot in zip(inputs, snapshots, strict=True):
        np.testing.assert_array_equal(actual, snapshot)
    assert output.ctypes.data == output_address
    assert kernels.update_object_relative_state_kernel.targetoptions["nopython"] is True
    assert kernels.update_object_relative_state_kernel.targetoptions["nogil"] is True
    assert kernels.update_object_relative_state_kernel.targetoptions["parallel"] is True
    assert kernels.update_object_relative_state_kernel.signatures


def test_joint_pos_limits_bit_parity() -> None:
    rng = np.random.default_rng(123)
    joint_pos = rng.standard_normal((8, 5), dtype=np.float32)
    limits = np.asarray([[-1.0, 1.0]] * 5, dtype=np.float32)
    asset = SimpleNamespace(data=SimpleNamespace(joint_pos=joint_pos, soft_joint_pos_limits=limits))
    env = SimpleNamespace(scene={"robot": asset})
    asset_cfg = SimpleNamespace(name="robot", joint_ids=np.array([4, 2, 0], dtype=np.intp))

    out = mt.joint_pos_limits(env, asset_cfg)
    selected = joint_pos[:, [4, 2, 0]]
    selected_limits = limits[[4, 2, 0]]
    expected = np.sum(
        np.square(
            np.maximum(selected_limits[:, 0] - selected, 0.0)
            + np.maximum(selected - selected_limits[:, 1], 0.0)
        ),
        axis=-1,
    )
    np.testing.assert_array_equal(out, expected)
