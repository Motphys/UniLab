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
from unilab.utils.rotation import np_quat_error_magnitude_squared_batched


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
