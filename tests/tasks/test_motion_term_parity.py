"""Numerical-contract tests for optimized motion-tracking manager terms."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch

from unilab.managers import RewardTermCfg, TerminationTermCfg
from unilab.tasks.motion_tracking.common import manager_terms as mt
from unilab.tasks.motion_tracking.common.motion_math import (
    _adaptive_failure_alpha,
    _adaptive_failure_counts,
    _gravity_z_in_body,
)
from unilab.utils.rotation import (
    np_quat_apply_inverse_batched,
    np_quat_error_magnitude_squared_batched,
    np_quat_from_euler_xyz,
    np_quat_mul_batched,
)


def _make_env(command: Any) -> SimpleNamespace:
    return SimpleNamespace(num_envs=command.num_envs, device=torch.device("cpu"))


def _unit_quat(value: np.ndarray) -> np.ndarray:
    value /= np.linalg.norm(value, axis=-1, keepdims=True)
    return value


def test_torch_anchor_orientation_matches_manager_projected_gravity() -> None:
    quaternions = np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
        ],
        dtype=np.float32,
    )
    gravity = np.tile(np.array([0.0, 0.0, -1.0], dtype=np.float32), (len(quaternions), 1))
    expected = np_quat_apply_inverse_batched(quaternions, gravity)[:, 2]
    actual = _gravity_z_in_body(torch.from_numpy(quaternions)).numpy()
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=2e-7)


def test_torch_adaptive_sampler_preserves_counts_without_failure() -> None:
    bin_indices = torch.tensor([0, 1, 1, 3], dtype=torch.int64)
    terminated = torch.tensor([True, False, True, False])
    failures = _adaptive_failure_counts(bin_indices, terminated, n_bins=4)
    torch.testing.assert_close(failures, torch.tensor([1.0, 1.0, 0.0, 0.0]))
    torch.testing.assert_close(
        _adaptive_failure_alpha(torch.zeros_like(terminated), 0.001),
        torch.tensor(0.0),
    )


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

    body_pos_w = torch.from_numpy(body_pos_w)
    robot_body_pos_w = torch.from_numpy(robot_body_pos_w)
    body_pos_relative_w = torch.from_numpy(body_pos_relative_w)
    body_quat_relative_w = torch.from_numpy(body_quat_relative_w)
    robot_body_quat_w = torch.from_numpy(robot_body_quat_w)
    body_lin_vel_w = torch.from_numpy(body_lin_vel_w)
    robot_body_lin_vel_w = torch.from_numpy(robot_body_lin_vel_w)
    body_ang_vel_w = torch.from_numpy(body_ang_vel_w)
    robot_body_ang_vel_w = torch.from_numpy(robot_body_ang_vel_w)

    command = SimpleNamespace(
        num_envs=num_envs,
        cfg=SimpleNamespace(body_names=tuple(f"b{i}" for i in range(num_bodies))),
        anchor_body_idx=anchor_body_idx,
        tensor_carrier=True,
        _body_ids=slice(None),
        body_pos_w=body_pos_w,
        robot_body_pos_w=robot_body_pos_w,
        anchor_pos_w=body_pos_w[:, anchor_body_idx],
        robot_anchor_pos_w=robot_body_pos_w[:, anchor_body_idx],
        anchor_quat_w=body_quat_relative_w[:, anchor_body_idx],
        robot_anchor_quat_w=robot_body_quat_w[:, anchor_body_idx],
        joint_pos=torch.from_numpy(rng.standard_normal((num_envs, 29), dtype=np.float32)),
        joint_vel=torch.from_numpy(rng.standard_normal((num_envs, 29), dtype=np.float32)),
        robot_joint_pos=torch.from_numpy(rng.standard_normal((num_envs, 29), dtype=np.float32)),
        robot_joint_vel=torch.from_numpy(rng.standard_normal((num_envs, 29), dtype=np.float32)),
        body_pos_relative_w=body_pos_relative_w,
        body_quat_relative_w=body_quat_relative_w,
        robot_body_quat_w=robot_body_quat_w,
        body_lin_vel_w=body_lin_vel_w,
        robot_body_lin_vel_w=robot_body_lin_vel_w,
        body_ang_vel_w=body_ang_vel_w,
        robot_body_ang_vel_w=robot_body_ang_vel_w,
    )

    snapshots = {
        key: value.detach().clone()
        for key, value in vars(command).items()
        if isinstance(value, torch.Tensor)
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
        from unilab.tasks.motion_tracking.common.tensor_rotation import (
            quat_conjugate,
            quat_mul,
        )

        rel = quat_mul(quat_conjugate(reference), actual)
        xyz = torch.linalg.vector_norm(rel[..., 1:4], dim=-1)
        angle = 2.0 * torch.atan2(xyz, rel[..., 0].abs().clamp(max=1.0))
        error = angle.square()
    else:
        error = (reference - actual).square().sum(dim=-1)
    return torch.exp(-error.mean(dim=-1) / std**2)


@pytest.mark.parametrize(
    ("function", "reference_name", "actual_name", "std", "mean"),
    [
        (
            mt.motion_global_anchor_position_error_exp,
            "anchor_pos_w",
            "robot_anchor_pos_w",
            0.3,
            False,
        ),
        (mt.motion_joint_position_error_exp, "joint_pos", "robot_joint_pos", 0.2, True),
        (mt.motion_joint_velocity_error_exp, "joint_vel", "robot_joint_vel", 1.0, True),
    ],
)
def test_tensor_scalar_error_rewards_match_numpy_and_do_not_mutate(
    monkeypatch: pytest.MonkeyPatch,
    body_setup,
    function,
    reference_name: str | None,
    actual_name: str | None,
    std: float,
    mean: bool,
) -> None:
    command, env, snapshots = body_setup
    tensor_command = command

    out = function(env, "motion", std=std)

    reference = snapshots[reference_name]
    actual = snapshots[actual_name]
    error = (reference - actual).square()
    reduction = error.mean(dim=-1) if mean else error.sum(dim=-1)
    expected = torch.exp(-reduction / std**2)
    assert isinstance(out, torch.Tensor)
    torch.testing.assert_close(out, expected, rtol=2e-6, atol=2e-7)
    torch.testing.assert_close(
        getattr(tensor_command, reference_name),
        snapshots[reference_name],
    )
    torch.testing.assert_close(
        getattr(tensor_command, actual_name),
        snapshots[actual_name],
    )


def test_tensor_joint_rewards_use_device_resident_actual_state(
    monkeypatch: pytest.MonkeyPatch, body_setup
) -> None:
    command, env, snapshots = body_setup
    motion_joint_pos = snapshots["joint_pos"].detach().clone()
    device_joint_pos = motion_joint_pos + 0.125
    motion_joint_vel = snapshots["joint_vel"].detach().clone()
    device_joint_vel = motion_joint_vel - 0.25
    tensor_command = SimpleNamespace(
        tensor_carrier=True,
        joint_pos=motion_joint_pos,
        device_robot_joint_pos=device_joint_pos,
        joint_vel=motion_joint_vel,
        device_robot_joint_vel=device_joint_vel,
    )
    monkeypatch.setattr(mt, "_command", lambda env, name: tensor_command)

    position_reward = mt.motion_joint_position_error_exp(env, "motion", std=0.5)
    velocity_reward = mt.motion_joint_velocity_error_exp(env, "motion", std=2.0)

    torch.testing.assert_close(
        position_reward,
        torch.exp(-(motion_joint_pos - device_joint_pos).square().mean(dim=-1) / 0.5**2),
        rtol=2e-6,
        atol=2e-7,
    )
    torch.testing.assert_close(
        velocity_reward,
        torch.exp(-(motion_joint_vel - device_joint_vel).square().mean(dim=-1) / 2.0**2),
        rtol=2e-6,
        atol=2e-7,
    )


def test_tensor_anchor_orientation_reward_matches_numpy(
    monkeypatch: pytest.MonkeyPatch, body_setup
) -> None:
    command, env, snapshots = body_setup
    motion_quat = _unit_quat(
        np.random.default_rng(17).standard_normal((command.num_envs, 4), dtype=np.float32)
    )
    robot_quat = _unit_quat(
        motion_quat
        + 0.08 * np.random.default_rng(19).standard_normal(motion_quat.shape, dtype=np.float32)
    )
    tensor_command = SimpleNamespace(
        num_envs=command.num_envs,
        cfg=command.cfg,
        tensor_carrier=True,
        anchor_quat_w=torch.from_numpy(motion_quat.copy()),
        robot_anchor_quat_w=torch.from_numpy(robot_quat.copy()),
    )
    monkeypatch.setattr(mt, "_command", lambda env, name: tensor_command)

    out = mt.motion_global_anchor_orientation_error_exp(env, "motion", std=0.4)
    expected = torch.from_numpy(
        np.exp(-np_quat_error_magnitude_squared_batched(motion_quat, robot_quat) / 0.4**2)
    )
    assert isinstance(out, torch.Tensor)
    torch.testing.assert_close(out, expected, rtol=2e-6, atol=2e-7)


def _tensor_command_from_snapshots(command, snapshots):
    return SimpleNamespace(
        num_envs=command.num_envs,
        cfg=command.cfg,
        anchor_body_idx=command.anchor_body_idx,
        tensor_carrier=True,
        **{
            name: value.detach().clone()
            for name, value in snapshots.items()
            if isinstance(value, torch.Tensor)
        },
    )


def test_tensor_anchor_position_termination_matches_numpy(
    monkeypatch: pytest.MonkeyPatch, body_setup
) -> None:
    command, env, snapshots = body_setup
    tensor_command = _tensor_command_from_snapshots(command, snapshots)
    monkeypatch.setattr(mt, "_command", lambda env, name: tensor_command)
    cfg = TerminationTermCfg(
        func=mt.bad_anchor_pos_z_only,
        params={"command_name": "motion", "threshold": 0.15},
    )
    term = mt.bad_anchor_pos_z_only(cfg, env)

    out = term(env, "motion", threshold=0.15)
    expected = (
        snapshots["body_pos_w"][:, command.anchor_body_idx, 2]
        - snapshots["robot_anchor_pos_w"][:, 2]
    ).abs() > 0.15
    assert isinstance(out, torch.Tensor)
    torch.testing.assert_close(out, expected)


def test_tensor_anchor_orientation_termination_matches_numpy(
    monkeypatch: pytest.MonkeyPatch, body_setup
) -> None:
    command, env, snapshots = body_setup
    motion_quat = _unit_quat(
        np.random.default_rng(21).standard_normal((command.num_envs, 4), dtype=np.float32)
    )
    robot_quat = _unit_quat(
        motion_quat
        + 0.1 * np.random.default_rng(23).standard_normal(motion_quat.shape, dtype=np.float32)
    )
    gravity = np.tile(np.array([0.0, 0.0, -1.0], dtype=np.float32), (command.num_envs, 1))
    tensor_command = SimpleNamespace(
        num_envs=command.num_envs,
        cfg=command.cfg,
        tensor_carrier=True,
        anchor_quat_w=torch.from_numpy(motion_quat.copy()),
        robot_anchor_quat_w=torch.from_numpy(robot_quat.copy()),
    )
    monkeypatch.setattr(mt, "_command", lambda env, name: tensor_command)

    out = mt.bad_anchor_ori(env, "motion", threshold=0.8)
    expected = (
        torch.from_numpy(np_quat_apply_inverse_batched(motion_quat, gravity)[:, 2])
        - torch.from_numpy(np_quat_apply_inverse_batched(robot_quat, gravity)[:, 2])
    ).abs() > 0.8
    assert isinstance(out, torch.Tensor)
    torch.testing.assert_close(out, expected)


@pytest.mark.parametrize(
    ("term_type", "reference_name", "actual_name", "threshold"),
    [
        (mt.bad_motion_body_pos_z_only, "body_pos_relative_w", "robot_body_pos_w", 0.5),
        (mt.bad_undesired_body_contacts, None, "robot_body_pos_w", 0.05),
    ],
)
def test_tensor_body_terminations_match_numpy(
    monkeypatch: pytest.MonkeyPatch,
    body_setup,
    term_type,
    reference_name: str | None,
    actual_name: str,
    threshold: float,
) -> None:
    command, env, snapshots = body_setup
    tensor_command = _tensor_command_from_snapshots(command, snapshots)
    monkeypatch.setattr(mt, "_command", lambda env, name: tensor_command)
    body_names = ("b0", "b3", "b11")
    term = term_type(
        TerminationTermCfg(
            func=term_type,
            params={
                "command_name": "motion",
                "threshold": threshold,
                "body_names": body_names,
            },
        ),
        env,
    )

    out = term(env, "motion", threshold=threshold, body_names=body_names)
    actual = snapshots[actual_name][:, [0, 3, 11], 2]
    if reference_name is None:
        expected = (actual < threshold).any(dim=-1)
    else:
        reference = snapshots[reference_name][:, [0, 3, 11], 2]
        expected = ((reference - actual).abs() > threshold).any(dim=-1)
    assert isinstance(out, torch.Tensor)
    torch.testing.assert_close(out, expected)


def test_tensor_motion_clip_end_uses_current_rows(monkeypatch, body_setup) -> None:
    command, env, snapshots = body_setup
    frames = np.array([1, 5, 10, 15], dtype=np.int64)
    clip_ends = np.array([2, 4, 10, 20], dtype=np.int64)
    tensor_command = SimpleNamespace(
        num_envs=command.num_envs,
        cfg=command.cfg,
        tensor_carrier=True,
        time_steps=torch.from_numpy(frames),
        tensor_current_clip_end_frames=torch.from_numpy(clip_ends),
    )
    monkeypatch.setattr(mt, "_command", lambda env, name: tensor_command)

    out = mt.motion_clip_end(env, "motion")

    assert isinstance(out, torch.Tensor)
    torch.testing.assert_close(out, torch.from_numpy(frames >= clip_ends))


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
def test_body_rewards_match_numpy_reference_without_mutation(
    monkeypatch: pytest.MonkeyPatch,
    body_setup,
    term_type,
    reference_name: str,
    actual_name: str,
    std: float,
    orientation: bool,
) -> None:
    command, env, snapshots = body_setup
    tensor_command = command
    term = term_type(_reward_cfg(), env)

    out = term(env, "motion", std=std)

    expected = _expected_body_reward(
        snapshots[reference_name],
        snapshots[actual_name],
        slice(None),
        std,
        orientation=orientation,
    )
    assert isinstance(out, torch.Tensor)
    torch.testing.assert_close(out, expected, rtol=2e-6, atol=2e-7)
    torch.testing.assert_close(
        getattr(tensor_command, reference_name),
        snapshots[reference_name],
    )
    torch.testing.assert_close(
        getattr(tensor_command, actual_name),
        snapshots[actual_name],
    )


def test_motion_metrics_tensor_peer_matches_numpy_and_scopes_rows() -> None:
    rng = np.random.default_rng(1820)
    num_envs, num_bodies, num_joints, anchor_body_idx = 257, 12, 29, 4
    motion_pos = rng.standard_normal((num_envs, num_bodies, 3), dtype=np.float32)
    robot_pos = rng.standard_normal((num_envs, num_bodies, 3), dtype=np.float32)
    motion_quat = _unit_quat(rng.standard_normal((num_envs, num_bodies, 4), dtype=np.float32))
    robot_quat = _unit_quat(rng.standard_normal((num_envs, num_bodies, 4), dtype=np.float32))
    motion_lin = rng.standard_normal((num_envs, num_bodies, 3), dtype=np.float32)
    robot_lin = rng.standard_normal((num_envs, num_bodies, 3), dtype=np.float32)
    motion_ang = rng.standard_normal((num_envs, num_bodies, 3), dtype=np.float32)
    robot_ang = rng.standard_normal((num_envs, num_bodies, 3), dtype=np.float32)
    relative_pos = rng.standard_normal((num_envs, num_bodies, 3), dtype=np.float32)
    relative_quat = _unit_quat(rng.standard_normal((num_envs, num_bodies, 4), dtype=np.float32))
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
    outputs = tuple(torch.full((num_envs,), -123.0, dtype=torch.float32) for _ in expected)
    tensor_inputs = tuple(torch.from_numpy(value) for value in inputs)

    selected = torch.tensor([0, 3, 128, 256], dtype=torch.int64)
    mt._update_motion_metrics_torch(selected, anchor_body_idx, *tensor_inputs, outputs)
    untouched = torch.ones(num_envs, dtype=torch.bool)
    untouched[selected] = False
    for actual, reference in zip(outputs, expected, strict=True):
        torch.testing.assert_close(actual[selected], torch.from_numpy(reference[selected]))
    for actual in outputs:
        assert torch.equal(actual[untouched], torch.full_like(actual[untouched], -123.0))

    mt._update_motion_metrics_torch(
        torch.arange(num_envs, dtype=torch.int64), anchor_body_idx, *tensor_inputs, outputs
    )
    for actual, reference in zip(outputs, expected, strict=True):
        torch.testing.assert_close(actual, torch.from_numpy(reference))
    for actual, snapshot in zip(tensor_inputs, inputs, strict=True):
        torch.testing.assert_close(actual, torch.from_numpy(snapshot))


def test_joint_pos_limits_matches_reference_equation() -> None:
    rng = np.random.default_rng(123)
    joint_pos = rng.standard_normal((8, 5), dtype=np.float32)
    limits = np.asarray([[-1.0, 1.0]] * 5, dtype=np.float32)
    asset = SimpleNamespace(data=SimpleNamespace(joint_pos=joint_pos, soft_joint_pos_limits=limits))
    env = SimpleNamespace(scene={"robot": asset}, device=torch.device("cpu"))
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
    torch.testing.assert_close(out, torch.from_numpy(expected), rtol=0.0, atol=0.0)


def test_tensor_command_publishes_sampler_advance_exactly_once() -> None:
    command = mt.MotionCommand.__new__(mt.MotionCommand)
    command._device = torch.device("cpu")
    command.last_step_timing_ms = {}
    command.time_steps = torch.tensor([4, 0, 9], dtype=torch.int32)
    command.tensor_sampler = mt.TensorMotionSampler(
        mode="adaptive",
        num_envs=3,
        num_frames=11,
        clip_offsets=torch.tensor([0], dtype=torch.int64),
        clip_end_frames=torch.tensor([100], dtype=torch.int32),
        bin_count=1,
        adaptive_lambda=0.8,
        adaptive_kernel_size=1,
        adaptive_uniform_ratio=0.1,
        adaptive_alpha=0.001,
        start_ratio=0.0,
        initial_frames=torch.tensor([4, 0, 9], dtype=torch.int32),
        initial_clip_end_frames=torch.tensor([100, 100, 100], dtype=torch.int32),
        device=torch.device("cpu"),
    )
    command.tensor_sampler.current_frames.copy_(command.time_steps)
    command.current_clip_end_frames = torch.tensor([90, 90, 90], dtype=torch.int32)
    command._clip_offsets_torch = torch.tensor([0], dtype=torch.int64)
    command._clip_end_frames_torch = torch.tensor([100], dtype=torch.int64)
    command._tensor_all_rows = torch.arange(3, dtype=torch.int64)
    command.cfg = SimpleNamespace(params=SimpleNamespace(truncate_on_clip_end=True))
    command._tensor_post_compute_env_ids = None
    command._tensor_resample_ingested = None
    command._env = SimpleNamespace(
        termination_manager=SimpleNamespace(terminated=torch.tensor([False, False, False])),
        reset_buf=torch.tensor([False, False, False]),
    )

    command._refresh_motion_torch = lambda *args, **kwargs: None
    command._update_command(None)

    torch.testing.assert_close(command.time_steps, torch.tensor([5, 1, 10], dtype=torch.int32))


def test_tensor_command_full_refresh_gathers_advanced_frames() -> None:
    command = mt.MotionCommand.__new__(mt.MotionCommand)
    command._device = torch.device("cpu")
    command.last_step_timing_ms = {}
    command.time_steps = torch.tensor([4, 0, 9], dtype=torch.int32)
    command.tensor_sampler = mt.TensorMotionSampler(
        mode="adaptive",
        num_envs=3,
        num_frames=11,
        clip_offsets=torch.tensor([0], dtype=torch.int64),
        clip_end_frames=torch.tensor([100], dtype=torch.int32),
        bin_count=1,
        adaptive_lambda=0.8,
        adaptive_kernel_size=1,
        adaptive_uniform_ratio=0.1,
        adaptive_alpha=0.001,
        start_ratio=0.0,
        initial_frames=torch.tensor([4, 0, 9], dtype=torch.int32),
        initial_clip_end_frames=torch.tensor([100, 100, 100], dtype=torch.int32),
        device=torch.device("cpu"),
    )
    command.tensor_sampler.current_frames.copy_(command.time_steps)
    command.current_clip_end_frames = torch.tensor([90, 90, 90], dtype=torch.int32)
    command._clip_offsets_torch = torch.tensor([0], dtype=torch.int64)
    command._clip_end_frames_torch = torch.tensor([100], dtype=torch.int64)
    command._tensor_all_rows = torch.arange(3, dtype=torch.int64)
    command.cfg = SimpleNamespace(params=SimpleNamespace(truncate_on_clip_end=True))
    command._tensor_post_compute_env_ids = None
    command._tensor_resample_ingested = None
    command._env = SimpleNamespace(
        termination_manager=SimpleNamespace(terminated=torch.tensor([False, False, False])),
        reset_buf=torch.tensor([False, False, False]),
    )
    captured: dict[str, Any] = {}
    command._motion_packet = lambda frames: frames

    def _capture_ingest(rows: torch.Tensor, packet: torch.Tensor) -> None:
        captured["rows"] = rows
        captured["frames"] = packet

    command._ingest_motion_packet = _capture_ingest
    command._update_command(None)

    torch.testing.assert_close(captured["rows"], torch.arange(3, dtype=torch.int64))
    torch.testing.assert_close(
        captured["frames"].to(torch.int64), torch.tensor([5, 1, 10], dtype=torch.int64)
    )


def test_tensor_command_syncs_sampler_mirrors_on_selected_rows_only() -> None:
    command = mt.MotionCommand.__new__(mt.MotionCommand)
    command._device = torch.device("cpu")
    command._tensor_all_rows = torch.arange(3, dtype=torch.int64)
    command._tensor_post_compute_env_ids = torch.tensor([0, 2], dtype=torch.int64)
    command.time_steps = torch.tensor([4, 20, 9], dtype=torch.int32)
    command.current_clip_end_frames = torch.tensor([90, 90, 90], dtype=torch.int32)
    command.tensor_sampler = mt.TensorMotionSampler(
        mode="adaptive",
        num_envs=3,
        num_frames=11,
        clip_offsets=torch.tensor([0], dtype=torch.int64),
        clip_end_frames=torch.tensor([100], dtype=torch.int32),
        bin_count=1,
        adaptive_lambda=0.8,
        adaptive_kernel_size=1,
        adaptive_uniform_ratio=0.1,
        adaptive_alpha=0.001,
        start_ratio=0.0,
        initial_frames=torch.tensor([7, 0, 8], dtype=torch.int32),
        initial_clip_end_frames=torch.tensor([100, 100, 100], dtype=torch.int32),
        device=torch.device("cpu"),
    )

    command._sync_tensor_sampler_state(torch.tensor([0, 2], dtype=torch.int64))

    torch.testing.assert_close(command.time_steps, torch.tensor([7, 20, 8], dtype=torch.int32))
    torch.testing.assert_close(
        command.current_clip_end_frames, torch.tensor([100, 90, 100], dtype=torch.int32)
    )


def test_tensor_clip_wrap_fails_closed_before_resampling_without_owner_transaction() -> None:
    command = mt.MotionCommand.__new__(mt.MotionCommand)
    command._device = torch.device("cpu")
    command.last_step_timing_ms = {}
    command.time_steps = torch.tensor([4, 0, 10], dtype=torch.int32)
    command.tensor_sampler = mt.TensorMotionSampler(
        mode="adaptive",
        num_envs=3,
        num_frames=11,
        clip_offsets=torch.tensor([0], dtype=torch.int64),
        clip_end_frames=torch.tensor([10], dtype=torch.int32),
        bin_count=1,
        adaptive_lambda=0.8,
        adaptive_kernel_size=1,
        adaptive_uniform_ratio=0.1,
        adaptive_alpha=0.001,
        start_ratio=0.0,
        initial_frames=torch.tensor([4, 0, 10], dtype=torch.int32),
        initial_clip_end_frames=torch.tensor([10, 10, 10], dtype=torch.int32),
        device=torch.device("cpu"),
    )
    command.tensor_sampler.current_frames.copy_(command.time_steps)
    command.current_clip_end_frames = torch.tensor([10, 10, 10], dtype=torch.int32)
    command._clip_offsets_torch = torch.tensor([0], dtype=torch.int64)
    command._clip_end_frames_torch = torch.tensor([10], dtype=torch.int64)
    command._tensor_all_rows = torch.arange(3, dtype=torch.int64)
    command.cfg = SimpleNamespace(params=SimpleNamespace(truncate_on_clip_end=False))
    command._tensor_post_compute_env_ids = None
    command._tensor_resample_ingested = None
    command.robot = SimpleNamespace(reset_state_tensor_active=False)
    command._env = SimpleNamespace(
        termination_manager=SimpleNamespace(terminated=torch.tensor([False, False, False])),
        reset_buf=torch.tensor([False, False, False]),
    )

    resampled: list[torch.Tensor] = []
    command._resample_command = lambda rows: resampled.append(rows)
    command._refresh_motion_torch = lambda *args, **kwargs: None

    with pytest.raises(RuntimeError, match="requires an active tensor reset transaction"):
        command._update_command(None)

    # Configuration is rejected before the sampler can advance the public frame
    # carrier or consume reset RNG/publish a replacement packet.
    assert resampled == []
    torch.testing.assert_close(command.time_steps, torch.tensor([4, 0, 10], dtype=torch.int32))
    torch.testing.assert_close(
        command.tensor_sampler.current_frames, torch.tensor([4, 0, 10], dtype=torch.int32)
    )


def test_motion_feature_layout_is_cached_without_recomputing_shapes() -> None:
    command = mt.MotionCommand.__new__(mt.MotionCommand)
    command._motion_feature_layout = None
    command.motion = SimpleNamespace(num_joints=3)
    command.cfg = SimpleNamespace(body_names=("pelvis", "torso"))

    calls = 0

    def counted_tails(_: object) -> dict[str, tuple[int, ...]]:
        nonlocal calls
        calls += 1
        return {
            "joint_pos": (command.motion.num_joints,),
            "joint_vel": (command.motion.num_joints,),
            "body_pos_w": (len(command.cfg.body_names), 3),
            "body_quat_w": (len(command.cfg.body_names), 4),
            "body_lin_vel_w": (len(command.cfg.body_names), 3),
            "body_ang_vel_w": (len(command.cfg.body_names), 3),
        }

    original = mt.MotionCommand._motion_feature_tail_shapes
    mt.MotionCommand._motion_feature_tail_shapes = counted_tails
    try:
        first_tails, first_offsets = command._cached_motion_feature_shapes()
        second_tails, second_offsets = command._cached_motion_feature_shapes()
    finally:
        mt.MotionCommand._motion_feature_tail_shapes = original

    assert calls == 1
    assert first_tails is second_tails
    assert first_offsets is second_offsets
    assert first_offsets == {
        "joint_pos": (0, 3),
        "joint_vel": (3, 6),
        "body_pos_w": (6, 12),
        "body_quat_w": (12, 20),
        "body_lin_vel_w": (20, 26),
        "body_ang_vel_w": (26, 32),
    }


def test_motion_reset_values_kernel_matches_eager_and_reuses_scratch() -> None:
    rng = np.random.default_rng(1811)
    count, num_joints, num_bodies = 39, 29, 14
    packet_joint_pos = rng.standard_normal((count, num_joints), dtype=np.float32)
    packet_joint_vel = rng.standard_normal((count, num_joints), dtype=np.float32)
    packet_body_pos = rng.standard_normal((count, num_bodies, 3), dtype=np.float32)
    packet_body_quat = _unit_quat(rng.standard_normal((count, num_bodies, 4), dtype=np.float32))
    packet_body_lin = rng.standard_normal((count, num_bodies, 3), dtype=np.float32)
    packet_body_ang = rng.standard_normal((count, num_bodies, 3), dtype=np.float32)
    origins = rng.standard_normal((count, 3), dtype=np.float32)
    pose = rng.uniform(-0.5, 0.5, (count, 6)).astype(np.float32)
    velocity = rng.standard_normal((count, 6), dtype=np.float32)
    joint_noise = rng.standard_normal((count, num_joints), dtype=np.float32) * 0.01
    soft_limits = np.stack(
        (
            rng.uniform(-2.0, -1.0, num_joints),
            rng.uniform(1.0, 2.0, num_joints),
        ),
        axis=-1,
    ).astype(np.float32)

    def tensor(value: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(value.copy())

    expected_joint = packet_joint_pos + joint_noise
    np.clip(expected_joint, soft_limits[:, 0], soft_limits[:, 1], out=expected_joint)
    expected_pose_quat = np_quat_from_euler_xyz(pose[:, 3], pose[:, 4], pose[:, 5])
    expected_root_quat = np_quat_mul_batched(expected_pose_quat, packet_body_quat[:, 0])
    expected_root = np.concatenate(
        (
            packet_body_pos[:, 0] + origins + pose[:, :3],
            expected_root_quat,
            packet_body_lin[:, 0] + velocity[:, :3],
            packet_body_ang[:, 0] + velocity[:, 3:],
        ),
        axis=-1,
    )

    joint_scratch = torch.full((256, num_joints), torch.nan, dtype=torch.float32)
    root_scratch = torch.full((256, 13), torch.nan, dtype=torch.float32)
    for selected in (count, 17, count):
        mt._motion_reset_values_kernel(
            tensor(packet_joint_pos[:selected]),
            tensor(packet_joint_vel[:selected]),
            tensor(packet_body_pos[:selected]),
            tensor(packet_body_quat[:selected]),
            tensor(packet_body_lin[:selected]),
            tensor(packet_body_ang[:selected]),
            tensor(origins[:selected]),
            tensor(pose[:selected]),
            tensor(velocity[:selected]),
            tensor(joint_noise[:selected]),
            tensor(soft_limits),
            joint_scratch[:selected],
            root_scratch[:selected],
        )

        torch.testing.assert_close(
            joint_scratch[:selected],
            torch.from_numpy(expected_joint[:selected]),
            rtol=2e-6,
            atol=2e-6,
        )
        torch.testing.assert_close(
            root_scratch[:selected],
            torch.from_numpy(expected_root[:selected]),
            rtol=2e-6,
            atol=2e-6,
        )
        # The unused suffix must never be exposed through the selected prefix.
        assert bool(torch.isfinite(joint_scratch[selected:]).all()) is False
        assert bool(torch.isfinite(root_scratch[selected:]).all()) is False

    compiled = mt._bind_compiled_motion_reset_values()
    if compiled is not mt._motion_reset_values_kernel:
        compiled(
            tensor(packet_joint_pos),
            tensor(packet_joint_vel),
            tensor(packet_body_pos),
            tensor(packet_body_quat),
            tensor(packet_body_lin),
            tensor(packet_body_ang),
            tensor(origins),
            tensor(pose),
            tensor(velocity),
            tensor(joint_noise),
            tensor(soft_limits),
            joint_scratch[:count],
            root_scratch[:count],
        )
        torch.testing.assert_close(
            joint_scratch[:count], torch.from_numpy(expected_joint), rtol=2e-6, atol=2e-6
        )
        torch.testing.assert_close(
            root_scratch[:count], torch.from_numpy(expected_root), rtol=2e-6, atol=2e-6
        )
