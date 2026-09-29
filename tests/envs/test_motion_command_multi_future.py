from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest

import unilab.tasks.motion_tracking.common.manager_terms as manager_terms
from unilab.tasks.motion_tracking.common.manager_terms import motion_command_multi_future
from unilab.utils.rotation import np_matrix_first_two_cols_from_quat

_NUM_ENVS = 3
_NUM_FRAMES = 20
_NUM_JOINTS = 2
_ANCHOR_IDX = 1


def _stub_command() -> SimpleNamespace:
    rng = np.random.default_rng(0)
    motion = SimpleNamespace(
        joint_pos=rng.normal(size=(_NUM_FRAMES, _NUM_JOINTS)).astype(np.float32),
        joint_vel=rng.normal(size=(_NUM_FRAMES, _NUM_JOINTS)).astype(np.float32),
        # Two bodies; only the anchor body is consumed.
        body_quat_w=rng.normal(size=(_NUM_FRAMES, 2, 4)).astype(np.float32),
    )
    motion.body_quat_w /= np.linalg.norm(motion.body_quat_w, axis=-1, keepdims=True)
    robot_quat = rng.normal(size=(_NUM_ENVS, 4)).astype(np.float32)
    robot_quat /= np.linalg.norm(robot_quat, axis=-1, keepdims=True)
    return SimpleNamespace(
        time_steps=np.array([0, 18, 5], dtype=np.int32),
        sampler=SimpleNamespace(current_clip_end_frames=np.array([9, 19, 15], dtype=np.int32)),
        motion=motion,
        anchor_body_idx=_ANCHOR_IDX,
        robot_anchor_quat_w=robot_quat,
    )


def _run(command: Any, future_steps: list[int]) -> np.ndarray:
    env = SimpleNamespace(num_envs=_NUM_ENVS)
    original = manager_terms._command
    manager_terms._command = lambda env, command_name: command
    try:
        return motion_command_multi_future(cast(Any, env), "motion", future_steps)
    finally:
        manager_terms._command = original


def test_motion_command_multi_future_layout_and_clamping() -> None:
    command = _stub_command()
    steps = [0, 5, 10]
    obs = _run(command, steps)

    frame_width = 2 * _NUM_JOINTS + 6
    assert obs.shape == (_NUM_ENVS, len(steps) * frame_width)
    assert obs.dtype == np.float32
    assert np.isfinite(obs).all()

    frames = np.array(
        [
            [0, 5, 9],  # clamped at clip end 9
            [18, 19, 19],  # clamped at clip end 19
            [5, 10, 15],
        ]
    )
    obs = obs.reshape(_NUM_ENVS, len(steps), frame_width)
    for env_idx in range(_NUM_ENVS):
        for step_idx in range(len(steps)):
            frame = frames[env_idx, step_idx]
            block = obs[env_idx, step_idx]
            np.testing.assert_allclose(
                block[:_NUM_JOINTS], command.motion.joint_pos[frame], atol=1e-6
            )
            np.testing.assert_allclose(
                block[_NUM_JOINTS : 2 * _NUM_JOINTS],
                command.motion.joint_vel[frame],
                atol=1e-6,
            )


def test_motion_command_multi_future_anchor_ori_matches_kernel_convention() -> None:
    command = _stub_command()
    obs = _run(command, [0]).reshape(_NUM_ENVS, -1)
    obs_6d = obs[:, 2 * _NUM_JOINTS :]

    robot = command.robot_anchor_quat_w
    conj = robot * np.array([1.0, -1.0, -1.0, -1.0], dtype=np.float32)
    motion_quat = command.motion.body_quat_w[command.time_steps, _ANCHOR_IDX]
    w1, x1, y1, z1 = conj[:, 0], conj[:, 1], conj[:, 2], conj[:, 3]
    w2, x2, y2, z2 = motion_quat[:, 0], motion_quat[:, 1], motion_quat[:, 2], motion_quat[:, 3]
    rel = np.stack(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        axis=-1,
    )
    np.testing.assert_allclose(obs_6d, np_matrix_first_two_cols_from_quat(rel), atol=1e-5)


def test_motion_command_multi_future_rejects_bad_steps() -> None:
    command = _stub_command()
    with pytest.raises(ValueError, match="future_steps"):
        _run(command, [])
    with pytest.raises(ValueError, match="future_steps"):
        _run(command, [0, -1])
