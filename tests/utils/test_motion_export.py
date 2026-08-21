"""Tests for the MuJoCo tracking FK export shared by scripts/motion converters."""

from __future__ import annotations

import numpy as np
import pytest

from unilab.assets import ASSETS_ROOT_PATH
from unilab.base.backend import compute_tracking_fk, get_named_bodies


def _g1_scene() -> str:
    return str(ASSETS_ROOT_PATH / "robots" / "g1" / "scene_flat.xml")


_G1_JOINT_NAMES = [
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
]


def test_get_named_bodies_matches_g1_scene() -> None:
    ids, names = get_named_bodies(_g1_scene())
    assert ids == list(range(1, len(names) + 1))
    assert "pelvis" in names
    assert len(names) == 30


def test_compute_tracking_fk_shapes_and_joint_roundtrip() -> None:
    pytest.importorskip("mujoco")
    rng = np.random.default_rng(0)
    num_frames = 4
    num_joints = len(_G1_JOINT_NAMES)

    base_poss = rng.standard_normal((num_frames, 3)).astype(np.float32) * 0.05
    base_poss[:, 2] += 0.8
    base_rots = np.zeros((num_frames, 4), dtype=np.float32)
    base_rots[:, 0] = 1.0
    base_lin_vels = np.zeros((num_frames, 3), dtype=np.float32)
    base_ang_vels = np.zeros((num_frames, 3), dtype=np.float32)
    dof_poss = rng.standard_normal((num_frames, num_joints)).astype(np.float32) * 0.1
    dof_vels = rng.standard_normal((num_frames, num_joints)).astype(np.float32) * 0.1

    arrays = compute_tracking_fk(
        _g1_scene(),
        joint_names=_G1_JOINT_NAMES,
        base_poss=base_poss,
        base_rots=base_rots,
        base_lin_vels=base_lin_vels,
        base_ang_vels=base_ang_vels,
        dof_poss=dof_poss,
        dof_vels=dof_vels,
    )

    _, body_names = get_named_bodies(_g1_scene())
    num_bodies = len(body_names) + 1  # + implicit world body 0

    assert arrays["joint_pos"].shape == (num_frames, num_joints)
    assert arrays["joint_vel"].shape == (num_frames, num_joints)
    assert arrays["body_pos_w"].shape == (num_frames, num_bodies, 3)
    assert arrays["body_quat_w"].shape == (num_frames, num_bodies, 4)
    assert arrays["body_lin_vel_w"].shape == (num_frames, num_bodies, 3)
    assert arrays["body_ang_vel_w"].shape == (num_frames, num_bodies, 3)
    for array in arrays.values():
        assert array.dtype == np.float32

    # Joint states written into qpos/qvel must read back exactly.
    np.testing.assert_array_equal(arrays["joint_pos"], dof_poss)
    np.testing.assert_array_equal(arrays["joint_vel"], dof_vels)

    # World body stays at the origin; pelvis tracks the commanded root state.
    np.testing.assert_array_equal(arrays["body_pos_w"][:, 0], 0.0)
    pelvis_id = body_names.index("pelvis") + 1
    np.testing.assert_allclose(arrays["body_pos_w"][:, pelvis_id], base_poss, atol=1e-5)
    np.testing.assert_allclose(arrays["body_quat_w"][:, pelvis_id], base_rots, atol=1e-5)


def test_compute_tracking_fk_unknown_joint_raises() -> None:
    pytest.importorskip("mujoco")
    base = np.zeros((1, 3), dtype=np.float32)
    quat = np.zeros((1, 4), dtype=np.float32)
    quat[:, 0] = 1.0
    with pytest.raises(ValueError, match="not_a_joint"):
        compute_tracking_fk(
            _g1_scene(),
            joint_names=["not_a_joint"],
            base_poss=base,
            base_rots=quat,
            base_lin_vels=base,
            base_ang_vels=base,
            dof_poss=np.zeros((1, 1), dtype=np.float32),
            dof_vels=np.zeros((1, 1), dtype=np.float32),
        )
