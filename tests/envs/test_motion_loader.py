from __future__ import annotations

import numpy as np

from unilab.envs.motion_tracking.g1.motion_loader import MotionLoader, MotionSampler


def _write_motion_npz(
    path,
    *,
    base_value: float,
    num_frames: int,
    num_joints: int = 2,
    num_bodies: int = 3,
    fps: int = 30,
) -> None:
    frame_values = np.arange(num_frames, dtype=np.float32)[:, None]
    joint_pos = base_value + np.repeat(frame_values, num_joints, axis=1)
    joint_vel = joint_pos + 100.0

    body_frame_values = np.arange(num_frames, dtype=np.float32)[:, None, None]
    body_pos_w = (
        base_value + np.ones((num_frames, num_bodies, 3), dtype=np.float32) * body_frame_values
    )
    body_quat_w = np.zeros((num_frames, num_bodies, 4), dtype=np.float32)
    body_quat_w[:, :, 0] = 1.0
    body_quat_w[:, :, 1] = base_value + body_frame_values[:, :, 0]
    body_lin_vel_w = body_pos_w + 10.0
    body_ang_vel_w = body_pos_w + 20.0

    np.savez(
        path,
        fps=np.array([fps], dtype=np.int32),
        joint_pos=joint_pos.astype(np.float32),
        joint_vel=joint_vel.astype(np.float32),
        body_pos_w=body_pos_w.astype(np.float32),
        body_quat_w=body_quat_w.astype(np.float32),
        body_lin_vel_w=body_lin_vel_w.astype(np.float32),
        body_ang_vel_w=body_ang_vel_w.astype(np.float32),
    )


def _write_box_motion_npz(
    path,
    *,
    base_value: float,
    num_frames: int,
    num_joints: int = 2,
    num_bodies: int = 3,
    fps: int = 30,
) -> None:
    frame_values = np.arange(num_frames, dtype=np.float32)[:, None]
    robot_joint_pos = base_value + np.repeat(frame_values, num_joints, axis=1)
    robot_joint_vel = robot_joint_pos + 100.0

    body_frame_values = np.arange(num_frames, dtype=np.float32)[:, None, None]
    body_pos_w = (
        base_value + np.ones((num_frames, num_bodies, 3), dtype=np.float32) * body_frame_values
    )
    body_quat_w = np.zeros((num_frames, num_bodies, 4), dtype=np.float32)
    body_quat_w[:, :, 0] = 1.0
    body_lin_vel_w = body_pos_w + 10.0
    body_ang_vel_w = body_pos_w + 20.0

    object_pos_w = np.concatenate(
        [
            base_value + frame_values,
            base_value + frame_values + 1.0,
            base_value + frame_values + 2.0,
        ],
        axis=1,
    ).astype(np.float32)
    object_quat_w = np.zeros((num_frames, 4), dtype=np.float32)
    object_quat_w[:, 0] = 1.0
    object_quat_w[:, 1] = base_value + np.arange(num_frames, dtype=np.float32)
    object_lin_vel_w = object_pos_w + 10.0
    object_ang_vel_w = object_pos_w + 20.0

    joint_pos = np.concatenate([robot_joint_pos, object_pos_w, object_quat_w], axis=1)
    joint_vel = np.concatenate([robot_joint_vel, object_lin_vel_w, object_ang_vel_w], axis=1)

    np.savez(
        path,
        fps=np.array([fps], dtype=np.int32),
        joint_names=np.array([f"joint_{i}" for i in range(num_joints)]),
        joint_pos=joint_pos.astype(np.float32),
        joint_vel=joint_vel.astype(np.float32),
        body_pos_w=body_pos_w.astype(np.float32),
        body_quat_w=body_quat_w.astype(np.float32),
        body_lin_vel_w=body_lin_vel_w.astype(np.float32),
        body_ang_vel_w=body_ang_vel_w.astype(np.float32),
        object_pos_w=object_pos_w,
        object_quat_w=object_quat_w,
        object_lin_vel_w=object_lin_vel_w,
        object_ang_vel_w=object_ang_vel_w,
    )


def test_motion_loader_accepts_single_path_or_path_list(tmp_path):
    motion_a = tmp_path / "motion_a.npz"
    motion_b = tmp_path / "motion_b.npz"
    _write_motion_npz(motion_a, base_value=0.0, num_frames=2)
    _write_motion_npz(motion_b, base_value=10.0, num_frames=3)

    single_loader = MotionLoader(str(motion_a))
    assert single_loader.num_clips == 1
    assert single_loader.num_frames == 2
    np.testing.assert_array_equal(single_loader.clip_offsets, np.array([0], dtype=np.int32))
    np.testing.assert_array_equal(single_loader.clip_end_frames, np.array([1], dtype=np.int32))

    multi_loader = MotionLoader([str(motion_a), str(motion_b)])
    assert multi_loader.num_clips == 2
    assert multi_loader.num_frames == 5
    np.testing.assert_array_equal(multi_loader.clip_lengths, np.array([2, 3], dtype=np.int32))
    np.testing.assert_array_equal(multi_loader.clip_offsets, np.array([0, 2], dtype=np.int32))
    np.testing.assert_array_equal(multi_loader.clip_end_frames, np.array([1, 4], dtype=np.int32))

    sampled = multi_loader.get_motion_at_frame(np.array([0, 1, 2, 4], dtype=np.int32))
    np.testing.assert_array_equal(sampled.joint_pos[:, 0], np.array([0.0, 1.0, 10.0, 12.0]))


def test_motion_loader_rejects_mismatched_multi_clip_metadata(tmp_path):
    motion_a = tmp_path / "motion_a.npz"
    motion_b = tmp_path / "motion_b.npz"
    _write_motion_npz(motion_a, base_value=0.0, num_frames=2, fps=30)
    _write_motion_npz(motion_b, base_value=10.0, num_frames=3, fps=60)

    with np.testing.assert_raises(ValueError):
        MotionLoader([str(motion_a), str(motion_b)])


def test_motion_sampler_start_mode_preserves_global_zero_frame(tmp_path):
    motion_a = tmp_path / "motion_a.npz"
    motion_b = tmp_path / "motion_b.npz"
    _write_motion_npz(motion_a, base_value=0.0, num_frames=2)
    _write_motion_npz(motion_b, base_value=10.0, num_frames=3)

    np.random.seed(0)
    loader = MotionLoader([str(motion_a), str(motion_b)])
    sampler = MotionSampler(loader, mode="start", num_envs=16)

    env_ids = np.arange(16, dtype=np.int32)
    frames = sampler.sample_frames(env_ids)

    np.testing.assert_array_equal(frames, np.zeros(16, dtype=np.int32))
    np.testing.assert_array_equal(sampler.current_clip_indices, np.zeros(16, dtype=np.int32))
    np.testing.assert_array_equal(sampler.current_clip_end_frames, np.full(16, 1, dtype=np.int32))


def test_motion_sampler_clip_start_mode_uses_clip_starts_for_multi_clip_loader(tmp_path):
    motion_a = tmp_path / "motion_a.npz"
    motion_b = tmp_path / "motion_b.npz"
    _write_motion_npz(motion_a, base_value=0.0, num_frames=2)
    _write_motion_npz(motion_b, base_value=10.0, num_frames=3)

    np.random.seed(0)
    loader = MotionLoader([str(motion_a), str(motion_b)])
    sampler = MotionSampler(loader, mode="clip_start", num_envs=16)

    env_ids = np.arange(16, dtype=np.int32)
    frames = sampler.sample_frames(env_ids)

    assert np.isin(frames, loader.clip_offsets).all()
    np.testing.assert_array_equal(
        sampler.current_clip_end_frames, loader.clip_end_frames[sampler.current_clip_indices]
    )


def test_motion_sampler_step_respects_current_clip_end(tmp_path):
    motion_a = tmp_path / "motion_a.npz"
    motion_b = tmp_path / "motion_b.npz"
    _write_motion_npz(motion_a, base_value=0.0, num_frames=2)
    _write_motion_npz(motion_b, base_value=10.0, num_frames=3)

    loader = MotionLoader([str(motion_a), str(motion_b)])
    sampler = MotionSampler(loader, mode="uniform", num_envs=2)

    sampler.current_frames[:] = np.array([1, 3], dtype=np.int32)
    sampler.current_clip_indices[:] = np.array([0, 1], dtype=np.int32)
    sampler.current_clip_end_frames[:] = np.array([1, 4], dtype=np.int32)

    done_env_ids = sampler.step()
    np.testing.assert_array_equal(done_env_ids, np.array([0], dtype=np.int64))
    np.testing.assert_array_equal(sampler.current_frames, np.array([2, 4], dtype=np.int32))


def test_box_motion_loader_reads_object_state_and_trims_robot_joints(tmp_path):
    from unilab.envs.motion_tracking.g1.motion_box_loader import BoxMotionLoader

    motion = tmp_path / "motion_box.npz"
    _write_box_motion_npz(motion, base_value=1.0, num_frames=2, num_joints=2)

    loader = BoxMotionLoader(str(motion))

    assert loader.has_object is True
    assert loader.num_joints == 2
    assert loader.joint_pos.shape == (2, 2)
    assert loader.joint_vel.shape == (2, 2)
    np.testing.assert_allclose(loader.object_pos_w, np.array([[1.0, 2.0, 3.0], [2.0, 3.0, 4.0]]))

    sampled = loader.get_motion_at_frame(np.array([0, 1], dtype=np.int32))
    np.testing.assert_allclose(sampled.joint_pos, np.array([[1.0, 1.0], [2.0, 2.0]]))
    np.testing.assert_allclose(sampled.joint_vel, np.array([[101.0, 101.0], [102.0, 102.0]]))
    np.testing.assert_allclose(
        sampled.object_quat_w, np.array([[1.0, 1.0, 0.0, 0.0], [1.0, 2.0, 0.0, 0.0]])
    )


def test_box_motion_loader_rejects_partial_object_key_sets(tmp_path):
    from unilab.envs.motion_tracking.g1.motion_box_loader import BoxMotionLoader

    motion = tmp_path / "motion_box_missing_keys.npz"
    _write_box_motion_npz(motion, base_value=1.0, num_frames=2, num_joints=2)

    with np.load(motion) as data:
        payload = {key: data[key] for key in data.files if key != "object_ang_vel_w"}
    np.savez(motion, **payload)

    with np.testing.assert_raises(ValueError):
        BoxMotionLoader(str(motion))


def test_box_motion_loader_rejects_multi_clip_object_presence_mismatch(tmp_path):
    from unilab.envs.motion_tracking.g1.motion_box_loader import BoxMotionLoader

    motion_without_object = tmp_path / "motion_without_object.npz"
    motion_with_object = tmp_path / "motion_with_object.npz"
    _write_motion_npz(motion_without_object, base_value=0.0, num_frames=2)
    _write_box_motion_npz(motion_with_object, base_value=10.0, num_frames=2, num_joints=2)

    with np.testing.assert_raises(ValueError):
        BoxMotionLoader([str(motion_without_object), str(motion_with_object)])
