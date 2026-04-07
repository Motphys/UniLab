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


def test_motion_sampler_start_mode_uses_clip_starts_for_multi_clip_loader(tmp_path):
    motion_a = tmp_path / "motion_a.npz"
    motion_b = tmp_path / "motion_b.npz"
    _write_motion_npz(motion_a, base_value=0.0, num_frames=2)
    _write_motion_npz(motion_b, base_value=10.0, num_frames=3)

    np.random.seed(0)
    loader = MotionLoader([str(motion_a), str(motion_b)])
    sampler = MotionSampler(loader, mode="start", num_envs=16)

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
