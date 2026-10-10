from __future__ import annotations

import numpy as np
import torch

from unilab.tasks.motion_tracking.common.motion_loader import MotionLoader


def _write_motion_npz(
    path,
    *,
    base_value: float,
    num_frames: int,
    num_joints: int = 2,
    num_bodies: int = 3,
    fps: int = 30,
    with_torque: bool = False,
    torque_limit: tuple[float, ...] | None = None,
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

    extra = {}
    if with_torque:
        extra["joint_torque"] = (joint_pos - base_value).astype(np.float32) + 7.0
        extra["joint_torque_limit"] = np.asarray(
            torque_limit if torque_limit is not None else (10.0,) * num_joints,
            dtype=np.float32,
        )

    np.savez(
        path,
        fps=np.array([fps], dtype=np.int32),
        joint_pos=joint_pos.astype(np.float32),
        joint_vel=joint_vel.astype(np.float32),
        body_pos_w=body_pos_w.astype(np.float32),
        body_quat_w=body_quat_w.astype(np.float32),
        body_lin_vel_w=body_lin_vel_w.astype(np.float32),
        body_ang_vel_w=body_ang_vel_w.astype(np.float32),
        **extra,
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


def test_motion_loader_publishes_immutable_torch_feature_table(tmp_path):
    motion_a = tmp_path / "motion_a.npz"
    motion_b = tmp_path / "motion_b.npz"
    _write_motion_npz(motion_a, base_value=0.0, num_frames=2)
    _write_motion_npz(motion_b, base_value=10.0, num_frames=3)
    loader = MotionLoader([str(motion_a), str(motion_b)])

    features = loader.motion_features_torch(torch.device("cpu"))

    expected_width = 2 * loader.num_joints + loader.num_bodies * (3 + 4 + 3 + 3)
    assert features.shape == (loader.num_frames, expected_width)
    assert features.dtype == torch.float32
    assert features.is_contiguous()
    torch.testing.assert_close(features[:, : loader.num_joints], torch.from_numpy(loader.joint_pos))


def test_motion_loader_rejects_mismatched_multi_clip_metadata(tmp_path):
    motion_a = tmp_path / "motion_a.npz"
    motion_b = tmp_path / "motion_b.npz"
    _write_motion_npz(motion_a, base_value=0.0, num_frames=2, fps=30)
    _write_motion_npz(motion_b, base_value=10.0, num_frames=3, fps=60)

    with np.testing.assert_raises(ValueError):
        MotionLoader([str(motion_a), str(motion_b)])


def test_motion_loader_reads_optional_joint_torque(tmp_path):
    motion = tmp_path / "motion_torque.npz"
    _write_motion_npz(motion, base_value=0.0, num_frames=4, num_joints=2, with_torque=True)

    loader = MotionLoader(str(motion))

    assert loader.has_joint_torque
    assert loader.joint_torque.shape == (4, 2)
    np.testing.assert_allclose(loader.joint_torque_limit, np.array([10.0, 10.0]))
    np.testing.assert_allclose(loader.joint_torque[2], np.array([9.0, 9.0]))

    sampled = loader.get_motion_at_frame(np.array([0, 3], dtype=np.int32))
    np.testing.assert_allclose(sampled.joint_torque[:, 0], np.array([7.0, 10.0]))

    buffer = loader.make_motion_data_buffer(2)
    assert buffer.joint_torque is not None and buffer.joint_torque.shape == (2, 2)
    out = loader.get_motion_at_frame(np.array([1, 2], dtype=np.int32), out=buffer)
    np.testing.assert_allclose(out.joint_torque[:, 0], np.array([8.0, 9.0]))


def test_motion_loader_without_torque_stays_torque_free(tmp_path):
    motion = tmp_path / "motion_plain.npz"
    _write_motion_npz(motion, base_value=0.0, num_frames=4)

    loader = MotionLoader(str(motion))

    assert not loader.has_joint_torque
    assert loader.joint_torque is None
    assert loader.joint_torque_limit is None

    sampled = loader.get_motion_at_frame(np.array([0, 3], dtype=np.int32))
    assert sampled.joint_torque is None

    buffer = loader.make_motion_data_buffer(2)
    assert buffer.joint_torque is None
    out = loader.get_motion_at_frame(np.array([1, 2], dtype=np.int32), out=buffer)
    assert out.joint_torque is None


def test_motion_loader_mixed_clip_torque_presence_warns_and_disables(tmp_path):
    motion_a = tmp_path / "motion_a.npz"
    motion_b = tmp_path / "motion_b.npz"
    _write_motion_npz(motion_a, base_value=0.0, num_frames=2, with_torque=True)
    _write_motion_npz(motion_b, base_value=10.0, num_frames=3, with_torque=False)

    import pytest

    with pytest.warns(UserWarning, match="torque"):
        loader = MotionLoader([str(motion_a), str(motion_b)])

    assert not loader.has_joint_torque
    assert loader.joint_torque is None
    assert loader.joint_torque_limit is None


def test_motion_loader_multi_clip_torque_concatenates(tmp_path):
    motion_a = tmp_path / "motion_a.npz"
    motion_b = tmp_path / "motion_b.npz"
    _write_motion_npz(motion_a, base_value=0.0, num_frames=2, with_torque=True)
    _write_motion_npz(motion_b, base_value=10.0, num_frames=3, with_torque=True)

    loader = MotionLoader([str(motion_a), str(motion_b)])

    assert loader.has_joint_torque
    assert loader.joint_torque.shape == (5, 2)
    np.testing.assert_allclose(loader.joint_torque[:, 0], np.array([7.0, 8.0, 7.0, 8.0, 9.0]))


def test_motion_loader_rejects_inconsistent_clip_torque_limits(tmp_path):
    motion_a = tmp_path / "motion_a.npz"
    motion_b = tmp_path / "motion_b.npz"
    _write_motion_npz(motion_a, base_value=0.0, num_frames=2, with_torque=True)
    _write_motion_npz(
        motion_b, base_value=10.0, num_frames=3, with_torque=True, torque_limit=(10.0, 20.0)
    )

    with np.testing.assert_raises(ValueError):
        MotionLoader([str(motion_a), str(motion_b)])


def test_motion_loader_rejects_partial_torque_key_set(tmp_path):
    motion = tmp_path / "motion_partial.npz"
    _write_motion_npz(motion, base_value=0.0, num_frames=2, with_torque=True)

    with np.load(motion) as data:
        payload = {key: data[key] for key in data.files if key != "joint_torque_limit"}
    np.savez(motion, **payload)

    with np.testing.assert_raises(ValueError):
        MotionLoader(str(motion))


def test_motion_loader_rejects_invalid_torque_limit(tmp_path):
    motion = tmp_path / "motion_bad_limit.npz"
    _write_motion_npz(
        motion, base_value=0.0, num_frames=2, with_torque=True, torque_limit=(10.0, 0.0)
    )

    with np.testing.assert_raises(ValueError):
        MotionLoader(str(motion))
