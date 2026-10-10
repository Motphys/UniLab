"""Shared motion loading and sampling for motion-tracking tasks."""

from __future__ import annotations

import math
import warnings
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal, cast

import numpy as np
import torch

from unilab.assets.hub import resolve_motion_files
from unilab.utils.rotation import np_quat_angular_velocity, np_quat_ensure_continuity


@dataclass
class MotionData:
    """Container for motion data at specific frame(s)."""

    # NumPy denotes the cold NPZ decode carrier; TensorMotionCommand stores
    # device-resident Torch carriers in the same owner-shaped container.
    joint_pos: np.ndarray | torch.Tensor  # (N, num_joints)
    joint_vel: np.ndarray | torch.Tensor  # (N, num_joints)
    body_pos_w: np.ndarray | torch.Tensor  # (N, num_bodies, 3)
    body_quat_w: np.ndarray | torch.Tensor  # (N, num_bodies, 4)
    body_lin_vel_w: np.ndarray | torch.Tensor  # (N, num_bodies, 3)
    body_ang_vel_w: np.ndarray | torch.Tensor  # (N, num_bodies, 3)
    joint_torque: np.ndarray | torch.Tensor | None = None  # optional torque carrier


def quat_slerp(q1: np.ndarray, q2: np.ndarray, t: float) -> np.ndarray:
    """Spherical linear interpolation between two quaternions (wxyz format).

    The computation runs in the input dtype; pass float64 arrays for a
    float64 interpolation path.
    """
    # Ensure shortest path
    dot = np.dot(q1, q2)
    if dot < 0:
        q2 = -q2
        dot = -dot

    # If quaternions are very close, use linear interpolation
    if dot > 0.9995:
        result = q1 + t * (q2 - q1)
        return result / np.linalg.norm(result)

    # Compute angle
    theta = np.arccos(np.clip(dot, -1, 1))
    sin_theta = np.sin(theta)

    # Compute interpolation weights
    w1 = np.sin((1 - t) * theta) / sin_theta
    w2 = np.sin(t * theta) / sin_theta

    return w1 * q1 + w2 * q2


@dataclass
class InterpolatedMotion:
    """Root/joint trajectory resampled to the output frame rate."""

    output_frames: int
    base_poss: np.ndarray  # (N, 3)
    base_rots: np.ndarray  # (N, 4), wxyz
    dof_poss: np.ndarray  # (N, num_joints)
    base_lin_vels: np.ndarray  # (N, 3)
    base_ang_vels: np.ndarray  # (N, 3)
    dof_vels: np.ndarray  # (N, num_joints)


def compute_motion_velocities(
    base_poss: np.ndarray,
    base_rots: np.ndarray,
    dof_poss: np.ndarray,
    dt: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Numerically differentiate a trajectory into base/dof velocities."""
    base_lin_vels = np.gradient(base_poss, dt, axis=0)
    dof_vels = np.gradient(dof_poss, dt, axis=0)
    base_ang_vels = np_quat_angular_velocity(base_rots, dt)
    return base_lin_vels, base_ang_vels, dof_vels


def compute_frame_blend(
    times: np.ndarray, duration: float, input_frames: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute frame indices and blend weights for interpolation."""
    phase = times / duration
    index_0 = np.floor(phase * (input_frames - 1)).astype(np.int32)
    index_1 = np.minimum(index_0 + 1, input_frames - 1)
    blend = phase * (input_frames - 1) - index_0
    return index_0, index_1, blend


def interpolate_motion(
    base_poss_input: np.ndarray,
    base_rots_input: np.ndarray,
    dof_poss_input: np.ndarray,
    *,
    input_fps: int,
    output_fps: int,
) -> InterpolatedMotion:
    """Resample a root+joint trajectory from ``input_fps`` to ``output_fps``.

    Positions and joint angles are linearly interpolated, root quaternions
    (wxyz) use slerp, and velocities are computed by numerical
    differentiation at the output rate.
    """
    input_dt = 1.0 / input_fps
    output_dt = 1.0 / output_fps
    input_frames = base_poss_input.shape[0]
    duration = (input_frames - 1) * input_dt

    times = np.arange(0, duration, output_dt, dtype=np.float32)
    output_frames = times.shape[0]
    index_0, index_1, blend = compute_frame_blend(times, duration, input_frames)

    # Linear interpolation for positions
    base_poss = (
        base_poss_input[index_0] * (1 - blend[:, None]) + base_poss_input[index_1] * blend[:, None]
    )

    # Spherical linear interpolation for quaternions
    base_rots = np.zeros((output_frames, 4), dtype=np.float32)
    for i in range(output_frames):
        base_rots[i] = quat_slerp(
            base_rots_input[index_0[i]], base_rots_input[index_1[i]], blend[i]
        )
    base_rots = np_quat_ensure_continuity(base_rots)

    # Linear interpolation for joint positions
    dof_poss = (
        dof_poss_input[index_0] * (1 - blend[:, None]) + dof_poss_input[index_1] * blend[:, None]
    )

    base_lin_vels, base_ang_vels, dof_vels = compute_motion_velocities(
        base_poss, base_rots, dof_poss, output_dt
    )
    return InterpolatedMotion(
        output_frames=output_frames,
        base_poss=base_poss,
        base_rots=base_rots,
        dof_poss=dof_poss,
        base_lin_vels=base_lin_vels,
        base_ang_vels=base_ang_vels,
        dof_vels=dof_vels,
    )


class MotionLoader:
    """Loads and provides access to motion data from NPZ files."""

    def __init__(self, motion_file: str | Sequence[str], body_indices: np.ndarray | None = None):
        """Initialize motion loader.

        Args:
            motion_file: Path to one NPZ file, or a sequence of NPZ files
            body_indices: Optional indices into the NPZ body axis. The exported
                motion files currently keep MuJoCo body-id layout, so these
                indices are expected to follow that convention.
        """
        motion_file = resolve_motion_files(motion_file)
        self.motion_files = self._normalize_motion_files(motion_file)

        joint_pos_list: list[np.ndarray] = []
        joint_vel_list: list[np.ndarray] = []
        body_pos_list: list[np.ndarray] = []
        body_quat_list: list[np.ndarray] = []
        body_lin_vel_list: list[np.ndarray] = []
        body_ang_vel_list: list[np.ndarray] = []
        joint_torque_list: list[np.ndarray] = []
        joint_torque_limit_list: list[np.ndarray] = []
        clip_lengths: list[int] = []

        self.fps = 0
        self.num_joints = 0
        self.num_bodies = 0

        for clip_idx, motion_path in enumerate(self.motion_files):
            with np.load(motion_path) as data:
                fps = int(np.asarray(data["fps"]).reshape(-1)[0])
                joint_pos = data["joint_pos"].astype(np.float32)
                joint_vel = data["joint_vel"].astype(np.float32)
                body_pos_w = data["body_pos_w"].astype(np.float32)
                body_quat_w = data["body_quat_w"].astype(np.float32)
                body_lin_vel_w = data["body_lin_vel_w"].astype(np.float32)
                body_ang_vel_w = data["body_ang_vel_w"].astype(np.float32)
                joint_torque = (
                    data["joint_torque"].astype(np.float32) if "joint_torque" in data else None
                )
                joint_torque_limit = (
                    data["joint_torque_limit"].astype(np.float32).reshape(-1)
                    if "joint_torque_limit" in data
                    else None
                )

            if body_indices is not None:
                body_pos_w = body_pos_w[:, body_indices]
                body_quat_w = body_quat_w[:, body_indices]
                body_lin_vel_w = body_lin_vel_w[:, body_indices]
                body_ang_vel_w = body_ang_vel_w[:, body_indices]

            num_frames = joint_pos.shape[0]
            if num_frames == 0:
                raise ValueError(f"Motion file '{motion_path}' contains no frames")
            if joint_vel.shape[0] != num_frames:
                raise ValueError(
                    f"Motion file '{motion_path}' has inconsistent frame counts between "
                    "'joint_pos' and 'joint_vel'"
                )
            for name, array in (
                ("body_pos_w", body_pos_w),
                ("body_quat_w", body_quat_w),
                ("body_lin_vel_w", body_lin_vel_w),
                ("body_ang_vel_w", body_ang_vel_w),
            ):
                if array.shape[0] != num_frames:
                    raise ValueError(
                        f"Motion file '{motion_path}' has inconsistent frame counts for '{name}'"
                    )
            if joint_torque is not None and (
                joint_torque.shape[0] != num_frames or joint_torque.shape[1] != joint_pos.shape[1]
            ):
                raise ValueError(
                    f"Motion file '{motion_path}' has 'joint_torque' shape "
                    f"{joint_torque.shape}, expected ({num_frames}, {joint_pos.shape[1]})"
                )
            if joint_torque_limit is not None and (
                joint_torque_limit.shape != (joint_pos.shape[1],)
                or not np.isfinite(joint_torque_limit).all()
                or np.any(joint_torque_limit <= 0.0)
            ):
                raise ValueError(
                    f"Motion file '{motion_path}' has invalid 'joint_torque_limit'; expected "
                    f"{joint_pos.shape[1]} finite positive entries, got {joint_torque_limit}"
                )

            if clip_idx == 0:
                self.fps = fps
                self.num_joints = joint_pos.shape[1]
                self.num_bodies = body_pos_w.shape[1]
            else:
                if fps != self.fps:
                    raise ValueError(
                        f"Motion file '{motion_path}' has fps={fps}, expected {self.fps}"
                    )
                if joint_pos.shape[1] != self.num_joints or joint_vel.shape[1] != self.num_joints:
                    raise ValueError(
                        f"Motion file '{motion_path}' has incompatible joint dimensions"
                    )
                if (
                    body_pos_w.shape[1] != self.num_bodies
                    or body_quat_w.shape[1] != self.num_bodies
                    or body_lin_vel_w.shape[1] != self.num_bodies
                    or body_ang_vel_w.shape[1] != self.num_bodies
                ):
                    raise ValueError(
                        f"Motion file '{motion_path}' has incompatible body dimensions"
                    )

            clip_lengths.append(num_frames)
            joint_pos_list.append(joint_pos)
            joint_vel_list.append(joint_vel)
            body_pos_list.append(body_pos_w)
            body_quat_list.append(body_quat_w)
            body_lin_vel_list.append(body_lin_vel_w)
            body_ang_vel_list.append(body_ang_vel_w)
            if (joint_torque is None) != (joint_torque_limit is None):
                raise ValueError(
                    f"Motion file '{motion_path}' has incomplete torque data; 'joint_torque' "
                    "and 'joint_torque_limit' must be provided together"
                )
            if joint_torque is not None and joint_torque_limit is not None:
                joint_torque_list.append(joint_torque)
                joint_torque_limit_list.append(joint_torque_limit)

        self.clip_lengths = np.asarray(clip_lengths, dtype=np.int32)
        self.num_clips = int(self.clip_lengths.shape[0])
        self.clip_offsets = np.zeros(self.num_clips, dtype=np.int32)
        if self.num_clips > 1:
            self.clip_offsets[1:] = np.cumsum(self.clip_lengths[:-1], dtype=np.int32)
        self.clip_end_frames = self.clip_offsets + self.clip_lengths - 1

        self.joint_pos = np.concatenate(joint_pos_list, axis=0)
        self.joint_vel = np.concatenate(joint_vel_list, axis=0)
        self.body_pos_w = np.concatenate(body_pos_list, axis=0)
        self.body_quat_w = np.concatenate(body_quat_list, axis=0)
        self.body_lin_vel_w = np.concatenate(body_lin_vel_list, axis=0)
        self.body_ang_vel_w = np.concatenate(body_ang_vel_list, axis=0)

        self.joint_torque: np.ndarray | None = None
        self.joint_torque_limit: np.ndarray | None = None
        if len(joint_torque_list) == self.num_clips:
            torque_limit = joint_torque_limit_list[0]
            for clip_limit in joint_torque_limit_list[1:]:
                if not np.allclose(clip_limit, torque_limit):
                    raise ValueError(
                        "Motion clips have inconsistent 'joint_torque_limit' values; "
                        "all clips must share the same per-joint torque limits"
                    )
            self.joint_torque = np.concatenate(joint_torque_list, axis=0)
            self.joint_torque_limit = torque_limit
        elif joint_torque_list:
            warnings.warn(
                "Motion files have inconsistent torque data presence "
                f"({len(joint_torque_list)} of {self.num_clips} clips provide 'joint_torque'); "
                "treating the whole motion as torque-free",
                stacklevel=2,
            )

        self.num_frames = int(self.joint_pos.shape[0])

    @staticmethod
    def _normalize_motion_files(motion_file: str | Sequence[str]) -> tuple[str, ...]:
        motion_files: tuple[str, ...]
        if isinstance(motion_file, str):
            motion_files = (motion_file,)
        elif isinstance(motion_file, Sequence):
            motion_files = tuple(motion_file)
        else:
            raise TypeError("motion_file must be a string path or a sequence of string paths")

        if not motion_files:
            raise ValueError("motion_file must contain at least one NPZ path")
        if any((not isinstance(path, str)) or (not path) for path in motion_files):
            raise ValueError("motion_file entries must be non-empty strings")
        return motion_files

    def get_clip_indices(self, frame_idx: np.ndarray) -> np.ndarray:
        """Map global frame indices to clip indices."""
        clip_indices = np.searchsorted(self.clip_offsets, frame_idx, side="right") - 1
        return np.asarray(clip_indices, dtype=np.int32)

    def motion_features_torch(self, device: str | torch.device) -> torch.Tensor:
        """Return the immutable motion feature table on the requested device.

        NPZ parsing and validation remain inside this cold-path leaf. Runtime
        consumers receive one contiguous float32 table with columns packed in
        the documented motion-feature order rather than mutable NumPy arrays.
        """
        arrays = (
            self.joint_pos,
            self.joint_vel,
            self.body_pos_w,
            self.body_quat_w,
            self.body_lin_vel_w,
            self.body_ang_vel_w,
        )
        host = np.concatenate(
            [np.asarray(value, dtype=np.float32).reshape(value.shape[0], -1) for value in arrays],
            axis=1,
        )
        return torch.from_numpy(np.ascontiguousarray(host)).to(device=device)

    @property
    def has_joint_torque(self) -> bool:
        """Whether every clip provided joint torque references."""
        return self.joint_torque is not None

    def make_motion_data_buffer(self, num_frames: int) -> MotionData:
        """Allocate a reusable ``MotionData`` buffer for frame-index gathers."""
        return MotionData(
            joint_pos=np.empty((num_frames, self.num_joints), dtype=self.joint_pos.dtype),
            joint_vel=np.empty((num_frames, self.num_joints), dtype=self.joint_vel.dtype),
            body_pos_w=np.empty((num_frames, self.num_bodies, 3), dtype=self.body_pos_w.dtype),
            body_quat_w=np.empty((num_frames, self.num_bodies, 4), dtype=self.body_quat_w.dtype),
            body_lin_vel_w=np.empty(
                (num_frames, self.num_bodies, 3), dtype=self.body_lin_vel_w.dtype
            ),
            body_ang_vel_w=np.empty(
                (num_frames, self.num_bodies, 3), dtype=self.body_ang_vel_w.dtype
            ),
            joint_torque=(
                np.empty((num_frames, self.num_joints), dtype=self.joint_torque.dtype)
                if self.joint_torque is not None
                else None
            ),
        )

    def get_motion_at_frame(
        self, frame_idx: np.ndarray, out: MotionData | None = None
    ) -> MotionData:
        """Get motion data at specified frame indices.

        Args:
            frame_idx: Frame indices (N,)
            out: Optional reusable output buffer.

        Returns:
            MotionData at specified frames
        """
        if out is not None:
            # Keep this cold loader contract NumPy-only. Runtime command owners
            # copy Torch carriers directly and never pass them through here.
            arrays = self._numpy_motion_arrays(out)
            np.take(self.joint_pos, frame_idx, axis=0, out=arrays[0])
            np.take(self.joint_vel, frame_idx, axis=0, out=arrays[1])
            np.take(self.body_pos_w, frame_idx, axis=0, out=arrays[2])
            np.take(self.body_quat_w, frame_idx, axis=0, out=arrays[3])
            np.take(self.body_lin_vel_w, frame_idx, axis=0, out=arrays[4])
            np.take(self.body_ang_vel_w, frame_idx, axis=0, out=arrays[5])
            if self.joint_torque is not None and out.joint_torque is not None:
                assert isinstance(out.joint_torque, np.ndarray)
                np.take(self.joint_torque, frame_idx, axis=0, out=out.joint_torque)
            return out

        return MotionData(
            joint_pos=self.joint_pos[frame_idx],
            joint_vel=self.joint_vel[frame_idx],
            body_pos_w=self.body_pos_w[frame_idx],
            body_quat_w=self.body_quat_w[frame_idx],
            body_lin_vel_w=self.body_lin_vel_w[frame_idx],
            body_ang_vel_w=self.body_ang_vel_w[frame_idx],
            joint_torque=(self.joint_torque[frame_idx] if self.joint_torque is not None else None),
        )

    @staticmethod
    def _numpy_motion_arrays(out: MotionData) -> tuple[np.ndarray, ...]:
        values = (
            out.joint_pos,
            out.joint_vel,
            out.body_pos_w,
            out.body_quat_w,
            out.body_lin_vel_w,
            out.body_ang_vel_w,
        )
        if any(isinstance(value, torch.Tensor) for value in values):
            raise TypeError("MotionLoader reusable buffers must be NumPy arrays")
        return cast("tuple[np.ndarray, ...]", values)
