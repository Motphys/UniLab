"""Motion loading and sampling for motion tracking tasks."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np

from unilab.assets.hub import resolve_motion_files


@dataclass
class MotionData:
    """Container for motion data at specific frame(s)."""

    joint_pos: np.ndarray  # (N, num_joints)
    joint_vel: np.ndarray  # (N, num_joints)
    body_pos_w: np.ndarray  # (N, num_bodies, 3)
    body_quat_w: np.ndarray  # (N, num_bodies, 4)
    body_lin_vel_w: np.ndarray  # (N, num_bodies, 3)
    body_ang_vel_w: np.ndarray  # (N, num_bodies, 3)


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

    def get_clip_end_frames(self, frame_idx: np.ndarray) -> np.ndarray:
        """Return the inclusive global end frame for each indexed clip."""
        clip_indices = self.get_clip_indices(frame_idx)
        return np.asarray(self.clip_end_frames[clip_indices], dtype=np.int32)

    def get_motion_at_frame(self, frame_idx: np.ndarray) -> MotionData:
        """Get motion data at specified frame indices.

        Args:
            frame_idx: Frame indices (N,)

        Returns:
            MotionData at specified frames
        """
        return MotionData(
            joint_pos=self.joint_pos[frame_idx],
            joint_vel=self.joint_vel[frame_idx],
            body_pos_w=self.body_pos_w[frame_idx],
            body_quat_w=self.body_quat_w[frame_idx],
            body_lin_vel_w=self.body_lin_vel_w[frame_idx],
            body_ang_vel_w=self.body_ang_vel_w[frame_idx],
        )


class MotionSampler:
    """Handles motion frame sampling with different strategies."""

    def __init__(
        self,
        motion_loader: MotionLoader,
        mode: Literal["start", "clip_start", "uniform", "adaptive"],
        num_envs: int,
        bin_count: int | None = None,
        adaptive_lambda: float = 0.8,
        adaptive_kernel_size: int = 1,
        adaptive_uniform_ratio: float = 0.1,
        adaptive_alpha: float = 0.001,
    ):
        """Initialize motion sampler.

        Args:
            motion_loader: Motion loader instance
            mode: Sampling mode ("start", "clip_start", "uniform", "adaptive")
            num_envs: Number of parallel environments
            bin_count: Number of bins for adaptive sampling (auto if None)
            adaptive_lambda: Decay factor for adaptive kernel
            adaptive_kernel_size: Kernel size for adaptive sampling
            adaptive_uniform_ratio: Uniform sampling ratio for adaptive mode
            adaptive_alpha: EMA alpha for failure count updates
        """
        self.motion_loader = motion_loader
        self.mode = mode
        self.num_envs = num_envs

        # Current frame indices for each environment
        self.current_frames = np.zeros(num_envs, dtype=np.int32)
        self.current_clip_indices = np.zeros(num_envs, dtype=np.int32)
        self.current_clip_end_frames = np.full(
            num_envs, motion_loader.clip_end_frames[0], dtype=np.int32
        )

        # Adaptive sampling parameters
        if bin_count is None:
            # Auto-compute bin count based on motion length and FPS
            self.bin_count = int(motion_loader.num_frames // motion_loader.fps) + 1
        else:
            self.bin_count = bin_count

        self.adaptive_lambda = adaptive_lambda
        self.adaptive_kernel_size = adaptive_kernel_size
        self.adaptive_uniform_ratio = adaptive_uniform_ratio
        self.adaptive_alpha = adaptive_alpha

        # Failure tracking for adaptive sampling
        self.bin_failed_count = np.zeros(self.bin_count, dtype=np.float32)
        self._current_bin_failed = np.zeros(self.bin_count, dtype=np.float32)

        # Precompute adaptive kernel
        self.kernel = np.array(
            [adaptive_lambda**i for i in range(adaptive_kernel_size)], dtype=np.float32
        )
        self.kernel = self.kernel / self.kernel.sum()

        # Metrics
        self.sampling_entropy = 0.0
        self.sampling_top1_prob = 0.0
        self.sampling_top1_bin = 0.0

    def sample_frames(self, env_ids: np.ndarray) -> np.ndarray:
        """Sample motion frames for specified environments.

        Args:
            env_ids: Environment indices to sample for

        Returns:
            Sampled frame indices
        """
        if self.mode == "start":
            return self._sample_start(env_ids)
        elif self.mode == "clip_start":
            return self._sample_clip_start(env_ids)
        elif self.mode == "uniform":
            return self._sample_uniform(env_ids)
        elif self.mode == "adaptive":
            return self._sample_adaptive(env_ids)
        else:
            raise ValueError(f"Unknown sampling mode: {self.mode}")

    def _sample_start(self, env_ids: np.ndarray) -> np.ndarray:
        """Always start from the global first frame (historical behavior)."""
        frames = np.zeros(len(env_ids), dtype=np.int32)
        self._set_sampled_frames(env_ids, frames)
        return frames

    def _sample_clip_start(self, env_ids: np.ndarray) -> np.ndarray:
        """Start from the first frame of a randomly chosen clip."""
        frames: np.ndarray
        if self.motion_loader.num_clips == 1:
            frames = np.zeros(len(env_ids), dtype=np.int32)
        else:
            clip_indices = np.random.randint(
                0, self.motion_loader.num_clips, len(env_ids), dtype=np.int32
            )
            frames = np.asarray(self.motion_loader.clip_offsets[clip_indices], dtype=np.int32)
        self._set_sampled_frames(env_ids, frames)
        return frames

    def _sample_uniform(self, env_ids: np.ndarray) -> np.ndarray:
        """Sample uniformly across motion."""
        frames = np.random.randint(0, self.motion_loader.num_frames, len(env_ids), dtype=np.int32)
        self._set_sampled_frames(env_ids, frames)

        # Update metrics
        self.sampling_entropy = 1.0  # Maximum entropy for uniform
        self.sampling_top1_prob = 1.0 / self.bin_count
        self.sampling_top1_bin = 0.5  # No specific bin preference

        return frames

    def _sample_adaptive(self, env_ids: np.ndarray) -> np.ndarray:
        """Sample adaptively based on failure statistics."""
        # Compute sampling probabilities
        sampling_probs = self.bin_failed_count + self.adaptive_uniform_ratio / float(self.bin_count)

        # Apply smoothing kernel (non-causal convolution)
        if self.adaptive_kernel_size > 1:
            # Pad and convolve
            padded = np.pad(sampling_probs, (0, self.adaptive_kernel_size - 1), mode="edge")
            sampling_probs = np.convolve(padded, self.kernel, mode="valid")

        # Normalize to probabilities
        sampling_probs = sampling_probs / sampling_probs.sum()

        # Sample bins
        sampled_bins = np.random.choice(self.bin_count, size=len(env_ids), p=sampling_probs)

        # Add random offset within bin
        bin_offsets = np.random.uniform(0.0, 1.0, len(env_ids))
        frames = (
            (sampled_bins + bin_offsets) / self.bin_count * (self.motion_loader.num_frames - 1)
        ).astype(np.int32)

        self._set_sampled_frames(env_ids, frames)

        # Update metrics
        H = -(sampling_probs * np.log(sampling_probs + 1e-12)).sum()
        H_norm = H / math.log(self.bin_count) if self.bin_count > 1 else 1.0
        pmax_idx = np.argmax(sampling_probs)
        pmax = sampling_probs[pmax_idx]

        self.sampling_entropy = H_norm
        self.sampling_top1_prob = float(pmax)
        self.sampling_top1_bin = float(pmax_idx) / self.bin_count

        return np.asarray(frames, dtype=np.int32)

    def update_failure_stats(
        self, terminated: np.ndarray, current_frames: np.ndarray | None = None
    ):
        """Update failure statistics for adaptive sampling.

        Args:
            terminated: Boolean array indicating which environments terminated
            current_frames: Optional current frame indices (uses internal if None)
        """
        if self.mode != "adaptive":
            return

        if current_frames is None:
            current_frames = self.current_frames

        # Find which bins failed
        if np.any(terminated):
            bin_indices = np.clip(
                (current_frames * self.bin_count) // max(self.motion_loader.num_frames, 1),
                0,
                self.bin_count - 1,
            )
            failed_bins = bin_indices[terminated]

            # Count failures per bin
            self._current_bin_failed[:] = 0
            for bin_idx in failed_bins:
                self._current_bin_failed[bin_idx] += 1

            # Update EMA of failure counts
            self.bin_failed_count = (
                self.adaptive_alpha * self._current_bin_failed
                + (1 - self.adaptive_alpha) * self.bin_failed_count
            )

    def _set_sampled_frames(self, env_ids: np.ndarray, frames: np.ndarray) -> None:
        self.current_frames[env_ids] = frames
        clip_indices = self.motion_loader.get_clip_indices(frames)
        self.current_clip_indices[env_ids] = clip_indices
        self.current_clip_end_frames[env_ids] = self.motion_loader.clip_end_frames[clip_indices]

    def step(self):
        """Advance all frames by one step."""
        self.current_frames += 1

        # Find environments that reached the end of their current clip.
        done_mask = self.current_frames > self.current_clip_end_frames
        return np.where(done_mask)[0]

    def get_current_motion(self) -> MotionData:
        """Get motion data at current frames for all environments."""
        return self.motion_loader.get_motion_at_frame(self.current_frames)
