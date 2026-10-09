"""Manager-native NumPy terms for motion tracking."""

from __future__ import annotations

import dataclasses
import math
import time
import warnings
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Literal, cast

import numpy as np
import torch

from unilab.envs.mdp.actions import JointPositionAction, JointPositionActionCfg
from unilab.managers import (
    CommandTerm,
    CommandTermCfg,
    ManagerTermBase,
    ManagerTermBaseCfg,
    ObservationTermCfg,
)
from unilab.managers.scene_entity_config import SceneEntityCfg
from unilab.utils.rotation import (
    np_quat_apply_inverse,
    np_quat_error_magnitude_squared_batched,
    np_quat_from_euler_xyz,
    np_quat_mul,
)

from .kernels import (
    configure_motion_kernel_runtime,
    reward_motion_body_ang_vel_kernel,
    reward_motion_body_lin_vel_kernel,
    reward_motion_body_ori_kernel,
    reward_motion_body_pos_kernel,
    termination_anchor_pos_kernel,
    update_motion_metrics_kernel,
    update_motion_relative_state_kernel,
)
from .motion_loader import MotionData, MotionLoader, MotionSampler
from .tensor_rotation import (
    quat_apply,
    quat_apply_inverse,
    quat_conjugate,
    quat_from_euler_xyz,
    quat_mul,
    quat_to_rot6,
)
from .tensor_sampler import TensorMotionSampler


def _quat_error_squared_torch(reference: torch.Tensor, actual: torch.Tensor) -> torch.Tensor:
    """Match the Numba kernel's absolute-dot shortest-path angular error."""
    rel = quat_mul(quat_conjugate(reference), actual)
    xyz = torch.linalg.vector_norm(rel[..., 1:4], dim=-1)
    angle = 2.0 * torch.atan2(xyz, rel[..., 0].abs().clamp(max=1.0))
    return angle.square()


def _update_motion_metrics_torch(
    rows: torch.Tensor,
    anchor_body_idx: int,
    motion_body_pos_w: torch.Tensor,
    robot_body_pos_w: torch.Tensor,
    motion_body_quat_w: torch.Tensor,
    robot_body_quat_w: torch.Tensor,
    motion_body_lin_vel_w: torch.Tensor,
    robot_body_lin_vel_w: torch.Tensor,
    motion_body_ang_vel_w: torch.Tensor,
    robot_body_ang_vel_w: torch.Tensor,
    body_pos_relative_w: torch.Tensor,
    body_quat_relative_w: torch.Tensor,
    motion_joint_pos: torch.Tensor,
    robot_joint_pos: torch.Tensor,
    motion_joint_vel: torch.Tensor,
    robot_joint_vel: torch.Tensor,
    outputs: tuple[torch.Tensor, ...],
) -> None:
    """Write all row-wise MotionCommand metrics without a host carrier."""
    if len(outputs) != 10:
        raise ValueError("motion Torch metrics require ten row-wise outputs")
    target = slice(None) if rows.numel() == motion_body_pos_w.shape[0] else rows
    anchor = (slice(None), anchor_body_idx)
    anchor_position_error = torch.linalg.vector_norm(
        motion_body_pos_w[target][anchor] - robot_body_pos_w[target][anchor], dim=-1
    )
    body_position_error = torch.linalg.vector_norm(
        body_pos_relative_w[target] - robot_body_pos_w[target], dim=-1
    ).mean(dim=-1)
    body_orientation_error = (
        _quat_error_squared_torch(body_quat_relative_w[target], robot_body_quat_w[target])
        .sqrt()
        .mean(dim=-1)
    )
    values = (
        anchor_position_error,
        _quat_error_squared_torch(
            motion_body_quat_w[target][anchor], robot_body_quat_w[target][anchor]
        ).sqrt(),
        torch.linalg.vector_norm(
            motion_body_lin_vel_w[target][anchor] - robot_body_lin_vel_w[target][anchor], dim=-1
        ),
        torch.linalg.vector_norm(
            motion_body_ang_vel_w[target][anchor] - robot_body_ang_vel_w[target][anchor], dim=-1
        ),
        body_position_error,
        body_orientation_error,
        torch.linalg.vector_norm(
            motion_body_lin_vel_w[target] - robot_body_lin_vel_w[target], dim=-1
        ).mean(dim=-1),
        torch.linalg.vector_norm(
            motion_body_ang_vel_w[target] - robot_body_ang_vel_w[target], dim=-1
        ).mean(dim=-1),
        torch.linalg.vector_norm(motion_joint_pos[target] - robot_joint_pos[target], dim=-1),
        torch.linalg.vector_norm(motion_joint_vel[target] - robot_joint_vel[target], dim=-1),
    )
    for output, value in zip(outputs, values, strict=True):
        output.index_copy_(0, rows, value)


def _update_motion_relative_state_torch(
    rows: torch.Tensor,
    anchor_body_idx: int,
    motion_body_pos_local_w: torch.Tensor,
    motion_body_pos_w: torch.Tensor,
    motion_body_quat_w: torch.Tensor,
    robot_body_pos_w: torch.Tensor,
    robot_body_quat_w: torch.Tensor,
    body_pos_relative_w: torch.Tensor,
    body_quat_relative_w: torch.Tensor,
    motion_anchor_pos_b: torch.Tensor,
    motion_anchor_ori_b: torch.Tensor,
    robot_body_pos_b: torch.Tensor,
    robot_body_ori_b: torch.Tensor,
) -> None:
    """Write all row-wise MotionCommand relative transforms on device."""
    target = slice(None) if rows.numel() == robot_body_pos_w.shape[0] else rows
    anchor = (slice(None), anchor_body_idx)
    motion_anchor_pos = motion_body_pos_local_w[target][anchor]
    motion_anchor_quat = motion_body_quat_w[target][anchor]
    robot_anchor_pos = robot_body_pos_w[target][anchor]
    robot_anchor_quat = robot_body_quat_w[target][anchor]

    yaw = quat_mul(robot_anchor_quat, quat_conjugate(motion_anchor_quat))
    yaw_w, yaw_z = yaw[..., 0], yaw[..., 3]
    half_yaw = 0.5 * torch.atan2(
        2.0 * (yaw_w * yaw_z + yaw[..., 1] * yaw[..., 2]),
        1.0 - 2.0 * (yaw[..., 2].square() + yaw_z.square()),
    )
    delta = torch.stack(
        (
            torch.cos(half_yaw),
            torch.zeros_like(half_yaw),
            torch.zeros_like(half_yaw),
            torch.sin(half_yaw),
        ),
        dim=-1,
    )
    delta_body = delta[:, None, :]
    body_quat_relative_w[target] = quat_mul(delta_body, motion_body_quat_w[target])

    local = motion_body_pos_local_w[target] - motion_anchor_pos[:, None, :]
    rotated = quat_apply(delta[:, None, :], local)
    relative = rotated + robot_anchor_pos[:, None, :]
    relative[..., 2] = motion_body_pos_local_w[target][..., 2]
    body_pos_relative_w[target] = relative

    anchor_delta = motion_body_pos_w[target][anchor] - robot_anchor_pos
    motion_anchor_pos_b[target] = quat_apply_inverse(robot_anchor_quat, anchor_delta)
    motion_anchor_ori_b[target] = quat_to_rot6(
        quat_mul(quat_conjugate(robot_anchor_quat), motion_anchor_quat)
    )

    robot_local = robot_body_pos_w[target] - robot_anchor_pos[:, None, :]
    robot_body_pos_b[target] = quat_apply_inverse(robot_anchor_quat[:, None, :], robot_local)
    robot_body_ori_b[target] = quat_to_rot6(
        quat_mul(quat_conjugate(robot_anchor_quat[:, None, :]), robot_body_quat_w[target])
    )


_MotionRelativeStateFn = Callable[..., None]
_update_motion_relative_state_compiled: _MotionRelativeStateFn | None = None
_update_motion_metrics_compiled: _MotionRelativeStateFn | None = None
_motion_reset_values_compiled: Callable[..., None] | None = None
_ingest_motion_packet_compiled: Callable[..., None] | None = None
_refresh_motion_robot_state_compiled: Callable[..., None] | None = None


def _compiled_motion_relative_state_available() -> bool:
    """Whether Torch compilation is available for the owner-relative kernel."""
    return hasattr(torch, "compile") and hasattr(torch.compiler, "is_compiling")


def _bind_compiled_motion_relative_state() -> _MotionRelativeStateFn:
    """Compile the owner-relative kernel once on the declared device."""
    global _update_motion_relative_state_compiled
    if _update_motion_relative_state_compiled is not None:
        return _update_motion_relative_state_compiled
    if not _compiled_motion_relative_state_available():
        return _update_motion_relative_state_torch
    _update_motion_relative_state_compiled = torch.compile(
        _update_motion_relative_state_torch,
        dynamic=True,
    )
    return _update_motion_relative_state_compiled


def _bind_compiled_motion_metrics() -> _MotionRelativeStateFn:
    """Compile the ten-output motion metric kernel once per process."""
    global _update_motion_metrics_compiled
    if _update_motion_metrics_compiled is not None:
        return _update_motion_metrics_compiled
    if not _compiled_motion_relative_state_available():
        return _update_motion_metrics_torch
    _update_motion_metrics_compiled = torch.compile(
        _update_motion_metrics_torch,
        dynamic=True,
    )
    return _update_motion_metrics_compiled


def _motion_reset_values_kernel(
    packet_joint_pos: torch.Tensor,
    packet_joint_vel: torch.Tensor,
    packet_body_pos_w: torch.Tensor,
    packet_body_quat_w: torch.Tensor,
    packet_body_lin_vel_w: torch.Tensor,
    packet_body_ang_vel_w: torch.Tensor,
    origins: torch.Tensor,
    pose: torch.Tensor,
    velocity: torch.Tensor,
    joint_noise: torch.Tensor,
    soft_limits: torch.Tensor,
    joint_output: torch.Tensor,
    root_output: torch.Tensor,
) -> None:
    """Construct one fused motion reset value/root carrier without host rows."""
    joint_output.copy_(packet_joint_pos)
    joint_output.add_(joint_noise)
    joint_output.clamp_(soft_limits[:, 0], soft_limits[:, 1])

    half_roll = 0.5 * pose[:, 3]
    half_pitch = 0.5 * pose[:, 4]
    half_yaw = 0.5 * pose[:, 5]
    cos_roll, sin_roll = torch.cos(half_roll), torch.sin(half_roll)
    cos_pitch, sin_pitch = torch.cos(half_pitch), torch.sin(half_pitch)
    cos_yaw, sin_yaw = torch.cos(half_yaw), torch.sin(half_yaw)
    pose_quat = torch.stack(
        (
            cos_roll * cos_pitch * cos_yaw + sin_roll * sin_pitch * sin_yaw,
            sin_roll * cos_pitch * cos_yaw - cos_roll * sin_pitch * sin_yaw,
            cos_roll * sin_pitch * cos_yaw + sin_roll * cos_pitch * sin_yaw,
            cos_roll * cos_pitch * sin_yaw - sin_roll * sin_pitch * cos_yaw,
        ),
        dim=-1,
    )
    motion_quat = packet_body_quat_w[:, 0]
    left_w, left_xyz = pose_quat[:, 0:1], pose_quat[:, 1:4]
    right_w, right_xyz = motion_quat[:, 0:1], motion_quat[:, 1:4]
    cross = torch.linalg.cross(left_xyz, right_xyz, dim=-1)
    root_quat = torch.cat(
        (
            left_w * right_w - (left_xyz * right_xyz).sum(dim=-1, keepdim=True),
            left_w * right_xyz + right_w * left_xyz + cross,
        ),
        dim=-1,
    )

    root_output[:, :3] = packet_body_pos_w[:, 0] + origins + pose[:, :3]
    root_output[:, 3:7] = root_quat
    root_output[:, 7:10] = packet_body_lin_vel_w[:, 0] + velocity[:, :3]
    root_output[:, 10:13] = packet_body_ang_vel_w[:, 0] + velocity[:, 3:]


def _bind_compiled_motion_reset_values() -> Callable[..., None]:
    """Compile the reset value/root construction once per process."""
    global _motion_reset_values_compiled
    if _motion_reset_values_compiled is not None:
        return _motion_reset_values_compiled
    if not _compiled_motion_relative_state_available():
        return _motion_reset_values_kernel
    _motion_reset_values_compiled = torch.compile(_motion_reset_values_kernel, dynamic=True)
    return _motion_reset_values_compiled


def _ingest_motion_packet_kernel(
    rows: torch.Tensor,
    packet: torch.Tensor,
    joint_pos: torch.Tensor,
    joint_vel: torch.Tensor,
    body_pos: torch.Tensor,
    body_quat: torch.Tensor,
    body_lin_vel: torch.Tensor,
    body_ang_vel: torch.Tensor,
    published_body_pos: torch.Tensor,
    command: torch.Tensor,
    origins: torch.Tensor,
    *,
    num_bodies: int,
    num_joints: int,
) -> None:
    """Scatter one motion feature packet into the owner's device carriers."""
    count = rows.numel()
    joint_width = num_joints
    vel_start, vel_end = joint_width, joint_width * 2
    body_start = vel_end
    quat_start = body_start + num_bodies * 3
    lin_start = quat_start + num_bodies * 4
    ang_start = lin_start + num_bodies * 3
    joint_pos.index_copy_(0, rows, packet[:, :joint_width])
    joint_vel.index_copy_(0, rows, packet[:, vel_start:vel_end])
    body_pos.index_copy_(0, rows, packet[:, body_start:quat_start].view(count, num_bodies, 3))
    body_quat.index_copy_(0, rows, packet[:, quat_start:lin_start].view(count, num_bodies, 4))
    body_lin_vel.index_copy_(0, rows, packet[:, lin_start:ang_start].view(count, num_bodies, 3))
    body_ang_vel.index_copy_(0, rows, packet[:, ang_start:].view(count, num_bodies, 3))
    published_body_pos.index_copy_(
        0,
        rows,
        packet[:, body_start:quat_start].view(count, num_bodies, 3) + origins[:, None, :],
    )
    command.index_copy_(0, rows, packet[:, : joint_width * 2])


def _bind_compiled_motion_packet_ingest() -> Callable[..., None]:
    """Compile the owner packet scatter once for full and selected rows."""
    global _ingest_motion_packet_compiled
    if _ingest_motion_packet_compiled is not None:
        return _ingest_motion_packet_compiled
    if not _compiled_motion_relative_state_available():
        return _ingest_motion_packet_kernel
    _ingest_motion_packet_compiled = torch.compile(
        _ingest_motion_packet_kernel,
        dynamic=True,
    )
    return _ingest_motion_packet_compiled


def _refresh_motion_robot_state_kernel(
    rows: torch.Tensor,
    view_pos: torch.Tensor,
    view_quat: torch.Tensor,
    view_lin_vel: torch.Tensor,
    view_ang_vel: torch.Tensor,
    robot_body_pos: torch.Tensor,
    robot_body_quat: torch.Tensor,
    robot_body_lin_vel: torch.Tensor,
    robot_body_ang_vel: torch.Tensor,
) -> None:
    """Copy authoritative selected-row body state into the command carrier."""
    target = slice(None) if rows.numel() == robot_body_pos.shape[0] else rows
    robot_body_pos[target] = view_pos[target]
    robot_body_quat[target] = view_quat[target]
    robot_body_lin_vel[target] = view_lin_vel[target]
    robot_body_ang_vel[target] = view_ang_vel[target]


def _bind_compiled_motion_robot_refresh() -> Callable[..., None]:
    """Compile the selected-row robot-state copy once."""
    global _refresh_motion_robot_state_compiled
    if _refresh_motion_robot_state_compiled is not None:
        return _refresh_motion_robot_state_compiled
    if not _compiled_motion_relative_state_available():
        return _refresh_motion_robot_state_kernel
    _refresh_motion_robot_state_compiled = torch.compile(
        _refresh_motion_robot_state_kernel,
        dynamic=True,
    )
    return _refresh_motion_robot_state_compiled


_MotionPostComputeFn = Callable[..., None]
_motion_post_compute_compiled: _MotionPostComputeFn | None = None


def _motion_post_compute_kernel(
    rows: torch.Tensor,
    anchor_body_idx: int,
    view_pos: torch.Tensor,
    view_quat: torch.Tensor,
    view_lin_vel: torch.Tensor,
    view_ang_vel: torch.Tensor,
    robot_body_pos: torch.Tensor,
    robot_body_quat: torch.Tensor,
    robot_body_lin_vel: torch.Tensor,
    robot_body_ang_vel: torch.Tensor,
    motion_body_pos_local_w: torch.Tensor,
    motion_body_pos_w: torch.Tensor,
    motion_body_quat_w: torch.Tensor,
    body_pos_relative_w: torch.Tensor,
    body_quat_relative_w: torch.Tensor,
    motion_anchor_pos_b: torch.Tensor,
    motion_anchor_ori_b: torch.Tensor,
    robot_body_pos_b: torch.Tensor,
    robot_body_ori_b: torch.Tensor,
) -> None:
    """Refresh selected robot state and relative motion state in one graph."""
    target = slice(None) if rows.numel() == robot_body_pos.shape[0] else rows
    robot_body_pos[target] = view_pos[target]
    robot_body_quat[target] = view_quat[target]
    robot_body_lin_vel[target] = view_lin_vel[target]
    robot_body_ang_vel[target] = view_ang_vel[target]
    _update_motion_relative_state_torch(
        rows,
        anchor_body_idx,
        motion_body_pos_local_w,
        motion_body_pos_w,
        motion_body_quat_w,
        robot_body_pos,
        robot_body_quat,
        body_pos_relative_w,
        body_quat_relative_w,
        motion_anchor_pos_b,
        motion_anchor_ori_b,
        robot_body_pos_b,
        robot_body_ori_b,
    )


def _bind_compiled_motion_post_compute() -> _MotionPostComputeFn:
    """Compile the owner post-compute refresh once on the declared device."""
    global _motion_post_compute_compiled
    if _motion_post_compute_compiled is not None:
        return _motion_post_compute_compiled
    if not _compiled_motion_relative_state_available():
        return _motion_post_compute_kernel
    _motion_post_compute_compiled = torch.compile(
        _motion_post_compute_kernel,
        dynamic=True,
    )
    return _motion_post_compute_compiled


if TYPE_CHECKING:
    from unilab.base.entity import Entity
    from unilab.managers._types import ManagerBasedRlEnv


SamplingMode = Literal["start", "clip_start", "uniform", "adaptive", "mixed"]
_RANGE_KEYS = ("x", "y", "z", "roll", "pitch", "yaw")
_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")
_TORQUE_SENSOR_SUFFIX = "_torque"
# Per-body observation motion layout: pos_w | quat_w | lin_vel_w | ang_vel_w.
_OBS_MOTION_BLOCK_WIDTHS = (3, 4, 3, 3)
_OBS_MOTION_WIDTH_PER_BODY = sum(_OBS_MOTION_BLOCK_WIDTHS)


def _range_matrix(value: dict[str, tuple[float, float]], *, name: str) -> np.ndarray:
    unknown = sorted(set(value) - set(_RANGE_KEYS))
    if unknown:
        raise ValueError(f"{name} has unknown axes {unknown}")
    try:
        ranges = np.asarray([value.get(key, (0.0, 0.0)) for key in _RANGE_KEYS], dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must map axes to numeric (min, max) pairs") from exc
    if ranges.shape != (6, 2) or not np.isfinite(ranges).all():
        raise ValueError(f"{name} must contain six finite (min, max) pairs")
    if np.any(ranges[:, 0] > ranges[:, 1]):
        raise ValueError(f"{name} contains a minimum greater than its maximum")
    ranges.setflags(write=False)
    return ranges


def _pair(value: tuple[float, float], *, name: str) -> tuple[float, float]:
    try:
        values = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be a numeric (min, max) pair") from exc
    if values.shape != (2,) or not np.isfinite(values).all():
        raise ValueError(f"{name} must be a finite (min, max) pair")
    lower, upper = float(values[0]), float(values[1])
    if lower > upper:
        raise ValueError(f"{name} minimum {lower} exceeds maximum {upper}")
    return lower, upper


def _validate_motion_command_cfg(cfg: MotionCommandCfg) -> None:
    if not isinstance(cfg.entity_name, str) or not cfg.entity_name:
        raise ValueError("MotionCommandCfg entity_name must be non-empty")
    if not isinstance(cfg.params, MotionCommandParamsCfg):
        raise TypeError("MotionCommandCfg params must be MotionCommandParamsCfg")
    if not cfg.motion_file:
        raise ValueError("MotionCommandCfg motion_file must be configured")
    if not cfg.anchor_body_name or cfg.anchor_body_name not in cfg.body_names:
        raise ValueError("MotionCommandCfg anchor_body_name must occur in body_names")
    if len(set(cfg.body_names)) != len(cfg.body_names):
        raise ValueError("MotionCommandCfg body_names must be unique")
    if cfg.sampling_mode not in ("start", "clip_start", "uniform", "adaptive", "mixed"):
        raise ValueError(f"MotionCommandCfg has unsupported sampling_mode {cfg.sampling_mode!r}")
    if not 0.0 <= cfg.params.sampling_start_ratio <= 1.0:
        raise ValueError("MotionCommandCfg sampling_start_ratio must be within [0, 1]")
    if not isinstance(cfg.params.truncate_on_clip_end, bool):
        raise TypeError("MotionCommandCfg truncate_on_clip_end must be bool")
    obs_body_names = cfg.params.obs_body_names
    if obs_body_names is not None:
        obs_names = tuple(obs_body_names)
        if not obs_names or len(set(obs_names)) != len(obs_names):
            raise ValueError("MotionCommandCfg obs_body_names must be non-empty and unique")
        obs_root = cfg.params.obs_root_body_name or obs_names[0]
        if obs_root not in obs_names:
            raise ValueError("MotionCommandCfg obs_root_body_name must occur in obs_body_names")


@dataclass
class MotionCommandParamsCfg:
    """Hydra-owned motion data and sampling parameters."""

    motion_file: str | list[str]
    anchor_body_name: str
    body_names: tuple[str, ...] | list[str]
    sampling_mode: SamplingMode = "adaptive"
    sampling_start_ratio: float = 0.0
    truncate_on_clip_end: bool = False
    pose_range: dict[str, tuple[float, float]] = field(default_factory=dict)
    velocity_range: dict[str, tuple[float, float]] = field(default_factory=dict)
    joint_position_range: tuple[float, float] = (-0.1, 0.1)
    joint_default_position_range: tuple[float, float] = (0.0, 0.0)
    adaptive_lambda: float = 0.8
    adaptive_kernel_size: int = 1
    adaptive_uniform_ratio: float = 0.1
    adaptive_alpha: float = 0.001
    # Optional mimic-lite observation body set. It selects a body-sliced view
    # of the same motion data while the reward-facing body contract is fixed.
    obs_body_names: tuple[str, ...] | list[str] | None = None
    obs_root_body_name: str | None = None
    # MimicLite observation placement is intentionally excluded from legacy
    # canonical fingerprints; owners must validate it separately.
    _semantic_fingerprint_excludes = frozenset({"obs_body_names", "obs_root_body_name"})


@dataclass(frozen=True)
class TensorMotionObsFuture:
    """Device-resident future reference gather over the observation bodies."""

    future_steps: tuple[int, ...]
    ref_joint_pos: torch.Tensor
    ref_body_pos_w: torch.Tensor
    ref_body_quat_w: torch.Tensor
    ref_body_lin_vel_w: torch.Tensor
    ref_body_ang_vel_w: torch.Tensor


@dataclass(kw_only=True)
class MotionCommandCfg(CommandTermCfg):
    """Community-shaped motion command with Hydra-owned nested parameters."""

    entity_name: str
    params: MotionCommandParamsCfg

    def build(self, env: ManagerBasedRlEnv) -> MotionCommand:
        return MotionCommand(self, env)

    @property
    def motion_file(self) -> str | list[str]:
        return self.params.motion_file

    @property
    def anchor_body_name(self) -> str:
        return self.params.anchor_body_name

    @property
    def body_names(self) -> tuple[str, ...]:
        return tuple(self.params.body_names)

    @property
    def sampling_mode(self) -> SamplingMode:
        return self.params.sampling_mode


@dataclass(kw_only=True)
class TensorMotionCommandCfg(MotionCommandCfg):
    """Device-resident motion command for tensor Manager owners."""

    def build(self, env: ManagerBasedRlEnv) -> MotionCommand:
        return TensorMotionCommand(self, env)


class MotionCommand(CommandTerm):
    """Motion reference command on UniLab's NumPy/entity runtime."""

    cfg: MotionCommandCfg
    obs_motion: MotionLoader | None
    _obs_terms_aux_memo: dict[Any, Any]

    def __init__(self, cfg: MotionCommandCfg, env: ManagerBasedRlEnv):
        _validate_motion_command_cfg(cfg)
        super().__init__(cfg, env)
        self.robot = cast("Entity", env.scene[cfg.entity_name])
        body_ids, body_names = self.robot.find_bodies(cfg.body_names, preserve_order=True)
        if tuple(body_names) != cfg.body_names:
            raise ValueError(
                f"MotionCommand body order {tuple(body_names)} does not match {cfg.body_names}"
            )
        self._robot_body_ids = np.asarray(body_ids, dtype=np.intp)
        self._robot_body_ids.setflags(write=False)
        obs_names = tuple(cfg.params.obs_body_names or ())
        self.obs_body_names = obs_names
        if obs_names:
            obs_ids, matched_obs_names = self.robot.find_bodies(obs_names, preserve_order=True)
            if tuple(matched_obs_names) != obs_names:
                raise ValueError(
                    f"MotionCommand obs body order {tuple(matched_obs_names)} does not match "
                    f"{obs_names}"
                )
            self._obs_robot_body_ids = np.asarray(obs_ids, dtype=np.intp)
            self._obs_robot_body_ids.setflags(write=False)
        else:
            self._obs_robot_body_ids = None
        self.obs_root_body_idx = (
            0 if not obs_names else obs_names.index(cfg.params.obs_root_body_name or obs_names[0])
        )
        self.obs_motion = None
        self._obs_terms_aux_memo = {}
        self._copy_robot_body_state = self.robot.bind_body_state_copy(self._robot_body_ids)
        motion_body_ids = self.robot.motion_body_ids[self._robot_body_ids]
        self.motion = self._make_motion_loader(cfg.motion_file, motion_body_ids)
        if self.motion.num_joints != len(self.robot.joint_names):
            raise ValueError(
                f"MotionCommand motion joint width {self.motion.num_joints} does not match "
                f"entity '{self.robot.name}' joint width {len(self.robot.joint_names)}"
            )
        if self.motion.num_bodies != len(cfg.body_names):
            raise ValueError(
                f"MotionCommand motion body width {self.motion.num_bodies} does not match "
                f"configured body width {len(cfg.body_names)}"
            )

        self.anchor_body_idx = cfg.body_names.index(cfg.anchor_body_name)
        self.sampler = MotionSampler(
            self.motion,
            mode=self.cfg.params.sampling_mode,
            num_envs=self.num_envs,
            adaptive_lambda=self.cfg.params.adaptive_lambda,
            adaptive_kernel_size=self.cfg.params.adaptive_kernel_size,
            adaptive_uniform_ratio=self.cfg.params.adaptive_uniform_ratio,
            adaptive_alpha=self.cfg.params.adaptive_alpha,
            start_ratio=self.cfg.params.sampling_start_ratio,
            rng=env.rng,
        )
        self._pose_range = _range_matrix(cfg.params.pose_range, name="MotionCommand pose_range")
        self._velocity_range = _range_matrix(
            cfg.params.velocity_range, name="MotionCommand velocity_range"
        )
        self._joint_position_range = _pair(
            cfg.params.joint_position_range,
            name="MotionCommand joint_position_range",
        )
        self._joint_default_position_range = _pair(
            cfg.params.joint_default_position_range,
            name="MotionCommand joint_default_position_range",
        )

        num_bodies = len(cfg.body_names)
        num_joints = self.motion.num_joints
        dtype = self.motion.joint_pos.dtype
        self.time_steps = self.sampler.current_frames
        self._motion_data = self.motion.make_motion_data_buffer(self.num_envs)
        self._command = np.empty((self.num_envs, num_joints * 2), dtype=dtype)
        self._body_pos_w = np.empty((self.num_envs, num_bodies, 3), dtype=dtype)
        self.body_pos_relative_w = np.empty_like(self._body_pos_w)
        self.body_quat_relative_w = np.empty((self.num_envs, num_bodies, 4), dtype=dtype)
        self.motion_anchor_pos_b = np.empty((self.num_envs, 3), dtype=dtype)
        self.motion_anchor_ori_b = np.empty((self.num_envs, 6), dtype=dtype)
        self.robot_body_pos_b = np.empty_like(self._body_pos_w)
        self.robot_body_ori_b = np.empty((self.num_envs, num_bodies, 6), dtype=dtype)
        self.joint_default_bias = np.zeros((self.num_envs, num_joints), dtype=dtype)
        self._robot_cache_step = -1
        self._all_env_ids = np.arange(self.num_envs, dtype=np.int32)
        self._all_env_ids.setflags(write=False)
        # Env ids of the most recent scoped (reset-path) compute; None after a
        # per-step compute. Written by `_update_command`, consumed by
        # `post_compute` to restrict refresh work to the reset rows.
        self._post_compute_env_ids: np.ndarray | None = None
        self._tensor_post_compute_env_ids: torch.Tensor | None = None
        # Reset rows whose motion-reference buffers were already ingested by
        # `_resample_command` during the in-flight reset; consumed by the
        # reset-path `_update_command` to skip the redundant `_refresh_motion`
        # gather (issue #1355).
        self._resample_ingested_ids: np.ndarray | None = None
        # Motion rows gathered by the in-flight `_resample_command`, exposed so
        # callers reuse the same gather instead of
        # re-reading the same frames.
        self._resample_motion: MotionData | None = None
        self._robot_body_pos_w = np.empty_like(self._body_pos_w)
        self._robot_body_quat_w = np.empty((self.num_envs, num_bodies, 4), dtype=dtype)
        self._robot_body_lin_vel_w = np.empty_like(self._body_pos_w)
        self._robot_body_ang_vel_w = np.empty_like(self._body_pos_w)
        self._bind_read_phase = False
        self._tensor_all_rows = torch.arange(self.num_envs, dtype=torch.int64, device=self._device)
        self._tensor_resample_ingested: torch.Tensor | None = None

        for name in (
            "error_anchor_pos",
            "error_anchor_rot",
            "error_anchor_lin_vel",
            "error_anchor_ang_vel",
            "error_body_pos",
            "error_body_rot",
            "error_body_lin_vel",
            "error_body_ang_vel",
            "error_joint_pos",
            "error_joint_vel",
            "sampling_entropy",
            "sampling_top1_prob",
            "sampling_top1_bin",
        ):
            self.metrics[name] = np.zeros(self.num_envs, dtype=dtype)
        self._prepare_tensor_carrier()
        self._defer_read_phase_binding()

    def _defer_read_phase_binding(self) -> None:
        """Defer public state-view binding until the Manager read phase exists.

        Command construction happens before ``EntityScene`` compiles its packed
        tensor read plan. NumPy commands complete their cold initialization
        eagerly because their entity facade is already available; a tensor
        command overrides this method and finishes only metadata allocation,
        then binds public views from ``bind_read_phase``.
        """
        self._refresh_motion()
        self._refresh_robot_state(force=True)
        # Configure the parallel kernel workers on the cold path so the first
        # measured manager step contains no Numba worker/JIT initialization.
        configure_motion_kernel_runtime()
        self._refresh_relative_state()
        self._update_metrics(
            torch.from_numpy(np.array(self._all_env_ids, copy=True)).to(self._device)
        )

    def bind_read_phase(self) -> None:
        if self._bind_read_phase:
            return
        self._defer_read_phase_binding()
        self._bind_read_phase = True

    def _prepare_tensor_carrier(self) -> None:
        """Hook for subclasses to replace cold carrier buffers before probing."""

    def memoize_aux(
        self, future: TensorMotionObsFuture, key: str, compute: Callable[[], Any]
    ) -> Any:
        """Cache one future-scoped auxiliary observation value per future object.

        Entries are keyed by future identity and rebound when the future is
        refreshed; the memo is replaced once it holds eight distinct futures.
        """
        memo = getattr(self, "_obs_terms_aux_memo", None)
        if memo is None or len(memo) >= 8:
            memo = {}
            self._obs_terms_aux_memo = memo
        entry = memo.get(id(future))
        if entry is None or entry[0] is not future:
            entry = (future, {})
            memo[id(future)] = entry
        values: dict[str, Any] = entry[1]
        if key not in values:
            values[key] = compute()
        return values[key]

    def _make_motion_loader(
        self,
        motion_file: str | list[str],
        body_indices: np.ndarray,
    ) -> MotionLoader:
        """Materialize the profile-owned motion loader on the cold path."""
        return MotionLoader(motion_file, body_indices=body_indices)

    def _refresh_motion(self, env_ids: np.ndarray | None = None) -> None:
        """Refresh motion-reference buffers from the current frame indices."""
        if env_ids is None:
            self.motion.get_motion_at_frame(self.time_steps, out=self._motion_data)
            np.add(
                self._motion_data.body_pos_w,
                self._env.scene.env_origins[:, None, :],
                out=self._body_pos_w,
            )
            width = self.motion.num_joints
            self._command[:, :width] = self._motion_data.joint_pos
            self._command[:, width:] = self._motion_data.joint_vel
            return
        self._ingest_motion_rows(env_ids, self.motion.get_motion_at_frame(self.time_steps[env_ids]))

    def _ingest_motion_rows(self, env_ids: np.ndarray, data: MotionData) -> None:
        """Scatter one gathered motion frame set into the reference buffers."""
        for motion_field in dataclasses.fields(data):
            value = getattr(data, motion_field.name)
            target = getattr(self._motion_data, motion_field.name)
            if value is None or target is None:
                continue
            target[env_ids] = value
        self._body_pos_w[env_ids] = data.body_pos_w + self._env.scene.env_origins[env_ids, None, :]
        width = self.motion.num_joints
        self._command[env_ids, :width] = data.joint_pos
        self._command[env_ids, width:] = data.joint_vel

    def _refresh_robot_state(
        self, *, force: bool = False, env_ids: np.ndarray | None = None
    ) -> None:
        step = self._env.common_step_counter
        if not force and self._robot_cache_step == step:
            return
        if env_ids is None:
            self._copy_robot_body_state(
                self._robot_body_pos_w,
                self._robot_body_quat_w,
                self._robot_body_lin_vel_w,
                self._robot_body_ang_vel_w,
            )
        else:
            # Partial-reset path (issue #1295): gather only the reset rows from
            # the backend instead of full-batch body reads sliced afterwards.
            # _robot_body_ids selects the tracked subset afterwards, so the
            # row getters fetch all entity bodies for just these rows.
            data = self.robot.data
            self._robot_body_pos_w[env_ids] = data.body_link_pos_w_rows(env_ids)[
                :, self._robot_body_ids
            ]
            self._robot_body_quat_w[env_ids] = data.body_link_quat_w_rows(env_ids)[
                :, self._robot_body_ids
            ]
            self._robot_body_lin_vel_w[env_ids] = data.body_link_lin_vel_w_rows(env_ids)[
                :, self._robot_body_ids
            ]
            self._robot_body_ang_vel_w[env_ids] = data.body_link_ang_vel_w_rows(env_ids)[
                :, self._robot_body_ids
            ]
        self._robot_cache_step = step

    def _refresh_relative_state(self, env_ids: np.ndarray | None = None) -> None:
        rows = self._all_env_ids if env_ids is None else env_ids
        update_motion_relative_state_kernel(
            rows,
            self.anchor_body_idx,
            self._motion_data.body_pos_w,
            self._body_pos_w,
            self._motion_data.body_quat_w,
            self._robot_body_pos_w,
            self._robot_body_quat_w,
            self.body_pos_relative_w,
            self.body_quat_relative_w,
            self.motion_anchor_pos_b,
            self.motion_anchor_ori_b,
            self.robot_body_pos_b,
            self.robot_body_ori_b,
        )

    @property
    def robot_joint_pos(self) -> np.ndarray:
        return self.robot.data.joint_pos

    @property
    def robot_joint_vel(self) -> np.ndarray:
        return self.robot.data.joint_vel

    @property
    def robot_body_pos_w(self) -> np.ndarray:
        self._refresh_robot_state()
        return self._robot_body_pos_w

    @property
    def robot_body_quat_w(self) -> np.ndarray:
        self._refresh_robot_state()
        return self._robot_body_quat_w

    @property
    def robot_body_lin_vel_w(self) -> np.ndarray:
        self._refresh_robot_state()
        return self._robot_body_lin_vel_w

    @property
    def robot_body_ang_vel_w(self) -> np.ndarray:
        self._refresh_robot_state()
        return self._robot_body_ang_vel_w

    @property
    def robot_anchor_pos_w(self) -> np.ndarray:
        return self.robot_body_pos_w[:, self.anchor_body_idx]

    @property
    def robot_anchor_quat_w(self) -> np.ndarray:
        return self.robot_body_quat_w[:, self.anchor_body_idx]

    @property
    def robot_anchor_lin_vel_w(self) -> np.ndarray:
        return self.robot_body_lin_vel_w[:, self.anchor_body_idx]

    @property
    def robot_anchor_ang_vel_w(self) -> np.ndarray:
        return self.robot_body_ang_vel_w[:, self.anchor_body_idx]

    @property
    def command(self) -> np.ndarray:
        return self._command

    @property
    def joint_pos(self) -> np.ndarray:
        return self._motion_data.joint_pos

    @property
    def joint_vel(self) -> np.ndarray:
        return self._motion_data.joint_vel

    @property
    def body_pos_w(self) -> np.ndarray:
        return self._body_pos_w

    @property
    def body_quat_w(self) -> np.ndarray:
        return self._motion_data.body_quat_w

    @property
    def body_lin_vel_w(self) -> np.ndarray:
        return self._motion_data.body_lin_vel_w

    @property
    def body_ang_vel_w(self) -> np.ndarray:
        return self._motion_data.body_ang_vel_w

    @property
    def anchor_pos_w(self) -> np.ndarray:
        return self._body_pos_w[:, self.anchor_body_idx]

    @property
    def anchor_quat_w(self) -> np.ndarray:
        return self._motion_data.body_quat_w[:, self.anchor_body_idx]

    @property
    def anchor_lin_vel_w(self) -> np.ndarray:
        return self._motion_data.body_lin_vel_w[:, self.anchor_body_idx]

    @property
    def anchor_ang_vel_w(self) -> np.ndarray:
        return self._motion_data.body_ang_vel_w[:, self.anchor_body_idx]

    def reset(
        self,
        env_ids: torch.Tensor | slice | None,
        *,
        publish_metrics: bool = True,
    ) -> dict[str, float]:
        if isinstance(env_ids, torch.Tensor):
            ids = env_ids.detach().cpu().numpy().astype(np.int32, copy=False)
        else:
            ids = np.arange(self.num_envs, dtype=np.int32)[env_ids or slice(None)]
        # Row-wise error metrics are consumed only here (CommandTerm.reset logs
        # per-episode means from these rows, then zeroes them). The per-step
        # compute path skips the full-batch metrics kernel (issue #1355), so
        # refresh exactly the rows being reset from the current post-step
        # buffers — the same inputs the former per-step refresh used, keeping
        # the consumed values bit-identical.
        self._update_error_metrics(ids)
        lower, upper = self._joint_default_position_range
        self.joint_default_bias[ids] = self._env.rng.uniform(
            lower, upper, size=(len(ids), self.motion.num_joints)
        )
        return super().reset(
            env_ids if isinstance(env_ids, torch.Tensor) else torch.from_numpy(ids),
            publish_metrics=publish_metrics,
        )

    def _update_metrics(self, env_ids: torch.Tensor | None = None) -> None:
        # The row-wise error metrics are consumed only by `reset()` (episode
        # log extras), which refreshes exactly the rows it reads. The per-step
        # call (env_ids=None) therefore skips the Numba kernel over all rows
        # (issue #1355); the reset path (env_ids set) refreshes the reset rows
        # so post-reset metrics track the post-reset state.
        if env_ids is not None:
            rows = (
                env_ids.detach().cpu().numpy()
                if isinstance(env_ids, torch.Tensor)
                else np.asarray(env_ids)
            )
            self._update_error_metrics(rows)
        # Sampler statistics are global scalars, so every row tracks them.
        # These scalar sampling settings and the Numba error kernel below are
        # still NumPy-owned; assert that migration boundary explicitly.
        self._numpy_metric("sampling_entropy").fill(self.sampler.sampling_entropy)
        self._numpy_metric("sampling_top1_prob").fill(self.sampler.sampling_top1_prob)
        self._numpy_metric("sampling_top1_bin").fill(self.sampler.sampling_top1_bin)

    def _numpy_metric(self, name: str) -> np.ndarray:
        value = self.metrics[name]
        if not isinstance(value, np.ndarray):
            raise TypeError(
                f"MotionCommand metric '{name}' must remain np.ndarray until its "
                "Numba kernel migrates to Torch."
            )
        return value

    def _update_error_metrics(self, rows: np.ndarray) -> None:
        """Recompute the row-wise error metrics for the given rows."""
        update_motion_metrics_kernel(
            rows,
            self.anchor_body_idx,
            self._body_pos_w,
            self._robot_body_pos_w,
            self._motion_data.body_quat_w,
            self._robot_body_quat_w,
            self._motion_data.body_lin_vel_w,
            self._robot_body_lin_vel_w,
            self._motion_data.body_ang_vel_w,
            self._robot_body_ang_vel_w,
            self.body_pos_relative_w,
            self.body_quat_relative_w,
            self._motion_data.joint_pos,
            self.robot_joint_pos,
            self._motion_data.joint_vel,
            self.robot_joint_vel,
            self._numpy_metric("error_anchor_pos"),
            self._numpy_metric("error_anchor_rot"),
            self._numpy_metric("error_anchor_lin_vel"),
            self._numpy_metric("error_anchor_ang_vel"),
            self._numpy_metric("error_body_pos"),
            self._numpy_metric("error_body_rot"),
            self._numpy_metric("error_body_lin_vel"),
            self._numpy_metric("error_body_ang_vel"),
            self._numpy_metric("error_joint_pos"),
            self._numpy_metric("error_joint_vel"),
        )

    def _resample_command(self, env_ids: torch.Tensor) -> None:
        """Resample motion frames and stage the corresponding state writes."""
        ids = env_ids.detach().cpu().numpy()
        frames = self.sampler.sample_frames(ids)
        motion = self.motion.get_motion_at_frame(frames)
        count = env_ids.numel()
        pose = self._env.rng.uniform(
            self._pose_range[:, 0], self._pose_range[:, 1], size=(count, 6)
        )
        velocity = self._env.rng.uniform(
            self._velocity_range[:, 0], self._velocity_range[:, 1], size=(count, 6)
        )
        root_pos = motion.body_pos_w[:, 0].copy()
        root_pos += self._env.scene.env_origins[ids]
        root_pos += pose[:, :3]
        root_quat = np_quat_mul(
            np_quat_from_euler_xyz(pose[:, 3], pose[:, 4], pose[:, 5]),
            motion.body_quat_w[:, 0],
        )
        root_lin_vel = motion.body_lin_vel_w[:, 0] + velocity[:, :3]
        root_ang_vel = motion.body_ang_vel_w[:, 0] + velocity[:, 3:]
        joint_pos = motion.joint_pos.copy()
        joint_pos += self._env.rng.uniform(
            *self._joint_position_range,
            size=joint_pos.shape,
        )
        limits = self.robot.data.soft_joint_pos_limits
        np.clip(joint_pos, limits[:, 0], limits[:, 1], out=joint_pos)
        self.robot.write_joint_state_to_sim(joint_pos, motion.joint_vel, env_ids=ids)
        root_state = np.concatenate((root_pos, root_quat, root_lin_vel, root_ang_vel), axis=-1)
        self.robot.write_root_state_to_sim(root_state, env_ids=ids)
        self._ingest_motion_rows(ids, motion)
        self._resample_ingested_ids = ids
        self._resample_motion = motion

    def _update_command(self, env_ids: torch.Tensor | None) -> None:
        self._post_compute_env_ids = (
            env_ids.detach().cpu().numpy() if isinstance(env_ids, torch.Tensor) else env_ids
        )
        self._tensor_post_compute_env_ids = env_ids if isinstance(env_ids, torch.Tensor) else None
        if env_ids is not None:
            ingested = self._resample_ingested_ids
            self._resample_ingested_ids = None
            host_rows = env_ids.detach().cpu().numpy()
            if ingested is None or not np.array_equal(ingested, host_rows):
                self._refresh_motion(host_rows)
            return
        self._resample_ingested_ids = None
        terminated = self._env.termination_manager.terminated
        if isinstance(terminated, torch.Tensor):
            terminated = terminated.detach().cpu().numpy()
        self.sampler.update_failure_stats(terminated)
        active_ids = np.flatnonzero(~self._env.reset_buf).astype(np.int32, copy=False)
        wrap_ids = self.sampler.step(active_ids)
        if len(wrap_ids) and not self.cfg.params.truncate_on_clip_end:
            self._resample_command(torch.from_numpy(wrap_ids).to(self._device))
        self._refresh_motion()

    def post_compute(self) -> None:
        # On the reset path only the reset rows changed (via the committed
        # set_state writes and the motion resample), so refresh just those rows.
        env_ids = self._post_compute_env_ids
        self._refresh_robot_state(force=True, env_ids=env_ids)
        self._refresh_relative_state(env_ids)


class TensorMotionCommand(MotionCommand):
    """Motion command whose public carrier and selected reset stay on device."""

    cfg: TensorMotionCommandCfg  # pyright: ignore[reportIncompatibleVariableOverride]

    def __init__(self, cfg: TensorMotionCommandCfg, env: ManagerBasedRlEnv):
        super().__init__(cfg, env)

    def _prepare_tensor_carrier(self) -> None:
        """Allocate Torch buffers before Manager probes the command carrier."""
        device = self._device
        self.last_step_timing_ms: dict[str, float] = {}
        self._motion_feature_layout: (
            tuple[dict[str, tuple[int, ...]], dict[str, tuple[int, int]]] | None
        ) = None
        self._last_reset_payload_validated = False
        num_bodies = len(self.cfg.body_names)
        num_joints = self.motion.num_joints
        obs_body_names = self._resolve_obs_body_names()
        num_obs_bodies = len(obs_body_names)
        self.time_steps = torch.as_tensor(
            np.array(self.sampler.current_frames, dtype=np.int32, copy=True), device=device
        )
        self.current_clip_end_frames = torch.as_tensor(
            np.array(self.sampler.current_clip_end_frames, dtype=np.int32, copy=True),
            device=device,
        )
        self._motion_features = self._make_motion_features(device)
        self._clip_offsets_torch = torch.as_tensor(
            self.motion.clip_offsets, dtype=torch.int64, device=device
        )
        self._clip_end_frames_torch = torch.as_tensor(
            self.motion.clip_end_frames, dtype=torch.int64, device=device
        )
        if obs_body_names:
            obs_body_ids = self._obs_robot_body_ids
            assert obs_body_ids is not None
            obs_motion_body_ids = self.robot.motion_body_ids[obs_body_ids]
            self.obs_motion = self._make_motion_loader(self.cfg.motion_file, obs_motion_body_ids)
            if (
                self.obs_motion.fps != self.motion.fps
                or self.obs_motion.num_frames != self.motion.num_frames
                or self.obs_motion.num_joints != self.motion.num_joints
                or self.obs_motion.num_bodies != num_obs_bodies
            ):
                raise ValueError(
                    "TensorMotionCommand observation motion view is inconsistent with "
                    "the reward motion view"
                )
            self._obs_motion_features = self._make_obs_motion_features(self.obs_motion, device)
        else:
            self.obs_motion = None
            self._obs_motion_features = torch.empty(
                (self.motion.num_frames, 0), dtype=torch.float32, device=device
            )
        self._motion_data = MotionData(
            joint_pos=cast("np.ndarray", torch.empty((self.num_envs, num_joints), device=device)),
            joint_vel=cast("np.ndarray", torch.empty((self.num_envs, num_joints), device=device)),
            body_pos_w=cast(
                "np.ndarray", torch.empty((self.num_envs, num_bodies, 3), device=device)
            ),
            body_quat_w=cast(
                "np.ndarray", torch.empty((self.num_envs, num_bodies, 4), device=device)
            ),
            body_lin_vel_w=cast(
                "np.ndarray", torch.empty((self.num_envs, num_bodies, 3), device=device)
            ),
            body_ang_vel_w=cast(
                "np.ndarray", torch.empty((self.num_envs, num_bodies, 3), device=device)
            ),
        )
        self._command = torch.empty((self.num_envs, num_joints * 2), device=device)
        self._body_pos_w = torch.empty((self.num_envs, num_bodies, 3), device=device)
        self.body_pos_relative_w = torch.empty_like(self._body_pos_w)
        self.body_quat_relative_w = torch.empty((self.num_envs, num_bodies, 4), device=device)
        self.motion_anchor_pos_b = torch.empty((self.num_envs, 3), device=device)
        self.motion_anchor_ori_b = torch.empty((self.num_envs, 6), device=device)
        self.robot_body_pos_b = torch.empty_like(self._body_pos_w)
        self.robot_body_ori_b = torch.empty((self.num_envs, num_bodies, 6), device=device)
        self.joint_default_bias = torch.zeros(
            (self.num_envs, num_joints), dtype=torch.float32, device=device
        )
        self._robot_body_pos_w = torch.empty_like(self._body_pos_w)
        self._robot_body_quat_w = torch.empty((self.num_envs, num_bodies, 4), device=device)
        self._robot_body_lin_vel_w = torch.empty_like(self._body_pos_w)
        self._robot_body_ang_vel_w = torch.empty_like(self._body_pos_w)
        self._obs_robot_body_pos_w = torch.empty(
            (self.num_envs, num_obs_bodies, 3), dtype=torch.float32, device=device
        )
        self._obs_robot_body_quat_w = torch.empty(
            (self.num_envs, num_obs_bodies, 4), dtype=torch.float32, device=device
        )
        self._obs_robot_body_lin_vel_w = torch.empty_like(self._obs_robot_body_pos_w)
        self._obs_robot_body_ang_vel_w = torch.empty_like(self._obs_robot_body_pos_w)
        self._robot_joint_pos = torch.empty(
            (self.num_envs, num_joints), dtype=torch.float32, device=device
        )
        self._robot_joint_vel = torch.empty(
            (self.num_envs, num_joints), dtype=torch.float32, device=device
        )
        self._env_origins = torch.as_tensor(
            np.array(self._env.scene.env_origins, dtype=np.float32, copy=True), device=device
        )
        self._soft_joint_limits = torch.as_tensor(
            np.array(self.robot.data.soft_joint_pos_limits, dtype=np.float32, copy=True),
            device=device,
        )
        self._pose_range_torch = torch.as_tensor(
            np.array(self._pose_range, copy=True), device=device
        )
        self._velocity_range_torch = torch.as_tensor(
            np.array(self._velocity_range, copy=True), device=device
        )
        self._joint_position_range_torch = torch.as_tensor(
            self._joint_position_range, device=device
        )
        self._reset_joint_values = torch.empty(
            (self.num_envs, num_joints), dtype=torch.float32, device=device
        )
        self._reset_root_state = torch.empty(
            (self.num_envs, 13), dtype=torch.float32, device=device
        )
        if self.cfg.params.sampling_mode not in ("adaptive", "mixed"):
            raise NotImplementedError(
                "TensorMotionCommand tensor sampler requires adaptive or mixed sampling"
            )
        self.tensor_sampler = TensorMotionSampler(
            mode=self.cfg.params.sampling_mode,
            num_envs=self.num_envs,
            num_frames=self.motion.num_frames,
            clip_offsets=self.motion.clip_offsets,
            clip_end_frames=self.motion.clip_end_frames,
            bin_count=self.sampler.bin_count,
            adaptive_lambda=self.cfg.params.adaptive_lambda,
            adaptive_kernel_size=self.cfg.params.adaptive_kernel_size,
            adaptive_uniform_ratio=self.cfg.params.adaptive_uniform_ratio,
            adaptive_alpha=self.cfg.params.adaptive_alpha,
            start_ratio=self.cfg.params.sampling_start_ratio,
            initial_frames=self.sampler.current_frames,
            initial_clip_end_frames=self.sampler.current_clip_end_frames,
            device=device,
        )
        for name in self.metrics:
            self.metrics[name] = torch.zeros(self.num_envs, dtype=torch.float32, device=device)
        self._obs_future_cache_step = -1
        self._obs_future_cache: dict[tuple[int, ...], TensorMotionObsFuture] = {}
        self._refresh_motion()

    def _resolve_obs_body_names(self) -> tuple[str, ...]:
        value = self.cfg.params.obs_body_names
        return () if value is None else tuple(value)

    @staticmethod
    def _make_obs_motion_features(motion: MotionLoader, device: torch.device) -> torch.Tensor:
        """Cache the observation body-sliced motion table on the device."""
        blocks = (
            motion.body_pos_w,
            motion.body_quat_w,
            motion.body_lin_vel_w,
            motion.body_ang_vel_w,
        )
        widths = tuple(int(np.asarray(value).shape[-1]) for value in blocks)
        if widths != _OBS_MOTION_BLOCK_WIDTHS:
            raise ValueError(
                "Observation motion feature blocks must follow the per-body "
                f"pos/quat/lin_vel/ang_vel layout {_OBS_MOTION_BLOCK_WIDTHS}, got {widths}"
            )
        # Interleave each body's state blocks so a row reshapes directly to
        # ``(bodies, pos_w | quat_w | lin_vel_w | ang_vel_w)`` in obs_future.
        host = np.concatenate(
            [
                np.asarray(value, dtype=np.float32).reshape(value.shape[0], motion.num_bodies, -1)
                for value in blocks
            ],
            axis=2,
        ).reshape(motion.num_frames, -1)
        return torch.from_numpy(np.ascontiguousarray(host)).to(device=device)

    def _make_motion_features(self, device: torch.device) -> torch.Tensor:
        """Cache the complete motion dataset as one device-resident table."""
        arrays = (
            self.motion.joint_pos,
            self.motion.joint_vel,
            self.motion.body_pos_w,
            self.motion.body_quat_w,
            self.motion.body_lin_vel_w,
            self.motion.body_ang_vel_w,
        )
        host = np.concatenate(
            [np.asarray(value, dtype=np.float32).reshape(value.shape[0], -1) for value in arrays],
            axis=1,
        )
        return torch.from_numpy(np.ascontiguousarray(host)).to(device=device)

    def _motion_packet(self, frames: np.ndarray | torch.Tensor) -> torch.Tensor:
        """Gather motion rows from the device-resident feature table."""
        rows = (
            frames
            if isinstance(frames, torch.Tensor)
            else torch.as_tensor(frames, dtype=torch.int64, device=self._device)
        )
        if rows.device != self._device or rows.dtype != torch.int64:
            rows = rows.to(device=self._device, dtype=torch.int64)
        return self._motion_features.index_select(0, rows)

    def obs_future(self, future_steps: Iterable[int]) -> TensorMotionObsFuture:
        """Gather future motion rows over the configured observation bodies."""
        if not self.obs_body_names:
            raise ValueError("MotionCommand observation bodies are not configured")
        steps = tuple(int(step) for step in future_steps)
        if not steps:
            raise ValueError("obs_future requires at least one future step")
        step_counter = self._env.common_step_counter
        if self._obs_future_cache_step != step_counter:
            self._obs_future_cache.clear()
            self._obs_future_cache_step = step_counter
        cached = self._obs_future_cache.get(steps)
        if cached is not None:
            return cached

        frames = cast(torch.Tensor, self.time_steps).to(dtype=torch.int64)
        offsets = self._clip_offsets_torch.index_select(0, self._frame_clip_indices(frames))
        ends = self.current_clip_end_frames.to(dtype=torch.int64)
        shifted = frames[:, None] + torch.as_tensor(steps, dtype=torch.int64, device=self._device)
        shifted = torch.maximum(torch.minimum(shifted, ends[:, None]), offsets[:, None])
        rows = shifted.reshape(-1)
        packets = self._obs_motion_features.index_select(0, rows).view(
            self.num_envs, len(steps), *self._obs_motion_features.shape[1:]
        )
        width_pos = _OBS_MOTION_BLOCK_WIDTHS[0]
        width_quat_end = width_pos + _OBS_MOTION_BLOCK_WIDTHS[1]
        width_lin_vel_end = width_quat_end + _OBS_MOTION_BLOCK_WIDTHS[2]
        body_packets = packets.view(
            self.num_envs, len(steps), len(self.obs_body_names), _OBS_MOTION_WIDTH_PER_BODY
        )
        result = TensorMotionObsFuture(
            future_steps=steps,
            ref_joint_pos=self._motion_features[:, : self.motion.num_joints]
            .index_select(0, rows)
            .view(self.num_envs, len(steps), self.motion.num_joints),
            ref_body_pos_w=body_packets[..., :width_pos] + self._env_origins[:, None, None, :],
            ref_body_quat_w=body_packets[..., width_pos:width_quat_end],
            ref_body_lin_vel_w=body_packets[..., width_quat_end:width_lin_vel_end],
            ref_body_ang_vel_w=body_packets[..., width_lin_vel_end:],
        )
        self._obs_future_cache[steps] = result
        return result

    def _frame_clip_indices(self, frames: torch.Tensor) -> torch.Tensor:
        """Map device frame indices to clip rows without a host transfer."""
        positions = torch.searchsorted(self._clip_offsets_torch[1:], frames, right=True)
        return positions.clamp_(min=0, max=self.motion.num_clips - 1)

    def _defer_read_phase_binding(self) -> None:
        """Torch carriers were allocated eagerly; defer state-view binding."""

    def bind_read_phase(self) -> None:
        if self._bind_read_phase:
            return
        self._bind_read_phase = True
        self._refresh_motion()
        read_plan = getattr(self._env.scene, "_tensor_read_plan", None)
        if read_plan is not None:
            read_plan.refresh()
            self._refresh_robot_state(force=True)
        else:
            self._seed_robot_state_torch_from_defaults()
        self._refresh_relative_state()
        self._update_metrics(self._tensor_all_rows)

    @property
    def tensor_carrier(self) -> bool:
        return True

    @property
    def tensor_body_names(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys((*self.cfg.body_names, *self.obs_body_names)))

    @property
    def tensor_current_clip_end_frames(self) -> torch.Tensor:
        return cast(torch.Tensor, self.current_clip_end_frames)

    @property
    def device_robot_joint_pos(self) -> torch.Tensor:
        read_plan = getattr(self._env.scene, "_tensor_read_plan", None)
        if read_plan is not None and read_plan.ready:
            self._robot_joint_pos = read_plan.joint_tensor_view(self.robot).joint_pos
        return self._robot_joint_pos

    @property
    def device_robot_joint_vel(self) -> torch.Tensor:
        read_plan = getattr(self._env.scene, "_tensor_read_plan", None)
        if read_plan is not None and read_plan.ready:
            self._robot_joint_vel = read_plan.joint_tensor_view(self.robot).joint_vel
        return self._robot_joint_vel

    @property
    def obs_robot_body_pos_w(self) -> torch.Tensor:
        self._refresh_robot_state()
        return self._obs_robot_body_pos_w

    @property
    def obs_robot_body_quat_w(self) -> torch.Tensor:
        self._refresh_robot_state()
        return self._obs_robot_body_quat_w

    @property
    def obs_robot_body_lin_vel_w(self) -> torch.Tensor:
        self._refresh_robot_state()
        return self._obs_robot_body_lin_vel_w

    @property
    def obs_robot_body_ang_vel_w(self) -> torch.Tensor:
        self._refresh_robot_state()
        return self._obs_robot_body_ang_vel_w

    @property
    def obs_robot_root_pos_w(self) -> torch.Tensor:
        return self.obs_robot_body_pos_w[:, self.obs_root_body_idx]

    @property
    def obs_robot_root_quat_w(self) -> torch.Tensor:
        return self.obs_robot_body_quat_w[:, self.obs_root_body_idx]

    def reset(
        self,
        env_ids: torch.Tensor | slice | None,
        *,
        publish_metrics: bool = True,
    ) -> dict[str, float]:
        rows = (
            env_ids.to(dtype=torch.int64)
            if isinstance(env_ids, torch.Tensor)
            else self._tensor_all_rows
            if env_ids is None
            else self._tensor_all_rows[env_ids]
        )
        if publish_metrics:
            self._update_torch_error_metrics(rows)
        if self._env.torch_rng is None:
            raise NotImplementedError(
                "TensorMotionCommand reset requires the Manager-owned Torch generator"
            )
        lower, upper = self._joint_default_position_range
        self.joint_default_bias[rows] = self._env.torch_rng.uniform(
            lower,
            upper,
            (rows.numel(), self.motion.num_joints),
            dtype=torch.float32,
        )
        self._last_reset_payload_validated = False
        return CommandTerm.reset(self, rows, publish_metrics=publish_metrics)

    def sampler_reset_diagnostics(self) -> dict[str, float]:
        return {
            "reset_done_sampler_host_transfer_count": float(
                self.tensor_sampler.diagnostics.reset_host_row_transfers
            )
        }

    def _refresh_motion(self, env_ids: np.ndarray | None = None) -> None:
        del env_ids
        self._refresh_motion_torch()

    def _refresh_relative_state(self, env_ids: np.ndarray | None = None) -> None:
        del env_ids
        self._refresh_relative_state_torch()

    def _refresh_robot_state(
        self, *, force: bool = False, env_ids: np.ndarray | None = None
    ) -> None:
        del env_ids
        self._refresh_robot_state_torch(force=force, rows=None)

    def _update_metrics(self, env_ids: torch.Tensor | None = None) -> None:
        if env_ids is not None:
            self._update_torch_error_metrics(env_ids)
        self._update_torch_sampling_metrics()

    @property
    def uses_tensor_reset_rows(self) -> bool:
        return True

    @property
    def last_reset_payload_validated(self) -> bool:
        """Whether the current reset payload already passed owner validation."""
        return bool(self._last_reset_payload_validated)

    def _resample_command(self, env_ids: torch.Tensor) -> None:
        rows = env_ids.to(dtype=torch.int64, device=self._device)
        sampler_started = time.perf_counter()
        if self._env.torch_rng is None:
            raise NotImplementedError("TensorMotionCommand reset requires the Manager Torch RNG")
        frames = self.tensor_sampler.sample_frames(rows, self._env.torch_rng.generator)
        sampler_ms = (time.perf_counter() - sampler_started) * 1000.0
        sampler_dispatch_ms = getattr(self.tensor_sampler, "last_reset_dispatch_ms", 0.0)
        packet_started = time.perf_counter()
        packet = self._motion_packet(frames)
        packet_ms = (time.perf_counter() - packet_started) * 1000.0
        tails, offsets = self._cached_motion_feature_shapes()
        if self._env.torch_rng is None:
            raise NotImplementedError(
                "TensorMotionCommand reset sampling requires the Manager-owned Torch generator"
            )
        pose = self._env.torch_rng.uniform(
            self._pose_range_torch[:, 0],
            self._pose_range_torch[:, 1],
            (rows.numel(), 6),
            dtype=torch.float32,
        )
        velocity = self._env.torch_rng.uniform(
            self._velocity_range_torch[:, 0],
            self._velocity_range_torch[:, 1],
            (rows.numel(), 6),
            dtype=torch.float32,
        )
        rng_ms = (time.perf_counter() - sampler_started) * 1000.0 - sampler_ms - packet_ms
        values_started = time.perf_counter()
        joint_start, joint_end = offsets["joint_pos"]
        vel_start, vel_end = offsets["joint_vel"]
        count = rows.numel()
        origins = (
            self._env_origins if count == self.num_envs else self._env_origins.index_select(0, rows)
        )
        joint_pos = self._reset_joint_values[:count]
        root_state = self._reset_root_state[:count]
        joint_noise = self._env.torch_rng.uniform(
            self._joint_position_range_torch[0],
            self._joint_position_range_torch[1],
            joint_pos.shape,
            dtype=torch.float32,
        )
        _bind_compiled_motion_reset_values()(
            packet[:, joint_start:joint_end],
            packet[:, vel_start:vel_end],
            packet[:, offsets["body_pos_w"][0] : offsets["body_pos_w"][1]].view(
                count, *tails["body_pos_w"]
            ),
            packet[:, offsets["body_quat_w"][0] : offsets["body_quat_w"][1]].view(
                count, *tails["body_quat_w"]
            ),
            packet[:, offsets["body_lin_vel_w"][0] : offsets["body_lin_vel_w"][1]].view(
                count, *tails["body_lin_vel_w"]
            ),
            packet[:, offsets["body_ang_vel_w"][0] : offsets["body_ang_vel_w"][1]].view(
                count, *tails["body_ang_vel_w"]
            ),
            origins,
            pose,
            velocity,
            joint_noise,
            self._soft_joint_limits,
            joint_pos,
            root_state,
        )
        motion_joint_vel = packet[:, vel_start:vel_end].contiguous()
        values_ms = (time.perf_counter() - values_started) * 1000.0
        root_values_ms = 0.0
        root_write_started = time.perf_counter()
        self.robot.write_motion_state_tensor_to_sim(
            root_state=root_state,
            position=joint_pos,
            velocity=motion_joint_vel,
            env_ids=rows,
        )
        root_write_ms = (time.perf_counter() - root_write_started) * 1000.0
        construction_ms = values_ms + root_values_ms + root_write_ms
        publish_started = time.perf_counter()
        self._ingest_motion_packet(rows, packet, origins=origins)
        publish_ms = (time.perf_counter() - publish_started) * 1000.0
        self._resample_ingested_ids = None
        self._resample_motion = None
        self._tensor_resample_ingested = env_ids
        self._last_reset_payload_validated = True
        self.last_reset_timing_ms.update(
            {
                "reset_done_motion_sampler_ms": sampler_ms,
                "reset_done_motion_sampler_dispatch_ms": sampler_dispatch_ms,
                "reset_done_motion_packet_ms": packet_ms,
                "reset_done_motion_reset_rng_ms": rng_ms,
                "reset_done_motion_reset_values_ms": values_ms + root_values_ms,
                "reset_done_motion_reset_construction_ms": construction_ms,
                "reset_done_motion_reset_write_ms": root_write_ms,
                "reset_done_motion_reset_publish_ms": publish_ms,
            }
        )
        sync_started = time.perf_counter()
        self._sync_tensor_sampler_state(rows)
        self.last_reset_timing_ms["reset_done_motion_sampler_sync_ms"] = (
            time.perf_counter() - sync_started
        ) * 1000.0

    def _update_command(self, env_ids: torch.Tensor | None) -> None:
        self._tensor_post_compute_env_ids = env_ids
        timing = getattr(self, "last_step_timing_ms", None)
        if timing is None:
            timing = {}
            self.last_step_timing_ms = timing
        if env_ids is not None:
            ingested = self._tensor_resample_ingested
            self._resample_ingested_ids = None
            self._tensor_resample_ingested = None
            if (
                ingested is None
                or ingested.shape != env_ids.shape
                or not bool(torch.equal(ingested, env_ids))
            ):
                self._refresh_motion_torch(env_ids)
            return
        self._resample_ingested_ids = None
        self._tensor_resample_ingested = None
        failure_started = time.perf_counter()
        terminated = cast(torch.Tensor, self._env.termination_manager.terminated)
        self.tensor_sampler.update_failure_stats(terminated)
        timing["update_state_motion_failure_stats_ms"] = (
            time.perf_counter() - failure_started
        ) * 1000.0
        sampler_started = time.perf_counter()
        wrap_rows = self._step_tensor_sampler()
        timing["update_state_motion_step_sampler_ms"] = (
            time.perf_counter() - sampler_started
        ) * 1000.0
        if wrap_rows.numel() and not self.cfg.params.truncate_on_clip_end:
            self._resample_command(wrap_rows)
        refresh_started = time.perf_counter()
        self._refresh_motion_torch()
        timing["update_state_motion_refresh_current_ms"] = (
            time.perf_counter() - refresh_started
        ) * 1000.0

    def _step_tensor_sampler(self) -> torch.Tensor:
        """Advance the device frame carrier once without a host row transfer."""
        active = ~cast(torch.Tensor, self._env.reset_buf)
        return self.tensor_sampler.step_full(
            active,
            cast(torch.Tensor, self.time_steps),
        )

    def post_compute(self) -> None:
        rows = self._tensor_post_compute_env_ids
        read_plan = getattr(self._env.scene, "_tensor_read_plan", None)
        view_started = time.perf_counter()
        if read_plan is None or not read_plan.ready:
            if not self._bind_read_phase:
                return
            raise RuntimeError("TensorMotionCommand requires a refreshed scene tensor read phase")
        view = read_plan.body_tensor_view(self.robot, self.cfg.body_names)
        obs_view = (
            read_plan.body_tensor_view(self.robot, self.obs_body_names)
            if self.obs_body_names
            else None
        )
        view_ms = (time.perf_counter() - view_started) * 1000.0
        robot_started = time.perf_counter()
        _bind_compiled_motion_post_compute()(
            self._tensor_all_rows if rows is None else rows,
            self.anchor_body_idx,
            view.pos_w,
            view.quat_w,
            view.lin_vel_w,
            view.ang_vel_w,
            self._robot_body_pos_w,
            self._robot_body_quat_w,
            self._robot_body_lin_vel_w,
            self._robot_body_ang_vel_w,
            cast("torch.Tensor", self._motion_data.body_pos_w),
            self._body_pos_w,
            cast("torch.Tensor", self._motion_data.body_quat_w),
            self.body_pos_relative_w,
            self.body_quat_relative_w,
            self.motion_anchor_pos_b,
            self.motion_anchor_ori_b,
            self.robot_body_pos_b,
            self.robot_body_ori_b,
        )
        if obs_view is not None:
            obs_target = slice(None) if rows is None else rows
            self._obs_robot_body_pos_w[obs_target] = obs_view.pos_w[obs_target]
            self._obs_robot_body_quat_w[obs_target] = obs_view.quat_w[obs_target]
            self._obs_robot_body_lin_vel_w[obs_target] = obs_view.lin_vel_w[obs_target]
            self._obs_robot_body_ang_vel_w[obs_target] = obs_view.ang_vel_w[obs_target]
        kernel_ms = (time.perf_counter() - robot_started) * 1000.0
        rebind_started = time.perf_counter()
        joint_view = read_plan.joint_tensor_view(self.robot)
        self._robot_joint_pos = joint_view.joint_pos
        self._robot_joint_vel = joint_view.joint_vel
        self._robot_cache_step = self._env.common_step_counter
        rebind_ms = (time.perf_counter() - rebind_started) * 1000.0
        self.last_post_compute_timing_ms = {
            "reset_done_motion_robot_refresh_ms": kernel_ms,
            "reset_done_motion_post_compute_view_ms": view_ms,
            "reset_done_motion_post_compute_kernel_ms": kernel_ms,
            "reset_done_motion_post_compute_rebind_ms": rebind_ms,
        }
        self.last_post_compute_timing_ms["reset_done_motion_relative_refresh_ms"] = 0.0

    def _sync_tensor_sampler_state(self, rows: torch.Tensor | None = None) -> None:
        selector = self._tensor_all_rows if rows is None else rows
        cast(torch.Tensor, self.time_steps).index_copy_(
            0, selector, self.tensor_sampler.current_frames.index_select(0, selector)
        )
        cast(torch.Tensor, self.current_clip_end_frames).index_copy_(
            0, selector, self.tensor_sampler.current_clip_end_frames.index_select(0, selector)
        )

    @property
    def sampler_host_transfers(self) -> int:
        """Count explicit sampler device-to-host transfers since the last read."""
        return self.tensor_sampler.diagnostics.total

    @sampler_host_transfers.setter
    def sampler_host_transfers(self, value: int) -> None:
        del value

    @staticmethod
    def _validate_cfg(cfg: MotionCommandCfg) -> None:
        _validate_motion_command_cfg(cfg)

    def _refresh_motion_torch(self, rows: torch.Tensor | None = None) -> None:
        """Gather motion rows and publish the device-resident command carrier."""
        frames: np.ndarray | torch.Tensor
        if rows is None:
            frames = (
                cast(torch.Tensor, self.time_steps)
                if isinstance(self.time_steps, torch.Tensor)
                else self.sampler.current_frames
            )
        elif isinstance(self.time_steps, torch.Tensor):
            frames = cast(torch.Tensor, self.time_steps)[rows]
        else:
            frames = self.sampler.current_frames[rows.detach().cpu().numpy()]
        self._ingest_motion_packet(
            self._tensor_all_rows if rows is None else rows, self._motion_packet(frames)
        )
        self._obs_future_cache_step = -1
        self.__dict__.get("_obs_future_cache", {}).clear()

    def _refresh_robot_state_torch(
        self, *, force: bool = False, rows: torch.Tensor | None = None
    ) -> None:
        step = self._env.common_step_counter
        if not force and self._robot_cache_step == step:
            return
        read_plan = getattr(self._env.scene, "_tensor_read_plan", None)
        if read_plan is None or not read_plan.ready:
            # Manager term construction probes command carriers before the scene
            # read plan exists. Before `bind_read_phase`, immutable defaults are
            # the explicit cold seed; after binding, a missing phase fails closed.
            if not self._bind_read_phase:
                self._seed_robot_state_torch_from_defaults()
                return
            raise RuntimeError("TensorMotionCommand requires a refreshed scene tensor read phase")
        view = read_plan.body_tensor_view(self.robot, self.cfg.body_names)
        row_selector = self._tensor_all_rows if rows is None else rows
        _bind_compiled_motion_robot_refresh()(
            row_selector,
            view.pos_w,
            view.quat_w,
            view.lin_vel_w,
            view.ang_vel_w,
            self._robot_body_pos_w,
            self._robot_body_quat_w,
            self._robot_body_lin_vel_w,
            self._robot_body_ang_vel_w,
        )
        if self.obs_body_names:
            obs_view = read_plan.body_tensor_view(self.robot, self.obs_body_names)
            self._obs_robot_body_pos_w[row_selector] = obs_view.pos_w[row_selector]
            self._obs_robot_body_quat_w[row_selector] = obs_view.quat_w[row_selector]
            self._obs_robot_body_lin_vel_w[row_selector] = obs_view.lin_vel_w[row_selector]
            self._obs_robot_body_ang_vel_w[row_selector] = obs_view.ang_vel_w[row_selector]
        joint_view = read_plan.joint_tensor_view(self.robot)
        self._robot_joint_pos = joint_view.joint_pos
        self._robot_joint_vel = joint_view.joint_vel
        self._robot_cache_step = step

    def _seed_robot_state_torch_from_defaults(self) -> None:
        """Cold proxy path: seed immutable defaults before CUDA view binding."""
        default_root = self.robot.data.default_root_state
        if default_root is None:
            self._robot_body_pos_w.zero_()
            self._robot_body_quat_w.zero_()
            self._robot_body_quat_w[..., 0] = 1.0
        else:
            root = torch.as_tensor(np.array(default_root, copy=True), device=self._device)
            self._robot_body_pos_w.copy_(root[:, None, 0:3])
            self._robot_body_quat_w.copy_(root[:, None, 3:7])
        self._robot_body_lin_vel_w.zero_()
        self._robot_body_ang_vel_w.zero_()
        default_joint_pos = torch.as_tensor(
            np.array(self.robot.data.default_joint_pos, dtype=np.float32, copy=True),
            device=self._device,
        )
        self._robot_joint_pos.copy_(default_joint_pos.expand(self.num_envs, -1))
        self._robot_joint_vel.zero_()
        if self.obs_body_names:
            default_root = self.robot.data.default_root_state
            if default_root is None:
                self._obs_robot_body_pos_w.zero_()
                self._obs_robot_body_quat_w.zero_()
                self._obs_robot_body_quat_w[..., 0] = 1.0
            else:
                root = torch.as_tensor(np.array(default_root, copy=True), device=self._device)
                self._obs_robot_body_pos_w.copy_(root[:, None, 0:3])
                self._obs_robot_body_quat_w.copy_(root[:, None, 3:7])
            self._obs_robot_body_lin_vel_w.zero_()
            self._obs_robot_body_ang_vel_w.zero_()
        self._robot_cache_step = self._env.common_step_counter

    def _motion_feature_tail_shapes(self) -> dict[str, tuple[int, ...]]:
        return {
            "joint_pos": (self.motion.num_joints,),
            "joint_vel": (self.motion.num_joints,),
            "body_pos_w": (len(self.cfg.body_names), 3),
            "body_quat_w": (len(self.cfg.body_names), 4),
            "body_lin_vel_w": (len(self.cfg.body_names), 3),
            "body_ang_vel_w": (len(self.cfg.body_names), 3),
        }

    def _cached_motion_feature_shapes(
        self,
    ) -> tuple[dict[str, tuple[int, ...]], dict[str, tuple[int, int]]]:
        cached = self._motion_feature_layout
        if cached is not None:
            return cached
        tails = self._motion_feature_tail_shapes()
        offset = 0
        offsets: dict[str, tuple[int, int]] = {}
        for name, tail in tails.items():
            width = int(np.prod(tail, dtype=np.int64))
            offsets[name] = (offset, offset + width)
            offset += width
        cached = (tails, offsets)
        self._motion_feature_layout = cached
        return cached

    def _ingest_motion_packet(
        self,
        rows: torch.Tensor,
        packet: torch.Tensor,
        *,
        origins: torch.Tensor | None = None,
    ) -> None:
        """Scatter one device motion packet into the command carriers."""
        # Selected reset can ingest newly sampled motion packets within the
        # same common-step counter. Explicitly invalidate the future cache;
        # otherwise reset observations reuse the pre-reset references.
        self._obs_future_cache_step = -1
        self._obs_future_cache.clear()
        count = rows.numel()
        selected_origins = (
            self._env_origins if count == self.num_envs else self._env_origins.index_select(0, rows)
        )
        if origins is None:
            origins = selected_origins
        _bind_compiled_motion_packet_ingest()(
            rows,
            packet,
            cast("torch.Tensor", self._motion_data.joint_pos),
            cast("torch.Tensor", self._motion_data.joint_vel),
            cast("torch.Tensor", self._motion_data.body_pos_w),
            cast("torch.Tensor", self._motion_data.body_quat_w),
            cast("torch.Tensor", self._motion_data.body_lin_vel_w),
            cast("torch.Tensor", self._motion_data.body_ang_vel_w),
            self._body_pos_w,
            self._command,
            origins,
            num_bodies=len(self.cfg.body_names),
            num_joints=self.motion.num_joints,
        )

    def _refresh_relative_state_torch(self, rows: torch.Tensor | None = None) -> None:
        row_selector = self._tensor_all_rows if rows is None else rows
        _bind_compiled_motion_relative_state()(
            row_selector,
            self.anchor_body_idx,
            cast("torch.Tensor", self._motion_data.body_pos_w),
            self._body_pos_w,
            cast("torch.Tensor", self._motion_data.body_quat_w),
            self._robot_body_pos_w,
            self._robot_body_quat_w,
            self.body_pos_relative_w,
            self.body_quat_relative_w,
            self.motion_anchor_pos_b,
            self.motion_anchor_ori_b,
            self.robot_body_pos_b,
            self.robot_body_ori_b,
        )

    def _torch_metrics(self) -> tuple[torch.Tensor, ...]:
        names = (
            "error_anchor_pos",
            "error_anchor_rot",
            "error_anchor_lin_vel",
            "error_anchor_ang_vel",
            "error_body_pos",
            "error_body_rot",
            "error_body_lin_vel",
            "error_body_ang_vel",
            "error_joint_pos",
            "error_joint_vel",
        )
        values = tuple(self.metrics[name] for name in names)
        if any(not isinstance(value, torch.Tensor) for value in values):
            raise TypeError("TensorMotionCommand row metrics must remain Torch tensors")
        return cast("tuple[torch.Tensor, ...]", values)

    def _update_torch_error_metrics(self, rows: torch.Tensor) -> None:
        _bind_compiled_motion_metrics()(
            rows.to(dtype=torch.int64),
            self.anchor_body_idx,
            self._body_pos_w,
            self._robot_body_pos_w,
            cast("torch.Tensor", self._motion_data.body_quat_w),
            self._robot_body_quat_w,
            cast("torch.Tensor", self._motion_data.body_lin_vel_w),
            self._robot_body_lin_vel_w,
            cast("torch.Tensor", self._motion_data.body_ang_vel_w),
            self._robot_body_ang_vel_w,
            self.body_pos_relative_w,
            self.body_quat_relative_w,
            cast("torch.Tensor", self._motion_data.joint_pos),
            self._robot_joint_pos,
            cast("torch.Tensor", self._motion_data.joint_vel),
            self._robot_joint_vel,
            self._torch_metrics(),
        )

    def _update_torch_sampling_metrics(self) -> None:
        values = (
            ("sampling_entropy", self.tensor_sampler.sampling_entropy),
            ("sampling_top1_prob", self.tensor_sampler.sampling_top1_prob),
            ("sampling_top1_bin", self.tensor_sampler.sampling_top1_bin),
        )
        for name, value in values:
            metric = self.metrics[name]
            if not isinstance(metric, torch.Tensor):
                raise TypeError("TensorMotionCommand sampler metrics must remain Torch tensors")
            metric.copy_(value.expand_as(metric))


@dataclass(kw_only=True)
class MotionJointPositionActionCfg(JointPositionActionCfg):
    command_name: str = "motion"
    simulate_action_latency: bool = False

    def build(self, env: ManagerBasedRlEnv) -> MotionJointPositionAction:
        return MotionJointPositionAction(self, env)


class MotionJointPositionAction(JointPositionAction):
    cfg: MotionJointPositionActionCfg  # pyright: ignore[reportIncompatibleVariableOverride]

    def __init__(self, cfg: MotionJointPositionActionCfg, env: ManagerBasedRlEnv):
        if not isinstance(cfg.simulate_action_latency, bool):
            raise TypeError("MotionJointPositionActionCfg simulate_action_latency must be bool")
        super().__init__(cfg, env)
        self._motion_command = _command(env, cfg.command_name)
        self._previous_raw_actions = torch.zeros_like(self._raw_actions)
        # MimicLite critic observations can read this target before the first
        # action application. Initialize deterministically rather than exposing
        # uninitialized device memory in initial/reset observations.
        self._tensor_motion_target = torch.zeros_like(self._processed_actions)

    @property
    def target(self) -> np.ndarray:
        """Most recently applied physical joint target in entity joint order."""
        return self._target

    @property
    def tensor_target(self) -> torch.Tensor:
        """Most recently applied physical target for tensor control planes."""
        return self._tensor_motion_target

    def process_actions(self, actions: torch.Tensor) -> None:
        self._previous_raw_actions.copy_(self._raw_actions)
        super().process_actions(actions)
        if not self.cfg.simulate_action_latency:
            return
        self._apply_affine(self._previous_raw_actions, self._processed_actions)

    def validate_actions(self) -> None:
        """Validate the current latency-adjusted raw input."""
        if not bool(torch.isfinite(self._raw_actions).all()):
            raise ValueError(f"{type(self).__name__} received NaN or Inf actions")

    def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
        super().reset(env_ids)
        selector = (
            slice(None)
            if env_ids is None
            else self._reset_selector(
                torch.arange(self.num_envs, device=self._device)[env_ids]
                if isinstance(env_ids, slice)
                else env_ids
            )
        )
        self._previous_raw_actions[selector] = 0.0

    def apply_actions(self) -> None:
        if isinstance(self._entity.data.control_buffer, torch.Tensor):
            encoder_bias = self._entity.data.encoder_bias_tensor.to(
                device=self._device, non_blocking=True
            )
            default_bias = self._motion_command.joint_default_bias
            if isinstance(default_bias, torch.Tensor):
                selected_default_bias = default_bias.index_select(1, self._target_index)
            else:
                selected_default_bias = torch.as_tensor(
                    default_bias[:, self._target_ids],
                    device=self._device,
                    dtype=torch.float32,
                )
            torch.add(
                self._processed_actions,
                selected_default_bias - encoder_bias.index_select(1, self._target_index),
                out=self._tensor_motion_target,
            )
            self._entity.set_joint_position_target(
                self._tensor_motion_target, joint_ids=self._target_ids
            )
            return
        processed = self._entity_values(self._processed_actions)
        encoder_bias = self._entity.data.encoder_bias[:, self._target_ids]
        default_bias = self._motion_command.joint_default_bias[:, self._target_ids]
        np.add(
            processed,
            default_bias,
            out=self._target,
        )
        self._target -= encoder_bias
        self._entity.set_joint_position_target(self._target, joint_ids=self._target_ids)


def _command(env: ManagerBasedRlEnv, command_name: str) -> MotionCommand:
    try:
        command = env.command_manager.get_term(command_name)
    except KeyError as exc:
        raise KeyError(f"Motion command term '{command_name}' not found") from exc
    if not isinstance(command, MotionCommand):
        raise TypeError(
            f"Command term '{command_name}' is {type(command).__name__}, expected MotionCommand"
        )
    return command


def motion_anchor_pos_b(env: ManagerBasedRlEnv, command_name: str) -> np.ndarray | torch.Tensor:
    return _command(env, command_name).motion_anchor_pos_b


def motion_anchor_ori_b(env: ManagerBasedRlEnv, command_name: str) -> np.ndarray | torch.Tensor:
    return _command(env, command_name).motion_anchor_ori_b


def robot_body_pos_b(env: ManagerBasedRlEnv, command_name: str) -> np.ndarray | torch.Tensor:
    command = _command(env, command_name)
    return command.robot_body_pos_b.reshape(env.num_envs, -1)


def robot_body_ori_b(env: ManagerBasedRlEnv, command_name: str) -> np.ndarray | torch.Tensor:
    command = _command(env, command_name)
    return command.robot_body_ori_b.reshape(env.num_envs, -1)


def motion_joint_pos_rel(env: ManagerBasedRlEnv, command_name: str) -> np.ndarray | torch.Tensor:
    command = _command(env, command_name)
    if getattr(command, "tensor_carrier", False):
        # MotionCommand.robot_joint_pos dispatches through the entity's host
        # facade; a tensor carrier owns the explicit device joint view.
        robot_joint_pos = getattr(command, "device_robot_joint_pos", None)
        if robot_joint_pos is None:
            robot_joint_pos = command.robot_joint_pos
        if not isinstance(robot_joint_pos, torch.Tensor):
            robot_joint_pos = torch.as_tensor(
                np.asarray(robot_joint_pos), dtype=torch.float32, device=env.device
            )
        return (
            cast(torch.Tensor, robot_joint_pos)
            - command.robot.data.default_joint_pos_torch(env.device)
            - cast(torch.Tensor, command.joint_default_bias)
        )
    return (
        command.robot_joint_pos - command.robot.data.default_joint_pos - command.joint_default_bias
    )


def motion_joint_pos_rel_biased(
    env: ManagerBasedRlEnv, command_name: str
) -> np.ndarray | torch.Tensor:
    """Joint position relative to the episode default, including encoder bias."""
    command = _command(env, command_name)
    if getattr(command, "tensor_carrier", False):
        robot_joint_pos = getattr(command, "device_robot_joint_pos", None)
        if robot_joint_pos is None:
            robot_joint_pos = command.robot_joint_pos
        return (
            cast(torch.Tensor, robot_joint_pos)
            + command.robot.data.encoder_bias_tensor.to(
                device=env.device, dtype=torch.float32, non_blocking=True
            )
            - command.robot.data.default_joint_pos_torch(env.device)
            - cast(torch.Tensor, command.joint_default_bias)
        )
    return (
        command.robot.data.joint_pos_biased
        - command.robot.data.default_joint_pos
        - command.joint_default_bias
    )


def _positive_std(value: float, *, term_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, np.number)):
        raise TypeError(f"{term_name} std must be a real number")
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{term_name} std must be finite and positive")
    return result


def motion_global_anchor_position_error_exp(
    env: ManagerBasedRlEnv, command_name: str, std: float
) -> np.ndarray | torch.Tensor:
    command = _command(env, command_name)
    scale = _positive_std(std, term_name="motion anchor position")
    if getattr(command, "tensor_carrier", False):
        anchor_delta = command.anchor_pos_w - command.robot_anchor_pos_w
        error = cast(torch.Tensor, anchor_delta).square().sum(dim=-1)
        return torch.exp(-error / (scale * scale))
    diff = command.anchor_pos_w - command.robot_anchor_pos_w
    np.square(diff, out=diff)
    error = np.sum(diff, axis=-1)
    np.divide(error, -(scale**2), out=error)
    return np.exp(error, out=error)


def motion_global_anchor_orientation_error_exp(
    env: ManagerBasedRlEnv, command_name: str, std: float
) -> np.ndarray | torch.Tensor:
    command = _command(env, command_name)
    scale = _positive_std(std, term_name="motion anchor orientation")
    if getattr(command, "tensor_carrier", False):
        motion_anchor_quat = cast(torch.Tensor, command.anchor_quat_w)
        robot_anchor_quat = cast(torch.Tensor, command.robot_anchor_quat_w)
        rel = quat_mul(quat_conjugate(motion_anchor_quat), robot_anchor_quat)
        xyz = torch.linalg.vector_norm(rel[..., 1:4], dim=-1)
        angle = 2.0 * torch.atan2(xyz, rel[..., 0].abs().clamp(max=1.0))
        return torch.exp(-angle.square() / (scale * scale))
    error = np_quat_error_magnitude_squared_batched(
        command.anchor_quat_w, command.robot_anchor_quat_w
    )
    np.divide(error, -(scale**2), out=error)
    return np.exp(error, out=error)


class _BodyTerm(ManagerTermBase):
    def __init__(self, cfg: ManagerTermBaseCfg, env: ManagerBasedRlEnv):
        super().__init__(env)
        command_name = cfg.params.get("command_name")
        if not isinstance(command_name, str) or not command_name:
            raise ValueError(f"{type(self).__name__} requires a non-empty command_name")
        self._command_name = command_name
        command = _command(env, command_name)
        body_names = cfg.params.get("body_names")
        if body_names is None:
            self._body_ids = slice(None)
        else:
            requested = tuple(body_names)
            missing = [name for name in requested if name not in command.cfg.body_names]
            if missing:
                raise ValueError(
                    f"Body names {missing} are not tracked by command '{command_name}'"
                )
            self._body_ids = np.asarray(
                [command.cfg.body_names.index(name) for name in requested], dtype=np.intp
            )
        # Lazily allocated scratch for squared-error reductions (issue #1296);
        # shapes depend on the selected body set, so they are sized on first use.
        self._diff_scratch: np.ndarray | None = None
        self._err_scratch: np.ndarray | None = None

    def _squared_error_3d(self, ref: np.ndarray, actual: np.ndarray) -> np.ndarray:
        """Per-body squared 3D error with reused scratch, same op order as the
        naive ``np.square(ref - actual).sum(axis=-1)`` (bit-identical)."""
        if (
            self._diff_scratch is None
            or self._err_scratch is None
            or self._diff_scratch.shape != ref.shape
        ):
            self._diff_scratch = np.empty(ref.shape, dtype=ref.dtype)
            self._err_scratch = np.empty(ref.shape[:-1], dtype=ref.dtype)
        diff = self._diff_scratch
        err = self._err_scratch
        np.subtract(ref, actual, out=diff)
        np.square(diff, out=diff)
        np.sum(diff, axis=-1, out=err)
        return err

    @staticmethod
    def _exp_neg_scaled(error: np.ndarray, scale: float) -> np.ndarray:
        """``np.exp(-error / scale**2)`` without intermediate temporaries; the
        input buffer is consumed and returned (callers own it)."""
        np.divide(error, -(scale**2), out=error)
        return np.exp(error, out=error)

    def _validate(self, command_name: str, std: float) -> tuple[MotionCommand, float]:
        if command_name != self._command_name:
            raise ValueError(
                f"{type(self).__name__} was bound to '{self._command_name}', got '{command_name}'"
            )
        return _command(self._env, command_name), _positive_std(std, term_name=type(self).__name__)


class _NumbaBodyTerm(_BodyTerm):
    """Shared cold-path setup for the four fixed parallel body reward kernels."""

    def __init__(self, cfg: ManagerTermBaseCfg, env: ManagerBasedRlEnv):
        super().__init__(cfg, env)
        configure_motion_kernel_runtime()
        command = _command(env, self._command_name)
        if isinstance(self._body_ids, slice):
            body_ids = np.arange(len(command.cfg.body_names), dtype=np.intp)
        else:
            body_ids = self._body_ids
        body_ids.setflags(write=False)
        self._kernel_body_ids = body_ids
        tensor_carrier = bool(getattr(command, "tensor_carrier", False))
        dtype: np.dtype[Any] = (
            np.dtype(np.float32) if tensor_carrier else np.dtype(command.body_pos_relative_w.dtype)
        )
        self._kernel_result = np.empty(self.num_envs, dtype=dtype)
        self._tensor_carrier = tensor_carrier

    def _kernel_std(self, scale: float) -> float:
        return cast(float, self._kernel_result.dtype.type(scale))

    def _body_reduce(
        self,
        command: MotionCommand,
        reference_attr: str,
        actual_attr: str,
        kernel: Callable[[np.ndarray, np.ndarray, np.ndarray, float, np.ndarray], None],
        scale: float,
    ) -> np.ndarray | torch.Tensor:
        """Dispatch one fixed body-error reduction on the command's carrier."""
        reference = getattr(command, reference_attr)
        actual = getattr(command, actual_attr)
        if self._tensor_carrier:
            assert isinstance(reference, torch.Tensor)
            assert isinstance(actual, torch.Tensor)
            return self._tensor_body_reduce(reference, actual, scale)
        kernel(
            reference,
            actual,
            self._kernel_body_ids,
            self._kernel_std(scale),
            self._kernel_result,
        )
        return self._kernel_result

    def _tensor_body_reduce(
        self,
        reference: torch.Tensor,
        actual: torch.Tensor,
        scale: float,
    ) -> torch.Tensor:
        """Tensor peer of the four fixed Numba squared-error reductions."""
        if reference.ndim == 3 and reference.shape[-1] == 4:
            rel = quat_mul(quat_conjugate(reference), actual)
            xyz = torch.linalg.vector_norm(rel[..., 1:4], dim=-1)
            angle = 2.0 * torch.atan2(xyz, rel[..., 0].abs().clamp(max=1.0))
            error = angle.square().sum(dim=-1)
        else:
            error = (reference - actual).square().sum(dim=(-1, -2))
        body_count = reference.shape[-2]
        return torch.exp(-error / (body_count * scale * scale))


class motion_relative_body_position_error_exp(_NumbaBodyTerm):
    def __init__(self, cfg: ManagerTermBaseCfg, env: ManagerBasedRlEnv):
        super().__init__(cfg, env)
        command = _command(env, self._command_name)
        self._body_reduce(
            command,
            "body_pos_relative_w",
            "robot_body_pos_w",
            reward_motion_body_pos_kernel,
            1.0,
        )

    def __call__(
        self,
        env: ManagerBasedRlEnv,
        command_name: str,
        std: float,
        body_names: tuple[str, ...] | None = None,
    ) -> np.ndarray | torch.Tensor:
        del env, body_names
        command, scale = self._validate(command_name, std)
        return self._body_reduce(
            command,
            "body_pos_relative_w",
            "robot_body_pos_w",
            reward_motion_body_pos_kernel,
            scale,
        )


class motion_relative_body_orientation_error_exp(_NumbaBodyTerm):
    def __init__(self, cfg: ManagerTermBaseCfg, env: ManagerBasedRlEnv):
        super().__init__(cfg, env)
        command = _command(env, self._command_name)
        self._body_reduce(
            command,
            "body_quat_relative_w",
            "robot_body_quat_w",
            reward_motion_body_ori_kernel,
            1.0,
        )

    def __call__(
        self,
        env: ManagerBasedRlEnv,
        command_name: str,
        std: float,
        body_names: tuple[str, ...] | None = None,
    ) -> np.ndarray | torch.Tensor:
        del env, body_names
        command, scale = self._validate(command_name, std)
        return self._body_reduce(
            command,
            "body_quat_relative_w",
            "robot_body_quat_w",
            reward_motion_body_ori_kernel,
            scale,
        )


class motion_global_body_linear_velocity_error_exp(_NumbaBodyTerm):
    def __init__(self, cfg: ManagerTermBaseCfg, env: ManagerBasedRlEnv):
        super().__init__(cfg, env)
        command = _command(env, self._command_name)
        self._body_reduce(
            command,
            "body_lin_vel_w",
            "robot_body_lin_vel_w",
            reward_motion_body_lin_vel_kernel,
            1.0,
        )

    def __call__(
        self,
        env: ManagerBasedRlEnv,
        command_name: str,
        std: float,
        body_names: tuple[str, ...] | None = None,
    ) -> np.ndarray | torch.Tensor:
        del env, body_names
        command, scale = self._validate(command_name, std)
        return self._body_reduce(
            command,
            "body_lin_vel_w",
            "robot_body_lin_vel_w",
            reward_motion_body_lin_vel_kernel,
            scale,
        )


class motion_global_body_angular_velocity_error_exp(_NumbaBodyTerm):
    def __init__(self, cfg: ManagerTermBaseCfg, env: ManagerBasedRlEnv):
        super().__init__(cfg, env)
        command = _command(env, self._command_name)
        self._body_reduce(
            command,
            "body_ang_vel_w",
            "robot_body_ang_vel_w",
            reward_motion_body_ang_vel_kernel,
            1.0,
        )

    def __call__(
        self,
        env: ManagerBasedRlEnv,
        command_name: str,
        std: float,
        body_names: tuple[str, ...] | None = None,
    ) -> np.ndarray | torch.Tensor:
        del env, body_names
        command, scale = self._validate(command_name, std)
        return self._body_reduce(
            command,
            "body_ang_vel_w",
            "robot_body_ang_vel_w",
            reward_motion_body_ang_vel_kernel,
            scale,
        )


class motion_relative_body_position_z_error_exp(_BodyTerm):
    def __call__(
        self,
        env: ManagerBasedRlEnv,
        command_name: str,
        std: float,
        body_names: tuple[str, ...] | None = None,
    ) -> np.ndarray | torch.Tensor:
        del env, body_names
        command, scale = self._validate(command_name, std)
        if getattr(command, "tensor_carrier", False):
            delta = (
                cast(torch.Tensor, command.body_pos_relative_w)[:, self._body_ids, 2]
                - cast(torch.Tensor, command.robot_body_pos_w)[:, self._body_ids, 2]
            )
            return torch.exp(-delta.square().mean(dim=-1) / (scale * scale))
        error = np.square(
            command.body_pos_relative_w[:, self._body_ids, 2]
            - command.robot_body_pos_w[:, self._body_ids, 2]
        )
        return self._exp_neg_scaled(error.mean(axis=-1), scale)


def motion_joint_position_error_exp(
    env: ManagerBasedRlEnv, command_name: str, std: float
) -> np.ndarray | torch.Tensor:
    command = _command(env, command_name)
    scale = _positive_std(std, term_name="motion joint position")
    if getattr(command, "tensor_carrier", False):
        robot_joint_pos = getattr(command, "device_robot_joint_pos", None)
        if robot_joint_pos is None:
            robot_joint_pos = command.robot_joint_pos
        joint_delta = command.joint_pos - robot_joint_pos
        error = cast(torch.Tensor, joint_delta).square().mean(dim=-1)
        return torch.exp(-error / (scale * scale))
    diff = command.joint_pos - command.robot_joint_pos
    np.square(diff, out=diff)
    error = diff.mean(axis=-1)
    np.divide(error, -(scale**2), out=error)
    return np.exp(error, out=error)


def motion_joint_velocity_error_exp(
    env: ManagerBasedRlEnv, command_name: str, std: float
) -> np.ndarray | torch.Tensor:
    command = _command(env, command_name)
    scale = _positive_std(std, term_name="motion joint velocity")
    if getattr(command, "tensor_carrier", False):
        robot_joint_vel = getattr(command, "device_robot_joint_vel", None)
        if robot_joint_vel is None:
            robot_joint_vel = command.robot_joint_vel
        joint_delta = command.joint_vel - robot_joint_vel
        error = cast(torch.Tensor, joint_delta).square().mean(dim=-1)
        return torch.exp(-error / (scale * scale))
    diff = command.joint_vel - command.robot_joint_vel
    np.square(diff, out=diff)
    error = diff.mean(axis=-1)
    np.divide(error, -(scale**2), out=error)
    return np.exp(error, out=error)


def joint_pos_limits(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> np.ndarray:
    """Penalize selected joint-limit violations through the entity facade."""
    asset = cast("Entity", env.scene[asset_cfg.name])
    joint_pos = asset.data.joint_pos[:, asset_cfg.joint_ids]
    limits = asset.data.soft_joint_pos_limits[asset_cfg.joint_ids]
    # Same op order as the naive form (maximum -> add -> square -> sum),
    # chained in place to avoid intermediate allocations.
    error = np.subtract(limits[:, 0], joint_pos)
    np.maximum(error, 0.0, out=error)
    upper = np.subtract(joint_pos, limits[:, 1])
    np.maximum(upper, 0.0, out=upper)
    error += upper
    np.square(error, out=error)
    return np.sum(error, axis=-1)


class undesired_body_contacts(_BodyTerm):
    def __call__(
        self,
        env: ManagerBasedRlEnv,
        command_name: str,
        threshold: float,
        body_names: tuple[str, ...] | None = None,
    ) -> np.ndarray | torch.Tensor:
        del env, body_names
        command = _command(self._env, command_name)
        if getattr(command, "tensor_carrier", False):
            robot_body_pos = cast(torch.Tensor, command.robot_body_pos_w)
            return (
                (robot_body_pos[:, self._body_ids, 2] < threshold)
                .sum(dim=-1)
                .to(dtype=torch.float32)
            )
        return np.sum(command.robot_body_pos_w[:, self._body_ids, 2] < threshold, axis=-1)


@dataclass(kw_only=True)
class MimicLiteAppliedTorqueObservationCfg(ObservationTermCfg):
    """Applied actuator torque read from the public packed sensor phase."""

    command_name: str = "motion"


class MimicLiteAppliedTorqueObservation(ManagerTermBase):
    cfg: MimicLiteAppliedTorqueObservationCfg

    def __init__(self, cfg: MimicLiteAppliedTorqueObservationCfg, env: ManagerBasedRlEnv):
        super().__init__(env)
        if not isinstance(cfg, MimicLiteAppliedTorqueObservationCfg):
            raise TypeError("MimicLite torque observation has an incompatible config")
        command = _command(env, cfg.command_name)
        self._entity_name = command.cfg.entity_name
        self._sensor_names = tuple(
            f"{name}{_TORQUE_SENSOR_SUFFIX}" for name in command.robot.joint_names
        )
        try:
            self._host_view = env.scene.bind_sensor_data(self._sensor_names)
        except (KeyError, TypeError, ValueError, NotImplementedError) as exc:
            raise type(exc)(
                "MimicLite torque observation could not materialize the host sensor "
                f"view for {self._sensor_names}: {exc}"
            ) from exc
        self._warned_host_fallback = False

    @property
    def entity_name(self) -> str:
        return self._entity_name

    @property
    def tensor_sensor_names(self) -> tuple[str, ...]:
        return self._sensor_names

    def __call__(self, env: ManagerBasedRlEnv) -> torch.Tensor:
        if env is not self._env:
            raise ValueError("MimicLite torque observation was called with an unbound env")
        read_plan = getattr(env.scene, "_tensor_read_plan", None)
        if read_plan is None:
            # Manager probing runs before the scene compiles its packed read
            # phase. Read the validated host view explicitly (same convention
            # as _NamedSensorObservation._read_tensor) and warn once.
            if not self._warned_host_fallback:
                warnings.warn(
                    "MimicLite torque observation read the host sensor view because "
                    "the scene tensor read plan is not compiled yet",
                    stacklevel=2,
                )
                self._warned_host_fallback = True
            device = getattr(env, "device", torch.device("cpu"))
            return torch.as_tensor(self._host_view.read(), dtype=torch.float32, device=device)
        views = read_plan.sensor_tensor_views(
            env.scene[self._entity_name], self._sensor_names
        ).values
        return torch.cat(tuple(views[name] for name in self._sensor_names), dim=-1)


class bad_anchor_pos_z_only(ManagerTermBase):
    """Anchor-height termination backed by a parallel, pre-warmed Numba kernel."""

    def __init__(self, cfg: ManagerTermBaseCfg, env: ManagerBasedRlEnv):
        super().__init__(env)
        configure_motion_kernel_runtime()
        command_name = cfg.params.get("command_name")
        if not isinstance(command_name, str) or not command_name:
            raise ValueError(f"{type(self).__name__} requires a non-empty command_name")
        self._command_name = command_name
        self._result = np.empty(self.num_envs, dtype=np.bool_)
        command = _command(env, command_name)
        tensor_carrier = bool(getattr(command, "tensor_carrier", False))
        configured_threshold = cfg.params.get("threshold", 0.0)
        threshold = (
            float(configured_threshold)
            if tensor_carrier
            else command.body_pos_w.dtype.type(configured_threshold)
        )
        if tensor_carrier:
            # Numba warmup is host-only; the tensor peer is pure Torch and has
            # no lazy dispatch to precompile.
            return
        termination_anchor_pos_kernel(
            command.body_pos_w,
            command.robot_body_pos_w,
            command.anchor_body_idx,
            threshold,
            self._result,
        )

    def __call__(
        self,
        env: ManagerBasedRlEnv,
        command_name: str,
        threshold: float,
    ) -> np.ndarray | torch.Tensor:
        del env
        if command_name != self._command_name:
            raise ValueError(
                f"{type(self).__name__} was bound to '{self._command_name}', got '{command_name}'"
            )
        command = _command(self._env, command_name)
        if getattr(command, "tensor_carrier", False):
            motion_anchor_pos = cast(torch.Tensor, command.body_pos_w)
            robot_anchor_pos = cast(torch.Tensor, command.robot_body_pos_w)
            return (
                motion_anchor_pos[:, command.anchor_body_idx, 2]
                - robot_anchor_pos[:, command.anchor_body_idx, 2]
            ).abs() > threshold
        threshold_value = command.body_pos_w.dtype.type(threshold)
        termination_anchor_pos_kernel(
            command.body_pos_w,
            command.robot_body_pos_w,
            command.anchor_body_idx,
            threshold_value,
            self._result,
        )
        return self._result


def bad_anchor_ori(
    env: ManagerBasedRlEnv,
    command_name: str,
    threshold: float,
    asset_cfg: SceneEntityCfg | None = None,
) -> np.ndarray | torch.Tensor:
    command = _command(env, command_name)
    if getattr(command, "tensor_carrier", False):
        motion_anchor_quat = cast(torch.Tensor, command.anchor_quat_w)
        robot_anchor_quat = cast(torch.Tensor, command.robot_anchor_quat_w)
        motion_z = 2.0 * (motion_anchor_quat[:, 1] ** 2 + motion_anchor_quat[:, 2] ** 2) - 1.0
        robot_z = 2.0 * (robot_anchor_quat[:, 1] ** 2 + robot_anchor_quat[:, 2] ** 2) - 1.0
        return (motion_z - robot_z).abs() > threshold
    asset = command.robot if asset_cfg is None else cast("Entity", env.scene[asset_cfg.name])
    gravity_vec_w = asset.data.gravity_vec_w
    motion_z = np_quat_apply_inverse(command.anchor_quat_w, gravity_vec_w)[:, 2]
    robot_z = np_quat_apply_inverse(command.robot_anchor_quat_w, gravity_vec_w)[:, 2]
    return np.abs(motion_z - robot_z) > threshold


class bad_motion_body_pos_z_only(_BodyTerm):
    def __call__(
        self,
        env: ManagerBasedRlEnv,
        command_name: str,
        threshold: float,
        body_names: tuple[str, ...] | None = None,
    ) -> np.ndarray | torch.Tensor:
        del env, body_names
        command = _command(self._env, command_name)
        if getattr(command, "tensor_carrier", False):
            reference = cast(torch.Tensor, command.body_pos_relative_w)
            actual = cast(torch.Tensor, command.robot_body_pos_w)
            error = (reference[:, self._body_ids, 2] - actual[:, self._body_ids, 2]).abs()
            return torch.any(error > threshold, dim=-1)
        error = np.abs(
            command.body_pos_relative_w[:, self._body_ids, 2]
            - command.robot_body_pos_w[:, self._body_ids, 2]
        )
        return np.any(error > threshold, axis=-1)


class bad_undesired_body_contacts(_BodyTerm):
    def __call__(
        self,
        env: ManagerBasedRlEnv,
        command_name: str,
        threshold: float,
        body_names: tuple[str, ...] | None = None,
    ) -> np.ndarray | torch.Tensor:
        del env, body_names
        command = _command(self._env, command_name)
        if getattr(command, "tensor_carrier", False):
            robot_body_pos = cast(torch.Tensor, command.robot_body_pos_w)
            return torch.any(robot_body_pos[:, self._body_ids, 2] < threshold, dim=-1)
        return np.any(command.robot_body_pos_w[:, self._body_ids, 2] < threshold, axis=-1)


def motion_clip_end(env: ManagerBasedRlEnv, command_name: str) -> np.ndarray | torch.Tensor:
    command = _command(env, command_name)
    if getattr(command, "tensor_carrier", False):
        current_clip_ends = getattr(command, "tensor_current_clip_end_frames", None)
        if current_clip_ends is not None:
            return cast(torch.Tensor, command.time_steps) >= cast(torch.Tensor, current_clip_ends)
    return command.time_steps >= command.sampler.current_clip_end_frames


class MotionAnchorObservation(ManagerTermBase):
    """Tensor-native anchor-relative observation over the scene body phase.

    The term is constructed after ``CommandManager`` and resolves the declared
    motion command through its public Manager accessor.  The resolved body
    namespace becomes a static ``tensor_body_names`` declaration so the Manager
    compiles one aggregate body read before any term executes.
    """

    def __init__(self, cfg: ManagerTermBaseCfg, env: ManagerBasedRlEnv):
        super().__init__(env)
        command_name = cfg.params.get("command_name", "motion")
        if not isinstance(command_name, str) or not command_name:
            raise ValueError("MotionAnchorObservation command_name must be a non-empty string")
        try:
            command = env.command_manager.get_term(command_name)
        except KeyError as exc:
            raise KeyError(
                f"Motion anchor observation command term '{command_name}' not found"
            ) from exc
        if not isinstance(command, MotionCommand):
            raise TypeError(
                "Motion anchor observation command term "
                f"'{command_name}' is {type(command).__name__}, expected MotionCommand"
            )
        self._command_name = command_name
        self._entity_name = command.cfg.entity_name
        self._body_names = tuple(command.cfg.body_names)
        self._anchor_body_idx = command.anchor_body_idx

    @property
    def tensor_body_names(self) -> tuple[str, ...]:
        return self._body_names

    @property
    def entity_name(self) -> str:
        return self._entity_name

    def _cold_observation(self, command: MotionCommand) -> torch.Tensor:
        """Return the immutable-shape probe used before the read plan exists."""
        del command
        width = 3 if isinstance(self, MotionAnchorPositionObservation) else 6
        return torch.zeros((self.num_envs, width), dtype=torch.float32)


class MotionAnchorPositionObservation(MotionAnchorObservation):
    def __call__(self, env: ManagerBasedRlEnv, command_name: str = "motion") -> torch.Tensor:
        if command_name != self._command_name:
            raise ValueError(
                f"Motion anchor position observation was bound to {self._command_name!r}, "
                f"received {command_name!r}"
            )
        read_plan = getattr(env.scene, "_tensor_read_plan", None)
        if read_plan is None:
            return self._cold_observation(_command(env, self._command_name))
        view = read_plan.body_tensor_view(self._entity_name, self._body_names)
        command = _command(env, self._command_name)
        anchor_pos = torch.as_tensor(
            command.body_pos_w[:, self._anchor_body_idx],
            dtype=torch.float32,
            device=view.pos_w.device,
        )
        robot_anchor_pos = view.pos_w[:, self._anchor_body_idx]
        robot_anchor_quat = view.quat_w[:, self._anchor_body_idx]
        return quat_apply_inverse(robot_anchor_quat, anchor_pos - robot_anchor_pos)


class MotionAnchorOrientationObservation(MotionAnchorObservation):
    def __call__(self, env: ManagerBasedRlEnv, command_name: str = "motion") -> torch.Tensor:
        if command_name != self._command_name:
            raise ValueError(
                f"Motion anchor orientation observation was bound to {self._command_name!r}, "
                f"received {command_name!r}"
            )
        read_plan = getattr(env.scene, "_tensor_read_plan", None)
        if read_plan is None:
            return self._cold_observation(_command(env, self._command_name))
        view = read_plan.body_tensor_view(self._entity_name, self._body_names)
        robot_quat = view.quat_w[:, self._anchor_body_idx]
        command = _command(env, self._command_name)
        motion_quat = torch.as_tensor(
            command.body_quat_w[:, self._anchor_body_idx],
            dtype=torch.float32,
            device=view.quat_w.device,
        )
        return quat_to_rot6(quat_mul(quat_conjugate(robot_quat), motion_quat))


__all__ = [
    "MotionCommand",
    "MotionCommandCfg",
    "MotionCommandParamsCfg",
    "TensorMotionCommandCfg",
    "TensorMotionCommand",
    "TensorMotionObsFuture",
    "MotionJointPositionAction",
    "MotionJointPositionActionCfg",
    "MotionAnchorObservation",
    "MimicLiteAppliedTorqueObservation",
    "MimicLiteAppliedTorqueObservationCfg",
    "MotionAnchorOrientationObservation",
    "MotionAnchorPositionObservation",
    "bad_anchor_ori",
    "bad_anchor_pos_z_only",
    "bad_motion_body_pos_z_only",
    "bad_undesired_body_contacts",
    "joint_pos_limits",
    "motion_anchor_ori_b",
    "motion_anchor_pos_b",
    "motion_clip_end",
    "motion_global_anchor_orientation_error_exp",
    "motion_global_anchor_position_error_exp",
    "motion_global_body_angular_velocity_error_exp",
    "motion_global_body_linear_velocity_error_exp",
    "motion_joint_pos_rel",
    "motion_joint_pos_rel_biased",
    "motion_joint_position_error_exp",
    "motion_joint_velocity_error_exp",
    "motion_relative_body_orientation_error_exp",
    "motion_relative_body_position_error_exp",
    "motion_relative_body_position_z_error_exp",
    "robot_body_ori_b",
    "robot_body_pos_b",
    "undesired_body_contacts",
]
