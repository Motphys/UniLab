"""Manager-native Torch terms for motion tracking."""

from __future__ import annotations

import math
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable, Literal, cast

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

from .motion_loader import MotionData, MotionLoader
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
    """Return the absolute-dot shortest-path angular error."""
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
    _motion_reset_values_compiled = torch.compile(_motion_reset_values_kernel, dynamic=False)
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


def _range_matrix(value: dict[str, tuple[float, float]], *, name: str) -> torch.Tensor:
    unknown = sorted(set(value) - set(_RANGE_KEYS))
    if unknown:
        raise ValueError(f"{name} has unknown axes {unknown}")
    try:
        ranges = torch.tensor(
            [value.get(key, (0.0, 0.0)) for key in _RANGE_KEYS],
            dtype=torch.float64,
        )
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must map axes to numeric (min, max) pairs") from exc
    if tuple(ranges.shape) != (6, 2) or not bool(torch.isfinite(ranges).all()):
        raise ValueError(f"{name} must contain six finite (min, max) pairs")
    if bool((ranges[:, 0] > ranges[:, 1]).any()):
        raise ValueError(f"{name} contains a minimum greater than its maximum")
    return ranges.clone()


def _pair(value: tuple[float, float], *, name: str) -> tuple[float, float]:
    try:
        if isinstance(value, (str, bytes)) or len(tuple(value)) != 2:
            raise ValueError(f"{name} must be a finite (min, max) pair")
        lower, upper = float(value[0]), float(value[1])
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be a numeric (min, max) pair") from exc
    if not (math.isfinite(lower) and math.isfinite(upper)):
        raise ValueError(f"{name} must be a finite (min, max) pair")
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


class MotionCommand(CommandTerm):
    """Device-resident motion reference command for Manager owners."""

    cfg: MotionCommandCfg

    def __init__(self, cfg: MotionCommandCfg, env: ManagerBasedRlEnv):
        _validate_motion_command_cfg(cfg)
        super().__init__(cfg, env)
        device = env.device
        if device.type == "cuda" and device.index is None:
            device = torch.device("cuda", index=torch.cuda.current_device())
        self._device = device
        self.robot = cast("Entity", env.scene[cfg.entity_name])
        body_ids, body_names = self.robot.find_bodies(cfg.body_names, preserve_order=True)
        if tuple(body_names) != cfg.body_names:
            raise ValueError(
                f"MotionCommand body order {tuple(body_names)} does not match {cfg.body_names}"
            )
        motion_body_ids = self.robot.motion_body_ids[body_ids]
        motion_body_indices = tuple(int(body_id) for body_id in motion_body_ids)
        self.motion = self._make_motion_loader(cfg.motion_file, motion_body_indices)
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
        self._clip_offsets_torch = self.motion.clip_offsets_torch(self._device)
        self._clip_end_frames_torch = self.motion.clip_end_frames_torch(self._device)
        self._motion_features = self._make_motion_features(self._device)
        self._motion_data = MotionData(
            joint_pos=torch.empty((self.num_envs, num_joints), device=self._device),
            joint_vel=torch.empty((self.num_envs, num_joints), device=self._device),
            body_pos_w=torch.empty((self.num_envs, num_bodies, 3), device=self._device),
            body_quat_w=torch.empty((self.num_envs, num_bodies, 4), device=self._device),
            body_lin_vel_w=torch.empty((self.num_envs, num_bodies, 3), device=self._device),
            body_ang_vel_w=torch.empty((self.num_envs, num_bodies, 3), device=self._device),
        )
        self._motion_packet_scratch = MotionData(
            joint_pos=torch.empty_like(self._motion_data.joint_pos),
            joint_vel=torch.empty_like(self._motion_data.joint_vel),
            body_pos_w=torch.empty_like(self._motion_data.body_pos_w),
            body_quat_w=torch.empty_like(self._motion_data.body_quat_w),
            body_lin_vel_w=torch.empty_like(self._motion_data.body_lin_vel_w),
            body_ang_vel_w=torch.empty_like(self._motion_data.body_ang_vel_w),
        )
        self._command = torch.empty((self.num_envs, num_joints * 2), device=self._device)
        self._command_finite_valid = torch.ones((), dtype=torch.bool, device=self._device)
        self._body_pos_w = torch.empty((self.num_envs, num_bodies, 3), device=self._device)
        self._body_pos_packet_scratch = torch.empty_like(self._body_pos_w)
        self.body_pos_relative_w = torch.empty_like(self._body_pos_w)
        self.body_quat_relative_w = torch.empty((self.num_envs, num_bodies, 4), device=self._device)
        self.motion_anchor_pos_b = torch.empty((self.num_envs, 3), device=self._device)
        self.motion_anchor_ori_b = torch.empty((self.num_envs, 6), device=self._device)
        self.robot_body_pos_b = torch.empty_like(self._body_pos_w)
        self.robot_body_ori_b = torch.empty((self.num_envs, num_bodies, 6), device=self._device)
        self.joint_default_bias = torch.zeros(
            (self.num_envs, num_joints), dtype=torch.float32, device=self._device
        )
        self._robot_cache_step = -1
        self._robot_body_pos_w = torch.empty_like(self._body_pos_w)
        self._robot_body_quat_w = torch.empty((self.num_envs, num_bodies, 4), device=self._device)
        self._robot_body_lin_vel_w = torch.empty_like(self._body_pos_w)
        self._robot_body_ang_vel_w = torch.empty_like(self._body_pos_w)
        self._robot_joint_pos = torch.empty(
            (self.num_envs, num_joints), dtype=torch.float32, device=self._device
        )
        self._robot_joint_vel = torch.empty(
            (self.num_envs, num_joints), dtype=torch.float32, device=self._device
        )
        self._env_origins = self._env.scene.env_origins_torch(self._device)
        self._soft_joint_limits = self.robot.data.soft_joint_pos_limits_torch(self._device)
        self._pose_range_torch = self._pose_range.to(device=self._device, dtype=torch.float32)
        self._velocity_range_torch = self._velocity_range.to(
            device=self._device, dtype=torch.float32
        )
        self._joint_position_range_torch = torch.as_tensor(
            self._joint_position_range, device=self._device
        )
        self._reset_joint_values = torch.empty(
            (self.num_envs, num_joints), dtype=torch.float32, device=self._device
        )
        self._reset_root_state = torch.empty(
            (self.num_envs, 13), dtype=torch.float32, device=self._device
        )
        self._reset_packet_velocity = torch.empty_like(self._reset_joint_values)
        self._reset_joint_values_scratch = torch.empty_like(self._reset_joint_values)
        self._reset_root_state_scratch = torch.empty_like(self._reset_root_state)
        self._reset_motion_packet = torch.empty(
            (self.num_envs, self._motion_features.shape[1]),
            dtype=torch.float32,
            device=self._device,
        )
        self._bind_read_phase = False
        self._tensor_all_rows = torch.arange(self.num_envs, dtype=torch.int64, device=self._device)
        self._tensor_all_reset_mask = torch.ones(
            self.num_envs, dtype=torch.bool, device=self._device
        )
        self._tensor_resample_ingested_mask: torch.Tensor | None = None
        self.last_step_timing_ms: dict[str, float] = {}
        self.last_post_compute_timing_ms: dict[str, float] = {}
        self.last_reset_timing_ms: dict[str, float] = {}
        self._motion_feature_layout: (
            tuple[dict[str, tuple[int, ...]], dict[str, tuple[int, int]]] | None
        ) = None
        self._last_reset_payload_validated = False
        self._sampler_dispatch_diagnostic_offset = 0

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
            self.metrics[name] = torch.zeros(
                self.num_envs, dtype=torch.float32, device=self._device
            )
        if self.cfg.params.sampling_mode not in ("adaptive", "mixed"):
            raise NotImplementedError(
                "MotionCommand tensor sampler requires adaptive or mixed sampling"
            )
        self._make_tensor_sampler()
        self._defer_read_phase_binding()

    def _defer_read_phase_binding(self) -> None:
        """Allocate immutable cold carriers only; defer state binding to read phase."""

    def bind_read_phase(self) -> None:
        if self._bind_read_phase:
            return
        self._bind_read_phase = True
        self._refresh_motion_torch()
        read_plan = getattr(self._env.scene, "_tensor_read_plan", None)
        if read_plan is not None:
            read_plan.refresh()
            self._refresh_robot_state(force=True)
        else:
            self._seed_robot_state_torch_from_defaults()
        self._refresh_relative_state()
        self._update_metrics()

    def _make_motion_loader(
        self,
        motion_file: str | list[str],
        body_indices: Sequence[int],
    ) -> MotionLoader:
        """Materialize the profile-owned motion loader on the cold path."""
        return MotionLoader(motion_file, body_indices=body_indices)

    def _make_motion_features(self, device: torch.device) -> torch.Tensor:
        """Cache the complete cold motion dataset as one device-resident table."""
        return self.motion.motion_features_torch(device)

    def _motion_packet(self, frames: torch.Tensor) -> torch.Tensor:
        """Gather motion rows from the device-resident feature table."""
        rows = frames.to(dtype=torch.int64, device=self._device)
        packet = self._reset_motion_packet
        torch.index_select(self._motion_features, 0, rows, out=packet)
        return packet

    def _make_tensor_sampler(self) -> None:
        if self._env.torch_rng is None:
            raise NotImplementedError("MotionCommand requires the Manager-owned Torch generator")
        self.tensor_sampler = TensorMotionSampler(
            mode=self.cfg.params.sampling_mode,
            num_envs=self.num_envs,
            num_frames=self.motion.num_frames,
            clip_offsets=self._clip_offsets_torch,
            clip_end_frames=self._clip_end_frames_torch,
            bin_count=int(self.motion.num_frames // self.motion.fps) + 1,
            adaptive_lambda=self.cfg.params.adaptive_lambda,
            adaptive_kernel_size=self.cfg.params.adaptive_kernel_size,
            adaptive_uniform_ratio=self.cfg.params.adaptive_uniform_ratio,
            adaptive_alpha=self.cfg.params.adaptive_alpha,
            start_ratio=self.cfg.params.sampling_start_ratio,
            initial_frames=torch.zeros(self.num_envs, dtype=torch.int32, device=self._device),
            initial_clip_end_frames=self._clip_end_frames_torch[:1]
            .to(dtype=torch.int32)
            .expand(self.num_envs)
            .contiguous(),
            device=self._device,
        )
        self.time_steps = self.tensor_sampler.current_frames
        self.current_clip_end_frames = self.tensor_sampler.current_clip_end_frames

    def _refresh_robot_state(
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
            raise RuntimeError("MotionCommand requires a refreshed scene tensor read phase")
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
        joint_view = read_plan.joint_tensor_view(self.robot)
        self._robot_joint_pos = joint_view.joint_pos
        self._robot_joint_vel = joint_view.joint_vel
        self._robot_cache_step = step

    def _refresh_relative_state(self, rows: torch.Tensor | None = None) -> None:
        _bind_compiled_motion_relative_state()(
            self._tensor_all_rows if rows is None else rows,
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

    @property
    def tensor_carrier(self) -> bool:
        return True

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
    def tensor_body_names(self) -> tuple[str, ...]:
        return tuple(self.cfg.body_names)

    @property
    def tensor_current_clip_end_frames(self) -> torch.Tensor:
        return cast(torch.Tensor, self.current_clip_end_frames)

    @property
    def robot_joint_pos(self) -> torch.Tensor:
        read_plan = getattr(self._env.scene, "_tensor_read_plan", None)
        if read_plan is not None and read_plan.ready:
            self._robot_joint_pos = read_plan.joint_tensor_view(self.robot).joint_pos
        return self._robot_joint_pos

    @property
    def robot_joint_vel(self) -> torch.Tensor:
        read_plan = getattr(self._env.scene, "_tensor_read_plan", None)
        if read_plan is not None and read_plan.ready:
            self._robot_joint_vel = read_plan.joint_tensor_view(self.robot).joint_vel
        return self._robot_joint_vel

    @property
    def robot_body_pos_w(self) -> torch.Tensor:
        self._refresh_robot_state()
        return self._robot_body_pos_w

    @property
    def robot_body_quat_w(self) -> torch.Tensor:
        self._refresh_robot_state()
        return self._robot_body_quat_w

    @property
    def robot_body_lin_vel_w(self) -> torch.Tensor:
        self._refresh_robot_state()
        return self._robot_body_lin_vel_w

    @property
    def robot_body_ang_vel_w(self) -> torch.Tensor:
        self._refresh_robot_state()
        return self._robot_body_ang_vel_w

    @property
    def robot_anchor_pos_w(self) -> torch.Tensor:
        return self.robot_body_pos_w[:, self.anchor_body_idx]

    @property
    def robot_anchor_quat_w(self) -> torch.Tensor:
        return self.robot_body_quat_w[:, self.anchor_body_idx]

    @property
    def robot_anchor_lin_vel_w(self) -> torch.Tensor:
        return self.robot_body_lin_vel_w[:, self.anchor_body_idx]

    @property
    def robot_anchor_ang_vel_w(self) -> torch.Tensor:
        return self.robot_body_ang_vel_w[:, self.anchor_body_idx]

    @property
    def command(self) -> torch.Tensor:
        torch.all(torch.isfinite(self._command), out=self._command_finite_valid)
        if not self.deferred_validation_boundaries and not bool(self._command_finite_valid):
            self._refresh_motion_torch()
        return self._command

    @property
    def joint_pos(self) -> torch.Tensor:
        return cast("torch.Tensor", self._motion_data.joint_pos)

    @property
    def joint_vel(self) -> torch.Tensor:
        return cast("torch.Tensor", self._motion_data.joint_vel)

    @property
    def body_pos_w(self) -> torch.Tensor:
        return self._body_pos_w

    @property
    def body_quat_w(self) -> torch.Tensor:
        return cast("torch.Tensor", self._motion_data.body_quat_w)

    @property
    def body_lin_vel_w(self) -> torch.Tensor:
        return cast("torch.Tensor", self._motion_data.body_lin_vel_w)

    @property
    def body_ang_vel_w(self) -> torch.Tensor:
        return cast("torch.Tensor", self._motion_data.body_ang_vel_w)

    @property
    def anchor_pos_w(self) -> torch.Tensor:
        return self._body_pos_w[:, self.anchor_body_idx]

    @property
    def anchor_quat_w(self) -> torch.Tensor:
        return cast("torch.Tensor", self._motion_data.body_quat_w)[:, self.anchor_body_idx]

    @property
    def anchor_lin_vel_w(self) -> torch.Tensor:
        return cast("torch.Tensor", self._motion_data.body_lin_vel_w)[:, self.anchor_body_idx]

    @property
    def anchor_ang_vel_w(self) -> torch.Tensor:
        return cast("torch.Tensor", self._motion_data.body_ang_vel_w)[:, self.anchor_body_idx]

    def reset(self, reset_mask: torch.Tensor, *, publish_metrics: bool = True) -> dict[str, float]:
        """Full-width motion command reset."""
        del publish_metrics
        if (
            reset_mask.shape != (self.num_envs,)
            or reset_mask.dtype != torch.bool
            or reset_mask.device != self._device
        ):
            raise ValueError(
                f"MotionCommand reset mask must have shape {(self.num_envs,)}, "
                f"bool dtype, and device {self._device}"
            )
        if self._env.torch_rng is None:
            raise NotImplementedError(
                "MotionCommand reset requires the Manager-owned Torch generator"
            )
        self._update_torch_error_metrics_mask()
        lower, upper = self._joint_default_position_range
        self.joint_default_bias.copy_(
            torch.where(
                reset_mask[:, None],
                self._env.torch_rng.uniform(
                    lower, upper, self.joint_default_bias.shape, dtype=torch.float32
                ),
                self.joint_default_bias,
            )
        )
        # MotionCommand bypasses CommandTerm.reset, so refresh the resample
        # timer here: a stale zero timer would force a spurious full-width
        # resample at the next compute, overwriting frames the reset just
        # sampled (and rows owned by an active owner-subset transaction).
        resample_lower, resample_upper = self._resampling_time_range
        sampled = self._torch_rng.uniform(
            resample_lower, resample_upper, (self.num_envs,), dtype=torch.float32
        )
        self.time_left.copy_(torch.where(reset_mask, sampled, self.time_left))
        self._resample_command(reset_mask)
        self._tensor_resample_ingested_mask = reset_mask
        self._last_reset_payload_validated = False
        return {}

    def _update_torch_error_metrics_mask(self) -> None:
        """Update full-width error metrics."""
        self._update_torch_error_metrics(self._tensor_all_rows)

    def sampler_reset_diagnostics(self) -> dict[str, float]:
        dispatches = self.tensor_sampler.diagnostics.reset_dispatches
        dispatch_count = dispatches - self._sampler_dispatch_diagnostic_offset
        self._sampler_dispatch_diagnostic_offset = dispatches
        return {
            "reset_done_sampler_dispatch_count": float(dispatch_count),
            "reset_done_sampler_host_transfer_count": float(
                self.tensor_sampler.diagnostics.reset_host_row_transfers
            ),
        }

    def _update_metrics(self) -> None:
        self._update_torch_error_metrics(self._tensor_all_rows)
        self._update_torch_sampling_metrics()

    @property
    def last_reset_payload_validated(self) -> bool:
        """Whether the current reset payload already passed owner validation."""
        return bool(self._last_reset_payload_validated)

    def _resample_command(self, reset_mask: torch.Tensor) -> None:
        """Full-width motion sampling and masked state publication."""
        sampler_started = time.perf_counter()
        frames = self.tensor_sampler.sample_frames(reset_mask, self._env.torch_rng.generator)
        packet = self._motion_packet(frames)
        tails, offsets = self._cached_motion_feature_shapes()
        pose = self._env.torch_rng.uniform(
            self._pose_range_torch[:, 0],
            self._pose_range_torch[:, 1],
            (self.num_envs, 6),
            dtype=torch.float32,
        )
        velocity = self._env.torch_rng.uniform(
            self._velocity_range_torch[:, 0],
            self._velocity_range_torch[:, 1],
            (self.num_envs, 6),
            dtype=torch.float32,
        )
        joint_noise = self._env.torch_rng.uniform(
            self._joint_position_range_torch[0],
            self._joint_position_range_torch[1],
            self._reset_joint_values.shape,
            dtype=torch.float32,
        )
        joint_start, joint_end = offsets["joint_pos"]
        vel_start, vel_end = offsets["joint_vel"]
        self._reset_packet_velocity.copy_(packet[:, vel_start:vel_end])
        joint_pos = self._reset_joint_values_scratch
        root_state = self._reset_root_state_scratch
        _bind_compiled_motion_reset_values()(
            packet[:, joint_start:joint_end],
            self._reset_packet_velocity,
            packet[:, offsets["body_pos_w"][0] : offsets["body_pos_w"][1]].view(
                self.num_envs, *tails["body_pos_w"]
            ),
            packet[:, offsets["body_quat_w"][0] : offsets["body_quat_w"][1]].view(
                self.num_envs, *tails["body_quat_w"]
            ),
            packet[:, offsets["body_lin_vel_w"][0] : offsets["body_lin_vel_w"][1]].view(
                self.num_envs, *tails["body_lin_vel_w"]
            ),
            packet[:, offsets["body_ang_vel_w"][0] : offsets["body_ang_vel_w"][1]].view(
                self.num_envs, *tails["body_ang_vel_w"]
            ),
            self._env_origins,
            pose,
            velocity,
            joint_noise,
            self._soft_joint_limits,
            joint_pos,
            root_state,
        )
        self.robot.write_motion_state_tensor_to_sim(
            root_state=root_state,
            position=joint_pos,
            velocity=self._reset_packet_velocity,
            reset_mask=reset_mask,
        )
        self._ingest_motion_packet_mask(reset_mask, packet)
        self.last_reset_timing_ms["reset_done_motion_reset_values_ms"] = (
            time.perf_counter() - sampler_started
        ) * 1000.0

    def _update_command(self) -> None:
        timing = self.last_step_timing_ms
        if not self.cfg.params.truncate_on_clip_end and not self.robot.reset_state_tensor_active:
            raise RuntimeError(
                "MotionCommand clip wrap requires an active tensor reset transaction"
            )
        failure_started = time.perf_counter()
        terminated = cast(torch.Tensor, self._env.termination_manager.terminated)
        self.tensor_sampler.update_failure_stats(terminated)
        timing["update_state_motion_failure_stats_ms"] = (
            time.perf_counter() - failure_started
        ) * 1000.0
        sampler_started = time.perf_counter()
        wrap_mask = self._step_tensor_sampler()
        timing["update_state_motion_step_sampler_ms"] = (
            time.perf_counter() - sampler_started
        ) * 1000.0
        if not self.cfg.params.truncate_on_clip_end:
            self._resample_command(wrap_mask)
        refresh_started = time.perf_counter()
        self._refresh_motion_torch()
        timing["update_state_motion_refresh_current_ms"] = (
            time.perf_counter() - refresh_started
        ) * 1000.0

    def _update_command_reset(self, reset_mask: torch.Tensor) -> None:
        """Consume the full-width reset mask without row comparison."""
        timing = self.last_step_timing_ms
        ingested = self._tensor_resample_ingested_mask
        self._tensor_resample_ingested_mask = None
        update_started = time.perf_counter()
        if ingested is None or ingested.data_ptr() != reset_mask.data_ptr():
            self._refresh_motion_torch()
        timing["update_state_command_update_dispatch_ms"] = (
            time.perf_counter() - update_started
        ) * 1000.0

    def post_compute_reset(self) -> None:
        """Run full-width post-compute after reset."""
        self.post_compute()

    def _step_tensor_sampler(self) -> torch.Tensor:
        """Advance the device frame carrier once without a host row transfer."""
        active = ~cast(torch.Tensor, self._env.reset_buf)
        return self.tensor_sampler.step_full(
            active,
            cast(torch.Tensor, self.time_steps),
        )

    def post_compute(self) -> None:
        read_plan = getattr(self._env.scene, "_tensor_read_plan", None)
        view_started = time.perf_counter()
        if read_plan is None or not read_plan.ready:
            if not self._bind_read_phase:
                return
            raise RuntimeError("MotionCommand requires a refreshed scene tensor read phase")
        view = read_plan.body_tensor_view(self.robot, self.cfg.body_names)
        view_ms = (time.perf_counter() - view_started) * 1000.0
        robot_started = time.perf_counter()
        _bind_compiled_motion_post_compute()(
            self._tensor_all_rows,
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

    def _sync_tensor_sampler_state(self) -> None:
        cast(torch.Tensor, self.time_steps).copy_(self.tensor_sampler.current_frames)
        cast(torch.Tensor, self.current_clip_end_frames).copy_(
            self.tensor_sampler.current_clip_end_frames
        )

    @staticmethod
    def _validate_cfg(cfg: MotionCommandCfg) -> None:
        _validate_motion_command_cfg(cfg)

    def _refresh_motion_torch(self) -> None:
        """Publish the device-resident command carrier at full width."""
        self._ingest_motion_packet_mask(
            self._tensor_all_reset_mask, self._motion_packet(self.time_steps)
        )

    def _seed_robot_state_torch_from_defaults(self) -> None:
        """Cold proxy path: seed immutable defaults before CUDA view binding."""
        try:
            root = self.robot.data.default_root_state_torch(self._device)
        except NotImplementedError:
            self._robot_body_pos_w.zero_()
            self._robot_body_quat_w.zero_()
            self._robot_body_quat_w[..., 0] = 1.0
        else:
            self._robot_body_pos_w.copy_(root[:, None, 0:3])
            self._robot_body_quat_w.copy_(root[:, None, 3:7])
        self._robot_body_lin_vel_w.zero_()
        self._robot_body_ang_vel_w.zero_()
        default_joint_pos = self.robot.data.default_joint_pos_torch(self._device)
        self._robot_joint_pos.copy_(default_joint_pos.expand(self.num_envs, -1))
        self._robot_joint_vel.zero_()
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
            width = math.prod(tail)
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

    def _ingest_motion_packet_mask(self, reset_mask: torch.Tensor, packet: torch.Tensor) -> None:
        """Merge a full-width motion packet into only selected rows."""
        scratch = self._motion_packet_scratch
        num_bodies = len(self.cfg.body_names)
        num_joints = self.motion.num_joints
        joint_end = num_joints
        vel_end = num_joints * 2
        body_end = vel_end + num_bodies * 3
        quat_end = body_end + num_bodies * 4
        lin_end = quat_end + num_bodies * 3
        ang_end = lin_end + num_bodies * 3
        torch.where(
            reset_mask[:, None],
            packet[:, :joint_end],
            cast(torch.Tensor, self._motion_data.joint_pos),
            out=cast(torch.Tensor, scratch.joint_pos),
        )
        torch.where(
            reset_mask[:, None],
            packet[:, joint_end:vel_end],
            cast(torch.Tensor, self._motion_data.joint_vel),
            out=cast(torch.Tensor, scratch.joint_vel),
        )
        torch.where(
            reset_mask[:, None, None],
            packet[:, quat_end:lin_end].view(self.num_envs, num_bodies, 3),
            cast(torch.Tensor, self._motion_data.body_lin_vel_w),
            out=cast(torch.Tensor, scratch.body_lin_vel_w),
        )
        torch.where(
            reset_mask[:, None, None],
            packet[:, lin_end:ang_end].view(self.num_envs, num_bodies, 3),
            cast(torch.Tensor, self._motion_data.body_ang_vel_w),
            out=cast(torch.Tensor, scratch.body_ang_vel_w),
        )
        torch.where(
            reset_mask[:, None, None],
            packet[:, vel_end:body_end].view(self.num_envs, num_bodies, 3),
            cast(torch.Tensor, self._motion_data.body_pos_w),
            out=cast(torch.Tensor, scratch.body_pos_w),
        )
        torch.where(
            reset_mask[:, None, None],
            packet[:, body_end:quat_end].view(self.num_envs, num_bodies, 4),
            cast(torch.Tensor, self._motion_data.body_quat_w),
            out=cast(torch.Tensor, scratch.body_quat_w),
        )
        self._motion_data, self._motion_packet_scratch = scratch, self._motion_data
        torch.add(
            packet[:, vel_end:body_end].view(self.num_envs, num_bodies, 3),
            self._env_origins[:, None, :],
            out=self._body_pos_packet_scratch,
        )
        torch.where(
            reset_mask[:, None, None],
            self._body_pos_packet_scratch,
            self._body_pos_w,
            out=self._body_pos_w,
        )
        torch.where(
            reset_mask[:, None],
            packet[:, :vel_end],
            self._command,
            out=self._command,
        )

    def _refresh_relative_state_torch(self) -> None:
        _bind_compiled_motion_relative_state()(
            self._tensor_all_rows,
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
            raise TypeError("MotionCommand row metrics must remain Torch tensors")
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
                raise TypeError("MotionCommand sampler metrics must remain Torch tensors")
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
        # Tensor observation consumers can read this target before the first
        # action application. Initialize deterministically rather than exposing
        # uninitialized device memory in initial/reset observations.
        self._tensor_motion_target = torch.zeros_like(self._processed_actions)
        self._selected_default_bias = torch.empty_like(self._tensor_motion_target)
        self._selected_encoder_bias = torch.empty_like(self._tensor_motion_target)
        self._control_adjustment = torch.empty_like(self._tensor_motion_target)

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
        """Publish the current latency-adjusted raw-input finite verdict."""
        self.publish_validation_verdict()

    def reset(self, reset_mask: torch.Tensor) -> None:
        super().reset(reset_mask)
        self._previous_raw_actions.masked_fill_(reset_mask.reshape(-1, 1), 0.0)

    def apply_actions(self) -> None:
        if isinstance(self._entity.data.control_buffer, torch.Tensor):
            encoder_bias = self._entity.data.encoder_bias_tensor.to(
                device=self._device, non_blocking=True
            )
            default_bias = self._motion_command.joint_default_bias
            if isinstance(default_bias, torch.Tensor):
                selected_default_bias = self._selected_default_bias
                torch.index_select(
                    default_bias,
                    1,
                    self._target_index,
                    out=selected_default_bias,
                )
            else:
                selected_default_bias = torch.as_tensor(
                    default_bias[:, self._target_ids],
                    device=self._device,
                    dtype=torch.float32,
                )
            selected_encoder_bias = self._selected_encoder_bias
            torch.index_select(
                encoder_bias,
                1,
                self._target_index,
                out=selected_encoder_bias,
            )
            control_adjustment = self._control_adjustment
            torch.sub(
                selected_default_bias,
                selected_encoder_bias,
                out=control_adjustment,
            )
            torch.add(
                self._processed_actions,
                control_adjustment,
                out=self._tensor_motion_target,
            )
            self._entity.set_joint_position_target(
                self._tensor_motion_target, joint_ids=self._target_ids
            )
            return
        raise TypeError("MotionJointPositionAction requires a Torch control plane")


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


def motion_anchor_pos_b(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
    return _command(env, command_name).motion_anchor_pos_b


def motion_anchor_ori_b(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
    return _command(env, command_name).motion_anchor_ori_b


def robot_body_pos_b(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
    command = _command(env, command_name)
    return command.robot_body_pos_b.reshape(env.num_envs, -1)


def robot_body_ori_b(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
    command = _command(env, command_name)
    return command.robot_body_ori_b.reshape(env.num_envs, -1)


def motion_joint_pos_rel(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
    command = _command(env, command_name)
    return (
        command.device_robot_joint_pos
        - command.robot.data.default_joint_pos_torch(env.device)
        - cast(torch.Tensor, command.joint_default_bias)
    )


def motion_joint_pos_rel_biased(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
    """Joint position relative to the episode default, including encoder bias."""
    command = _command(env, command_name)
    return (
        command.device_robot_joint_pos
        + command.robot.data.encoder_bias_tensor.to(
            device=env.device, dtype=torch.float32, non_blocking=True
        )
        - command.robot.data.default_joint_pos_torch(env.device)
        - cast(torch.Tensor, command.joint_default_bias)
    )


def _positive_std(value: float, *, term_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{term_name} std must be a real number")
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{term_name} std must be finite and positive")
    return result


def motion_global_anchor_position_error_exp(
    env: ManagerBasedRlEnv, command_name: str, std: float
) -> torch.Tensor:
    command = _command(env, command_name)
    scale = _positive_std(std, term_name="motion anchor position")
    anchor_delta = command.anchor_pos_w - command.robot_anchor_pos_w
    error = cast(torch.Tensor, anchor_delta).square().sum(dim=-1)
    return torch.exp(-error / (scale * scale))


def motion_global_anchor_orientation_error_exp(
    env: ManagerBasedRlEnv, command_name: str, std: float
) -> torch.Tensor:
    command = _command(env, command_name)
    scale = _positive_std(std, term_name="motion anchor orientation")
    motion_anchor_quat = cast(torch.Tensor, command.anchor_quat_w)
    robot_anchor_quat = cast(torch.Tensor, command.robot_anchor_quat_w)
    rel = quat_mul(quat_conjugate(motion_anchor_quat), robot_anchor_quat)
    xyz = torch.linalg.vector_norm(rel[..., 1:4], dim=-1)
    angle = 2.0 * torch.atan2(xyz, rel[..., 0].abs().clamp(max=1.0))
    return torch.exp(-angle.square() / (scale * scale))


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
            self._body_ids = torch.as_tensor(
                [command.cfg.body_names.index(name) for name in requested],
                dtype=torch.long,
                device=env.device,
            )

    def _validate(self, command_name: str, std: float) -> tuple[MotionCommand, float]:
        if command_name != self._command_name:
            raise ValueError(
                f"{type(self).__name__} was bound to '{self._command_name}', received "
                f"'{command_name}'"
            )
        return _command(self._env, command_name), _positive_std(std, term_name=type(self).__name__)

    def reset(self, reset_mask: torch.Tensor) -> None:
        """Mask boundary for stateless motion reward reads."""
        del reset_mask

    @staticmethod
    def _body_reduce(
        command: MotionCommand,
        reference_attr: str,
        actual_attr: str,
        scale: float,
    ) -> torch.Tensor:
        """Apply one fixed body-error reduction with a Torch carrier."""
        body_ids = getattr(command, "_body_ids", slice(None))
        reference = cast(torch.Tensor, getattr(command, reference_attr))[:, body_ids]
        actual = cast(torch.Tensor, getattr(command, actual_attr))[:, body_ids]
        if reference.shape[-1] == 4:
            rel = quat_mul(quat_conjugate(reference), actual)
            xyz = torch.linalg.vector_norm(rel[..., 1:4], dim=-1)
            angle = 2.0 * torch.atan2(xyz, rel[..., 0].abs().clamp(max=1.0))
            error = angle.square().sum(dim=-1)
        else:
            error = (reference - actual).square().sum(dim=-1).sum(dim=-1)
        body_count = reference.shape[-2]
        return torch.exp(-error / (body_count * scale * scale))


class motion_relative_body_position_error_exp(_BodyTerm):
    def __call__(
        self,
        env: ManagerBasedRlEnv,
        command_name: str,
        std: float,
        body_names: tuple[str, ...] | None = None,
    ) -> torch.Tensor:
        del env, body_names
        command, scale = self._validate(command_name, std)
        return self._body_reduce(command, "body_pos_relative_w", "robot_body_pos_w", scale)


class motion_relative_body_orientation_error_exp(_BodyTerm):
    def __call__(
        self,
        env: ManagerBasedRlEnv,
        command_name: str,
        std: float,
        body_names: tuple[str, ...] | None = None,
    ) -> torch.Tensor:
        del env, body_names
        command, scale = self._validate(command_name, std)
        return self._body_reduce(command, "body_quat_relative_w", "robot_body_quat_w", scale)


class motion_global_body_linear_velocity_error_exp(_BodyTerm):
    def __call__(
        self,
        env: ManagerBasedRlEnv,
        command_name: str,
        std: float,
        body_names: tuple[str, ...] | None = None,
    ) -> torch.Tensor:
        del env, body_names
        command, scale = self._validate(command_name, std)
        return self._body_reduce(command, "body_lin_vel_w", "robot_body_lin_vel_w", scale)


class motion_global_body_angular_velocity_error_exp(_BodyTerm):
    def __call__(
        self,
        env: ManagerBasedRlEnv,
        command_name: str,
        std: float,
        body_names: tuple[str, ...] | None = None,
    ) -> torch.Tensor:
        del env, body_names
        command, scale = self._validate(command_name, std)
        return self._body_reduce(command, "body_ang_vel_w", "robot_body_ang_vel_w", scale)


def motion_joint_position_error_exp(
    env: ManagerBasedRlEnv, command_name: str, std: float
) -> torch.Tensor:
    command = _command(env, command_name)
    scale = _positive_std(std, term_name="motion joint position")
    error = (command.joint_pos - command.device_robot_joint_pos).square().mean(dim=-1)
    return torch.exp(-error / (scale * scale))


def motion_joint_velocity_error_exp(
    env: ManagerBasedRlEnv, command_name: str, std: float
) -> torch.Tensor:
    command = _command(env, command_name)
    scale = _positive_std(std, term_name="motion joint velocity")
    error = (command.joint_vel - command.device_robot_joint_vel).square().mean(dim=-1)
    return torch.exp(-error / (scale * scale))


def joint_pos_limits(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Penalize selected joint-limit violations through the packed joint read."""
    asset = cast("Entity", env.scene[asset_cfg.name])
    read_plan = getattr(env.scene, "_tensor_read_plan", None)
    if read_plan is None:
        raise NotImplementedError(
            "Motion tracking term 'joint_pos_limits' requires the Manager packed "
            "joint-state read plan"
        )
    joint_pos = read_plan.joint_tensor_view(asset).joint_pos[:, asset_cfg.joint_ids]
    limits = asset.data.soft_joint_pos_limits_torch(env.device)[asset_cfg.joint_ids]
    lower_violation = (limits[:, 0] - joint_pos).clamp_min(0.0)
    upper_violation = (joint_pos - limits[:, 1]).clamp_min(0.0)
    return (lower_violation + upper_violation).square().sum(dim=-1)


class undesired_body_contacts(_BodyTerm):
    def __call__(
        self,
        env: ManagerBasedRlEnv,
        command_name: str,
        threshold: float,
        body_names: tuple[str, ...] | None = None,
    ) -> torch.Tensor:
        del env, body_names
        command = _command(self._env, command_name)
        robot_body_pos = cast(torch.Tensor, command.robot_body_pos_w)
        return (
            (robot_body_pos[:, self._body_ids, 2] < threshold).sum(dim=-1).to(dtype=torch.float32)
        )


class bad_anchor_pos_z_only(ManagerTermBase):
    """Anchor-height termination on the command's Torch carrier."""

    def __init__(self, cfg: ManagerTermBaseCfg, env: ManagerBasedRlEnv):
        super().__init__(env)
        command_name = cfg.params.get("command_name")
        if not isinstance(command_name, str) or not command_name:
            raise ValueError(f"{type(self).__name__} requires a non-empty command_name")
        self._command_name = command_name

    def __call__(
        self,
        env: ManagerBasedRlEnv,
        command_name: str,
        threshold: float,
    ) -> torch.Tensor:
        del env
        if command_name != self._command_name:
            raise ValueError(
                f"{type(self).__name__} was bound to '{self._command_name}', received "
                f"'{command_name}'"
            )
        command = _command(self._env, command_name)
        motion_anchor_pos = cast(torch.Tensor, command.body_pos_w)
        robot_anchor_pos = cast(torch.Tensor, command.robot_body_pos_w)
        return (
            motion_anchor_pos[:, command.anchor_body_idx, 2]
            - robot_anchor_pos[:, command.anchor_body_idx, 2]
        ).abs() > threshold

    def reset(self, reset_mask: torch.Tensor) -> None:
        """Mask boundary for the stateless termination read."""
        del reset_mask


def bad_anchor_ori(
    env: ManagerBasedRlEnv,
    command_name: str,
    threshold: float,
    asset_cfg: SceneEntityCfg | None = None,
) -> torch.Tensor:
    del asset_cfg
    command = _command(env, command_name)
    motion_anchor_quat = cast(torch.Tensor, command.anchor_quat_w)
    robot_anchor_quat = cast(torch.Tensor, command.robot_anchor_quat_w)
    motion_z = 2.0 * (motion_anchor_quat[:, 1] ** 2 + motion_anchor_quat[:, 2] ** 2) - 1.0
    robot_z = 2.0 * (robot_anchor_quat[:, 1] ** 2 + robot_anchor_quat[:, 2] ** 2) - 1.0
    return (motion_z - robot_z).abs() > threshold


class bad_motion_body_pos_z_only(_BodyTerm):
    def __call__(
        self,
        env: ManagerBasedRlEnv,
        command_name: str,
        threshold: float,
        body_names: tuple[str, ...] | None = None,
    ) -> torch.Tensor:
        del env, body_names
        command = _command(self._env, command_name)
        reference = cast(torch.Tensor, command.body_pos_relative_w)
        actual = cast(torch.Tensor, command.robot_body_pos_w)
        error = (reference[:, self._body_ids, 2] - actual[:, self._body_ids, 2]).abs()
        return torch.any(error > threshold, dim=-1)


def motion_clip_end(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
    command = _command(env, command_name)
    return cast(torch.Tensor, command.time_steps) >= command.tensor_current_clip_end_frames


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
    "MotionJointPositionAction",
    "MotionJointPositionActionCfg",
    "MotionAnchorObservation",
    "MotionAnchorOrientationObservation",
    "MotionAnchorPositionObservation",
    "bad_anchor_ori",
    "bad_anchor_pos_z_only",
    "bad_motion_body_pos_z_only",
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
    "robot_body_ori_b",
    "robot_body_pos_b",
    "undesired_body_contacts",
]
