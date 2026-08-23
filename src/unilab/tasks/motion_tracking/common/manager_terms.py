"""Manager-native NumPy terms for motion tracking."""

from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, Literal, cast

import numpy as np

from unilab.envs.mdp.actions import JointPositionAction, JointPositionActionCfg
from unilab.managers import CommandTerm, CommandTermCfg, ManagerTermBase, ManagerTermBaseCfg
from unilab.managers.scene_entity_config import SceneEntityCfg
from unilab.utils.geometry import (
    np_write_relative_anchor_transform_pos_rot6d,
)
from unilab.utils.rotation import (
    np_quat_apply_inverse,
    np_quat_error_magnitude_squared_batched,
    np_quat_from_euler_xyz,
    np_quat_mul,
)

from .motion_loader import MotionData, MotionLoader, MotionSampler
from .observations import write_body_ori6_in_anchor_frame, write_body_pos_in_anchor_frame
from .transforms import update_relative_transforms

if TYPE_CHECKING:
    from unilab.base.entity import Entity
    from unilab.managers._types import ManagerBasedRlEnv


SamplingMode = Literal["start", "clip_start", "uniform", "adaptive", "mixed"]
_RANGE_KEYS = ("x", "y", "z", "roll", "pitch", "yaw")
_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


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
    """Motion reference command on UniLab's NumPy/entity runtime."""

    cfg: MotionCommandCfg

    def __init__(self, cfg: MotionCommandCfg, env: ManagerBasedRlEnv):
        self._validate_cfg(cfg)
        super().__init__(cfg, env)
        self.robot = cast("Entity", env.scene[cfg.entity_name])
        body_ids, body_names = self.robot.find_bodies(cfg.body_names, preserve_order=True)
        if tuple(body_names) != cfg.body_names:
            raise ValueError(
                f"MotionCommand body order {tuple(body_names)} does not match {cfg.body_names}"
            )
        self._robot_body_ids = np.asarray(body_ids, dtype=np.intp)
        self._robot_body_ids.setflags(write=False)
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
            mode=cfg.params.sampling_mode,
            num_envs=self.num_envs,
            adaptive_lambda=cfg.params.adaptive_lambda,
            adaptive_kernel_size=cfg.params.adaptive_kernel_size,
            adaptive_uniform_ratio=cfg.params.adaptive_uniform_ratio,
            adaptive_alpha=cfg.params.adaptive_alpha,
            start_ratio=cfg.params.sampling_start_ratio,
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
        self._delta_pos_w = np.empty((self.num_envs, 3), dtype=dtype)
        self._delta_ori_w = np.empty((self.num_envs, 4), dtype=dtype)
        self._body_vec_error = np.empty_like(self._body_pos_w)
        self._env_error = np.empty(self.num_envs, dtype=dtype)
        self._reward_term = np.empty(self.num_envs, dtype=dtype)
        self._robot_cache_step = -1
        # Env ids of the most recent scoped (reset-path) compute; None after a
        # per-step compute. Written by `_update_command`, consumed by
        # `post_compute` to restrict refresh work to the reset rows.
        self._post_compute_env_ids: np.ndarray | None = None
        self._robot_body_pos_w = np.empty_like(self._body_pos_w)
        self._robot_body_quat_w = np.empty((self.num_envs, num_bodies, 4), dtype=dtype)
        self._robot_body_lin_vel_w = np.empty_like(self._body_pos_w)
        self._robot_body_ang_vel_w = np.empty_like(self._body_pos_w)

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
        self._refresh_motion()
        self._refresh_robot_state(force=True)
        self._refresh_relative_state()

    def _make_motion_loader(
        self,
        motion_file: str | list[str],
        body_indices: np.ndarray,
    ) -> MotionLoader:
        """Materialize the profile-owned motion loader on the cold path."""
        return MotionLoader(motion_file, body_indices=body_indices)

    @staticmethod
    def _validate_cfg(cfg: MotionCommandCfg) -> None:
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
            raise ValueError(
                f"MotionCommandCfg has unsupported sampling_mode {cfg.sampling_mode!r}"
            )
        if not 0.0 <= cfg.params.sampling_start_ratio <= 1.0:
            raise ValueError("MotionCommandCfg sampling_start_ratio must be within [0, 1]")
        if not isinstance(cfg.params.truncate_on_clip_end, bool):
            raise TypeError("MotionCommandCfg truncate_on_clip_end must be bool")

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

    def reset(self, env_ids: np.ndarray | slice | None) -> dict[str, float]:
        ids = (
            np.arange(self.num_envs, dtype=np.int32)
            if env_ids is None
            else np.arange(self.num_envs, dtype=np.int32)[env_ids]
            if isinstance(env_ids, slice)
            else env_ids
        )
        lower, upper = self._joint_default_position_range
        self.joint_default_bias[ids] = self._env.rng.uniform(
            lower, upper, size=(len(ids), self.motion.num_joints)
        )
        return super().reset(ids)

    def _refresh_motion(self, env_ids: np.ndarray | None = None) -> None:
        """Refresh motion-reference buffers from the current frame indices.

        With env_ids=None all rows are refreshed in place; with env_ids only
        those rows are gathered and scattered (partial-reset path). Rows outside
        env_ids keep the values produced by the last per-step refresh, which are
        still valid because untouched envs did not advance or resample frames.
        """
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
        data = self.motion.get_motion_at_frame(self.time_steps[env_ids])
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
        sel: np.ndarray | slice = slice(None) if env_ids is None else env_ids
        body_index = self._robot_body_ids
        self._robot_body_pos_w[sel] = self.robot.data.body_link_pos_w[sel][:, body_index]
        self._robot_body_quat_w[sel] = self.robot.data.body_link_quat_w[sel][:, body_index]
        self._robot_body_lin_vel_w[sel] = self.robot.data.body_link_lin_vel_w[sel][:, body_index]
        self._robot_body_ang_vel_w[sel] = self.robot.data.body_link_ang_vel_w[sel][:, body_index]
        self._robot_cache_step = step

    def _refresh_relative_state(self, env_ids: np.ndarray | None = None) -> None:
        if env_ids is not None:
            self._refresh_relative_state_rows(env_ids)
            return
        update_relative_transforms(
            self,
            self._motion_data,
            self._robot_body_pos_w,
            self._robot_body_quat_w,
        )
        np_write_relative_anchor_transform_pos_rot6d(
            self.robot_anchor_pos_w,
            self.robot_anchor_quat_w,
            self.anchor_pos_w,
            self.anchor_quat_w,
            self.motion_anchor_pos_b,
            self.motion_anchor_ori_b,
        )
        write_body_pos_in_anchor_frame(
            self.robot_anchor_pos_w,
            self.robot_anchor_quat_w,
            self._robot_body_pos_w,
            self.robot_body_pos_b,
            body_vec_error=self._body_vec_error,
        )
        write_body_ori6_in_anchor_frame(
            self.robot_anchor_quat_w,
            self._robot_body_quat_w,
            self.robot_body_ori_b,
        )

    def _refresh_relative_state_rows(self, env_ids: np.ndarray) -> None:
        """Row-scoped variant of `_refresh_relative_state` for partial resets.

        Computes the same transforms on the gathered reset rows and scatters the
        results back; untouched rows keep their per-step values.
        """
        num_rows = len(env_ids)
        num_bodies = self._robot_body_pos_w.shape[1]
        dtype = self._body_pos_w.dtype
        robot_pos_rows = self._robot_body_pos_w[env_ids]
        robot_quat_rows = self._robot_body_quat_w[env_ids]
        motion_rows = SimpleNamespace(
            body_pos_w=self._motion_data.body_pos_w[env_ids],
            body_quat_w=self._motion_data.body_quat_w[env_ids],
        )
        # `update_relative_transforms` reads/writes these attributes on the env;
        # a namespace with row-sized scratch keeps the shared buffers untouched.
        scratch = SimpleNamespace(
            anchor_body_idx=self.anchor_body_idx,
            _delta_pos_w=np.empty((num_rows, 3), dtype=dtype),
            _delta_ori_w=np.empty((num_rows, 4), dtype=dtype),
            body_quat_relative_w=np.empty((num_rows, num_bodies, 4), dtype=dtype),
            body_pos_relative_w=np.empty((num_rows, num_bodies, 3), dtype=dtype),
            _body_vec_error=np.empty((num_rows, num_bodies, 3), dtype=dtype),
            _env_error=np.empty(num_rows, dtype=dtype),
            _reward_term=np.empty(num_rows, dtype=dtype),
        )
        update_relative_transforms(scratch, motion_rows, robot_pos_rows, robot_quat_rows)
        self.body_pos_relative_w[env_ids] = scratch.body_pos_relative_w
        self.body_quat_relative_w[env_ids] = scratch.body_quat_relative_w

        anchor_idx = self.anchor_body_idx
        robot_anchor_pos_rows = robot_pos_rows[:, anchor_idx]
        robot_anchor_quat_rows = robot_quat_rows[:, anchor_idx]
        anchor_pos_rows = self._body_pos_w[env_ids][:, anchor_idx]
        anchor_quat_rows = motion_rows.body_quat_w[:, anchor_idx]
        motion_anchor_pos_b = np.empty((num_rows, 3), dtype=dtype)
        motion_anchor_ori_b = np.empty((num_rows, 6), dtype=dtype)
        np_write_relative_anchor_transform_pos_rot6d(
            robot_anchor_pos_rows,
            robot_anchor_quat_rows,
            anchor_pos_rows,
            anchor_quat_rows,
            motion_anchor_pos_b,
            motion_anchor_ori_b,
        )
        self.motion_anchor_pos_b[env_ids] = motion_anchor_pos_b
        self.motion_anchor_ori_b[env_ids] = motion_anchor_ori_b
        robot_body_pos_b = np.empty((num_rows, num_bodies, 3), dtype=dtype)
        write_body_pos_in_anchor_frame(
            robot_anchor_pos_rows,
            robot_anchor_quat_rows,
            robot_pos_rows,
            robot_body_pos_b,
            body_vec_error=scratch._body_vec_error,
        )
        self.robot_body_pos_b[env_ids] = robot_body_pos_b
        robot_body_ori_b = np.empty((num_rows, num_bodies, 6), dtype=dtype)
        write_body_ori6_in_anchor_frame(
            robot_anchor_quat_rows,
            robot_quat_rows,
            robot_body_ori_b,
        )
        self.robot_body_ori_b[env_ids] = robot_body_ori_b

    def _update_metrics(self, env_ids: np.ndarray | None = None) -> None:
        # All error metrics are row-wise functions of the motion/robot buffers;
        # on the reset path only the reset rows are recomputed since other rows
        # are unchanged since the per-step update.
        sel: np.ndarray | slice = slice(None) if env_ids is None else env_ids
        self.metrics["error_anchor_pos"][sel] = np.linalg.norm(
            self.anchor_pos_w[sel] - self.robot_anchor_pos_w[sel], axis=-1
        )
        self.metrics["error_anchor_rot"][sel] = np.sqrt(
            np_quat_error_magnitude_squared_batched(
                self.anchor_quat_w[sel], self.robot_anchor_quat_w[sel]
            )
        )
        self.metrics["error_anchor_lin_vel"][sel] = np.linalg.norm(
            self.anchor_lin_vel_w[sel] - self.robot_anchor_lin_vel_w[sel], axis=-1
        )
        self.metrics["error_anchor_ang_vel"][sel] = np.linalg.norm(
            self.anchor_ang_vel_w[sel] - self.robot_anchor_ang_vel_w[sel], axis=-1
        )
        self.metrics["error_body_pos"][sel] = np.linalg.norm(
            self.body_pos_relative_w[sel] - self.robot_body_pos_w[sel], axis=-1
        ).mean(axis=-1)
        self.metrics["error_body_rot"][sel] = np.sqrt(
            np_quat_error_magnitude_squared_batched(
                self.body_quat_relative_w[sel], self.robot_body_quat_w[sel]
            )
        ).mean(axis=-1)
        self.metrics["error_body_lin_vel"][sel] = np.linalg.norm(
            self.body_lin_vel_w[sel] - self.robot_body_lin_vel_w[sel], axis=-1
        ).mean(axis=-1)
        self.metrics["error_body_ang_vel"][sel] = np.linalg.norm(
            self.body_ang_vel_w[sel] - self.robot_body_ang_vel_w[sel], axis=-1
        ).mean(axis=-1)
        self.metrics["error_joint_pos"][sel] = np.linalg.norm(
            self.joint_pos[sel] - self.robot_joint_pos[sel], axis=-1
        )
        self.metrics["error_joint_vel"][sel] = np.linalg.norm(
            self.joint_vel[sel] - self.robot_joint_vel[sel], axis=-1
        )
        # Sampler statistics are global scalars, so every row tracks them.
        self.metrics["sampling_entropy"].fill(self.sampler.sampling_entropy)
        self.metrics["sampling_top1_prob"].fill(self.sampler.sampling_top1_prob)
        self.metrics["sampling_top1_bin"].fill(self.sampler.sampling_top1_bin)

    def _resample_command(self, env_ids: np.ndarray) -> None:
        frames = self.sampler.sample_frames(env_ids)
        motion = self.motion.get_motion_at_frame(frames)
        count = len(env_ids)
        pose = self._env.rng.uniform(
            self._pose_range[:, 0], self._pose_range[:, 1], size=(count, 6)
        )
        velocity = self._env.rng.uniform(
            self._velocity_range[:, 0], self._velocity_range[:, 1], size=(count, 6)
        )
        root_pos = motion.body_pos_w[:, 0].copy()
        root_pos += self._env.scene.env_origins[env_ids]
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
        self.robot.write_joint_state_to_sim(joint_pos, motion.joint_vel, env_ids=env_ids)
        root_state = np.concatenate((root_pos, root_quat, root_lin_vel, root_ang_vel), axis=-1)
        self.robot.write_root_state_to_sim(root_state, env_ids=env_ids)

    def _update_command(self, env_ids: np.ndarray | None) -> None:
        self._post_compute_env_ids = env_ids
        if env_ids is not None:
            self._refresh_motion(env_ids)
            return
        self.sampler.update_failure_stats(self._env.termination_manager.terminated)
        active_ids = np.flatnonzero(~self._env.reset_buf).astype(np.int32, copy=False)
        wrap_ids = self.sampler.step(active_ids)
        if len(wrap_ids) and not self.cfg.params.truncate_on_clip_end:
            self._resample_command(wrap_ids)
        self._refresh_motion()

    def post_compute(self) -> None:
        # On the reset path only the reset rows changed (via the committed
        # set_state writes and the motion resample), so refresh just those rows.
        env_ids = self._post_compute_env_ids
        self._refresh_robot_state(force=True, env_ids=env_ids)
        self._refresh_relative_state(env_ids)


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
        self._previous_raw_actions = np.zeros_like(self._raw_actions)

    @property
    def target(self) -> np.ndarray:
        """Most recently applied physical joint target in entity joint order."""
        return self._target

    def process_actions(self, actions: np.ndarray) -> None:
        self._previous_raw_actions[:] = self._raw_actions
        super().process_actions(actions)
        if not self.cfg.simulate_action_latency:
            return
        np.multiply(self._previous_raw_actions, self._scale, out=self._processed_actions)
        np.add(self._processed_actions, self._offset, out=self._processed_actions)
        if self._clip is not None:
            np.clip(
                self._processed_actions,
                self._clip[..., 0],
                self._clip[..., 1],
                out=self._processed_actions,
            )

    def reset(self, env_ids: np.ndarray | slice | None = None) -> None:
        super().reset(env_ids)
        ids = slice(None) if env_ids is None else env_ids
        self._previous_raw_actions[ids] = 0.0

    def apply_actions(self) -> None:
        encoder_bias = self._entity.data.encoder_bias[:, self._target_ids]
        np.add(
            self._processed_actions,
            self._motion_command.joint_default_bias[:, self._target_ids],
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


def motion_anchor_pos_b(env: ManagerBasedRlEnv, command_name: str) -> np.ndarray:
    return _command(env, command_name).motion_anchor_pos_b


def motion_anchor_ori_b(env: ManagerBasedRlEnv, command_name: str) -> np.ndarray:
    return _command(env, command_name).motion_anchor_ori_b


def robot_body_pos_b(env: ManagerBasedRlEnv, command_name: str) -> np.ndarray:
    command = _command(env, command_name)
    return command.robot_body_pos_b.reshape(env.num_envs, -1)


def robot_body_ori_b(env: ManagerBasedRlEnv, command_name: str) -> np.ndarray:
    command = _command(env, command_name)
    return command.robot_body_ori_b.reshape(env.num_envs, -1)


def motion_joint_pos_rel(env: ManagerBasedRlEnv, command_name: str) -> np.ndarray:
    command = _command(env, command_name)
    return (
        command.robot_joint_pos - command.robot.data.default_joint_pos - command.joint_default_bias
    )


def motion_joint_pos_rel_biased(env: ManagerBasedRlEnv, command_name: str) -> np.ndarray:
    """Joint position relative to the episode default, including encoder bias."""
    command = _command(env, command_name)
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
) -> np.ndarray:
    command = _command(env, command_name)
    error = np.sum(np.square(command.anchor_pos_w - command.robot_anchor_pos_w), axis=-1)
    return np.exp(-error / _positive_std(std, term_name="motion anchor position") ** 2)


def motion_global_anchor_orientation_error_exp(
    env: ManagerBasedRlEnv, command_name: str, std: float
) -> np.ndarray:
    command = _command(env, command_name)
    error = np_quat_error_magnitude_squared_batched(
        command.anchor_quat_w, command.robot_anchor_quat_w
    )
    return np.exp(-error / _positive_std(std, term_name="motion anchor orientation") ** 2)


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

    def _validate(self, command_name: str, std: float) -> tuple[MotionCommand, float]:
        if command_name != self._command_name:
            raise ValueError(
                f"{type(self).__name__} was bound to '{self._command_name}', got '{command_name}'"
            )
        return _command(self._env, command_name), _positive_std(std, term_name=type(self).__name__)


class motion_relative_body_position_error_exp(_BodyTerm):
    def __call__(
        self,
        env: ManagerBasedRlEnv,
        command_name: str,
        std: float,
        body_names: tuple[str, ...] | None = None,
    ) -> np.ndarray:
        del env, body_names
        command, scale = self._validate(command_name, std)
        error = np.sum(
            np.square(
                command.body_pos_relative_w[:, self._body_ids]
                - command.robot_body_pos_w[:, self._body_ids]
            ),
            axis=-1,
        )
        return np.exp(-error.mean(axis=-1) / scale**2)


class motion_relative_body_orientation_error_exp(_BodyTerm):
    def __call__(
        self,
        env: ManagerBasedRlEnv,
        command_name: str,
        std: float,
        body_names: tuple[str, ...] | None = None,
    ) -> np.ndarray:
        del env, body_names
        command, scale = self._validate(command_name, std)
        error = np_quat_error_magnitude_squared_batched(
            command.body_quat_relative_w[:, self._body_ids],
            command.robot_body_quat_w[:, self._body_ids],
        )
        return np.exp(-error.mean(axis=-1) / scale**2)


class motion_global_body_linear_velocity_error_exp(_BodyTerm):
    def __call__(
        self,
        env: ManagerBasedRlEnv,
        command_name: str,
        std: float,
        body_names: tuple[str, ...] | None = None,
    ) -> np.ndarray:
        del env, body_names
        command, scale = self._validate(command_name, std)
        error = np.sum(
            np.square(
                command.body_lin_vel_w[:, self._body_ids]
                - command.robot_body_lin_vel_w[:, self._body_ids]
            ),
            axis=-1,
        )
        return np.exp(-error.mean(axis=-1) / scale**2)


class motion_global_body_angular_velocity_error_exp(_BodyTerm):
    def __call__(
        self,
        env: ManagerBasedRlEnv,
        command_name: str,
        std: float,
        body_names: tuple[str, ...] | None = None,
    ) -> np.ndarray:
        del env, body_names
        command, scale = self._validate(command_name, std)
        error = np.sum(
            np.square(
                command.body_ang_vel_w[:, self._body_ids]
                - command.robot_body_ang_vel_w[:, self._body_ids]
            ),
            axis=-1,
        )
        return np.exp(-error.mean(axis=-1) / scale**2)


class motion_relative_body_position_z_error_exp(_BodyTerm):
    def __call__(
        self,
        env: ManagerBasedRlEnv,
        command_name: str,
        std: float,
        body_names: tuple[str, ...] | None = None,
    ) -> np.ndarray:
        del env, body_names
        command, scale = self._validate(command_name, std)
        error = np.square(
            command.body_pos_relative_w[:, self._body_ids, 2]
            - command.robot_body_pos_w[:, self._body_ids, 2]
        )
        return np.exp(-error.mean(axis=-1) / scale**2)


def motion_joint_position_error_exp(
    env: ManagerBasedRlEnv, command_name: str, std: float
) -> np.ndarray:
    command = _command(env, command_name)
    error = np.square(command.joint_pos - command.robot_joint_pos).mean(axis=-1)
    return np.exp(-error / _positive_std(std, term_name="motion joint position") ** 2)


def motion_joint_velocity_error_exp(
    env: ManagerBasedRlEnv, command_name: str, std: float
) -> np.ndarray:
    command = _command(env, command_name)
    error = np.square(command.joint_vel - command.robot_joint_vel).mean(axis=-1)
    return np.exp(-error / _positive_std(std, term_name="motion joint velocity") ** 2)


def joint_pos_limits(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> np.ndarray:
    """Penalize selected joint-limit violations through the entity facade."""
    asset = cast("Entity", env.scene[asset_cfg.name])
    joint_pos = asset.data.joint_pos[:, asset_cfg.joint_ids]
    limits = asset.data.soft_joint_pos_limits[asset_cfg.joint_ids]
    lower_error = np.maximum(limits[:, 0] - joint_pos, 0.0)
    upper_error = np.maximum(joint_pos - limits[:, 1], 0.0)
    return np.sum(np.square(lower_error + upper_error), axis=-1)


class undesired_body_contacts(_BodyTerm):
    def __call__(
        self,
        env: ManagerBasedRlEnv,
        command_name: str,
        threshold: float,
        body_names: tuple[str, ...] | None = None,
    ) -> np.ndarray:
        del env, body_names
        command = _command(self._env, command_name)
        return np.sum(command.robot_body_pos_w[:, self._body_ids, 2] < threshold, axis=-1)


def bad_anchor_pos_z_only(
    env: ManagerBasedRlEnv, command_name: str, threshold: float
) -> np.ndarray:
    command = _command(env, command_name)
    return np.abs(command.anchor_pos_w[:, 2] - command.robot_anchor_pos_w[:, 2]) > threshold


def bad_anchor_ori(
    env: ManagerBasedRlEnv,
    command_name: str,
    threshold: float,
    asset_cfg: SceneEntityCfg | None = None,
) -> np.ndarray:
    command = _command(env, command_name)
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
    ) -> np.ndarray:
        del env, body_names
        command = _command(self._env, command_name)
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
    ) -> np.ndarray:
        del env, body_names
        command = _command(self._env, command_name)
        return np.any(command.robot_body_pos_w[:, self._body_ids, 2] < threshold, axis=-1)


def motion_clip_end(env: ManagerBasedRlEnv, command_name: str) -> np.ndarray:
    command = _command(env, command_name)
    return command.time_steps >= command.sampler.current_clip_end_frames


__all__ = [
    "MotionCommand",
    "MotionCommandCfg",
    "MotionCommandParamsCfg",
    "MotionJointPositionAction",
    "MotionJointPositionActionCfg",
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
