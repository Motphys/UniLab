"""Manager-native NumPy terms for motion tracking."""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal, cast

import numpy as np

from unilab.envs.mdp.actions import JointPositionAction, JointPositionActionCfg
from unilab.managers import CommandTerm, CommandTermCfg, ManagerTermBase, ManagerTermBaseCfg
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

if TYPE_CHECKING:
    from unilab.base.entity import Entity
    from unilab.managers._types import ManagerBasedRlEnv, ManagerSensorView


SamplingMode = Literal["start", "clip_start", "uniform", "adaptive", "mixed"]
_RANGE_KEYS = ("x", "y", "z", "roll", "pitch", "yaw")
_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")
# Name suffix of the per-actuator ``actuatorfrc`` sensors declared in the task
# scene XML; one scalar sensor per entity joint, named ``<joint_name><suffix>``.
_TORQUE_SENSOR_SUFFIX = "_torque"


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
    # Optional mimic-lite-style observation body set. When set, the command
    # loads a second body-sliced view of the motion and tracks the robot state
    # of these bodies so observation terms can build future-step reference
    # windows without touching the reward-facing ``body_names`` set.
    obs_body_names: tuple[str, ...] | list[str] | None = None
    # Body used as the observation root (reference/robot frame anchor for the
    # root-diff and body-diff observations). Defaults to obs_body_names[0].
    obs_root_body_name: str | None = None


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


@dataclass(frozen=True)
class MotionObsFuture:
    """Future-step reference gather over the configured observation body set.

    All body quantities are world-frame with per-env origins already added to
    positions, gathered at ``clamp(current_frame + step)`` per env within that
    env's current clip bounds. ``S = len(future_steps)``, ``B`` the number of
    configured observation bodies, ``J`` the motion joint width.
    """

    future_steps: tuple[int, ...]
    ref_joint_pos: np.ndarray  # (E, S, J)
    ref_body_pos_w: np.ndarray  # (E, S, B, 3)
    ref_body_quat_w: np.ndarray  # (E, S, B, 4)
    ref_body_lin_vel_w: np.ndarray  # (E, S, B, 3)
    ref_body_ang_vel_w: np.ndarray  # (E, S, B, 3)


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
        # Future-step observation gather cache, keyed by the env step counter.
        # Initialized before any `_refresh_motion`/`_ingest_motion_rows` call
        # (both invalidate it); populated lazily by `obs_future`.
        self._obs_future_cache_step = -1
        self._obs_future_cache: dict[tuple[int, ...], MotionObsFuture] = {}
        obs_body_names = cfg.params.obs_body_names
        if obs_body_names is None:
            self.obs_motion: MotionLoader | None = None
            self.obs_body_names: tuple[str, ...] | None = None
            self.obs_root_body_idx = 0
            self._obs_robot_body_ids: np.ndarray | None = None
            self._copy_obs_robot_body_state = None
            self._obs_robot_body_pos_w: np.ndarray | None = None
            self._obs_robot_body_quat_w: np.ndarray | None = None
            self._obs_robot_body_lin_vel_w: np.ndarray | None = None
            self._obs_robot_body_ang_vel_w: np.ndarray | None = None
        else:
            obs_names = tuple(obs_body_names)
            obs_ids, matched_names = self.robot.find_bodies(obs_names, preserve_order=True)
            if tuple(matched_names) != obs_names:
                raise ValueError(
                    f"MotionCommand obs body order {tuple(matched_names)} does not match "
                    f"{obs_names}"
                )
            self._obs_robot_body_ids = np.asarray(obs_ids, dtype=np.intp)
            self._obs_robot_body_ids.setflags(write=False)
            obs_motion_body_ids = self.robot.motion_body_ids[self._obs_robot_body_ids]
            self.obs_motion = self._make_motion_loader(cfg.motion_file, obs_motion_body_ids)
            if (
                self.obs_motion.fps != self.motion.fps
                or self.obs_motion.num_frames != self.motion.num_frames
                or self.obs_motion.num_joints != self.motion.num_joints
            ):
                raise ValueError(
                    "MotionCommand obs motion view is inconsistent with the main motion "
                    f"(fps {self.obs_motion.fps}/{self.motion.fps}, frames "
                    f"{self.obs_motion.num_frames}/{self.motion.num_frames}, joints "
                    f"{self.obs_motion.num_joints}/{self.motion.num_joints})"
                )
            if self.obs_motion.num_bodies != len(obs_names):
                raise ValueError(
                    f"MotionCommand obs motion body width {self.obs_motion.num_bodies} does "
                    f"not match configured obs body width {len(obs_names)}"
                )
            self.obs_body_names = obs_names
            obs_root = cfg.params.obs_root_body_name or obs_names[0]
            self.obs_root_body_idx = obs_names.index(obs_root)
            self._copy_obs_robot_body_state = self.robot.bind_body_state_copy(
                self._obs_robot_body_ids
            )
            dtype = self.motion.joint_pos.dtype
            num_obs_bodies = len(obs_names)
            self._obs_robot_body_pos_w = np.empty((self.num_envs, num_obs_bodies, 3), dtype=dtype)
            self._obs_robot_body_quat_w = np.empty((self.num_envs, num_obs_bodies, 4), dtype=dtype)
            self._obs_robot_body_lin_vel_w = np.empty_like(self._obs_robot_body_pos_w)
            self._obs_robot_body_ang_vel_w = np.empty_like(self._obs_robot_body_pos_w)
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
        self._joint_torque_sensor_view: ManagerSensorView | None = None
        self._robot_cache_step = -1
        self._all_env_ids = np.arange(self.num_envs, dtype=np.int32)
        self._all_env_ids.setflags(write=False)
        # Env ids of the most recent scoped (reset-path) compute; None after a
        # per-step compute. Written by `_update_command`, consumed by
        # `post_compute` to restrict refresh work to the reset rows.
        self._post_compute_env_ids: np.ndarray | None = None
        # Reset rows whose motion-reference buffers were already ingested by
        # `_resample_command` during the in-flight reset; consumed by the
        # reset-path `_update_command` to skip the redundant `_refresh_motion`
        # gather (issue #1355).
        self._resample_ingested_ids: np.ndarray | None = None
        # Motion rows gathered by the in-flight `_resample_command`, exposed so
        # subclasses (e.g. BoxMotionCommand) reuse the same gather instead of
        # re-reading the same frames.
        self._resample_motion: MotionData | None = None
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
        # Configure and compile both fused kernels on the cold path so the first
        # measured manager step contains no Numba worker/JIT initialization.
        configure_motion_kernel_runtime()
        self._refresh_relative_state()
        self._update_metrics(self._all_env_ids)

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
        obs_body_names = cfg.params.obs_body_names
        if obs_body_names is not None:
            if not obs_body_names:
                raise ValueError("MotionCommandCfg obs_body_names must be non-empty when set")
            if len(set(obs_body_names)) != len(obs_body_names):
                raise ValueError("MotionCommandCfg obs_body_names must be unique")
            obs_root = cfg.params.obs_root_body_name or tuple(obs_body_names)[0]
            if obs_root not in obs_body_names:
                raise ValueError("MotionCommandCfg obs_root_body_name must occur in obs_body_names")

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
    def joint_torque_ref(self) -> np.ndarray | None:
        """Per-env reference joint torques at the current frames, or None.

        Rows follow the same per-env frame gather as every other motion
        reference (``self.time_steps``), so uniform/RSI sampling modes line up
        with the existing tracking rewards. None when the loaded motion files
        carry no torque fields.
        """
        return self._motion_data.joint_torque

    @property
    def joint_torque_limit(self) -> np.ndarray | None:
        """Per-joint torque limits (num_joints,), or None when torque-free."""
        return self.motion.joint_torque_limit

    @property
    def robot_joint_torque(self) -> np.ndarray:
        """Applied joint actuator torques in entity joint order.

        Read from the per-actuator ``actuatorfrc`` sensors
        (``<joint_name>_torque``) declared in the task scene XML — the same
        quantity (MuJoCo ``data.actuator_force``) the offline pipeline exports
        as ``joint_torque``. The sensor view binds lazily on first access so
        runs that never enable the torque reward pay no binding cost.
        """
        if self._joint_torque_sensor_view is None:
            names = tuple(f"{name}{_TORQUE_SENSOR_SUFFIX}" for name in self.robot.joint_names)
            try:
                self._joint_torque_sensor_view = self._env.scene.bind_sensor_data(names)
            except (KeyError, TypeError, ValueError, NotImplementedError) as exc:
                raise type(exc)(
                    f"MotionCommand torque sensors {names} could not be bound; the task scene "
                    f"XML must declare one 'actuatorfrc' sensor per joint: {exc}"
                ) from exc
        return self._joint_torque_sensor_view.read()

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

    def _require_obs_state(self) -> None:
        if self.obs_motion is None:
            raise ValueError(
                "MotionCommand observation bodies are not configured; set params.obs_body_names"
            )

    @property
    def obs_robot_body_pos_w(self) -> np.ndarray:
        """Robot obs-body world positions (origins included), in obs body order."""
        self._require_obs_state()
        self._refresh_robot_state()
        return cast(np.ndarray, self._obs_robot_body_pos_w)

    @property
    def obs_robot_body_quat_w(self) -> np.ndarray:
        self._require_obs_state()
        self._refresh_robot_state()
        return cast(np.ndarray, self._obs_robot_body_quat_w)

    @property
    def obs_robot_body_lin_vel_w(self) -> np.ndarray:
        self._require_obs_state()
        self._refresh_robot_state()
        return cast(np.ndarray, self._obs_robot_body_lin_vel_w)

    @property
    def obs_robot_body_ang_vel_w(self) -> np.ndarray:
        self._require_obs_state()
        self._refresh_robot_state()
        return cast(np.ndarray, self._obs_robot_body_ang_vel_w)

    @property
    def obs_robot_root_pos_w(self) -> np.ndarray:
        return self.obs_robot_body_pos_w[:, self.obs_root_body_idx]

    @property
    def obs_robot_root_quat_w(self) -> np.ndarray:
        return self.obs_robot_body_quat_w[:, self.obs_root_body_idx]

    def obs_future(self, future_steps: Iterable[int]) -> MotionObsFuture:
        """Gather reference motion at ``current_frame + step`` per env.

        Frame indices are clamped per env to that env's current clip bounds
        (mimic-lite ``get_slice`` boundary semantics), so negative/overshooting
        steps repeat the clip's first/last frame. Results are cached per env
        step and per distinct ``future_steps`` tuple; the cache is invalidated
        whenever the motion-reference buffers are refreshed (per-step advance
        and reset-path resample both flow through `_refresh_motion` /
        `_ingest_motion_rows`).
        """
        self._require_obs_state()
        steps = tuple(int(step) for step in future_steps)
        if not steps:
            raise ValueError("obs_future requires at least one future step")
        step_counter = self._env.common_step_counter
        if self._obs_future_cache_step != step_counter:
            self._obs_future_cache.clear()
            self._obs_future_cache_step = step_counter
        cached = self._obs_future_cache.get(steps)
        if cached is None:
            cached = self._gather_obs_future(steps)
            self._obs_future_cache[steps] = cached
        return cached

    def _gather_obs_future(self, steps: tuple[int, ...]) -> MotionObsFuture:
        motion = self.obs_motion
        assert motion is not None  # guaranteed by `_require_obs_state`
        frames = self.time_steps[:, None].astype(np.int64) + np.asarray(steps, dtype=np.int64)
        lower = motion.clip_offsets[self.sampler.current_clip_indices][:, None]
        upper = self.sampler.current_clip_end_frames[:, None].astype(np.int64)
        frames = np.clip(frames, lower, upper)
        body_pos_w = motion.body_pos_w[frames] + self._env.scene.env_origins[:, None, None, :]
        return MotionObsFuture(
            future_steps=steps,
            ref_joint_pos=motion.joint_pos[frames],
            ref_body_pos_w=body_pos_w,
            ref_body_quat_w=motion.body_quat_w[frames],
            ref_body_lin_vel_w=motion.body_lin_vel_w[frames],
            ref_body_ang_vel_w=motion.body_ang_vel_w[frames],
        )

    def reset(self, env_ids: np.ndarray | slice | None) -> dict[str, float]:
        ids = (
            np.arange(self.num_envs, dtype=np.int32)
            if env_ids is None
            else np.arange(self.num_envs, dtype=np.int32)[env_ids]
            if isinstance(env_ids, slice)
            else env_ids
        )
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
        return super().reset(ids)

    def _refresh_motion(self, env_ids: np.ndarray | None = None) -> None:
        """Refresh motion-reference buffers from the current frame indices.

        With env_ids=None all rows are refreshed in place; with env_ids only
        those rows are gathered and scattered (partial-reset path). Rows outside
        env_ids keep the values produced by the last per-step refresh, which are
        still valid because untouched envs did not advance or resample frames.

        Subclass contract (issue #1355): on the reset path the row-scoped
        refresh may be skipped when `_resample_command` already ingested the
        same rows through `_ingest_motion_rows`. A subclass that overrides this
        method to refresh additional buffers must override
        `_ingest_motion_rows` with the same additions (see BoxMotionCommand).
        """
        # Any motion-reference refresh invalidates the future-step obs cache:
        # resampled rows changed frames within the same env step counter.
        self._obs_future_cache_step = -1
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
        """Scatter one gathered motion frame set into the reference buffers.

        Shared by the row-scoped `_refresh_motion` and by `_resample_command`,
        so the reset path gathers each reset row's motion frame exactly once
        (issue #1355).
        """
        # Rows ingested here changed frames; invalidate the obs cache on the
        # direct `_resample_command` path as well (`_refresh_motion` already
        # invalidates at entry).
        self._obs_future_cache_step = -1
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
            if self._copy_obs_robot_body_state is not None:
                self._copy_obs_robot_body_state(
                    cast(np.ndarray, self._obs_robot_body_pos_w),
                    cast(np.ndarray, self._obs_robot_body_quat_w),
                    cast(np.ndarray, self._obs_robot_body_lin_vel_w),
                    cast(np.ndarray, self._obs_robot_body_ang_vel_w),
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
            obs_ids = self._obs_robot_body_ids
            if obs_ids is not None:
                obs_pos_w = cast(np.ndarray, self._obs_robot_body_pos_w)
                obs_quat_w = cast(np.ndarray, self._obs_robot_body_quat_w)
                obs_lin_vel_w = cast(np.ndarray, self._obs_robot_body_lin_vel_w)
                obs_ang_vel_w = cast(np.ndarray, self._obs_robot_body_ang_vel_w)
                obs_pos_w[env_ids] = data.body_link_pos_w_rows(env_ids)[:, obs_ids]
                obs_quat_w[env_ids] = data.body_link_quat_w_rows(env_ids)[:, obs_ids]
                obs_lin_vel_w[env_ids] = data.body_link_lin_vel_w_rows(env_ids)[:, obs_ids]
                obs_ang_vel_w[env_ids] = data.body_link_ang_vel_w_rows(env_ids)[:, obs_ids]
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

    def _update_metrics(self, env_ids: np.ndarray | None = None) -> None:
        # The row-wise error metrics are consumed only by `reset()` (episode
        # log extras), which refreshes exactly the rows it reads. The per-step
        # call (env_ids=None) therefore skips the Numba kernel over all rows
        # (issue #1355); the reset path (env_ids set) refreshes the reset rows
        # so post-reset metrics track the post-reset state.
        if env_ids is not None:
            self._update_error_metrics(env_ids)
        # Sampler statistics are global scalars, so every row tracks them.
        self.metrics["sampling_entropy"].fill(self.sampler.sampling_entropy)
        self.metrics["sampling_top1_prob"].fill(self.sampler.sampling_top1_prob)
        self.metrics["sampling_top1_bin"].fill(self.sampler.sampling_top1_bin)

    def _update_error_metrics(self, rows: np.ndarray) -> None:
        """Recompute the row-wise error metrics for the given rows.

        All row-wise metrics are written by one Numba kernel.  Passing an
        explicit all-row index buffer for normal steps lets the same kernel
        serve partial-reset rows without retaining a NumPy runtime formula.
        """
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
            self.metrics["error_anchor_pos"],
            self.metrics["error_anchor_rot"],
            self.metrics["error_anchor_lin_vel"],
            self.metrics["error_anchor_ang_vel"],
            self.metrics["error_body_pos"],
            self.metrics["error_body_rot"],
            self.metrics["error_body_lin_vel"],
            self.metrics["error_body_ang_vel"],
            self.metrics["error_joint_pos"],
            self.metrics["error_joint_vel"],
        )

    def _resample_command(self, env_ids: np.ndarray) -> None:
        """Resample motion frames and stage the corresponding state writes.

        The base implementation gathers the resampled frames once, ingests them
        into the motion-reference buffers via `_ingest_motion_rows`, and exposes
        the gather as `self._resample_motion` (issue #1355). Subclass contract:
        call ``super()._resample_command(env_ids)`` first and reuse
        `self._resample_motion` for additional writes instead of re-gathering;
        a subclass that does not call super() leaves `_resample_ingested_ids`
        unset, and the reset-path `_update_command` falls back to a fresh
        `_refresh_motion(env_ids)` gather.
        """
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
        # Keep the motion-reference buffers in sync with the resampled frames so
        # the reset-path `_update_command` does not gather the same rows again
        # (issue #1355). Subclasses reuse `self._resample_motion` for their own
        # reset writes instead of re-gathering the same frames.
        self._ingest_motion_rows(env_ids, motion)
        self._resample_ingested_ids = env_ids
        self._resample_motion = motion

    def _update_command(self, env_ids: np.ndarray | None) -> None:
        self._post_compute_env_ids = env_ids
        if env_ids is not None:
            ingested = self._resample_ingested_ids
            self._resample_ingested_ids = None
            if ingested is None or not np.array_equal(ingested, env_ids):
                self._refresh_motion(env_ids)
            return
        self._resample_ingested_ids = None
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
        # The base class allocates `_target` uninitialized; observation terms
        # (applied_action) can read it before the first `apply_actions`, so
        # start from a deterministic zero target.
        self._target = np.zeros_like(self._processed_actions)
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
    diff = command.anchor_pos_w - command.robot_anchor_pos_w
    np.square(diff, out=diff)
    error = np.sum(diff, axis=-1)
    scale = _positive_std(std, term_name="motion anchor position")
    np.divide(error, -(scale**2), out=error)
    return np.exp(error, out=error)


def motion_global_anchor_orientation_error_exp(
    env: ManagerBasedRlEnv, command_name: str, std: float
) -> np.ndarray:
    command = _command(env, command_name)
    error = np_quat_error_magnitude_squared_batched(
        command.anchor_quat_w, command.robot_anchor_quat_w
    )
    scale = _positive_std(std, term_name="motion anchor orientation")
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
        self._kernel_result = np.empty(self.num_envs, dtype=command.body_pos_relative_w.dtype)

    def _kernel_std(self, scale: float) -> float:
        return cast(float, self._kernel_result.dtype.type(scale))


class motion_relative_body_position_error_exp(_NumbaBodyTerm):
    def __init__(self, cfg: ManagerTermBaseCfg, env: ManagerBasedRlEnv):
        super().__init__(cfg, env)
        command = _command(env, self._command_name)
        reward_motion_body_pos_kernel(
            command.body_pos_relative_w,
            command.robot_body_pos_w,
            self._kernel_body_ids,
            self._kernel_std(1.0),
            self._kernel_result,
        )

    def __call__(
        self,
        env: ManagerBasedRlEnv,
        command_name: str,
        std: float,
        body_names: tuple[str, ...] | None = None,
    ) -> np.ndarray:
        del env, body_names
        command, scale = self._validate(command_name, std)
        reward_motion_body_pos_kernel(
            command.body_pos_relative_w,
            command.robot_body_pos_w,
            self._kernel_body_ids,
            self._kernel_std(scale),
            self._kernel_result,
        )
        return self._kernel_result


class motion_relative_body_orientation_error_exp(_NumbaBodyTerm):
    def __init__(self, cfg: ManagerTermBaseCfg, env: ManagerBasedRlEnv):
        super().__init__(cfg, env)
        command = _command(env, self._command_name)
        reward_motion_body_ori_kernel(
            command.body_quat_relative_w,
            command.robot_body_quat_w,
            self._kernel_body_ids,
            self._kernel_std(1.0),
            self._kernel_result,
        )

    def __call__(
        self,
        env: ManagerBasedRlEnv,
        command_name: str,
        std: float,
        body_names: tuple[str, ...] | None = None,
    ) -> np.ndarray:
        del env, body_names
        command, scale = self._validate(command_name, std)
        reward_motion_body_ori_kernel(
            command.body_quat_relative_w,
            command.robot_body_quat_w,
            self._kernel_body_ids,
            self._kernel_std(scale),
            self._kernel_result,
        )
        return self._kernel_result


class motion_global_body_linear_velocity_error_exp(_NumbaBodyTerm):
    def __init__(self, cfg: ManagerTermBaseCfg, env: ManagerBasedRlEnv):
        super().__init__(cfg, env)
        command = _command(env, self._command_name)
        reward_motion_body_lin_vel_kernel(
            command.body_lin_vel_w,
            command.robot_body_lin_vel_w,
            self._kernel_body_ids,
            self._kernel_std(1.0),
            self._kernel_result,
        )

    def __call__(
        self,
        env: ManagerBasedRlEnv,
        command_name: str,
        std: float,
        body_names: tuple[str, ...] | None = None,
    ) -> np.ndarray:
        del env, body_names
        command, scale = self._validate(command_name, std)
        reward_motion_body_lin_vel_kernel(
            command.body_lin_vel_w,
            command.robot_body_lin_vel_w,
            self._kernel_body_ids,
            self._kernel_std(scale),
            self._kernel_result,
        )
        return self._kernel_result


class motion_global_body_angular_velocity_error_exp(_NumbaBodyTerm):
    def __init__(self, cfg: ManagerTermBaseCfg, env: ManagerBasedRlEnv):
        super().__init__(cfg, env)
        command = _command(env, self._command_name)
        reward_motion_body_ang_vel_kernel(
            command.body_ang_vel_w,
            command.robot_body_ang_vel_w,
            self._kernel_body_ids,
            self._kernel_std(1.0),
            self._kernel_result,
        )

    def __call__(
        self,
        env: ManagerBasedRlEnv,
        command_name: str,
        std: float,
        body_names: tuple[str, ...] | None = None,
    ) -> np.ndarray:
        del env, body_names
        command, scale = self._validate(command_name, std)
        reward_motion_body_ang_vel_kernel(
            command.body_ang_vel_w,
            command.robot_body_ang_vel_w,
            self._kernel_body_ids,
            self._kernel_std(scale),
            self._kernel_result,
        )
        return self._kernel_result


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
        return self._exp_neg_scaled(error.mean(axis=-1), scale)


def motion_joint_position_error_exp(
    env: ManagerBasedRlEnv, command_name: str, std: float
) -> np.ndarray:
    command = _command(env, command_name)
    diff = command.joint_pos - command.robot_joint_pos
    np.square(diff, out=diff)
    error = diff.mean(axis=-1)
    scale = _positive_std(std, term_name="motion joint position")
    np.divide(error, -(scale**2), out=error)
    return np.exp(error, out=error)


def motion_joint_velocity_error_exp(
    env: ManagerBasedRlEnv, command_name: str, std: float
) -> np.ndarray:
    command = _command(env, command_name)
    diff = command.joint_vel - command.robot_joint_vel
    np.square(diff, out=diff)
    error = diff.mean(axis=-1)
    scale = _positive_std(std, term_name="motion joint velocity")
    np.divide(error, -(scale**2), out=error)
    return np.exp(error, out=error)


def motion_joint_torque(env: ManagerBasedRlEnv, command_name: str, std: float) -> np.ndarray:
    """KDTO+T-style joint torque tracking: ``exp(-mean(((tau - tau*) / tau_max)^2) / std**2)``.

    ``tau`` is the simulator's applied joint actuator torque (per-actuator
    ``actuatorfrc`` sensors, i.e. MuJoCo ``data.actuator_force``), ``tau*`` the
    motion reference torque at each env's current frame, and ``tau_max`` the
    per-joint torque limit from the motion file. Returns zeros when the loaded
    motion carries no torque fields so torque-free baselines keep running even
    if the term is configured with a nonzero weight.
    """
    command = _command(env, command_name)
    ref = command.joint_torque_ref
    limit = command.joint_torque_limit
    if ref is None or limit is None:
        return np.zeros(env.num_envs, dtype=command.joint_pos.dtype)
    scale = _positive_std(std, term_name="motion joint torque")
    diff = command.robot_joint_torque - ref
    np.divide(diff, limit, out=diff)
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
    ) -> np.ndarray:
        del env, body_names
        command = _command(self._env, command_name)
        return np.sum(command.robot_body_pos_w[:, self._body_ids, 2] < threshold, axis=-1)


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
        threshold = command.body_pos_w.dtype.type(cfg.params.get("threshold", 0.0))
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
    ) -> np.ndarray:
        del env
        if command_name != self._command_name:
            raise ValueError(
                f"{type(self).__name__} was bound to '{self._command_name}', got '{command_name}'"
            )
        command = _command(self._env, command_name)
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
    "MotionObsFuture",
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
    "motion_joint_torque",
    "motion_joint_velocity_error_exp",
    "motion_relative_body_orientation_error_exp",
    "motion_relative_body_position_error_exp",
    "motion_relative_body_position_z_error_exp",
    "robot_body_ori_b",
    "robot_body_pos_b",
    "undesired_body_contacts",
]
