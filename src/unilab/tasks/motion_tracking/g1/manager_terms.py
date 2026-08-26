"""G1 profile-specific NumPy manager terms for motion tracking."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

import numpy as np

from unilab.managers import ManagerTermBase, ManagerTermBaseCfg
from unilab.managers.scene_entity_config import SceneEntityCfg
from unilab.tasks.motion_tracking.common.kernels import update_object_relative_state_kernel
from unilab.tasks.motion_tracking.common.manager_terms import (
    MotionCommand,
    MotionCommandCfg,
    MotionJointPositionAction,
)
from unilab.utils.rotation import np_quat_error_magnitude_squared_batched

from .motion_box_loader import BoxMotionData, BoxMotionLoader

if TYPE_CHECKING:
    from unilab.base.entity import Entity
    from unilab.managers._types import ManagerBasedRlEnv


_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


@dataclass(kw_only=True)
class BoxMotionCommandCfg(MotionCommandCfg):
    """Motion command extended with one independently simulated object."""

    object_entity_name: str

    def build(self, env: ManagerBasedRlEnv) -> BoxMotionCommand:
        return BoxMotionCommand(self, env)


class BoxMotionCommand(MotionCommand):
    cfg: BoxMotionCommandCfg  # pyright: ignore[reportIncompatibleVariableOverride]

    def __init__(self, cfg: BoxMotionCommandCfg, env: ManagerBasedRlEnv):
        if not isinstance(cfg.object_entity_name, str) or not cfg.object_entity_name:
            raise ValueError("BoxMotionCommandCfg object_entity_name must be non-empty")
        self.object = cast("Entity", env.scene[cfg.object_entity_name])
        self._object_pos_w = np.empty((env.num_envs, 3), dtype=np.float32)
        self._object_obs_b = np.empty((env.num_envs, 12), dtype=np.float32)
        super().__init__(cfg, env)
        if not isinstance(self._motion_data, BoxMotionData):
            raise TypeError("BoxMotionCommand requires BoxMotionData")
        self._refresh_object_state()

    def _make_motion_loader(
        self,
        motion_file: str | list[str],
        body_indices: np.ndarray,
    ) -> BoxMotionLoader:
        return BoxMotionLoader(motion_file, body_indices=body_indices)

    @property
    def box_motion(self) -> BoxMotionData:
        return cast(BoxMotionData, self._motion_data)

    @property
    def object_pos_w(self) -> np.ndarray:
        return self._object_pos_w

    @property
    def object_quat_w(self) -> np.ndarray:
        value = self.box_motion.object_quat_w
        if value is None:
            raise RuntimeError("Box motion object quaternion was not materialized")
        return value

    @property
    def object_state_b(self) -> np.ndarray:
        return self._object_obs_b

    def _refresh_motion(self, env_ids: np.ndarray | None = None) -> None:
        super()._refresh_motion(env_ids)
        value = self.box_motion.object_pos_w
        if value is None:
            raise RuntimeError("Box motion object position was not materialized")
        if env_ids is None:
            np.add(value, self._env.scene.env_origins, out=self._object_pos_w)
        else:
            self._object_pos_w[env_ids] = value[env_ids] + self._env.scene.env_origins[env_ids]

    def _resample_command(self, env_ids: np.ndarray) -> None:
        super()._resample_command(env_ids)
        motion = cast(BoxMotionData, self.motion.get_motion_at_frame(self.time_steps[env_ids]))
        values = (
            motion.object_pos_w,
            motion.object_quat_w,
            motion.object_lin_vel_w,
            motion.object_ang_vel_w,
        )
        if any(value is None for value in values):
            raise RuntimeError("Box motion reset requires complete object state")
        object_pos = cast(np.ndarray, motion.object_pos_w).copy()
        object_pos += self._env.scene.env_origins[env_ids]
        object_state = np.concatenate(
            (
                object_pos,
                cast(np.ndarray, motion.object_quat_w),
                cast(np.ndarray, motion.object_lin_vel_w),
                cast(np.ndarray, motion.object_ang_vel_w),
            ),
            axis=-1,
        )
        self.object.write_root_state_to_sim(object_state, env_ids=env_ids)

    def _refresh_object_state(self, env_ids: np.ndarray | None = None) -> None:
        rows = self._all_env_ids if env_ids is None else env_ids
        update_object_relative_state_kernel(
            rows,
            self.robot_anchor_pos_w,
            self.robot_anchor_quat_w,
            self.object.data.root_link_pos_w,
            self.object.data.root_link_quat_w,
            self.object.data.root_link_lin_vel_w,
            self._object_obs_b,
        )

    def post_compute(self) -> None:
        super().post_compute()
        self._refresh_object_state(self._post_compute_env_ids)


def _box_command(env: ManagerBasedRlEnv, command_name: str) -> BoxMotionCommand:
    try:
        command = env.command_manager.get_term(command_name)
    except KeyError as exc:
        raise KeyError(f"Box motion command term '{command_name}' not found") from exc
    if not isinstance(command, BoxMotionCommand):
        raise TypeError(
            f"Command term '{command_name}' is {type(command).__name__}, expected BoxMotionCommand"
        )
    return command


def object_state_b(env: ManagerBasedRlEnv, command_name: str) -> np.ndarray:
    return _box_command(env, command_name).object_state_b


def object_global_position_error_exp(
    env: ManagerBasedRlEnv,
    command_name: str,
    std: float,
) -> np.ndarray:
    command = _box_command(env, command_name)
    if not np.isfinite(std) or std <= 0.0:
        raise ValueError("object position std must be finite and positive")
    error = np.sum(
        np.square(command.object_pos_w - command.object.data.root_link_pos_w),
        axis=-1,
    )
    return np.exp(-error / float(std) ** 2)


def object_global_orientation_error_exp(
    env: ManagerBasedRlEnv,
    command_name: str,
    std: float,
) -> np.ndarray:
    command = _box_command(env, command_name)
    if not np.isfinite(std) or std <= 0.0:
        raise ValueError("object orientation std must be finite and positive")
    error = np_quat_error_magnitude_squared_batched(
        command.object_quat_w,
        command.object.data.root_link_quat_w,
    )
    return np.exp(-error / float(std) ** 2)


def bad_object_position(
    env: ManagerBasedRlEnv,
    command_name: str,
    threshold: float,
) -> np.ndarray:
    command = _box_command(env, command_name)
    error = np.linalg.norm(command.object_pos_w - command.object.data.root_link_pos_w, axis=-1)
    return error > threshold


def bad_object_orientation(
    env: ManagerBasedRlEnv,
    command_name: str,
    threshold: float,
) -> np.ndarray:
    command = _box_command(env, command_name)
    error = np.sqrt(
        np_quat_error_magnitude_squared_batched(
            command.object_quat_w,
            command.object.data.root_link_quat_w,
        )
    )
    return error > threshold


class randomize_encoder_bias(ManagerTermBase):
    """Per-reset encoder calibration error through the public Entity data surface."""

    def __init__(self, cfg: ManagerTermBaseCfg, env: ManagerBasedRlEnv):
        super().__init__(env)
        asset_cfg = cfg.params.get("asset_cfg", _DEFAULT_ASSET_CFG)
        if not isinstance(asset_cfg, SceneEntityCfg):
            raise TypeError("randomize_encoder_bias asset_cfg must be SceneEntityCfg")
        bias_range = np.asarray(cfg.params.get("bias_range"), dtype=np.float64)
        if bias_range.shape != (2,) or not np.isfinite(bias_range).all():
            raise ValueError("randomize_encoder_bias bias_range must be a finite pair")
        if bias_range[0] > bias_range[1]:
            raise ValueError("randomize_encoder_bias minimum exceeds maximum")
        self._range = (float(bias_range[0]), float(bias_range[1]))
        self._entity = cast("Entity", env.scene[asset_cfg.name])
        raw_ids = np.arange(self._entity.num_joints)[asset_cfg.joint_ids]
        self._joint_ids = np.asarray(raw_ids, dtype=np.intp)

    def __call__(
        self,
        env: ManagerBasedRlEnv,
        env_ids: np.ndarray | None,
        bias_range: tuple[float, float],
        asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    ) -> None:
        del bias_range, asset_cfg
        ids = np.arange(env.num_envs, dtype=np.int32) if env_ids is None else env_ids
        samples = env.rng.uniform(
            *self._range,
            size=(len(ids), len(self._joint_ids)),
        )
        self._entity.data.encoder_bias[np.ix_(ids, self._joint_ids)] = samples


class joint_acc_l2(ManagerTermBase):
    """Squared finite-difference joint acceleration with reset-aware state."""

    def __init__(self, cfg: ManagerTermBaseCfg, env: ManagerBasedRlEnv):
        super().__init__(env)
        asset_cfg = cfg.params.get("asset_cfg", _DEFAULT_ASSET_CFG)
        if not isinstance(asset_cfg, SceneEntityCfg):
            raise TypeError("joint_acc_l2 asset_cfg must be SceneEntityCfg")
        self._entity = cast("Entity", env.scene[asset_cfg.name])
        self._joint_ids = np.arange(self._entity.num_joints, dtype=np.intp)[asset_cfg.joint_ids]
        self._previous = self._entity.data.joint_vel[:, self._joint_ids].copy()

    def reset(self, env_ids: np.ndarray | slice | None) -> None:
        ids = np.arange(self.num_envs, dtype=np.intp)
        if env_ids is not None:
            ids = ids[env_ids]
        self._previous[ids] = self._entity.data.joint_vel[np.ix_(ids, self._joint_ids)]

    def __call__(
        self,
        env: ManagerBasedRlEnv,
        asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    ) -> np.ndarray:
        del asset_cfg
        velocity = self._entity.data.joint_vel[:, self._joint_ids]
        acceleration = (velocity - self._previous) / env.step_dt
        self._previous[:] = velocity
        return np.sum(np.square(acceleration), axis=-1)


class joint_torque_l2(ManagerTermBase):
    """Position-controller torque estimate using cold-path actuator gain binding."""

    def __init__(self, cfg: ManagerTermBaseCfg, env: ManagerBasedRlEnv):
        super().__init__(env)
        asset_cfg = cfg.params.get("asset_cfg", _DEFAULT_ASSET_CFG)
        if not isinstance(asset_cfg, SceneEntityCfg):
            raise TypeError("joint_torque_l2 asset_cfg must be SceneEntityCfg")
        action_name = cfg.params.get("action_name", "joint_pos")
        if not isinstance(action_name, str) or not action_name:
            raise ValueError("joint_torque_l2 action_name must be non-empty")
        action = env.action_manager.get_term(action_name)
        if not isinstance(action, MotionJointPositionAction):
            raise TypeError("joint_torque_l2 requires MotionJointPositionAction")
        self._action = action
        self._entity = cast("Entity", env.scene[asset_cfg.name])
        actuator_ids, kp, kd = self._entity.bind_actuator_gain_write(
            asset_cfg.actuator_ids,
            term_name="joint_torque_l2",
        )
        selected_names = tuple(self._entity.actuator_names[int(index)] for index in actuator_ids)
        if selected_names != tuple(action.target_names):
            raise ValueError(
                "joint_torque_l2 actuator order does not match the action target order: "
                f"{selected_names} != {tuple(action.target_names)}"
            )
        self._kp = kp
        self._kd = kd
        self._joint_ids = action.target_ids
        self._torque = np.empty_like(action.target)

    def __call__(
        self,
        env: ManagerBasedRlEnv,
        action_name: str = "joint_pos",
        asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    ) -> np.ndarray:
        del env, action_name, asset_cfg
        np.subtract(
            self._action.target,
            self._entity.data.joint_pos[:, self._joint_ids],
            out=self._torque,
        )
        self._torque *= self._kp
        self._torque -= self._kd * self._entity.data.joint_vel[:, self._joint_ids]
        return np.sum(np.square(self._torque), axis=-1)


__all__ = [
    "BoxMotionCommand",
    "BoxMotionCommandCfg",
    "bad_object_orientation",
    "bad_object_position",
    "joint_acc_l2",
    "joint_torque_l2",
    "object_global_orientation_error_exp",
    "object_global_position_error_exp",
    "object_state_b",
    "randomize_encoder_bias",
]
