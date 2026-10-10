"""G1-specific Manager-Based motion-tracking terms."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import numpy as np
import torch

from unilab.managers import ManagerTermBase, ManagerTermBaseCfg
from unilab.managers.scene_entity_config import SceneEntityCfg
from unilab.tasks.motion_tracking.common.manager_terms import MotionJointPositionAction

if TYPE_CHECKING:
    from unilab.base.entity import Entity
    from unilab.managers._types import ManagerBasedRlEnv


_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


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

    def reset(self, env_ids: torch.Tensor | np.ndarray | slice | None) -> None:
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
        self._torque = torch.empty_like(action.tensor_target)
        self._target_index = torch.from_numpy(action.target_ids.copy()).to(
            device=action.tensor_target.device, dtype=torch.int64
        )
        self._kp_tensor = torch.as_tensor(
            kp, device=action.tensor_target.device, dtype=torch.float32
        )
        self._kd_tensor = torch.as_tensor(
            kd, device=action.tensor_target.device, dtype=torch.float32
        )

    def __call__(
        self,
        env: ManagerBasedRlEnv,
        action_name: str = "joint_pos",
        asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    ) -> torch.Tensor:
        del env, action_name, asset_cfg
        read_plan = getattr(self._env.scene, "_tensor_read_plan", None)
        if read_plan is not None and read_plan.ready:
            joint_view = read_plan.joint_tensor_view(self._entity)
        else:
            joint_view = self._entity.joint_tensor_view(self._action.tensor_target.device)
        joint_pos = joint_view.joint_pos.index_select(1, self._target_index)
        joint_vel = joint_view.joint_vel.index_select(1, self._target_index)
        torch.sub(
            self._action.tensor_target,
            joint_pos,
            out=self._torque,
        )
        self._torque *= self._kp_tensor
        self._torque.sub_(self._kd_tensor * joint_vel)
        return torch.sum(torch.square(self._torque), dim=-1)


__all__ = [
    "joint_acc_l2",
    "joint_torque_l2",
]
