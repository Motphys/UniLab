"""Exact A2Arm actor/critic frame assembly and history semantics."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from unilab.managers import ManagerTermBase, ObservationTermCfg
from unilab.utils.rotation import np_quat_apply, np_quat_apply_inverse, np_yaw_quat

from .constants import (
    ACTOR_HISTORY,
    ACTOR_STEP_DIM,
    CRITIC_HISTORY,
    CRITIC_STEP_DIM,
    NUM_ACTIONS,
    NUM_COMMANDS,
    NUM_LEG,
)
from .state import A2ArmPosForceState, cart2sphere

if TYPE_CHECKING:
    from unilab.managers import ObservationTermCfg
    from unilab.managers._types import ManagerBasedRlEnv


def _roll_pitch(quat: np.ndarray) -> np.ndarray:
    w, x, y, z = np.moveaxis(quat, -1, 0)
    return np.stack(
        [
            np.arctan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y)),
            np.arcsin(np.clip(2.0 * (w * y - z * x), -1.0, 1.0)),
        ],
        axis=-1,
    )


class A2ArmActorHistoryCfg(ObservationTermCfg):
    func: Any = None


class A2ArmCriticHistoryCfg(ObservationTermCfg):
    func: Any = None


class _HistoryTerm(ManagerTermBase):
    def __init__(self, cfg: ObservationTermCfg, env: ManagerBasedRlEnv):
        super().__init__(env)
        self._state: A2ArmPosForceState = env.command_manager.get_term("task_state")
        self._action = env.action_manager.get_term("joint_pd")
        self._actor_history = np.zeros(
            (env.num_envs, ACTOR_HISTORY, ACTOR_STEP_DIM), dtype=np.float32
        )
        self._critic_history = np.zeros(
            (env.num_envs, CRITIC_HISTORY, CRITIC_STEP_DIM), dtype=np.float32
        )
        self._clip = float(cfg.params.get("clip", 100.0))
        self._noise_level = float(cfg.params.get("noise_level", 1.0))
        self._actor_noise = bool(cfg.params.get("actor_noise", True))

    def reset(self, env_ids: np.ndarray | slice | None = None) -> None:
        ids = slice(None) if env_ids is None else env_ids
        self._actor_history[ids] = 0.0
        self._critic_history[ids] = 0.0

    def _frame(self) -> tuple[np.ndarray, np.ndarray]:
        self._state.prepare_transition(int(self._env.common_step_counter))
        robot = self._env.scene["robot"]
        quat = robot.data.root_link_quat_w
        roll_pitch = _roll_pitch(quat)
        ang_vel = robot.data.root_link_ang_vel_b
        dof_pos = robot.data.joint_pos
        dof_vel = robot.data.joint_vel
        default = robot.data.default_joint_pos
        command = self._state.command.copy()
        command_scale = np.asarray(
            [2.0, 2.0, 0.25, 0.5, 1.0, 1.3, 1.0, 1.0, 1.0, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01],
            dtype=np.float32,
        )
        scaled_command = command * command_scale
        phase = self._state.gait_phase[:, :1]
        actor_core = np.concatenate(
            [
                roll_pitch,
                ang_vel * 0.25,
                dof_pos - default,
                dof_vel * 0.05,
                self._action.raw_action,
                np.sin(2.0 * np.pi * phase),
                np.cos(2.0 * np.pi * phase),
                scaled_command,
            ],
            axis=1,
        ).astype(np.float32)
        if actor_core.shape[1] != ACTOR_STEP_DIM:
            raise RuntimeError(f"A2Arm actor frame drifted to {actor_core.shape[1]}")
        if self._actor_noise:
            noise = np.zeros_like(actor_core)
            noise[:, 0:2] = self._env.rng.uniform(-0.05, 0.05, size=(self._env.num_envs, 2))
            noise[:, 2:5] = self._env.rng.uniform(-0.2, 0.2, size=(self._env.num_envs, 3))
            noise[:, 5 : 5 + NUM_ACTIONS] = self._env.rng.uniform(
                -0.01, 0.01, size=(self._env.num_envs, NUM_ACTIONS)
            )
            noise[:, 5 + NUM_ACTIONS : 5 + 2 * NUM_ACTIONS] = self._env.rng.uniform(
                -0.075, 0.075, size=(self._env.num_envs, NUM_ACTIONS)
            )
            actor_core += noise.astype(np.float32) * self._noise_level

        sensor = self._state._sensor_view.read()
        endpoint_pos = sensor[:, 0:3]
        arm_base_pos = sensor[:, 7:10]
        arm_base_quat = sensor[:, 10:14]
        ee_world = arm_base_pos + np_quat_apply(arm_base_quat, endpoint_pos)
        center = self._state.goal_center_world()
        ee_local = np_quat_apply_inverse(np_yaw_quat(quat), ee_world - center)
        ee_sphere = cart2sphere(ee_local) * np.asarray([0.5, 1.0, 1.3], dtype=np.float32)
        force_ee = np_quat_apply_inverse(np_yaw_quat(quat), self._state.force_ee_world) * 0.01
        force_base = np_quat_apply_inverse(np_yaw_quat(quat), self._state.force_base_world) * 0.01
        cse = np.concatenate(
            [robot.data.root_link_lin_vel_b * 2.0, ee_sphere, force_ee, force_base], axis=1
        )
        base_yaw = np_yaw_quat(quat)
        force_cmd_world = np_quat_apply(base_yaw, self._state.force_ee_command)
        goal_offset_world = (
            self._state.current_goal_world
            + (self._state.force_ee_world + force_cmd_world) / self._state.cfg.gripper_force_kp
        )
        goal_offset_local = np_quat_apply_inverse(base_yaw, goal_offset_world - center)
        goal_offset = cart2sphere(goal_offset_local) * np.asarray([0.5, 1.0, 1.3], dtype=np.float32)
        dr_block = np.concatenate(
            [
                self._state.dr_friction,
                self._state.dr_base_mass,
                self._state.dr_base_com,
                self._state.dr_gripper_mass,
            ],
            axis=1,
        )
        critic_core = np.concatenate(
            [
                roll_pitch,
                ang_vel * 0.25,
                dof_pos - default,
                dof_vel * 0.05,
                self._action.raw_action,
                np.sin(2.0 * np.pi * phase),
                np.cos(2.0 * np.pi * phase),
                scaled_command,
            ],
            axis=1,
        )
        critic = np.concatenate(
            [
                cse,
                dof_pos[:, :NUM_LEG] - self._state.reference_dof_pos,
                dr_block,
                self._action.motor_strength - 1.0,
                self._state.stance_mask,
                self._state.foot_contact.astype(np.float32),
                robot.data.projected_gravity_b,
                goal_offset,
                critic_core,
            ],
            axis=1,
        ).astype(np.float32)
        if critic.shape[1] != CRITIC_STEP_DIM:
            raise RuntimeError(f"A2Arm critic frame drifted to {critic.shape[1]}")
        return actor_core, critic

    def __call__(self, env: ManagerBasedRlEnv, **params: Any) -> np.ndarray:
        del env, params
        actor, critic = self._frame()
        self._actor_history[:, :-1] = self._actor_history[:, 1:]
        self._actor_history[:, -1] = actor
        self._critic_history[:, 1:] = self._critic_history[:, :-1]
        self._critic_history[:, 0] = critic
        if isinstance(self, A2ArmActorHistory):
            return np.clip(
                self._actor_history.reshape(self._env.num_envs, -1), -self._clip, self._clip
            )
        return np.clip(
            self._critic_history.reshape(self._env.num_envs, -1), -self._clip, self._clip
        )


class A2ArmActorHistory(_HistoryTerm):
    pass


class A2ArmCriticHistory(_HistoryTerm):
    pass


__all__ = [
    "A2ArmActorHistory",
    "A2ArmCriticHistory",
    "A2ArmActorHistoryCfg",
    "A2ArmCriticHistoryCfg",
]
