"""Exact A2Arm actor/critic frame assembly and history semantics."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar, cast

import numpy as np

from unilab.managers import ManagerTermBase, ObservationTermCfg
from unilab.tasks.locomotion.a2arm.actions import A2ArmPdAction
from unilab.utils.rotation import np_quat_apply, np_quat_apply_inverse, np_yaw_quat

from .constants import (
    ACTOR_HISTORY,
    ACTOR_STEP_DIM,
    CRITIC_HISTORY,
    CRITIC_STEP_DIM,
    NUM_ACTIONS,
    NUM_LEG,
)
from .state import A2ArmPosForceState, cart2sphere

if TYPE_CHECKING:
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


def _add_actor_noise(actor_core: np.ndarray, rng: Any, noise_level: float) -> np.ndarray:
    """Add the legacy actor corruption after applying per-field observation scales."""
    noise = np.zeros_like(actor_core)
    num_envs = actor_core.shape[0]
    noise[:, 0:2] = rng.uniform(-0.05, 0.05, size=(num_envs, 2))
    noise[:, 2:5] = rng.uniform(-0.2, 0.2, size=(num_envs, 3))
    noise[:, 5 : 5 + NUM_ACTIONS] = rng.uniform(-0.01, 0.01, size=(num_envs, NUM_ACTIONS))
    noise[:, 5 + NUM_ACTIONS : 5 + 2 * NUM_ACTIONS] = rng.uniform(
        -0.075, 0.075, size=(num_envs, NUM_ACTIONS)
    )
    # UniFP injected gyro noise before multiplying angular velocity by 0.25.
    noise[:, 2:5] *= 0.25
    actor_core += noise.astype(np.float32) * float(noise_level)
    return actor_core


class A2ArmActorHistoryCfg(ObservationTermCfg):
    history_role: ClassVar[str] = "actor"
    func: Any = None


class A2ArmCriticHistoryCfg(ObservationTermCfg):
    history_role: ClassVar[str] = "critic"
    func: Any = None


class _HistoryTerm(ManagerTermBase):
    supports_row_scoped_reset = True

    def __init__(self, cfg: ObservationTermCfg, env: ManagerBasedRlEnv):
        super().__init__(env)
        self._state: A2ArmPosForceState = env.command_manager.get_term("task_state")
        self._action = cast(A2ArmPdAction, env.action_manager.get_term("joint_pd"))
        self._history_role = str(getattr(cfg, "history_role", "actor"))
        if self._history_role not in {"actor", "critic"}:
            raise ValueError(
                f"A2Arm history_role must be 'actor' or 'critic', got {self._history_role!r}"
            )
        default_history = ACTOR_HISTORY if self._history_role == "actor" else CRITIC_HISTORY
        self._history_length = int(cfg.params.get("history_length", default_history))
        if self._history_length <= 0:
            raise ValueError(
                f"A2Arm {self._history_role} history_length must be positive, "
                f"got {self._history_length}"
            )
        step_dim = ACTOR_STEP_DIM if self._history_role == "actor" else CRITIC_STEP_DIM
        self._history = np.zeros(
            (env.num_envs, self._history_length, step_dim),
            dtype=np.float32,
        )
        self._clip = float(cfg.params.get("clip", 100.0))
        self._noise_level = float(cfg.params.get("noise_level", 1.0))
        self._actor_noise = bool(cfg.params.get("actor_noise", True))
        # ObservationManager calls each term once during construction to infer
        # its shape.  That probe must not advance task history or consume the
        # task RNG; the first real frame is produced during reset.
        self._skip_shape_probe = True

    def reset(self, env_ids: np.ndarray | slice | None = None) -> None:
        ids = slice(None) if env_ids is None else env_ids
        self._history[ids] = 0.0

    def _frame(self, env_ids: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
        ids: np.ndarray | slice = (
            slice(None) if env_ids is None else np.asarray(env_ids, dtype=np.intp)
        )
        robot = self._env.scene["robot"]
        quat = robot.data.root_link_quat_w[ids]
        roll_pitch = _roll_pitch(quat)
        ang_vel = robot.data.root_link_ang_vel_b[ids]
        dof_pos = robot.data.joint_pos[ids]
        dof_vel = robot.data.joint_vel[ids]
        default = robot.data.default_joint_pos[ids]
        command = self._state.command[ids].copy()
        command_scale = np.asarray(
            [2.0, 2.0, 0.25, 0.5, 1.0, 1.3, 1.0, 1.0, 1.0, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01],
            dtype=np.float32,
        )
        scaled_command = command * command_scale
        phase = self._state.gait_phase[ids, :1]
        actor_core = np.concatenate(
            [
                roll_pitch,
                ang_vel * 0.25,
                dof_pos - default,
                dof_vel * 0.05,
                self._action.raw_action[ids],
                np.sin(2.0 * np.pi * phase),
                np.cos(2.0 * np.pi * phase),
                scaled_command,
            ],
            axis=1,
        ).astype(np.float32)
        if actor_core.shape[1] != ACTOR_STEP_DIM:
            raise RuntimeError(f"A2Arm actor frame drifted to {actor_core.shape[1]}")
        # The critic carries the same uncorrupted actor frame as its trailing
        # block. Copy it before applying actor-only observation noise so the two
        # layouts cannot drift independently.
        critic_core = actor_core.copy()
        if self._actor_noise:
            if env_ids is None:
                _add_actor_noise(actor_core, self._env.rng, self._noise_level)
            else:
                # The legacy raw-observation path drew noise for the complete
                # batch even while rebuilding a partial reset. Preserve that
                # RNG stream, then scatter only the selected rows into the
                # row-scoped history term.
                full_noise = np.zeros((self._env.num_envs, ACTOR_STEP_DIM), dtype=np.float32)
                _add_actor_noise(full_noise, self._env.rng, self._noise_level)
                actor_core += full_noise[np.asarray(env_ids, dtype=np.intp)]

        ee_world = self._state.ee_world_pos()[ids]
        center = self._state.goal_center_world()[ids]
        ee_local = np_quat_apply_inverse(np_yaw_quat(quat), ee_world - center)
        ee_sphere = cart2sphere(ee_local) * np.asarray([0.5, 1.0, 1.3], dtype=np.float32)
        force_ee = np_quat_apply_inverse(np_yaw_quat(quat), self._state.force_ee_world[ids]) * 0.01
        force_base = (
            np_quat_apply_inverse(np_yaw_quat(quat), self._state.force_base_world[ids]) * 0.01
        )
        cse = np.concatenate(
            [robot.data.root_link_lin_vel_b[ids] * 2.0, ee_sphere, force_ee, force_base], axis=1
        )
        base_yaw = np_yaw_quat(quat)
        force_cmd_world = np_quat_apply(base_yaw, self._state.force_ee_command[ids])
        goal_offset_world = (
            self._state.current_goal_world[ids]
            + (self._state.force_ee_world[ids] + force_cmd_world) / self._state.cfg.gripper_force_kp
        )
        goal_offset_local = np_quat_apply_inverse(base_yaw, goal_offset_world - center)
        goal_offset = cart2sphere(goal_offset_local) * np.asarray([0.5, 1.0, 1.3], dtype=np.float32)
        dr_block = np.concatenate(
            [
                self._state.dr_friction[ids],
                self._state.dr_base_mass[ids],
                self._state.dr_base_com[ids],
                self._state.dr_gripper_mass[ids],
            ],
            axis=1,
        )
        critic = np.concatenate(
            [
                cse,
                dof_pos[:, :NUM_LEG] - self._state.reference_dof_pos[ids],
                dr_block,
                self._action.motor_strength[ids] - 1.0,
                self._state.stance_mask[ids],
                self._state.foot_contact[ids].astype(np.float32),
                robot.data.projected_gravity_b[ids],
                goal_offset,
                critic_core,
            ],
            axis=1,
        ).astype(np.float32)
        if critic.shape[1] != CRITIC_STEP_DIM:
            raise RuntimeError(f"A2Arm critic frame drifted to {critic.shape[1]}")
        return actor_core, critic

    def __call__(
        self,
        env: ManagerBasedRlEnv,
        *,
        env_ids: np.ndarray | None = None,
        **params: Any,
    ) -> np.ndarray:
        del env, params
        if self._skip_shape_probe:
            self._skip_shape_probe = False
            return np.zeros(
                (self._env.num_envs, self._history.shape[1] * self._history.shape[2]),
                dtype=np.float32,
            )
        actor, critic = self._frame(env_ids)
        frame = actor if self._history_role == "actor" else critic
        if env_ids is None:
            if self._history_role == "actor":
                self._history[:, :-1] = self._history[:, 1:]
                self._history[:, -1] = frame
            else:
                self._history[:, 1:] = self._history[:, :-1]
                self._history[:, 0] = frame
        else:
            ids = np.asarray(env_ids, dtype=np.intp)
            if self._history_role == "actor":
                self._history[ids, :-1] = self._history[ids, 1:]
                self._history[ids, -1] = frame
            else:
                self._history[ids, 1:] = self._history[ids, :-1]
                self._history[ids, 0] = frame
        return np.clip(self._history.reshape(self._env.num_envs, -1), -self._clip, self._clip)


class A2ArmActorHistory(_HistoryTerm):
    pass


class A2ArmCriticHistory(_HistoryTerm):
    pass


__all__ = [
    "A2ArmActorHistory",
    "A2ArmActorHistoryCfg",
    "A2ArmCriticHistory",
    "A2ArmCriticHistoryCfg",
]
