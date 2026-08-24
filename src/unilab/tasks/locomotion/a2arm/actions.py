"""A2Arm per-substep Python motor-PD action term."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from unilab.managers import ActionTerm, ActionTermCfg

from .constants import ACTUATOR_NAMES, NUM_ACTIONS

if TYPE_CHECKING:
    from unilab.managers._types import ManagerBasedRlEnv


@dataclass(kw_only=True)
class A2ArmPdActionCfg(ActionTermCfg):
    actuator_names: tuple[str, ...] = ACTUATOR_NAMES
    action_scale: tuple[float, ...] = (0.25,) * 12 + (0.25,) * 5
    kp: tuple[float, ...] = (100.0, 100.0, 150.0) * 4 + (90.0, 120.0, 70.0, 30.0, 30.0)
    kd: tuple[float, ...] = (4.0, 4.0, 6.0) * 4 + (5.5, 10.5, 5.5, 1.0, 1.0)
    torque_limits: tuple[float, ...] = (120.0, 120.0, 180.0) * 4 + (30.0, 30.0, 30.0, 10.0, 10.0)
    motor_strength: tuple[float, ...] = (1.0,) * NUM_ACTIONS
    randomize_motor_strength: bool = True
    leg_motor_strength_range: tuple[float, float] = (0.85, 1.15)
    arm_motor_strength_range: tuple[float, float] = (0.85, 1.15)
    clip_actions: float = 100.0
    action_delay_steps: int = 0
    simulate_action_latency: bool = False

    def build(self, env: ManagerBasedRlEnv) -> A2ArmPdAction:
        return A2ArmPdAction(self, env)


class A2ArmPdAction(ActionTerm):
    cfg: A2ArmPdActionCfg

    def __init__(self, cfg: A2ArmPdActionCfg, env: ManagerBasedRlEnv):
        if len(cfg.actuator_names) != NUM_ACTIONS:
            raise ValueError(f"A2ArmPdAction requires {NUM_ACTIONS} actuator names")
        super().__init__(cfg, env)
        actuator_ids, names = self._entity.find_actuators(cfg.actuator_names, preserve_order=True)
        if tuple(names) != tuple(cfg.actuator_names):
            raise ValueError(f"A2ArmPdAction actuator order mismatch: {names}")
        joint_names = tuple(
            name + "_joint" if name.startswith(("FL_", "FR_", "RL_", "RR_")) else name
            for name in cfg.actuator_names
        )
        joint_ids, _ = self._entity.find_joints(joint_names, preserve_order=True)
        self._actuator_ids = np.asarray(actuator_ids, dtype=np.int32)
        self._joint_ids = np.asarray(joint_ids, dtype=np.int32)
        self._scale = np.asarray(cfg.action_scale, dtype=np.float32)
        self._kp = np.asarray(cfg.kp, dtype=np.float32)
        self._kd = np.asarray(cfg.kd, dtype=np.float32)
        self._limits = np.asarray(cfg.torque_limits, dtype=np.float32)
        self._motor_strength = np.broadcast_to(
            np.asarray(cfg.motor_strength, dtype=np.float32), (self.num_envs, NUM_ACTIONS)
        ).copy()
        self._raw_action = np.zeros((self.num_envs, NUM_ACTIONS), dtype=np.float32)
        self._previous_raw_action = np.zeros_like(self._raw_action)
        self._processed_action = np.zeros_like(self._raw_action)
        self._applied_torque = np.zeros_like(self._raw_action)
        self._delay = max(0, int(cfg.action_delay_steps))
        self._history = np.zeros((self.num_envs, self._delay + 1, NUM_ACTIONS), dtype=np.float32)
        self._state = env.command_manager.get_term("task_state")

    @property
    def action_dim(self) -> int:
        return NUM_ACTIONS

    @property
    def raw_action(self) -> np.ndarray:
        return self._raw_action

    @property
    def previous_raw_action(self) -> np.ndarray:
        return self._previous_raw_action

    @property
    def processed_action(self) -> np.ndarray:
        return self._processed_action

    @property
    def applied_torque(self) -> np.ndarray:
        return self._applied_torque

    @property
    def motor_strength(self) -> np.ndarray:
        return self._motor_strength

    @property
    def torque_limits(self) -> np.ndarray:
        """Per-actuator absolute torque limits used by reward terms."""
        return self._limits

    def set_motor_strength(self, env_ids: np.ndarray, values: np.ndarray) -> None:
        ids = np.asarray(env_ids, dtype=np.intp)
        values = np.asarray(values, dtype=np.float32)
        if values.shape != (len(ids), NUM_ACTIONS):
            raise ValueError(
                f"A2ArmPdAction motor strength expected {(len(ids), NUM_ACTIONS)}, got {values.shape}"
            )
        self._motor_strength[ids] = values

    def process_actions(self, actions: np.ndarray) -> None:
        if actions.shape != self._raw_action.shape:
            raise ValueError(
                f"A2ArmPdAction expected {self._raw_action.shape}, got {actions.shape}"
            )
        self._previous_raw_action[:] = self._raw_action
        np.clip(
            actions,
            -float(self.cfg.clip_actions),
            float(self.cfg.clip_actions),
            out=self._raw_action,
        )
        if self._delay:
            self._history[:, :-1] = self._history[:, 1:]
            self._history[:, -1] = self._raw_action
            executed = self._history[:, 0]
        elif self.cfg.simulate_action_latency:
            executed = self._previous_raw_action
        else:
            executed = self._raw_action
        self._state.prepare_control_step(int(self._env.step_counter))
        self._processed_action[:] = self._entity.data.default_joint_pos * 0.0
        self._processed_action[:] = (
            self._entity.data.default_joint_pos + executed * self._scale * self._motor_strength
        )

    def apply_actions(self) -> None:
        q = self._entity.data.joint_pos
        qd = self._entity.data.joint_vel
        torque = self._kp * (self._processed_action - q) - self._kd * qd
        np.clip(torque, -self._limits, self._limits, out=self._applied_torque)
        self._entity.data.write_ctrl(self._applied_torque, actuator_ids=self._actuator_ids)

    def reset(self, env_ids: np.ndarray | slice | None = None) -> None:
        ids = slice(None) if env_ids is None else env_ids
        self._raw_action[ids] = 0.0
        self._previous_raw_action[ids] = 0.0
        self._processed_action[ids] = 0.0
        self._applied_torque[ids] = 0.0
        self._history[ids] = 0.0
        if self.cfg.randomize_motor_strength:
            count = len(np.arange(self.num_envs)[ids]) if isinstance(ids, slice) else len(ids)
            reset_ids = (
                np.arange(self.num_envs, dtype=np.int32)[ids]
                if isinstance(ids, slice)
                else np.asarray(ids, dtype=np.int32)
            )
            leg = self._env.rng.uniform(*self.cfg.leg_motor_strength_range, size=(count, 12))
            arm = self._env.rng.uniform(*self.cfg.arm_motor_strength_range, size=(count, 5))
            self._motor_strength[reset_ids] = np.concatenate([leg, arm], axis=1).astype(np.float32)


__all__ = ["A2ArmPdAction", "A2ArmPdActionCfg"]
