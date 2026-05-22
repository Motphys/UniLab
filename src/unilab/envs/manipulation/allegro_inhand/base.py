from __future__ import annotations

from dataclasses import dataclass, field

import gymnasium as gym
import numpy as np

from unilab.base.backend import SimBackend
from unilab.base.base import EnvCfg
from unilab.base.np_env import NpEnv, NpEnvState
from unilab.dtype_config import get_global_dtype


@dataclass
class NoiseConfig:
    level: float = 1.0
    scale_joint_angle: float = 0.02


@dataclass
class ControlConfig:
    action_scale: float = 1.0 / 24.0
    kp: float = 1.0
    kd: float = 0.1


@dataclass
class AllegroBaseCfg(EnvCfg):
    sim_dt: float = 0.005
    ctrl_dt: float = 0.05
    noise_config: NoiseConfig = field(default_factory=NoiseConfig)
    control_config: ControlConfig = field(default_factory=ControlConfig)


class AllegroBaseEnv(NpEnv):
    _NUM_HAND_DOF: int = 16
    _FINGERTIP_BODY_NAMES: tuple[str, ...] = ("ff_tip", "mf_tip", "rf_tip", "th_tip")
    _cfg: AllegroBaseCfg
    _init_qpos: np.ndarray
    _init_qvel: np.ndarray

    def __init__(self, cfg: AllegroBaseCfg, backend: SimBackend, num_envs: int = 1):
        super().__init__(cfg, backend, num_envs)

        self._np_dtype = get_global_dtype()
        actuator_range = np.asarray(self._backend.get_actuator_ctrl_range(), dtype=self._np_dtype)
        if actuator_range.shape[0] < self._NUM_HAND_DOF:
            raise ValueError(
                f"Model has {actuator_range.shape[0]} actuators, expected at least {self._NUM_HAND_DOF}"
            )
        self._ctrl_lower = np.asarray(actuator_range[: self._NUM_HAND_DOF, 0], dtype=self._np_dtype)
        self._ctrl_upper = np.asarray(actuator_range[: self._NUM_HAND_DOF, 1], dtype=self._np_dtype)

        self._init_action_space()
        self._num_action = self._action_space.shape[0]
        if self._num_action != self._NUM_HAND_DOF:
            raise ValueError(f"Expected {self._NUM_HAND_DOF} actuators, got {self._num_action}")

        self._init_buffers()
        self.nq = int(self._init_qpos.shape[0])
        self.nv = int(self._init_qvel.shape[0])

        self._ball_body_ids = self._backend.get_body_ids(["ball"])
        self._fingertip_body_ids = self._backend.get_body_ids(self._FINGERTIP_BODY_NAMES)

    def _init_action_space(self) -> None:
        self._action_space = gym.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self._NUM_HAND_DOF,),
            dtype=np.float32,
        )

    @property
    def action_space(self) -> gym.spaces.Box:
        return self._action_space  # type: ignore[no-any-return]

    def _init_buffers(self) -> None:
        self.default_angles = np.zeros((self._num_action,), dtype=self._np_dtype)
        self._init_qpos = self._resolve_init_qpos()
        self.default_angles = np.asarray(
            self._init_qpos[: self._NUM_HAND_DOF], dtype=self._np_dtype
        )
        self._init_qvel = np.asarray(self._backend.get_init_qvel(), dtype=self._np_dtype)

    def _resolve_init_qpos(self) -> np.ndarray:
        for key_name in ("home", "stand", "default"):
            try:
                return np.asarray(self._backend.get_keyframe_qpos(key_name), dtype=self._np_dtype)
            except Exception:
                continue
        raise ValueError("Could not resolve initial qpos from backend keyframes")

    def apply_action(self, actions: np.ndarray, state: NpEnvState) -> np.ndarray:
        clipped_actions = np.asarray(np.clip(actions, -1.0, 1.0), dtype=self._np_dtype)
        state.info["last_actions"] = state.info.get(
            "current_actions", np.zeros_like(clipped_actions)
        )
        state.info["current_actions"] = clipped_actions

        prev_ctrl = state.info.get(
            "prev_ctrl",
            np.broadcast_to(
                self.default_angles, (clipped_actions.shape[0], self._num_action)
            ).copy(),
        )
        new_ctrl = prev_ctrl + self._cfg.control_config.action_scale * clipped_actions
        new_ctrl = np.clip(new_ctrl, self._ctrl_lower, self._ctrl_upper)
        prev_ctrl = np.asarray(new_ctrl, dtype=self._np_dtype)
        state.info["prev_ctrl"] = prev_ctrl
        return prev_ctrl

    def get_hand_dof_pos(self) -> np.ndarray:
        return np.asarray(
            self._backend.get_dof_pos()[:, : self._NUM_HAND_DOF],
            dtype=self._np_dtype,
        )

    def get_hand_dof_vel(self) -> np.ndarray:
        return np.asarray(
            self._backend.get_dof_vel()[:, : self._NUM_HAND_DOF],
            dtype=self._np_dtype,
        )

    def get_ball_pos(self) -> np.ndarray:
        return np.asarray(
            self._backend.get_body_pos_w(self._ball_body_ids)[:, 0, :],
            dtype=self._np_dtype,
        )

    def get_ball_quat(self) -> np.ndarray:
        return np.asarray(
            self._backend.get_body_quat_w(self._ball_body_ids)[:, 0, :],
            dtype=self._np_dtype,
        )

    def get_ball_linvel(self) -> np.ndarray:
        return np.asarray(
            self._backend.get_body_lin_vel_w(self._ball_body_ids)[:, 0, :],
            dtype=self._np_dtype,
        )

    def get_ball_angvel(self) -> np.ndarray:
        return np.asarray(
            self._backend.get_body_ang_vel_w(self._ball_body_ids)[:, 0, :],
            dtype=self._np_dtype,
        )

    def get_fingertip_pos(self) -> np.ndarray:
        return np.asarray(
            self._backend.get_body_pos_w(self._fingertip_body_ids),
            dtype=self._np_dtype,
        )

    def get_sensor_data(self, name: str) -> np.ndarray:
        return np.asarray(self._backend.get_sensor_data(name), dtype=self._np_dtype)


AllegroBaseMjEnv = AllegroBaseEnv
