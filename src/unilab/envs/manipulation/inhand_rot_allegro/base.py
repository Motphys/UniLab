from __future__ import annotations

from dataclasses import dataclass, field

import gymnasium as gym
import mujoco
import numpy as np

from unilab.base.backend import SimBackend
from unilab.base.base import EnvCfg
from unilab.base.dtype_config import get_global_dtype
from unilab.base.np_env import NpEnv, NpEnvState


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
    model_file: str = ""
    sim_dt: float = 0.005
    ctrl_dt: float = 0.05
    noise_config: NoiseConfig = field(default_factory=NoiseConfig)
    control_config: ControlConfig = field(default_factory=ControlConfig)


def get_model_ctrl_limits(model, dtype: np.dtype) -> tuple[np.ndarray, np.ndarray]:
    if hasattr(model, "actuator_ctrlrange"):
        low = np.asarray(model.actuator_ctrlrange[:, 0], dtype=dtype)
        high = np.asarray(model.actuator_ctrlrange[:, 1], dtype=dtype)
    else:
        low = np.asarray(model.actuator_ctrl_limits[0, :], dtype=dtype)
        high = np.asarray(model.actuator_ctrl_limits[1, :], dtype=dtype)
    return low, high


class AllegroBaseMjEnv(NpEnv):
    _NUM_HAND_DOF: int = 16
    _cfg: AllegroBaseCfg

    def __init__(self, cfg: AllegroBaseCfg, backend: SimBackend, num_envs: int = 1):
        super().__init__(cfg, backend, num_envs)

        self._np_dtype = get_global_dtype()

        model = self._backend.model
        body = getattr(self._backend, "_body", None)
        if body is not None and hasattr(body, "joints"):
            print("[AllegroBaseMjEnv] body joint order for get_dof_pos():")
            for i, joint in enumerate(body.joints):
                print(i, getattr(joint, "name", "<unnamed>"))
        if body is not None and hasattr(body, "dofs"):
            print("[AllegroBaseMjEnv] body dof order for get_dof_pos():")
            for i, dof in enumerate(body.dofs):
                print(i, getattr(dof, "name", "<unnamed>"))
        if hasattr(model, "dof_damping"):
            model.dof_damping[: self._NUM_HAND_DOF] = cfg.control_config.kd
            model.actuator_gainprm[:, 0] = cfg.control_config.kp
            model.actuator_biasprm[:, 1] = -cfg.control_config.kp

        self._ctrl_lower, self._ctrl_upper = get_model_ctrl_limits(model, self._np_dtype)
        self._init_action_space()
        self._num_action = self._action_space.shape[0]
        if self._num_action != self._NUM_HAND_DOF:
            raise ValueError(f"Expected {self._NUM_HAND_DOF} actuators, got {self._num_action}")

        self._init_buffer()
        self.nq = int(self._init_qpos.shape[0])
        self.nv = int(self._init_qvel.shape[0])
        self._idx_qpos = 1
        self._idx_qvel = 1 + self.nq
        self._ball_body_ids = self._resolve_body_ids(["ball"])

    def _init_action_space(self) -> None:
        self._action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(self._NUM_HAND_DOF,), dtype=float
        )

    @property
    def action_space(self) -> gym.spaces.Box:
        return self._action_space  # type: ignore[no-any-return]

    def _init_buffer(self) -> None:
        self.default_angles = np.zeros((self._num_action,), dtype=self._np_dtype)
        model = self._backend.model
        if hasattr(model, "key_qpos"):
            key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
            if key_id >= 0:
                self._init_qpos = np.asarray(model.key_qpos[key_id].copy(), dtype=self._np_dtype)
                self._init_ctrl = np.asarray(model.key_ctrl[key_id].copy(), dtype=self._np_dtype)
                self.default_angles = self._init_qpos[: self._NUM_HAND_DOF]
            else:
                raise ValueError("Keyframe 'home' not found in MuJoCo model")
            self._init_qvel = np.zeros((model.nv,), dtype=self._np_dtype)
        elif hasattr(model, "keyframes") and model.num_keyframes > 0:
            keyframe = model.keyframes[0]
            self._init_qpos = np.asarray(keyframe.dof_pos, dtype=self._np_dtype)
            self._init_ctrl = np.asarray(
                self._init_qpos[: self._NUM_HAND_DOF], dtype=self._np_dtype
            )
            self.default_angles = self._init_qpos[: self._NUM_HAND_DOF]
            self._init_qvel = np.zeros((model.num_dof_vel,), dtype=self._np_dtype)
        else:
            raise ValueError(
                "No keyframe found in model. Model must have either MuJoCo key_qpos or Motrix keyframes."
            )

    def _resolve_body_ids(self, body_names: list[str]) -> np.ndarray:
        model = self._backend.model
        if hasattr(model, "body"):
            body_ids = [
                mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name) for name in body_names
            ]
        else:
            body_ids = [model.get_link_index(name) for name in body_names]
        if any(body_id is None or body_id < 0 for body_id in body_ids):
            raise ValueError(f"Failed to resolve body ids for {body_names}")
        return np.asarray(body_ids, dtype=np.int32)

    def apply_action(self, actions: np.ndarray, state: NpEnvState) -> np.ndarray:
        clipped_actions = np.clip(actions, -1.0, 1.0)
        state.info["last_actions"] = np.array(state.info["current_actions"])
        state.info["current_actions"] = clipped_actions

        scale = self._cfg.control_config.action_scale
        new_ctrl = state.info["prev_ctrl"] + scale * clipped_actions
        new_ctrl = np.clip(new_ctrl, self._ctrl_lower, self._ctrl_upper)
        state.info["prev_ctrl"] = new_ctrl
        return np.asarray(new_ctrl)

    def get_hand_dof_pos(self) -> np.ndarray:
        if hasattr(self._backend, "get_physics_state"):  # TODO: wait for backend update
            physics_state = self._backend.get_physics_state()
            return np.asarray(
                physics_state[:, self._idx_qpos : self._idx_qpos + self._NUM_HAND_DOF]
            )
        return np.asarray(self._backend.get_dof_pos())

    def get_hand_dof_vel(self) -> np.ndarray:
        if hasattr(self._backend, "get_physics_state"):  # TODO: wait for backend update
            physics_state = self._backend.get_physics_state()
            return np.asarray(
                physics_state[:, self._idx_qvel : self._idx_qvel + self._NUM_HAND_DOF]
            )
        return np.asarray(self._backend.get_dof_vel())

    def get_ball_pos(self) -> np.ndarray:
        return np.asarray(self._backend.get_body_pos_w(self._ball_body_ids)[:, 0, :])

    def get_ball_quat(self) -> np.ndarray:
        return np.asarray(self._backend.get_body_quat_w(self._ball_body_ids)[:, 0, :])

    def get_ball_linvel(self) -> np.ndarray:
        return np.asarray(self._backend.get_body_lin_vel_w(self._ball_body_ids)[:, 0, :])

    def get_ball_angvel(self) -> np.ndarray:
        return np.asarray(self._backend.get_body_ang_vel_w(self._ball_body_ids)[:, 0, :])
