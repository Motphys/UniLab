from __future__ import annotations

import gymnasium as gym
import mujoco
import numpy as np
from dataclasses import dataclass, field

from unilab.envs.base import EnvCfg
from unilab.envs.mujoco_env.mj_env import MjNpEnv, MjNpEnvState

# ─────────────────────────── Configuration ────────────────────────────


@dataclass
class NoiseConfig:
    level: float = 1.0
    scale_joint_angle: float = 0.02   # rad


@dataclass
class ControlConfig:
    # Incremental position target: ctrl = clip(prev_ctrl + action_scale * action, lo, hi)
    # Mirrors HORA's  targets = prev_targets + (1/24) * actions
    action_scale: float = 1.0 / 24.0
    # PD gains for torque computation (used in reward calculation)
    kp: float = 3.0
    kd: float = 0.1


@dataclass
class AllegroBaseCfg(EnvCfg):
    model_file: str = ""
    sim_dt: float = 0.005   # 5 ms physics step
    ctrl_dt: float = 0.05   # 50 ms control step  →  10 substeps
    noise_config: NoiseConfig = field(default_factory=NoiseConfig)
    control_config: ControlConfig = field(default_factory=ControlConfig)


# ─────────────────────────── Environment ──────────────────────────────

# Physics-state (FULLPHYSICS) layout for scene.xml
# (nq=23, nv=22, na=0  →  nstate=46)
#
#   ps[0]       time
#   ps[1:17]    hand qpos  (16 joints: ff×4, mf×4, rf×4, th×4)
#   ps[17:20]   ball position (x, y, z)
#   ps[20:24]   ball quaternion (w, x, y, z)
#   ps[24:40]   hand qvel  (16 velocities)
#   ps[40:43]   ball linear  velocity
#   ps[43:46]   ball angular velocity  ← rotation reward is computed here


class AllegroBaseMjEnv(MjNpEnv):
    # Slot constants – filled once in __init__ from the verified model.
    _NUM_HAND_DOF: int = 16

    def __init__(self, cfg: AllegroBaseCfg, num_envs: int = 1):
        super().__init__(cfg, num_envs)

        # Set PD gains in MuJoCo model
        self._model.actuator_gainprm[:, 0] = cfg.control_config.kp
        self._model.actuator_biasprm[:, 1] = -cfg.control_config.kp
        self._model.dof_damping[:self._NUM_HAND_DOF] = cfg.control_config.kd

        self.nq = self._model.nq   # 23
        self.nv = self._model.nv   # 22

        # physics_state offsets
        self._idx_qpos = 1
        self._idx_qvel = 1 + self.nq  # 24

        # hand occupies the first 16 DOFs
        assert self._model.nu == self._NUM_HAND_DOF, (
            f"Expected {self._NUM_HAND_DOF} actuators, got {self._model.nu}"
        )

        # ball positions inside physics_state
        self._ps_ball_pos  = self._idx_qpos + self._NUM_HAND_DOF       # 17
        self._ps_ball_quat = self._idx_qpos + self._NUM_HAND_DOF + 3   # 20
        self._ps_ball_linv = self._idx_qvel + self._NUM_HAND_DOF       # 40
        self._ps_ball_angv = self._idx_qvel + self._NUM_HAND_DOF + 3   # 43

        # joint limits (shape: (16,))
        self._ctrl_lower = self._model.actuator_ctrlrange[:, 0].astype(self._np_dtype)
        self._ctrl_upper = self._model.actuator_ctrlrange[:, 1].astype(self._np_dtype)

        self._init_action_space()
        self._num_action = self._action_space.shape[0]
        self._init_buffer()

    # ── Spaces ──────────────────────────────────────────────────────

    def _init_action_space(self):
        # Policy outputs are in [-1, 1]; action_scale converts them to rad increments.
        self._action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(self._NUM_HAND_DOF,), dtype=float
        )

    @property
    def action_space(self) -> gym.spaces.Box:
        return self._action_space

    # ── Buffers ─────────────────────────────────────────────────────

    def _init_buffer(self):
        key_id = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_KEY, "home")
        if key_id < 0:
            raise ValueError("Keyframe 'home' not found in model.")

        self._init_qpos = self._model.key_qpos[key_id].copy()   # (nq,) float64
        self._init_ctrl = self._model.key_ctrl[key_id].copy()   # (nu,) float64

        # Default hand pose (first 16 entries of qpos)
        self.default_angles = self._init_qpos[:self._NUM_HAND_DOF].astype(self._np_dtype)

    # ── Action / control ────────────────────────────────────────────

    def apply_action(self, actions: np.ndarray, state: MjNpEnvState) -> np.ndarray:
        """Incremental position control:  ctrl = clip(prev_ctrl + scale * action)."""
        state.info["last_actions"] = np.array(state.info["current_actions"])
        state.info["current_actions"] = actions

        scale = self._cfg.control_config.action_scale
        new_ctrl = state.info["prev_ctrl"] + scale * actions
        new_ctrl = np.clip(new_ctrl, self._ctrl_lower, self._ctrl_upper)
        state.info["prev_ctrl"] = new_ctrl
        return new_ctrl

    # ── State accessors ─────────────────────────────────────────────

    def get_hand_dof_pos(self, state: MjNpEnvState) -> np.ndarray:
        """(num_envs, 16) hand joint angles."""
        return state.physics_state[:, self._idx_qpos : self._idx_qpos + self._NUM_HAND_DOF]

    def get_hand_dof_vel(self, state: MjNpEnvState) -> np.ndarray:
        """(num_envs, 16) hand joint velocities."""
        return state.physics_state[:, self._idx_qvel : self._idx_qvel + self._NUM_HAND_DOF]

    def get_ball_pos(self, state: MjNpEnvState) -> np.ndarray:
        """(num_envs, 3) ball position in world frame."""
        return state.physics_state[:, self._ps_ball_pos : self._ps_ball_pos + 3]

    def get_ball_quat(self, state: MjNpEnvState) -> np.ndarray:
        """(num_envs, 4) ball quaternion (w, x, y, z)."""
        return state.physics_state[:, self._ps_ball_quat : self._ps_ball_quat + 4]

    def get_ball_linvel(self, state: MjNpEnvState) -> np.ndarray:
        """(num_envs, 3) ball linear velocity in world frame."""
        return state.physics_state[:, self._ps_ball_linv : self._ps_ball_linv + 3]

    def get_ball_angvel(self, state: MjNpEnvState) -> np.ndarray:
        """(num_envs, 3) ball angular velocity in world frame."""
        return state.physics_state[:, self._ps_ball_angv : self._ps_ball_angv + 3]
