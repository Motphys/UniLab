"""G1 SAC environment - inherits from PPO for code reuse."""
from __future__ import annotations

from dataclasses import dataclass, field
from etils import epath
import numpy as np

from unilab.envs import registry
from unilab.envs.backend import create_backend
from unilab.envs.dtype_config import get_global_dtype
from unilab.envs.locomotion.g1.base import G1BaseCfg, G1BaseEnv
from unilab.envs.locomotion.g1.joystick import G1JoystickPPO, InitState


@dataclass
class Commands:
    """对齐 holosoma: 多方向命令采样"""
    vel_limit = [
        [-0.6, -0.4, -0.8],  # [vx_min, vy_min, vyaw_min]
        [1.0, 0.4, 0.8]      # [vx_max, vy_max, vyaw_max]
    ]

@dataclass
class RewardConfigSAC:
    """对齐 holosoma G1 FastSAC 奖励权重"""
    scales: dict[str, float] = field(
        default_factory=lambda: {
            "tracking_lin_vel": 2.0,      # holosoma: 2.0
            "tracking_ang_vel": 1.5,      # holosoma: 1.5
            "ang_vel_xy": -1.0,           # holosoma: -1.0 (penalty_ang_vel_xy)
            "orientation": -10.0,         # holosoma: -10.0 (penalty_orientation)
            "action_rate": -2.0,          # holosoma: -2.0 (penalty_action_rate)
            "pose": -0.5,                 # holosoma: -0.5 (weighted pose penalty)
            "close_feet_xy": -10.0,       # holosoma: -10.0 (penalty_close_feet_xy)
            "feet_ori": -5.0,             # holosoma: -5.0 (penalty_feet_ori)
            "feet_phase": 5.0,            # holosoma: 5.0 (gait phase reward)
            "alive": 10.0,                # holosoma: 10.0
        }
    )
    tracking_sigma: float = 0.25
    base_height_target: float = 0.754
    min_base_height: float = 0.55
    max_tilt_deg: float = 25.0
    # gait 参数
    gait_frequency: float = 1.5
    # feet_phase 参数
    swing_height: float = 0.09
    feet_phase_tracking_sigma: float = 0.008
    # close_feet_xy 参数
    close_feet_threshold: float = 0.15
    # pose 权重（29 个关节）
    pose_weights: list[float] = field(
        default_factory=lambda: [
            0.01, 1.0, 5.0, 0.01, 5.0, 5.0,  # 左腿
            0.01, 1.0, 5.0, 0.01, 5.0, 5.0,  # 右腿
            50.0, 50.0, 50.0,                # 腰部
            50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0,  # 左臂
            50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0,  # 右臂
        ]
    )


@registry.envcfg("G1WalkTaskMjSAC")
@dataclass
class G1JoystickSACCfg(G1BaseCfg):
    model_file: str = str(epath.Path(__file__).parent / "xml" / "scene_flat.xml")
    max_episode_seconds: float = 20.0
    init_state: InitState = field(default_factory=InitState)
    commands: Commands = field(default_factory=Commands)
    reward_config: RewardConfigSAC = field(default_factory=RewardConfigSAC)


@registry.env("G1WalkTaskMjSAC", sim_backend="mujoco")
@registry.env("G1WalkTaskMjSAC", sim_backend="motrix")
class G1WalkTaskMjSAC(G1JoystickPPO):
    """G1 SAC environment - inherits from PPO, overrides rewards."""

    def __init__(self, cfg: G1JoystickSACCfg, num_envs=1, backend_type="mujoco"):
        backend = create_backend(backend_type, cfg.model_file, num_envs, cfg.sim_dt, body_name=cfg.asset.body_name)
        G1BaseEnv.__init__(self, cfg, backend, num_envs)
        self._enable_reward_log = True
        self._gait_phase_delta = float(2.0 * np.pi * cfg.reward_config.gait_frequency * cfg.ctrl_dt)
        self._pose_weights = np.array(cfg.reward_config.pose_weights, dtype=get_global_dtype())
        self._init_obs_space()
        self._init_reward_functions()

    def _init_reward_functions(self):
        """对齐 holosoma G1 FastSAC 奖励函数"""
        self._reward_fns = {
            "tracking_lin_vel": self._reward_tracking_lin_vel,
            "tracking_ang_vel": self._reward_tracking_ang_vel,
            "ang_vel_xy": self._reward_ang_vel_xy,
            "orientation": self._reward_orientation,
            "action_rate": self._reward_action_rate,
            "pose": self._reward_pose,
            "close_feet_xy": self._reward_close_feet_xy,
            "feet_ori": self._reward_feet_ori,
            "feet_phase": self._reward_feet_phase,
            "alive": self._reward_alive,
        }

    def _reward_orientation(self, info, linvel, gyro, gravity, dof_pos, dof_vel, qpos):
        """惩罚姿态偏差（roll/pitch）"""
        return np.square(gravity[:, 0]) + np.square(gravity[:, 1])

    def _reward_pose(self, info, linvel, gyro, gravity, dof_pos, dof_vel, qpos):
        """加权惩罚偏离默认姿态"""
        diff = dof_pos - self.default_angles
        return np.sum(self._pose_weights * np.square(diff), axis=1)

    def _reward_close_feet_xy(self, info, linvel, gyro, gravity, dof_pos, dof_vel, qpos):
        """惩罚双脚过近"""
        left_foot = self._backend.get_sensor_data("left_foot_pos")
        right_foot = self._backend.get_sensor_data("right_foot_pos")
        feet_dist = np.linalg.norm(left_foot[:, :2] - right_foot[:, :2], axis=1)
        threshold = self._cfg.reward_config.close_feet_threshold
        return np.where(feet_dist < threshold, np.square(feet_dist - threshold), 0.0)

    def _reward_feet_ori(self, info, linvel, gyro, gravity, dof_pos, dof_vel, qpos):
        """惩罚脚部姿态偏差"""
        left_foot_quat = self._backend.get_sensor_data("left_foot_quat")
        right_foot_quat = self._backend.get_sensor_data("right_foot_quat")
        # 简化：惩罚 quat 的 x,y 分量（roll/pitch）
        return np.square(left_foot_quat[:, 0]) + np.square(left_foot_quat[:, 1]) + \
               np.square(right_foot_quat[:, 0]) + np.square(right_foot_quat[:, 1])

    def _reward_feet_phase(self, info, linvel, gyro, gravity, dof_pos, dof_vel, qpos):
        """步态相位奖励：鼓励正确的摆动腿高度"""
        left_foot = self._backend.get_sensor_data("left_foot_pos")
        right_foot = self._backend.get_sensor_data("right_foot_pos")
        gait_phase = info.get("gait_phase", np.zeros((self._num_envs, 1), dtype=get_global_dtype()))

        cfg = self._cfg.reward_config
        target_height = cfg.swing_height * np.abs(np.sin(gait_phase[:, 0]))
        left_error = np.square(left_foot[:, 2] - target_height)
        right_error = np.square(right_foot[:, 2] - target_height)
        return np.exp(-(left_error + right_error) / cfg.feet_phase_tracking_sigma)

    def _reward_alive(self, info, linvel, gyro, gravity, dof_pos, dof_vel, qpos):
        return np.ones((self._num_envs,), dtype=get_global_dtype())
