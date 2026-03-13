"""G1 SAC environment - inherits from PPO for code reuse."""
from __future__ import annotations

from dataclasses import dataclass, field
from etils import epath
import numpy as np

from unilab.base import registry
from unilab.base.backend import create_backend
from unilab.base.dtype_config import get_global_dtype
from unilab.envs.locomotion.g1.base import G1BaseCfg, G1BaseEnv
from unilab.envs.locomotion.g1.joystick import G1JoystickPPO, InitState
from unilab.base.curriculum import EpisodeLengthTracker, PenaltyCurriculum


@dataclass
class ControlConfigSAC:
    action_scale: float = 1.  # holosoma 0.25
    simulate_action_latency: bool = False

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
            "penalty_ang_vel_xy": -1.0,
            "penalty_orientation": -10.0,
            "penalty_action_rate": -2.0,
            "pose": -0.5,                 # holosoma: -0.5 (weighted pose penalty)
            # "penalty_close_feet_xy": -10.0,
            "penalty_feet_ori": -25.0,    # holosoma: 5.0 (feet ori penalty)
            "feet_phase": 5.0,            # holosoma: 5.0 (gait phase reward)
            "alive": 10.0,                # holosoma: 10.0
        }
    )
    tracking_sigma: float = 0.25
    base_height_target: float = 0.754
    min_base_height: float = 0.3  # 放宽以允许更多探索
    max_tilt_deg: float = 65.0  # 放宽以允许更多探索
    # gait 参数
    gait_frequency: float = 1.5
    # feet_phase 参数
    feet_phase_swing_height: float = 0.09
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
    control_config: ControlConfigSAC = field(default_factory=ControlConfigSAC)


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

        # Curriculum learning - 更宽松的初始配置
        self._episode_tracker = EpisodeLengthTracker(num_envs)
        self._penalty_curriculum = PenaltyCurriculum(
            self, enabled=True,
            initial_scale=0.1,      # 从0.1开始，更容易
            min_scale=0.1,          # 最小0.1
            max_scale=1.0,          # 最大1.0
            level_down_threshold=50.0,   # 低于50步降低难度
            level_up_threshold=500.0,    # 高于500步增加难度
            degree=0.002            # 调整速度加快
        )

        self._init_obs_space()
        self._init_reward_functions()

    def _init_reward_functions(self):
        """对齐 holosoma G1 FastSAC 奖励函数"""
        self._reward_fns = {
            "tracking_lin_vel": self._reward_tracking_lin_vel,
            "tracking_ang_vel": self._reward_tracking_ang_vel,
            "penalty_ang_vel_xy": self._reward_ang_vel_xy,
            "penalty_orientation": self._reward_orientation,
            "penalty_action_rate": self._reward_action_rate,
            "pose": self._reward_pose,
            # "penalty_close_feet_xy": self._reward_close_feet_xy,
            "penalty_feet_ori": self._reward_feet_ori,
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
        # MuJoCo quat: [w,x,y,z], 惩罚 x,y 分量（roll/pitch）
        return np.square(left_foot_quat[:, 1]) + np.square(left_foot_quat[:, 2]) + \
               np.square(right_foot_quat[:, 1]) + np.square(right_foot_quat[:, 2])

    def _reward_alive(self, info, linvel, gyro, gravity, dof_pos, dof_vel, qpos):
        return np.ones((self._num_envs,), dtype=get_global_dtype())

    def _reward_lin_vel_z(self, info, linvel, gyro, gravity, dof_pos, dof_vel, qpos):
        """惩罚 z 方向线速度"""
        return np.square(linvel[:, 2])

    def _reward_base_height(self, info, linvel, gyro, gravity, dof_pos, dof_vel, qpos):
        """惩罚基座高度偏差"""
        base_height = qpos[:, 2]
        return np.square(base_height - self._cfg.reward_config.base_height_target)

    def _reward_torques(self, info, linvel, gyro, gravity, dof_pos, dof_vel, qpos):
        """惩罚力矩"""
        torques = info.get("torques", np.zeros((self._num_envs, self._num_action), dtype=get_global_dtype()))
        return np.sum(np.abs(torques), axis=1)

    def _reward_energy(self, info, linvel, gyro, gravity, dof_pos, dof_vel, qpos):
        """惩罚能量消耗"""
        torques = info.get("torques", np.zeros((self._num_envs, self._num_action), dtype=get_global_dtype()))
        return np.sum(np.abs(dof_vel) * np.abs(torques), axis=1)

    def _reward_dof_acc(self, info, linvel, gyro, gravity, dof_pos, dof_vel, qpos):
        """惩罚关节加速度"""
        qacc = info.get("qacc", np.zeros((self._num_envs, self._num_action), dtype=get_global_dtype()))
        return np.sum(np.square(qacc), axis=1)

    def _reward_upright(self, info, linvel, gyro, gravity, dof_pos, dof_vel, qpos):
        """奖励直立姿态（mjlab flat_orientation）"""
        xy_squared = np.sum(np.square(gravity[:, :2]), axis=1)
        return np.exp(-xy_squared / 0.25)

    def _reward_feet_air_time(self, info, linvel, gyro, gravity, dof_pos, dof_vel, qpos):
        """奖励脚离地时间"""
        air_time = info.get("feet_air_time", np.zeros((self._num_envs, 2), dtype=get_global_dtype()))
        in_range = (air_time > 0.05) & (air_time < 0.5)
        return np.sum(in_range.astype(float), axis=1)

    def update_state(self, state):
        """Override to add curriculum update."""
        # Call parent first to compute terminated/truncated
        state = super().update_state(state)

        # Track episode lengths AFTER parent update (when terminated is set)
        # Note: steps will be incremented in np_env.step() after this returns
        if np.any(state.done):
            done_indices = np.where(state.done)[0]
            # Add 1 because steps will be incremented after update_state
            episode_lengths = state.info["steps"][done_indices] + 1
            self._episode_tracker.update(episode_lengths)
            self._penalty_curriculum.update(self._episode_tracker.average_length)

            # Always log curriculum metrics when episode ends
            if "log" not in state.info:
                state.info["log"] = {}
            state.info["log"]["curriculum/average_episode_length"] = float(self._episode_tracker.average_length)
            state.info["log"]["curriculum/penalty_scale"] = float(self._penalty_curriculum.current_scale)

        return state
