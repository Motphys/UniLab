"""G1 SAC environment - inherits from PPO for code reuse."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from etils import epath

from unilab.assets import ASSETS_ROOT_PATH
from unilab.base import registry
from unilab.base.backend import create_backend
from unilab.base.curriculum import EpisodeLengthTracker, PenaltyCurriculum
from unilab.base.dtype_config import get_global_dtype
from unilab.envs.locomotion.common import rewards
from unilab.envs.locomotion.common.commands import Commands
from unilab.envs.locomotion.common.rewards import RewardContext
from unilab.envs.locomotion.g1.base import G1BaseCfg, G1BaseEnv
from unilab.envs.locomotion.g1.joystick import (
    G1DomainRandConfig,
    G1JoystickDomainRandomizationProvider,
    G1JoystickPPO,
    InitState,
)


@dataclass
class ControlConfigSAC:
    action_scale: float = 1.0
    simulate_action_latency: bool = False


@dataclass
class RewardConfigSAC:
    """对齐 holosoma G1 FastSAC 奖励权重"""

    scales: dict[str, float]
    tracking_sigma: float
    base_height_target: float
    min_base_height: float
    max_tilt_deg: float
    gait_frequency: float
    feet_phase_swing_height: float
    feet_phase_tracking_sigma: float
    close_feet_threshold: float
    pose_weights: list[float]


@registry.envcfg("G1WalkTaskMjSAC")
@dataclass
class G1JoystickSACCfg(G1BaseCfg):
    reward_config: RewardConfigSAC | None = None
    model_file: str = str(ASSETS_ROOT_PATH / "robots" / "g1" / "scene_flat.xml")
    max_episode_seconds: float = 20.0
    init_state: InitState = field(default_factory=InitState)
    commands: Commands = field(default_factory=Commands)
    control_config: ControlConfigSAC = field(default_factory=ControlConfigSAC)  # type: ignore[assignment]
    domain_rand: G1DomainRandConfig = field(default_factory=G1DomainRandConfig)
    gait_phase_init_mode: str = "offset_phase"
    reset_base_qvel_limit: float = 0.5


@registry.env("G1WalkTaskMjSAC", sim_backend="mujoco")
@registry.env("G1WalkTaskMjSAC", sim_backend="motrix")
class G1WalkTaskMjSAC(G1JoystickPPO):
    """G1 SAC environment - inherits from PPO, overrides rewards."""

    def __init__(self, cfg: G1JoystickSACCfg, num_envs=1, backend_type="mujoco"):
        if cfg.reward_config is None:
            raise ValueError("reward_config must be provided via Hydra configuration")
        backend = create_backend(
            backend_type,
            cfg.model_file,
            num_envs,
            cfg.sim_dt,
            base_name=cfg.asset.base_name,
            iterations=cfg.iterations,
        )
        G1BaseEnv.__init__(self, cfg, backend, num_envs)
        self._enable_reward_log = True
        self._reward_cfg = cfg.reward_config
        self._gait_phase_delta = float(2.0 * np.pi * cfg.reward_config.gait_frequency * cfg.ctrl_dt)
        self._pose_weights = np.array(cfg.reward_config.pose_weights, dtype=get_global_dtype())

        # Curriculum learning - 更宽松的初始配置
        self._episode_tracker = EpisodeLengthTracker(num_envs)
        self._penalty_curriculum = PenaltyCurriculum(
            self,
            enabled=True,
            initial_scale=0.5,
            min_scale=0.5,
            max_scale=1.0,
            level_down_threshold=150.0,
            level_up_threshold=750.0,
            degree=0.001,
        )

        self._init_reward_functions()
        if cfg.domain_rand.randomize_kp or cfg.domain_rand.randomize_kd:
            base_kp, base_kd = backend.get_actuator_gains()
            dr_provider = G1JoystickDomainRandomizationProvider(base_kp=base_kp, base_kd=base_kd)
        else:
            dr_provider = G1JoystickDomainRandomizationProvider()
        self._init_domain_randomization(dr_provider)

    def _compute_obs(
        self, info: dict, linvel, gyro, gravity, dof_pos, dof_vel
    ) -> dict[str, np.ndarray]:
        diff = dof_pos - self.default_angles
        command = info["commands"]
        last_actions = info.get("current_actions", np.zeros_like(diff))
        gait_phase = info.get("gait_phase", np.zeros((self._num_envs, 2), dtype=get_global_dtype()))
        actor = np.concatenate(
            [gyro * 0.25, -gravity, diff, dof_vel * 0.05, last_actions, command, gait_phase],
            axis=1,
            dtype=get_global_dtype(),
        )
        return {
            "obs": actor,
            "privileged": np.asarray(linvel * 2.0, dtype=get_global_dtype()),
        }

    def _init_reward_functions(self):
        """对齐 holosoma G1 FastSAC 奖励函数"""
        from typing import Any

        self._reward_fns: dict[str, Any] = {
            "tracking_lin_vel": rewards.tracking_lin_vel,
            "tracking_ang_vel": rewards.tracking_ang_vel,
            "penalty_ang_vel_xy": rewards.ang_vel_xy,
            "penalty_orientation": rewards.orientation,
            "penalty_action_rate": rewards.action_rate,
            "pose": rewards.weighted_pose,
            # "penalty_close_feet_xy": self._reward_close_feet_xy,
            "penalty_feet_ori": self._reward_feet_ori,
            "feet_phase": self._reward_feet_phase,
            "alive": rewards.alive,
        }

    def _reward_close_feet_xy(self, ctx: RewardContext):
        """惩罚双脚过近"""
        left_foot = self._backend.get_sensor_data("left_foot_pos")
        right_foot = self._backend.get_sensor_data("right_foot_pos")
        feet_dist = np.linalg.norm(left_foot[:, :2] - right_foot[:, :2], axis=1)
        threshold = self._cfg.reward_config.close_feet_threshold  # type: ignore[union-attr]
        return np.where(feet_dist < threshold, np.square(feet_dist - threshold), 0.0)

    def _reward_feet_ori(self, ctx: RewardContext):
        """惩罚脚部姿态偏差"""
        left_foot_quat = self._backend.get_sensor_data("left_foot_quat")
        right_foot_quat = self._backend.get_sensor_data("right_foot_quat")
        return (
            np.square(left_foot_quat[:, 1])
            + np.square(left_foot_quat[:, 2])
            + np.square(right_foot_quat[:, 1])
            + np.square(right_foot_quat[:, 2])
        )

    def _reward_feet_air_time(self, ctx: RewardContext):
        """奖励脚离地时间"""
        air_time = ctx.info.get(
            "feet_air_time", np.zeros((self._num_envs, 2), dtype=get_global_dtype())
        )
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
            state.info["log"]["curriculum/average_episode_length"] = float(
                self._episode_tracker.average_length
            )
            state.info["log"]["curriculum/penalty_scale"] = float(
                self._penalty_curriculum.current_scale
            )

        return state
