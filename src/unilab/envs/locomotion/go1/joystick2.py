from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from unilab.assets import ASSETS_ROOT_PATH
from unilab.base import registry
from unilab.base.backend import create_backend
from unilab.base.np_env import NpEnvState
from unilab.envs.locomotion.common import rewards
from unilab.envs.locomotion.common.commands import Commands
from unilab.envs.locomotion.common.domain_rand import DomainRandConfig
from unilab.envs.locomotion.common.dr_provider import LocomotionDRProvider
from unilab.envs.locomotion.common.rewards import RewardContext
from unilab.envs.locomotion.go1.base import Go1BaseCfg, Go1BaseEnv
from unilab.base.scene import SceneCfg
from unilab.dtype_config import get_global_dtype

@dataclass
class InitState:
    pos = [0.0, 0.0, 0.45]


@dataclass
class RewardConfig:
    scales: dict[str, float]
    tracking_sigma: float
    feet_air_time_threshold: float


@dataclass
class JoystickSensor:
    local_linvel = "local_linvel"
    gyro = "gyro"
    feet_force = ["FL_foot_contact", "FR_foot_contact", "RL_foot_contact", "RR_foot_contact"]
    feet_pos = ["FL_pos", "FR_pos", "RL_pos", "RR_pos"]


@registry.envcfg("Go1JoystickFlat2")
@dataclass
class Go1JoystickCfg2(Go1BaseCfg):
    scene: SceneCfg = field(
        default_factory=lambda: SceneCfg(
            model_file=str(ASSETS_ROOT_PATH / "robots" / "go1" / "scene_flat.xml")
        )
    )
    max_episode_seconds: float = 20.0
    init_state: InitState = field(default_factory=InitState)
    commands: Commands = field(default_factory=Commands)
    reward_config: RewardConfig | None = None
    sensor: JoystickSensor = field(default_factory=JoystickSensor)  # type: ignore[assignment]
    domain_rand: DomainRandConfig = field(
        default_factory=lambda: DomainRandConfig(
            randomize_base_mass=True,
            random_com=True,
            push_robots=True,
        )
    )


class Go1JoystickDomainRandomizationProvider(LocomotionDRProvider):
    def _compute_reset_obs(
        self,
        env: Any,
        env_ids: Any,
        info_updates: Any,
        linvel: Any,
        gyro: Any,
        gravity: Any,
        dof_pos: Any,
        dof_vel: Any,
    ) -> dict[str, np.ndarray]:
        return env._compute_obs(  # type: ignore[no-any-return]
            info_updates, linvel, gyro, gravity, dof_pos, dof_vel
        )


@registry.env("Go1JoystickFlat2", sim_backend="mujoco")
@registry.env("Go1JoystickFlat2", sim_backend="motrix")
class Go1WalkTask2(Go1BaseEnv):
    _cfg: Go1JoystickCfg2

    def __init__(self, cfg: Go1JoystickCfg2, num_envs=1, backend_type="mujoco"):
        if cfg.reward_config is None:
            raise ValueError("reward_config must be provided via Hydra configuration")
        backend = create_backend(
            backend_type,
            cfg.scene,
            num_envs,
            cfg.sim_dt,
            base_name=cfg.asset.base_name,
            push_body_name=cfg.domain_rand.push_body_name,
            position_actuator_gains={"kp": cfg.control_config.Kp, "kd": cfg.control_config.Kd},
            motrix_max_iterations=cfg.motrix_max_iterations,
            post_step_forward_sensor=cfg.post_step_forward_sensor,
        )
        super().__init__(cfg, backend, num_envs)
        self._enable_reward_log = True
        self._reward_cfg = cfg.reward_config
        self._init_reward_functions()
        self.feet_force = np.zeros((num_envs, len(cfg.sensor.feet_force), 3), dtype=np.float32)
        self.feet_air_time = np.zeros((num_envs, len(cfg.sensor.feet_force)), dtype=np.float32)
        self.feet_contacts = np.zeros((num_envs, len(cfg.sensor.feet_force)), dtype=bool)
        self._last_dof_vel = np.zeros((num_envs, self._num_action), dtype=get_global_dtype())
        self._init_domain_randomization(Go1JoystickDomainRandomizationProvider())
        self.feet_pos = np.zeros((num_envs, len(cfg.sensor.feet_pos), 3), dtype=np.float32)

    @property
    def obs_groups_spec(self) -> dict[str, int]:
        # gyro(3) + gravity(3) + diff(12) + dof_vel(12) + action(12) + cmd(3) = 45
        return {"obs": 45, "critic": 48}

    def _init_reward_functions(self):
        self._reward_fns: dict[str, Any] = {
            "tracking_lin_vel": rewards.tracking_lin_vel,
            "tracking_ang_vel": rewards.tracking_ang_vel,
            "lin_vel_z": rewards.lin_vel_z,
            "ang_vel_xy": rewards.ang_vel_xy,
            "action_rate": rewards.action_rate,
            "torques": rewards.dof_torques_l2,
            "feet_air_time": self._reward_feet_air_time,
            "orientation": rewards.orientation,
            "dof_acc": rewards.dof_acc,
            "feet_clearance": self._reward_feet_clearance,
        }

    def reset(self, env_indices: np.ndarray) -> tuple[dict[str, np.ndarray], dict]:
        obs, info = super().reset(env_indices)
        env_ids = np.asarray(env_indices, dtype=np.int32)
        self.feet_air_time[env_ids] = 0.0
        self.feet_contacts[env_ids] = False
        self._last_dof_vel[env_ids] = 0.0
        return obs, info

    def update_state(self, state: NpEnvState) -> NpEnvState:
        linvel = self.get_local_linvel()
        gyro = self.get_gyro()
        gravity = self._backend.get_sensor_data("upvector")
        dof_pos = self.get_dof_pos()
        dof_vel = self.get_dof_vel()

        self.feet_force[:, :, :] = 0
        for i in range(len(self._cfg.sensor.feet_force)):
            self.feet_force[:, i, :] = self._backend.get_sensor_data(self._cfg.sensor.feet_force[i])
        for i in range(len(self._cfg.sensor.feet_pos)):
            self.feet_pos[:, i, :] = self._backend.get_sensor_data(self._cfg.sensor.feet_pos[i])
        
        terminated = gravity[:, 2] <= 0.5
        
        state.info["qacc"] = (
            dof_vel - self._last_dof_vel
        ) / self._cfg.ctrl_dt
        #判断是否触地
        self.feet_contacts = self.feet_force[:, :, 0].astype(bool)

        reward = self._compute_reward(state.info, linvel, gyro, dof_pos)
        # 更新滞空时间
        self.feet_air_time += self._cfg.ctrl_dt
        self.feet_air_time *= ~self.feet_contacts
        
        obs = self._compute_obs(
            state.info, linvel, gyro, gravity, dof_pos, dof_vel
        )
        return state.replace(obs=obs, reward=reward, terminated=terminated)

    def _compute_obs(
        self, info: dict, linvel, gyro, gravity, dof_pos, dof_vel
    ) -> dict[str, np.ndarray]:
        noise_cfg = self._cfg.noise_config
        diff = dof_pos - self.default_angles
        gyro = self._obs_noise(gyro, noise_cfg.scale_gyro)
        gravity = self._obs_noise(gravity, noise_cfg.scale_gravity)
        diff = self._obs_noise(diff, noise_cfg.scale_joint_angle)
        dof_vel = self._obs_noise(dof_vel, noise_cfg.scale_joint_vel)
        linvel = self._obs_noise(linvel, noise_cfg.scale_linvel)
        command = info["commands"]
        last_actions = info.get("current_actions", np.zeros_like(diff))
        obs = np.concatenate(
            [gyro, -gravity, diff, dof_vel, last_actions, command],
            axis=1,
            dtype=get_global_dtype(),
        )
        critic = np.concatenate([obs, linvel], axis=1, dtype=get_global_dtype())
        return {"obs": obs, "critic": critic}
    
    def apply_action(self, actions: np.ndarray, state: NpEnvState) -> np.ndarray: 
        state.info["torques"] = self._compute_torques(actions)
        self._last_dof_vel[:] = self.get_dof_vel()
        return super().apply_action(actions, state)

    def _compute_reward(self, info: dict, linvel, gyro, dof_pos) -> np.ndarray:
        cfg = self._reward_cfg
        ctx = RewardContext(
            info=info,
            linvel=linvel,
            gyro=gyro,
            dof_pos=dof_pos,
            num_envs=self._num_envs,
            default_angles=self.default_angles,
            tracking_sigma=cfg.tracking_sigma,
            base_height=self._backend.get_base_pos()[:, 2],
            gravity=self._backend.get_sensor_data("upvector")
        )
        return rewards.run_reward_dispatch(
            scales=cfg.scales,
            fns=self._reward_fns,
            ctx=ctx,
            info=info,
            enable_log=self._enable_reward_log,
            ctrl_dt=self._cfg.ctrl_dt,
        )
        
    
    def _reward_feet_air_time(self,ctx: RewardContext):
        # Reward long steps
        first_contact = (self.feet_air_time > 0.0) * self.feet_contacts
        # reward only on first contact with the ground
        rew_airTime = np.sum((self.feet_air_time - self._reward_cfg.feet_air_time_threshold) * first_contact, axis=1)
        # no reward for zero command
        commands = ctx.info["commands"]
        rew_airTime *= np.linalg.norm(commands[:, :2], axis=1) > 0.1
        return rew_airTime
    
    def _reward_feet_clearance(self, ctx: RewardContext) -> np.ndarray:
        is_swing = self.feet_air_time > 0.0
        swing_rew = np.clip(self.feet_pos[:, :, 2] / 0.05, 0, 1) * is_swing
        reward: np.ndarray = np.sum(swing_rew, axis=1) / len(self._cfg.sensor.feet_pos)
        return reward
    
    
    def _compute_torques(self, actions):
        actions_scaled = actions * self._cfg.control_config.action_scale
        torques = self._cfg.control_config.Kp * (
            actions_scaled + self.default_angles - self.get_dof_pos()
        ) - self._cfg.control_config.Kd * self.get_dof_vel()
        return torques


