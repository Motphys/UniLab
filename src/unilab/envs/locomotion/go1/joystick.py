from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from etils import epath

from unilab.assets import ASSETS_ROOT_PATH
from unilab.base import registry
from unilab.base.backend import create_backend
from unilab.base.np_env import NpEnvState
from unilab.base.scene import SceneCfg
from unilab.dtype_config import get_global_dtype
from unilab.envs.common.rotation import np_quat_mul, np_yaw_to_quat
from unilab.envs.locomotion.common import rewards
from unilab.envs.locomotion.common.commands import Commands
from unilab.envs.locomotion.common.domain_rand import DomainRandConfig
from unilab.envs.locomotion.common.dr_provider import LocomotionDRProvider
from unilab.envs.locomotion.common.rewards import RewardContext
from unilab.envs.locomotion.common.terrain_spawn import (
    TerrainCurriculumCfg,
    TerrainSpawnManager,
)
from unilab.envs.locomotion.go1.base import Go1BaseCfg, Go1BaseEnv


@dataclass
class InitState:
    pos = [0.0, 0.0, 0.45]


@dataclass
class RewardConfig:
    scales: dict[str, float]
    tracking_sigma: float
    base_height_target: float


@dataclass
class JoystickSensor:
    local_linvel = "local_linvel"
    gyro = "gyro"
    feet_force = ["FL_foot_contact", "FR_foot_contact", "RL_foot_contact", "RR_foot_contact"]
    feet_pos = ["FL_pos", "FR_pos", "RL_pos", "RR_pos"]


@registry.envcfg("Go1JoystickFlat")
@dataclass
class Go1JoystickCfg(Go1BaseCfg):
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
            info_updates, linvel, gyro, gravity, dof_pos, dof_vel, env.feet_phase[env_ids]
        )


@registry.env("Go1JoystickFlat", sim_backend="mujoco")
@registry.env("Go1JoystickFlat", sim_backend="motrix")
class Go1WalkTask(Go1BaseEnv):
    _cfg: Go1JoystickCfg

    def __init__(self, cfg: Go1JoystickCfg, num_envs=1, backend_type="mujoco"):
        if cfg.reward_config is None:
            raise ValueError("reward_config must be provided via Hydra configuration")
        self._scene_terrain_origins: np.ndarray | None = None
        scene_cfg = cfg.scene
        terrain_generator = scene_cfg.terrain.generator if scene_cfg.terrain is not None else None
        backend = create_backend(
            backend_type,
            cfg.scene,
            num_envs,
            cfg.sim_dt,
            base_name=cfg.asset.base_name,
            push_body_name=cfg.domain_rand.push_body_name,
            position_actuator_gains={"kp": cfg.control_config.Kp, "kd": cfg.control_config.Kd},
            motrix_max_iterations=cfg.motrix_max_iterations,
        )
        self._terrain_surface_sampler = getattr(backend, "terrain_surface_sampler", None)
        terrain_origins = getattr(backend, "terrain_origins", None)
        if terrain_origins is not None:
            self._scene_terrain_origins = terrain_origins
        super().__init__(cfg, backend, num_envs)
        self._enable_reward_log = True
        self._reward_cfg = cfg.reward_config
        self._init_reward_functions()
        if self._scene_terrain_origins is not None and terrain_generator is not None:
            self._spawn = TerrainSpawnManager(
                num_envs,
                self._scene_terrain_origins,
                cell_size=float(terrain_generator.size[0]),
                cfg=getattr(cfg, "terrain_curriculum", TerrainCurriculumCfg()),
                terrain_surface_sampler=self._terrain_surface_sampler,
            )
        self.phase = np.zeros((num_envs,), dtype=np.float32)
        self.feet_phase = np.zeros((num_envs, len(cfg.sensor.feet_force)), dtype=np.float32)
        self.gait_frequency = 2
        self.feet_force = np.zeros((num_envs, len(cfg.sensor.feet_force), 3), dtype=np.float32)
        self._init_domain_randomization(Go1JoystickDomainRandomizationProvider())
        self.feet_pos = np.zeros((num_envs, len(cfg.sensor.feet_pos), 3), dtype=np.float32)

    @property
    def obs_groups_spec(self) -> dict[str, int]:
        # gyro(3) + gravity(3) + diff(12) + dof_vel(12) + action(12) + cmd(3) + phase(4) = 49
        return {"obs": 49, "critic": 52}

    def _init_reward_functions(self):
        self._reward_fns: dict[str, Any] = {
            "tracking_lin_vel": rewards.tracking_lin_vel,
            "tracking_ang_vel": rewards.tracking_ang_vel,
            "lin_vel_z": rewards.lin_vel_z,
            "ang_vel_xy": rewards.ang_vel_xy,
            "base_height": rewards.base_height,
            "action_rate": rewards.action_rate,
            "similar_to_default": rewards.similar_to_default,
            "swing_feet_z": self._reward_swing_feet_z,
        }

    def update_state(self, state: NpEnvState) -> NpEnvState:
        self.phase = np.fmod(self.phase + self._cfg.ctrl_dt * self.gait_frequency, 1.0)
        self.feet_phase[:, 0] = self.phase
        self.feet_phase[:, 3] = self.phase

        self.feet_phase[:, 1] = (self.phase + 0.5) % 1
        self.feet_phase[:, 2] = (self.phase + 0.5) % 1

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
        reward = self._compute_reward(state.info, linvel, gyro, dof_pos)
        obs = self._compute_obs(
            state.info, linvel, gyro, gravity, dof_pos, dof_vel, self.feet_phase
        )
        return state.replace(obs=obs, reward=reward, terminated=terminated)

    def _compute_obs(
        self, info: dict, linvel, gyro, gravity, dof_pos, dof_vel, feet_phase
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
            [gyro, -gravity, diff, dof_vel, last_actions, command, feet_phase],
            axis=1,
            dtype=get_global_dtype(),
        )
        critic = np.concatenate([obs, linvel], axis=1, dtype=get_global_dtype())
        return {"obs": obs, "critic": critic}

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
            base_height_target=cfg.base_height_target,
            base_height=self._backend.get_base_pos()[:, 2],
        )
        return rewards.run_reward_dispatch(
            scales=cfg.scales,
            fns=self._reward_fns,
            ctx=ctx,
            info=info,
            enable_log=self._enable_reward_log,
            ctrl_dt=self._cfg.ctrl_dt,
        )

    def _reward_contact(self, ctx: RewardContext) -> np.ndarray:
        contact = self.feet_force[:, :, 2] > 0.1
        res = np.zeros(self.num_envs, dtype=np.float32)
        for i in range(len(self._cfg.sensor.feet_force)):
            is_contact = (self.feet_phase[:, i] < 0.6) | (self.gait_frequency < 1.0e-8)
            res += ~(contact[:, i] ^ is_contact)
        return res

    def _reward_swing_feet_z(self, ctx: RewardContext) -> np.ndarray:
        is_swing = self.feet_phase >= 0.6
        target_height = 0.1
        height_error = np.square(self.feet_pos[:, :, 2] - target_height)
        swing_rew = np.exp(-height_error / 0.01) * is_swing
        reward: np.ndarray = np.sum(swing_rew, axis=1) / len(self._cfg.sensor.feet_pos)
        return reward
