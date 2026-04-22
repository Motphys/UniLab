from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, cast

import numpy as np
from etils import epath

from unilab.assets import ASSETS_ROOT_PATH
from unilab.base import registry
from unilab.base.backend import create_backend
from unilab.base.dtype_config import get_global_dtype
from unilab.base.np_env import NpEnvState
from unilab.envs.locomotion.common import rewards
from unilab.envs.locomotion.common.commands import Commands
from unilab.envs.locomotion.common.domain_rand import DomainRandConfig
from unilab.envs.locomotion.common.dr_provider import LocomotionDRProvider
from unilab.envs.locomotion.common.rewards import RewardContext
from unilab.envs.locomotion.go2.base import Go2BaseCfg, Go2BaseEnv


@dataclass
class InitState:
    pos = [0.0, 0.0, 0.42]


@dataclass
class Go2DomainRandConfig(DomainRandConfig):
    randomize_kp: bool = True
    kp_multiplier_range: list[float] = field(default_factory=lambda: [0.9, 1.1])

    randomize_kd: bool = True
    kd_multiplier_range: list[float] = field(default_factory=lambda: [0.9, 1.1])


@dataclass
class RewardConfig:
    scales: dict[str, float]
    tracking_sigma: float
    base_height_target: float
    target_foot_height: float = 0.1


@dataclass
class JoystickSensor:
    local_linvel = "local_linvel"
    gyro = "gyro"
    feet_force = ["FL_foot_contact", "FR_foot_contact", "RL_foot_contact", "RR_foot_contact"]
    feet_pos = ["FL_pos", "FR_pos", "RL_pos", "RR_pos"]
    global_pos = "global_position"
    ternamate_contact = [
        "base1_contact",
        "base2_contact",
        "base3_contact",
        "FL_hip_contact",
        "FR_hip_contact",
        "FL_thigh_contact",
        "FR_thigh_contact",
        "FL_calf_contact1",
        "FL_calf_contact2",
        "FR_calf_contact1",
        "FR_calf_contact2",
    ]
    penalty_contact = [
        "RL_hip_contact",
        "RR_hip_contact",
        "RL_thigh_contact",
        "RR_thigh_contact",
        "RL_calf_contact1",
        "RL_calf_contact2",
        "RR_calf_contact1",
        "RR_calf_contact2",
    ]


@registry.envcfg("Go2HandStand")
@dataclass
class Go2HandStandCfg(Go2BaseCfg):
    model_file: str = str(ASSETS_ROOT_PATH / "robots" / "go2" / "scene_flat.xml")
    max_episode_seconds: float = 20.0
    init_state: InitState = field(default_factory=InitState)
    commands: Commands = field(default_factory=Commands)
    reward_config: RewardConfig | None = None
    sensor: JoystickSensor = field(default_factory=JoystickSensor)  # type: ignore[assignment]
    domain_rand: Go2DomainRandConfig = field(default_factory=Go2DomainRandConfig)


class Go2HandStandDomainRandomizationProvider(LocomotionDRProvider):
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
        height = env.torso_height[env_ids].reshape(-1, 1)
        return env._compute_obs(  # type: ignore[no-any-return]
            info_updates, linvel, gyro, gravity, dof_pos, dof_vel, height
        )


@registry.env("Go2HandStand", sim_backend="mujoco")
@registry.env("Go2HandStand", sim_backend="motrix")
class Go2HandStandTask(Go2BaseEnv):
    _cfg: Go2HandStandCfg

    def __init__(self, cfg: Go2HandStandCfg, num_envs=1, backend_type="mujoco"):
        if cfg.reward_config is None:
            raise ValueError("reward_config must be provided via Hydra configuration")
        backend = create_backend(
            backend_type,
            cfg.model_file,
            num_envs,
            cfg.sim_dt,
            base_name=cfg.asset.base_name,
            position_actuator_gains={"kp": cfg.control_config.Kp, "kd": cfg.control_config.Kd},
            iterations=cfg.iterations,
        )
        super().__init__(cfg, backend, num_envs)
        self._enable_reward_log = True
        self._reward_cfg = cfg.reward_config
        self._init_reward_functions()
        self._init_domain_randomization(Go2HandStandDomainRandomizationProvider())
        self.phase = np.zeros((num_envs,), dtype=np.float32)
        self.feet_phase = np.zeros((num_envs, len(cfg.sensor.feet_force)), dtype=np.float32)
        self.gait_frequency = 2
        self.feet_force = np.zeros((num_envs, len(cfg.sensor.feet_force), 1), dtype=np.float32)
        self.feet_pos = np.zeros((num_envs, len(cfg.sensor.feet_pos), 3), dtype=np.float32)
        self.torso_height = np.zeros((num_envs,), dtype=np.float32)
        self._z_des = 0.55
        self._desired_gravity = np.array([-1, 0, 0])
        self.feet_geom_names = [0, 1]
        self._joint_ids = [0, 1, 2, 3, 4, 5, 6, 9]
        self._tar_ids = [6, 7, 8, 9, 10, 11]
        self.target_angle = np.array([0, 1.82, -1.16, 0.0, 1.82, -1.16])

    @property
    def obs_groups_spec(self) -> dict[str, int]:
        # gyro(3) + gravity(3) + diff(12) + dof_vel(12) + action(12)  = 42
        return {"obs": 42, "critic": 46}

    def _init_reward_functions(self):
        self._reward_fns: dict[str, Any] = {
            "height": self._reward_height,
            "contact": self._cost_contact,
            "oritentation": self._reward_orientation,
            "pose": self._cost_pose,
            "penalty_contact": self._reward_penalty_contact,
            "action_rate": rewards.action_rate,
            "tar": self._reward_tar,
        }

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
        self.torso_height = self._backend.get_sensor_data(self._cfg.sensor.global_pos)[:, -1]
        contact_arrays = []
        for name in self._cfg.sensor.ternamate_contact:
            arr = self._backend.get_sensor_data(name)
            contact_arrays.append(arr)
        result = np.concatenate(contact_arrays, axis=1)

        terminated_z = gravity[:, 2] <= -0.25
        terminated_contact = np.any(result, axis=1)
        terminated = np.logical_or(terminated_contact, terminated_z)
        reward = self._compute_reward(state.info, linvel, gyro, dof_pos)
        obs = self._compute_obs(
            state.info, linvel, gyro, gravity, dof_pos, dof_vel, self.torso_height.reshape(-1, 1)
        )
        return state.replace(obs=obs, reward=reward, terminated=terminated)

    def _compute_obs(
        self, info: dict, linvel, gyro, gravity, dof_pos, dof_vel, height
    ) -> dict[str, np.ndarray]:
        noise_cfg = self._cfg.noise_config
        diff = dof_pos - self.default_angles
        gyro = self._obs_noise(gyro, noise_cfg.scale_gyro)
        gravity = self._obs_noise(gravity, noise_cfg.scale_gravity)
        diff = self._obs_noise(diff, noise_cfg.scale_joint_angle)
        dof_vel = self._obs_noise(dof_vel, noise_cfg.scale_joint_vel)
        linvel = self._obs_noise(linvel, noise_cfg.scale_linvel)
        # command = info["commands"]
        last_actions = info.get("current_actions", np.zeros_like(diff))
        obs = np.concatenate(
            [gyro, -gravity, diff, dof_vel, last_actions],
            axis=1,
            dtype=get_global_dtype(),
        )
        critic = np.concatenate([obs, linvel, height], axis=1, dtype=get_global_dtype())
        return {"obs": obs, "critic": critic}

    # state = jp.hstack([
    #     noisy_linvel,
    #     noisy_gyro,
    #     noisy_gravity,
    #     noisy_joint_angles - self._default_pose,
    #     noisy_joint_vel,
    #     info["last_act"],
    # ])
    # privileged_state = jp.hstack([ TODO
    #     state,
    #     gyro,
    #     accelerometer,
    #     linvel,
    #     angvel,
    #     joint_angles,
    #     joint_vel,
    #     data.actuator_force,
    #     torso_height,
    # ])

    def _compute_reward(self, info: dict, linvel, gyro, dof_pos) -> np.ndarray:
        dtype = get_global_dtype()
        reward = np.zeros((self._num_envs,), dtype=dtype)
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

        step_count = info.get("steps", np.zeros((self._num_envs,), dtype=np.uint32))
        should_log = self._enable_reward_log and (int(step_count[0]) % 4 == 0)
        log = {} if should_log else info.get("log", {})

        for name, scale in cfg.scales.items():
            if scale == 0 or name not in self._reward_fns:
                continue
            rew = self._reward_fns[name](ctx)
            weighted_rew = rew * scale
            reward += weighted_rew
            if should_log:
                log[f"reward/{name}"] = float(np.mean(weighted_rew))

        info["log"] = log
        return reward * self._cfg.ctrl_dt

    # ── reward functions (robot-specific) ────────────────────────────

    def _reward_swing_feet_z(self, ctx: RewardContext) -> np.ndarray:
        is_swing = self.feet_phase >= 0.6
        target_height = 0.1
        height_error = np.square(self.feet_pos[:, :, 2] - target_height)
        swing_rew = np.exp(-height_error / 0.01) * is_swing
        reward: np.ndarray = np.sum(swing_rew, axis=1) / len(self._cfg.sensor.feet_pos)
        return reward

    def _reward_foot_drag(self, ctx: RewardContext) -> np.ndarray:
        foot_pos = self.get_foot_pos()
        foot_heights = foot_pos[..., 2]
        foot_contact = self.get_foot_contact()
        is_swing = foot_contact < 0.5
        safe_height = self._reward_cfg.target_foot_height / 2.0
        height_error = np.clip(safe_height - foot_heights, 0.0, None)
        error = np.square(height_error) * is_swing
        drag_penalty: np.ndarray = np.sum(error, axis=1)
        return drag_penalty

    def _reward_penalty_contact(self, ctx: RewardContext) -> np.ndarray:
        contact_arrays = []
        for name in self._cfg.sensor.penalty_contact:
            arr = self._backend.get_sensor_data(name)
            contact_arrays.append(arr)
        result = np.concatenate(contact_arrays, axis=1)
        return np.asarray(np.any(result, axis=1))

    def _reward_contact(self, ctx: RewardContext) -> np.ndarray:
        contact = self.feet_force[:, :, 2] > 0.1
        res = np.zeros(self._num_envs, dtype=np.float32)
        for i in range(len(self._cfg.sensor.feet_force)):
            is_contact = (self.feet_phase[:, i] < 0.6) | (self.gait_frequency < 1.0e-8)
            res += (contact[:, i] == is_contact).astype(np.float32)
        return res / len(self._cfg.sensor.feet_force)

    def _reward_height(self, ctx: RewardContext) -> np.ndarray:
        height = np.minimum(self.torso_height, self._z_des)
        error = self._z_des - height
        return np.exp(-error / 0.25)

    def _reward_orientation(self, ctx: RewardContext) -> np.ndarray:
        gravity = -1 * self._backend.get_sensor_data("upvector")
        cos_dist = gravity @ self._desired_gravity
        normalized = 0.5 * cos_dist + 0.5
        return np.asarray(np.square(normalized))

    def _cost_contact(self, ctx: RewardContext) -> np.ndarray:
        feet_contact = self.feet_force[:, self.feet_geom_names, :]
        return np.asarray(np.any(feet_contact, axis=1).squeeze())

    # def _cost_pose(self, ctx: RewardContext) -> np.ndarray:
    #     dof_pos = self.get_dof_pos()
    #     error = dof_pos[:, self._joint_ids] - self.default_angles[self._joint_ids]
    #     return np.sum(np.square(error), axis=1)
    def _cost_pose(self, ctx: RewardContext) -> np.ndarray:
        dof_pos = self.get_dof_pos()
        error = dof_pos[:, self._joint_ids] - self.default_angles[self._joint_ids]
        # 告诉 Mypy：把它当成 np.ndarray 看待
        return cast(np.ndarray, np.sum(np.square(error), axis=1))
    def _reward_tar(self, ctx: RewardContext) -> np.ndarray:
        dof_pos = self.get_dof_pos()
        error = dof_pos[:, self._tar_ids] - self.target_angle
        error = np.sum(np.square(error), axis=1)
        mask = (self.torso_height >= self._z_des * 0.8).astype(np.float32)
        return cast(np.ndarray, np.exp(-error / 1) * mask)

    # def _cost_pose(self, qpos: jax.Array) -> jax.Array:
    # return jp.sum(jp.square(qpos[self._joint_ids] - self._joint_pose))
