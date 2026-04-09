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
from unilab.dr import (
    DomainRandomizationCapabilities,
    DomainRandomizationProvider,
    IntervalRandomizationPlan,
    ResetPlan,
)
from unilab.dr.dr_utils import (
    build_common_reset_randomization,
    build_interval_push_plan,
    validate_common_reset_randomization,
    validate_interval_push_support,
    zero_actions,
)
from unilab.envs.locomotion.common.commands import Commands
from unilab.envs.locomotion.common.domain_rand import DomainRandConfig
from unilab.envs.locomotion.go2.base import Go2BaseCfg, Go2BaseEnv
from unilab.utils.math_utils import np_quat_mul, np_yaw_to_quat


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
class LegacyMotrixProfile:
    enabled: bool = False
    command_vel_limit: list[list[float]] | None = None


@dataclass
class JoystickSensor:
    local_linvel = "local_linvel"
    gyro = "gyro"
    feet_force = ["FL_foot_contact", "FR_foot_contact", "RL_foot_contact", "RR_foot_contact"]
    feet_pos = ["FL_pos", "FR_pos", "RL_pos", "RR_pos"]


@registry.envcfg("Go2JoystickFlatTerrain")
@dataclass
class Go2JoystickCfg(Go2BaseCfg):
    model_file: str = str(ASSETS_ROOT_PATH / "robots" / "go2" / "scene_flat.xml")
    max_episode_seconds: float = 20.0
    init_state: InitState = field(default_factory=InitState)
    commands: Commands = field(default_factory=Commands)
    reward_config: RewardConfig | None = None
    legacy_motrix_profile: LegacyMotrixProfile = field(default_factory=LegacyMotrixProfile)
    sensor: JoystickSensor = field(default_factory=JoystickSensor)  # type: ignore[assignment]
    domain_rand: Go2DomainRandConfig = field(default_factory=Go2DomainRandConfig)

    def apply_legacy_motrix_profile(self) -> None:
        profile = self.legacy_motrix_profile
        if not profile.enabled:
            return
        if profile.command_vel_limit is not None:
            self.commands.vel_limit = [list(v) for v in profile.command_vel_limit]


class Go2JoystickDomainRandomizationProvider(DomainRandomizationProvider):
    def validate(self, env: Any, capabilities: DomainRandomizationCapabilities) -> None:
        validate_common_reset_randomization(env, capabilities)
        validate_interval_push_support(env, capabilities)

    def build_interval_randomization_plan(
        self, env: Any, step_counter: int
    ) -> IntervalRandomizationPlan | None:
        return build_interval_push_plan(env, step_counter)

    def _sample_commands(self, env: Any, num_reset: int) -> np.ndarray:
        low = np.asarray(env.cfg.commands.vel_limit[0], dtype=get_global_dtype())
        high = np.asarray(env.cfg.commands.vel_limit[1], dtype=get_global_dtype())
        return np.asarray(
            np.random.uniform(low=low, high=high, size=(num_reset, 3)), dtype=get_global_dtype()
        )

    def build_reset_plan(self, env: Any, env_ids: np.ndarray) -> ResetPlan:
        num_reset = len(env_ids)
        qpos = np.tile(env._init_qpos, (num_reset, 1))
        qvel = np.tile(env._init_qvel, (num_reset, 1))
        qpos[:, 0:2] += np.random.uniform(-0.5, 0.5, (num_reset, 2))
        yaw = np.random.uniform(-np.pi, np.pi, (num_reset,))
        qpos[:, 3:7] = np_quat_mul(qpos[:, 3:7], np_yaw_to_quat(yaw))
        qvel[:, 0:6] = np.random.uniform(-0.5, 0.5, (num_reset, 6))
        info_updates = {
            "commands": self._sample_commands(env, num_reset),
            "current_actions": zero_actions(num_reset, env._num_action),
            "last_actions": zero_actions(num_reset, env._num_action),
        }
        return ResetPlan(
            env_ids=env_ids,
            qpos=qpos,
            qvel=qvel,
            info_updates=info_updates,
            randomization=build_common_reset_randomization(env, num_reset),
        )

    def build_reset_observation(
        self, env: Any, env_ids: np.ndarray, info_updates: dict[str, Any]
    ) -> dict[str, np.ndarray]:
        linvel = env.get_local_linvel()[env_ids]
        gyro = env.get_gyro()[env_ids]
        gravity = env._backend.get_sensor_data("upvector")[env_ids]
        dof_pos = env.get_dof_pos()[env_ids]
        dof_vel = env.get_dof_vel()[env_ids]
        return cast(
            dict[str, np.ndarray],
            env._compute_obs(
                info_updates, linvel, gyro, gravity, dof_pos, dof_vel, env.feet_phase[env_ids]
            ),
        )


@registry.env("Go2JoystickFlatTerrain", sim_backend="mujoco")
@registry.env("Go2JoystickFlatTerrain", sim_backend="motrix")
class Go2WalkTask(Go2BaseEnv):
    _cfg: Go2JoystickCfg

    def __init__(self, cfg: Go2JoystickCfg, num_envs=1, backend_type="mujoco"):
        if cfg.reward_config is None:
            raise ValueError("reward_config must be provided via Hydra configuration")
        cfg.apply_legacy_motrix_profile()
        backend = create_backend(
            backend_type,
            cfg.model_file,
            num_envs,
            cfg.sim_dt,
            base_name=cfg.asset.base_name,
            position_actuator_gains={"kp": cfg.control_config.Kp, "kd": cfg.control_config.Kd},
        )
        super().__init__(cfg, backend, num_envs)
        self._enable_reward_log = True
        self._reward_cfg = cfg.reward_config
        self._init_reward_functions()
        self._init_domain_randomization(Go2JoystickDomainRandomizationProvider())
        self.phase = np.zeros((num_envs,), dtype=np.float32)
        self.feet_phase = np.zeros((num_envs, len(cfg.sensor.feet_force)), dtype=np.float32)
        self.gait_frequency = 2
        self.feet_force = np.zeros((num_envs, len(cfg.sensor.feet_force), 3), dtype=np.float32)
        self.feet_pos = np.zeros((num_envs, len(cfg.sensor.feet_pos), 3), dtype=np.float32)

    @property
    def obs_groups_spec(self) -> dict[str, int]:
        # gyro(3) + gravity(3) + diff(12) + dof_vel(12) + action(12) + cmd(3) + phase(4) = 49
        return {"obs": 49, "privileged": 3}

    def _init_reward_functions(self):
        self._reward_fns = {
            "tracking_lin_vel": self._reward_tracking_lin_vel,
            "tracking_ang_vel": self._reward_tracking_ang_vel,
            "lin_vel_z": self._reward_lin_vel_z,
            "ang_vel_xy": self._reward_ang_vel_xy,
            "base_height": self._reward_base_height,
            "action_rate": self._reward_action_rate,
            "similar_to_default": self._reward_similar_to_default,
            # "alive": self._reward_alive,
            "swing_feet_z": self._reward_swing_feet_z,
            "contact": self._reward_contact,
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
        diff = dof_pos - self.default_angles
        command = info["commands"]
        last_actions = info.get("current_actions", np.zeros_like(diff))
        obs = np.concatenate(
            [gyro, -gravity, diff, dof_vel, last_actions, command, feet_phase],
            axis=1,
            dtype=get_global_dtype(),
        )
        return {"obs": obs, "privileged": linvel}

    def _compute_reward(self, info: dict, linvel, gyro, dof_pos) -> np.ndarray:
        dtype = get_global_dtype()
        reward = np.zeros((self._num_envs,), dtype=dtype)
        cfg = self._reward_cfg

        step_count = info.get("steps", np.zeros((self._num_envs,), dtype=np.uint32))
        should_log = self._enable_reward_log and (int(step_count[0]) % 4 == 0)
        log = {} if should_log else info.get("log", {})

        for name, scale in cfg.scales.items():
            if scale == 0 or name not in self._reward_fns:
                continue
            rew = self._reward_fns[name](info, linvel, gyro, dof_pos)
            weighted_rew = rew * scale
            reward += weighted_rew
            if should_log:
                log[f"reward/{name}"] = float(np.mean(weighted_rew))

        info["log"] = log
        return reward * self._cfg.ctrl_dt

    # ── reward functions ──────────────────────────────────────────────

    def _reward_tracking_lin_vel(self, info: dict, linvel, gyro, dof_pos) -> np.ndarray:
        commands = info["commands"]
        lin_vel_error = np.sum(np.square(commands[:, :2] - linvel[:, :2]), axis=1)
        return np.asarray(np.exp(-lin_vel_error / self._reward_cfg.tracking_sigma))

    def _reward_tracking_ang_vel(self, info: dict, linvel, gyro, dof_pos) -> np.ndarray:
        commands = info["commands"]
        ang_vel_error = np.square(commands[:, 2] - gyro[:, 2])
        return np.asarray(np.exp(-ang_vel_error / self._reward_cfg.tracking_sigma))

    def _reward_lin_vel_z(self, info: dict, linvel, gyro, dof_pos) -> np.ndarray:
        return np.asarray(np.square(linvel[:, 2]))

    def _reward_ang_vel_xy(self, info: dict, linvel, gyro, dof_pos) -> np.ndarray:
        return np.asarray(np.sum(np.square(gyro[:, :2]), axis=1))

    def _reward_base_height(self, info: dict, linvel, gyro, dof_pos) -> np.ndarray:
        base_height = self._backend.get_base_pos()[:, 2]
        return np.asarray(np.square(base_height - self._reward_cfg.base_height_target))

    def _reward_action_rate(self, info: dict, linvel, gyro, dof_pos) -> np.ndarray:
        action_diff = info["current_actions"] - info["last_actions"]
        return np.asarray(np.sum(np.square(action_diff), axis=1))

    def _reward_similar_to_default(self, info: dict, linvel, gyro, dof_pos) -> np.ndarray:
        # print(self.default_angles)
        return np.asarray(np.sum(np.abs(dof_pos - self.default_angles), axis=1))

    def _reward_alive(self, info: dict, linvel, gyro, dof_pos) -> np.ndarray:
        return np.ones((self._num_envs,), dtype=get_global_dtype())

    def _reward_swing_feet_z(self, info: dict, linvel, gyro, dof_pos) -> np.ndarray:
        is_swing = self.feet_phase >= 0.6
        target_height = 0.1
        height_error = np.square(self.feet_pos[:, :, 2] - target_height)
        swing_rew = np.exp(-height_error / 0.01) * is_swing
        return np.asarray(np.sum(swing_rew, axis=1) / len(self._cfg.sensor.feet_pos))

    def _reward_foot_drag(self, info: dict, linvel, gyro, dof_pos) -> np.ndarray:
        foot_pos = self.get_foot_pos()
        foot_heights = foot_pos[..., 2]
        foot_contact = self.get_foot_contact()
        is_swing = foot_contact < 0.5
        safe_height = self._reward_cfg.target_foot_height / 2.0
        height_error = np.clip(safe_height - foot_heights, 0.0, None)
        error = np.square(height_error) * is_swing
        return np.asarray(np.sum(error, axis=1))

    def _reward_contact(self, info: dict, linvel, gyro, dof_pos) -> np.ndarray:
        contact = self.feet_force[:, :, 2] > 0.1
        res = np.zeros(self._num_envs, dtype=np.float32)
        for i in range(len(self._cfg.sensor.feet_force)):
            is_contact = (self.feet_phase[:, i] < 0.6) | (self.gait_frequency < 1.0e-8)
            res += (contact[:, i] == is_contact).astype(np.float32)
        return res / len(self._cfg.sensor.feet_force)
