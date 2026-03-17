from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from etils import epath

from unilab.base import registry
from unilab.base.backend import create_backend
from unilab.base.dtype_config import get_global_dtype
from unilab.base.np_env import NpEnvState
from unilab.envs.locomotion.go2.base import Go2BaseCfg, Go2BaseEnv
from unilab.utils.math_utils import np_quat_mul, np_yaw_to_quat


@dataclass
class InitState:
    pos = [0.0, 0.0, 0.42]


@dataclass
class Commands:
    vel_limit = [
        [-0.6, -0.4, -0.8],  # [vx_min, vy_min, vyaw_min]
        [1.0, 0.4, 0.8],  # [vx_max, vy_max, vyaw_max]
    ]


@dataclass
class Domain_Rand:
    # randomize_friction = True
    # friction_range = [0.5, 1.25]
    randomize_base_mass = True
    added_mass_range = [-1.5, 1.5]

    random_com = True
    com_offset_x = [-0.05, 0.05]

    push_robots = True
    push_interval = 750  # step
    max_force = [1, 1, 0.5]


@dataclass
class RewardConfig:
    scales: dict[str, float] = field(
        default_factory=lambda: {
            "tracking_lin_vel": 1.0,
            "tracking_ang_vel": 0.2,
            "lin_vel_z": -5.0,
            "ang_vel_xy": -0.02,
            "base_height": -100.0,
            "action_rate": -0.005,
            "similar_to_default": -0.1,
            "alive": 0.0,
            "foot_lift_reward": 0.2,
            "foot_drag_penalty": -0.0,
        }
    )
    tracking_sigma: float = 0.25
    base_height_target: float = 0.3
    target_foot_height: float = 0.08
    foot_clearance_sigma: float = 0.02


@registry.envcfg("Go2JoystickFlatTerrain")
@dataclass
class Go2JoystickCfg(Go2BaseCfg):
    model_file: str = str(epath.Path(__file__).parent / "xml" / "scene_flat.xml")
    max_episode_seconds: float = 20.0
    init_state: InitState = field(default_factory=InitState)
    commands: Commands = field(default_factory=Commands)
    reward_config: RewardConfig = field(default_factory=RewardConfig)
    domain_rand: Domain_Rand = field(default_factory=Domain_Rand)


@registry.env("Go2JoystickFlatTerrain", sim_backend="mujoco")
@registry.env("Go2JoystickFlatTerrain", sim_backend="motrix")
class Go2WalkTask(Go2BaseEnv):
    _cfg: Go2JoystickCfg

    def __init__(self, cfg: Go2JoystickCfg, num_envs=1, backend_type="mujoco"):
        backend = create_backend(
            backend_type, cfg.model_file, num_envs, cfg.sim_dt, base_name=cfg.asset.base_name
        )
        super().__init__(cfg, backend, num_envs)
        self._enable_reward_log = True
        self._init_reward_functions()

    @property
    def obs_groups_spec(self) -> dict[str, int]:
        # gyro(3) + gravity(3) + diff(12) + dof_vel(12) + action(12) + cmd(3) = 45
        return {"actor": 45, "privileged": 3}

    def _init_reward_functions(self):
        self._reward_fns = {
            "tracking_lin_vel": self._reward_tracking_lin_vel,
            "tracking_ang_vel": self._reward_tracking_ang_vel,
            "lin_vel_z": self._reward_lin_vel_z,
            "ang_vel_xy": self._reward_ang_vel_xy,
            "base_height": self._reward_base_height,
            "action_rate": self._reward_action_rate,
            "similar_to_default": self._reward_similar_to_default,
            "alive": self._reward_alive,
            "foot_lift_reward": self._reward_foot_lift,
            "foot_drag_penalty": self._reward_foot_drag,
        }

    def update_state(self, state: NpEnvState) -> NpEnvState:
        linvel = self.get_local_linvel()
        gyro = self.get_gyro()
        gravity = self._backend.get_sensor_data("upvector")
        dof_pos = self.get_dof_pos()
        dof_vel = self.get_dof_vel()
        terminated = gravity[:, 2] <= 0.5
        reward = self._compute_reward(state.info, linvel, gyro, dof_pos)
        obs = self._compute_obs(state.info, linvel, gyro, gravity, dof_pos, dof_vel)
        return state.replace(obs=obs, reward=reward, terminated=terminated)

    def _compute_obs(
        self, info: dict, linvel, gyro, gravity, dof_pos, dof_vel
    ) -> dict[str, np.ndarray]:
        diff = dof_pos - self.default_angles
        command = info["commands"]
        last_actions = info.get("current_actions", np.zeros_like(diff))
        actor = np.concatenate(
            [gyro, -gravity, diff, dof_vel, last_actions, command],
            axis=1,
            dtype=get_global_dtype(),
        )
        return {"actor": actor, "privileged": linvel}

    def _compute_reward(self, info: dict, linvel, gyro, dof_pos) -> np.ndarray:
        dtype = get_global_dtype()
        reward = np.zeros((self._num_envs,), dtype=dtype)
        cfg = self._cfg.reward_config

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
        return np.asarray(np.exp(-lin_vel_error / self._cfg.reward_config.tracking_sigma))

    def _reward_tracking_ang_vel(self, info: dict, linvel, gyro, dof_pos) -> np.ndarray:
        commands = info["commands"]
        ang_vel_error = np.square(commands[:, 2] - gyro[:, 2])
        return np.asarray(np.exp(-ang_vel_error / self._cfg.reward_config.tracking_sigma))

    def _reward_lin_vel_z(self, info: dict, linvel, gyro, dof_pos) -> np.ndarray:
        return np.asarray(np.square(linvel[:, 2]))

    def _reward_ang_vel_xy(self, info: dict, linvel, gyro, dof_pos) -> np.ndarray:
        return np.asarray(np.sum(np.square(gyro[:, :2]), axis=1))

    def _reward_base_height(self, info: dict, linvel, gyro, dof_pos) -> np.ndarray:
        base_height = self._backend.get_base_pos()[:, 2]
        return np.asarray(np.square(base_height - self._cfg.reward_config.base_height_target))

    def _reward_action_rate(self, info: dict, linvel, gyro, dof_pos) -> np.ndarray:
        action_diff = info["current_actions"] - info["last_actions"]
        return np.asarray(np.sum(np.square(action_diff), axis=1))

    def _reward_similar_to_default(self, info: dict, linvel, gyro, dof_pos) -> np.ndarray:
        return np.asarray(np.sum(np.abs(dof_pos - self.default_angles), axis=1))

    def _reward_alive(self, info: dict, linvel, gyro, dof_pos) -> np.ndarray:
        return np.ones((self._num_envs,), dtype=get_global_dtype())

    def _reward_foot_lift(self, info: dict, linvel, gyro, dof_pos) -> np.ndarray:
        foot_pos = self.get_foot_pos()
        foot_heights = foot_pos[..., 2]
        foot_contact = self.get_foot_contact()
        is_swing = foot_contact < 0.5
        target_height = self._cfg.reward_config.target_foot_height
        sigma = self._cfg.reward_config.foot_clearance_sigma
        error_sq = np.square(foot_heights - target_height)
        reward = np.exp(-error_sq / sigma) * is_swing
        return np.asarray(np.sum(reward, axis=1))

    def _reward_foot_drag(self, info: dict, linvel, gyro, dof_pos) -> np.ndarray:
        foot_pos = self.get_foot_pos()
        foot_heights = foot_pos[..., 2]
        foot_contact = self.get_foot_contact()
        is_swing = foot_contact < 0.5
        safe_height = self._cfg.reward_config.target_foot_height / 2.0
        height_error = np.clip(safe_height - foot_heights, 0.0, None)
        error = np.square(height_error) * is_swing
        return np.asarray(np.sum(error, axis=1))

    def reset(self, env_indices: np.ndarray):
        num_reset = len(env_indices)
        qpos = np.tile(self._init_qpos, (num_reset, 1))
        qvel = np.tile(self._init_qvel, (num_reset, 1))

        # Domain Randomization
        dxy = np.random.uniform(-0.5, 0.5, (num_reset, 2))
        qpos[:, 0:2] += dxy
        yaw = np.random.uniform(-np.pi, np.pi, (num_reset,))
        quat_yaw = np_yaw_to_quat(yaw)
        qpos[:, 3:7] = np_quat_mul(qpos[:, 3:7], quat_yaw)
        qvel[:, 0:6] = np.random.uniform(-0.5, 0.5, (num_reset, 6))

        self._backend.set_state(env_indices, qpos, qvel)

        commands = np.random.uniform(
            low=self._cfg.commands.vel_limit[0],
            high=self._cfg.commands.vel_limit[1],
            size=(num_reset, 3),
        )

        info = {
            "commands": commands,
            "current_actions": np.zeros((num_reset, self._num_action), dtype=get_global_dtype()),
            "last_actions": np.zeros((num_reset, self._num_action), dtype=get_global_dtype()),
        }

        linvel = self.get_local_linvel()[env_indices]
        gyro = self.get_gyro()[env_indices]
        gravity = self._backend.get_sensor_data("upvector")[env_indices]
        dof_pos = self.get_dof_pos()[env_indices]
        dof_vel = self.get_dof_vel()[env_indices]
        obs = self._compute_obs(info, linvel, gyro, gravity, dof_pos, dof_vel)
        return obs, obs, info
