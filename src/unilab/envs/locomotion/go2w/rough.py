from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from unilab.assets import ASSETS_ROOT_PATH
from unilab.base import registry
from unilab.base.np_env import NpEnvState
from unilab.base.scene import SceneCfg, TerrainSceneCfg
from unilab.dr import DomainRandomizationManager, ResetPlan
from unilab.dr.dr_utils import zero_actions
from unilab.dtype_config import get_global_dtype
from unilab.envs.common.rotation import np_quat_from_euler_xyz, np_quat_mul
from unilab.envs.locomotion.common import rewards
from unilab.envs.locomotion.common.commands import Commands
from unilab.envs.locomotion.common.height_scan import (
    HeightScanConfig,
    base_height_from_scan,
    height_scan_obs,
    init_height_scan_sensor,
    raw_height_scan_obs,
    terrain_out_of_bounds,
)
from unilab.envs.locomotion.common.rewards import RewardContext
from unilab.envs.locomotion.common.terrain_spawn import (
    TerrainCurriculumCfg,
    TerrainSpawnManager,
)
from unilab.envs.locomotion.go2.rough import Go2RoughTerrainCfg, RoughTerminationConfig
from unilab.envs.locomotion.go2w.base import NUM_GO2W_ACTIONS, NUM_LEG_ACTIONS
from unilab.envs.locomotion.go2w.joystick import (
    Go2WJoystickCfg,
    Go2WJoystickDomainRandomizationProvider,
    Go2WJoystickEnv,
    build_go2w_backend_reset_randomization,
    sample_go2w_heading_commands,
    wrap_to_pi_np,
    yaw_from_quat_np,
)

# pyright: reportIncompatibleVariableOverride=false, reportAttributeAccessIssue=false, reportCallIssue=false


GO2W_HEIGHT_SCAN_SCALE = 5.0


@dataclass
class TerrainScanConfig(HeightScanConfig):
    """Backward-compatible alias used by Go2W rough yaml configs."""

    hfield_name: str = "terrain_hfield"
    geom_name: str = "floor"
    scale: float = GO2W_HEIGHT_SCAN_SCALE


@dataclass
class Go2WRoughCommands(Commands):
    vel_limit: list[list[float]] = field(
        default_factory=lambda: [[-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]]
    )
    resampling_time: float = 10.0
    heading_command: bool = True
    heading_range: list[float] = field(default_factory=lambda: [-np.pi, np.pi])


@registry.envcfg("Go2WJoystickRough")
@dataclass
class Go2WJoystickRoughCfg(Go2WJoystickCfg):
    """Go2W rough terrain task with procedurally generated sub-terrains."""

    scene: SceneCfg = field(
        default_factory=lambda: SceneCfg(
            model_file=str(ASSETS_ROOT_PATH / "robots" / "go2w" / "go2w.xml"),
            terrain=TerrainSceneCfg(
                generator=Go2RoughTerrainCfg(),
                hfield_name="terrain_hfield",
                geom_name="floor",
            ),
        )
    )
    commands: Go2WRoughCommands = field(default_factory=Go2WRoughCommands)
    terrain_scan: TerrainScanConfig = field(default_factory=TerrainScanConfig)
    termination_config: RoughTerminationConfig = field(default_factory=RoughTerminationConfig)
    terrain_curriculum: TerrainCurriculumCfg = field(default_factory=TerrainCurriculumCfg)


class Go2WJoystickRoughDomainRandomizationProvider(Go2WJoystickDomainRandomizationProvider):
    def _sample_commands(self, env: Any, num_reset: int) -> np.ndarray:
        low = np.asarray(env.cfg.commands.vel_limit[0], dtype=get_global_dtype())
        high = np.asarray(env.cfg.commands.vel_limit[1], dtype=get_global_dtype())
        commands = np.asarray(
            np.random.uniform(low=low, high=high, size=(num_reset, 3)),
            dtype=get_global_dtype(),
        )
        _zero_small_xy_commands(commands)
        if env.cfg.commands.heading_command:
            commands[:, 2] = 0.0
        return commands

    def build_reset_plan(self, env: Any, env_ids: np.ndarray) -> ResetPlan:
        num_reset = len(env_ids)
        qpos = np.tile(env._init_qpos, (num_reset, 1))
        qvel = np.tile(env._init_qvel, (num_reset, 1))
        qpos[:, 0:2] += np.random.uniform(-0.5, 0.5, (num_reset, 2))
        qpos[:, 2] += np.random.uniform(0.25, 0.5, (num_reset,))
        qpos[:, 0:3] += env._spawn.origins_for(env_ids)
        roll = np.random.uniform(-3.14, 3.14, (num_reset,))
        pitch = np.random.uniform(-3.14, 3.14, (num_reset,))
        yaw = np.random.uniform(-3.14, 3.14, (num_reset,))
        qpos[:, 3:7] = np_quat_mul(qpos[:, 3:7], np_quat_from_euler_xyz(roll, pitch, yaw))
        qvel[:, 0:6] = np.asarray(
            np.random.uniform(-0.5, 0.5, size=(num_reset, 6)), dtype=get_global_dtype()
        )

        motor_kp, motor_kd = env.sample_reset_motor_gains(num_reset)
        env.set_motor_gains(env_ids, motor_kp, motor_kd)
        commands = self._sample_commands(env, num_reset)
        info_updates: dict[str, Any] = {
            "commands": commands,
            "current_actions": zero_actions(num_reset, env._num_action),
            "last_actions": zero_actions(num_reset, env._num_action),
            "motor_kp": motor_kp.astype(get_global_dtype()),
            "motor_kd": motor_kd.astype(get_global_dtype()),
            "torques": np.zeros((num_reset, env._num_action), dtype=get_global_dtype()),
        }
        if getattr(env.cfg.commands, "heading_command", False):
            info_updates["heading_commands"] = sample_go2w_heading_commands(env, num_reset)
        return ResetPlan(
            env_ids=env_ids,
            qpos=qpos,
            qvel=qvel,
            info_updates=info_updates,
            randomization=build_go2w_backend_reset_randomization(env, num_reset),
        )


@registry.env("Go2WJoystickRough", sim_backend="mujoco")
class Go2WJoystickRoughEnv(Go2WJoystickEnv):
    _cfg: Go2WJoystickRoughCfg

    def __init__(self, cfg: Go2WJoystickRoughCfg, num_envs=1, backend_type="mujoco"):
        super().__init__(cfg, num_envs=num_envs, backend_type=backend_type)
        terrain_origins = getattr(self._backend, "terrain_origins", None)
        terrain_generator = cfg.scene.terrain.generator if cfg.scene.terrain is not None else None
        if terrain_origins is not None and terrain_generator is not None:
            self._spawn = TerrainSpawnManager(
                num_envs,
                terrain_origins,
                cell_size=float(terrain_generator.size[0]),
                cfg=cfg.terrain_curriculum,
                terrain_surface_sampler=getattr(self._backend, "terrain_surface_sampler", None),
            )
        self._dr_manager = DomainRandomizationManager(
            self, Go2WJoystickRoughDomainRandomizationProvider()
        )
        init_height_scan_sensor(self, cfg.terrain_scan, cfg.asset.base_name)

    @property
    def obs_groups_spec(self) -> dict[str, int]:
        return {"obs": 53, "critic": 56 + self._height_scan_dim}

    def _init_reward_functions(self) -> None:
        def gated(fn):
            return lambda ctx: fn(ctx) * self._upright_scale(ctx.gravity)

        def _joint_pos_penalty(ctx: RewardContext) -> np.ndarray:
            return self._reward_joint_pos_penalty(ctx) * self._upright_scale(ctx.gravity)

        def _stand_still(ctx: RewardContext) -> np.ndarray:
            return self._reward_stand_still(ctx) * self._upright_scale(ctx.gravity)

        self._reward_fns = {
            "tracking_lin_vel": gated(rewards.tracking_lin_vel),
            "tracking_ang_vel": gated(rewards.tracking_ang_vel),
            "lin_vel_z": gated(rewards.lin_vel_z),
            "ang_vel_xy": gated(rewards.ang_vel_xy),
            "base_height": gated(rewards.base_height),
            "orientation": gated(rewards.orientation),
            "similar_to_default": gated(rewards.similar_to_default),
            "torques": gated(self._reward_torques_l2),
            "joint_torques_l2": gated(self._reward_joint_torques_l2),
            "energy": gated(rewards.energy),
            "dof_vel": gated(self._reward_dof_vel),
            "dof_acc": gated(self._reward_dof_acc),
            "joint_acc_l2": gated(self._reward_dof_acc),
            "wheel_acc": gated(self._reward_wheel_acc),
            "joint_acc_wheel_l2": gated(self._reward_wheel_acc),
            "joint_pos_limits": gated(rewards.joint_pos_limits),
            "stand_still": _stand_still,
            "hip_pos": gated(self._reward_hip_pos),
            "dof_error": gated(self._reward_dof_error),
            "joint_pos_penalty": _joint_pos_penalty,
            "joint_power": gated(self._reward_joint_power),
            "joint_mirror": gated(self._reward_joint_mirror),
            "alive": rewards.alive,
            "upward": rewards.upward,
            "wheel_vel": gated(self._reward_wheel_vel),
            "action_rate": rewards.action_rate,
        }

    def _upright_scale(self, gravity: np.ndarray | None) -> np.ndarray:
        return rewards.upright_scale(gravity, self._num_envs)

    def _compute_obs(
        self,
        info: dict,
        linvel: np.ndarray,
        gyro: np.ndarray,
        gravity: np.ndarray,
        dof_pos: np.ndarray,
        dof_vel: np.ndarray,
    ) -> dict[str, np.ndarray]:
        noise_cfg = self._cfg.noise_config
        leg_diff = dof_pos[:, :NUM_LEG_ACTIONS] - self.default_angles[:NUM_LEG_ACTIONS]
        policy_gyro = self._obs_noise(gyro, noise_cfg.scale_gyro) * 0.25
        policy_gravity = self._obs_noise(-gravity, noise_cfg.scale_gravity)
        policy_leg_diff = self._obs_noise(leg_diff, noise_cfg.scale_joint_angle)
        policy_dof_vel = self._obs_noise(dof_vel, noise_cfg.scale_joint_vel) * 0.05
        num_obs = gyro.shape[0]
        last_actions = info.get(
            "current_actions", np.zeros((num_obs, NUM_GO2W_ACTIONS), dtype=dof_pos.dtype)
        )
        commands = info["commands"]

        obs = np.concatenate(
            [
                policy_gyro,
                policy_gravity,
                commands,
                policy_leg_diff,
                policy_dof_vel,
                last_actions,
            ],
            axis=1,
            dtype=get_global_dtype(),
        )
        critic_base = np.concatenate(
            [linvel, gyro, -gravity, commands, leg_diff, dof_vel, last_actions],
            axis=1,
            dtype=get_global_dtype(),
        )
        critic = np.concatenate(
            [critic_base, height_scan_obs(self, self._cfg.terrain_scan, num_obs)],
            axis=1,
            dtype=get_global_dtype(),
        )
        return {"obs": obs, "critic": critic}

    def _reward_base_height_values(self, num_obs: int) -> np.ndarray:
        height = base_height_from_scan(self, num_obs)
        if height.shape[0] != num_obs:
            return super()._reward_base_height_values(num_obs)
        return height

    def _update_commands(self, info: dict) -> None:
        commands = info.get("commands")
        if commands is None:
            return

        commands_arr = np.asarray(commands, dtype=get_global_dtype())
        resampling_time = float(self._cfg.commands.resampling_time)
        if resampling_time > 0.0:
            interval_steps = max(int(round(resampling_time / self._cfg.ctrl_dt)), 1)
            steps = np.asarray(info.get("steps", np.zeros((self._num_envs,), dtype=np.uint32)))
            resample_mask = (steps > 0) & ((steps % interval_steps) == 0)
            if np.any(resample_mask):
                num_resample = int(np.count_nonzero(resample_mask))
                low = np.asarray(self._cfg.commands.vel_limit[0], dtype=get_global_dtype())
                high = np.asarray(self._cfg.commands.vel_limit[1], dtype=get_global_dtype())
                sampled = np.random.uniform(low=low, high=high, size=(num_resample, 3)).astype(
                    get_global_dtype()
                )
                _zero_small_xy_commands(sampled)
                commands_arr[resample_mask] = sampled
                if self._cfg.commands.heading_command:
                    heading_commands = self._ensure_heading_commands(info, commands_arr.shape[0])
                    heading_commands[resample_mask] = sample_go2w_heading_commands(
                        self, num_resample
                    )
                    info["heading_commands"] = heading_commands

        if self._cfg.commands.heading_command:
            heading_commands = self._ensure_heading_commands(info, commands_arr.shape[0])
            base_quat = np.asarray(self._backend.get_base_quat(), dtype=get_global_dtype())
            if base_quat.shape[0] == commands_arr.shape[0]:
                heading = yaw_from_quat_np(base_quat)
                commands_arr[:, 2] = np.clip(
                    0.5 * wrap_to_pi_np(heading_commands - heading), -2.0, 2.0
                )
        info["commands"] = commands_arr

    def _compute_terminated(self, gravity: np.ndarray) -> np.ndarray:
        del gravity
        return np.zeros((self._num_envs,), dtype=bool)

    def _raw_height_scan_obs(self, num_obs: int) -> tuple[np.ndarray | None, np.ndarray | None]:
        return raw_height_scan_obs(self, num_obs)

    def _compute_truncated(self, state: NpEnvState) -> np.ndarray:
        truncated = super()._compute_truncated(state)
        if self._cfg.termination_config.terrain_out_of_bounds:
            terrain_scene = self._cfg.scene.terrain
            terrain_cfg = terrain_scene.generator if terrain_scene is not None else None
            np.logical_or(
                truncated,
                terrain_out_of_bounds(
                    self,
                    terrain_cfg,
                    float(self._cfg.termination_config.terrain_distance_buffer),
                ),
                out=truncated,
            )
        return truncated


registry.register_env("Go2WJoystickRough", Go2WJoystickRoughEnv, sim_backend="motrix")


def _zero_small_xy_commands(commands: np.ndarray) -> None:
    moving = np.linalg.norm(commands[:, :2], axis=1) > 0.001
    commands[:, :2] *= moving[:, None]
