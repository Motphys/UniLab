"""Go1 joystick rough-terrain task with procedural sub-terrains.

The reward set is intentionally minimal: only the *common* RewardContext-pure
functions are used (no Go2-specific hip_pos / joint_mirror / feet_gait).
Contact-timer / sensor-reading rewards from Go2 are also omitted so that this
file stays a thin layer over ``Go1WalkTask``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from unilab.assets import ASSETS_ROOT_PATH
from unilab.base import registry
from unilab.base.np_env import NpEnvState
from unilab.base.scene import SceneCfg, TerrainSceneCfg
from unilab.dr import DomainRandomizationManager, ResetPlan
from unilab.dr.dr_utils import build_common_reset_randomization, zero_actions
from unilab.dtype_config import get_global_dtype
from unilab.envs.common.rotation import np_quat_from_euler_xyz, np_quat_mul
from unilab.envs.locomotion.common import rewards
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
from unilab.envs.locomotion.go1.base import ControlConfig
from unilab.envs.locomotion.go1.joystick import (
    Go1JoystickCfg,
    Go1JoystickDomainRandomizationProvider,
    Go1WalkTask,
    JoystickSensor,
    RewardConfig,
)
from unilab.envs.locomotion.common.commands import Commands
from unilab.terrains import (
    SubTerrainCfg,
    TerrainGeneratorCfg,
    flat,
    hf_pyramid_slope,
    hf_pyramid_slope_inv,
    pyramid_stairs,
    pyramid_stairs_inv,
    random_rough,
    wave_terrain,
)

# pyright: reportIncompatibleVariableOverride=false, reportAttributeAccessIssue=false, reportCallIssue=false

GO1_HEIGHT_SCAN_SCALE = 5.0


@dataclass
class TerrainScanConfig(HeightScanConfig):
    scale: float = GO1_HEIGHT_SCAN_SCALE


@dataclass
class RoughControlConfig(ControlConfig):
    clip_actions: float = 100.0


@dataclass
class RoughCommands(Commands):
    vel_limit: list[list[float]] = field(
        default_factory=lambda: [[-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]]
    )
    resampling_time: float = 10.0
    heading_command: bool = True
    heading_range: list[float] = field(default_factory=lambda: [-np.pi, np.pi])


@dataclass
class RoughRewardConfig(RewardConfig):
    stand_still_command_threshold: float = 0.1
    joint_pos_penalty_stand_still_scale: float = 5.0
    joint_pos_penalty_velocity_threshold: float = 0.5
    joint_pos_penalty_command_threshold: float = 0.1


@dataclass
class RoughTerminationConfig:
    terrain_out_of_bounds: bool = True
    terrain_distance_buffer: float = 3.0


@dataclass(kw_only=True)
class Go1RoughTerrainCfg(TerrainGeneratorCfg):
    size: tuple[float, float] = (8.0, 8.0)
    num_rows: int = 6
    num_cols: int = 6
    border_width: float = 1.0
    add_lights: bool = True
    horizontal_scale: float = 0.2

    sub_terrains: dict[str, SubTerrainCfg] = field(
        default_factory=lambda: {
            "flat": flat(proportion=0.0),
            "pyramid_stairs": pyramid_stairs(
                proportion=0.1,
                step_height_range=(0.025, 0.10),
                step_width=0.4,
                platform_width=3.0,
                border_width=0.2,
            ),
            "pyramid_stairs_inv": pyramid_stairs_inv(
                proportion=0.1,
                step_height_range=(0.025, 0.10),
                step_width=0.4,
                platform_width=3.0,
                border_width=0.2,
            ),
            "hf_pyramid_slope": hf_pyramid_slope(
                proportion=0.2,
                slope_range=(0.0, 0.3),
                platform_width=2.0,
                border_width=0.2,
            ),
            "hf_pyramid_slope_inv": hf_pyramid_slope_inv(
                proportion=0.2,
                slope_range=(0.0, 0.3),
                platform_width=2.0,
                border_width=0.2,
            ),
            "random_rough": random_rough(
                proportion=0.3,
                noise_range=(0.01, 0.06),
                noise_step=0.01,
                border_width=0.2,
            ),
            "wave_terrain": wave_terrain(
                proportion=0.3,
                amplitude_range=(0.0, 0.12),
                num_waves=4,
                border_width=0.2,
            ),
        }
    )


@registry.envcfg("Go1JoystickRough")
@dataclass
class Go1JoystickRoughCfg(Go1JoystickCfg):
    scene: SceneCfg = field(
        default_factory=lambda: SceneCfg(
            model_file=str(ASSETS_ROOT_PATH / "robots" / "go1" / "go1.xml"),
            fragment_files=[
                str(ASSETS_ROOT_PATH / "robots" / "go1" / "locomotion_task.xml"),
            ],
            terrain=TerrainSceneCfg(
                generator=Go1RoughTerrainCfg(),
                hfield_name="terrain_hfield",
                geom_name="floor",
            ),
        )
    )
    control_config: RoughControlConfig = field(default_factory=RoughControlConfig)
    commands: RoughCommands = field(default_factory=RoughCommands)
    terrain_scan: TerrainScanConfig = field(default_factory=TerrainScanConfig)
    termination_config: RoughTerminationConfig = field(default_factory=RoughTerminationConfig)
    terrain_curriculum: TerrainCurriculumCfg = field(default_factory=TerrainCurriculumCfg)
    sensor: JoystickSensor = field(default_factory=JoystickSensor)
    reward_config: RoughRewardConfig | None = None


class Go1JoystickRoughDomainRandomizationProvider(Go1JoystickDomainRandomizationProvider):
    def build_reset_plan(self, env: Any, env_ids: np.ndarray) -> ResetPlan:
        num_reset = len(env_ids)
        qpos = np.tile(env._init_qpos, (num_reset, 1))
        qvel = np.tile(env._init_qvel, (num_reset, 1))
        qpos[:, 0:2] += np.random.uniform(-0.5, 0.5, (num_reset, 2))
        qpos[:, 2] += np.random.uniform(0.05, 0.2, (num_reset,))
        qpos[:, 0:3] += env._spawn.origins_for(env_ids)
        yaw = np.random.uniform(-3.14, 3.14, (num_reset,))
        qpos[:, 3:7] = np_quat_mul(
            qpos[:, 3:7],
            np_quat_from_euler_xyz(
                np.zeros_like(yaw), np.zeros_like(yaw), yaw
            ),
        )
        qvel[:, 0:6] = np.asarray(
            np.random.uniform(-0.3, 0.3, size=(num_reset, 6)), dtype=get_global_dtype()
        )
        commands = self._sample_commands(env, num_reset)
        _zero_small_xy_commands(commands)
        if env.cfg.commands.heading_command:
            commands[:, 2] = 0.0
        info_updates: dict[str, Any] = {
            "commands": commands,
            "current_actions": zero_actions(num_reset, env._num_action),
            "last_actions": zero_actions(num_reset, env._num_action),
            "qacc": np.zeros((num_reset, env._num_action), dtype=get_global_dtype()),
            "torques": np.zeros((num_reset, env._num_action), dtype=get_global_dtype()),
        }
        if env.cfg.commands.heading_command:
            info_updates["heading_commands"] = _sample_heading_commands(env, num_reset)
        env._spawn.record_episode_start(env_ids, qpos[:, 0:3])
        return ResetPlan(
            env_ids=env_ids,
            qpos=qpos,
            qvel=qvel,
            info_updates=info_updates,
            randomization=build_common_reset_randomization(env, num_reset),
        )


@registry.env("Go1JoystickRough", sim_backend="mujoco")
class Go1JoystickRoughEnv(Go1WalkTask):
    _cfg: Go1JoystickRoughCfg
    _reward_cfg: RoughRewardConfig

    def __init__(self, cfg: Go1JoystickRoughCfg, num_envs=1, backend_type="mujoco"):
        super().__init__(cfg, num_envs=num_envs, backend_type=backend_type)
        # Replace the default no-op spawn manager with one that places each env
        # on a terrain tile when the scene has a procedural terrain attached.
        terrain_origins = getattr(self._backend, "terrain_origins", None)
        terrain_generator = (
            cfg.scene.terrain.generator if cfg.scene.terrain is not None else None
        )
        if terrain_origins is not None and terrain_generator is not None:
            terrain_surface_sampler = getattr(
                self._backend, "terrain_surface_sampler", None
            )
            self._spawn = TerrainSpawnManager(
                num_envs,
                terrain_origins,
                cell_size=float(terrain_generator.size[0]),
                cfg=cfg.terrain_curriculum,
                terrain_surface_sampler=terrain_surface_sampler,
            )
        self._dr_manager = DomainRandomizationManager(
            self, Go1JoystickRoughDomainRandomizationProvider()
        )
        self._last_dof_vel_for_acc = np.zeros(
            (num_envs, self._num_action), dtype=get_global_dtype()
        )
        joint_range = self._backend.get_joint_range()
        self._joint_range = (
            np.asarray(joint_range, dtype=get_global_dtype()) if joint_range is not None else None
        )
        init_height_scan_sensor(self, cfg.terrain_scan, cfg.asset.base_name)

    @property
    def obs_groups_spec(self) -> dict[str, int]:
        base_spec = super().obs_groups_spec
        return {"obs": base_spec["obs"], "critic": base_spec["critic"] + self._height_scan_dim}

    def _init_reward_functions(self):
        def _joint_pos_penalty(ctx: RewardContext) -> np.ndarray:
            cfg = self._reward_cfg
            return rewards.joint_pos_penalty(
                ctx,
                stand_still_scale=cfg.joint_pos_penalty_stand_still_scale,
                velocity_threshold=cfg.joint_pos_penalty_velocity_threshold,
                command_threshold=cfg.joint_pos_penalty_command_threshold,
            )

        def _stand_still(ctx: RewardContext) -> np.ndarray:
            return rewards.stand_still(
                ctx, command_threshold=self._reward_cfg.stand_still_command_threshold
            )

        self._reward_fns: dict[str, Any] = {
            "tracking_lin_vel": rewards.tracking_lin_vel,
            "tracking_ang_vel": rewards.tracking_ang_vel,
            "lin_vel_z": rewards.lin_vel_z,
            "ang_vel_xy": rewards.ang_vel_xy,
            "orientation": rewards.orientation,
            "base_height": rewards.base_height,
            "action_rate": rewards.action_rate,
            "action_rate_l2": rewards.action_rate,
            "similar_to_default": rewards.similar_to_default,
            "dof_torques_l2": rewards.dof_torques_l2,
            "joint_torques_l2": rewards.dof_torques_l2,
            "dof_acc_l2": rewards.dof_acc_l2,
            "joint_acc_l2": rewards.dof_acc_l2,
            "joint_pos_limits": rewards.joint_pos_limits,
            "joint_power": rewards.joint_power,
            "stand_still": _stand_still,
            "joint_pos_penalty": _joint_pos_penalty,
            "upward": rewards.upward,
            "alive": rewards.alive,
            "swing_feet_z": self._reward_swing_feet_z,
        }

    def update_state(self, state: NpEnvState) -> NpEnvState:
        state = super().update_state(state)
        # Add height-scan to critic observation, terrain-out-of-bounds truncation,
        # and curriculum logging.
        state = self._maybe_log_curriculum(state)
        return state

    def _reward_swing_feet_z(self, ctx: RewardContext) -> np.ndarray:
        """Swing-phase foot-lift reward, terrain-robust.

        Uses foot z **relative to the base** instead of world z so the target
        is meaningful regardless of terrain height. Gated on ``commands`` so it
        doesn't fight the stand_still penalty when the command is zero.
        """
        is_swing = self.feet_phase >= 0.6
        # Foot height below the base when standing is ~0.27 m (init_state.pos).
        # During swing we want the foot ~0.10 m above the local terrain, i.e.
        # ~0.17 m below the base.
        target_rel_z = -0.17
        base_z = self._backend.get_base_pos()[:, 2:3]
        rel_z = self.feet_pos[:, :, 2] - base_z
        height_error = np.square(rel_z - target_rel_z)
        swing_rew = np.exp(-height_error / 0.01) * is_swing
        moving = np.linalg.norm(ctx.info["commands"], axis=1) > 0.1
        reward = np.sum(swing_rew, axis=1) / len(self._cfg.sensor.feet_pos)
        return np.asarray(reward * moving, dtype=get_global_dtype())

    def _compute_obs(
        self, info: dict, linvel, gyro, gravity, dof_pos, dof_vel, feet_phase
    ) -> dict[str, np.ndarray]:
        obs = super()._compute_obs(info, linvel, gyro, gravity, dof_pos, dof_vel, feet_phase)
        num_obs = obs["critic"].shape[0]
        obs["critic"] = np.concatenate(
            [obs["critic"], height_scan_obs(self, self._cfg.terrain_scan, num_obs)],
            axis=1,
            dtype=get_global_dtype(),
        )
        return obs

    def _compute_reward(self, info: dict, linvel, gyro, dof_pos) -> np.ndarray:
        # Reuse parent dispatch but inject extra context fields (joint_range,
        # base_height from height-scan, gravity, dof_vel, torques, qacc).
        dtype = get_global_dtype()
        reward = np.zeros((self._num_envs,), dtype=dtype)
        cfg = self._reward_cfg

        dof_vel = self.get_dof_vel()
        gravity = self._backend.get_sensor_data("upvector")
        info["torques"] = self._estimate_pd_torques(info, dof_pos, dof_vel)
        info["qacc"] = self._estimate_dof_acc(dof_vel)

        ctx = RewardContext(
            info=info,
            linvel=linvel,
            gyro=gyro,
            dof_pos=dof_pos,
            num_envs=self._num_envs,
            default_angles=self.default_angles,
            tracking_sigma=cfg.tracking_sigma,
            base_height_target=cfg.base_height_target,
            base_height=base_height_from_scan(self, self._num_envs),
            gravity=gravity,
            dof_vel=dof_vel,
            joint_range=self._joint_range,
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

    def _raw_height_scan_obs(self, num_obs: int) -> tuple[np.ndarray | None, np.ndarray | None]:
        return raw_height_scan_obs(self, num_obs)

    def _estimate_dof_acc(self, dof_vel: np.ndarray) -> np.ndarray:
        qacc = np.asarray((dof_vel - self._last_dof_vel_for_acc) / self._cfg.ctrl_dt)
        self._last_dof_vel_for_acc[:] = dof_vel
        return np.asarray(qacc, dtype=get_global_dtype())

    def _estimate_pd_torques(
        self, info: dict, dof_pos: np.ndarray, dof_vel: np.ndarray
    ) -> np.ndarray:
        actions = np.asarray(
            info.get("current_actions", np.zeros((dof_pos.shape[0], self._num_action))),
            dtype=get_global_dtype(),
        )
        targets = actions * float(self._cfg.control_config.action_scale) + self.default_angles
        torques = (
            float(self._cfg.control_config.Kp) * (targets - dof_pos)
            - float(self._cfg.control_config.Kd) * dof_vel
        )
        return np.asarray(torques, dtype=get_global_dtype())

    def _maybe_log_curriculum(self, state: NpEnvState) -> NpEnvState:
        done = state.terminated | state.truncated
        if not np.any(done):
            return state
        done_indices = np.where(done)[0]
        stats = self._spawn.update_on_done(
            done_indices, self._backend.get_base_pos()[done_indices]
        )
        if stats:
            if "log" not in state.info:
                state.info["log"] = {}
            for k, v in stats.items():
                state.info["log"][f"terrain_curriculum/{k}"] = float(v)
        return state


def _zero_small_xy_commands(commands: np.ndarray) -> None:
    moving = np.linalg.norm(commands[:, :2], axis=1) > 0.001
    commands[:, :2] *= moving[:, None]


def _sample_heading_commands(env: Any, num_samples: int) -> np.ndarray:
    heading_range = np.asarray(env.cfg.commands.heading_range, dtype=get_global_dtype())
    low, high = float(np.min(heading_range)), float(np.max(heading_range))
    return np.asarray(np.random.uniform(low, high, size=(num_samples,)), dtype=get_global_dtype())


registry.register_env("Go1JoystickRough", Go1JoystickRoughEnv, sim_backend="motrix")
