"""Go1 joystick rough-terrain task.

This mirrors ``Go2JoystickRough``'s reward set and runtime contracts as
closely as Go1's asset / actuator layout allows. The leg ordering (FR, FL, RR,
RL × hip/thigh/knee = 12 actuators) and ``FL/FR/RL/RR`` foot/site naming match
Go2 byte-for-byte, so the same gait, hip-deviation and feet-related rewards
work directly.

Differences from Go2 rough:
* Go1's xml uses class-tagged collision geoms (unnamed) for hip / thigh / calf;
  ``undesired_contact`` therefore uses ``body2=`` selectors and drops the
  per-calf split (calf body subtree contains the foot site).
* Go1 has no ``base1/base2/base3`` collision geom names; trunk contacts are
  collapsed into a single ``trunk_contact`` sensor.
* Calf contact is omitted from the undesired-contact list for the reason above.
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
from unilab.envs.common.rotation import (
    np_quat_apply_inverse,
    np_quat_from_euler_xyz,
    np_quat_mul,
)
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
from unilab.envs.locomotion.go1.base import ControlConfig
from unilab.envs.locomotion.go1.joystick import (
    Go1JoystickCfg,
    Go1JoystickDomainRandomizationProvider,
    Go1WalkTask,
    JoystickSensor,
    RewardConfig,
)
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
# Joint ordering (FR, FL, RR, RL × hip/thigh/knee) — hip indices are the
# abduction joints, identical to Go2.
GO1_HIP_INDICES = np.asarray([0, 3, 6, 9], dtype=np.int32)
GO1_FRONT_LEFT = 0
GO1_FRONT_RIGHT = 1
GO1_REAR_LEFT = 2
GO1_REAR_RIGHT = 3


@dataclass
class TerrainScanConfig(HeightScanConfig):
    scale: float = GO1_HEIGHT_SCAN_SCALE


@dataclass
class RoughControlConfig(ControlConfig):
    hip_action_scale: float = 0.125
    non_hip_action_scale: float = 0.25
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
    contact_threshold: float = 1.0
    contact_forces_threshold: float = 100.0
    feet_air_time_threshold: float = 0.5
    feet_height_body_target: float = -0.2
    feet_height_body_tanh_mult: float = 2.0
    feet_gait_std: float = float(np.sqrt(0.5))
    feet_gait_max_err: float = 0.2
    feet_gait_velocity_threshold: float = 0.5
    feet_gait_command_threshold: float = 0.1


@dataclass
class RoughJoystickSensor(JoystickSensor):
    feet_vel = ["FL_vel", "FR_vel", "RL_vel", "RR_vel"]
    undesired_contact = [
        "base1_contact",
        "base2_contact",
        "base3_contact",
        "FL_hip_contact",
        "FR_hip_contact",
        "RL_hip_contact",
        "RR_hip_contact",
        "FL_thigh_contact",
        "FR_thigh_contact",
        "RL_thigh_contact",
        "RR_thigh_contact",
        "FL_calf_contact1",
        "FR_calf_contact1",
        "RL_calf_contact1",
        "RR_calf_contact1",
        "FL_calf_contact2",
        "FR_calf_contact2",
        "RL_calf_contact2",
        "RR_calf_contact2",
    ]


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
    sensor: RoughJoystickSensor = field(default_factory=RoughJoystickSensor)
    reward_config: RoughRewardConfig | None = None


class Go1JoystickRoughDomainRandomizationProvider(Go1JoystickDomainRandomizationProvider):
    def _sample_commands(self, env: Any, num_reset: int) -> np.ndarray:
        commands = super()._sample_commands(env, num_reset)
        _zero_small_xy_commands(commands)
        if env.cfg.commands.heading_command:
            commands[:, 2] = 0.0
        return commands

    def build_reset_plan(self, env: Any, env_ids: np.ndarray) -> ResetPlan:
        num_reset = len(env_ids)
        qpos = np.tile(env._init_qpos, (num_reset, 1))
        qvel = np.tile(env._init_qvel, (num_reset, 1))
        qpos[:, 0:2] += np.random.uniform(-0.5, 0.5, (num_reset, 2))
        qpos[:, 2] += np.random.uniform(0.1, 0.3, (num_reset,))
        qpos[:, 0:3] += env._spawn.origins_for(env_ids)
        # Random roll/pitch/yaw forces the policy to learn recovery from any
        # initial orientation (matches Go2 rough).
        roll = np.random.uniform(-3.14, 3.14, (num_reset,))
        pitch = np.random.uniform(-3.14, 3.14, (num_reset,))
        yaw = np.random.uniform(-3.14, 3.14, (num_reset,))
        qpos[:, 3:7] = np_quat_mul(qpos[:, 3:7], np_quat_from_euler_xyz(roll, pitch, yaw))
        qvel[:, 0:6] = np.asarray(
            np.random.uniform(-0.5, 0.5, size=(num_reset, 6)), dtype=get_global_dtype()
        )
        commands = self._sample_commands(env, num_reset)
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

        # Replace default no-op spawn manager with terrain-aware one.
        terrain_origins = getattr(self._backend, "terrain_origins", None)
        terrain_generator = (
            cfg.scene.terrain.generator if cfg.scene.terrain is not None else None
        )
        if terrain_origins is not None and terrain_generator is not None:
            self._spawn = TerrainSpawnManager(
                num_envs,
                terrain_origins,
                cell_size=float(terrain_generator.size[0]),
                cfg=cfg.terrain_curriculum,
                terrain_surface_sampler=getattr(self._backend, "terrain_surface_sampler", None),
            )
        self._dr_manager = DomainRandomizationManager(
            self, Go1JoystickRoughDomainRandomizationProvider()
        )

        # PD-estimation buffers and joint limits for the lifted common rewards.
        self._last_dof_vel_for_acc = np.zeros(
            (num_envs, self._num_action), dtype=get_global_dtype()
        )
        joint_range = self._backend.get_joint_range()
        self._joint_range = (
            np.asarray(joint_range, dtype=get_global_dtype()) if joint_range is not None else None
        )

        # Per-joint action scale: smaller scale on the hip abduction joints
        # tames sideways flailing during early training (mirrors Go2 rough).
        self._action_scale = np.full(
            (self._num_action,),
            float(cfg.control_config.non_hip_action_scale),
            dtype=get_global_dtype(),
        )
        self._action_scale[GO1_HIP_INDICES] = float(cfg.control_config.hip_action_scale)

        # Foot kinematics + contact-timer state for the gait/air-time rewards.
        self.feet_vel = np.zeros((num_envs, len(cfg.sensor.feet_vel), 3), dtype=np.float32)
        self._last_foot_contact = np.zeros((num_envs, len(cfg.sensor.feet_force)), dtype=bool)
        self._current_air_time = np.zeros(
            (num_envs, len(cfg.sensor.feet_force)), dtype=np.float32
        )
        self._current_contact_time = np.zeros(
            (num_envs, len(cfg.sensor.feet_force)), dtype=np.float32
        )
        self._last_air_time = np.zeros(
            (num_envs, len(cfg.sensor.feet_force)), dtype=np.float32
        )
        self._last_contact_time = np.zeros(
            (num_envs, len(cfg.sensor.feet_force)), dtype=np.float32
        )
        self._first_foot_contact = np.zeros((num_envs, len(cfg.sensor.feet_force)), dtype=bool)

        init_height_scan_sensor(self, cfg.terrain_scan, cfg.asset.base_name)

    @property
    def obs_groups_spec(self) -> dict[str, int]:
        # Match the rough format: policy obs = 45, critic = 48 + height_scan.
        return {"obs": 45, "critic": 48 + self._height_scan_dim}

    def reset(self, env_indices: np.ndarray) -> tuple[dict[str, np.ndarray], dict]:
        env_ids = np.asarray(env_indices, dtype=np.int32)
        obs, info = super().reset(env_ids)
        dof_vel = self.get_dof_vel()
        if dof_vel.shape[0] == self._num_envs:
            self._last_dof_vel_for_acc[env_ids] = dof_vel[env_ids]
        self._reset_contact_timers(env_ids)
        return obs, info

    def _upright_scale(self, gravity: np.ndarray | None) -> np.ndarray:
        return rewards.upright_scale(gravity, self._num_envs)

    def _init_reward_functions(self):
        scale_gravity = self._upright_scale

        def gated(fn):
            return lambda ctx: fn(ctx) * scale_gravity(ctx.gravity)

        def _joint_pos_penalty(ctx: RewardContext) -> np.ndarray:
            cfg = self._reward_cfg
            return rewards.joint_pos_penalty(
                ctx,
                stand_still_scale=cfg.joint_pos_penalty_stand_still_scale,
                velocity_threshold=cfg.joint_pos_penalty_velocity_threshold,
                command_threshold=cfg.joint_pos_penalty_command_threshold,
            ) * scale_gravity(ctx.gravity)

        def _stand_still(ctx: RewardContext) -> np.ndarray:
            return rewards.stand_still(
                ctx, command_threshold=self._reward_cfg.stand_still_command_threshold
            ) * scale_gravity(ctx.gravity)

        self._reward_fns: dict[str, Any] = {
            "tracking_lin_vel": gated(rewards.tracking_lin_vel),
            "tracking_ang_vel": gated(rewards.tracking_ang_vel),
            "lin_vel_z": gated(rewards.lin_vel_z),
            "ang_vel_xy": gated(rewards.ang_vel_xy),
            "dof_torques_l2": gated(rewards.dof_torques_l2),
            "joint_torques_l2": gated(rewards.dof_torques_l2),
            "dof_acc_l2": gated(rewards.dof_acc_l2),
            "joint_acc_l2": gated(rewards.dof_acc_l2),
            "joint_pos_limits": gated(rewards.joint_pos_limits),
            "joint_power": gated(rewards.joint_power),
            "stand_still": _stand_still,
            "hip_pos": self._reward_hip_pos,
            "joint_pos_penalty": _joint_pos_penalty,
            "joint_mirror": self._reward_joint_mirror,
            "action_rate": rewards.action_rate,
            "action_rate_l2": rewards.action_rate,
            "undesired_contacts": self._reward_undesired_contacts,
            "contact_forces": self._reward_contact_forces,
            "feet_air_time": self._reward_feet_air_time,
            "feet_air_time_variance": self._reward_feet_air_time_variance,
            "feet_contact_without_cmd": self._reward_feet_contact_without_cmd,
            "feet_slide": self._reward_feet_slide,
            "feet_height_body": self._reward_feet_height_body,
            "feet_gait": self._reward_feet_gait,
            "upward": rewards.upward,
            "alive": rewards.alive,
            "swing_feet_z": self._reward_swing_feet_z,
        }

    def apply_action(self, actions: np.ndarray, state: NpEnvState) -> np.ndarray:
        clipped = np.asarray(
            np.clip(
                actions,
                -float(self._cfg.control_config.clip_actions),
                float(self._cfg.control_config.clip_actions),
            ),
            dtype=get_global_dtype(),
        )
        state.info["last_actions"] = state.info.get("current_actions", np.zeros_like(clipped))
        state.info["current_actions"] = clipped
        exec_actions = (
            state.info["last_actions"]
            if self._cfg.control_config.simulate_action_latency
            else clipped
        )
        return np.asarray(
            exec_actions * self._action_scale + self.default_angles, dtype=get_global_dtype()
        )

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
        for i in range(len(self._cfg.sensor.feet_vel)):
            self.feet_vel[:, i, :] = self._backend.get_sensor_data(self._cfg.sensor.feet_vel[i])

        self._update_contact_timers(self._foot_contact_mask())
        state.info["qacc"] = self._estimate_dof_acc(dof_vel)
        state.info["torques"] = self._estimate_pd_torques(state.info, dof_pos, dof_vel)

        # Never terminate on tilt — let the policy learn recovery.
        terminated = np.zeros((self._num_envs,), dtype=bool)
        reward = self._compute_rough_reward(state.info, linvel, gyro, gravity, dof_pos, dof_vel)
        obs = self._compute_obs(state.info, linvel, gyro, gravity, dof_pos, dof_vel)
        state = state.replace(obs=obs, reward=reward, terminated=terminated)

        done = state.terminated | state.truncated
        if np.any(done):
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

    def _compute_obs(
        self,
        info: dict,
        linvel: np.ndarray,
        gyro: np.ndarray,
        gravity: np.ndarray,
        dof_pos: np.ndarray,
        dof_vel: np.ndarray,
        *args,  # absorbs feet_phase passed by parent reset path; unused here.
    ) -> dict[str, np.ndarray]:
        del args
        # Rough-style obs (no feet_phase in actor obs), matching Go2 rough.
        noise_cfg = self._cfg.noise_config
        diff = dof_pos - self.default_angles
        policy_gyro = self._obs_noise(gyro, noise_cfg.scale_gyro) * 0.25
        policy_gravity = self._obs_noise(-gravity, noise_cfg.scale_gravity)
        policy_diff = self._obs_noise(diff, noise_cfg.scale_joint_angle)
        policy_dof_vel = self._obs_noise(dof_vel, noise_cfg.scale_joint_vel) * 0.05
        last_actions = info.get("current_actions", np.zeros_like(diff))
        commands = info["commands"]
        obs = np.concatenate(
            [policy_gyro, policy_gravity, commands, policy_diff, policy_dof_vel, last_actions],
            axis=1,
            dtype=get_global_dtype(),
        )
        critic_base = np.concatenate(
            [linvel, gyro, -gravity, commands, diff, dof_vel, last_actions],
            axis=1,
            dtype=get_global_dtype(),
        )
        critic = np.concatenate(
            [critic_base, height_scan_obs(self, self._cfg.terrain_scan, critic_base.shape[0])],
            axis=1,
            dtype=get_global_dtype(),
        )
        return {"obs": obs, "critic": critic}

    def _compute_rough_reward(
        self,
        info: dict,
        linvel: np.ndarray,
        gyro: np.ndarray,
        gravity: np.ndarray,
        dof_pos: np.ndarray,
        dof_vel: np.ndarray,
    ) -> np.ndarray:
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
        if self._cfg.control_config.simulate_action_latency:
            actions = np.asarray(info.get("last_actions", actions), dtype=get_global_dtype())
        targets = actions * self._action_scale + self.default_angles
        torques = (
            float(self._cfg.control_config.Kp) * (targets - dof_pos)
            - float(self._cfg.control_config.Kd) * dof_vel
        )
        return np.asarray(torques, dtype=get_global_dtype())

    # ── contact timer ─────────────────────────────────────────────────────

    def _foot_contact_mask(self) -> np.ndarray:
        contact_force = np.linalg.norm(self.feet_force, axis=2)
        return np.asarray(contact_force > self._reward_cfg.contact_threshold, dtype=bool)

    def _reset_contact_timers(self, env_ids: np.ndarray) -> None:
        self._current_air_time[env_ids] = 0.0
        self._current_contact_time[env_ids] = 0.0
        self._last_air_time[env_ids] = 0.0
        self._last_contact_time[env_ids] = 0.0
        self._first_foot_contact[env_ids] = False
        self._last_foot_contact[env_ids] = self._foot_contact_mask()[env_ids]

    def _update_contact_timers(self, contact: np.ndarray) -> None:
        first_contact = contact & ~self._last_foot_contact
        first_air = ~contact & self._last_foot_contact
        self._first_foot_contact[:] = first_contact
        self._last_air_time[first_contact] = self._current_air_time[first_contact]
        self._last_contact_time[first_air] = self._current_contact_time[first_air]
        self._current_air_time[contact] = 0.0
        self._current_air_time[~contact] += self._cfg.ctrl_dt
        self._current_contact_time[~contact] = 0.0
        self._current_contact_time[contact] += self._cfg.ctrl_dt
        self._last_foot_contact[:] = contact

    # ── foot kinematics in body frame ─────────────────────────────────────

    def _relative_foot_vel_body(self) -> np.ndarray:
        base_quat = np.asarray(self._backend.get_base_quat(), dtype=get_global_dtype())
        base_linvel = np.asarray(
            self._backend.get_sensor_data("global_linvel"), dtype=get_global_dtype()
        )
        relative_vel = self.feet_vel - base_linvel[:, None, :]
        flat = relative_vel.reshape(self._num_envs * relative_vel.shape[1], 3)
        quat = np.repeat(base_quat, relative_vel.shape[1], axis=0)
        return np_quat_apply_inverse(quat, flat).reshape(relative_vel.shape)

    def _relative_foot_pos_body(self) -> np.ndarray:
        base_quat = np.asarray(self._backend.get_base_quat(), dtype=get_global_dtype())
        base_pos = np.asarray(self._backend.get_base_pos(), dtype=get_global_dtype())
        relative_pos = self.feet_pos - base_pos[:, None, :]
        flat = relative_pos.reshape(self._num_envs * relative_pos.shape[1], 3)
        quat = np.repeat(base_quat, relative_pos.shape[1], axis=0)
        return np_quat_apply_inverse(quat, flat).reshape(relative_pos.shape)

    # ── Go2-style env-coupled rewards (ported verbatim, Go1 names) ────────

    def _reward_hip_pos(self, ctx: RewardContext) -> np.ndarray:
        diff = ctx.dof_pos[:, GO1_HIP_INDICES] - self.default_angles[GO1_HIP_INDICES]
        return np.asarray(
            np.sum(np.square(diff), axis=1) * self._upright_scale(ctx.gravity),
            dtype=get_global_dtype(),
        )

    def _reward_joint_mirror(self, ctx: RewardContext) -> np.ndarray:
        fr_rl = ctx.dof_pos[:, 0:3] - ctx.dof_pos[:, 9:12]
        fl_rr = ctx.dof_pos[:, 3:6] - ctx.dof_pos[:, 6:9]
        mirror = 0.5 * (np.sum(np.square(fr_rl), axis=1) + np.sum(np.square(fl_rr), axis=1))
        return np.asarray(mirror * self._upright_scale(ctx.gravity), dtype=get_global_dtype())

    def _reward_undesired_contacts(self, ctx: RewardContext) -> np.ndarray:
        contacts = [
            _force_norm_columns(
                np.asarray(self._backend.get_sensor_data(name), dtype=get_global_dtype()),
                ctx.num_envs,
            )
            for name in self._cfg.sensor.undesired_contact
        ]
        if not contacts:
            return np.zeros((ctx.num_envs,), dtype=get_global_dtype())
        contact_force = np.concatenate(contacts, axis=1)
        contact_count = np.sum(contact_force > self._reward_cfg.contact_threshold, axis=1)
        return np.asarray(
            contact_count * self._upright_scale(ctx.gravity), dtype=get_global_dtype()
        )

    def _reward_contact_forces(self, ctx: RewardContext) -> np.ndarray:
        force_norm = np.linalg.norm(self.feet_force, axis=2)
        violation = np.clip(force_norm - self._reward_cfg.contact_forces_threshold, 0.0, None)
        return np.asarray(
            np.sum(violation, axis=1) * self._upright_scale(ctx.gravity),
            dtype=get_global_dtype(),
        )

    def _reward_feet_air_time(self, ctx: RewardContext) -> np.ndarray:
        cfg = self._reward_cfg
        reward = np.sum(
            (self._last_air_time - cfg.feet_air_time_threshold) * self._first_foot_contact,
            axis=1,
        )
        moving = np.linalg.norm(ctx.info["commands"], axis=1) > 0.1
        return np.asarray(
            reward * moving * self._upright_scale(ctx.gravity), dtype=get_global_dtype()
        )

    def _reward_feet_air_time_variance(self, ctx: RewardContext) -> np.ndarray:
        air_var = np.var(np.clip(self._last_air_time, 0.0, 0.5), axis=1)
        contact_var = np.var(np.clip(self._last_contact_time, 0.0, 0.5), axis=1)
        return np.asarray(
            (air_var + contact_var) * self._upright_scale(ctx.gravity), dtype=get_global_dtype()
        )

    def _reward_feet_contact_without_cmd(self, ctx: RewardContext) -> np.ndarray:
        reward = np.sum(self._first_foot_contact, axis=1)
        stopped = np.linalg.norm(ctx.info["commands"], axis=1) < 0.1
        return np.asarray(
            reward * stopped * self._upright_scale(ctx.gravity), dtype=get_global_dtype()
        )

    def _reward_feet_slide(self, ctx: RewardContext) -> np.ndarray:
        foot_vel_body = self._relative_foot_vel_body()
        lateral_vel = np.linalg.norm(foot_vel_body[:, :, :2], axis=2)
        reward = np.sum(lateral_vel * self._foot_contact_mask(), axis=1)
        return np.asarray(reward * self._upright_scale(ctx.gravity), dtype=get_global_dtype())

    def _reward_feet_height_body(self, ctx: RewardContext) -> np.ndarray:
        cfg = self._reward_cfg
        foot_pos_body = self._relative_foot_pos_body()
        foot_vel_body = self._relative_foot_vel_body()
        z_error = np.square(foot_pos_body[:, :, 2] - cfg.feet_height_body_target)
        velocity_tanh = np.tanh(
            cfg.feet_height_body_tanh_mult * np.linalg.norm(foot_vel_body[:, :, :2], axis=2)
        )
        moving = np.linalg.norm(ctx.info["commands"], axis=1) > 0.1
        reward = np.sum(z_error * velocity_tanh, axis=1)
        return np.asarray(
            reward * moving * self._upright_scale(ctx.gravity), dtype=get_global_dtype()
        )

    def _reward_feet_gait(self, ctx: RewardContext) -> np.ndarray:
        cfg = self._reward_cfg
        command_norm = np.linalg.norm(ctx.info["commands"], axis=1)
        body_vel = np.linalg.norm(ctx.linvel[:, :2], axis=1)
        enabled = (command_norm > cfg.feet_gait_command_threshold) | (
            body_vel > cfg.feet_gait_velocity_threshold
        )
        air = self._current_air_time
        contact = self._current_contact_time
        sync_fl_rr = _gait_sync_reward(
            air, contact, GO1_FRONT_LEFT, GO1_REAR_RIGHT, cfg.feet_gait_std, cfg.feet_gait_max_err
        )
        sync_fr_rl = _gait_sync_reward(
            air, contact, GO1_FRONT_RIGHT, GO1_REAR_LEFT, cfg.feet_gait_std, cfg.feet_gait_max_err
        )
        async_fl_fr = _gait_async_reward(
            air, contact, GO1_FRONT_LEFT, GO1_FRONT_RIGHT, cfg.feet_gait_std, cfg.feet_gait_max_err
        )
        async_rr_rl = _gait_async_reward(
            air, contact, GO1_REAR_RIGHT, GO1_REAR_LEFT, cfg.feet_gait_std, cfg.feet_gait_max_err
        )
        async_fl_rl = _gait_async_reward(
            air, contact, GO1_FRONT_LEFT, GO1_REAR_LEFT, cfg.feet_gait_std, cfg.feet_gait_max_err
        )
        async_fr_rr = _gait_async_reward(
            air, contact, GO1_FRONT_RIGHT, GO1_REAR_RIGHT, cfg.feet_gait_std, cfg.feet_gait_max_err
        )
        reward = sync_fl_rr * sync_fr_rl * async_fl_fr * async_rr_rl * async_fl_rl * async_fr_rr
        return np.asarray(
            reward * enabled * self._upright_scale(ctx.gravity), dtype=get_global_dtype()
        )

    def _reward_swing_feet_z(self, ctx: RewardContext) -> np.ndarray:
        """Swing-phase foot-lift reward, terrain-robust (body-relative z)."""
        is_swing = self.feet_phase >= 0.6
        # Stance ~0.27 m below the base; we want the foot ~0.10 m higher in swing.
        target_rel_z = -0.17
        base_z = self._backend.get_base_pos()[:, 2:3]
        rel_z = self.feet_pos[:, :, 2] - base_z
        height_error = np.square(rel_z - target_rel_z)
        swing_rew = np.exp(-height_error / 0.01) * is_swing
        moving = np.linalg.norm(ctx.info["commands"], axis=1) > 0.1
        reward = np.sum(swing_rew, axis=1) / len(self._cfg.sensor.feet_pos)
        return np.asarray(reward * moving, dtype=get_global_dtype())


# ── module helpers ────────────────────────────────────────────────────────


def _force_norm_columns(force: np.ndarray, num_envs: int) -> np.ndarray:
    force = np.asarray(force, dtype=get_global_dtype()).reshape(num_envs, -1)
    if force.shape[1] == 0:
        return force
    if force.shape[1] % 3 == 0:
        return np.linalg.norm(force.reshape(num_envs, -1, 3), axis=2)
    return np.abs(force)


def _gait_sync_reward(
    air: np.ndarray,
    contact: np.ndarray,
    foot_0: int,
    foot_1: int,
    std: float,
    max_err: float,
) -> np.ndarray:
    se_air = np.clip(np.square(air[:, foot_0] - air[:, foot_1]), 0.0, max_err**2)
    se_contact = np.clip(np.square(contact[:, foot_0] - contact[:, foot_1]), 0.0, max_err**2)
    return np.exp(-(se_air + se_contact) / std)


def _gait_async_reward(
    air: np.ndarray,
    contact: np.ndarray,
    foot_0: int,
    foot_1: int,
    std: float,
    max_err: float,
) -> np.ndarray:
    se_act_0 = np.clip(np.square(air[:, foot_0] - contact[:, foot_1]), 0.0, max_err**2)
    se_act_1 = np.clip(np.square(contact[:, foot_0] - air[:, foot_1]), 0.0, max_err**2)
    return np.exp(-(se_act_0 + se_act_1) / std)


def _zero_small_xy_commands(commands: np.ndarray) -> None:
    moving = np.linalg.norm(commands[:, :2], axis=1) > 0.001
    commands[:, :2] *= moving[:, None]


def _sample_heading_commands(env: Any, num_samples: int) -> np.ndarray:
    heading_range = np.asarray(env.cfg.commands.heading_range, dtype=get_global_dtype())
    if heading_range.shape != (2,):
        raise ValueError(f"commands.heading_range must have shape (2,), got {heading_range.shape}")
    low, high = float(np.min(heading_range)), float(np.max(heading_range))
    return np.asarray(np.random.uniform(low, high, size=(num_samples,)), dtype=get_global_dtype())


registry.register_env("Go1JoystickRough", Go1JoystickRoughEnv, sim_backend="motrix")
