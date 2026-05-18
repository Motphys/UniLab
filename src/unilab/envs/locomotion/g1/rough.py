"""G1 joystick rough-terrain task."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from unilab.assets import ASSETS_ROOT_PATH
from unilab.base import registry
from unilab.base.np_env import NpEnvState
from unilab.base.scene import SceneCfg, TerrainSceneCfg
from unilab.dtype_config import get_global_dtype
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
from unilab.envs.locomotion.g1.joystick import (
    LEFT_FOOT_CONTACT_SENSORS,
    RIGHT_FOOT_CONTACT_SENSORS,
    G1RewardConfig,
    G1WalkEnv,
    G1WalkEnvCfg,
    compute_aggregated_foot_contact,
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

G1_HEIGHT_SCAN_SCALE = 5.0

# Joint indices on the 29-DoF G1, derived from the joint order in g1.xml.
# 0..5: left leg, 6..11: right leg, 12..14: waist, 15..21: left arm, 22..28: right arm.
G1_LEG_HIP_INDICES = np.asarray([1, 2, 7, 8], dtype=np.int32)  # hip_roll + hip_yaw on each leg
G1_ANKLE_INDICES = np.asarray([4, 5, 10, 11], dtype=np.int32)  # ankle_pitch + ankle_roll
G1_ARM_INDICES = np.asarray(
    [15, 16, 17, 18, 22, 23, 24, 25],
    dtype=np.int32,
)  # shoulder pitch/roll/yaw + elbow (each side)
G1_WAIST_INDICES = np.asarray([12], dtype=np.int32)


@dataclass
class TerrainScanConfig(HeightScanConfig):
    scale: float = G1_HEIGHT_SCAN_SCALE


@dataclass
class RoughTerminationConfig:
    terrain_out_of_bounds: bool = True
    terrain_distance_buffer: float = 3.0


@dataclass
class G1JoystickRoughRewardConfig(G1RewardConfig):
    """Reward config for G1 rough — extends G1RewardConfig with biped fields."""

    feet_air_time_threshold: float = 0.4
    feet_air_time_command_threshold: float = 0.1
    only_positive_rewards: bool = False


@dataclass(kw_only=True)
class G1RoughTerrainCfg(TerrainGeneratorCfg):
    size: tuple[float, float] = (8.0, 8.0)
    num_rows: int = 6
    num_cols: int = 6
    border_width: float = 1.0
    add_lights: bool = True
    horizontal_scale: float = 0.2

    sub_terrains: dict[str, SubTerrainCfg] = field(
        default_factory=lambda: {
            "flat": flat(proportion=0.2),
            "pyramid_stairs": pyramid_stairs(
                proportion=0.2,
                step_height_range=(0.0, 0.1),
                step_width=0.4,
                platform_width=3.0,
                border_width=0.2,
            ),
            "pyramid_stairs_inv": pyramid_stairs_inv(
                proportion=0.2,
                step_height_range=(0.0, 0.1),
                step_width=0.4,
                platform_width=3.0,
                border_width=0.2,
            ),
            "hf_pyramid_slope": hf_pyramid_slope(
                proportion=0.1,
                slope_range=(0.0, 0.3),
                platform_width=2.0,
                border_width=0.2,
            ),
            "hf_pyramid_slope_inv": hf_pyramid_slope_inv(
                proportion=0.1,
                slope_range=(0.0, 0.3),
                platform_width=2.0,
                border_width=0.2,
            ),
            "random_rough": random_rough(
                proportion=0.1,
                noise_range=(0.02, 0.10),
                noise_step=0.02,
                border_width=0.2,
            ),
            "wave_terrain": wave_terrain(
                proportion=0.1,
                amplitude_range=(0.0, 0.2),
                num_waves=4,
                border_width=0.2,
            ),
        }
    )


@registry.envcfg("G1JoystickRough")
@dataclass
class G1JoystickRoughCfg(G1WalkEnvCfg):
    scene: SceneCfg = field(
        default_factory=lambda: SceneCfg(
            model_file=str(ASSETS_ROOT_PATH / "robots" / "g1" / "g1.xml"),
            fragment_files=[
                str(ASSETS_ROOT_PATH / "robots" / "g1" / "locomotion_task.xml"),
            ],
            terrain=TerrainSceneCfg(
                generator=G1RoughTerrainCfg(),
                hfield_name="terrain_hfield",
                geom_name="floor",
            ),
        )
    )
    reward_config: G1JoystickRoughRewardConfig | None = None
    terrain_scan: TerrainScanConfig = field(default_factory=TerrainScanConfig)
    termination_config: RoughTerminationConfig = field(default_factory=RoughTerminationConfig)
    terrain_curriculum: TerrainCurriculumCfg = field(default_factory=TerrainCurriculumCfg)


@registry.env("G1JoystickRough", sim_backend="mujoco")
class G1JoystickRoughEnv(G1WalkEnv):
    _cfg: G1JoystickRoughCfg
    _reward_cfg: G1JoystickRoughRewardConfig

    def __init__(self, cfg: G1JoystickRoughCfg, num_envs=1, backend_type="mujoco"):
        super().__init__(cfg, num_envs=num_envs, backend_type=backend_type)
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
        init_height_scan_sensor(self, cfg.terrain_scan, cfg.asset.base_name)
        joint_range = self._backend.get_joint_range()
        self._joint_range = (
            np.asarray(joint_range, dtype=get_global_dtype()) if joint_range is not None else None
        )
        self._last_dof_vel_for_acc = np.zeros(
            (num_envs, self._num_action), dtype=get_global_dtype()
        )
        # Biped contact timer state (left, right).
        self._last_foot_contact = np.zeros((num_envs, 2), dtype=bool)
        self._current_air_time = np.zeros((num_envs, 2), dtype=np.float32)
        self._current_contact_time = np.zeros((num_envs, 2), dtype=np.float32)

    @property
    def obs_groups_spec(self) -> dict[str, int]:
        base_spec = super().obs_groups_spec
        return {"obs": base_spec["obs"], "critic": base_spec["critic"] + self._height_scan_dim}

    def _init_reward_functions(self):
        def _joint_deviation_hip(ctx: RewardContext) -> np.ndarray:
            return rewards.joint_deviation_l1(ctx, G1_LEG_HIP_INDICES)

        def _joint_deviation_arms(ctx: RewardContext) -> np.ndarray:
            return rewards.joint_deviation_l1(ctx, G1_ARM_INDICES)

        def _joint_deviation_torso(ctx: RewardContext) -> np.ndarray:
            return rewards.joint_deviation_l1(ctx, G1_WAIST_INDICES)

        def _dof_pos_limits_ankle(ctx: RewardContext) -> np.ndarray:
            if ctx.joint_range is None:
                return np.zeros((ctx.num_envs,), dtype=get_global_dtype())
            ankle_lower = ctx.joint_range[G1_ANKLE_INDICES, 0]
            ankle_upper = ctx.joint_range[G1_ANKLE_INDICES, 1]
            ankle_pos = ctx.dof_pos[:, G1_ANKLE_INDICES]
            low_error = np.clip(ankle_lower - ankle_pos, 0.0, None)
            high_error = np.clip(ankle_pos - ankle_upper, 0.0, None)
            return np.asarray(
                np.sum(low_error + high_error, axis=1), dtype=get_global_dtype()
            )

        def _feet_air_time(ctx: RewardContext) -> np.ndarray:
            return rewards.feet_air_time_positive_biped(
                ctx,
                threshold=self._reward_cfg.feet_air_time_threshold,
                command_threshold=self._reward_cfg.feet_air_time_command_threshold,
            )

        # G1WalkEnv populated this dispatch in its __init__; we now overwrite it
        # entirely with the biped-style mix the user asked for.
        self._reward_fns: dict[str, Any] = {
            # tracking
            "tracking_lin_vel": rewards.tracking_lin_vel,
            "tracking_ang_vel": rewards.tracking_ang_vel,
            "track_lin_vel_xy_yaw_frame_exp": rewards.track_lin_vel_xy_yaw_frame_exp,
            "track_ang_vel_z_world_exp": rewards.track_ang_vel_z_world_exp,
            # penalties
            "lin_vel_z": rewards.lin_vel_z,
            "ang_vel_xy": rewards.ang_vel_xy,
            "orientation": rewards.orientation,
            "flat_orientation_l2": rewards.orientation,
            "base_height": rewards.base_height,
            "action_rate": rewards.action_rate,
            "action_rate_l2": rewards.action_rate,
            "dof_torques_l2": rewards.dof_torques_l2,
            "dof_acc_l2": rewards.dof_acc_l2,
            "joint_pos_limits": rewards.joint_pos_limits,
            "dof_pos_limits_ankle": _dof_pos_limits_ankle,
            "joint_deviation_hip": _joint_deviation_hip,
            "joint_deviation_arms": _joint_deviation_arms,
            "joint_deviation_torso": _joint_deviation_torso,
            # biped
            "feet_air_time": _feet_air_time,
            "feet_slide": self._reward_feet_slide_biped,
            # survival
            "alive": rewards.alive,
        }

    # ── biped feet_slide: reads ankle (foot) world linvel ──────────────────

    def _reward_feet_slide_biped(self, ctx: RewardContext) -> np.ndarray:
        del ctx
        left_vel = np.asarray(
            self._backend.get_sensor_data("left_foot_linvel"), dtype=get_global_dtype()
        )
        right_vel = np.asarray(
            self._backend.get_sensor_data("right_foot_linvel"), dtype=get_global_dtype()
        )
        left_speed = np.linalg.norm(left_vel[:, :2], axis=1)
        right_speed = np.linalg.norm(right_vel[:, :2], axis=1)
        left_contact = compute_aggregated_foot_contact(self._backend, LEFT_FOOT_CONTACT_SENSORS)
        right_contact = compute_aggregated_foot_contact(self._backend, RIGHT_FOOT_CONTACT_SENSORS)
        return np.asarray(
            left_speed * left_contact + right_speed * right_contact, dtype=get_global_dtype()
        )

    # ── update loop: extend G1WalkEnv to include contact timers,
    #    info["torques"] / info["qacc"], height scan & out-of-bounds ─────

    def update_state(self, state: NpEnvState) -> NpEnvState:
        self._step_contact_timers()
        # Stash for the reward dispatch which reads ctx.info.
        state.info["current_air_time"] = self._current_air_time
        state.info["current_contact_time"] = self._current_contact_time

        # Estimate PD torques (G1 uses position actuators with kp/kd from XML).
        dof_pos = self.get_dof_pos()
        dof_vel = self.get_dof_vel()
        state.info["torques"] = self._estimate_pd_torques(state.info, dof_pos, dof_vel)
        state.info["qacc"] = self._estimate_dof_acc(dof_vel)

        state = super().update_state(state)
        state = self._maybe_extend_truncated(state)
        return state

    def _step_contact_timers(self) -> None:
        left = compute_aggregated_foot_contact(self._backend, LEFT_FOOT_CONTACT_SENSORS)
        right = compute_aggregated_foot_contact(self._backend, RIGHT_FOOT_CONTACT_SENSORS)
        contact = np.stack([left, right], axis=1).astype(bool)
        air = ~contact
        self._current_air_time = np.where(
            air, self._current_air_time + self._cfg.ctrl_dt, 0.0
        ).astype(np.float32)
        self._current_contact_time = np.where(
            contact, self._current_contact_time + self._cfg.ctrl_dt, 0.0
        ).astype(np.float32)
        self._last_foot_contact = contact

    def _maybe_extend_truncated(self, state: NpEnvState) -> NpEnvState:
        if not self._cfg.termination_config.terrain_out_of_bounds:
            return state
        terrain_scene = self._cfg.scene.terrain
        terrain_cfg = terrain_scene.generator if terrain_scene is not None else None
        oob = terrain_out_of_bounds(
            self, terrain_cfg, float(self._cfg.termination_config.terrain_distance_buffer)
        )
        truncated = state.truncated | oob
        return state.replace(truncated=truncated)

    def _compute_obs(
        self, info: dict, linvel, gyro, gravity, dof_pos, dof_vel
    ) -> dict[str, np.ndarray]:
        obs = super()._compute_obs(info, linvel, gyro, gravity, dof_pos, dof_vel)
        num_obs = obs["critic"].shape[0]
        obs["critic"] = np.concatenate(
            [obs["critic"], height_scan_obs(self, self._cfg.terrain_scan, num_obs)],
            axis=1,
            dtype=get_global_dtype(),
        )
        return obs

    def _build_reward_context(
        self, info: dict, linvel, gyro, gravity, dof_pos, dof_vel
    ) -> RewardContext:
        ctx = super()._build_reward_context(info, linvel, gyro, gravity, dof_pos, dof_vel)
        ctx.joint_range = self._joint_range
        ctx.base_height = base_height_from_scan(self, self._num_envs)
        ctx.linvel_yaw = self._compute_yaw_frame_linvel()
        return ctx

    def _compute_yaw_frame_linvel(self) -> np.ndarray:
        base_linvel = np.asarray(
            self._backend.get_sensor_data("global_linvel")
            if False
            else self._backend.get_sensor_data("local_linvel"),
            dtype=get_global_dtype(),
        )
        return base_linvel

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
        kp_kd = self._backend.get_actuator_gains()
        if kp_kd is None:
            return np.zeros_like(dof_pos)
        kp, kd = kp_kd
        kp = np.asarray(kp, dtype=get_global_dtype())
        kd = np.asarray(kd, dtype=get_global_dtype())
        targets = actions * float(self._cfg.control_config.action_scale) + self.default_angles
        return np.asarray(kp * (targets - dof_pos) - kd * dof_vel, dtype=get_global_dtype())


registry.register_env("G1JoystickRough", G1JoystickRoughEnv, sim_backend="motrix")
