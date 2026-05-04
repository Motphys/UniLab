"""G1 joystick locomotion environments."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from etils import epath

from unilab.assets import ASSETS_ROOT_PATH
from unilab.base import registry
from unilab.base.augmentation import SymmetryObsLayout
from unilab.base.backend import create_backend
from unilab.base.curriculum import EpisodeLengthTracker, PenaltyCurriculum
from unilab.base.np_env import NpEnvState
from unilab.dtype_config import get_global_dtype
from unilab.envs.locomotion.common import rewards
from unilab.envs.locomotion.common.commands import Commands
from unilab.envs.locomotion.common.domain_rand import DomainRandConfig
from unilab.envs.locomotion.common.dr_provider import LocomotionDRProvider
from unilab.envs.locomotion.common.rewards import RewardContext
from unilab.envs.locomotion.g1.base import G1BaseCfg, G1BaseEnv


@dataclass
class G1DomainRandConfig(DomainRandConfig):
    randomize_kp: bool = True
    kp_multiplier_range: list[float] = field(default_factory=lambda: [0.9, 1.1])

    randomize_kd: bool = True
    kd_multiplier_range: list[float] = field(default_factory=lambda: [0.9, 1.1])


@dataclass
class InitState:
    pos = [0.0, 0.0, 0.754]


def sample_gait_phase_pairs(rng, num_samples: int, mode: str) -> np.ndarray:
    if mode == "independent":
        return np.asarray(
            np.column_stack(
                [
                    rng.uniform(0.0, 2.0 * np.pi, size=(num_samples,)),
                    rng.uniform(0.0, 2.0 * np.pi, size=(num_samples,)),
                ]
            ),
            dtype=get_global_dtype(),
        )

    phase = rng.uniform(0.0, 2.0 * np.pi, size=(num_samples,))
    return np.asarray(np.column_stack([phase, phase + np.pi]), dtype=get_global_dtype())


def sample_reset_base_qvel(rng, num_samples: int, limit: float) -> np.ndarray:
    return np.asarray(rng.uniform(-limit, limit, size=(num_samples, 6)), dtype=get_global_dtype())


def build_upper_body_pose_weights(pose_weights: list[float]) -> np.ndarray:
    weights = np.asarray(pose_weights, dtype=get_global_dtype()).copy()
    weights[:12] = 0.0
    return np.asarray(weights, dtype=get_global_dtype())


def compute_feet_phase_height_targets(
    gait_phase: np.ndarray, swing_height: float
) -> tuple[np.ndarray, np.ndarray]:
    def cubic_bezier_height(phi: np.ndarray, swing_height: float) -> np.ndarray:
        phi_normalized = np.fmod(phi + np.pi, 2 * np.pi) - np.pi
        x = (phi_normalized + np.pi) / (2 * np.pi)

        def cubic_bezier_interpolation(
            y_start: np.ndarray, y_end: np.ndarray, t: np.ndarray
        ) -> np.ndarray:
            y_diff = y_end - y_start
            bezier = t**3 + 3 * (t**2 * (1 - t))
            return np.asarray(y_start + y_diff * bezier, dtype=get_global_dtype())

        stance = cubic_bezier_interpolation(np.zeros_like(x), np.full_like(x, swing_height), 2 * x)
        swing = cubic_bezier_interpolation(
            np.full_like(x, swing_height), np.zeros_like(x), 2 * x - 1
        )
        return np.where(x <= 0.5, stance, swing)

    left_target = cubic_bezier_height(gait_phase[:, 0], swing_height)
    right_target = cubic_bezier_height(gait_phase[:, 1], swing_height)
    return left_target, right_target


LEFT_FOOT_CONTACT_SENSORS = [f"left_foot_contact_{i}" for i in range(4)]
RIGHT_FOOT_CONTACT_SENSORS = [f"right_foot_contact_{i}" for i in range(4)]


def _scalarize_sensor_values(sensor_values: np.ndarray) -> np.ndarray:
    sensor_array = np.asarray(sensor_values, dtype=get_global_dtype())
    if sensor_array.ndim == 1:
        return sensor_array
    if sensor_array.ndim == 2 and sensor_array.shape[1] == 1:
        return sensor_array[:, 0]
    raise ValueError(f"Expected scalar sensor values, got shape {sensor_array.shape}")


def compute_aggregated_foot_contact(backend: Any, sensor_names: list[str]) -> np.ndarray:
    contacts = [_scalarize_sensor_values(backend.get_sensor_data(name)) for name in sensor_names]
    return np.asarray(np.any(np.stack(contacts, axis=1) > 0.5, axis=1), dtype=np.bool_)


def compute_feet_phase_contact_targets(
    gait_phase: np.ndarray, swing_height: float
) -> tuple[np.ndarray, np.ndarray]:
    left_target, right_target = compute_feet_phase_height_targets(gait_phase, swing_height)
    contact_height_threshold = swing_height * 0.5
    return left_target <= contact_height_threshold, right_target <= contact_height_threshold


def compute_forward_speed_gate(linvel: np.ndarray, min_forward_speed: float) -> np.ndarray:
    forward_speed = np.maximum(linvel[:, 0], 0.0)
    return np.asarray(forward_speed >= min_forward_speed, dtype=get_global_dtype())


def compute_forward_command_mask(commands: np.ndarray) -> np.ndarray:
    return np.asarray(np.maximum(commands[:, 0], 0.0) > 1.0e-6, dtype=get_global_dtype())


@dataclass
class G1RewardConfig:
    scales: dict[str, float]
    tracking_sigma: float
    gait_frequency: float
    feet_phase_swing_height: float
    feet_phase_tracking_sigma: float
    base_height_target: float
    min_base_height: float
    max_tilt_deg: float
    min_forward_speed_for_gait_reward: float = 0.0
    close_feet_threshold: float = 0.15
    pose_weights: list[float] = field(
        default_factory=lambda: [
            0.01,
            1.0,
            5.0,
            0.01,
            5.0,
            5.0,
            0.01,
            1.0,
            5.0,
            0.01,
            5.0,
            5.0,
            50.0,
            50.0,
            50.0,
            50.0,
            50.0,
            50.0,
            50.0,
            50.0,
            50.0,
            50.0,
            50.0,
            50.0,
            50.0,
            50.0,
            50.0,
            50.0,
            50.0,
        ]
    )


@dataclass
class G1WalkLegacyRewardConfig(G1RewardConfig):
    pass


@dataclass
class CurriculumConfig:
    enabled: bool = False
    initial_scale: float = 0.5
    min_scale: float = 0.5
    max_scale: float = 1.0
    level_down_threshold: float = 150.0
    level_up_threshold: float = 750.0
    degree: float = 0.001


@dataclass
class G1WalkEnvCfg(G1BaseCfg):
    model_file: str = str(ASSETS_ROOT_PATH / "robots" / "g1" / "scene_flat.xml")
    max_episode_seconds: float = 20.0
    init_state: InitState = field(default_factory=InitState)
    commands: Commands = field(default_factory=Commands)
    reward_config: G1RewardConfig | None = None
    domain_rand: G1DomainRandConfig = field(default_factory=G1DomainRandConfig)
    gait_phase_init_mode: str = "offset_phase"
    reset_base_qvel_limit: float = 0.5
    curriculum: CurriculumConfig = field(default_factory=CurriculumConfig)


class G1WalkDomainRandomizationProvider(LocomotionDRProvider):
    def __init__(self, *, base_kp: np.ndarray | None = None, base_kd: np.ndarray | None = None):
        self._base_kp = base_kp
        self._base_kd = base_kd

    def _get_base_actuator_gains(self, env: Any) -> tuple[np.ndarray | None, np.ndarray | None]:
        return self._base_kp, self._base_kd

    def _get_qvel_limit(self, env: Any) -> float:
        return float(env.cfg.reset_base_qvel_limit)

    def _build_extra_info_updates(self, env: Any, num_reset: int) -> dict[str, np.ndarray]:
        return {"gait_phase": self._sample_gait_phase(env, num_reset)}

    def _sample_gait_phase(self, env: Any, num_reset: int) -> np.ndarray:
        mode = env.cfg.gait_phase_init_mode
        if mode == "independent":
            left = np.random.uniform(0.0, 2.0 * np.pi, size=(num_reset,))
            right = np.random.uniform(0.0, 2.0 * np.pi, size=(num_reset,))
            return np.asarray(np.column_stack([left, right]), dtype=get_global_dtype())

        phase = np.random.uniform(0.0, 2.0 * np.pi, size=(num_reset,))
        return np.asarray(np.column_stack([phase, phase + np.pi]), dtype=get_global_dtype())

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
        return env._compute_obs(info_updates, linvel, gyro, gravity, dof_pos, dof_vel)  # type: ignore[no-any-return]


class G1WalkEnv(G1BaseEnv):
    _cfg: G1WalkEnvCfg
    _reward_cfg: Any

    def __init__(self, cfg: G1WalkEnvCfg, num_envs=1, backend_type="mujoco"):
        if cfg.reward_config is None:
            raise ValueError("reward_config must be provided via Hydra configuration")
        backend = create_backend(
            backend_type,
            cfg.model_file,
            num_envs,
            cfg.sim_dt,
            base_name=cfg.asset.base_name,
            push_body_name=cfg.domain_rand.push_body_name,
            iterations=cfg.iterations,
        )
        super().__init__(cfg, backend, num_envs)
        self._enable_reward_log = True
        self._reward_cfg = cfg.reward_config

        self._gait_phase_delta = float(
            2.0 * math.pi * self._reward_cfg.gait_frequency * cfg.ctrl_dt
        )
        self._pose_weights = np.array(self._reward_cfg.pose_weights, dtype=get_global_dtype())
        if self._pose_weights.shape[0] != self._num_action:
            raise ValueError("pose_weights length mismatch")
        self._upper_body_pose_weights = build_upper_body_pose_weights(self._reward_cfg.pose_weights)
        self._episode_tracker: EpisodeLengthTracker | None = None
        self._penalty_curriculum: PenaltyCurriculum | None = None
        if cfg.curriculum.enabled:
            self._episode_tracker = EpisodeLengthTracker(num_envs)
            self._penalty_curriculum = PenaltyCurriculum(
                self,
                enabled=True,
                initial_scale=cfg.curriculum.initial_scale,
                min_scale=cfg.curriculum.min_scale,
                max_scale=cfg.curriculum.max_scale,
                level_down_threshold=cfg.curriculum.level_down_threshold,
                level_up_threshold=cfg.curriculum.level_up_threshold,
                degree=cfg.curriculum.degree,
            )

        self._init_reward_functions()
        if cfg.domain_rand.randomize_kp or cfg.domain_rand.randomize_kd:
            base_kp, base_kd = backend.get_actuator_gains()
            dr_provider = G1WalkDomainRandomizationProvider(base_kp=base_kp, base_kd=base_kd)
        else:
            dr_provider = G1WalkDomainRandomizationProvider()
        self._init_domain_randomization(dr_provider)

    @property
    def obs_groups_spec(self) -> dict[str, int]:
        # gyro(3) + gravity(3) + diff(29) + dof_vel(29) + action(29) + cmd(3) + phase(2) = 98
        return {"obs": 98, "critic": 101}

    def _init_reward_functions(self):
        self._reward_fns: dict[str, Any] = {
            "tracking_lin_vel": rewards.tracking_lin_vel,
            "tracking_ang_vel": rewards.tracking_ang_vel,
            "forward_progress": rewards.forward_progress,
            "under_speed": rewards.under_speed,
            "lin_vel_z": rewards.lin_vel_z,
            "orientation": rewards.orientation,
            "penalty_orientation": rewards.orientation,
            "ang_vel_xy": rewards.ang_vel_xy,
            "penalty_ang_vel_xy": rewards.ang_vel_xy,
            "action_rate": rewards.action_rate,
            "penalty_action_rate": rewards.action_rate,
            "base_height": rewards.base_height,
            "pose": rewards.weighted_pose,
            "upper_body_pose": self._reward_upper_body_pose,
            "penalty_close_feet_xy": self._reward_close_feet_xy,
            "penalty_feet_ori": self._reward_feet_ori,
            "feet_phase": self._reward_feet_phase,
            "feet_phase_contrast": self._reward_feet_phase_contrast,
            "feet_phase_contact": self._reward_feet_phase_contact,
            "feet_double_stance": self._reward_feet_double_stance,
            "feet_air_time": self._reward_feet_air_time,
            "alive": rewards.alive,
        }

    def update_state(self, state: NpEnvState) -> NpEnvState:
        linvel = self.get_local_linvel()
        gyro = self.get_gyro()
        gravity = self._backend.get_sensor_data("upvector")
        dof_pos = self.get_dof_pos()
        dof_vel = self.get_dof_vel()

        max_tilt_rad = np.deg2rad(self._reward_cfg.max_tilt_deg)
        tilt = np.arccos(np.clip(gravity[:, 2], -1, 1))
        terminated = np.logical_or(
            tilt > max_tilt_rad,
            self._backend.get_base_pos()[:, 2] < self._reward_cfg.min_base_height,
        )

        reward = self._compute_reward(state.info, linvel, gyro, gravity, dof_pos, dof_vel)
        obs = self._compute_obs(state.info, linvel, gyro, gravity, dof_pos, dof_vel)
        state = state.replace(obs=obs, reward=reward, terminated=terminated)

        done = state.terminated | state.truncated
        if self._episode_tracker is None or self._penalty_curriculum is None or not np.any(done):
            return state

        done_indices = np.where(done)[0]
        episode_lengths = state.info["steps"][done_indices] + 1
        self._episode_tracker.update(episode_lengths)
        self._penalty_curriculum.update(self._episode_tracker.average_length)

        if "log" not in state.info:
            state.info["log"] = {}
        state.info["log"]["curriculum/average_episode_length"] = float(
            self._episode_tracker.average_length
        )
        state.info["log"]["curriculum/penalty_scale"] = float(
            self._penalty_curriculum.current_scale
        )
        return state

    def _compute_obs(
        self, info: dict, linvel, gyro, gravity, dof_pos, dof_vel
    ) -> dict[str, np.ndarray]:
        noise_cfg = self._cfg.noise_config
        diff = dof_pos - self.default_angles
        command = info["commands"]
        last_actions = info.get("current_actions", np.zeros_like(diff))
        gait_phase = info.get("gait_phase", np.zeros((self._num_envs, 2), dtype=get_global_dtype()))
        walk_profile = self._uses_walk_observation_profile()

        noisy_gyro = self._obs_noise(gyro, noise_cfg.scale_gyro)
        noisy_gravity = self._obs_noise(gravity, noise_cfg.scale_gravity)
        noisy_diff = self._obs_noise(diff, noise_cfg.scale_joint_angle)
        noisy_dof_vel = self._obs_noise(dof_vel, noise_cfg.scale_joint_vel)
        actor_gyro_scale = 0.25 if walk_profile else 1.0
        actor_dof_vel_scale = 0.05 if walk_profile else 1.0

        actor = np.concatenate(
            [
                noisy_gyro * actor_gyro_scale,
                -noisy_gravity,
                noisy_diff,
                noisy_dof_vel * actor_dof_vel_scale,
                last_actions,
                command,
                gait_phase,
            ],
            axis=1,
            dtype=get_global_dtype(),
        )

        critic_gyro_scale = 0.25 if walk_profile else 1.0
        critic_dof_vel_scale = 0.05 if walk_profile else 1.0
        critic_linvel_scale = 2.0 if walk_profile else 1.0
        critic_base = np.concatenate(
            [
                gyro * critic_gyro_scale,
                -gravity,
                diff,
                dof_vel * critic_dof_vel_scale,
                last_actions,
                command,
                gait_phase,
            ],
            axis=1,
            dtype=get_global_dtype(),
        )
        critic = np.concatenate(
            [
                critic_base,
                np.asarray(linvel * critic_linvel_scale, dtype=get_global_dtype()),
            ],
            axis=1,
            dtype=get_global_dtype(),
        )

        return {"obs": actor, "critic": critic}

    def _uses_walk_observation_profile(self) -> bool:
        scales = getattr(getattr(self, "_reward_cfg", None), "scales", None)
        if scales is None:
            reward_cfg = getattr(self._cfg, "reward_config", None)
            scales = getattr(reward_cfg, "scales", None)

        if scales is not None:
            if any(
                key in scales
                for key in (
                    "penalty_orientation",
                    "penalty_ang_vel_xy",
                    "penalty_action_rate",
                    "alive",
                )
            ):
                return True
            if any(key in scales for key in ("orientation", "ang_vel_xy", "action_rate")):
                return False

        curriculum = getattr(self._cfg, "curriculum", None)
        return bool(curriculum is not None and curriculum.enabled)

    def _actor_symmetry_obs_layout(self) -> SymmetryObsLayout:
        return (
            ("gyro", 3),
            ("gravity", 3),
            ("dof_pos", self._num_action),
            ("dof_vel", self._num_action),
            ("actions", self._num_action),
            ("command", 3),
            ("gait_phase", 2),
        )

    def get_symmetry_obs_layouts(self) -> dict[str, SymmetryObsLayout]:
        actor_layout = self._actor_symmetry_obs_layout()
        return {
            "obs": actor_layout,
            "critic": (*actor_layout, ("linvel", 3)),
        }

    def build_symmetry_augmentation(self, *, device: str):
        if self._backend.backend_type != "mujoco":
            return None
        from unilab.envs.locomotion.g1.symmetry import G1SymmetryAugmentation

        return G1SymmetryAugmentation(
            self._backend.model,
            self.get_symmetry_obs_layouts(),
            device=device,
        )

    def _build_reward_context(
        self, info: dict, linvel, gyro, gravity, dof_pos, dof_vel
    ) -> RewardContext:
        return RewardContext(
            info=info,
            linvel=linvel,
            gyro=gyro,
            dof_pos=dof_pos,
            num_envs=self._num_envs,
            default_angles=self.default_angles,
            tracking_sigma=self._reward_cfg.tracking_sigma,
            base_height_target=self._reward_cfg.base_height_target,
            base_height=self._backend.get_base_pos()[:, 2],
            gravity=gravity,
            dof_vel=dof_vel,
            pose_weights=self._pose_weights,
        )

    def _compute_reward(self, info: dict, linvel, gyro, gravity, dof_pos, dof_vel) -> np.ndarray:
        dtype = get_global_dtype()
        reward = np.zeros((self._num_envs,), dtype=dtype)
        cfg = self._reward_cfg
        ctx = self._build_reward_context(info, linvel, gyro, gravity, dof_pos, dof_vel)

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

    def _reward_feet_phase(self, ctx: RewardContext):
        """步态相位奖励：鼓励正确的摆动腿高度"""
        left_foot = self._backend.get_sensor_data("left_foot_pos")
        right_foot = self._backend.get_sensor_data("right_foot_pos")
        gait_phase = ctx.info.get(
            "gait_phase", np.zeros((self._num_envs, 2), dtype=get_global_dtype())
        )
        swing_height = self._reward_cfg.feet_phase_swing_height
        left_target, right_target = compute_feet_phase_height_targets(gait_phase, swing_height)
        left_error = np.square(left_foot[:, 2] - left_target)
        right_error = np.square(right_foot[:, 2] - right_target)
        reward = np.exp(-(left_error + right_error) / self._reward_cfg.feet_phase_tracking_sigma)
        return np.asarray(reward * self._gait_reward_gate(ctx.linvel), dtype=get_global_dtype())

    def _gait_reward_gate(self, linvel: np.ndarray) -> np.ndarray:
        min_forward_speed = getattr(self._reward_cfg, "min_forward_speed_for_gait_reward", 0.0)
        return compute_forward_speed_gate(linvel, min_forward_speed)

    def _reward_feet_phase_contrast(self, ctx: RewardContext):
        left_foot = self._backend.get_sensor_data("left_foot_pos")
        right_foot = self._backend.get_sensor_data("right_foot_pos")
        gait_phase = ctx.info.get(
            "gait_phase", np.zeros((self._num_envs, 2), dtype=get_global_dtype())
        )
        swing_height = self._reward_cfg.feet_phase_swing_height
        left_target, right_target = compute_feet_phase_height_targets(gait_phase, swing_height)
        actual_delta = left_foot[:, 2] - right_foot[:, 2]
        target_delta = left_target - right_target
        error = np.square(actual_delta - target_delta)
        reward = np.exp(-error / self._reward_cfg.feet_phase_tracking_sigma)
        return np.asarray(reward * self._gait_reward_gate(ctx.linvel), dtype=get_global_dtype())

    def _reward_feet_phase_contact(self, ctx: RewardContext):
        gait_phase = ctx.info.get(
            "gait_phase", np.zeros((self._num_envs, 2), dtype=get_global_dtype())
        )
        swing_height = self._reward_cfg.feet_phase_swing_height
        left_target_contact, right_target_contact = compute_feet_phase_contact_targets(
            gait_phase, swing_height
        )
        left_contact = compute_aggregated_foot_contact(self._backend, LEFT_FOOT_CONTACT_SENSORS)
        right_contact = compute_aggregated_foot_contact(self._backend, RIGHT_FOOT_CONTACT_SENSORS)
        left_match = np.asarray(left_contact == left_target_contact, dtype=get_global_dtype())
        right_match = np.asarray(right_contact == right_target_contact, dtype=get_global_dtype())
        reward = np.asarray(0.5 * (left_match + right_match), dtype=get_global_dtype())
        return np.asarray(reward * self._gait_reward_gate(ctx.linvel), dtype=get_global_dtype())

    def _reward_feet_double_stance(self, ctx: RewardContext):
        commands = ctx.info.get("commands", np.zeros((self._num_envs, 3), dtype=get_global_dtype()))
        left_contact = compute_aggregated_foot_contact(self._backend, LEFT_FOOT_CONTACT_SENSORS)
        right_contact = compute_aggregated_foot_contact(self._backend, RIGHT_FOOT_CONTACT_SENSORS)
        double_stance = np.asarray(
            np.logical_and(left_contact, right_contact), dtype=get_global_dtype()
        )
        return np.asarray(
            double_stance * compute_forward_command_mask(commands), dtype=get_global_dtype()
        )

    def _reward_feet_ori(self, ctx: RewardContext):
        left_foot_quat = self._backend.get_sensor_data("left_foot_quat")
        right_foot_quat = self._backend.get_sensor_data("right_foot_quat")
        return (
            np.square(left_foot_quat[:, 1])
            + np.square(left_foot_quat[:, 2])
            + np.square(right_foot_quat[:, 1])
            + np.square(right_foot_quat[:, 2])
        )

    def _reward_close_feet_xy(self, ctx: RewardContext):
        left_foot = self._backend.get_sensor_data("left_foot_pos")
        right_foot = self._backend.get_sensor_data("right_foot_pos")
        feet_dist = np.linalg.norm(left_foot[:, :2] - right_foot[:, :2], axis=1)
        return np.where(
            feet_dist < self._reward_cfg.close_feet_threshold,
            np.square(feet_dist - self._reward_cfg.close_feet_threshold),
            0.0,
        )

    def _reward_feet_air_time(self, ctx: RewardContext):
        air_time = ctx.info.get(
            "feet_air_time", np.zeros((self._num_envs, 2), dtype=get_global_dtype())
        )
        in_range = (air_time > 0.05) & (air_time < 0.5)
        return np.sum(in_range.astype(float), axis=1)

    def _reward_upper_body_pose(self, ctx: RewardContext):
        diff = ctx.dof_pos - self.default_angles
        return np.asarray(
            np.sum(self._upper_body_pose_weights * np.square(diff), axis=1),
            dtype=get_global_dtype(),
        )

    def apply_action(self, actions: np.ndarray, state: NpEnvState) -> np.ndarray:
        state.info["last_actions"] = state.info.get("current_actions", np.zeros_like(actions))
        state.info["current_actions"] = actions

        gait_phase = state.info.get(
            "gait_phase", np.zeros((self._num_envs, 2), dtype=get_global_dtype())
        )
        gait_phase[:, 0] = (gait_phase[:, 0] + self._gait_phase_delta) % (2 * np.pi)
        gait_phase[:, 1] = (gait_phase[:, 1] + self._gait_phase_delta) % (2 * np.pi)
        state.info["gait_phase"] = gait_phase

        ctrl: np.ndarray = actions * self._cfg.control_config.action_scale + self.default_angles
        return ctrl


def _walk_curriculum() -> CurriculumConfig:
    return CurriculumConfig(
        enabled=True,
        initial_scale=0.5,
        min_scale=0.5,
        max_scale=1.0,
        level_down_threshold=150.0,
        level_up_threshold=750.0,
        degree=0.001,
    )


@dataclass
class G1WalkControlConfig:
    action_scale: float = 1.0
    simulate_action_latency: bool = False


@dataclass
class G1WalkRewardConfig(G1RewardConfig):
    """对齐 holosoma G1 walking 奖励权重。"""


@registry.envcfg("G1WalkFlat")
@dataclass
class G1WalkFlatCfg(G1WalkEnvCfg):
    reward_config: G1WalkRewardConfig | None = None
    model_file: str = str(ASSETS_ROOT_PATH / "robots" / "g1" / "scene_flat.xml")
    control_config: G1WalkControlConfig = field(default_factory=G1WalkControlConfig)  # type: ignore[assignment]
    curriculum: CurriculumConfig = field(default_factory=_walk_curriculum)


@registry.envcfg("G1WalkRough")
@dataclass
class G1WalkRoughCfg(G1WalkFlatCfg):
    model_file: str = str(ASSETS_ROOT_PATH / "robots" / "g1" / "scene_rough.xml")


registry.register_env("G1WalkFlat", G1WalkEnv, sim_backend="mujoco")
registry.register_env("G1WalkFlat", G1WalkEnv, sim_backend="motrix")
registry.register_env("G1WalkRough", G1WalkEnv, sim_backend="mujoco")
registry.register_env("G1WalkRough", G1WalkEnv, sim_backend="motrix")
