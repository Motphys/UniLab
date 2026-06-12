from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, cast

import numpy as np
from etils import epath

from unilab.assets import ASSETS_ROOT_PATH
from unilab.base import registry
from unilab.base.backend import create_backend
from unilab.base.np_env import NpEnvState
from unilab.base.scene import SceneCfg
from unilab.dr import ResetPlan, ResetRandomizationPayload
from unilab.dtype_config import get_global_dtype
from unilab.envs.common.rotation import np_quat_apply, np_quat_apply_inverse
from unilab.envs.locomotion.common import rewards
from unilab.envs.locomotion.common.commands import Commands
from unilab.envs.locomotion.common.domain_rand import DomainRandConfig
from unilab.envs.locomotion.common.dr_provider import LocomotionDRProvider
from unilab.envs.locomotion.common.rewards import RewardContext
from unilab.envs.locomotion.go2.base import ControlConfig, Go2BaseCfg, Go2BaseEnv, NoiseConfig


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
    knee_height_target: float = 0.08


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


@dataclass
class Go2HandStandCfg(Go2BaseCfg):
    scene: SceneCfg = field(
        default_factory=lambda: SceneCfg(
            model_file=str(ASSETS_ROOT_PATH / "robots" / "go2" / "scene_flat.xml")
        )
    )
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
        env.feet_phase[env_ids, :] = 0
        env.feet_phase[:, 2] = 0.0
        env.feet_phase[:, 3] = 0.5
        env._feet_air_time[env_ids, :] = 0.0
        env._last_contacts[env_ids, :] = False

        return env._compute_obs(  # type: ignore[no-any-return]
            info_updates,
            linvel,
            gyro,
            gravity,
            dof_pos,
            dof_vel,
            height,
        )


class Go2HandStandTask(Go2BaseEnv):
    _cfg: Go2HandStandCfg

    def __init__(self, cfg: Go2HandStandCfg, num_envs=1, backend_type="mujoco"):
        if cfg.reward_config is None:
            raise ValueError("reward_config must be provided via Hydra configuration")
        backend = create_backend(
            backend_type,
            cfg.scene,
            num_envs,
            cfg.sim_dt,
            base_name=cfg.asset.base_name,
            push_body_name=cfg.domain_rand.push_body_name,
            add_body_sensors=bool(getattr(cfg, "add_body_sensors", False)),
            position_actuator_gains={"kp": cfg.control_config.Kp, "kd": cfg.control_config.Kd},
            motrix_max_iterations=cfg.motrix_max_iterations,
            post_step_forward_sensor=cfg.post_step_forward_sensor,
        )
        super().__init__(cfg, backend, num_envs)
        self._enable_reward_log = True
        self._reward_cfg = cfg.reward_config
        self._init_reward_functions()
        self._init_task_domain_randomization()
        self.phase = np.zeros((num_envs,), dtype=np.float32)
        self.feet_phase = np.zeros((num_envs, len(cfg.sensor.feet_force)), dtype=np.float32)
        self.feet_phase[:, 2] = 0.0
        self.feet_phase[:, 3] = 0.5
        self.gait_frequency = 2
        self.feet_force = np.zeros((num_envs, len(cfg.sensor.feet_force), 1), dtype=np.float32)
        self._feet_air_time = np.zeros((num_envs, len(cfg.sensor.feet_force)), dtype=np.float32)
        self._last_contacts = np.zeros((num_envs, 2), dtype=bool)
        self.feet_pos = np.zeros((num_envs, len(cfg.sensor.feet_pos), 3), dtype=np.float32)
        self.torso_height = np.zeros((num_envs,), dtype=np.float32)
        self._z_des = 0.55
        self._desired_gravity = np.array([-1, 0, 0])
        self.feet_geom_names = [0, 1]
        self._joint_ids = [0, 1, 2, 3, 4, 5, 6, 9]
        self._tar_ids = [6, 7, 8, 9, 10, 11]
        self.target_angle = np.array([0, 1.82, -1.16, 0.0, 1.82, -1.16])

    def _init_task_domain_randomization(self) -> None:
        self._init_domain_randomization(Go2HandStandDomainRandomizationProvider())

    @property
    def obs_groups_spec(self) -> dict[str, int]:
        return {"obs": 42, "critic": 46}

    def _init_reward_functions(self):
        self._reward_fns: dict[str, Any] = {}

    def update_state(self, state: NpEnvState) -> NpEnvState:
        return state

    def _compute_obs(
        self,
        info: dict,
        linvel,
        gyro,
        gravity,
        dof_pos,
        dof_vel,
        height,
    ) -> dict[str, np.ndarray]:
        return {"obs": np.zeros((self._num_envs, 1)), "critic": np.zeros((self._num_envs, 1))}

    def _compute_reward(self, info: dict, linvel, gyro, dof_pos) -> np.ndarray:
        return np.zeros((self._num_envs,), dtype=get_global_dtype())

    def _cost_pose(self, ctx: RewardContext) -> np.ndarray:
        dof_pos = self.get_dof_pos()
        error = dof_pos[:, self._joint_ids] - self.default_angles[self._joint_ids]
        return cast(np.ndarray, np.sum(np.square(error), axis=1))

    def _reward_penalty_contact(self, ctx: RewardContext) -> np.ndarray:
        contact_arrays = []
        for name in self._cfg.sensor.penalty_contact:
            arr = self._backend.get_sensor_data(name)
            contact_arrays.append(arr)
        result = np.concatenate(contact_arrays, axis=1)
        return np.asarray(np.any(result, axis=1))

    def _reward_tar(self, ctx: RewardContext) -> np.ndarray:
        dof_pos = self.get_dof_pos()
        error = dof_pos[:, self._tar_ids] - self.target_angle
        error = np.sum(np.square(error), axis=1)
        mask = (self.torso_height >= self._z_des * 0.8).astype(np.float32)
        return cast(np.ndarray, np.exp(-error / 1) * mask)


_GO2_DOF_TO_CTRL = np.array([3, 4, 5, 0, 1, 2, 9, 10, 11, 6, 7, 8], dtype=np.int32)
_WORLD_GRAVITY = np.array([0.0, 0.0, -1.0], dtype=np.float32)
_BODY_FORWARD = np.array([1.0, 0.0, 0.0], dtype=np.float32)
_FOOTSTAND_FRAME_OBS_DIM = 45
_FOOTSTAND_PRIVILEGED_TAIL_DIM = 49
_FOOTSTAND_MIN_OBS_HISTORY_LEN = 15
_FOOTSTAND_FRONT_FEET = [0, 1]
_FOOTSTAND_REAR_FEET = [2, 3]
_FOOTSTAND_FRONT_LEG_IDS = [0, 1, 2, 3, 4, 5]
_FOOTSTAND_REAR_LEG_IDS = [6, 7, 8, 9, 10, 11]
_FOOTSTAND_REAR_LEFT_LEG_IDS = [6, 7, 8]
_FOOTSTAND_REAR_RIGHT_LEG_IDS = [9, 10, 11]
_FOOTSTAND_REAR_MIRROR_SIGNS = np.array([-1.0, 1.0, 1.0], dtype=np.float32)
_FOOTSTAND_FRONT_LEG_TARGET = np.array([0.0, 1.82, -1.16, 0.0, 1.82, -1.16])
_FOOTSTAND_KNEE_BODY_NAMES = ("FL_calf", "FR_calf", "RL_calf", "RR_calf")
_FOOTSTAND_CONTACT_THRESHOLD = 0.1
_FOOTSTAND_STAND_HEIGHT_FRACTION = 0.8
_FOOTSTAND_STAND_ORIENTATION_THRESHOLD = 0.5


@dataclass
class FootstandNoiseConfig(NoiseConfig):
    level: float = 1.0
    scale_joint_angle: float = 0.01
    scale_joint_vel: float = 1.5
    scale_gyro: float = 0.2
    scale_gravity: float = 0.05
    scale_linvel: float = 0.1


@dataclass
class FootstandControlConfig(ControlConfig):
    clip_actions: float = 1.0


@dataclass
class Go2FootStandDomainRandConfig(Go2DomainRandConfig):
    randomize_kp: bool = False
    randomize_kd: bool = False
    randomize_base_mass: bool = False
    random_com: bool = False
    push_robots: bool = False

    randomize_floor_friction: bool = True
    floor_friction_range: list[float] = field(default_factory=lambda: [0.4, 1.0])

    randomize_link_mass: bool = True
    link_mass_scale_range: list[float] = field(default_factory=lambda: [0.9, 1.1])
    torso_added_mass_range: list[float] = field(default_factory=lambda: [-1.0, 1.0])

    randomize_torso_com: bool = True
    torso_com_offset_range: list[float] = field(default_factory=lambda: [-0.05, 0.05])

    randomize_dof_armature: bool = True
    dof_armature_scale_range: list[float] = field(default_factory=lambda: [1.0, 1.05])

    randomize_reset_joint_qpos: bool = True
    reset_joint_qpos_range: list[float] = field(default_factory=lambda: [-0.05, 0.05])


@dataclass
class FootstandSensor(JoystickSensor):
    accelerometer = "accelerometer"
    global_angvel = "global_angvel"
    ternamate_contact = [
        "RL_hip_contact",
        "RR_hip_contact",
        "RL_thigh_contact",
        "RR_thigh_contact",
        "RL_calf_contact1",
        "RL_calf_contact2",
        "RR_calf_contact1",
        "RR_calf_contact2",
    ]
    penalty_contact = [
        "FL_hip_contact",
        "FR_hip_contact",
        "FL_thigh_contact",
        "FR_thigh_contact",
        "FL_calf_contact1",
        "FL_calf_contact2",
        "FR_calf_contact1",
        "FR_calf_contact2",
    ]


@registry.envcfg("Go2FootStand")
@dataclass
class Go2FootStandCfg(Go2HandStandCfg):
    max_episode_seconds: float = 10.0
    add_body_sensors: bool = True
    obs_history_len: int = _FOOTSTAND_MIN_OBS_HISTORY_LEN
    soft_joint_pos_limit_factor: float = 0.9
    energy_termination_threshold: float = np.inf
    termination_grace_steps: int = 100
    termination_height_fraction: float = 0.8
    termination_orientation_threshold: float = 0.2
    noise_config: FootstandNoiseConfig = field(default_factory=FootstandNoiseConfig)  # type: ignore[assignment]
    control_config: FootstandControlConfig = field(  # type: ignore[assignment]
        default_factory=lambda: FootstandControlConfig(action_scale=0.3)
    )
    sensor: FootstandSensor = field(default_factory=FootstandSensor)  # type: ignore[assignment]
    domain_rand: Go2FootStandDomainRandConfig = field(default_factory=Go2FootStandDomainRandConfig)  # type: ignore[assignment]


class Go2FootStandDomainRandomizationProvider(Go2HandStandDomainRandomizationProvider):
    def _get_reset_randomization_baselines(
        self, env: Any
    ) -> tuple[np.ndarray | None, np.ndarray | None, int | None, np.ndarray | None]:
        return (
            env._base_body_mass,
            env._base_geom_friction,
            env._floor_geom_id,
            env._base_dof_armature,
        )

    def build_reset_plan(self, env: Any, env_ids: np.ndarray) -> ResetPlan:
        plan = super().build_reset_plan(env, env_ids)
        qpos = np.asarray(plan.qpos, dtype=get_global_dtype()).copy()
        domain_rand = env.cfg.domain_rand
        if domain_rand.randomize_reset_joint_qpos:
            low, high = domain_rand.reset_joint_qpos_range
            qpos[:, -env._num_action :] += np.random.uniform(
                low, high, size=(len(env_ids), env._num_action)
            ).astype(qpos.dtype)

        return ResetPlan(
            env_ids=plan.env_ids,
            qpos=qpos,
            qvel=plan.qvel,
            info_updates=plan.info_updates,
            randomization=self._merge_reset_randomization(
                plan.randomization,
                env._build_playground_reset_randomization(len(env_ids)),
            ),
        )

    @staticmethod
    def _merge_reset_randomization(
        base: ResetRandomizationPayload | None,
        override: ResetRandomizationPayload | None,
    ) -> ResetRandomizationPayload | None:
        if base is None or base.is_empty():
            return override
        if override is None or override.is_empty():
            return base
        return ResetRandomizationPayload(
            base_mass_delta=base.base_mass_delta,
            base_com_offset=base.base_com_offset,
            gravity=base.gravity,
            body_iquat=override.body_iquat if override.body_iquat is not None else base.body_iquat,
            body_inertia=override.body_inertia
            if override.body_inertia is not None
            else base.body_inertia,
            body_ipos=override.body_ipos if override.body_ipos is not None else base.body_ipos,
            body_mass=override.body_mass if override.body_mass is not None else base.body_mass,
            dof_armature=override.dof_armature
            if override.dof_armature is not None
            else base.dof_armature,
            geom_friction=override.geom_friction
            if override.geom_friction is not None
            else base.geom_friction,
            kp=base.kp,
            kd=base.kd,
        )

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
        height = env._backend.get_sensor_data(env._cfg.sensor.global_pos)[env_ids, -1].reshape(
            -1, 1
        )
        local_gravity = env._get_local_gravity()[env_ids]
        accelerometer = env._backend.get_sensor_data(env._cfg.sensor.accelerometer)[env_ids]
        global_angvel = env._backend.get_sensor_data(env._cfg.sensor.global_angvel)[env_ids]
        env.torso_height[env_ids] = height[:, 0]
        env._last_dof_vel_for_acc[env_ids, :] = dof_vel
        env._last_terminated[env_ids] = False
        env._motor_targets[env_ids] = env._dof_to_ctrl_order(dof_pos)
        target_dof = env._ctrl_to_dof_order(env._motor_targets[env_ids])
        info_updates["torques"] = np.asarray(
            env._cfg.control_config.Kp * (target_dof - dof_pos)
            - env._cfg.control_config.Kd * dof_vel,
            dtype=get_global_dtype(),
        )

        return env._compute_obs(  # type: ignore[no-any-return]
            info_updates,
            linvel,
            gyro,
            local_gravity,
            dof_pos,
            dof_vel,
            height,
            accelerometer,
            global_angvel,
            env_ids=env_ids,
        )


@registry.env("Go2FootStand", sim_backend="mujoco")
class Go2FootStandTask(Go2HandStandTask):
    _cfg: Go2FootStandCfg

    def __init__(self, cfg: Go2FootStandCfg, num_envs=1, backend_type="mujoco"):
        super().__init__(cfg, num_envs=num_envs, backend_type=backend_type)
        self._z_des = 0.53
        self._desired_forward_vec = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        self._init_footstand_pose_targets()
        self._init_soft_joint_limits()
        self._init_motor_target_limits()
        self._last_dof_vel_for_acc = np.zeros((num_envs, self._num_action), dtype=np.float32)
        self._last_terminated = np.zeros((num_envs,), dtype=bool)
        self._motor_targets = np.zeros((num_envs, self._num_action), dtype=get_global_dtype())
        self._obs_history = np.zeros(
            (num_envs, self._obs_history_len, _FOOTSTAND_FRAME_OBS_DIM),
            dtype=get_global_dtype(),
        )
        self._critic_obs_history = np.zeros_like(self._obs_history)
        self._base_geom_friction = self._backend.get_geom_friction()
        self._floor_geom_id = self._backend.get_geom_id(self._cfg.asset.ground)
        self._base_body_id = self._backend.get_body_id(self._cfg.asset.base_name)
        self._base_body_mass = self._backend.get_body_mass()
        self._base_body_ipos = self._backend.get_body_ipos()
        self._base_dof_armature = self._backend.get_dof_armature()
        self._knee_body_ids = self._backend.get_body_ids(_FOOTSTAND_KNEE_BODY_NAMES)
        self._init_domain_randomization(Go2FootStandDomainRandomizationProvider())

    def _init_task_domain_randomization(self) -> None:
        pass

    def _init_footstand_pose_targets(self) -> None:
        self.feet_geom_names = list(_FOOTSTAND_FRONT_FEET)
        self._joint_ids = list(_FOOTSTAND_REAR_LEG_IDS)
        self._tar_ids = list(_FOOTSTAND_FRONT_LEG_IDS)
        self.target_angle = np.asarray(_FOOTSTAND_FRONT_LEG_TARGET, dtype=get_global_dtype())

    @property
    def obs_groups_spec(self) -> dict[str, int]:
        # Playground state:
        # linvel(3) + gyro(3) + gravity(3) + diff(12) + dof_vel(12) + last_action(12) = 45.
        # UniLab stacks the actor state for short-horizon dynamics; critic appends current privileged tail.
        obs_dim = _FOOTSTAND_FRAME_OBS_DIM * self._obs_history_len
        return {"obs": obs_dim, "critic": obs_dim + _FOOTSTAND_PRIVILEGED_TAIL_DIM}

    @property
    def _obs_history_len(self) -> int:
        return max(_FOOTSTAND_MIN_OBS_HISTORY_LEN, int(self._cfg.obs_history_len))

    def _build_playground_reset_randomization(
        self, num_reset: int
    ) -> ResetRandomizationPayload | None:
        domain_rand = self._cfg.domain_rand
        payload = ResetRandomizationPayload()

        if domain_rand.randomize_floor_friction:
            low, high = domain_rand.floor_friction_range
            geom_friction = np.broadcast_to(
                self._base_geom_friction, (num_reset, *self._base_geom_friction.shape)
            ).copy()
            geom_friction[:, self._floor_geom_id, 0] = np.random.uniform(
                low, high, size=(num_reset,)
            )
            payload.geom_friction = geom_friction

        body_mass = None
        if domain_rand.randomize_link_mass:
            low, high = domain_rand.link_mass_scale_range
            scale = np.random.uniform(low, high, size=(num_reset, self._base_body_mass.size))
            body_mass = self._base_body_mass.reshape(1, -1) * scale
        if domain_rand.torso_added_mass_range is not None:
            low, high = domain_rand.torso_added_mass_range
            if body_mass is None:
                body_mass = np.broadcast_to(
                    self._base_body_mass, (num_reset, self._base_body_mass.size)
                ).copy()
            body_mass[:, self._base_body_id] += np.random.uniform(low, high, size=(num_reset,))
        if body_mass is not None:
            payload.body_mass = body_mass.astype(np.float64, copy=False)

        if domain_rand.randomize_torso_com:
            low, high = domain_rand.torso_com_offset_range
            body_ipos = np.broadcast_to(
                self._base_body_ipos, (num_reset, *self._base_body_ipos.shape)
            ).copy()
            body_ipos[:, self._base_body_id, :] += np.random.uniform(low, high, size=(num_reset, 3))
            payload.body_ipos = body_ipos

        if domain_rand.randomize_dof_armature:
            low, high = domain_rand.dof_armature_scale_range
            dof_armature = np.broadcast_to(
                self._base_dof_armature, (num_reset, self._base_dof_armature.size)
            ).copy()
            dof_armature[:, -self._num_action :] *= np.random.uniform(
                low, high, size=(num_reset, self._num_action)
            )
            payload.dof_armature = dof_armature

        return None if payload.is_empty() else payload

    def _init_reward_functions(self):
        self._reward_fns: dict[str, Any] = {
            "height": self._reward_height,
            "contact": self._cost_contact,
            "orientation": self._reward_orientation,
            "oritentation": self._reward_orientation,
            "action_rate": rewards.action_rate,
            "termination": self._reward_termination,
            "dof_pos_limits": self._cost_joint_pos_limits,
            "torques": self._cost_torques,
            "pose": self._cost_pose,
            "penalty_contact": self._reward_penalty_contact,
            "tar": self._reward_tar,
            "rear_feet_contact": self._reward_rear_feet_contact,
            "rear_leg_symmetry": self._cost_rear_leg_symmetry,
            "front_leg_motion": self._cost_front_leg_motion,
            "upright_stability": self._cost_upright_stability,
            "knee_clearance": self._cost_knee_clearance,
            "stay_still": self._cost_stay_still,
            "energy": rewards.energy,
            "dof_acc": rewards.dof_acc,
        }

    def _dof_to_ctrl_order(self, values: np.ndarray) -> np.ndarray:
        return np.asarray(values[:, _GO2_DOF_TO_CTRL], dtype=get_global_dtype())

    def _ctrl_to_dof_order(self, values: np.ndarray) -> np.ndarray:
        return np.asarray(values[:, _GO2_DOF_TO_CTRL], dtype=get_global_dtype())

    def _get_local_gravity(self) -> np.ndarray:
        gravity = np.broadcast_to(_WORLD_GRAVITY, (self._num_envs, 3))
        return np.asarray(
            np_quat_apply_inverse(self._backend.get_base_quat(), gravity), dtype=get_global_dtype()
        )

    def _get_body_forward(self) -> np.ndarray:
        forward = np.broadcast_to(_BODY_FORWARD, (self._num_envs, 3))
        return np.asarray(
            np_quat_apply(self._backend.get_base_quat(), forward), dtype=get_global_dtype()
        )

    def _reward_height(self, ctx: RewardContext) -> np.ndarray:
        del ctx
        error = np.abs(self._z_des - self.torso_height)
        return np.asarray(np.exp(-error / 0.1), dtype=get_global_dtype())

    def _standing_mask(self) -> np.ndarray:
        height_ready = self.torso_height >= self._z_des * _FOOTSTAND_STAND_HEIGHT_FRACTION
        orientation_ready = self._orientation_score() >= _FOOTSTAND_STAND_ORIENTATION_THRESHOLD
        return np.asarray(height_ready & orientation_ready, dtype=get_global_dtype())

    def apply_action(self, actions: np.ndarray, state: NpEnvState) -> np.ndarray:
        clip_actions = float(getattr(self._cfg.control_config, "clip_actions", np.inf))
        actions_np = np.asarray(actions, dtype=get_global_dtype())
        if np.isfinite(clip_actions):
            actions_np = np.clip(actions_np, -clip_actions, clip_actions)

        state.info["last_actions"] = state.info.get("current_actions", np.zeros_like(actions_np))
        state.info["current_actions"] = actions_np
        exec_actions = (
            state.info["last_actions"]
            if self._cfg.control_config.simulate_action_latency
            else actions_np
        )
        self._motor_targets += exec_actions * self._cfg.control_config.action_scale
        self._clip_motor_targets()
        return np.asarray(self._motor_targets, dtype=get_global_dtype())

    def update_state(self, state: NpEnvState) -> NpEnvState:
        linvel = self.get_local_linvel()
        gyro = self.get_gyro()
        upvector = self._backend.get_sensor_data("upvector")
        gravity = self._get_local_gravity()
        accelerometer = self._backend.get_sensor_data(self._cfg.sensor.accelerometer)
        global_angvel = self._backend.get_sensor_data(self._cfg.sensor.global_angvel)
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

        state.info["qacc"] = self._estimate_dof_acc(dof_vel)
        state.info["torques"] = self._estimate_pd_torques(state.info, dof_pos, dof_vel)
        orientation = self._orientation_score()
        step_count = state.info.get("steps", np.zeros((self._num_envs,), dtype=np.uint32))
        grace_elapsed = step_count >= self._cfg.termination_grace_steps
        terminated_z = upvector[:, 2] < -0.25
        terminated_contact = np.any(result, axis=1)
        terminated_low_height = (
            self.torso_height < self._z_des * self._cfg.termination_height_fraction
        )
        terminated_bad_orientation = orientation < self._cfg.termination_orientation_threshold
        terminated_pose = grace_elapsed & (terminated_low_height | terminated_bad_orientation)
        energy = np.sum(np.abs(state.info["torques"]) * np.abs(dof_vel), axis=1)
        terminated_energy = energy > self._cfg.energy_termination_threshold
        terminated = np.logical_or.reduce(
            (terminated_contact, terminated_z, terminated_energy, terminated_pose)
        )
        self._last_terminated = terminated.copy()
        reward = self._compute_reward(state.info, linvel, gyro, dof_pos, dof_vel)
        obs = self._compute_obs(
            state.info,
            linvel,
            gyro,
            gravity,
            dof_pos,
            dof_vel,
            self.torso_height.reshape(-1, 1),
            accelerometer,
            global_angvel,
        )
        return state.replace(obs=obs, reward=reward, terminated=terminated)

    def _compute_reward(
        self,
        info: dict,
        linvel: np.ndarray,
        gyro: np.ndarray,
        dof_pos: np.ndarray,
        dof_vel: np.ndarray | None = None,
    ) -> np.ndarray:
        dtype = get_global_dtype()
        reward = np.zeros((self._num_envs,), dtype=dtype)
        cfg = self._reward_cfg
        if dof_vel is None:
            dof_vel = self.get_dof_vel()

        ctx = RewardContext(
            info=info,
            linvel=linvel,
            gyro=gyro,
            dof_pos=dof_pos,
            dof_vel=dof_vel,
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
        return np.clip(reward * self._cfg.ctrl_dt, 0.0, 10000.0)

    def _compute_obs(
        self,
        info: dict,
        linvel: np.ndarray,
        gyro: np.ndarray,
        gravity: np.ndarray,
        dof_pos: np.ndarray,
        dof_vel: np.ndarray,
        height: np.ndarray,
        accelerometer: np.ndarray | None = None,
        global_angvel: np.ndarray | None = None,
        env_ids: np.ndarray | None = None,
    ) -> dict[str, np.ndarray]:
        noise_cfg = self._cfg.noise_config
        diff = dof_pos - self.default_angles
        noisy_linvel = self._obs_noise(linvel, noise_cfg.scale_linvel)
        noisy_gyro = self._obs_noise(gyro, noise_cfg.scale_gyro)
        noisy_gravity = self._obs_noise(gravity, noise_cfg.scale_gravity)
        noisy_diff = self._obs_noise(diff, noise_cfg.scale_joint_angle)
        noisy_dof_vel = self._obs_noise(dof_vel, noise_cfg.scale_joint_vel)
        last_actions = info.get("last_actions", np.zeros_like(diff))

        obs_frame = np.concatenate(
            [noisy_linvel, noisy_gyro, noisy_gravity, noisy_diff, noisy_dof_vel, last_actions],
            axis=1,
            dtype=get_global_dtype(),
        )
        critic_frame = np.concatenate(
            [linvel, gyro, gravity, diff, dof_vel, last_actions],
            axis=1,
            dtype=get_global_dtype(),
        )
        obs = self._update_obs_history(obs_frame, env_ids=env_ids, history_attr="_obs_history")
        critic_obs = self._update_obs_history(
            critic_frame,
            env_ids=env_ids,
            history_attr="_critic_obs_history",
        )
        torques = np.asarray(info.get("torques", np.zeros_like(dof_pos)), dtype=get_global_dtype())
        if accelerometer is None:
            accelerometer = np.zeros_like(gyro)
        if global_angvel is None:
            global_angvel = np.zeros_like(gyro)
        critic = np.concatenate(
            [
                critic_obs,
                gyro,
                accelerometer,
                linvel,
                global_angvel,
                dof_pos,
                dof_vel,
                torques,
                height,
            ],
            axis=1,
            dtype=get_global_dtype(),
        )
        return {"obs": obs, "critic": critic}

    def _update_obs_history(
        self,
        frame_obs: np.ndarray,
        *,
        env_ids: np.ndarray | None = None,
        history_attr: str = "_obs_history",
    ) -> np.ndarray:
        frame_obs = np.asarray(frame_obs, dtype=get_global_dtype())
        batch_size = int(frame_obs.shape[0])
        history_len = self._obs_history_len
        expected_shape = (self._num_envs, history_len, _FOOTSTAND_FRAME_OBS_DIM)
        history = getattr(self, history_attr, None)
        if history is None or history.shape != expected_shape:
            if env_ids is None and batch_size == self._num_envs:
                history = np.zeros(expected_shape, dtype=get_global_dtype())
                setattr(self, history_attr, history)
            elif env_ids is not None:
                history = np.zeros(expected_shape, dtype=get_global_dtype())
                setattr(self, history_attr, history)
            else:
                repeated = np.broadcast_to(
                    frame_obs[:, None, :], (batch_size, history_len, frame_obs.shape[1])
                ).copy()
                return np.asarray(repeated.reshape(batch_size, -1), dtype=get_global_dtype())

        assert history is not None
        if env_ids is None:
            if batch_size != self._num_envs:
                repeated = np.broadcast_to(
                    frame_obs[:, None, :], (batch_size, history_len, frame_obs.shape[1])
                ).copy()
                return np.asarray(repeated.reshape(batch_size, -1), dtype=get_global_dtype())
            history[:, :-1] = history[:, 1:]
            history[:, -1] = frame_obs
            selected_history = history
        else:
            env_ids = np.asarray(env_ids, dtype=np.int32)
            selected_history = np.broadcast_to(
                frame_obs[:, None, :], (batch_size, history_len, frame_obs.shape[1])
            ).copy()
            history[env_ids] = selected_history

        return np.asarray(selected_history.reshape(batch_size, -1), dtype=get_global_dtype())

    def _init_soft_joint_limits(self) -> None:
        joint_range = self._backend.get_joint_range()
        if joint_range is None:
            self._soft_lowers = np.full((self._num_action,), -np.inf, dtype=np.float32)
            self._soft_uppers = np.full((self._num_action,), np.inf, dtype=np.float32)
            return

        joint_range = np.asarray(joint_range, dtype=np.float32)
        centers = (joint_range[:, 0] + joint_range[:, 1]) / 2.0
        widths = joint_range[:, 1] - joint_range[:, 0]
        factor = self._cfg.soft_joint_pos_limit_factor
        self._soft_lowers = centers - 0.5 * widths * factor
        self._soft_uppers = centers + 0.5 * widths * factor

    def _init_motor_target_limits(self) -> None:
        joint_range = self._backend.get_joint_range()
        if joint_range is None:
            self._target_lowers = np.full((self._num_action,), -np.inf, dtype=get_global_dtype())
            self._target_uppers = np.full((self._num_action,), np.inf, dtype=get_global_dtype())
            return

        joint_range = np.asarray(joint_range, dtype=get_global_dtype())
        lowers = joint_range[:, 0]
        uppers = joint_range[:, 1]
        if lowers.size == _GO2_DOF_TO_CTRL.size:
            lowers = self._dof_to_ctrl_order(lowers.reshape(1, -1))[0]
            uppers = self._dof_to_ctrl_order(uppers.reshape(1, -1))[0]
        self._target_lowers = np.asarray(lowers, dtype=get_global_dtype())
        self._target_uppers = np.asarray(uppers, dtype=get_global_dtype())

    def _clip_motor_targets(self) -> None:
        lowers = getattr(self, "_target_lowers", None)
        uppers = getattr(self, "_target_uppers", None)
        if lowers is None or uppers is None:
            return
        np.clip(self._motor_targets, lowers, uppers, out=self._motor_targets)

    def _reward_termination(self, ctx: RewardContext) -> np.ndarray:
        return self._last_terminated.astype(get_global_dtype())

    def _cost_joint_pos_limits(self, ctx: RewardContext) -> np.ndarray:
        out_of_limits = -np.clip(ctx.dof_pos - self._soft_lowers, None, 0.0)
        out_of_limits += np.clip(ctx.dof_pos - self._soft_uppers, 0.0, None)
        return cast(np.ndarray, np.sum(out_of_limits, axis=1))

    def _cost_stay_still(self, ctx: RewardContext) -> np.ndarray:
        linvel = self._backend.get_base_lin_vel()
        angvel = self._backend.get_base_ang_vel()
        return cast(np.ndarray, np.sum(np.square(linvel[:, :2]), axis=1) + np.square(angvel[:, 2]))

    def _cost_torques(self, ctx: RewardContext) -> np.ndarray:
        torques = np.asarray(ctx.info.get("torques", np.zeros_like(ctx.dof_pos)))
        return cast(np.ndarray, np.sum(np.square(torques), axis=1))

    def _estimate_dof_acc(self, dof_vel: np.ndarray) -> np.ndarray:
        qacc = np.asarray((dof_vel - self._last_dof_vel_for_acc) / self._cfg.ctrl_dt)
        self._last_dof_vel_for_acc = np.asarray(dof_vel, dtype=np.float32).copy()
        return np.asarray(qacc, dtype=get_global_dtype())

    def _estimate_pd_torques(
        self, info: dict, dof_pos: np.ndarray, dof_vel: np.ndarray
    ) -> np.ndarray:
        del info
        targets = self._ctrl_to_dof_order(self._motor_targets)
        torques = self._cfg.control_config.Kp * (targets - dof_pos)
        torques -= self._cfg.control_config.Kd * dof_vel
        return np.asarray(torques, dtype=get_global_dtype())

    def _reward_orientation(self, ctx: RewardContext) -> np.ndarray:
        del ctx
        return self._orientation_score()

    def _orientation_score(self) -> np.ndarray:
        forward = self._get_body_forward()
        cos_dist = forward @ self._desired_forward_vec
        normalized = 0.5 * cos_dist + 0.5
        return np.asarray(np.square(normalized), dtype=get_global_dtype())

    def _cost_contact(self, ctx: RewardContext) -> np.ndarray:
        del ctx
        feet_contact = self.feet_force[:, self.feet_geom_names, 0] > _FOOTSTAND_CONTACT_THRESHOLD
        return np.asarray(np.any(feet_contact, axis=1), dtype=get_global_dtype())

    def _reward_rear_feet_contact(self, ctx: RewardContext) -> np.ndarray:
        del ctx
        rear_contact = self.feet_force[:, _FOOTSTAND_REAR_FEET, 0] > _FOOTSTAND_CONTACT_THRESHOLD
        return np.asarray(np.mean(rear_contact, axis=1), dtype=get_global_dtype())

    def _cost_rear_leg_symmetry(self, ctx: RewardContext) -> np.ndarray:
        rear_left = ctx.dof_pos[:, _FOOTSTAND_REAR_LEFT_LEG_IDS]
        rear_right = ctx.dof_pos[:, _FOOTSTAND_REAR_RIGHT_LEG_IDS]
        mirrored_right = rear_right * _FOOTSTAND_REAR_MIRROR_SIGNS
        cost = np.mean(np.square(rear_left - mirrored_right), axis=1)
        rising_mask = 1.0 - self._standing_mask()
        return np.asarray(cost * rising_mask, dtype=get_global_dtype())

    def _cost_front_leg_motion(self, ctx: RewardContext) -> np.ndarray:
        assert ctx.dof_vel is not None
        front_leg_vel = ctx.dof_vel[:, _FOOTSTAND_FRONT_LEG_IDS]
        cost = np.mean(np.square(front_leg_vel), axis=1)
        return np.asarray(cost * self._standing_mask(), dtype=get_global_dtype())

    def _cost_upright_stability(self, ctx: RewardContext) -> np.ndarray:
        del ctx
        linvel = self._backend.get_base_lin_vel()
        angvel = self._backend.get_base_ang_vel()
        cost = np.sum(np.square(linvel), axis=1) + 0.25 * np.sum(np.square(angvel), axis=1)
        return np.asarray(cost * self._standing_mask(), dtype=get_global_dtype())

    def _cost_knee_clearance(self, ctx: RewardContext) -> np.ndarray:
        del ctx
        target = max(float(self._reward_cfg.knee_height_target), 1e-6)
        knee_height = self._backend.get_body_pos_w(self._knee_body_ids)[:, :, 2]
        clearance_error = np.clip(target - knee_height, 0.0, None) / target
        return np.asarray(np.mean(np.square(clearance_error), axis=1), dtype=get_global_dtype())
