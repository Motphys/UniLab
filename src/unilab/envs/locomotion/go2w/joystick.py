from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, cast

import numpy as np

from unilab.assets import ASSETS_ROOT_PATH
from unilab.base import registry
from unilab.base.backend import create_backend
from unilab.base.np_env import NpEnvState
from unilab.base.scene import SceneCfg
from unilab.dr import DomainRandomizationCapabilities, ResetPlan, ResetRandomizationPayload
from unilab.dr.dr_utils import (
    build_interval_push_plan,
    validate_interval_push_support,
    zero_actions,
)
from unilab.dtype_config import get_global_dtype
from unilab.envs.common.rotation import (
    np_quat_mul,
    np_yaw_to_quat,
)
from unilab.envs.locomotion.common import rewards
from unilab.envs.locomotion.common.commands import (
    Commands,
    apply_heading_yaw_feedback,
    zero_small_xy_commands,
)
from unilab.envs.locomotion.common.commands import (
    sample_heading_commands as sample_go2w_heading_commands,
)
from unilab.envs.locomotion.common.domain_rand import DomainRandConfig
from unilab.envs.locomotion.common.dr_provider import LocomotionDRProvider
from unilab.envs.locomotion.common.rewards import RewardContext
from unilab.envs.locomotion.go2w.base import (
    DEFAULT_GO2W_ANGLES,
    NUM_GO2W_ACTIONS,
    NUM_LEG_ACTIONS,
    NUM_WHEEL_ACTIONS,
    Go2WBaseCfg,
    Go2WBaseEnv,
    compute_go2w_motor_ctrl,
    stack_joint_sensors,
)

GO2W_HIP_INDICES = np.asarray([0, 3, 6, 9], dtype=np.int32)


@dataclass
class InitState:
    pos = [0.0, 0.0, 0.42]


@dataclass
class Go2WDomainRandConfig(DomainRandConfig):
    randomize_init_yaw: bool = True
    init_z_range: list[float] = field(default_factory=lambda: [0.0, 0.2])
    init_roll_range: list[float] = field(default_factory=lambda: [0.0, 0.0])
    init_pitch_range: list[float] = field(default_factory=lambda: [0.0, 0.0])
    init_yaw_range: list[float] = field(default_factory=lambda: [-np.pi, np.pi])

    randomize_kp: bool = True
    kp_multiplier_range: list[float] = field(default_factory=lambda: [0.9, 1.1])

    randomize_kd: bool = True
    kd_multiplier_range: list[float] = field(default_factory=lambda: [0.9, 1.1])


@dataclass
class RewardConfig:
    scales: dict[str, float]
    tracking_sigma: float
    base_height_target: float
    only_positive_rewards: bool = False
    joint_pos_penalty_stand_still_scale: float = 5.0
    joint_pos_penalty_velocity_threshold: float = 0.5
    joint_pos_penalty_command_threshold: float = 0.1


@dataclass
class JoystickSensor:
    local_linvel = "local_linvel"
    gyro = "gyro"
    gravity = "upvector"


@registry.envcfg("Go2WJoystickFlat")
@dataclass
class Go2WJoystickCfg(Go2WBaseCfg):
    scene: SceneCfg = field(
        default_factory=lambda: SceneCfg(
            model_file=str(ASSETS_ROOT_PATH / "robots" / "go2w" / "scene_flat.xml")
        )
    )
    max_episode_seconds: float = 20.0
    init_state: InitState = field(default_factory=InitState)
    commands: Commands = field(default_factory=Commands)
    reward_config: RewardConfig | None = None
    sensor: JoystickSensor = field(default_factory=JoystickSensor)  # type: ignore[assignment]
    domain_rand: Go2WDomainRandConfig = field(default_factory=Go2WDomainRandConfig)


def build_go2w_backend_reset_randomization(
    env: Any, num_reset: int
) -> ResetRandomizationPayload | None:
    """Build reset DR payloads that are valid for a motor-actuator Go2W model.

    kp/kd are intentionally excluded here. Go2W samples them through the same
    config path as Go2, but applies them inside its owner pre-step motor control.
    """
    domain_rand = getattr(env.cfg, "domain_rand", None)
    if domain_rand is None:
        return None

    payload = ResetRandomizationPayload()
    if getattr(domain_rand, "randomize_base_mass", False):
        low, high = domain_rand.added_mass_range
        payload.base_mass_delta = np.random.uniform(low, high, size=(num_reset,))

    if getattr(domain_rand, "random_com", False):
        low, high = domain_rand.com_offset_x
        base_com_offset = np.zeros((num_reset, 3), dtype=np.float64)
        base_com_offset[:, 0] = np.random.uniform(low, high, size=(num_reset,))
        payload.base_com_offset = base_com_offset

    if getattr(domain_rand, "randomize_gravity", False):
        gravity_range = np.asarray(domain_rand.gravity_range, dtype=np.float64)
        if gravity_range.shape != (2, 3):
            raise ValueError(
                f"domain_rand.gravity_range must have shape (2, 3), got {gravity_range.shape}"
            )
        low = np.minimum(gravity_range[0], gravity_range[1])
        high = np.maximum(gravity_range[0], gravity_range[1])
        payload.gravity = np.random.uniform(low=low, high=high, size=(num_reset, 3))

    return None if payload.is_empty() else payload


def sample_go2w_reset_yaw(domain_rand: Go2WDomainRandConfig, num_reset: int) -> np.ndarray:
    if not domain_rand.randomize_init_yaw:
        return np.zeros((num_reset,), dtype=get_global_dtype())

    yaw_range = np.asarray(domain_rand.init_yaw_range, dtype=np.float64)
    if yaw_range.shape != (2,):
        raise ValueError(f"domain_rand.init_yaw_range must have shape (2,), got {yaw_range.shape}")
    low, high = float(np.min(yaw_range)), float(np.max(yaw_range))
    return np.asarray(np.random.uniform(low, high, size=(num_reset,)), dtype=get_global_dtype())


class Go2WJoystickDomainRandomizationProvider(LocomotionDRProvider):
    def validate(self, env: Any, capabilities: DomainRandomizationCapabilities) -> None:
        payload = build_go2w_backend_reset_randomization(env, num_reset=1)
        if payload is not None:
            unsupported = capabilities.get_unsupported_reset_terms(payload.requested_terms())
            if unsupported:
                names = ", ".join(sorted(unsupported))
                raise NotImplementedError(
                    f"{env._backend.backend_type} backend does not support Go2W reset randomization terms: {names}"
                )
        validate_interval_push_support(env, capabilities)

    def build_interval_randomization_plan(self, env: Any, step_counter: int):
        return build_interval_push_plan(env, step_counter)

    def build_reset_plan(self, env: Any, env_ids: np.ndarray) -> ResetPlan:
        num_reset = len(env_ids)
        qpos = np.tile(env._init_qpos, (num_reset, 1))
        qvel = np.tile(env._init_qvel, (num_reset, 1))
        qpos[:, 0:2] += np.random.uniform(-0.5, 0.5, (num_reset, 2))
        qpos[:, 0:3] += env._spawn.origins_for(env_ids)
        yaw = sample_go2w_reset_yaw(env.cfg.domain_rand, num_reset)
        qpos[:, 3:7] = np_quat_mul(qpos[:, 3:7], np_yaw_to_quat(yaw))
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
        del env_ids
        return cast(
            dict[str, np.ndarray],
            env._compute_obs(info_updates, linvel, gyro, gravity, dof_pos, dof_vel),
        )

    def _sample_commands(self, env: Any, num_reset: int) -> np.ndarray:
        commands = super()._sample_commands(env, num_reset)
        zero_small_xy_commands(commands)
        standing_prob = float(getattr(env.cfg.commands, "rel_standing_envs", 0.0))
        if standing_prob > 0.0:
            standing = np.random.uniform(size=(num_reset,)) < min(standing_prob, 1.0)
            commands[standing] = 0.0
        if getattr(env.cfg.commands, "heading_command", False):
            commands[:, 2] = 0.0
        return commands


@registry.env("Go2WJoystickFlat", sim_backend="mujoco")
class Go2WJoystickEnv(Go2WBaseEnv):
    _cfg: Go2WJoystickCfg

    def __init__(self, cfg: Go2WJoystickCfg, num_envs=1, backend_type="mujoco"):
        if cfg.reward_config is None:
            raise ValueError("reward_config must be provided via Hydra configuration")
        backend = create_backend(
            backend_type,
            cfg.scene,
            num_envs,
            cfg.sim_dt,
            base_name=cfg.asset.base_name,
            push_body_name=cfg.domain_rand.push_body_name,
            motrix_max_iterations=cfg.motrix_max_iterations,
        )
        super().__init__(cfg, backend, num_envs)
        self._np_dtype = get_global_dtype()
        self._leg_action_scale = self._build_leg_action_scale()
        self._reward_cfg = cfg.reward_config
        self._enable_reward_log = True
        ctrl_range = np.asarray(self._backend.get_actuator_ctrl_range(), dtype=np.float64)
        self._validate_motor_control_contract(ctrl_range, num_envs)
        self._ctrl_lower = ctrl_range[:, 0].astype(self._np_dtype)
        self._ctrl_upper = ctrl_range[:, 1].astype(self._np_dtype)
        joint_range = self._backend.get_joint_range()
        self._leg_joint_range = (
            np.asarray(joint_range[:NUM_LEG_ACTIONS], dtype=get_global_dtype())
            if joint_range is not None
            else None
        )
        self._base_motor_kp = np.full((NUM_LEG_ACTIONS,), cfg.control_config.Kp, dtype=np.float64)
        self._base_motor_kd = np.full((NUM_LEG_ACTIONS,), cfg.control_config.Kd, dtype=np.float64)
        self._base_wheel_kd = np.full(
            (NUM_WHEEL_ACTIONS,), cfg.control_config.wheel_Kd, dtype=np.float64
        )
        self._motor_kp = np.broadcast_to(self._base_motor_kp, (num_envs, NUM_LEG_ACTIONS)).copy()
        self._motor_kd = np.broadcast_to(self._base_motor_kd, (num_envs, NUM_LEG_ACTIONS)).copy()
        self._wheel_kd = np.broadcast_to(self._base_wheel_kd, (num_envs, NUM_WHEEL_ACTIONS)).copy()
        self._last_motor_ctrl = np.zeros((num_envs, NUM_GO2W_ACTIONS), dtype=self._np_dtype)
        self._last_dof_vel_for_acc = np.zeros(
            (num_envs, NUM_GO2W_ACTIONS), dtype=get_global_dtype()
        )
        self._backend.set_pre_step_control(self._pre_step_motor_control)
        self._init_reward_functions()
        self._init_domain_randomization(Go2WJoystickDomainRandomizationProvider())

    @property
    def obs_groups_spec(self) -> dict[str, int]:
        return {"obs": 53, "critic": 72}

    def reset(self, env_indices: np.ndarray) -> tuple[dict[str, np.ndarray], dict]:
        env_ids = np.asarray(env_indices, dtype=np.int32)
        obs, info = super().reset(env_ids)
        dof_vel = self.get_dof_vel()
        if dof_vel.shape[0] == self._num_envs:
            self._last_dof_vel_for_acc[env_ids] = dof_vel[env_ids]
        return obs, info

    def _validate_motor_control_contract(self, ctrl_range: np.ndarray, num_envs: int) -> None:
        if self._backend.num_actuators != NUM_GO2W_ACTIONS:
            raise ValueError(
                f"Go2W requires {NUM_GO2W_ACTIONS} motor actuators, got {self._backend.num_actuators}"
            )
        if ctrl_range.shape != (NUM_GO2W_ACTIONS, 2):
            raise ValueError(
                f"Go2W actuator ctrl_range must have shape ({NUM_GO2W_ACTIONS}, 2), "
                f"got {ctrl_range.shape}"
            )
        pos = stack_joint_sensors(self._backend, "pos", dtype=self.default_angles.dtype)
        vel = stack_joint_sensors(self._backend, "vel", dtype=self.default_angles.dtype)
        expected_shape = (num_envs, NUM_GO2W_ACTIONS)
        if pos.shape != expected_shape:
            raise ValueError(f"Go2W joint position sensor stack must have shape {expected_shape}")
        if vel.shape != expected_shape:
            raise ValueError(f"Go2W joint velocity sensor stack must have shape {expected_shape}")

    def _build_leg_action_scale(self) -> np.ndarray:
        scale = np.full(
            (NUM_LEG_ACTIONS,),
            float(self._cfg.control_config.action_scale),
            dtype=self._np_dtype,
        )
        hip_action_scale = self._cfg.control_config.hip_action_scale
        if hip_action_scale is not None:
            scale[GO2W_HIP_INDICES] = float(hip_action_scale)
        return scale

    def _init_reward_functions(self) -> None:
        self._reward_fns: dict[str, Any] = {
            "tracking_lin_vel": rewards.tracking_lin_vel,
            "tracking_ang_vel": rewards.tracking_ang_vel,
            "lin_vel_z": rewards.lin_vel_z,
            "ang_vel_xy": rewards.ang_vel_xy,
            "base_height": rewards.base_height,
            "action_rate": rewards.action_rate,
            "similar_to_default": rewards.similar_to_default,
            "orientation": rewards.orientation,
            "torques": self._reward_torques_l2,
            "joint_torques_l2": self._reward_joint_torques_l2,
            "energy": rewards.energy,
            "dof_vel": self._reward_dof_vel,
            "dof_acc": self._reward_dof_acc,
            "joint_acc_l2": self._reward_dof_acc,
            "wheel_acc": self._reward_wheel_acc,
            "joint_acc_wheel_l2": self._reward_wheel_acc,
            "stand_still": self._reward_stand_still,
            "hip_pos": self._reward_hip_pos,
            "dof_error": self._reward_dof_error,
            "joint_pos_penalty": self._reward_joint_pos_penalty,
            "joint_power": self._reward_joint_power,
            "joint_mirror": self._reward_joint_mirror,
            "alive": rewards.alive,
            "upward": rewards.upward,
            "wheel_vel": self._reward_wheel_vel,
        }

    def sample_reset_motor_gains(self, num_reset: int) -> tuple[np.ndarray, np.ndarray]:
        kp = np.broadcast_to(self._base_motor_kp, (num_reset, NUM_LEG_ACTIONS)).copy()
        kd = np.broadcast_to(self._base_motor_kd, (num_reset, NUM_LEG_ACTIONS)).copy()
        domain_rand = self._cfg.domain_rand
        if domain_rand.randomize_kp:
            low, high = domain_rand.kp_multiplier_range
            kp *= np.random.uniform(low, high, size=(num_reset, 1))
        if domain_rand.randomize_kd:
            low, high = domain_rand.kd_multiplier_range
            kd *= np.random.uniform(low, high, size=(num_reset, 1))
        return kp, kd

    def set_motor_gains(self, env_ids: np.ndarray, kp: np.ndarray, kd: np.ndarray) -> None:
        self._motor_kp[env_ids] = np.asarray(kp, dtype=np.float64)
        self._motor_kd[env_ids] = np.asarray(kd, dtype=np.float64)

    def apply_action(self, actions: np.ndarray, state: NpEnvState) -> np.ndarray:
        clipped_actions = np.asarray(
            np.clip(
                actions,
                -self._cfg.control_config.clip_actions,
                self._cfg.control_config.clip_actions,
            ),
            dtype=self._np_dtype,
        )
        state.info["last_actions"] = state.info.get(
            "current_actions", np.zeros_like(clipped_actions)
        )
        state.info["current_actions"] = clipped_actions
        exec_actions = (
            state.info["last_actions"]
            if self._cfg.control_config.simulate_action_latency
            else clipped_actions
        )

        leg_targets = (
            exec_actions[:, :NUM_LEG_ACTIONS]
            * self._leg_action_scale
            + self.default_angles[:NUM_LEG_ACTIONS]
        )
        wheel_velocity_targets = (
            exec_actions[:, NUM_LEG_ACTIONS:] * self._cfg.control_config.wheel_action_scale
        )
        return np.concatenate([leg_targets, wheel_velocity_targets], axis=1, dtype=self._np_dtype)

    def _pre_step_motor_control(self, backend: Any, policy_ctrl: np.ndarray) -> np.ndarray:
        joint_pos = stack_joint_sensors(backend, "pos", dtype=self.default_angles.dtype)
        joint_vel = stack_joint_sensors(backend, "vel", dtype=self.default_angles.dtype)
        motor_ctrl = compute_go2w_motor_ctrl(
            policy_ctrl,
            joint_pos,
            joint_vel,
            self._motor_kp,
            self._motor_kd,
            self._wheel_kd,
            self._ctrl_lower,
            self._ctrl_upper,
            self._last_motor_ctrl,
        )
        return motor_ctrl

    def update_state(self, state: NpEnvState) -> NpEnvState:
        self._update_commands(state.info)
        linvel = self.get_local_linvel()
        gyro = self.get_gyro()
        gravity = self._backend.get_sensor_data(self._cfg.sensor.gravity)
        dof_pos = self.get_dof_pos()
        dof_vel = self.get_dof_vel()
        state.info["torques"] = self._last_motor_ctrl.copy()
        state.info["qacc"] = self._estimate_dof_acc(dof_vel)
        terminated = self._compute_terminated(gravity)
        reward = self._compute_reward(state.info, linvel, gyro, gravity, dof_pos, dof_vel)
        obs = self._compute_obs(state.info, linvel, gyro, gravity, dof_pos, dof_vel)
        return state.replace(obs=obs, reward=reward, terminated=terminated)

    def _compute_terminated(self, gravity: np.ndarray) -> np.ndarray:
        return gravity[:, 2] <= 0.5

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
        leg_vel = dof_vel[:, :NUM_LEG_ACTIONS]
        wheel_vel = dof_vel[:, NUM_LEG_ACTIONS:]
        gyro = self._obs_noise(gyro, noise_cfg.scale_gyro)
        gravity = self._obs_noise(gravity, noise_cfg.scale_gravity)
        leg_diff = self._obs_noise(leg_diff, noise_cfg.scale_joint_angle)
        leg_vel = self._obs_noise(leg_vel, noise_cfg.scale_joint_vel)
        wheel_vel = self._obs_noise(wheel_vel, noise_cfg.scale_wheel_vel)
        linvel = self._obs_noise(linvel, noise_cfg.scale_linvel)
        num_obs = gyro.shape[0]
        last_actions = info.get("current_actions", np.zeros((num_obs, self._num_action)))
        motor_ctrl = info.get("torques", np.zeros((num_obs, self._num_action), dtype=dof_pos.dtype))

        obs = np.concatenate(
            [gyro, -gravity, leg_diff, leg_vel, wheel_vel, last_actions, info["commands"]],
            axis=1,
            dtype=get_global_dtype(),
        )
        critic = np.concatenate(
            [obs, linvel, motor_ctrl],
            axis=1,
            dtype=get_global_dtype(),
        )
        return {"obs": obs, "critic": critic}

    def _compute_reward(self, info: dict, linvel, gyro, gravity, dof_pos, dof_vel) -> np.ndarray:
        dtype = get_global_dtype()
        num_obs = linvel.shape[0]
        ctx = RewardContext(
            info=info,
            linvel=linvel,
            gyro=gyro,
            dof_pos=dof_pos[:, :NUM_LEG_ACTIONS],
            dof_vel=dof_vel,
            num_envs=num_obs,
            default_angles=DEFAULT_GO2W_ANGLES[:NUM_LEG_ACTIONS].astype(dtype),
            tracking_sigma=self._reward_cfg.tracking_sigma,
            base_height_target=self._reward_cfg.base_height_target,
            base_height=self._reward_base_height_values(num_obs),
            gravity=gravity,
            joint_range=self._leg_joint_range,
        )
        return rewards.run_reward_dispatch(
            scales=self._reward_cfg.scales,
            fns=self._reward_fns,
            ctx=ctx,
            info=info,
            enable_log=self._enable_reward_log,
            ctrl_dt=self._cfg.ctrl_dt,
            only_positive=self._reward_cfg.only_positive_rewards,
        )

    def _update_commands(self, info: dict) -> None:
        commands = info.get("commands")
        if commands is None:
            return

        commands_arr = np.asarray(commands, dtype=get_global_dtype())
        resampling_time = float(getattr(self._cfg.commands, "resampling_time", 0.0))
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
                zero_small_xy_commands(sampled)
                standing_prob = float(getattr(self._cfg.commands, "rel_standing_envs", 0.0))
                if standing_prob > 0.0:
                    standing = np.random.uniform(size=(num_resample,)) < min(standing_prob, 1.0)
                    sampled[standing] = 0.0
                commands_arr[resample_mask] = sampled
                if getattr(self._cfg.commands, "heading_command", False):
                    heading_commands = self._ensure_heading_commands(info, commands_arr.shape[0])
                    heading_commands[resample_mask] = sample_go2w_heading_commands(
                        self, num_resample
                    )
                    info["heading_commands"] = heading_commands

        if getattr(self._cfg.commands, "heading_command", False):
            heading_commands = self._ensure_heading_commands(info, commands_arr.shape[0])
            base_quat = np.asarray(self._backend.get_base_quat(), dtype=get_global_dtype())
            if base_quat.shape[0] == commands_arr.shape[0]:
                stiffness = float(getattr(self._cfg.commands, "heading_control_stiffness", 0.5))
                apply_heading_yaw_feedback(
                    commands_arr, base_quat, heading_commands, stiffness=stiffness
                )
        info["commands"] = commands_arr

    def _ensure_heading_commands(self, info: dict, num_obs: int) -> np.ndarray:
        heading_commands = info.get("heading_commands")
        if heading_commands is None or np.asarray(heading_commands).shape != (num_obs,):
            heading_commands = sample_go2w_heading_commands(self, num_obs)
            info["heading_commands"] = heading_commands
        heading_commands = np.asarray(heading_commands, dtype=get_global_dtype())
        info["heading_commands"] = heading_commands
        return heading_commands

    def _estimate_dof_acc(self, dof_vel: np.ndarray) -> np.ndarray:
        qacc = np.asarray((dof_vel - self._last_dof_vel_for_acc) / self._cfg.ctrl_dt)
        self._last_dof_vel_for_acc[:] = dof_vel
        return np.asarray(qacc, dtype=get_global_dtype())

    def _reward_base_height_values(self, num_obs: int) -> np.ndarray:
        base_pos = np.asarray(self._backend.get_base_pos(), dtype=get_global_dtype())
        if base_pos.shape[0] != num_obs:
            return np.zeros((num_obs,), dtype=get_global_dtype())
        return np.asarray(base_pos[:, 2], dtype=get_global_dtype())

    def _reward_wheel_vel(self, ctx: RewardContext) -> np.ndarray:
        assert ctx.dof_vel is not None
        wheel_vel = ctx.dof_vel[:, NUM_LEG_ACTIONS:]
        return np.asarray(np.sum(np.square(wheel_vel), axis=1), dtype=get_global_dtype())

    def _reward_torques_l2(self, ctx: RewardContext) -> np.ndarray:
        torques = np.asarray(
            ctx.info.get("torques", np.zeros((ctx.num_envs, self._num_action))),
            dtype=get_global_dtype(),
        )
        return np.asarray(np.sum(np.square(torques), axis=1), dtype=get_global_dtype())

    def _reward_joint_torques_l2(self, ctx: RewardContext) -> np.ndarray:
        torques = np.asarray(
            ctx.info.get("torques", np.zeros((ctx.num_envs, self._num_action))),
            dtype=get_global_dtype(),
        )
        return np.asarray(
            np.sum(np.square(torques[:, :NUM_LEG_ACTIONS]), axis=1),
            dtype=get_global_dtype(),
        )

    def _reward_dof_vel(self, ctx: RewardContext) -> np.ndarray:
        assert ctx.dof_vel is not None
        return np.asarray(
            np.sum(np.square(ctx.dof_vel[:, :NUM_LEG_ACTIONS]), axis=1),
            dtype=get_global_dtype(),
        )

    def _reward_dof_acc(self, ctx: RewardContext) -> np.ndarray:
        qacc = np.asarray(
            ctx.info.get("qacc", np.zeros((ctx.num_envs, NUM_GO2W_ACTIONS))),
            dtype=get_global_dtype(),
        )
        return np.asarray(np.sum(np.square(qacc[:, :NUM_LEG_ACTIONS]), axis=1), dtype=qacc.dtype)

    def _reward_wheel_acc(self, ctx: RewardContext) -> np.ndarray:
        qacc = np.asarray(
            ctx.info.get("qacc", np.zeros((ctx.num_envs, NUM_GO2W_ACTIONS))),
            dtype=get_global_dtype(),
        )
        return np.asarray(np.sum(np.square(qacc[:, NUM_LEG_ACTIONS:]), axis=1), dtype=qacc.dtype)

    def _reward_stand_still(self, ctx: RewardContext) -> np.ndarray:
        commands = ctx.info["commands"]
        stopped = np.linalg.norm(commands[:, :2], axis=1) < 0.1
        dof_error = np.sum(np.abs(ctx.dof_pos - DEFAULT_GO2W_ANGLES[:NUM_LEG_ACTIONS]), axis=1)
        return np.asarray(dof_error * stopped, dtype=get_global_dtype())

    def _reward_hip_pos(self, ctx: RewardContext) -> np.ndarray:
        diff = ctx.dof_pos[:, GO2W_HIP_INDICES] - DEFAULT_GO2W_ANGLES[GO2W_HIP_INDICES]
        return np.asarray(np.sum(np.square(diff), axis=1), dtype=get_global_dtype())

    def _reward_dof_error(self, ctx: RewardContext) -> np.ndarray:
        diff = ctx.dof_pos - DEFAULT_GO2W_ANGLES[:NUM_LEG_ACTIONS]
        return np.asarray(np.sum(np.square(diff), axis=1), dtype=get_global_dtype())

    def _reward_joint_pos_penalty(self, ctx: RewardContext) -> np.ndarray:
        return rewards.joint_pos_penalty(
            ctx,
            stand_still_scale=self._reward_cfg.joint_pos_penalty_stand_still_scale,
            velocity_threshold=self._reward_cfg.joint_pos_penalty_velocity_threshold,
            command_threshold=self._reward_cfg.joint_pos_penalty_command_threshold,
        )

    def _reward_joint_power(self, ctx: RewardContext) -> np.ndarray:
        assert ctx.dof_vel is not None
        torques = np.asarray(
            ctx.info.get("torques", np.zeros((ctx.num_envs, self._num_action))),
            dtype=get_global_dtype(),
        )
        return np.asarray(
            np.sum(np.abs(ctx.dof_vel[:, :NUM_LEG_ACTIONS] * torques[:, :NUM_LEG_ACTIONS]), axis=1),
            dtype=get_global_dtype(),
        )

    def _reward_joint_mirror(self, ctx: RewardContext) -> np.ndarray:
        fr_rl = ctx.dof_pos[:, 0:3] - ctx.dof_pos[:, 9:12]
        fl_rr = ctx.dof_pos[:, 3:6] - ctx.dof_pos[:, 6:9]
        mirror = 0.5 * (np.sum(np.square(fr_rl), axis=1) + np.sum(np.square(fl_rr), axis=1))
        return np.asarray(mirror, dtype=get_global_dtype())


registry.register_env("Go2WJoystickFlat", Go2WJoystickEnv, sim_backend="motrix")
