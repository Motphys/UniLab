from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, cast

import numpy as np

from unilab.assets import ASSETS_ROOT_PATH
from unilab.base import registry
from unilab.base.backend import create_backend
from unilab.base.np_env import NpEnvState
from unilab.dr import DomainRandomizationCapabilities, ResetPlan, ResetRandomizationPayload
from unilab.dr.dr_utils import (
    build_interval_push_plan,
    validate_interval_push_support,
    zero_actions,
)
from unilab.dtype_config import get_global_dtype
from unilab.envs.common.rotation import np_quat_mul, np_yaw_to_quat
from unilab.envs.locomotion.common import rewards
from unilab.envs.locomotion.common.commands import Commands
from unilab.envs.locomotion.common.domain_rand import DomainRandConfig
from unilab.envs.locomotion.common.dr_provider import LocomotionDRProvider
from unilab.envs.locomotion.common.rewards import RewardContext
from unilab.envs.locomotion.go2w.base import (
    DEFAULT_GO2W_ANGLES,
    NUM_GO2W_ACTIONS,
    NUM_LEG_ACTIONS,
    Go2WBaseCfg,
    Go2WBaseEnv,
    compute_go2w_motor_ctrl,
    stack_joint_sensors,
)


@dataclass
class InitState:
    pos = [0.0, 0.0, 0.4]


@dataclass
class Go2WDomainRandConfig(DomainRandConfig):
    randomize_kp: bool = True
    kp_multiplier_range: list[float] = field(default_factory=lambda: [0.9, 1.1])

    randomize_kd: bool = True
    kd_multiplier_range: list[float] = field(default_factory=lambda: [0.9, 1.1])


@dataclass
class RewardConfig:
    scales: dict[str, float]
    tracking_sigma: float
    base_height_target: float


@dataclass
class JoystickSensor:
    local_linvel = "local_linvel"
    gyro = "gyro"
    gravity = "upvector"


@registry.envcfg("Go2WJoystickFlat")
@dataclass
class Go2WJoystickCfg(Go2WBaseCfg):
    model_file: str = str(ASSETS_ROOT_PATH / "robots" / "go2w" / "scene_flat.xml")
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
        qpos[:, 0:3] += env._env_origins[env_ids]
        yaw = np.random.uniform(-np.pi, np.pi, (num_reset,))
        qpos[:, 3:7] = np_quat_mul(qpos[:, 3:7], np_yaw_to_quat(yaw))
        qvel[:, 0:6] = np.asarray(
            np.random.uniform(-0.5, 0.5, size=(num_reset, 6)), dtype=get_global_dtype()
        )

        motor_kp, motor_kd = env.sample_reset_motor_gains(num_reset)
        env.set_motor_gains(env_ids, motor_kp, motor_kd)

        info_updates: dict[str, Any] = {
            "commands": self._sample_commands(env, num_reset),
            "current_actions": zero_actions(num_reset, env._num_action),
            "last_actions": zero_actions(num_reset, env._num_action),
            "motor_kp": motor_kp.astype(get_global_dtype()),
            "motor_kd": motor_kd.astype(get_global_dtype()),
            "torques": np.zeros((num_reset, env._num_action), dtype=get_global_dtype()),
        }
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


@registry.env("Go2WJoystickFlat", sim_backend="mujoco")
class Go2WJoystickEnv(Go2WBaseEnv):
    _cfg: Go2WJoystickCfg

    def __init__(self, cfg: Go2WJoystickCfg, num_envs=1, backend_type="mujoco"):
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
        self._np_dtype = get_global_dtype()
        self._reward_cfg = cfg.reward_config
        self._enable_reward_log = True
        ctrl_range = np.asarray(self._backend.get_actuator_ctrl_range(), dtype=np.float64)
        self._validate_motor_control_contract(ctrl_range, num_envs)
        self._ctrl_lower = ctrl_range[:, 0].astype(self._np_dtype)
        self._ctrl_upper = ctrl_range[:, 1].astype(self._np_dtype)
        self._base_motor_kp = np.full((NUM_LEG_ACTIONS,), cfg.control_config.Kp, dtype=np.float64)
        self._base_motor_kd = np.full((NUM_LEG_ACTIONS,), cfg.control_config.Kd, dtype=np.float64)
        self._motor_kp = np.broadcast_to(self._base_motor_kp, (num_envs, NUM_LEG_ACTIONS)).copy()
        self._motor_kd = np.broadcast_to(self._base_motor_kd, (num_envs, NUM_LEG_ACTIONS)).copy()
        self._last_motor_ctrl = np.zeros((num_envs, NUM_GO2W_ACTIONS), dtype=self._np_dtype)
        self._backend.set_pre_step_control(self._pre_step_motor_control)
        self._init_reward_functions()
        self._init_domain_randomization(Go2WJoystickDomainRandomizationProvider())

    @property
    def obs_groups_spec(self) -> dict[str, int]:
        return {"obs": 53, "critic": 72}

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

    def _init_reward_functions(self) -> None:
        self._reward_fns: dict[str, Any] = {
            "tracking_lin_vel": rewards.tracking_lin_vel,
            "tracking_ang_vel": rewards.tracking_ang_vel,
            "lin_vel_z": rewards.lin_vel_z,
            "ang_vel_xy": rewards.ang_vel_xy,
            "base_height": rewards.base_height,
            "action_rate": rewards.action_rate,
            "similar_to_default": rewards.similar_to_default,
            "torques": rewards.torques,
            "energy": rewards.energy,
            "alive": rewards.alive,
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
            exec_actions[:, :NUM_LEG_ACTIONS] * self._cfg.control_config.action_scale
            + self.default_angles[:NUM_LEG_ACTIONS]
        )
        wheel_torque = (
            exec_actions[:, NUM_LEG_ACTIONS:] * self._cfg.control_config.wheel_action_scale
        )
        return np.concatenate([leg_targets, wheel_torque], axis=1, dtype=self._np_dtype)

    def _pre_step_motor_control(self, backend: Any, policy_ctrl: np.ndarray) -> np.ndarray:
        joint_pos = stack_joint_sensors(backend, "pos", dtype=self.default_angles.dtype)
        joint_vel = stack_joint_sensors(backend, "vel", dtype=self.default_angles.dtype)
        motor_ctrl = compute_go2w_motor_ctrl(
            policy_ctrl,
            joint_pos,
            joint_vel,
            self._motor_kp,
            self._motor_kd,
            self._ctrl_lower,
            self._ctrl_upper,
            self._last_motor_ctrl,
        )
        return motor_ctrl

    def update_state(self, state: NpEnvState) -> NpEnvState:
        linvel = self.get_local_linvel()
        gyro = self.get_gyro()
        gravity = self._backend.get_sensor_data(self._cfg.sensor.gravity)
        dof_pos = self.get_dof_pos()
        dof_vel = self.get_dof_vel()
        state.info["torques"] = self._last_motor_ctrl.copy()
        terminated = gravity[:, 2] <= 0.5
        reward = self._compute_reward(state.info, linvel, gyro, dof_pos, dof_vel)
        obs = self._compute_obs(state.info, linvel, gyro, gravity, dof_pos, dof_vel)
        return state.replace(obs=obs, reward=reward, terminated=terminated)

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
        last_actions = info.get("current_actions", np.zeros((self._num_envs, self._num_action)))

        obs = np.concatenate(
            [gyro, -gravity, leg_diff, leg_vel, wheel_vel, last_actions, info["commands"]],
            axis=1,
            dtype=get_global_dtype(),
        )
        critic = np.concatenate(
            [obs, linvel, self._last_motor_ctrl],
            axis=1,
            dtype=get_global_dtype(),
        )
        return {"obs": obs, "critic": critic}

    def _compute_reward(self, info: dict, linvel, gyro, dof_pos, dof_vel) -> np.ndarray:
        dtype = get_global_dtype()
        reward = np.zeros((self._num_envs,), dtype=dtype)
        ctx = RewardContext(
            info=info,
            linvel=linvel,
            gyro=gyro,
            dof_pos=dof_pos[:, :NUM_LEG_ACTIONS],
            dof_vel=dof_vel,
            num_envs=self._num_envs,
            default_angles=DEFAULT_GO2W_ANGLES[:NUM_LEG_ACTIONS].astype(dtype),
            tracking_sigma=self._reward_cfg.tracking_sigma,
            base_height_target=self._reward_cfg.base_height_target,
            base_height=self._backend.get_base_pos()[:, 2],
        )

        step_count = info.get("steps", np.zeros((self._num_envs,), dtype=np.uint32))
        should_log = self._enable_reward_log and (int(step_count[0]) % 4 == 0)
        log = {} if should_log else info.get("log", {})

        for name, scale in self._reward_cfg.scales.items():
            if scale == 0 or name not in self._reward_fns:
                continue
            rew = self._reward_fns[name](ctx)
            weighted_rew = rew * scale
            reward += weighted_rew
            if should_log:
                log[f"reward/{name}"] = float(np.mean(weighted_rew))

        info["log"] = log
        return reward * self._cfg.ctrl_dt

    def _reward_wheel_vel(self, ctx: RewardContext) -> np.ndarray:
        del ctx
        wheel_vel = self.get_dof_vel()[:, NUM_LEG_ACTIONS:]
        return np.asarray(np.sum(np.square(wheel_vel), axis=1), dtype=get_global_dtype())
