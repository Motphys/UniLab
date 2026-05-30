from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from unilab.assets import ASSETS_ROOT_PATH
from unilab.base import registry
from unilab.base.backend import create_backend
from unilab.base.np_env import NpEnvState
from unilab.base.scene import SceneCfg
from unilab.dr.types import ResetPlan
from unilab.dtype_config import get_global_dtype
from unilab.envs.common.rotation import np_matrix_from_quat, np_quat_from_euler_xyz
from unilab.envs.locomotion.common import rewards
from unilab.envs.locomotion.common.commands import Commands
from unilab.envs.locomotion.common.domain_rand import DomainRandConfig
from unilab.envs.locomotion.common.dr_provider import LocomotionDRProvider
from unilab.envs.locomotion.common.rewards import RewardContext
from unilab.envs.locomotion.go2_arm.base import (
    DEFAULT_LEG_ANGLES,
    Go2ArmBaseCfg,
    Go2ArmBaseEnv,
    Go2ArmSensor,
    build_go2_arm_position_gains,
)


def _default_go2_arm_model_file() -> str:
    return str(ASSETS_ROOT_PATH / "robots" / "go2_arm" / "scene_flat.xml")


def _default_go2_arm_scene() -> SceneCfg:
    return SceneCfg(model_file=_default_go2_arm_model_file())


def _resolve_go2_arm_scene(cfg: "Go2ArmManipLocoCfg") -> SceneCfg:
    scene = cfg.scene
    default_model_file = _default_go2_arm_model_file()
    if scene is None:
        scene = SceneCfg(model_file=cfg.model_file)
    elif cfg.model_file != default_model_file and scene.model_file == default_model_file:
        scene = SceneCfg(
            model_file=cfg.model_file,
            fragment_files=list(scene.fragment_files),
            terrain=scene.terrain,
        )
    cfg.scene = scene
    return scene


def _sphere2cart(sphere: np.ndarray) -> np.ndarray:
    """Convert (..., 3)[l, phi, theta] to (..., 3)[x, y, z]."""
    l = sphere[..., 0]
    phi = sphere[..., 1]
    theta = sphere[..., 2]
    x = l * np.cos(phi) * np.cos(theta)
    y = l * np.sin(theta)
    z = l * np.sin(phi) * np.cos(theta)
    return np.stack([x, y, z], axis=-1)


def _cart2sphere(cart: np.ndarray) -> np.ndarray:
    """Convert (..., 3)[x, y, z] to (..., 3)[l, phi, theta]."""
    cart = np.asarray(cart)
    l_sq = np.sum(cart**2, axis=-1, keepdims=True)
    l = np.sqrt(np.maximum(l_sq, 1e-12))
    phi = np.arctan2(cart[..., 2:3], cart[..., 0:1])
    theta = np.arcsin(np.clip(cart[..., 1:2] / l, -1.0, 1.0))
    return np.concatenate([l, phi, theta], axis=-1)


@dataclass
class InitState:
    pos: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.42])


@dataclass
class Go2ArmDomainRandConfig(DomainRandConfig):
    randomize_kp: bool = True
    kp_multiplier_range: list[float] = field(default_factory=lambda: [0.9, 1.1])

    randomize_kd: bool = True
    kd_multiplier_range: list[float] = field(default_factory=lambda: [0.9, 1.1])


@dataclass
class EEGoalConfig:
    """End-effector goal config in spherical coordinates."""

    # Spherical sampling ranges.
    sphere_l_range: list[float] = field(default_factory=lambda: [0.3, 0.6])
    sphere_phi_range: list[float] = field(default_factory=lambda: [-1.2566, 1.0472])
    sphere_theta_range: list[float] = field(default_factory=lambda: [-2.3562, 2.3562])
    # Trajectory timing.
    traj_time_range: list[float] = field(default_factory=lambda: [1.0, 3.0])
    hold_time_range: list[float] = field(default_factory=lambda: [0.5, 2.0])
    # Collision checks.
    collision_upper_limits: list[float] = field(default_factory=lambda: [0.3, 0.15, 0.05 - 0.165])
    collision_lower_limits: list[float] = field(
        default_factory=lambda: [-0.2, -0.15, -0.35 - 0.165]
    )
    underground_limit: float = -0.57
    num_collision_check_samples: int = 10
    num_resample_attempts: int = 10
    # End-effector target orientation sampling (XYZ Euler to wxyz quaternion).
    default_orn_roll: float = float(np.pi / 2.0)
    arm_induced_pitch: float = 0.78
    delta_orn_r: list[float] = field(default_factory=lambda: [-0.5, 0.5])
    delta_orn_p: list[float] = field(default_factory=lambda: [-0.5, 0.5])
    delta_orn_y: list[float] = field(default_factory=lambda: [-0.5, 0.5])
    # Initial goal used as the reset-time start point.
    init_ee_cart: list[float] = field(default_factory=lambda: [0.30, 0.0, 0.25])


@dataclass
class CommandsConfig(Commands):
    # Periodic command resampling time in seconds. None disables mid-episode resampling.
    resample_time_s: float | None = None
    # Probability of explicitly sampling a zero-velocity command for stable standing.
    zero_command_prob: float = 0.2


@dataclass
class CurriculumConfig:
    """Expand velocity command ranges when mean tracking_lin_vel exceeds a threshold."""

    enable: bool = False
    # Expansion threshold: per-step episode mean tracking_lin_vel must exceed this value.
    threshold: float = 0.8
    # Expansion step applied on each trigger: [vx, vy, vyaw].
    step_size: list[float] = field(default_factory=lambda: [0.1, 0.05, 0.1])
    # Absolute velocity-range limits to prevent unbounded expansion.
    max_vel_limit: list[float] = field(default_factory=lambda: [1.0, 0.4, 0.8])


@dataclass
class RewardConfig:
    scales: dict[str, float]
    tracking_sigma: float
    base_height_target: float
    target_foot_height: float = 0.1
    object_sigma: float = 0.1
    # Soft limits for 12 leg joints in radians. Empty lists disable the reward term.
    leg_dof_upper_limits: list[float] = field(default_factory=list)
    leg_dof_lower_limits: list[float] = field(default_factory=list)
    dof_pos_limit_margin: float = 0.01


@dataclass
class HistoryConfig:
    """Actor/critic observation history lengths. A value of 1 disables history."""

    num_actor_history: int = 1
    num_critic_history: int = 1


@dataclass
class ArmStageConfig:
    freeze_arm_joints: bool = False
    disable_ee_goal_trajectory: bool = False
    fixed_ee_goal_cart: list[float] = field(default_factory=lambda: [0.30, 0.0, 0.25])


@registry.envcfg("Go2ArmManipLoco")
@dataclass
class Go2ArmManipLocoCfg(Go2ArmBaseCfg):
    scene: SceneCfg = field(default_factory=_default_go2_arm_scene)
    model_file: str = field(default_factory=_default_go2_arm_model_file)
    max_episode_seconds: float = 20.0
    init_state: InitState = field(default_factory=InitState)
    commands: CommandsConfig = field(default_factory=CommandsConfig)  # type: ignore[assignment]
    reward_config: RewardConfig | None = None
    sensor: Go2ArmSensor = field(default_factory=Go2ArmSensor)  # type: ignore[assignment]
    domain_rand: Go2ArmDomainRandConfig = field(default_factory=Go2ArmDomainRandConfig)
    goal_ee: EEGoalConfig = field(default_factory=EEGoalConfig)
    history: HistoryConfig = field(default_factory=HistoryConfig)
    arm_stage: ArmStageConfig = field(default_factory=ArmStageConfig)
    curriculum: CurriculumConfig = field(default_factory=CurriculumConfig)


class Go2ArmManipLocoDRProvider(LocomotionDRProvider):
    def __init__(
        self,
        *,
        base_kp: np.ndarray | None = None,
        base_kd: np.ndarray | None = None,
        base_body_mass: np.ndarray | None = None,
        base_geom_friction: np.ndarray | None = None,
        ground_geom_id: int | None = None,
        base_dof_armature: np.ndarray | None = None,
    ):
        self._base_kp = base_kp
        self._base_kd = base_kd
        self._base_body_mass = base_body_mass
        self._base_geom_friction = base_geom_friction
        self._ground_geom_id = ground_geom_id
        self._base_dof_armature = base_dof_armature

    def _sample_commands(self, env: Any, num_reset: int) -> np.ndarray:
        commands = super()._sample_commands(env, num_reset)
        return env._postprocess_velocity_commands(commands)

    def _get_base_actuator_gains(self, env: Any) -> tuple[np.ndarray | None, np.ndarray | None]:
        return self._base_kp, self._base_kd

    def _get_reset_randomization_baselines(
        self, env: Any
    ) -> tuple[np.ndarray | None, np.ndarray | None, int | None, np.ndarray | None]:
        return (
            self._base_body_mass,
            self._base_geom_friction,
            self._ground_geom_id,
            self._base_dof_armature,
        )

    def build_reset_plan(self, env: Any, env_ids: np.ndarray) -> ResetPlan:
        plan = super().build_reset_plan(env, env_ids)
        env.reset_ee_goals(env_ids)
        # Update command curriculum at episode end before resetting timers.
        env._update_command_curriculum(env_ids)
        # Reset command timers. reset_ee_goals already clears _arm_goal_timer.
        env._cmd_timer[env_ids] = 0
        env._arm_goal_timer[env_ids] = 0
        # Clear history buffers for reset environments.
        env._history_obs_buf[env_ids] = 0.0
        env._history_critic_buf[env_ids] = 0.0
        env.phase[env_ids] = 0.0
        env._write_feet_phase(env_ids, env._command_is_moving(plan.info_updates["commands"]))
        return plan

    def _compute_reset_obs(
        self,
        env: Any,
        env_ids: np.ndarray,
        info_updates: dict[str, Any],
        linvel: np.ndarray,
        gyro: np.ndarray,
        gravity: np.ndarray,
        dof_pos: np.ndarray,
        dof_vel: np.ndarray,
    ) -> dict[str, np.ndarray]:
        ee_local_pos, _ = env.get_ee_local_pose()
        # info_updates may contain global info arrays; slice entries for env_ids.
        n = len(env_ids)
        sliced_info: dict[str, Any] = {}
        for k, v in info_updates.items():
            if isinstance(v, np.ndarray) and v.ndim >= 1 and v.shape[0] == env._num_envs:
                sliced_info[k] = v[env_ids]
            else:
                sliced_info[k] = v
        raw = env._compute_raw_obs(  # type: ignore[no-any-return]
            sliced_info,
            linvel,
            gyro,
            gravity,
            dof_pos,
            dof_vel,
            ee_local_pos[env_ids],
            env.curr_ee_goal_cart[env_ids],
            env.feet_phase[env_ids],
        )
        del n
        return env._update_history(raw, env_ids=env_ids)  # type: ignore[no-any-return]


@registry.env("Go2ArmManipLoco", sim_backend="motrix")
@registry.env("Go2ArmManipLoco", sim_backend="mujoco")
class Go2ArmManipLocoEnv(Go2ArmBaseEnv):
    _cfg: Go2ArmManipLocoCfg

    def __init__(self, cfg: Go2ArmManipLocoCfg, num_envs=1, backend_type="mujoco"):
        if cfg.reward_config is None:
            raise ValueError("reward_config must be provided via Hydra configuration")
        if backend_type not in {"mujoco", "motrix"}:
            raise ValueError(
                "Go2ArmManipLoco supports only the mujoco and motrix backends, "
                f"got {backend_type!r}"
            )

        scene = _resolve_go2_arm_scene(cfg)
        backend_kwargs: dict[str, Any] = {
            "base_name": cfg.asset.base_name,
            "push_body_name": cfg.domain_rand.push_body_name,
        }
        if backend_type == "motrix":
            backend_kwargs["motrix_max_iterations"] = cfg.iterations
        else:
            backend_kwargs["position_actuator_gains"] = build_go2_arm_position_gains(
                cfg.control_config
            )
            backend_kwargs["iterations"] = cfg.iterations
        backend = create_backend(
            backend_type,
            scene,
            num_envs,
            cfg.sim_dt,
            **backend_kwargs,
        )
        super().__init__(cfg, backend, num_envs)
        if self._num_action != 18:
            raise ValueError(f"Go2ArmManipLoco expects 18 actuators, got {self._num_action}")
        if not 0.0 <= cfg.commands.zero_command_prob <= 1.0:
            raise ValueError(
                "env.commands.zero_command_prob must be in [0, 1], "
                f"got {cfg.commands.zero_command_prob}"
            )

        self._enable_reward_log = True
        self._reward_cfg = cfg.reward_config
        self._leg_pose_weights = np.array([1.0, 1.0, 0.1] * 4 + [0.0] * 6, dtype=get_global_dtype())
        self._init_reward_functions()
        self._init_ee_goal_buffers(num_envs)
        self._current_ee_local_pos = np.zeros((num_envs, 3), dtype=get_global_dtype())
        self.phase = np.zeros((num_envs,), dtype=np.float32)
        self.feet_phase = np.zeros((num_envs, len(cfg.sensor.feet_force)), dtype=np.float32)
        self.gait_frequency = 2.0
        self.feet_force = np.zeros((num_envs, len(cfg.sensor.feet_force), 3), dtype=np.float32)
        self.feet_pos = np.zeros((num_envs, len(cfg.sensor.feet_pos), 3), dtype=np.float32)

        # Mid-episode command resampling. None disables periodic resampling.
        if cfg.commands.resample_time_s is not None:
            self._cmd_resample_steps: int | None = max(
                1, int(cfg.commands.resample_time_s / cfg.ctrl_dt)
            )
            self._cmd_timer = np.random.randint(
                0, self._cmd_resample_steps, size=(num_envs,), dtype=np.int32
            )
        else:
            self._cmd_resample_steps = None
            self._cmd_timer = np.zeros((num_envs,), dtype=np.int32)

        # Per-env episode tracking_lin_vel accumulator for command curriculum.
        self._episode_sum_tracking_vel = np.zeros(num_envs, dtype=np.float64)
        self._episode_steps = np.zeros(num_envs, dtype=np.int32)

        # History buffers.
        # Actor obs excludes linvel (first 3 dims) to avoid bypassing the estimator.
        # raw_obs layout: linvel(3)+gyro(3)+(-gravity)(3)+command(3)+feet_phase(4)+
        #               diff(18)+dof_vel(18)+ee_local_pos(3)+ee_goal_cart(3)+
        #               ee_error(3)+last_actions(18) = 79 dims.
        _CRITIC_ONE = 79  # Single-step critic obs dim, including privileged linvel.
        _ACTOR_ONE = 76  # Single-step actor obs dim after removing linvel[0:3].
        H_a = cfg.history.num_actor_history
        H_c = cfg.history.num_critic_history
        self._actor_one_step_dim = _ACTOR_ONE
        self._critic_one_step_dim = _CRITIC_ONE
        self._history_obs_buf = np.zeros((num_envs, H_a * _ACTOR_ONE), dtype=get_global_dtype())
        self._history_critic_buf = np.zeros((num_envs, H_c * _CRITIC_ONE), dtype=get_global_dtype())

        base_kp: np.ndarray | None = None
        base_kd: np.ndarray | None = None
        if cfg.domain_rand.randomize_kp or cfg.domain_rand.randomize_kd:
            base_kp, base_kd = backend.get_actuator_gains()

        base_body_mass: np.ndarray | None = None
        if cfg.domain_rand.randomize_body_mass:
            base_body_mass = backend.get_body_mass()

        base_geom_friction: np.ndarray | None = None
        ground_geom_id: int | None = None
        if cfg.domain_rand.randomize_ground_friction:
            base_geom_friction = backend.get_geom_friction()
            ground_geom_id = backend.get_geom_id(cfg.asset.ground)

        base_dof_armature: np.ndarray | None = None
        if cfg.domain_rand.randomize_dof_armature:
            base_dof_armature = backend.get_dof_armature()

        dr_provider = Go2ArmManipLocoDRProvider(
            base_kp=base_kp,
            base_kd=base_kd,
            base_body_mass=base_body_mass,
            base_geom_friction=base_geom_friction,
            ground_geom_id=ground_geom_id,
            base_dof_armature=base_dof_armature,
        )
        self._init_domain_randomization(dr_provider)

    @property
    def obs_groups_spec(self) -> dict[str, int]:
        H_a = self._cfg.history.num_actor_history
        H_c = self._cfg.history.num_critic_history
        return {"obs": H_a * self._actor_one_step_dim, "critic": H_c * self._critic_one_step_dim}

    def _init_ee_goal_buffers(self, num_envs: int) -> None:
        dtype = get_global_dtype()
        self.curr_ee_goal_cart = np.zeros((num_envs, 3), dtype=dtype)
        self.curr_ee_goal_sphere = np.zeros((num_envs, 3), dtype=dtype)
        self.ee_goal_orn_euler = np.zeros((num_envs, 3), dtype=dtype)
        self.ee_goal_orn_quat = np.tile(
            np.asarray([1.0, 0.0, 0.0, 0.0], dtype=dtype),
            (num_envs, 1),
        )
        self.ee_goal_orn_delta_rpy = np.zeros((num_envs, 3), dtype=dtype)
        # Goal position in world coordinates, used for render-time visualization.
        self.curr_ee_goal_world = np.zeros((num_envs, 3), dtype=dtype)
        self._ee_start_sphere = np.zeros((num_envs, 3), dtype=dtype)
        self._ee_goal_sphere = np.zeros((num_envs, 3), dtype=dtype)
        self._arm_goal_timer = np.zeros((num_envs,), dtype=np.int32)
        self._traj_steps = np.ones((num_envs,), dtype=np.int32)
        self._traj_total_steps = np.ones((num_envs,), dtype=np.int32)

    def _sample_timing(self, env_ids: np.ndarray) -> None:
        """Sample movement and hold durations for env_ids."""
        cfg = self._cfg.goal_ee
        dt = self._cfg.ctrl_dt
        traj_t = np.random.uniform(*cfg.traj_time_range, size=len(env_ids))
        hold_t = np.random.uniform(*cfg.hold_time_range, size=len(env_ids))
        traj_s = np.maximum(1, np.round(traj_t / dt).astype(np.int32))
        hold_s = np.maximum(0, np.round(hold_t / dt).astype(np.int32))
        self._traj_steps[env_ids] = traj_s
        self._traj_total_steps[env_ids] = traj_s + hold_s

    def _collision_check_sphere(self, starts: np.ndarray, goals: np.ndarray) -> np.ndarray:
        """Check spherical lerp paths for collisions after Cartesian conversion."""
        cfg = self._cfg.goal_ee
        dtype = get_global_dtype()
        n = max(2, cfg.num_collision_check_samples)
        t = np.linspace(0.0, 1.0, n, dtype=dtype)  # (n,)
        path_sphere = (
            starts[:, None, :] + (goals - starts)[:, None, :] * t[None, :, None]
        )  # (N, n, 3)
        path_cart = _sphere2cart(path_sphere.reshape(-1, 3)).reshape(len(starts), n, 3)
        upper = np.asarray(cfg.collision_upper_limits, dtype=dtype)
        lower = np.asarray(cfg.collision_lower_limits, dtype=dtype)
        inside_collision_box = np.all(path_cart < upper, axis=2) & np.all(path_cart > lower, axis=2)
        collision_mask = np.any(inside_collision_box, axis=1)
        underground_mask = np.any(path_cart[..., 2] < float(cfg.underground_limit), axis=1)
        return collision_mask | underground_mask

    def _sample_goal_spheres(self, env_ids: np.ndarray, start_spheres: np.ndarray) -> None:
        """Sample goal spheres and write them into _ee_goal_sphere[env_ids]."""
        cfg = self._cfg.goal_ee
        dtype = get_global_dtype()
        init_sphere = _cart2sphere(np.asarray(cfg.init_ee_cart, dtype=dtype)[None, :])[0]
        candidates = np.broadcast_to(init_sphere, (len(env_ids), 3)).copy()
        remaining = np.arange(len(env_ids), dtype=np.int32)
        for _ in range(max(1, cfg.num_resample_attempts)):
            l = np.random.uniform(*cfg.sphere_l_range, size=len(remaining)).astype(dtype)
            phi = np.random.uniform(*cfg.sphere_phi_range, size=len(remaining)).astype(dtype)
            theta = np.random.uniform(*cfg.sphere_theta_range, size=len(remaining)).astype(dtype)
            new_goals = np.stack([l, phi, theta], axis=1)
            candidates[remaining] = new_goals
            unsafe = self._collision_check_sphere(start_spheres[remaining], new_goals)
            remaining = remaining[unsafe]
            if len(remaining) == 0:
                break
        self._ee_goal_sphere[env_ids] = candidates

    def _sample_ee_goal_orn_delta(self, env_ids: np.ndarray, *, is_init: bool) -> None:
        if len(env_ids) == 0:
            return
        if is_init:
            self.ee_goal_orn_delta_rpy[env_ids] = 0.0
            return

        dtype = get_global_dtype()
        ranges = (
            self._cfg.goal_ee.delta_orn_r,
            self._cfg.goal_ee.delta_orn_p,
            self._cfg.goal_ee.delta_orn_y,
        )
        for axis, bounds in enumerate(ranges):
            low_high = np.asarray(bounds, dtype=dtype)
            if low_high.shape != (2,):
                raise ValueError("goal_ee delta orientation ranges must have shape (2,)")
            if low_high[1] < low_high[0]:
                raise ValueError("goal_ee delta orientation range high must be >= low")
            self.ee_goal_orn_delta_rpy[env_ids, axis] = np.random.uniform(
                low=low_high[0],
                high=low_high[1],
                size=(len(env_ids),),
            ).astype(dtype)

    def _update_curr_ee_goal_orientation(self, env_ids: np.ndarray) -> None:
        if len(env_ids) == 0:
            return
        dtype = get_global_dtype()
        goal_cfg = self._cfg.goal_ee
        goal_local = self.curr_ee_goal_cart[env_ids]
        goal_sphere = self.curr_ee_goal_sphere[env_ids]
        delta = self.ee_goal_orn_delta_rpy[env_ids]

        default_yaw = np.arctan2(goal_local[:, 1], goal_local[:, 0])
        default_pitch = -goal_sphere[:, 1] + float(goal_cfg.arm_induced_pitch)
        roll = float(goal_cfg.default_orn_roll) + delta[:, 0]
        pitch = default_pitch + delta[:, 1]
        yaw = default_yaw + delta[:, 2]

        self.ee_goal_orn_euler[env_ids] = np.stack([roll, pitch, yaw], axis=1).astype(dtype)
        self.ee_goal_orn_quat[env_ids] = np.atleast_2d(
            np_quat_from_euler_xyz(roll, pitch, yaw)
        ).astype(dtype)

    def reset_ee_goals(self, env_ids: np.ndarray) -> None:
        """Reset EE goals by sampling the first segment from init_ee_cart."""
        env_ids = np.asarray(env_ids, dtype=np.int32).reshape(-1)
        if len(env_ids) == 0:
            return
        stage_cfg = self._cfg.arm_stage
        if stage_cfg.disable_ee_goal_trajectory:
            fixed_goal = np.asarray(stage_cfg.fixed_ee_goal_cart, dtype=get_global_dtype())
            if fixed_goal.shape != (3,):
                raise ValueError(
                    f"env.arm_stage.fixed_ee_goal_cart must have shape (3,), got {fixed_goal.shape}"
                )
            fixed_sphere = _cart2sphere(fixed_goal[None, :])[0]
            self._ee_start_sphere[env_ids] = fixed_sphere
            self._ee_goal_sphere[env_ids] = fixed_sphere
            self._traj_steps[env_ids] = 1
            self._traj_total_steps[env_ids] = 1
            self._arm_goal_timer[env_ids] = 0
            self.curr_ee_goal_cart[env_ids] = fixed_goal
            self.curr_ee_goal_sphere[env_ids] = fixed_sphere
            self._sample_ee_goal_orn_delta(env_ids, is_init=True)
            self._update_curr_ee_goal_orientation(env_ids)
            return
        dtype = get_global_dtype()
        init_sphere = _cart2sphere(
            np.asarray(self._cfg.goal_ee.init_ee_cart, dtype=dtype)[None, :]
        )[0]
        self._ee_start_sphere[env_ids] = init_sphere
        self._sample_goal_spheres(
            env_ids,
            np.broadcast_to(init_sphere, (len(env_ids), 3)).copy(),
        )
        self._sample_timing(env_ids)
        self._arm_goal_timer[env_ids] = 0
        self.curr_ee_goal_sphere[env_ids] = init_sphere
        self.curr_ee_goal_cart[env_ids] = _sphere2cart(
            np.broadcast_to(init_sphere, (len(env_ids), 3))
        )
        self._sample_ee_goal_orn_delta(env_ids, is_init=True)
        self._update_curr_ee_goal_orientation(env_ids)

    def _update_command_curriculum(self, env_ids: np.ndarray) -> None:
        """Update velocity command ranges at episode end from tracking_lin_vel.

        This follows the go2_arx_robot.py rule:
          mean(episode_sum[env_ids] / episode_steps[env_ids]) > threshold
        The unweighted tracking_lin_vel maximum is 1.0 per step.
        """
        cur = self._cfg.curriculum
        if not cur.enable:
            return
        ep_steps = np.maximum(self._episode_steps[env_ids], 1)
        mean_per_step = float(np.mean(self._episode_sum_tracking_vel[env_ids] / ep_steps))
        if mean_per_step > cur.threshold:
            step = np.asarray(cur.step_size, dtype=np.float64)
            max_limit = np.asarray(cur.max_vel_limit, dtype=np.float64)
            low = np.asarray(self._cfg.commands.vel_limit[0], dtype=np.float64)
            high = np.asarray(self._cfg.commands.vel_limit[1], dtype=np.float64)
            low = np.clip(low - step, -max_limit, 0.0)
            high = np.clip(high + step, 0.0, max_limit)
            self._cfg.commands.vel_limit = [low.tolist(), high.tolist()]
        # Clear episode statistics for reset environments.
        self._episode_sum_tracking_vel[env_ids] = 0.0
        self._episode_steps[env_ids] = 0

    # Command clipping threshold: small vx/vy/vyaw commands are zeroed out.
    _CMD_CLIP: float = 0.1

    def _command_is_moving(self, commands: np.ndarray) -> np.ndarray:
        command_arr = np.asarray(commands)
        return np.any(np.abs(command_arr[:, :3]) > self._CMD_CLIP, axis=1)

    def _normalize_velocity_commands(self, commands: np.ndarray) -> np.ndarray:
        normalized = np.asarray(commands, dtype=get_global_dtype()).copy()
        normalized[~self._command_is_moving(normalized)] = 0.0
        return normalized

    def _postprocess_velocity_commands(self, commands: np.ndarray) -> np.ndarray:
        processed = self._normalize_velocity_commands(commands)
        prob = float(self._cfg.commands.zero_command_prob)
        if prob <= 0.0 or processed.shape[0] == 0:
            return processed
        zero_mask = np.random.random(size=(processed.shape[0],)) < prob
        processed[zero_mask] = 0.0
        return processed

    def _write_feet_phase(self, env_ids: np.ndarray | slice, is_moving: np.ndarray) -> None:
        phase = self.phase[env_ids]
        feet_phase = self.feet_phase[env_ids].copy()
        feet_phase[:, 0] = phase
        feet_phase[:, 3] = phase
        feet_phase[:, 1] = (phase + 0.5) % 1.0
        feet_phase[:, 2] = (phase + 0.5) % 1.0
        feet_phase[~is_moving] = 0.0
        self.feet_phase[env_ids] = feet_phase

    def _resample_commands(self, env_ids: np.ndarray, info: dict) -> None:
        """Resample velocity commands and zero out small commands."""
        if len(env_ids) == 0:
            return
        low = np.asarray(self._cfg.commands.vel_limit[0], dtype=get_global_dtype())
        high = np.asarray(self._cfg.commands.vel_limit[1], dtype=get_global_dtype())
        new_cmds = np.random.uniform(low=low, high=high, size=(len(env_ids), 3)).astype(
            get_global_dtype()
        )
        new_cmds = self._postprocess_velocity_commands(new_cmds)
        if "commands" in info:
            info["commands"][env_ids] = new_cmds

    def apply_action(self, actions: np.ndarray, state: NpEnvState) -> np.ndarray:
        state.info["last_actions"] = state.info.get("current_actions", np.zeros_like(actions))
        stage_cfg = self._cfg.arm_stage
        if stage_cfg.freeze_arm_joints:
            effective_actions = actions.copy()
            effective_actions[:, 12:18] = 0.0
        else:
            effective_actions = actions
        state.info["current_actions"] = effective_actions
        exec_actions = (
            state.info["last_actions"]
            if self._cfg.control_config.simulate_action_latency
            else effective_actions
        )

        ee_local_pos, ee_local_quat = self.get_ee_local_pose()
        dq_ik = self.compute_arm_ik_delta(
            self.curr_ee_goal_cart,
            ee_local_pos,
            self.ee_goal_orn_quat,
            ee_local_quat,
        )

        leg_ctrl = (
            exec_actions[:, :12] * self._cfg.control_config.action_scale + self.default_angles[:12]
        )
        if stage_cfg.freeze_arm_joints:
            arm_ctrl = np.broadcast_to(self.default_angles[12:18], (self._num_envs, 6)).astype(
                get_global_dtype(),
                copy=False,
            )
        else:
            arm_ctrl = (
                self.get_arm_dof_pos()
                + exec_actions[:, 12:18] * self._cfg.control_config.arm_action_scale
                + self._cfg.ik.gain * dq_ik
            )
        ctrl = np.concatenate([leg_ctrl, arm_ctrl], axis=1, dtype=get_global_dtype())
        return np.clip(ctrl, self.action_space.low, self.action_space.high)

    def _init_reward_functions(self) -> None:
        self._reward_fns: dict[str, Any] = {
            # Tracking rewards.
            "tracking_lin_vel": rewards.tracking_lin_vel,
            "tracking_ang_vel": rewards.tracking_ang_vel,
            # Velocity and orientation penalties.
            "lin_vel_z": rewards.lin_vel_z,
            "ang_vel_xy": rewards.ang_vel_xy,
            "roll": rewards.roll,  # Requires ctx.gravity.
            # Height and joint-pose terms.
            "base_height": rewards.base_height,
            "similar_to_default": rewards.similar_to_default,  # Aligns with Go2 Joystick.
            "leg_pose": rewards.weighted_pose,  # Weighted leg L2 term.
            "dof_pos_limits": self._reward_dof_pos_limits,  # Leg soft limits.
            # Action and effort penalties.
            "action_rate": rewards.action_rate,
            "torques": rewards.torques,  # L1 torque over all 18 DOFs.
            "energy": rewards.energy,  # Requires ctx.dof_vel and info["torques"].
            "dof_vel": self._reward_dof_vel,  # L2 velocity over all 18 DOFs.
            "dof_acc": rewards.dof_acc,  # Requires info["qacc"].
            # Standing penalty.
            "stand_still": self._reward_stand_still,  # Penalizes leg pose at zero command.
            # Survival.
            "alive": rewards.alive,
            # Gait terms.
            "swing_feet_z": self._reward_swing_feet_z,
            "foot_drag": self._reward_foot_drag,
            "contact": self._reward_contact,
            # Manipulation rewards.
            "object_distance": self._reward_object_distance,
            "object_distance_l2": self._reward_object_distance_l2,
            # Arm collision penalty.
            "arm_collision": self._reward_arm_collision,
        }

    def update_state(self, state: NpEnvState) -> NpEnvState:
        # Mid-episode command resampling, enabled only when resample_time_s is set.
        if self._cmd_resample_steps is not None:
            self._cmd_timer += 1
            resample_ids = np.where(self._cmd_timer >= self._cmd_resample_steps)[0].astype(np.int32)
            if len(resample_ids) > 0:
                self._resample_commands(resample_ids, state.info)
                self._cmd_timer[resample_ids] = 0

        # Gait phase update: zero commands reset the phase to a full-stance pattern.
        # This gives contact a four-feet contact target and naturally disables swing_feet_z.
        cmd = state.info.get("commands", np.zeros((self._num_envs, 3), dtype=np.float32))
        is_moving = self._command_is_moving(cmd)
        advanced = np.fmod(self.phase + self._cfg.ctrl_dt * self.gait_frequency, 1.0)
        self.phase = np.where(is_moving, advanced, 0.0)
        self._write_feet_phase(slice(None), is_moving)

        # EE goal trajectory update.
        stage_cfg = self._cfg.arm_stage
        if stage_cfg.disable_ee_goal_trajectory:
            fixed_goal = np.asarray(stage_cfg.fixed_ee_goal_cart, dtype=get_global_dtype())
            if fixed_goal.shape != (3,):
                raise ValueError(
                    f"env.arm_stage.fixed_ee_goal_cart must have shape (3,), got {fixed_goal.shape}"
                )
            self.curr_ee_goal_cart[:] = fixed_goal
            self.curr_ee_goal_sphere[:] = _cart2sphere(fixed_goal[None, :])[0]
        else:
            self._arm_goal_timer += 1
            expired = np.where(self._arm_goal_timer >= self._traj_total_steps)[0].astype(np.int32)
            if len(expired) > 0:
                self._ee_start_sphere[expired] = self._ee_goal_sphere[expired].copy()
                self._sample_goal_spheres(expired, self._ee_start_sphere[expired])
                self._sample_ee_goal_orn_delta(expired, is_init=False)
                self._sample_timing(expired)
                self._arm_goal_timer[expired] = 0
            # Spherical interpolation, updated every step.
            t_frac = np.clip(self._arm_goal_timer / self._traj_steps, 0.0, 1.0).astype(
                get_global_dtype()
            )[:, None]  # (num_envs, 1)
            curr_sphere = (
                self._ee_start_sphere + (self._ee_goal_sphere - self._ee_start_sphere) * t_frac
            )
            self.curr_ee_goal_sphere[:] = curr_sphere
            self.curr_ee_goal_cart[:] = _sphere2cart(curr_sphere)
        self._update_curr_ee_goal_orientation(np.arange(self._num_envs, dtype=np.int32))
        # Compute the world-space goal position for render-time visualization.
        ab_pos = self._backend.get_sensor_data("armbasepoint_world_pos")  # (N, 3)
        ab_quat = self._backend.get_sensor_data("armbasepoint_world_quat")  # (N, 4)
        R = np_matrix_from_quat(ab_quat)  # (N, 3, 3)
        self.curr_ee_goal_world[:] = ab_pos + np.einsum("nij,nj->ni", R, self.curr_ee_goal_cart)

        linvel = self.get_local_linvel()
        gyro = self.get_gyro()
        gravity = self._backend.get_sensor_data("upvector")
        dof_pos = self.get_dof_pos()
        dof_vel = self.get_dof_vel()
        ee_local_pos, _ = self.get_ee_local_pose()
        self._current_ee_local_pos = ee_local_pos

        self.feet_force[:, :, :] = 0
        for i, sensor_name in enumerate(self._cfg.sensor.feet_force):
            self.feet_force[:, i, :] = self._backend.get_sensor_data(sensor_name)
        for i, sensor_name in enumerate(self._cfg.sensor.feet_pos):
            self.feet_pos[:, i, :] = self._backend.get_sensor_data(sensor_name)

        terminated = gravity[:, 2] <= 0.5
        reward = self._compute_reward(
            state.info, linvel, gyro, gravity, dof_pos, dof_vel, ee_local_pos
        )
        obs = self._compute_obs(
            state.info,
            linvel,
            gyro,
            gravity,
            dof_pos,
            dof_vel,
            ee_local_pos,
            self.curr_ee_goal_cart,
            self.feet_phase,
        )
        return state.replace(obs=obs, reward=reward, terminated=terminated)

    def _compute_raw_obs(
        self,
        info: dict,
        linvel: np.ndarray,
        gyro: np.ndarray,
        gravity: np.ndarray,
        dof_pos: np.ndarray,
        dof_vel: np.ndarray,
        ee_local_pos: np.ndarray,
        ee_goal_cart: np.ndarray,
        feet_phase: np.ndarray,
    ) -> np.ndarray:
        """Compute single-step 79-dim obs (no history).

        Layout: linvel(3)+gyro(3)+(-gravity)(3)+command(3)+feet_phase(4)+
                diff(18)+dof_vel(18)+ee_local_pos(3)+ee_goal_cart(3)+ee_error(3)+
                last_actions(18) = 79
        """
        noise_cfg = self._cfg.noise_config
        diff = dof_pos - self.default_angles
        linvel = self._obs_noise(linvel, noise_cfg.scale_linvel)
        gyro = self._obs_noise(gyro, noise_cfg.scale_gyro)
        gravity = self._obs_noise(gravity, noise_cfg.scale_gravity)
        diff = self._obs_noise(diff, noise_cfg.scale_joint_angle)
        dof_vel = self._obs_noise(dof_vel, noise_cfg.scale_joint_vel)
        ee_local_pos = self._obs_noise(ee_local_pos, noise_cfg.scale_ee_pos)
        n = len(dof_pos)
        command = info["commands"] if info["commands"].shape[0] == n else info["commands"][:n]
        last_actions = info.get(
            "current_actions", np.zeros((n, self._num_action), dtype=get_global_dtype())
        )
        ee_error = ee_local_pos - ee_goal_cart
        return np.concatenate(
            [
                linvel,  # 3
                gyro,  # 3
                -gravity,  # 3
                command,  # 3
                feet_phase,  # 4
                diff,  # 18
                dof_vel,  # 18
                ee_local_pos,  # 3
                ee_goal_cart,  # 3
                ee_error,  # 3
                last_actions,  # 18
            ],
            axis=1,
            dtype=get_global_dtype(),
        )

    def _update_history(
        self,
        raw_obs: np.ndarray,
        env_ids: np.ndarray | None = None,
    ) -> dict[str, np.ndarray]:
        """Update history buffers and return obs dict (with or without env_ids slice).

        Actor buffer stores obs WITHOUT linvel (raw_obs[:, 3:], 72-dim) so the actor
        cannot shortcut the estimator.  Critic buffer stores full 75-dim obs (linvel
        at [0:3] is used as estimator supervision signal).
        """
        A = self._actor_one_step_dim  # 72
        C = self._critic_one_step_dim  # 75
        H_a = self._cfg.history.num_actor_history
        H_c = self._cfg.history.num_critic_history
        actor_step = raw_obs[:, 3:] if raw_obs.ndim == 2 else raw_obs[3:]
        if env_ids is None:
            if H_a > 1:
                self._history_obs_buf = np.roll(self._history_obs_buf, -A, axis=1)
            self._history_obs_buf[:, -A:] = actor_step
            if H_c > 1:
                self._history_critic_buf = np.roll(self._history_critic_buf, -C, axis=1)
            self._history_critic_buf[:, -C:] = raw_obs
            return {
                "obs": self._history_obs_buf.copy(),
                "critic": self._history_critic_buf.copy(),
            }
        else:
            if H_a > 1:
                self._history_obs_buf[env_ids] = np.roll(self._history_obs_buf[env_ids], -A, axis=1)
            self._history_obs_buf[env_ids, -A:] = actor_step
            if H_c > 1:
                self._history_critic_buf[env_ids] = np.roll(
                    self._history_critic_buf[env_ids], -C, axis=1
                )
            self._history_critic_buf[env_ids, -C:] = raw_obs
            return {
                "obs": self._history_obs_buf[env_ids].copy(),
                "critic": self._history_critic_buf[env_ids].copy(),
            }

    def _compute_obs(
        self,
        info: dict,
        linvel: np.ndarray,
        gyro: np.ndarray,
        gravity: np.ndarray,
        dof_pos: np.ndarray,
        dof_vel: np.ndarray,
        ee_local_pos: np.ndarray,
        ee_goal_cart: np.ndarray,
        feet_phase: np.ndarray,
    ) -> dict[str, np.ndarray]:
        raw = self._compute_raw_obs(
            info, linvel, gyro, gravity, dof_pos, dof_vel, ee_local_pos, ee_goal_cart, feet_phase
        )
        return self._update_history(raw)

    def _compute_reward(
        self,
        info: dict,
        linvel: np.ndarray,
        gyro: np.ndarray,
        gravity: np.ndarray,
        dof_pos: np.ndarray,
        dof_vel: np.ndarray,
        ee_local_pos: np.ndarray,
    ) -> np.ndarray:
        dtype = get_global_dtype()
        reward = np.zeros((self._num_envs,), dtype=dtype)
        cfg = self._reward_cfg
        self._current_ee_local_pos = ee_local_pos
        ctx = RewardContext(
            info=info,
            linvel=linvel,
            gyro=gyro,
            gravity=gravity,
            dof_pos=dof_pos,
            dof_vel=dof_vel,
            num_envs=self._num_envs,
            default_angles=self.default_angles,
            tracking_sigma=cfg.tracking_sigma,
            base_height_target=cfg.base_height_target,
            base_height=self._backend.get_base_pos()[:, 2],
            pose_weights=self._leg_pose_weights,
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
            if name == "tracking_lin_vel":
                self._episode_sum_tracking_vel += rew.astype(np.float64)
            if should_log:
                log[f"reward/{name}"] = float(np.mean(weighted_rew))

        self._episode_steps += 1
        info["log"] = log
        return reward * self._cfg.ctrl_dt

    def _reward_swing_feet_z(self, _ctx: RewardContext) -> np.ndarray:
        is_swing = self.feet_phase >= 0.6
        height_error = np.square(self.feet_pos[:, :, 2] - self._reward_cfg.target_foot_height)
        swing_rew = np.exp(-height_error / 0.01) * is_swing
        reward: np.ndarray = np.sum(swing_rew, axis=1) / len(self._cfg.sensor.feet_pos)
        return reward

    def _reward_foot_drag(self, _ctx: RewardContext) -> np.ndarray:
        foot_heights = self.feet_pos[..., 2]
        foot_contact = self.get_foot_contact()
        is_swing = foot_contact < 0.5
        safe_height = self._reward_cfg.target_foot_height / 2.0
        height_error = np.clip(safe_height - foot_heights, 0.0, None)
        error = np.square(height_error) * is_swing
        drag_penalty: np.ndarray = np.sum(error, axis=1)
        return drag_penalty

    def _reward_contact(self, _ctx: RewardContext) -> np.ndarray:
        contact = self.feet_force[:, :, 2] > 0.1
        res = np.zeros(self._num_envs, dtype=np.float32)
        for i in range(len(self._cfg.sensor.feet_force)):
            is_contact = (self.feet_phase[:, i] < 0.6) | (self.gait_frequency < 1.0e-8)
            res += (contact[:, i] == is_contact).astype(np.float32)
        return res / len(self._cfg.sensor.feet_force)

    def _reward_object_distance(self, _ctx: RewardContext) -> np.ndarray:
        dis_err = np.sum(
            np.square(self._current_ee_local_pos - self.curr_ee_goal_cart),
            axis=1,
        )
        return np.exp(-dis_err / self._reward_cfg.object_sigma)  # type: ignore[no-any-return]

    def _reward_object_distance_l2(self, _ctx: RewardContext) -> np.ndarray:
        return np.sum(
            np.square(self._current_ee_local_pos - self.curr_ee_goal_cart),
            axis=1,
        )

    def _reward_stand_still(self, ctx: RewardContext) -> np.ndarray:
        """Penalize leg deviation from the default pose when command is near zero."""
        commands = ctx.info["commands"]
        is_still = (~self._command_is_moving(commands)).astype(get_global_dtype())
        assert ctx.dof_pos is not None
        dof_error = np.sum(np.abs(ctx.dof_pos[:, :12] - ctx.default_angles[:12]), axis=1)
        return is_still * dof_error

    def _reward_dof_vel(self, ctx: RewardContext) -> np.ndarray:
        """L2 velocity penalty over all 18 joints."""
        assert ctx.dof_vel is not None
        return np.sum(np.square(ctx.dof_vel), axis=1)  # type: ignore[no-any-return]

    def _reward_dof_pos_limits(self, ctx: RewardContext) -> np.ndarray:
        """Leg soft-limit penalty configured through reward_config limits."""
        cfg = self._reward_cfg
        if not cfg.leg_dof_upper_limits or not cfg.leg_dof_lower_limits:
            return np.zeros(self._num_envs, dtype=get_global_dtype())
        dtype = get_global_dtype()
        upper = np.asarray(cfg.leg_dof_upper_limits, dtype=dtype)
        lower = np.asarray(cfg.leg_dof_lower_limits, dtype=dtype)
        m = cfg.dof_pos_limit_margin
        leg_pos = ctx.dof_pos[:, :12]
        over = np.square(np.maximum(leg_pos - upper + m, 0.0))
        under = np.square(np.maximum(lower + m - leg_pos, 0.0))
        return np.sum(over + under, axis=1)  # type: ignore[no-any-return]

    _ARM_TOUCH_SENSORS = (
        "arm_touch_base",
        "arm_touch_link1",
        "arm_touch_link2",
        "arm_touch_link3",
        "arm_touch_link4",
        "arm_touch_link5",
        "arm_touch_link6",
        "arm_touch_eef",
        "arm_touch_g2base",
    )

    def _reward_arm_collision(self, _ctx: RewardContext) -> np.ndarray:
        """Sum arm-link contact forces. The scale should be negative."""
        total = np.zeros(self._num_envs, dtype=get_global_dtype())
        for name in self._ARM_TOUCH_SENSORS:
            total += self._backend.get_sensor_data(name)[:, 0]
        return total
