from __future__ import annotations

import abc
from typing import Any, cast

import numpy as np
from etils import epath

from unilab.base.dtype_config import get_global_dtype
from unilab.dr.types import (
    DomainRandomizationCapabilities,
    IntervalRandomizationPlan,
    ResetPlan,
    ResetRandomizationPayload,
)
from unilab.utils.math_utils import (
    np_quat_apply,
    np_quat_from_euler_xyz,
    np_quat_inv,
    np_quat_mul,
    np_sample_uniform,
    np_yaw_to_quat,
)


def _zero_actions(num_reset: int, num_action: int) -> np.ndarray:
    return np.zeros((num_reset, num_action), dtype=get_global_dtype())


def _build_common_reset_randomization(env: Any, num_reset: int) -> ResetRandomizationPayload | None:
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

    return None if payload.is_empty() else payload


def _validate_common_reset_randomization(
    env: Any, capabilities: DomainRandomizationCapabilities
) -> None:
    payload = _build_common_reset_randomization(env, num_reset=1)
    if payload is None:
        return
    unsupported = payload.requested_terms() - capabilities.supported_reset_terms
    if unsupported:
        terms = ", ".join(sorted(unsupported))
        raise NotImplementedError(
            f"{env._backend.backend_type} backend does not support reset randomization terms: {terms}"
        )


def _build_push_plan(env: Any, step_counter: int) -> IntervalRandomizationPlan | None:
    domain_rand = getattr(env.cfg, "domain_rand", None)
    if domain_rand is None or not getattr(domain_rand, "push_robots", False):
        return None
    if step_counter % domain_rand.push_interval != 0:
        return None
    return IntervalRandomizationPlan(
        push_perturbation_limit=np.asarray(domain_rand.max_force, dtype=np.float64)
    )


def _validate_interval_push_support(
    env: Any, capabilities: DomainRandomizationCapabilities
) -> None:
    domain_rand = getattr(env.cfg, "domain_rand", None)
    if domain_rand is None or not getattr(domain_rand, "push_robots", False):
        return
    if not capabilities.supports_interval_push:
        raise NotImplementedError(
            f"{env._backend.backend_type} backend does not support interval push"
        )


class DomainRandomizationProvider(abc.ABC):
    @abc.abstractmethod
    def validate(self, env: Any, capabilities: DomainRandomizationCapabilities) -> None:
        pass

    @abc.abstractmethod
    def build_reset_plan(self, env: Any, env_ids: np.ndarray) -> ResetPlan:
        pass

    @abc.abstractmethod
    def build_reset_observation(
        self, env: Any, env_ids: np.ndarray, info_updates: dict[str, Any]
    ) -> dict[str, np.ndarray]:
        pass

    def build_interval_randomization_plan(
        self, env: Any, step_counter: int
    ) -> IntervalRandomizationPlan | None:
        return None


class _BaseLocomotionProvider(DomainRandomizationProvider):
    def validate(self, env: Any, capabilities: DomainRandomizationCapabilities) -> None:
        _validate_common_reset_randomization(env, capabilities)
        _validate_interval_push_support(env, capabilities)

    def build_interval_randomization_plan(
        self, env: Any, step_counter: int
    ) -> IntervalRandomizationPlan | None:
        return _build_push_plan(env, step_counter)

    def _build_qpos_qvel(self, env: Any, num_reset: int) -> tuple[np.ndarray, np.ndarray]:
        qpos = np.tile(env._init_qpos, (num_reset, 1))
        qvel = np.tile(env._init_qvel, (num_reset, 1))

        qpos[:, 0:2] += np.random.uniform(-0.5, 0.5, (num_reset, 2))
        yaw = np.random.uniform(-np.pi, np.pi, (num_reset,))
        quat_yaw = np_yaw_to_quat(yaw)
        qpos[:, 3:7] = np_quat_mul(qpos[:, 3:7], quat_yaw)
        qvel[:, 0:6] = self._sample_base_velocity(env, num_reset)
        return qpos, qvel

    def _sample_commands(self, env: Any, num_reset: int) -> np.ndarray:
        low = np.asarray(env.cfg.commands.vel_limit[0], dtype=get_global_dtype())
        high = np.asarray(env.cfg.commands.vel_limit[1], dtype=get_global_dtype())
        return np.asarray(
            np.random.uniform(low=low, high=high, size=(num_reset, 3)), dtype=get_global_dtype()
        )

    @abc.abstractmethod
    def _sample_base_velocity(self, env: Any, num_reset: int) -> np.ndarray:
        pass


class Go1JoystickProvider(_BaseLocomotionProvider):
    def _sample_base_velocity(self, env: Any, num_reset: int) -> np.ndarray:
        return np.random.uniform(-0.5, 0.5, (num_reset, 6))

    def build_reset_plan(self, env: Any, env_ids: np.ndarray) -> ResetPlan:
        num_reset = len(env_ids)
        qpos, qvel = self._build_qpos_qvel(env, num_reset)
        info_updates = {
            "commands": self._sample_commands(env, num_reset),
            "current_actions": _zero_actions(num_reset, env._num_action),
            "last_actions": _zero_actions(num_reset, env._num_action),
        }
        return ResetPlan(
            env_ids=env_ids,
            qpos=qpos,
            qvel=qvel,
            info_updates=info_updates,
            randomization=_build_common_reset_randomization(env, num_reset),
        )

    def build_reset_observation(
        self, env: Any, env_ids: np.ndarray, info_updates: dict[str, Any]
    ) -> dict[str, np.ndarray]:
        linvel = env.get_local_linvel()[env_ids]
        gyro = env.get_gyro()[env_ids]
        gravity = env._backend.get_sensor_data("upvector")[env_ids]
        dof_pos = env.get_dof_pos()[env_ids]
        dof_vel = env.get_dof_vel()[env_ids]
        return cast(
            dict[str, np.ndarray],
            env._compute_obs(
                info_updates, linvel, gyro, gravity, dof_pos, dof_vel, env.feet_phase[env_ids]
            ),
        )


class Go2JoystickProvider(_BaseLocomotionProvider):
    def _sample_base_velocity(self, env: Any, num_reset: int) -> np.ndarray:
        return np.random.uniform(-0.5, 0.5, (num_reset, 6))

    def build_reset_plan(self, env: Any, env_ids: np.ndarray) -> ResetPlan:
        num_reset = len(env_ids)
        qpos, qvel = self._build_qpos_qvel(env, num_reset)
        info_updates = {
            "commands": self._sample_commands(env, num_reset),
            "current_actions": _zero_actions(num_reset, env._num_action),
            "last_actions": _zero_actions(num_reset, env._num_action),
        }
        return ResetPlan(
            env_ids=env_ids,
            qpos=qpos,
            qvel=qvel,
            info_updates=info_updates,
            randomization=_build_common_reset_randomization(env, num_reset),
        )

    def build_reset_observation(
        self, env: Any, env_ids: np.ndarray, info_updates: dict[str, Any]
    ) -> dict[str, np.ndarray]:
        linvel = env.get_local_linvel()[env_ids]
        gyro = env.get_gyro()[env_ids]
        gravity = env._backend.get_sensor_data("upvector")[env_ids]
        dof_pos = env.get_dof_pos()[env_ids]
        dof_vel = env.get_dof_vel()[env_ids]
        return cast(
            dict[str, np.ndarray],
            env._compute_obs(info_updates, linvel, gyro, gravity, dof_pos, dof_vel),
        )


class G1JoystickProvider(_BaseLocomotionProvider):
    def _sample_base_velocity(self, env: Any, num_reset: int) -> np.ndarray:
        limit = float(env.cfg.reset_base_qvel_limit)
        return np.asarray(
            np.random.uniform(-limit, limit, size=(num_reset, 6)), dtype=get_global_dtype()
        )

    def _sample_gait_phase(self, env: Any, num_reset: int) -> np.ndarray:
        mode = env.cfg.gait_phase_init_mode
        if mode == "independent":
            left = np.random.uniform(0.0, 2.0 * np.pi, size=(num_reset,))
            right = np.random.uniform(0.0, 2.0 * np.pi, size=(num_reset,))
            return np.asarray(np.column_stack([left, right]), dtype=get_global_dtype())

        phase = np.random.uniform(0.0, 2.0 * np.pi, size=(num_reset,))
        return np.asarray(np.column_stack([phase, phase + np.pi]), dtype=get_global_dtype())

    def build_reset_plan(self, env: Any, env_ids: np.ndarray) -> ResetPlan:
        num_reset = len(env_ids)
        qpos, qvel = self._build_qpos_qvel(env, num_reset)
        info_updates = {
            "commands": self._sample_commands(env, num_reset),
            "current_actions": _zero_actions(num_reset, env._num_action),
            "last_actions": _zero_actions(num_reset, env._num_action),
            "gait_phase": self._sample_gait_phase(env, num_reset),
        }
        return ResetPlan(
            env_ids=env_ids,
            qpos=qpos,
            qvel=qvel,
            info_updates=info_updates,
            randomization=_build_common_reset_randomization(env, num_reset),
        )

    def build_reset_observation(
        self, env: Any, env_ids: np.ndarray, info_updates: dict[str, Any]
    ) -> dict[str, np.ndarray]:
        linvel = env.get_local_linvel()[env_ids]
        gyro = env.get_gyro()[env_ids]
        gravity = env._backend.get_sensor_data("upvector")[env_ids]
        dof_pos = env.get_dof_pos()[env_ids]
        dof_vel = env.get_dof_vel()[env_ids]
        return cast(
            dict[str, np.ndarray],
            env._compute_obs(info_updates, linvel, gyro, gravity, dof_pos, dof_vel),
        )


class G1MotionTrackingProvider(DomainRandomizationProvider):
    def validate(self, env: Any, capabilities: DomainRandomizationCapabilities) -> None:
        _validate_common_reset_randomization(env, capabilities)
        _validate_interval_push_support(env, capabilities)

    def build_interval_randomization_plan(
        self, env: Any, step_counter: int
    ) -> IntervalRandomizationPlan | None:
        return _build_push_plan(env, step_counter)

    def build_reset_plan(self, env: Any, env_ids: np.ndarray) -> ResetPlan:
        dtype = get_global_dtype()
        num_reset = len(env_ids)

        motion_frames = env.motion_sampler.sample_frames(env_ids)
        motion_data = env.motion_loader.get_motion_at_frame(motion_frames)

        root_pos = motion_data.body_pos_w[:, 0].copy()
        root_ori = motion_data.body_quat_w[:, 0].copy()
        root_lin_vel = motion_data.body_lin_vel_w[:, 0].copy()
        root_ang_vel = motion_data.body_ang_vel_w[:, 0].copy()
        joint_pos = motion_data.joint_pos.copy()
        joint_vel = motion_data.joint_vel.copy()

        pose_rand = env.cfg.pose_randomization
        pose_ranges = [
            (pose_rand.x[0], pose_rand.x[1]),
            (pose_rand.y[0], pose_rand.y[1]),
            (pose_rand.z[0], pose_rand.z[1]),
            (pose_rand.roll[0], pose_rand.roll[1]),
            (pose_rand.pitch[0], pose_rand.pitch[1]),
            (pose_rand.yaw[0], pose_rand.yaw[1]),
        ]
        pose_samples = np.array(
            [[np.random.uniform(low, high) for low, high in pose_ranges] for _ in range(num_reset)],
            dtype=dtype,
        )
        root_pos += pose_samples[:, 0:3]
        root_ori = np_quat_mul(
            np_quat_from_euler_xyz(pose_samples[:, 3], pose_samples[:, 4], pose_samples[:, 5]),
            root_ori,
        )

        vel_rand = env.cfg.velocity_randomization
        vel_ranges = [
            (vel_rand.x[0], vel_rand.x[1]),
            (vel_rand.y[0], vel_rand.y[1]),
            (vel_rand.z[0], vel_rand.z[1]),
            (vel_rand.roll[0], vel_rand.roll[1]),
            (vel_rand.pitch[0], vel_rand.pitch[1]),
            (vel_rand.yaw[0], vel_rand.yaw[1]),
        ]
        vel_samples = np.array(
            [[np.random.uniform(low, high) for low, high in vel_ranges] for _ in range(num_reset)],
            dtype=dtype,
        )
        root_lin_vel += vel_samples[:, :3]
        root_ang_vel += vel_samples[:, 3:]

        joint_pos += np_sample_uniform(
            env.cfg.joint_position_range[0],
            env.cfg.joint_position_range[1],
            joint_pos.shape,
            dtype=np.float32,
        )
        joint_range = env._get_joint_range()
        if joint_range is not None:
            joint_pos = np.clip(joint_pos, joint_range[:, 0], joint_range[:, 1])

        qpos = np.tile(env._init_qpos, (num_reset, 1))
        qvel = np.tile(env._init_qvel, (num_reset, 1))
        qpos[:, 0:3] = root_pos
        qpos[:, 3:7] = root_ori
        qpos[:, 7:] = joint_pos

        qvel[:, 0:3] = root_lin_vel
        qvel[:, 3:6] = np_quat_apply(np_quat_inv(root_ori), root_ang_vel)
        qvel[:, 6:] = joint_vel

        info_updates = {
            "current_actions": _zero_actions(num_reset, env._num_action),
            "last_actions": _zero_actions(num_reset, env._num_action),
        }
        return ResetPlan(
            env_ids=env_ids,
            qpos=qpos,
            qvel=qvel,
            info_updates=info_updates,
            randomization=_build_common_reset_randomization(env, num_reset),
        )

    def build_reset_observation(
        self, env: Any, env_ids: np.ndarray, info_updates: dict[str, Any]
    ) -> dict[str, np.ndarray]:
        motion_data = env.motion_loader.get_motion_at_frame(
            env.motion_sampler.current_frames[env_ids]
        )
        linvel = env.get_local_linvel()[env_ids]
        gyro = env.get_gyro()[env_ids]
        dof_pos = env.get_dof_pos()[env_ids]
        dof_vel = env.get_dof_vel()[env_ids]
        all_pos_w, all_quat_w = env._get_body_pose_w()
        return cast(
            dict[str, np.ndarray],
            env._compute_obs(
                info_updates,
                motion_data,
                linvel,
                gyro,
                dof_pos,
                dof_vel,
                all_pos_w[env_ids],
                all_quat_w[env_ids],
            ),
        )


class _BaseAllegroInhandProvider(DomainRandomizationProvider):
    def validate(self, env: Any, capabilities: DomainRandomizationCapabilities) -> None:
        _validate_common_reset_randomization(env, capabilities)

    @abc.abstractmethod
    def _load_grasp_cache(self, env: Any) -> np.ndarray | None:
        pass

    def _sample_hand_and_ball_state(
        self, env: Any, num_reset: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        grasp_cache = self._load_grasp_cache(env)
        dr = env.cfg.domain_rand
        if grasp_cache is not None:
            idx = np.random.randint(0, len(grasp_cache), size=num_reset)
            sampled = grasp_cache[idx]
            return sampled[:, :16], sampled[:, 16:19], sampled[:, 19:23]

        hand_qpos = np.broadcast_to(env.default_angles, (num_reset, env._NUM_HAND_DOF)).copy()
        hand_qpos += np.random.uniform(-dr.joint_noise, dr.joint_noise, hand_qpos.shape).astype(
            np.float64
        )
        hand_qpos = np.clip(
            hand_qpos,
            env._ctrl_lower.astype(np.float64),
            env._ctrl_upper.astype(np.float64),
        )
        ball_init_pos = env._init_qpos[env._NUM_HAND_DOF : env._NUM_HAND_DOF + 3]
        ball_pos = np.broadcast_to(ball_init_pos, (num_reset, 3)).copy()
        ball_pos[:, 2] += dr.ball_z_offset
        ball_quat = np.tile([1.0, 0.0, 0.0, 0.0], (num_reset, 1))
        return hand_qpos, ball_pos, ball_quat

    def build_reset_plan(self, env: Any, env_ids: np.ndarray) -> ResetPlan:
        num_reset = len(env_ids)
        hand_qpos, ball_pos, ball_quat = self._sample_hand_and_ball_state(env, num_reset)
        qpos = np.concatenate([hand_qpos, ball_pos, ball_quat], axis=1).astype(np.float64)
        qvel = np.zeros((num_reset, env.nv), dtype=np.float64)
        qvel[:, env._NUM_HAND_DOF : env._NUM_HAND_DOF + 3] = np.random.uniform(
            -env.cfg.domain_rand.ball_vel_noise,
            env.cfg.domain_rand.ball_vel_noise,
            (num_reset, 3),
        )

        dtype = get_global_dtype()
        init_ctrl = hand_qpos.astype(dtype)
        ball_pos_f32 = ball_pos.astype(dtype)
        info_updates = self._build_info_updates(env, init_ctrl, ball_pos_f32, ball_quat)
        return ResetPlan(
            env_ids=env_ids,
            qpos=qpos,
            qvel=qvel,
            info_updates=info_updates,
            randomization=None,
        )

    @abc.abstractmethod
    def _build_info_updates(
        self, env: Any, init_ctrl: np.ndarray, ball_pos: np.ndarray, ball_quat: np.ndarray
    ) -> dict[str, Any]:
        pass


class AllegroRotationProvider(_BaseAllegroInhandProvider):
    def _load_grasp_cache(self, env: Any) -> np.ndarray | None:
        if env._grasp_cache_loaded:
            return cast(np.ndarray | None, env._grasp_cache)
        if env.cfg.gen_grasp:
            env._grasp_cache = None
            env._grasp_cache_loaded = True
            return None
        cache_path = env.cfg.grasp_cache_path or str(
            epath.Path(__file__).parents[1]
            / "envs"
            / "manipulation"
            / "inhand_rot_allegro"
            / "grasps"
            / "grasp_50k.npy"
        )
        if not epath.Path(cache_path).exists():
            raise FileNotFoundError(f"Grasp cache not found: {cache_path}")
        env._grasp_cache = np.load(cache_path).astype(np.float64)
        env._grasp_cache_loaded = True
        return cast(np.ndarray | None, env._grasp_cache)

    def _build_info_updates(
        self, env: Any, init_ctrl: np.ndarray, ball_pos: np.ndarray, ball_quat: np.ndarray
    ) -> dict[str, Any]:
        dtype = get_global_dtype()
        dof_pos_norm = 2.0 * (init_ctrl - env._dof_mid) / (env._dof_range + 1e-8)
        init_obs = np.concatenate([dof_pos_norm, init_ctrl, ball_pos], axis=1, dtype=dtype)
        obs_lag_history = np.broadcast_to(
            init_obs[:, None, :],
            (init_ctrl.shape[0], env._NUM_LAG_STEPS, env._NUM_OBS_PER_STEP),
        ).copy()
        return {
            "current_actions": np.zeros((init_ctrl.shape[0], env._num_action), dtype=dtype),
            "last_actions": np.zeros((init_ctrl.shape[0], env._num_action), dtype=dtype),
            "prev_ctrl": init_ctrl,
            "init_pose": init_ctrl.copy(),
            "prev_dof_pos": init_ctrl.copy(),
            "prev_ball_pos": ball_pos.copy(),
            "prev_ball_quat": ball_quat.astype(dtype).copy(),
            "obs_lag_history": obs_lag_history,
        }

    def build_reset_observation(
        self, env: Any, env_ids: np.ndarray, info_updates: dict[str, Any]
    ) -> dict[str, np.ndarray]:
        return cast(
            dict[str, np.ndarray],
            env._compute_obs(
                info_updates,
                info_updates["prev_ctrl"],
                info_updates["prev_ball_pos"],
            ),
        )


class AllegroRotationSacProvider(_BaseAllegroInhandProvider):
    def _load_grasp_cache(self, env: Any) -> np.ndarray | None:
        if env._grasp_cache_loaded:
            return cast(np.ndarray | None, env._grasp_cache)
        if env.cfg.gen_grasp:
            env._grasp_cache = None
            env._grasp_cache_loaded = True
            return None
        cache_path = env.cfg.grasp_cache_path or str(
            epath.Path(__file__).parents[1]
            / "envs"
            / "manipulation"
            / "inhand_rot_allegro"
            / "grasps"
            / "grasp_50k.npy"
        )
        if not epath.Path(cache_path).exists():
            raise FileNotFoundError(f"Grasp cache not found: {cache_path}")
        env._grasp_cache = np.load(cache_path).astype(np.float64)
        env._grasp_cache_loaded = True
        return cast(np.ndarray | None, env._grasp_cache)

    def _build_info_updates(
        self, env: Any, init_ctrl: np.ndarray, ball_pos: np.ndarray, ball_quat: np.ndarray
    ) -> dict[str, Any]:
        dof_pos_norm = 2.0 * (init_ctrl - env._dof_mid) / (env._dof_range + 1e-8)
        init_obs = np.concatenate([dof_pos_norm, init_ctrl, ball_pos], axis=1)
        obs_lag_history = np.broadcast_to(
            init_obs[:, None, :],
            (init_ctrl.shape[0], env._NUM_LAG_STEPS, env._NUM_OBS_PER_STEP),
        ).copy()
        return {
            "current_actions": np.zeros((init_ctrl.shape[0], env._num_action), dtype=env._np_dtype),
            "last_actions": np.zeros((init_ctrl.shape[0], env._num_action), dtype=env._np_dtype),
            "prev_ctrl": init_ctrl,
            "init_pose": init_ctrl.copy(),
            "prev_dof_pos": init_ctrl.copy(),
            "prev_ball_pos": ball_pos.copy(),
            "prev_ball_quat": ball_quat.astype(env._np_dtype).copy(),
            "obs_lag_history": obs_lag_history,
        }

    def build_reset_observation(
        self, env: Any, env_ids: np.ndarray, info_updates: dict[str, Any]
    ) -> dict[str, np.ndarray]:
        return cast(dict[str, np.ndarray], env._get_obs(info_updates))


_PROVIDERS: dict[str, type[DomainRandomizationProvider]] = {
    "go1_joystick": Go1JoystickProvider,
    "go2_joystick": Go2JoystickProvider,
    "g1_joystick": G1JoystickProvider,
    "g1_motion_tracking": G1MotionTrackingProvider,
    "allegro_rotation": AllegroRotationProvider,
    "allegro_rotation_sac": AllegroRotationSacProvider,
}


def build_provider(provider_key: str) -> DomainRandomizationProvider:
    try:
        provider_cls = _PROVIDERS[provider_key]
    except KeyError as exc:  # pragma: no cover - configuration error
        raise ValueError(f"Unknown domain-randomization provider: {provider_key}") from exc
    return provider_cls()
