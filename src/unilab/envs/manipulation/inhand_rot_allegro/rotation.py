"""Allegro in-hand rotation environment."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, cast

import numpy as np
from etils import epath

from unilab.assets import ASSETS_ROOT_PATH
from unilab.base import registry
from unilab.base.backend import create_backend
from unilab.base.dtype_config import get_global_dtype
from unilab.base.np_env import NpEnvState
from unilab.dr import DomainRandomizationCapabilities, DomainRandomizationProvider, ResetPlan
from unilab.dr.dr_utils import validate_common_reset_randomization
from unilab.envs.manipulation.inhand_rot_allegro.base import AllegroBaseCfg, AllegroBaseMjEnv
from unilab.utils.math_utils import np_quat_conjugate, np_quat_mul, np_quat_to_axis_angle


def normalize_rotation_axis(rotation_axis: tuple[float, float, float]) -> np.ndarray:
    axis = np.asarray(rotation_axis, dtype=get_global_dtype())
    return np.asarray(axis / np.linalg.norm(axis), dtype=get_global_dtype())


def compute_ball_angvel(
    ball_quat: np.ndarray, prev_ball_quat: np.ndarray, ctrl_dt: float
) -> np.ndarray:
    rel_quat = np_quat_mul(ball_quat, np_quat_conjugate(prev_ball_quat))
    return np.asarray(np_quat_to_axis_angle(rel_quat) / ctrl_dt, dtype=get_global_dtype())


def compute_pd_torques(
    targets: np.ndarray, dof_pos: np.ndarray, dof_vel: np.ndarray, kp: float, kd: float
) -> np.ndarray:
    torques = kp * (targets - dof_pos) - kd * dof_vel
    return np.asarray(np.clip(torques, -0.5, 0.5), dtype=get_global_dtype())


def build_obs_lag_history(
    init_obs: np.ndarray, num_lag_steps: int, num_obs_per_step: int
) -> np.ndarray:
    num_envs = init_obs.shape[0]
    return np.broadcast_to(
        init_obs[:, None, :],
        (num_envs, num_lag_steps, num_obs_per_step),
    ).copy()


def sample_cached_grasps(
    grasp_cache: np.ndarray, num_reset: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    idx = np.random.randint(0, len(grasp_cache), size=num_reset)
    sampled = grasp_cache[idx]
    return sampled[:, :16], sampled[:, 16:19], sampled[:, 19:23]


@dataclass
class RewardConfigPPO:
    scales: dict[str, float]
    angvel_clip_min: float
    angvel_clip_max: float
    reset_z_threshold: float


@dataclass
class Domain_Rand:
    randomize_base_mass: bool = False
    added_mass_range: list[float] = field(default_factory=lambda: [0.0, 0.0])
    random_com: bool = False
    com_offset_x: list[float] = field(default_factory=lambda: [0.0, 0.0])
    push_robots: bool = False
    push_interval: int = 0
    max_force: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    joint_noise: float = 0.0
    ball_vel_noise: float = 0.0
    ball_z_offset: float = 0.0


@registry.envcfg("AllegroInhandRotation")
@dataclass
class AllegroRotationPPOCfg(AllegroBaseCfg):
    model_file: str = str(ASSETS_ROOT_PATH / "robots" / "allegro_hand" / "scene.xml")
    max_episode_seconds: float = 20.0
    reward_config: RewardConfigPPO | None = None
    domain_rand: Domain_Rand = field(default_factory=Domain_Rand)
    rotation_axis: tuple[float, float, float] = (0.0, 0.0, 1.0)
    grasp_cache_path: str = ""
    gen_grasp: bool = False


class AllegroRotationDomainRandomizationProvider(DomainRandomizationProvider):
    def validate(self, env: Any, capabilities: DomainRandomizationCapabilities) -> None:
        validate_common_reset_randomization(env, capabilities)

    def _load_grasp_cache(self, env: Any) -> np.ndarray | None:
        if env._grasp_cache_loaded:
            return cast(np.ndarray | None, env._grasp_cache)
        if env.cfg.gen_grasp:
            env._grasp_cache = None
            env._grasp_cache_loaded = True
            return None
        cache_path = env.cfg.grasp_cache_path or str(
            epath.Path(__file__).with_name("grasps") / "grasp_50k.npy"
        )
        if not epath.Path(cache_path).exists():
            raise FileNotFoundError(f"Grasp cache not found: {cache_path}")
        env._grasp_cache = np.load(cache_path).astype(np.float64)
        env._grasp_cache_loaded = True
        return cast(np.ndarray | None, env._grasp_cache)

    def build_reset_plan(self, env: Any, env_ids: np.ndarray) -> ResetPlan:
        num_reset = len(env_ids)
        grasp_cache = self._load_grasp_cache(env)
        dr = env.cfg.domain_rand
        if grasp_cache is not None:
            idx = np.random.randint(0, len(grasp_cache), size=num_reset)
            sampled = grasp_cache[idx]
            hand_qpos = sampled[:, :16]
            ball_pos = sampled[:, 16:19]
            ball_quat = sampled[:, 19:23]
        else:
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
        dof_pos_norm = 2.0 * (init_ctrl - env._dof_mid) / (env._dof_range + 1e-8)
        init_obs = np.concatenate([dof_pos_norm, init_ctrl, ball_pos_f32], axis=1, dtype=dtype)
        obs_lag_history = np.broadcast_to(
            init_obs[:, None, :],
            (init_ctrl.shape[0], env._NUM_LAG_STEPS, env._NUM_OBS_PER_STEP),
        ).copy()
        info_updates = {
            "current_actions": np.zeros((init_ctrl.shape[0], env._num_action), dtype=dtype),
            "last_actions": np.zeros((init_ctrl.shape[0], env._num_action), dtype=dtype),
            "prev_ctrl": init_ctrl,
            "init_pose": init_ctrl.copy(),
            "prev_dof_pos": init_ctrl.copy(),
            "prev_ball_pos": ball_pos_f32.copy(),
            "prev_ball_quat": ball_quat.astype(dtype).copy(),
            "obs_lag_history": obs_lag_history,
        }
        return ResetPlan(
            env_ids=env_ids,
            qpos=qpos,
            qvel=qvel,
            info_updates=info_updates,
            randomization=None,
        )

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


# ─────────────────────────── Environment ──────────────────────────────


@registry.env("AllegroInhandRotation", sim_backend="mujoco")
@registry.env("AllegroInhandRotation", sim_backend="motrix")
class AllegroRotationMj(AllegroBaseMjEnv):
    _cfg: AllegroRotationPPOCfg
    _reward_cfg: RewardConfigPPO

    _NUM_OBS_PER_STEP = 35
    _NUM_LAG_STEPS = 3

    def __init__(
        self, cfg: AllegroRotationPPOCfg, num_envs: int = 1, backend_type: str = "mujoco"
    ) -> None:
        if cfg.reward_config is None:
            raise ValueError("reward_config must be provided via Hydra configuration")
        backend = create_backend(
            backend_type,
            cfg.model_file,
            num_envs,
            cfg.sim_dt,
            base_name="palm",
            add_body_sensors=True,
            position_actuator_gains={
                "kp": cfg.control_config.kp,
                "kd": cfg.control_config.kd,
                "actuator_ids": slice(0, 16),
            },
        )
        super().__init__(cfg, backend, num_envs)
        self._enable_reward_log = True
        self._reward_cfg = cfg.reward_config

        self._dof_range = self._ctrl_upper - self._ctrl_lower
        self._dof_mid = (self._ctrl_upper + self._ctrl_lower) / 2.0
        self._rot_axis = normalize_rotation_axis(cfg.rotation_axis)
        self._grasp_cache: np.ndarray | None = None
        self._grasp_cache_loaded = False

        self._init_reward_functions()
        self._init_domain_randomization(AllegroRotationDomainRandomizationProvider())

    @property
    def obs_groups_spec(self) -> dict[str, int]:
        return {"obs": self._NUM_OBS_PER_STEP * self._NUM_LAG_STEPS}

    def _init_reward_functions(self) -> None:
        self._reward_fns = {
            "rotate": self._reward_rotate,
            "obj_linvel": self._reward_obj_linvel,
            "pose_diff": self._reward_pose_diff,
            "torque": self._reward_torque,
            "work": self._reward_work,
            "drop": self._reward_drop,
        }

    def _reward_rotate(
        self,
        info: dict[str, Any],
        dof_pos: np.ndarray,
        dof_vel: np.ndarray,
        ball_pos: np.ndarray,
        ball_linvel: np.ndarray,
        ball_angvel: np.ndarray,
        torques: np.ndarray,
        terminated: np.ndarray,
    ) -> np.ndarray:
        del info, dof_pos, dof_vel, ball_pos, ball_linvel, torques, terminated
        vec_dot = ball_angvel @ self._rot_axis
        return np.asarray(
            np.clip(vec_dot, self._reward_cfg.angvel_clip_min, self._reward_cfg.angvel_clip_max)
        )

    def _reward_obj_linvel(
        self,
        info: dict[str, Any],
        dof_pos: np.ndarray,
        dof_vel: np.ndarray,
        ball_pos: np.ndarray,
        ball_linvel: np.ndarray,
        ball_angvel: np.ndarray,
        torques: np.ndarray,
        terminated: np.ndarray,
    ) -> np.ndarray:
        del info, dof_pos, dof_vel, ball_pos, ball_angvel, torques, terminated
        return np.asarray(np.sum(np.abs(ball_linvel), axis=1))

    def _reward_pose_diff(
        self,
        info: dict[str, Any],
        dof_pos: np.ndarray,
        dof_vel: np.ndarray,
        ball_pos: np.ndarray,
        ball_linvel: np.ndarray,
        ball_angvel: np.ndarray,
        torques: np.ndarray,
        terminated: np.ndarray,
    ) -> np.ndarray:
        del dof_vel, ball_pos, ball_linvel, ball_angvel, torques, terminated
        diff = dof_pos - info["init_pose"]
        return np.asarray(np.sum(np.square(diff), axis=1))

    def _reward_torque(
        self,
        info: dict[str, Any],
        dof_pos: np.ndarray,
        dof_vel: np.ndarray,
        ball_pos: np.ndarray,
        ball_linvel: np.ndarray,
        ball_angvel: np.ndarray,
        torques: np.ndarray,
        terminated: np.ndarray,
    ) -> np.ndarray:
        del info, dof_pos, dof_vel, ball_pos, ball_linvel, ball_angvel, terminated
        return np.asarray(np.sum(np.square(torques), axis=1))

    def _reward_work(
        self,
        info: dict[str, Any],
        dof_pos: np.ndarray,
        dof_vel: np.ndarray,
        ball_pos: np.ndarray,
        ball_linvel: np.ndarray,
        ball_angvel: np.ndarray,
        torques: np.ndarray,
        terminated: np.ndarray,
    ) -> np.ndarray:
        del info, dof_pos, ball_pos, ball_linvel, ball_angvel, terminated
        work = np.sum(torques * dof_vel, axis=1)
        return np.asarray(np.square(work))

    def _reward_drop(
        self,
        info: dict[str, Any],
        dof_pos: np.ndarray,
        dof_vel: np.ndarray,
        ball_pos: np.ndarray,
        ball_linvel: np.ndarray,
        ball_angvel: np.ndarray,
        torques: np.ndarray,
        terminated: np.ndarray,
    ) -> np.ndarray:
        del info, dof_pos, dof_vel, ball_pos, ball_linvel, ball_angvel, torques
        return np.asarray(terminated, dtype=get_global_dtype())

    def update_state(self, state: NpEnvState) -> NpEnvState:
        dof_pos = self.get_hand_dof_pos()
        ball_pos = self.get_ball_pos()
        ball_quat = self.get_ball_quat()

        dof_vel = (dof_pos - state.info.get("prev_dof_pos", dof_pos)) / self._cfg.ctrl_dt
        ball_linvel = (ball_pos - state.info.get("prev_ball_pos", ball_pos)) / self._cfg.ctrl_dt

        prev_ball_quat = state.info.get("prev_ball_quat", ball_quat)
        ball_angvel = compute_ball_angvel(ball_quat, prev_ball_quat, self._cfg.ctrl_dt)

        state.info["prev_dof_pos"] = dof_pos.copy()
        state.info["prev_ball_pos"] = ball_pos.copy()
        state.info["prev_ball_quat"] = ball_quat.copy()

        targets = state.info["prev_ctrl"]
        torques = compute_pd_torques(
            targets=targets,
            dof_pos=dof_pos,
            dof_vel=dof_vel,
            kp=self._cfg.control_config.kp,
            kd=self._cfg.control_config.kd,
        )
        terminated = ball_pos[:, 2] < self._reward_cfg.reset_z_threshold

        reward = self._compute_reward(
            state.info, dof_pos, dof_vel, ball_pos, ball_linvel, ball_angvel, torques, terminated
        )
        obs = self._compute_obs(state.info, dof_pos, ball_pos)
        return state.replace(obs=obs, reward=reward, terminated=terminated)

    def _compute_reward(
        self,
        info: dict[str, Any],
        dof_pos: np.ndarray,
        dof_vel: np.ndarray,
        ball_pos: np.ndarray,
        ball_linvel: np.ndarray,
        ball_angvel: np.ndarray,
        torques: np.ndarray,
        terminated: np.ndarray,
    ) -> np.ndarray:
        dtype = get_global_dtype()
        reward = np.zeros(self._num_envs, dtype=dtype)
        step_count = info.get("steps", np.zeros(self._num_envs, dtype=np.uint32))
        should_log = self._enable_reward_log and (int(step_count[0]) % 4 == 0)
        log = {} if should_log else info.get("log", {})

        for name, scale in self._reward_cfg.scales.items():
            if scale == 0 or name not in self._reward_fns:
                continue
            rew = self._reward_fns[name](
                info, dof_pos, dof_vel, ball_pos, ball_linvel, ball_angvel, torques, terminated
            )
            weighted_rew = rew * scale
            reward += weighted_rew
            if should_log:
                log[f"reward/{name}"] = float(np.mean(weighted_rew))

        if should_log:
            log["reward/total"] = float(np.mean(reward))

        info["log"] = log
        return reward * self._cfg.ctrl_dt

    def _compute_obs(
        self, info: dict[str, Any], dof_pos: np.ndarray, ball_pos: np.ndarray
    ) -> dict[str, np.ndarray]:
        dtype = get_global_dtype()
        targets = info["prev_ctrl"]
        dof_pos_norm = 2.0 * (dof_pos - self._dof_mid) / (self._dof_range + 1e-8)

        noise_cfg = self._cfg.noise_config
        if noise_cfg.level > 0.0:
            dof_pos_norm += (
                np.random.uniform(-1.0, 1.0, dof_pos_norm.shape).astype(dtype)
                * noise_cfg.level
                * noise_cfg.scale_joint_angle
            )

        current_obs = np.concatenate(
            [dof_pos_norm, targets, ball_pos.astype(dtype)], axis=1, dtype=dtype
        )

        num_envs = dof_pos.shape[0]
        obs_lag_history = info.get(
            "obs_lag_history",
            np.zeros(
                (num_envs, self._NUM_LAG_STEPS, self._NUM_OBS_PER_STEP),
                dtype=dtype,
            ),
        )
        obs_lag_history[:, :-1] = obs_lag_history[:, 1:]
        obs_lag_history[:, -1] = current_obs
        info["obs_lag_history"] = obs_lag_history

        return {
            "obs": np.asarray(obs_lag_history.reshape(num_envs, -1), dtype=dtype),
        }

    def _load_grasp_cache(self) -> None:
        if self._cfg.gen_grasp:
            self._grasp_cache = None
        else:
            default_path = epath.Path(__file__).parent / "grasps" / "grasp_50k.npy"
            cache_path = self._cfg.grasp_cache_path or str(default_path)
            if not epath.Path(cache_path).exists():
                raise FileNotFoundError(f"Grasp cache not found: {cache_path}")
            self._grasp_cache = np.load(cache_path).astype(np.float64)
        self._grasp_cache_loaded = True

    def _sample_reset_state(
        self, num_reset: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        dr = self._cfg.domain_rand
        if self._grasp_cache is not None:
            hand_qpos, ball_pos, ball_quat = sample_cached_grasps(self._grasp_cache, num_reset)
        else:
            hand_qpos = np.broadcast_to(self.default_angles, (num_reset, self._NUM_HAND_DOF)).copy()
            hand_qpos += np.random.uniform(-dr.joint_noise, dr.joint_noise, hand_qpos.shape).astype(
                np.float64
            )
            hand_qpos = np.clip(
                hand_qpos,
                self._ctrl_lower.astype(np.float64),
                self._ctrl_upper.astype(np.float64),
            )

            ball_init_pos = self._init_qpos[self._NUM_HAND_DOF : self._NUM_HAND_DOF + 3]
            ball_pos = np.broadcast_to(ball_init_pos, (num_reset, 3)).copy()
            ball_pos[:, 2] += dr.ball_z_offset
            ball_quat = np.tile([1.0, 0.0, 0.0, 0.0], (num_reset, 1))

        qvel = np.zeros((num_reset, self.nv), dtype=np.float64)
        qvel[:, self._NUM_HAND_DOF : self._NUM_HAND_DOF + 3] = np.random.uniform(
            -dr.ball_vel_noise, dr.ball_vel_noise, (num_reset, 3)
        )
        return hand_qpos, ball_pos, ball_quat, qvel

    def reset(self, env_indices: np.ndarray) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        num_reset = len(env_indices)

        if not self._grasp_cache_loaded:
            self._load_grasp_cache()

        hand_qpos, ball_pos, ball_quat, qvel = self._sample_reset_state(num_reset)
        qpos = np.concatenate([hand_qpos, ball_pos, ball_quat], axis=1).astype(np.float64)
        self._backend.set_state(env_indices, qpos, qvel)

        dtype = get_global_dtype()
        init_ctrl = hand_qpos.astype(dtype)
        ball_pos_f32 = ball_pos.astype(dtype)
        dof_pos_norm = 2.0 * (init_ctrl - self._dof_mid) / (self._dof_range + 1e-8)
        init_obs = np.concatenate([dof_pos_norm, init_ctrl, ball_pos_f32], axis=1, dtype=dtype)
        obs_lag_history = build_obs_lag_history(
            init_obs, self._NUM_LAG_STEPS, self._NUM_OBS_PER_STEP
        )

        info = {
            "current_actions": np.zeros((num_reset, self._num_action), dtype=dtype),
            "last_actions": np.zeros((num_reset, self._num_action), dtype=dtype),
            "prev_ctrl": init_ctrl,
            "init_pose": init_ctrl.copy(),
            "prev_dof_pos": init_ctrl.copy(),
            "prev_ball_pos": ball_pos_f32.copy(),
            "prev_ball_quat": ball_quat.astype(dtype).copy(),
            "obs_lag_history": obs_lag_history,
        }

        obs_batch = self._compute_obs(info, init_ctrl, ball_pos_f32)
        return obs_batch, info


RewardConfig = RewardConfigPPO
DomainRandConfig = Domain_Rand
AllegroRotationCfg = AllegroRotationPPOCfg
