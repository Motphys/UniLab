from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, cast

import numpy as np

from unilab.base import registry
from unilab.base.backend import create_backend
from unilab.base.dtype_config import get_global_dtype
from unilab.base.np_env import NpEnvState
from unilab.dr import DomainRandomizationCapabilities, DomainRandomizationProvider, ResetPlan
from unilab.dr.dr_utils import build_common_reset_randomization, validate_common_reset_randomization
from unilab.envs.manipulation.sharpa_inhand.base import (
    SharpaInhandBaseCfg,
    SharpaInhandBaseEnv,
    apply_random_rotation_to_positions,
    format_scale_tag,
    repeat_obs_history,
    resolve_grasp_cache_file,
    sample_bucketed_grasp_cache,
)
from unilab.utils.math_utils import np_quat_conjugate, np_quat_mul, np_quat_to_axis_angle


@dataclass
class RewardConfig:
    scales: dict[str, float] = field(
        default_factory=lambda: {
            "rotate": 2.5,
            "obj_linvel": -0.3,
            "pose_diff": -0.4,
            "torque": -0.1,
            "work": -0.5,
            "object_pos": 0.003,
        }
    )
    angvel_clip_min: float = -0.5
    angvel_clip_max: float = 0.5


@registry.envcfg("SharpaInhandRotation")
@dataclass
class SharpaInhandRotationCfg(SharpaInhandBaseCfg):
    reward_config: RewardConfig | None = None


class SharpaInhandRotationDRProvider(DomainRandomizationProvider):
    def validate(self, env: Any, capabilities: DomainRandomizationCapabilities) -> None:
        unsupported = validate_common_reset_randomization(env, capabilities)
        if unsupported:
            names = ", ".join(sorted(unsupported))
            raise NotImplementedError(
                f"{env._backend.backend_type} backend does not support reset randomization terms: {names}"
            )

    def _load_grasp_cache(self, env: Any) -> np.ndarray:
        if getattr(env, "_grasp_cache", None) is not None:
            return cast(np.ndarray, env._grasp_cache)

        cache_file = resolve_grasp_cache_file(env.cfg.grasp_cache_path, env.cfg.scale_range)
        if not cache_file.exists():
            hand_qpos = np.asarray(env.default_angles, dtype=np.float64)
            object_pos = np.asarray(env._init_qpos[env._obj_pos_slice], dtype=np.float64)
            object_quat = np.asarray(env._init_qpos[env._obj_quat_slice], dtype=np.float64)
            seed_pose = np.concatenate([hand_qpos, object_pos, object_quat], axis=0)
            env._grasp_cache = np.tile(seed_pose[None, :], (max(1, int(env._num_scales)), 1))
            print(
                "[sharpa_inhand] WARNING: grasp cache not found at "
                f"{cache_file} (scale tag: {format_scale_tag(env.cfg.scale_range)}); "
                "falling back to deterministic init-pose reset."
            )
            return cast(np.ndarray, env._grasp_cache)

        env._grasp_cache = np.load(cache_file).astype(np.float64)
        return cast(np.ndarray, env._grasp_cache)

    def _sample_random_quaternion(self, num_envs: int) -> np.ndarray:
        u1 = np.random.rand(num_envs)
        u2 = np.random.rand(num_envs) * 2.0 * np.pi
        u3 = np.random.rand(num_envs) * 2.0 * np.pi

        q1 = np.sqrt(1.0 - u1) * np.sin(u2)
        q2 = np.sqrt(1.0 - u1) * np.cos(u2)
        q3 = np.sqrt(u1) * np.sin(u3)
        q4 = np.sqrt(u1) * np.cos(u3)

        return np.stack([q4, q1, q2, q3], axis=1).astype(np.float64)

    def _build_info_updates(
        self,
        env: Any,
        hand_qpos: np.ndarray,
        object_pos: np.ndarray,
        object_quat: np.ndarray,
        reset_height_lower: np.ndarray,
        reset_height_upper: np.ndarray,
        rot_axis: np.ndarray,
    ) -> dict[str, np.ndarray]:
        num_reset = hand_qpos.shape[0]
        dtype = get_global_dtype()

        p_gain = np.full((num_reset, env._num_action), env.cfg.control_config.p_gain, dtype=dtype)
        d_gain = np.full((num_reset, env._num_action), env.cfg.control_config.d_gain, dtype=dtype)
        if env.cfg.randomize_pd_gains:
            p_scale = env._sample_pd_scales(
                env.cfg.randomize_p_gain_scale_lower,
                env.cfg.randomize_p_gain_scale_upper,
                shape=(num_reset, env._num_action),
            )
            d_scale = env._sample_pd_scales(
                env.cfg.randomize_d_gain_scale_lower,
                env.cfg.randomize_d_gain_scale_upper,
                shape=(num_reset, env._num_action),
            )
            p_gain *= p_scale
            d_gain *= d_scale

        priv_info = np.zeros((num_reset, env.cfg.priv_info_dim), dtype=dtype)
        if env.cfg.randomize_friction:
            priv_info[:, 3] = np.random.uniform(
                env.cfg.randomize_friction_scale_lower,
                env.cfg.randomize_friction_scale_upper,
                size=(num_reset,),
            ).astype(dtype)
        if env.cfg.randomize_mass:
            priv_info[:, 4] = np.random.uniform(
                env.cfg.randomize_mass_lower,
                env.cfg.randomize_mass_upper,
                size=(num_reset,),
            ).astype(dtype)
        if env.cfg.randomize_com:
            priv_info[:, 5:8] = np.random.uniform(
                env.cfg.randomize_com_lower,
                env.cfg.randomize_com_upper,
                size=(num_reset, 3),
            ).astype(dtype)

        # NOTE: source task randomizes friction/mass/com in physics directly.
        # UniLab backend contract currently does not expose those object-level mutation hooks,
        # so we preserve privileged channels but keep runtime physics mutation as TODO.

        tactile = np.zeros((num_reset, env._num_tactile), dtype=dtype)
        contact_pos = np.zeros((num_reset, env._num_tactile * 3), dtype=dtype)
        hand_qpos_f = hand_qpos.astype(dtype)
        dof_norm = env._normalize_joint_pos(hand_qpos_f)
        init_frame = np.concatenate([dof_norm, hand_qpos_f, tactile, contact_pos], axis=1).astype(
            dtype
        )
        obs_lag_history = repeat_obs_history(init_frame, env.cfg.obs_history_len).astype(dtype)

        object_default_pose = np.concatenate([object_pos, object_quat], axis=1).astype(dtype)
        priv_info[:, 0:3] = object_pos.astype(dtype) - object_default_pose[:, 0:3]

        return {
            "current_actions": np.zeros((num_reset, env._num_action), dtype=dtype),
            "last_actions": np.zeros((num_reset, env._num_action), dtype=dtype),
            "prev_targets": hand_qpos_f.copy(),
            "init_pose": hand_qpos_f.copy(),
            "prev_hand_pos": hand_qpos_f.copy(),
            "prev_object_pos": object_pos.astype(dtype).copy(),
            "prev_object_quat": object_quat.astype(dtype).copy(),
            "object_default_pose": object_default_pose,
            "reset_height_lower": reset_height_lower.astype(dtype),
            "reset_height_upper": reset_height_upper.astype(dtype),
            "rot_axis": rot_axis.astype(dtype),
            "p_gain": p_gain,
            "d_gain": d_gain,
            "priv_info": priv_info,
            "obs_lag_history": obs_lag_history,
            "proprio_hist": env._update_proprio_history(obs_lag_history),
        }

    def build_reset_plan(self, env: Any, env_ids: np.ndarray) -> ResetPlan:
        num_reset = len(env_ids)
        if num_reset == 0:
            return ResetPlan(
                env_ids=env_ids,
                qpos=np.zeros((0, env.nq), dtype=np.float64),
                qvel=np.zeros((0, env.nv), dtype=np.float64),
                info_updates={},
                randomization=None,
            )

        grasp_cache = self._load_grasp_cache(env)
        sampled_pose = sample_bucketed_grasp_cache(
            grasp_cache,
            env.scale_ids[env_ids],
            env._num_scales,
        )

        hand_qpos = sampled_pose[:, : env._num_action]
        object_pos = sampled_pose[:, env._num_action : env._num_action + 3]
        object_quat = sampled_pose[:, env._num_action + 3 : env._num_action + 7]

        rot_axis = np.broadcast_to(env._rot_axis, (num_reset, 3)).copy().astype(np.float64)

        if env.cfg.reset_random_quat:
            random_quat = self._sample_random_quaternion(num_reset)
            object_pos = apply_random_rotation_to_positions(
                object_pos,
                center=np.zeros((num_reset, 3), dtype=np.float64),
                random_quat=random_quat,
            )
            object_quat = env._rotate_quat(object_quat, random_quat)
            rot_axis = env._rotate_axis(rot_axis, random_quat)

        qpos = np.zeros((num_reset, env.nq), dtype=np.float64)
        qpos[:, : env._num_action] = hand_qpos
        qpos[:, env._obj_pos_slice] = object_pos
        qpos[:, env._obj_quat_slice] = object_quat

        qvel = np.zeros((num_reset, env.nv), dtype=np.float64)

        height_range = env.cfg.reset_height_upper - env.cfg.reset_height_lower
        reset_height_lower = object_pos[:, 2] - 0.5 * height_range
        reset_height_upper = object_pos[:, 2] + 0.5 * height_range

        info_updates = self._build_info_updates(
            env,
            hand_qpos=hand_qpos,
            object_pos=object_pos,
            object_quat=object_quat,
            reset_height_lower=reset_height_lower,
            reset_height_upper=reset_height_upper,
            rot_axis=rot_axis,
        )

        return ResetPlan(
            env_ids=env_ids,
            qpos=qpos,
            qvel=qvel,
            info_updates=info_updates,
            randomization=build_common_reset_randomization(env, num_reset),
        )

    def build_reset_observation(
        self,
        env: Any,
        env_ids: np.ndarray,
        info_updates: dict[str, Any],
    ) -> dict[str, np.ndarray]:
        del env_ids
        return cast(
            dict[str, np.ndarray],
            env._compute_obs_from_inputs(
                info_updates,
                dof_pos=np.asarray(info_updates["prev_targets"]),
                object_pos=np.asarray(info_updates["prev_object_pos"]),
                tactile=np.zeros(
                    (len(info_updates["prev_targets"]), env._num_tactile), dtype=env._np_dtype
                ),
                contact_pos=np.zeros(
                    (len(info_updates["prev_targets"]), env._num_tactile * 3), dtype=env._np_dtype
                ),
            ),
        )


@registry.env("SharpaInhandRotation", sim_backend="mujoco")
@registry.env("SharpaInhandRotation", sim_backend="motrix")
class SharpaInhandRotationEnv(SharpaInhandBaseEnv):
    _cfg: SharpaInhandRotationCfg
    _reward_cfg: RewardConfig

    def __init__(
        self,
        cfg: SharpaInhandRotationCfg,
        num_envs: int = 1,
        backend_type: str = "motrix",
    ) -> None:
        if cfg.reward_config is None:
            raise ValueError("reward_config must be provided via Hydra configuration")

        backend = create_backend(
            backend_type,
            cfg.model_file,
            num_envs,
            cfg.sim_dt,
            base_name=cfg.base_name,
            iterations=cfg.iterations,
        )
        super().__init__(cfg, backend, num_envs)

        expected_frame_dim = (
            self._num_action + self._num_action + self._num_tactile + self._num_tactile * 3
        )
        if cfg.frame_obs_dim != expected_frame_dim:
            raise ValueError(
                f"frame_obs_dim must be {expected_frame_dim} for current task layout, got {cfg.frame_obs_dim}"
            )

        self._reward_cfg = cfg.reward_config
        self._enable_reward_log = True
        self._grasp_cache: np.ndarray | None = None

        axis = np.asarray(cfg.rot_axis, dtype=self._np_dtype)
        axis_norm = np.linalg.norm(axis)
        if axis_norm <= 1.0e-8:
            raise ValueError("rot_axis must be non-zero")
        self._rot_axis = np.asarray(axis / axis_norm, dtype=self._np_dtype)

        self._init_domain_randomization(SharpaInhandRotationDRProvider())

    @property
    def obs_groups_spec(self) -> dict[str, int]:
        return {
            "obs": self._cfg.obs_lag_steps * self._cfg.frame_obs_dim,
            "privileged": self._cfg.priv_info_dim,
        }

    def _compute_obs_from_inputs(
        self,
        info: dict[str, Any],
        dof_pos: np.ndarray,
        object_pos: np.ndarray,
        tactile: np.ndarray,
        contact_pos: np.ndarray,
    ) -> dict[str, np.ndarray]:
        dof_norm = self._normalize_joint_pos(dof_pos)

        if self._cfg.joint_noise_scale > 0.0:
            dof_norm += (
                np.random.uniform(-1.0, 1.0, size=dof_norm.shape).astype(self._np_dtype)
                * self._cfg.joint_noise_scale
            )

        targets = np.asarray(info.get("prev_targets", dof_pos), dtype=self._np_dtype)
        frame = np.concatenate([dof_norm, targets, tactile, contact_pos], axis=1).astype(
            self._np_dtype
        )

        history = info.get("obs_lag_history")
        if history is None:
            history = repeat_obs_history(frame, self._cfg.obs_history_len).astype(self._np_dtype)
        else:
            history = np.asarray(history, dtype=self._np_dtype)
            history[:, :-1] = history[:, 1:]
            history[:, -1] = frame

        info["obs_lag_history"] = history
        info["proprio_hist"] = self._update_proprio_history(history)

        priv_info = np.asarray(
            info.get(
                "priv_info",
                np.zeros((self._num_envs, self._cfg.priv_info_dim), dtype=self._np_dtype),
            ),
            dtype=self._np_dtype,
        )

        object_default_pose = np.asarray(
            info.get(
                "object_default_pose",
                np.zeros((self._num_envs, 7), dtype=self._np_dtype),
            ),
            dtype=self._np_dtype,
        )
        if priv_info.shape[1] >= 3:
            priv_info[:, 0:3] = object_pos - object_default_pose[:, 0:3]

        info["priv_info"] = priv_info

        obs = np.asarray(
            history[:, -self._cfg.obs_lag_steps :].reshape(self._num_envs, -1),
            dtype=self._np_dtype,
        )
        return {"obs": obs, "privileged": priv_info}

    def _compute_reward(
        self,
        info: dict[str, Any],
        dof_pos: np.ndarray,
        dof_vel: np.ndarray,
        object_pos: np.ndarray,
        object_linvel: np.ndarray,
        object_angvel: np.ndarray,
        torques: np.ndarray,
    ) -> np.ndarray:
        rot_axis = np.asarray(
            info.get("rot_axis", np.broadcast_to(self._rot_axis, (self._num_envs, 3))),
            dtype=self._np_dtype,
        )
        rotate_reward = np.clip(
            np.sum(object_angvel * rot_axis, axis=1),
            self._reward_cfg.angvel_clip_min,
            self._reward_cfg.angvel_clip_max,
        )
        object_linvel_penalty = np.sum(np.abs(object_linvel), axis=1)
        pos_diff_penalty = np.sum(np.square(dof_pos - self.default_angles), axis=1)
        torque_penalty = np.sum(np.square(torques), axis=1)
        work_penalty = np.square(np.sum(torques * dof_vel, axis=1))

        object_default_pose = np.asarray(
            info.get(
                "object_default_pose",
                np.zeros((self._num_envs, 7), dtype=self._np_dtype),
            ),
            dtype=self._np_dtype,
        )
        object_pos_reward = 1.0 / (
            np.linalg.norm(object_pos - object_default_pose[:, 0:3], axis=1) + 0.001
        )

        reward_terms: dict[str, np.ndarray] = {
            "rotate": np.asarray(rotate_reward, dtype=self._np_dtype),
            "obj_linvel": np.asarray(object_linvel_penalty, dtype=self._np_dtype),
            "pose_diff": np.asarray(pos_diff_penalty, dtype=self._np_dtype),
            "torque": np.asarray(torque_penalty, dtype=self._np_dtype),
            "work": np.asarray(work_penalty, dtype=self._np_dtype),
            "object_pos": np.asarray(object_pos_reward, dtype=self._np_dtype),
        }

        reward = np.zeros((self._num_envs,), dtype=self._np_dtype)
        step_count = info.get("steps", np.zeros((self._num_envs,), dtype=np.uint32))
        should_log = self._enable_reward_log and (int(step_count[0]) % 4 == 0)
        log = {} if should_log else info.get("log", {})

        for name, scale in self._reward_cfg.scales.items():
            if scale == 0.0 or name not in reward_terms:
                continue
            weighted = reward_terms[name] * scale
            reward += weighted
            if should_log:
                log[f"reward/{name}"] = float(np.mean(weighted))

        if should_log:
            log["reward/total"] = float(np.mean(reward))
        info["log"] = log

        return np.asarray(reward, dtype=self._np_dtype)

    def update_state(self, state: NpEnvState) -> NpEnvState:
        dof_pos = self.get_hand_dof_pos()
        dof_vel = self.get_hand_dof_vel()
        object_pos = self.get_object_pos()
        object_quat = self.get_object_quat()

        prev_object_pos = np.asarray(
            state.info.get("prev_object_pos", object_pos), dtype=self._np_dtype
        )
        prev_object_quat = np.asarray(
            state.info.get("prev_object_quat", object_quat), dtype=self._np_dtype
        )

        object_linvel = (object_pos - prev_object_pos) / self._cfg.ctrl_dt
        object_angvel = (
            np_quat_to_axis_angle(np_quat_mul(object_quat, np_quat_conjugate(prev_object_quat)))
            / self._cfg.ctrl_dt
        )

        targets = np.asarray(
            state.info.get(
                "prev_targets",
                np.broadcast_to(self.default_angles, (self._num_envs, self._num_action)).copy(),
            ),
            dtype=self._np_dtype,
        )
        p_gain, d_gain = self._resolve_pd_gains(state.info)
        torques = np.asarray(
            np.clip(p_gain * (targets - dof_pos) - d_gain * dof_vel, -0.5, 0.5),
            dtype=self._np_dtype,
        )

        tactile = self._compute_tactile_observation()
        contact_pos = self._compute_contact_positions(tactile)

        reward = self._compute_reward(
            state.info,
            dof_pos=dof_pos,
            dof_vel=dof_vel,
            object_pos=object_pos,
            object_linvel=object_linvel,
            object_angvel=object_angvel,
            torques=torques,
        )

        reset_height_lower = np.asarray(
            state.info.get(
                "reset_height_lower",
                np.full((self._num_envs,), self._cfg.reset_height_lower, dtype=self._np_dtype),
            ),
            dtype=self._np_dtype,
        )
        reset_height_upper = np.asarray(
            state.info.get(
                "reset_height_upper",
                np.full((self._num_envs,), self._cfg.reset_height_upper, dtype=self._np_dtype),
            ),
            dtype=self._np_dtype,
        )
        terminated = (object_pos[:, 2] > reset_height_upper) | (
            object_pos[:, 2] < reset_height_lower
        )

        obs = self._compute_obs_from_inputs(
            state.info,
            dof_pos=dof_pos,
            object_pos=object_pos,
            tactile=tactile,
            contact_pos=contact_pos,
        )

        state.info["prev_hand_pos"] = dof_pos.copy()
        state.info["prev_object_pos"] = object_pos.copy()
        state.info["prev_object_quat"] = object_quat.copy()
        state.info["torques"] = torques
        state.info["object_linvel"] = object_linvel
        state.info["object_angvel"] = object_angvel

        return state.replace(
            obs=obs,
            reward=reward,
            terminated=np.asarray(terminated, dtype=bool),
        )


SharpaWaveRewardConfig = RewardConfig
SharpaWaveRotationCfg = SharpaInhandRotationCfg
