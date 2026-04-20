from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, cast

import numpy as np

from unilab.base import registry
from unilab.base.backend import create_backend
from unilab.base.dtype_config import get_global_dtype
from unilab.base.np_env import NpEnvState
from unilab.dr import (
    DomainRandomizationCapabilities,
    DomainRandomizationProvider,
    GeomSizeOverride,
    InitRandomizationPlan,
    ModelVariantSpec,
    ResetPlan,
)
from unilab.dr.dr_utils import build_common_reset_randomization, validate_common_reset_randomization
from unilab.envs.manipulation.sharpa_inhand.base import (
    SharpaInhandBaseCfg,
    SharpaInhandBaseEnv,
    apply_random_rotation_to_positions,
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
    zero_action_test_mode: bool = False
    # "separated": source-style actor obs and privileged info carried separately.
    # "flattened": simple actor obs with privileged info appended into "obs".
    observation_mode: str = "separated"


class SharpaInhandRotationDRProvider(DomainRandomizationProvider):
    def validate(self, env: Any, capabilities: DomainRandomizationCapabilities) -> None:
        unsupported = validate_common_reset_randomization(env, capabilities)
        if unsupported:
            names = ", ".join(sorted(unsupported))
            raise NotImplementedError(
                f"{env._backend.backend_type} backend does not support reset randomization terms: {names}"
            )

    def build_init_randomization_plan(self, env: Any) -> InitRandomizationPlan | None:
        if env._backend.backend_type != "mujoco":
            return None
        base_size = getattr(env, "_object_geom_base_size", None)

        if base_size is None:
            return None

        model_variants = tuple(
            ModelVariantSpec(
                geom_size_overrides=(
                    GeomSizeOverride(
                        geom_name=env.cfg.object_geom_name,
                        size=tuple(np.asarray(base_size * scale, dtype=np.float64)),
                    ),
                )
            )
            for scale in np.asarray(env.scale_values, dtype=np.float64)
        )
        return InitRandomizationPlan(
            model_assignments=np.asarray(env.scale_ids, dtype=np.int32).copy(),
            model_variants=model_variants,
        )

    def _load_grasp_cache(self, env: Any) -> np.ndarray:
        if getattr(env, "_grasp_cache", None) is not None:
            return cast(np.ndarray, env._grasp_cache)

        cache_file = resolve_grasp_cache_file(env.cfg.grasp_cache_path, env.cfg.scale_range)
        if not cache_file.exists():
            raise RuntimeError(f"No saved grasping states found at {cache_file}")

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
        env_ids: np.ndarray,
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

        critic_info = env._build_reset_critic_info(num_reset, env_ids).astype(dtype)

        tactile = np.zeros((num_reset, env._num_tactile), dtype=dtype)
        contact_pos = np.zeros((num_reset, env._num_tactile * 3), dtype=dtype)
        hand_qpos_f = hand_qpos.astype(dtype)
        targets = hand_qpos_f.copy()
        object_pos_f = object_pos.astype(dtype)
        init_frame = env._build_policy_frame(
            dof_pos=hand_qpos_f,
            targets=targets,
            tactile=tactile,
            contact_pos=contact_pos,
        )
        obs_lag_history = repeat_obs_history(init_frame, env.cfg.obs_history_len).astype(dtype)

        object_default_pose = np.concatenate(
            [object_pos_f, object_quat.astype(dtype)], axis=1
        ).astype(dtype)
        critic_info = env._fill_critic_info(
            critic_info=critic_info,
            object_pos=object_pos_f,
            object_default_pose=object_default_pose,
        )

        info_updates = {
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
            "critic_info": critic_info,
            "obs_lag_history": obs_lag_history,
            "proprio_hist": env._update_proprio_history(obs_lag_history),
        }
        return info_updates

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
            env_ids=env_ids,
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
        tactile, contact_pos = env._policy_frame_zeros(len(info_updates["prev_targets"]))
        return cast(
            dict[str, np.ndarray],
            env._compute_obs_from_inputs(
                info_updates,
                dof_pos=np.asarray(info_updates["prev_targets"]),
                object_pos=np.asarray(info_updates["prev_object_pos"]),
                tactile=tactile,
                contact_pos=contact_pos,
            ),
        )


@registry.env("SharpaInhandRotation", sim_backend="mujoco")
@registry.env("SharpaInhandRotation", sim_backend="motrix")
class SharpaInhandRotationEnv(SharpaInhandBaseEnv):
    _cfg: SharpaInhandRotationCfg
    _reward_cfg: RewardConfig
    _OBS_MODE_ALIASES: dict[str, str] = {
        "separated": "separated",
        "flattened": "flattened",
    }
    _CRITIC_BASE_DIM = 8

    def __init__(
        self,
        cfg: SharpaInhandRotationCfg,
        num_envs: int = 1,
        backend_type: str = "motrix",
        dr_provider: DomainRandomizationProvider | None = None,
    ) -> None:
        if cfg.reward_config is None:
            raise ValueError("reward_config must be provided via Hydra configuration")

        backend = create_backend(
            backend_type,
            cfg.model_file,
            num_envs,
            cfg.sim_dt,
            base_name=cfg.base_name,
            add_body_sensors=True,
            iterations=cfg.iterations,
        )
        super().__init__(cfg, backend, num_envs)

        self._observation_mode = self._resolve_observation_mode(cfg.observation_mode)
        expected_critic_info_dim = self._expected_critic_info_dim()
        if cfg.critic_info_dim != expected_critic_info_dim:
            raise ValueError(
                "critic_info_dim must be "
                f"{expected_critic_info_dim} for current task layout, got {cfg.critic_info_dim}"
            )
        policy_frame_dim = self._policy_frame_dim()
        self.obs_buf_lag_history = np.zeros(
            (num_envs, cfg.obs_history_len, policy_frame_dim), dtype=self._np_dtype
        )
        self.proprio_hist_buf = np.zeros(
            (num_envs, cfg.prop_hist_len, policy_frame_dim), dtype=self._np_dtype
        )
        self.critic_info_buf = np.zeros((num_envs, expected_critic_info_dim), dtype=self._np_dtype)

        if cfg.torque_control:
            raise NotImplementedError(
                "Sharpa torque_control=True is not implemented with the current position-actuator XML setup. "
                "Set env.torque_control=false. Virtual torques are still computed explicitly for reward terms."
            )

        self._reward_cfg = cfg.reward_config
        self._zero_action_test_mode = bool(cfg.zero_action_test_mode)
        self._enable_reward_log = True
        self._grasp_cache: np.ndarray | None = None

        axis = np.asarray(cfg.rot_axis, dtype=self._np_dtype)
        axis_norm = np.linalg.norm(axis)
        if axis_norm <= 1.0e-8:
            raise ValueError("rot_axis must be non-zero")
        self._rot_axis = np.asarray(axis / axis_norm, dtype=self._np_dtype)

        provider = dr_provider if dr_provider is not None else SharpaInhandRotationDRProvider()
        self._init_domain_randomization(provider)

    def apply_action(self, actions: np.ndarray, state: NpEnvState) -> np.ndarray:
        actions_np = np.asarray(actions, dtype=self._np_dtype)
        if self._zero_action_test_mode:
            actions_np = np.zeros_like(actions_np, dtype=self._np_dtype)
        return super().apply_action(actions_np, state)

    def _scale_randomization_enabled(self) -> bool:
        lower = float(self._cfg.scale_range[0])
        upper = float(self._cfg.scale_range[1])
        num_scales = int(self._cfg.scale_range[2])
        return num_scales > 1 or not np.isclose(lower, upper)

    def _expected_critic_info_dim(self) -> int:
        return self._CRITIC_BASE_DIM + int(self._scale_randomization_enabled())

    def _critic_info_layout(self) -> dict[str, slice]:
        """Describe the flat critic_info channel layout.

        Args:
            None.

        Returns:
            Mapping from logical field names to channel slices.
        """
        layout = {
            "object_pos_delta": slice(0, 3),
            "friction": slice(3, 4),
            "mass": slice(4, 5),
            "com": slice(5, 8),
        }
        if self._scale_randomization_enabled():
            layout["scale"] = slice(self._CRITIC_BASE_DIM, self._CRITIC_BASE_DIM + 1)
        return layout

    def _assign_critic_info_field(
        self,
        critic_info: np.ndarray,
        field_name: str,
        values: np.ndarray,
    ) -> None:
        """Assign one critic_info field according to the declared layout.

        Args:
            critic_info: Critic-info buffer to update in place.
            field_name: Logical field name from the layout.
            values: Batch-major values for the field.

        Returns:
            None. The critic_info array is updated in place.
        """
        field_slice = self._critic_info_layout().get(field_name)
        if field_slice is None:
            return
        field_values = np.asarray(values, dtype=self._np_dtype).reshape(critic_info.shape[0], -1)
        critic_info[:, field_slice] = field_values

    def _build_reset_critic_info(self, batch_size: int, env_ids: np.ndarray) -> np.ndarray:
        """Build reset-time critic_info for all randomized object properties.

        Args:
            batch_size: Number of reset environments.
            env_ids: Global environment ids for this reset batch.

        Returns:
            Critic-info tensor with shape (batch_size, critic_info_dim).
        """
        critic_info = np.zeros((batch_size, self._cfg.critic_info_dim), dtype=self._np_dtype)

        if self._cfg.randomize_friction:
            self._assign_critic_info_field(
                critic_info,
                "friction",
                np.random.uniform(
                    self._cfg.randomize_friction_scale_lower,
                    self._cfg.randomize_friction_scale_upper,
                    size=(batch_size, 1),
                ),
            )
        if self._cfg.randomize_mass:
            self._assign_critic_info_field(
                critic_info,
                "mass",
                np.random.uniform(
                    self._cfg.randomize_mass_lower,
                    self._cfg.randomize_mass_upper,
                    size=(batch_size, 1),
                ),
            )
        if self._cfg.randomize_com:
            self._assign_critic_info_field(
                critic_info,
                "com",
                np.random.uniform(
                    self._cfg.randomize_com_lower,
                    self._cfg.randomize_com_upper,
                    size=(batch_size, 3),
                ),
            )
        if self._scale_randomization_enabled():
            self._assign_critic_info_field(
                critic_info,
                "scale",
                self.scale_values[self.scale_ids[env_ids]].reshape(batch_size, 1),
            )
        return critic_info

    def _policy_frame_dim(self) -> int:
        dim = self._num_action + self._num_action
        if self._cfg.enable_tactile:
            dim += self._num_tactile
        if self._cfg.enable_contact_pos:
            dim += self._num_tactile * 3
        return dim

    def _policy_frame_zeros(self, batch_size: int) -> tuple[np.ndarray, np.ndarray]:
        """Build zero-filled optional policy inputs for reset observations.

        Args:
            batch_size: Number of environments in the batch.

        Returns:
            Tuple of tactile and contact-position arrays sized for the current config.
        """
        tactile_dim = self._num_tactile if self._cfg.enable_tactile else 0
        contact_pos_dim = self._num_tactile * 3 if self._cfg.enable_contact_pos else 0
        return (
            np.zeros((batch_size, tactile_dim), dtype=self._np_dtype),
            np.zeros((batch_size, contact_pos_dim), dtype=self._np_dtype),
        )

    def _policy_frame_parts(
        self,
        dof_norm: np.ndarray,
        targets: np.ndarray,
        tactile: np.ndarray,
        contact_pos: np.ndarray,
    ) -> list[np.ndarray]:
        """Collect policy-frame components according to the configured observation layout.

        Args:
            dof_norm: Normalized hand joint positions.
            targets: Hand joint targets.
            tactile: Tactile features.
            contact_pos: Contact-position features.

        Returns:
            Ordered list of arrays that should be concatenated into the policy frame.
        """
        parts = [dof_norm, targets]
        if self._cfg.enable_tactile:
            parts.append(np.asarray(tactile, dtype=self._np_dtype))
        if self._cfg.enable_contact_pos:
            parts.append(np.asarray(contact_pos, dtype=self._np_dtype))
        return parts

    def _fill_critic_info(
        self,
        critic_info: np.ndarray,
        object_pos: np.ndarray,
        object_default_pose: np.ndarray,
    ) -> np.ndarray:
        """Populate privileged channels that are derived from runtime state.

        Args:
            critic_info: Critic info buffer to update in place.
            object_pos: Current object positions with shape (batch, 3).
            object_default_pose: Reset-time object pose cache with shape (batch, 7).

        Returns:
            Updated critic info array with shape (batch, critic_info_dim).
        """
        self._assign_critic_info_field(
            critic_info,
            "object_pos_delta",
            object_pos - object_default_pose[:, 0:3],
        )
        return critic_info

    @classmethod
    def _resolve_observation_mode(cls, observation_mode: str) -> str:
        normalized_mode = str(observation_mode).strip().lower()
        resolved_mode = cls._OBS_MODE_ALIASES.get(normalized_mode)
        if resolved_mode is None:
            supported_modes = "', '".join(sorted(cls._OBS_MODE_ALIASES))
            raise ValueError(
                f"observation_mode must be one of '{supported_modes}', got {observation_mode!r}"
            )
        return resolved_mode

    def _build_policy_frame(
        self,
        dof_pos: np.ndarray,
        targets: np.ndarray,
        tactile: np.ndarray,
        contact_pos: np.ndarray,
    ) -> np.ndarray:
        dof_pos_f = np.asarray(dof_pos, dtype=self._np_dtype)
        targets_f = np.asarray(targets, dtype=self._np_dtype)

        dof_norm = self._normalize_joint_pos(dof_pos_f)
        if self._cfg.joint_noise_scale > 0.0:
            dof_norm += (
                np.random.uniform(-1.0, 1.0, size=dof_norm.shape).astype(self._np_dtype)
                * self._cfg.joint_noise_scale
            )

        return np.asarray(
            np.concatenate(
                self._policy_frame_parts(
                    dof_norm=dof_norm,
                    targets=targets_f,
                    tactile=tactile,
                    contact_pos=contact_pos,
                ),
                axis=1,
            ),
            dtype=self._np_dtype,
        )

    @property
    def obs_groups_spec(self) -> dict[str, int]:
        policy_obs_dim = self._cfg.obs_lag_steps * self._policy_frame_dim()
        if self._observation_mode == "flattened":
            return {"obs": policy_obs_dim + self._cfg.critic_info_dim}
        return {"obs": policy_obs_dim, "critic": policy_obs_dim + self._cfg.critic_info_dim}

    def _build_critic_info(
        self,
        info: dict[str, Any],
        batch_size: int,
        object_pos: np.ndarray,
    ) -> np.ndarray:
        """Build privileged critic info for the current batch.

        Args:
            info: Mutable state info dictionary carrying reset-time caches.
            batch_size: Number of environments in the current batch.
            object_pos: Current object positions with shape (batch, 3).

        Returns:
            Privileged info array with shape (batch, critic_info_dim).
        """
        critic_info = np.asarray(
            info.get(
                "critic_info",
                np.zeros((batch_size, self._cfg.critic_info_dim), dtype=self._np_dtype),
            ),
            dtype=self._np_dtype,
        )

        object_default_pose = np.asarray(
            info.get(
                "object_default_pose",
                np.zeros((batch_size, 7), dtype=self._np_dtype),
            ),
            dtype=self._np_dtype,
        )
        critic_info = self._fill_critic_info(
            critic_info=critic_info,
            object_pos=object_pos,
            object_default_pose=object_default_pose,
        )

        info["critic_info"] = critic_info
        return critic_info

    def _pack_observations(
        self,
        policy_obs: np.ndarray,
        critic_info: np.ndarray,
    ) -> dict[str, np.ndarray]:
        """Pack actor and privileged info into the env observation groups.

        Args:
            policy_obs: Actor observation tensor with shape (batch, actor_dim).
            critic_info: Privileged tensor with shape (batch, critic_info_dim).

        Returns:
            Observation groups that satisfy the UniLab env contract.
        """
        if self._observation_mode == "flattened":
            flattened_obs = np.concatenate([policy_obs, critic_info], axis=1).astype(self._np_dtype)
            return {"obs": flattened_obs}

        critic_obs = np.concatenate([policy_obs, critic_info], axis=1).astype(self._np_dtype)
        return {"obs": policy_obs, "critic": critic_obs}

    def _compute_obs_from_inputs(
        self,
        info: dict[str, Any],
        dof_pos: np.ndarray,
        object_pos: np.ndarray,
        tactile: np.ndarray,
        contact_pos: np.ndarray,
    ) -> dict[str, np.ndarray]:
        targets = np.asarray(info.get("prev_targets", dof_pos), dtype=self._np_dtype)
        frame = self._build_policy_frame(
            dof_pos=dof_pos,
            targets=targets,
            tactile=tactile,
            contact_pos=contact_pos,
        )
        batch_size = int(frame.shape[0])

        history = info.get("obs_lag_history")
        if history is None:
            history = repeat_obs_history(frame, self._cfg.obs_history_len).astype(self._np_dtype)
        else:
            history = np.asarray(history, dtype=self._np_dtype)
            history[:, :-1] = history[:, 1:]
            history[:, -1] = frame

        info["obs_lag_history"] = history
        info["proprio_hist"] = self._update_proprio_history(history)

        obs = np.asarray(
            history[:, -self._cfg.obs_lag_steps :].reshape(batch_size, -1),
            dtype=self._np_dtype,
        )
        critic_info = self._build_critic_info(info, batch_size=batch_size, object_pos=object_pos)
        return self._pack_observations(obs, critic_info)

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
        # Explicit virtual torque used for reward parity with source Sharpa formulation.
        virtual_torques = np.asarray(
            p_gain * (targets - dof_pos) - d_gain * dof_vel,
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
            torques=virtual_torques,
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
        state.info["hand_dof_vel"] = dof_vel.copy()
        state.info["prev_object_pos"] = object_pos.copy()
        state.info["prev_object_quat"] = object_quat.copy()
        state.info["torques"] = virtual_torques
        state.info["virtual_torques"] = virtual_torques.copy()
        state.info["object_linvel"] = object_linvel
        state.info["object_angvel"] = object_angvel

        return state.replace(
            obs=obs,
            reward=reward,
            terminated=np.asarray(terminated, dtype=bool),
        )


SharpaWaveRewardConfig = RewardConfig
SharpaWaveRotationCfg = SharpaInhandRotationCfg
