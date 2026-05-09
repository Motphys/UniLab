from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, cast

import numpy as np

from unilab.base import registry
from unilab.base.backend import create_backend
from unilab.base.np_env import NpEnvState
from unilab.dr import (
    DomainRandomizationCapabilities,
    DomainRandomizationProvider,
    GeomSizeOverride,
    InitRandomizationPlan,
    IntervalRandomizationPlan,
    ModelVariantSpec,
    ResetPlan,
)
from unilab.dr.dr_utils import build_common_reset_randomization, validate_common_reset_randomization
from unilab.dr.types import (
    RESET_TERM_BODY_IPOS,
    RESET_TERM_BODY_MASS,
    RESET_TERM_GEOM_FRICTION,
    RESET_TERM_GRAVITY,
    RESET_TERM_KD,
    RESET_TERM_KP,
    ResetRandomizationPayload,
)
from unilab.dtype_config import get_global_dtype
from unilab.envs.common.rotation import (
    np_quat_apply,
    np_quat_conjugate,
    np_quat_mul,
    np_quat_to_axis_angle,
)
from unilab.envs.manipulation.sharpa_inhand.base import (
    SharpaInhandBaseCfg,
    SharpaInhandBaseEnv,
    repeat_obs_history,
    resolve_grasp_cache_file,
    sample_scale_grasp_caches,
)


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
    critic_info_dim: int = 9
    reward_config: RewardConfig | None = None
    zero_action_test_mode: bool = False


def sample_random_quaternion(num_envs: int) -> np.ndarray:
    """Sample uniformly distributed random quaternions in wxyz convention.

    Args:
        num_envs: Number of quaternions to sample.

    Returns:
        Quaternion array with shape ``(num_envs, 4)``.
    """
    u1 = np.random.rand(num_envs)
    u2 = np.random.rand(num_envs) * 2.0 * np.pi
    u3 = np.random.rand(num_envs) * 2.0 * np.pi

    q1 = np.sqrt(1.0 - u1) * np.sin(u2)
    q2 = np.sqrt(1.0 - u1) * np.cos(u2)
    q3 = np.sqrt(u1) * np.sin(u3)
    q4 = np.sqrt(u1) * np.cos(u3)

    return np.stack([q4, q1, q2, q3], axis=1).astype(np.float64)


class SharpaInhandRotationDRProvider(DomainRandomizationProvider):
    def validate(self, env: Any, capabilities: DomainRandomizationCapabilities) -> None:
        unsupported = validate_common_reset_randomization(env, capabilities)
        domain_rand = getattr(env.cfg, "domain_rand", None)
        if domain_rand is not None and getattr(domain_rand, "randomize_gravity_direction", False):
            if getattr(domain_rand, "randomize_gravity", False):
                raise ValueError(
                    "Use only one Sharpa gravity randomization mode: "
                    "domain_rand.randomize_gravity_direction or domain_rand.randomize_gravity"
                )
            if not capabilities.supports_reset_term(RESET_TERM_GRAVITY):
                unsupported = unsupported | frozenset({RESET_TERM_GRAVITY})
        if (
            domain_rand is not None
            and domain_rand.force_scale > 0.0
            and not capabilities.supports_interval_body_force
        ):
            raise NotImplementedError(
                f"{env._backend.backend_type} backend does not support interval body force perturbation"
            )
        if domain_rand is not None and domain_rand.randomize_pd_gains:
            if not capabilities.supports_reset_term(RESET_TERM_KP):
                unsupported = unsupported | frozenset({RESET_TERM_KP})
            if not capabilities.supports_reset_term(RESET_TERM_KD):
                unsupported = unsupported | frozenset({RESET_TERM_KD})
        if (
            domain_rand is not None
            and domain_rand.randomize_com
            and not capabilities.supports_reset_term(RESET_TERM_BODY_IPOS)
        ):
            unsupported = unsupported | frozenset({RESET_TERM_BODY_IPOS})
        if (
            domain_rand is not None
            and domain_rand.randomize_mass
            and not capabilities.supports_reset_term(RESET_TERM_BODY_MASS)
        ):
            unsupported = unsupported | frozenset({RESET_TERM_BODY_MASS})
        if (
            domain_rand is not None
            and domain_rand.randomize_friction
            and not capabilities.supports_reset_term(RESET_TERM_GEOM_FRICTION)
        ):
            unsupported = unsupported | frozenset({RESET_TERM_GEOM_FRICTION})
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

    def _load_grasp_cache(self, env: Any) -> tuple[np.ndarray, ...]:
        """Load one grasp cache file for each configured object scale.

        Args:
            env: Sharpa rotation env instance.

        Returns:
            Tuple of cache arrays ordered the same as ``env.scale_values``.
        """
        if getattr(env, "_grasp_cache", None) is not None:
            return cast(tuple[np.ndarray, ...], env._grasp_cache)

        grasp_caches: list[np.ndarray] = []
        missing_files: list[str] = []
        for scale_value in np.asarray(env.scale_values, dtype=np.float64):
            cache_file = resolve_grasp_cache_file(env.cfg.grasp_cache_path, float(scale_value))
            if not cache_file.exists():
                missing_files.append(str(cache_file))
                continue
            grasp_caches.append(np.load(cache_file).astype(np.float64))

        if missing_files:
            missing = ", ".join(missing_files)
            raise RuntimeError(f"Missing Sharpa grasp cache file(s): {missing}")

        env._grasp_cache = tuple(grasp_caches)
        return cast(tuple[np.ndarray, ...], env._grasp_cache)

    def _sample_reset_pd_gains(
        self,
        env: Any,
        num_reset: int,
        *,
        dtype: np.dtype[Any],
    ) -> tuple[np.ndarray, np.ndarray]:
        """Sample absolute reset-time PD gains from split-around-1 scale ranges.

        Args:
            env: Sharpa rotation env instance.
            num_reset: Number of environments being reset.
            dtype: Output dtype for the returned gain arrays.

        Returns:
            Tuple of absolute ``(p_gain, d_gain)`` arrays with shape
            ``(num_reset, env._num_action)``.
        """
        p_gain = np.broadcast_to(env._default_p_gain, (num_reset, env._num_action)).astype(
            dtype, copy=True
        )
        d_gain = np.broadcast_to(env._default_d_gain, (num_reset, env._num_action)).astype(
            dtype, copy=True
        )
        domain_rand = env.cfg.domain_rand
        if domain_rand.randomize_pd_gains:
            p_scale = env._sample_pd_scales(
                domain_rand.randomize_p_gain_scale_lower,
                domain_rand.randomize_p_gain_scale_upper,
                shape=(num_reset, env._num_action),
            )
            d_scale = env._sample_pd_scales(
                domain_rand.randomize_d_gain_scale_lower,
                domain_rand.randomize_d_gain_scale_upper,
                shape=(num_reset, env._num_action),
            )
            p_gain *= p_scale
            d_gain *= d_scale
        return p_gain, d_gain

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
        p_gain: np.ndarray,
        d_gain: np.ndarray,
        friction_scale: np.ndarray | None,
        randomized_mass: np.ndarray | None,
        randomized_com_offset: np.ndarray | None,
        gravity: np.ndarray | None,
    ) -> dict[str, np.ndarray]:
        num_reset = hand_qpos.shape[0]
        dtype = get_global_dtype()

        critic_info = env._build_reset_critic_info(
            num_reset,
            env_ids,
            friction_scale=friction_scale,
            randomized_mass=randomized_mass,
            randomized_com_offset=randomized_com_offset,
            gravity=gravity,
        ).astype(dtype)

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
        object_pos_anchor = env._build_object_pos_anchor(object_pos_f).astype(dtype)
        critic_info = env._fill_critic_info(
            critic_info=critic_info,
            object_pos=object_pos_f,
            object_pos_anchor=object_pos_anchor,
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
            "object_pos_anchor": object_pos_anchor,
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

        friction_scale = env._sample_friction_scale(num_reset)
        randomized_mass = env._sample_object_mass(num_reset)
        randomized_com_offset = env._sample_object_com_offset(num_reset)
        gravity = env._sample_reset_gravity(num_reset)
        p_gain, d_gain = self._sample_reset_pd_gains(env, num_reset, dtype=get_global_dtype())
        grasp_cache = self._load_grasp_cache(env)
        sampled_pose = sample_scale_grasp_caches(grasp_cache, env.scale_ids[env_ids])

        hand_qpos = sampled_pose[:, : env._num_action]
        object_pos = sampled_pose[:, env._num_action : env._num_action + 3]
        object_quat = sampled_pose[:, env._num_action + 3 : env._num_action + 7]

        rot_axis = np.broadcast_to(env._rot_axis, (num_reset, 3)).copy().astype(np.float64)

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
            p_gain=p_gain,
            d_gain=d_gain,
            friction_scale=friction_scale,
            randomized_mass=randomized_mass,
            randomized_com_offset=randomized_com_offset,
            gravity=gravity,
        )
        # Match the source task by clearing any cached external object force on reset.
        env._random_object_force[env_ids] = 0.0
        env._clear_tactile_history(env_ids)

        return ResetPlan(
            env_ids=env_ids,
            qpos=qpos,
            qvel=qvel,
            info_updates=info_updates,
            randomization=env._build_reset_randomization(
                num_reset,
                p_gain=p_gain,
                d_gain=d_gain,
                friction_scale=friction_scale,
                randomized_mass=randomized_mass,
                randomized_com_offset=randomized_com_offset,
                gravity=gravity,
            ),
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

    def build_interval_randomization_plan(
        self,
        env: Any,
        step_counter: int,
    ) -> IntervalRandomizationPlan | None:
        """Build Sharpa object-force perturbations for the upcoming control step.

        Args:
            env: Sharpa rotation env instance.
            step_counter: Global environment step counter.

        Returns:
            Interval randomization plan carrying direct object-force perturbations,
            or ``None`` when object-force injection is disabled.
        """
        del step_counter
        domain_rand = env.cfg.domain_rand
        if domain_rand.force_scale <= 0.0:
            return None

        decay = float(
            np.power(
                domain_rand.force_decay,
                env.cfg.ctrl_dt / max(domain_rand.force_decay_interval, 1.0e-8),
            )
        )
        env._random_object_force *= decay

        random_mask = np.random.rand(env._num_envs) < float(domain_rand.random_force_prob_scalar)
        if np.any(random_mask):
            object_mass = env._resolve_current_object_mass()
            env._random_object_force[random_mask] = (
                np.random.randn(int(np.sum(random_mask)), 3).astype(np.float64)
                * object_mass[random_mask, None]
                * float(domain_rand.force_scale)
            )

        return IntervalRandomizationPlan(
            body_ids=np.asarray([env._object_body_id], dtype=np.int32),
            body_force=env._random_object_force[:, None, :].copy(),
        )


@registry.env("SharpaInhandRotation", sim_backend="mujoco")
class SharpaInhandRotationEnv(SharpaInhandBaseEnv):
    _cfg: SharpaInhandRotationCfg
    _reward_cfg: RewardConfig
    _OBS_MODE_ALIASES: dict[str, str] = {
        "separated": "separated",
        "flattened": "flattened",
    }
    _CRITIC_BASE_DIM_WITHOUT_OPTIONALS = 8

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
            cfg.scene,
            num_envs,
            cfg.sim_dt,
            base_name=cfg.base_name,
            push_body_name=cfg.domain_rand.push_body_name,
            add_body_sensors=True,
            motrix_max_iterations=cfg.motrix_max_iterations,
        )
        super().__init__(cfg, backend, num_envs)

        self._observation_mode = self._resolve_observation_mode(cfg.obs.observation_mode)
        expected_critic_info_dim = self._expected_critic_info_dim()
        if cfg.critic_info_dim != expected_critic_info_dim:
            cfg.critic_info_dim = expected_critic_info_dim
        policy_frame_dim = self._policy_frame_dim()
        self.obs_buf_lag_history = np.zeros(
            (num_envs, cfg.obs_history_len, policy_frame_dim), dtype=self._np_dtype
        )
        self.proprio_hist_buf = np.zeros(
            (num_envs, cfg.prop_hist_len, policy_frame_dim), dtype=self._np_dtype
        )
        self.critic_info_buf = np.zeros((num_envs, expected_critic_info_dim), dtype=self._np_dtype)
        self._friction_geom_ids = self._resolve_friction_geom_ids()
        self._base_geom_friction = self._resolve_base_geom_friction()
        self._base_gravity = self._resolve_base_gravity()
        self._object_body_id = self._resolve_object_body_id()
        self._base_body_mass = self._resolve_base_body_mass()
        self._base_body_ipos = self._resolve_base_body_ipos()
        self._default_p_gain, self._default_d_gain = self._load_default_pd_gains()
        self._random_object_force = np.zeros((num_envs, 3), dtype=np.float64)

        if cfg.control_config.torque_control:
            raise NotImplementedError(
                "Sharpa torque_control=True is not implemented with the current position-actuator XML setup. "
                "Set env.control_config.torque_control=false. Virtual torques are still computed explicitly for reward terms."
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
        return self._num_scales > 1 or not np.allclose(self.scale_values, self.scale_values[0])

    def _expected_critic_info_dim(self) -> int:
        priv_info_cfg = self._cfg.priv_info
        dim = self._CRITIC_BASE_DIM_WITHOUT_OPTIONALS
        if priv_info_cfg.include_friction_scale:
            dim += 1
        if priv_info_cfg.include_gravity_direction:
            dim += 3
        return dim

    def _resolve_friction_geom_ids(self) -> dict[str, np.ndarray]:
        """Resolve MuJoCo geom ids touched by Sharpa friction randomization.

        Args:
            None.

        Returns:
            Mapping with object, elastomer, and metal collision geom id arrays.
        """
        try:
            object_geom_id = self._backend.get_geom_id(self._cfg.object_geom_name)
            base_body_id = self._backend.get_body_id(self._cfg.base_name)
            hand_body_ids = set(
                int(body_id) for body_id in self._backend.get_body_subtree_ids(base_body_id)
            )
            geom_body_ids = self._backend.get_geom_body_ids()
            geom_contype, geom_conaffinity = self._backend.get_geom_contact_masks()
            geom_names = self._backend.get_geom_names()
        except NotImplementedError:
            empty = np.zeros((0,), dtype=np.int32)
            return {"object": empty, "elastomer": empty, "metal": empty}
        elastomer_ids: list[int] = []
        metal_ids: list[int] = []
        for geom_id, geom_name in enumerate(geom_names):
            if int(geom_body_ids[geom_id]) not in hand_body_ids:
                continue
            if int(geom_contype[geom_id]) == 0 and int(geom_conaffinity[geom_id]) == 0:
                continue
            if "elastomer" in geom_name:
                elastomer_ids.append(geom_id)
            else:
                metal_ids.append(geom_id)

        if not elastomer_ids:
            raise ValueError("No Sharpa elastomer collision geoms found for friction randomization")
        if not metal_ids:
            raise ValueError("No Sharpa metal collision geoms found for friction randomization")

        return {
            "object": np.asarray([object_geom_id], dtype=np.int32),
            "elastomer": np.asarray(elastomer_ids, dtype=np.int32),
            "metal": np.asarray(metal_ids, dtype=np.int32),
        }

    def _resolve_object_body_id(self) -> int:
        """Resolve the MuJoCo body id of the manipulated object.

        Args:
            None.

        Returns:
            The MuJoCo object body id, or -1 when the backend is not MuJoCo.
        """
        if self._object_body_ids.size == 0:
            return -1
        return int(self._object_body_ids[0])

    def _resolve_current_object_mass(self) -> np.ndarray:
        """Resolve the current object mass used by force perturbation sampling.

        Args:
            None.

        Returns:
            Array of shape ``(num_envs,)`` with current object masses.
        """
        if self.state is not None:
            critic_info = np.asarray(
                self.state.info.get(
                    "critic_info",
                    np.zeros((self._num_envs, self._cfg.critic_info_dim), dtype=self._np_dtype),
                ),
                dtype=np.float64,
            )
            mass = critic_info[:, self._critic_info_layout()["mass"]].reshape(self._num_envs)
            if np.all(mass > 0.0):
                return mass
        if self._base_body_mass is None or self._object_body_id < 0:
            raise ValueError("MuJoCo object body-mass cache is unavailable")
        return np.full(
            (self._num_envs,),
            float(self._base_body_mass[self._object_body_id]),
            dtype=np.float64,
        )

    def _resolve_base_geom_friction(self) -> np.ndarray | None:
        """Cache MuJoCo model friction vectors used as torsional/rolling templates.

        Args:
            None.

        Returns:
            Full geom friction table, or None when the backend has no geom-friction hook.
        """
        try:
            return np.asarray(self._backend.get_geom_friction(), dtype=np.float64).copy()
        except NotImplementedError:
            return None

    def _resolve_base_gravity(self) -> np.ndarray:
        """Cache the default gravity vector for privileged-info fallbacks.

        Args:
            None.

        Returns:
            Gravity vector with shape ``(3,)``.
        """
        try:
            return np.asarray(self._backend.get_gravity(), dtype=np.float64).copy()
        except NotImplementedError:
            pass
        return np.asarray([0.0, 0.0, -9.81], dtype=np.float64)

    def _resolve_base_body_mass(self) -> np.ndarray | None:
        """Cache the MuJoCo body-mass table used as reset randomization baseline.

        Args:
            None.

        Returns:
            Full body-mass table, or None when the backend is not MuJoCo.
        """
        try:
            return np.asarray(self._backend.get_body_mass(), dtype=np.float64).copy()
        except NotImplementedError:
            return None

    def _resolve_base_body_ipos(self) -> np.ndarray | None:
        """Cache the MuJoCo inertial-position table used for object COM randomization.

        Args:
            None.

        Returns:
            Full body inertial-position table, or None when the backend is not MuJoCo.
        """
        try:
            return np.asarray(self._backend.get_body_ipos(), dtype=np.float64).copy()
        except NotImplementedError:
            return None

    def _sample_friction_scale(self, batch_size: int) -> np.ndarray | None:
        """Sample one friction multiplier per reset environment.

        Args:
            batch_size: Number of reset environments.

        Returns:
            Shape (batch_size, 1) multipliers, or None when disabled.
        """
        domain_rand = self._cfg.domain_rand
        if not domain_rand.randomize_friction:
            return None
        return np.random.uniform(
            domain_rand.randomize_friction_scale_lower,
            domain_rand.randomize_friction_scale_upper,
            size=(batch_size, 1),
        ).astype(np.float64)

    def _sample_object_mass(self, batch_size: int) -> np.ndarray | None:
        """Sample reset-time object masses.

        Args:
            batch_size: Number of reset environments.

        Returns:
            Shape (batch_size, 1) absolute object masses, or None when disabled.
        """
        domain_rand = self._cfg.domain_rand
        if not domain_rand.randomize_mass:
            return None
        return np.random.uniform(
            domain_rand.randomize_mass_lower,
            domain_rand.randomize_mass_upper,
            size=(batch_size, 1),
        ).astype(np.float64)

    def _sample_object_com_offset(self, batch_size: int) -> np.ndarray | None:
        """Sample reset-time object COM offsets.

        Args:
            batch_size: Number of reset environments.

        Returns:
            Shape (batch_size, 3) COM offsets, or None when disabled.
        """
        domain_rand = self._cfg.domain_rand
        if not domain_rand.randomize_com:
            return None
        return np.random.uniform(
            domain_rand.randomize_com_lower,
            domain_rand.randomize_com_upper,
            size=(batch_size, 3),
        ).astype(np.float64)

    def _friction_profile(self, material: str, base_sliding_friction: float) -> np.ndarray:
        """Build one MuJoCo friction vector while preserving XML friction ratios.

        Args:
            material: One of object, elastomer, or metal.
            base_sliding_friction: Sliding-friction coefficient before random scaling.

        Returns:
            Shape (3,) MuJoCo friction vector.
        """
        if self._base_geom_friction is None:
            raise ValueError("MuJoCo base geom friction cache is unavailable")

        geom_ids = self._friction_geom_ids[material]
        template = np.asarray(self._base_geom_friction[int(geom_ids[0])], dtype=np.float64)
        sliding = float(template[0])
        if sliding <= 0.0:
            raise ValueError(f"{material} base sliding friction must be positive, got {sliding}")
        return np.asarray(template / sliding * base_sliding_friction, dtype=np.float64)

    def _sample_reset_gravity(self, batch_size: int) -> np.ndarray | None:
        """Sample the exact reset-time gravity vector applied by Sharpa.

        Args:
            batch_size: Number of reset environments.

        Returns:
            Gravity array with shape ``(batch_size, 3)``, or ``None`` when reset-time
            gravity randomization is disabled.
        """
        gravity = self._build_gravity_direction_randomization(batch_size)
        if gravity is not None:
            return gravity
        domain_rand = self._cfg.domain_rand
        if not getattr(domain_rand, "randomize_gravity", False):
            return None
        gravity_range = np.asarray(domain_rand.gravity_range, dtype=np.float64)
        if gravity_range.shape != (2, 3):
            raise ValueError(
                f"domain_rand.gravity_range must have shape (2, 3), got {gravity_range.shape}"
            )
        low = np.minimum(gravity_range[0], gravity_range[1])
        high = np.maximum(gravity_range[0], gravity_range[1])
        return np.random.uniform(low=low, high=high, size=(batch_size, 3)).astype(np.float64)

    def _resolved_privileged_gravity(
        self,
        batch_size: int,
        gravity: np.ndarray | None,
    ) -> np.ndarray:
        """Resolve the gravity vector written into privileged info for one reset batch.

        Args:
            batch_size: Number of reset environments.
            gravity: Sampled reset gravity, if reset-time randomization is active.

        Returns:
            Gravity array with shape ``(batch_size, 3)``.
        """
        if gravity is not None:
            return np.asarray(gravity, dtype=self._np_dtype)
        return np.broadcast_to(self._base_gravity, (batch_size, 3)).astype(
            self._np_dtype, copy=True
        )

    def _build_friction_randomization(
        self,
        batch_size: int,
        friction_scale: np.ndarray | None,
    ) -> np.ndarray | None:
        """Build the reset-time MuJoCo geom_friction randomization table.

        Args:
            batch_size: Number of reset environments.
            friction_scale: Shape (batch_size, 1) sampled friction multipliers.

        Returns:
            Shape (batch_size, ngeom, 3) friction table, or None when disabled.
        """
        if friction_scale is None:
            return None
        if self._base_geom_friction is None:
            raise ValueError("MuJoCo base geom friction cache is unavailable")

        geom_friction = np.broadcast_to(
            self._base_geom_friction,
            (batch_size, *self._base_geom_friction.shape),
        ).copy()
        scale = np.asarray(friction_scale, dtype=np.float64).reshape(batch_size, 1, 1)
        domain_rand = self._cfg.domain_rand
        material_profiles = {
            "object": self._friction_profile("object", domain_rand.object_base_friction),
            "elastomer": self._friction_profile("elastomer", domain_rand.elastomer_base_friction),
            "metal": self._friction_profile("metal", domain_rand.metal_base_friction),
        }
        for material, profile in material_profiles.items():
            geom_friction[:, self._friction_geom_ids[material], :] = scale * profile.reshape(
                1, 1, 3
            )
        return geom_friction

    def _build_object_mass_randomization(
        self,
        batch_size: int,
        randomized_mass: np.ndarray | None,
    ) -> np.ndarray | None:
        """Build the reset-time MuJoCo body_mass table for object mass randomization.

        Args:
            batch_size: Number of reset environments.
            randomized_mass: Shape (batch_size, 1) sampled absolute object masses.

        Returns:
            Shape (batch_size, nbody) mass table, or None when disabled.
        """
        if randomized_mass is None:
            return None
        if self._base_body_mass is None or self._object_body_id < 0:
            raise ValueError("MuJoCo base body-mass cache is unavailable")
        body_mass = np.broadcast_to(
            self._base_body_mass, (batch_size, self._base_body_mass.size)
        ).copy()
        body_mass[:, self._object_body_id] = np.asarray(randomized_mass, dtype=np.float64).reshape(
            batch_size
        )
        return body_mass

    def _build_object_com_randomization(
        self,
        batch_size: int,
        randomized_com_offset: np.ndarray | None,
    ) -> np.ndarray | None:
        """Build the reset-time MuJoCo body_ipos table for object COM randomization.

        Args:
            batch_size: Number of reset environments.
            randomized_com_offset: Shape (batch_size, 3) sampled COM offsets.

        Returns:
            Shape (batch_size, nbody, 3) inertial-position table, or None when disabled.
        """
        if randomized_com_offset is None:
            return None
        if self._base_body_ipos is None or self._object_body_id < 0:
            raise ValueError("MuJoCo base body-ipos cache is unavailable")
        body_ipos = np.broadcast_to(
            self._base_body_ipos,
            (batch_size, *self._base_body_ipos.shape),
        ).copy()
        body_ipos[:, self._object_body_id, :] += np.asarray(randomized_com_offset, dtype=np.float64)
        return body_ipos

    def _build_gravity_direction_randomization(self, batch_size: int) -> np.ndarray | None:
        """Sample fixed-magnitude gravity vectors with randomized directions.

        Args:
            batch_size: Number of reset environments.

        Returns:
            Gravity vectors with shape ``(batch_size, 3)``, or None when disabled.
        """
        domain_rand = self._cfg.domain_rand
        if not getattr(domain_rand, "randomize_gravity_direction", False):
            return None
        magnitude = float(getattr(domain_rand, "gravity_direction_magnitude", 9.81))
        if magnitude <= 0.0:
            raise ValueError(f"gravity_direction_magnitude must be positive, got {magnitude}")
        gravity = np.zeros((batch_size, 3), dtype=np.float64)
        gravity[:, 2] = -magnitude
        random_quat = sample_random_quaternion(batch_size)
        # A uniform quaternion and its inverse have the same distribution, so
        # rotating gravity samples the same relative gravity directions induced
        # by uniformly rotating the global hand/object/task frame.
        return np.asarray(np_quat_apply(random_quat, gravity), dtype=np.float64)

    def _build_reset_randomization(
        self,
        batch_size: int,
        *,
        p_gain: np.ndarray | None,
        d_gain: np.ndarray | None,
        friction_scale: np.ndarray | None,
        randomized_mass: np.ndarray | None,
        randomized_com_offset: np.ndarray | None,
        gravity: np.ndarray | None,
    ) -> ResetRandomizationPayload | None:
        """Build reset-randomization payloads owned by the Sharpa rotation env.

        Args:
            batch_size: Number of reset environments.
            friction_scale: Shape (batch_size, 1) sampled friction multipliers.

        Returns:
            Reset-randomization payload, or None when no backend randomization is requested.
        """
        payload = build_common_reset_randomization(self, batch_size)
        if self._cfg.domain_rand.randomize_pd_gains:
            if payload is None:
                payload = ResetRandomizationPayload()
            payload.kp = np.asarray(p_gain, dtype=np.float64)
            payload.kd = np.asarray(d_gain, dtype=np.float64)
        body_mass = self._build_object_mass_randomization(batch_size, randomized_mass)
        body_ipos = self._build_object_com_randomization(batch_size, randomized_com_offset)
        geom_friction = self._build_friction_randomization(batch_size, friction_scale)
        if body_mass is not None:
            if payload is None:
                payload = ResetRandomizationPayload()
            payload.body_mass = body_mass
        if body_ipos is not None:
            if payload is None:
                payload = ResetRandomizationPayload()
            payload.body_ipos = body_ipos
        if geom_friction is not None:
            if payload is None:
                payload = ResetRandomizationPayload()
            payload.geom_friction = geom_friction
        if gravity is not None:
            if payload is None:
                payload = ResetRandomizationPayload()
            payload.gravity = gravity
        return payload

    def _critic_info_layout(self) -> dict[str, slice]:
        """Describe the flat critic_info channel layout.

        Args:
            None.

        Returns:
            Mapping from logical field names to channel slices.
        """
        offset = 0
        layout = {"object_pos_delta": slice(offset, offset + 3)}
        offset += 3
        priv_info_cfg = self._cfg.priv_info
        if priv_info_cfg.include_friction_scale:
            layout["friction"] = slice(offset, offset + 1)
            offset += 1
        layout["mass"] = slice(offset, offset + 1)
        offset += 1
        layout["com"] = slice(offset, offset + 3)
        offset += 3
        layout["scale"] = slice(offset, offset + 1)
        offset += 1
        if priv_info_cfg.include_gravity_direction:
            layout["gravity"] = slice(offset, offset + 3)
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

    def _build_reset_critic_info(
        self,
        batch_size: int,
        env_ids: np.ndarray,
        *,
        friction_scale: np.ndarray | None,
        randomized_mass: np.ndarray | None,
        randomized_com_offset: np.ndarray | None,
        gravity: np.ndarray | None,
    ) -> np.ndarray:
        """Build reset-time critic_info for all randomized object properties.

        Args:
            batch_size: Number of reset environments.
            env_ids: Global environment ids for this reset batch.

        Returns:
            Critic-info tensor with shape (batch_size, critic_info_dim).
        """
        critic_info = np.zeros((batch_size, self._cfg.critic_info_dim), dtype=self._np_dtype)

        priv_info_cfg = self._cfg.priv_info
        if priv_info_cfg.include_friction_scale:
            self._assign_critic_info_field(
                critic_info,
                "friction",
                (
                    friction_scale
                    if friction_scale is not None
                    else np.ones((batch_size, 1), dtype=np.float64)
                ),
            )
        if randomized_mass is not None:
            self._assign_critic_info_field(
                critic_info,
                "mass",
                randomized_mass,
            )
        if randomized_com_offset is not None:
            self._assign_critic_info_field(
                critic_info,
                "com",
                randomized_com_offset,
            )
        if self._scale_randomization_enabled():
            self._assign_critic_info_field(
                critic_info,
                "scale",
                self.scale_values[self.scale_ids[env_ids]].reshape(batch_size, 1),
            )
        if priv_info_cfg.include_gravity_direction:
            self._assign_critic_info_field(
                critic_info,
                "gravity",
                self._resolved_privileged_gravity(batch_size, gravity),
            )
        return critic_info

    def _policy_frame_dim(self) -> int:
        obs_cfg = self._cfg.obs
        dim = self._num_action + self._num_action
        if obs_cfg.enable_tactile:
            dim += self._num_tactile
        if obs_cfg.enable_contact_pos:
            dim += self._num_tactile * 3
        return dim

    def _policy_frame_zeros(self, batch_size: int) -> tuple[np.ndarray, np.ndarray]:
        """Build zero-filled optional policy inputs for reset observations.

        Args:
            batch_size: Number of environments in the batch.

        Returns:
            Tuple of tactile and contact-position arrays sized for the current config.
        """
        obs_cfg = self._cfg.obs
        tactile_dim = self._num_tactile if obs_cfg.enable_tactile else 0
        contact_pos_dim = self._num_tactile * 3 if obs_cfg.enable_contact_pos else 0
        return (
            np.zeros((batch_size, tactile_dim), dtype=self._np_dtype),
            np.zeros((batch_size, contact_pos_dim), dtype=self._np_dtype),
        )

    def _fixed_default_object_pose(self, batch_size: int) -> np.ndarray:
        """Build the fixed default object pose from the backend init state.

        Args:
            batch_size: Number of environments in the batch.

        Returns:
            Array with shape ``(batch_size, 7)`` containing the default object
            position and quaternion loaded from the backend/model init qpos.
        """
        object_pos = np.broadcast_to(
            np.asarray(self._init_qpos[self._obj_pos_slice], dtype=self._np_dtype),
            (batch_size, 3),
        ).copy()
        object_quat = np.broadcast_to(
            np.asarray(self._init_qpos[self._obj_quat_slice], dtype=self._np_dtype),
            (batch_size, 4),
        ).copy()
        return np.concatenate([object_pos, object_quat], axis=1).astype(self._np_dtype)

    def _build_object_pos_anchor(
        self,
        object_pos: np.ndarray,
    ) -> np.ndarray:
        """Choose the reward/privileged-info object-position anchor.

        Args:
            object_pos: Reset-time object positions with shape ``(batch_size, 3)``.

        Returns:
            Array with shape ``(batch_size, 3)``. When the config flag is set,
            the fixed XML/default object position is used; otherwise the sampled
            reset position is returned to preserve the legacy UniLab behavior.
        """
        object_pos = np.asarray(object_pos, dtype=self._np_dtype)
        if self._cfg.use_default_object_pose_for_object_pos_anchor:
            return self._fixed_default_object_pose(object_pos.shape[0])[:, 0:3]
        return object_pos.copy()

    def _resolve_object_pos_anchor(
        self,
        info: dict[str, Any],
        batch_size: int,
    ) -> np.ndarray:
        """Resolve the cached object-position anchor for one runtime batch.

        Args:
            info: Mutable env-state info dictionary.
            batch_size: Number of environments in the current batch.

        Returns:
            Array with shape ``(batch_size, 3)`` used by the object-position
            reward and privileged-info delta channels.
        """
        cached_anchor = info.get("object_pos_anchor")
        if cached_anchor is not None:
            return np.asarray(cached_anchor, dtype=self._np_dtype)

        object_default_pose = info.get("object_default_pose")
        if object_default_pose is not None:
            return np.asarray(object_default_pose, dtype=self._np_dtype)[:, 0:3]

        if self._cfg.use_default_object_pose_for_object_pos_anchor:
            return self._fixed_default_object_pose(batch_size)[:, 0:3]
        return np.zeros((batch_size, 3), dtype=self._np_dtype)

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
        obs_cfg = self._cfg.obs
        if obs_cfg.enable_tactile:
            parts.append(np.asarray(tactile, dtype=self._np_dtype))
        if obs_cfg.enable_contact_pos:
            parts.append(np.asarray(contact_pos, dtype=self._np_dtype))
        return parts

    def _fill_critic_info(
        self,
        critic_info: np.ndarray,
        object_pos: np.ndarray,
        object_pos_anchor: np.ndarray,
    ) -> np.ndarray:
        """Populate privileged channels that are derived from runtime state.

        Args:
            critic_info: Critic info buffer to update in place.
            object_pos: Current object positions with shape (batch, 3).
            object_pos_anchor: Object-position anchor with shape (batch, 3).

        Returns:
            Updated critic info array with shape (batch, critic_info_dim).
        """
        self._assign_critic_info_field(
            critic_info,
            "object_pos_delta",
            object_pos - object_pos_anchor,
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
        joint_noise_scale = float(self._cfg.domain_rand.joint_noise_scale)
        if joint_noise_scale > 0.0:
            dof_norm += (
                np.random.uniform(-1.0, 1.0, size=dof_norm.shape).astype(self._np_dtype)
                * joint_noise_scale
            )

        frame = np.asarray(
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
        return self._clip_observation_values(frame)

    def _clip_observation_values(self, values: np.ndarray) -> np.ndarray:
        """Clamp observation tensors to a configurable absolute value bound.

        Args:
            values: Observation-like array to clip.

        Returns:
            Clipped array with the same shape. Non-positive ``clip_obs`` disables the
            clamp so callers can opt out when needed.
        """
        clip_max = float(getattr(self._cfg, "clip_obs", 5.0))
        values = np.asarray(values, dtype=self._np_dtype)
        if clip_max <= 0.0:
            return values
        return np.asarray(np.clip(values, -clip_max, clip_max), dtype=self._np_dtype)

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

        object_pos_anchor = self._resolve_object_pos_anchor(info, batch_size)
        critic_info = self._fill_critic_info(
            critic_info=critic_info,
            object_pos=object_pos,
            object_pos_anchor=object_pos_anchor,
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
            flattened_obs = self._clip_observation_values(
                np.concatenate([policy_obs, critic_info], axis=1).astype(self._np_dtype)
            )
            return {"obs": flattened_obs}

        return {
            "obs": self._clip_observation_values(policy_obs),
            "critic": self._clip_observation_values(
                np.concatenate([policy_obs, critic_info], axis=1).astype(self._np_dtype)
            ),
        }

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

        object_pos_anchor = self._resolve_object_pos_anchor(info, object_pos.shape[0])
        object_pos_reward = 1.0 / (np.linalg.norm(object_pos - object_pos_anchor, axis=1) + 0.001)

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

        return np.asarray(reward, dtype=self._np_dtype) * self._cfg.ctrl_dt

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
