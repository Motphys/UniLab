from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, cast

import numpy as np

from unilab.base import registry
from unilab.base.dtype_config import get_global_dtype
from unilab.base.np_env import NpEnvState
from unilab.dr import DomainRandomizationCapabilities, DomainRandomizationProvider, ResetPlan
from unilab.dr.dr_utils import build_common_reset_randomization, validate_common_reset_randomization
from unilab.envs.manipulation.sharpa_inhand.base import (
    SharpaInhandBaseCfg,
    repeat_obs_history,
    resolve_grasp_cache_file,
)
from unilab.envs.manipulation.sharpa_inhand.rotation import RewardConfig, SharpaInhandRotationEnv
from unilab.utils.math_utils import np_quat_error_magnitude


@dataclass
class SharpaInhandRotationGraspCfg(SharpaInhandBaseCfg):
    max_episode_seconds: float = 12.0
    torque_control: bool = False

    reset_height_lower: float = 0.61406
    reset_height_upper: float = 0.62406
    reset_angle_diff: float = 30.0 / 180.0 * np.pi
    reset_random_quat: bool = False

    grasp_cache_path: str = ""

    randomize_pd_gains: bool = False
    randomize_friction: bool = False
    randomize_com: bool = False
    randomize_mass: bool = True
    randomize_mass_lower: float = 0.05
    randomize_mass_upper: float = 0.051

    force_scale: float = 0.0
    random_force_prob_scalar: float = 0.0
    gravity_curriculum: bool = False

    reward_config: RewardConfig = field(
        default_factory=lambda: RewardConfig(
            scales={
                "rotate": 0.0,
                "obj_linvel": 0.0,
                "pose_diff": 0.0,
                "torque": 0.0,
                "work": 0.0,
                "object_pos": 0.0,
            }
        )
    )

    grasp_collection_target: int = 50_000
    grasp_auto_save: bool = True


@registry.envcfg("SharpaInhandRotationGrasp")
@dataclass
class SharpaInhandGraspEnvCfg(SharpaInhandRotationGraspCfg):
    pass


class SharpaInhandGraspDRProvider(DomainRandomizationProvider):
    def validate(self, env: Any, capabilities: DomainRandomizationCapabilities) -> None:
        unsupported = validate_common_reset_randomization(env, capabilities)
        if unsupported:
            names = ", ".join(sorted(unsupported))
            raise NotImplementedError(
                f"{env._backend.backend_type} backend does not support reset randomization terms: {names}"
            )

    def _build_info_updates(
        self,
        env: Any,
        hand_qpos: np.ndarray,
        object_pos: np.ndarray,
        object_quat: np.ndarray,
    ) -> dict[str, np.ndarray]:
        num_reset = hand_qpos.shape[0]
        dtype = get_global_dtype()

        tactile = np.zeros((num_reset, env._num_tactile), dtype=dtype)
        contact_pos = np.zeros((num_reset, env._num_tactile * 3), dtype=dtype)
        hand_qpos_f = hand_qpos.astype(dtype)
        dof_norm = env._normalize_joint_pos(hand_qpos_f)
        init_frame = np.concatenate([dof_norm, hand_qpos_f, tactile, contact_pos], axis=1).astype(
            dtype
        )
        obs_lag_history = repeat_obs_history(init_frame, env.cfg.obs_history_len).astype(dtype)

        p_gain = np.full((num_reset, env._num_action), env.cfg.control_config.p_gain, dtype=dtype)
        d_gain = np.full((num_reset, env._num_action), env.cfg.control_config.d_gain, dtype=dtype)

        priv_info = np.zeros((num_reset, env.cfg.priv_info_dim), dtype=dtype)
        if env.cfg.randomize_mass:
            priv_info[:, 4] = np.random.uniform(
                env.cfg.randomize_mass_lower,
                env.cfg.randomize_mass_upper,
                size=(num_reset,),
            ).astype(dtype)

        object_default_pose = np.concatenate([object_pos, object_quat], axis=1).astype(dtype)

        return {
            "current_actions": np.zeros((num_reset, env._num_action), dtype=dtype),
            "last_actions": np.zeros((num_reset, env._num_action), dtype=dtype),
            "prev_targets": hand_qpos_f.copy(),
            "init_pose": hand_qpos_f.copy(),
            "prev_hand_pos": hand_qpos_f.copy(),
            "prev_object_pos": object_pos.astype(dtype).copy(),
            "prev_object_quat": object_quat.astype(dtype).copy(),
            "object_default_pose": object_default_pose,
            "reset_height_lower": np.full((num_reset,), env.cfg.reset_height_lower, dtype=dtype),
            "reset_height_upper": np.full((num_reset,), env.cfg.reset_height_upper, dtype=dtype),
            "rot_axis": np.broadcast_to(env._rot_axis, (num_reset, 3)).astype(dtype),
            "p_gain": p_gain,
            "d_gain": d_gain,
            "priv_info": priv_info,
            "obs_lag_history": obs_lag_history,
            "proprio_hist": env._update_proprio_history(obs_lag_history),
        }

    def build_reset_plan(self, env: Any, env_ids: np.ndarray) -> ResetPlan:
        env._collect_successful_grasps(env_ids)

        num_reset = len(env_ids)
        if num_reset == 0:
            return ResetPlan(
                env_ids=env_ids,
                qpos=np.zeros((0, env.nq), dtype=np.float64),
                qvel=np.zeros((0, env.nv), dtype=np.float64),
                info_updates={},
                randomization=None,
            )

        rand = 2.0 * np.random.rand(num_reset, env._num_action) - 1.0
        hand_qpos = np.broadcast_to(env.default_angles, (num_reset, env._num_action)).copy()
        hand_qpos += 0.15 * rand
        hand_qpos = np.clip(hand_qpos, env._ctrl_lower, env._ctrl_upper)

        object_pos = np.broadcast_to(env._init_qpos[env._obj_pos_slice], (num_reset, 3)).copy()
        object_quat = np.broadcast_to(env._init_qpos[env._obj_quat_slice], (num_reset, 4)).copy()

        qpos = np.zeros((num_reset, env.nq), dtype=np.float64)
        qpos[:, : env._num_action] = hand_qpos
        qpos[:, env._obj_pos_slice] = object_pos
        qpos[:, env._obj_quat_slice] = object_quat

        qvel = np.zeros((num_reset, env.nv), dtype=np.float64)

        info_updates = self._build_info_updates(
            env,
            hand_qpos=hand_qpos,
            object_pos=object_pos,
            object_quat=object_quat,
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


@registry.env("SharpaInhandRotationGrasp", sim_backend="motrix")
class SharpaInhandRotationGraspEnv(SharpaInhandRotationEnv):
    _cfg: SharpaInhandRotationGraspCfg

    def __init__(
        self,
        cfg: SharpaInhandRotationGraspCfg,
        num_envs: int = 1,
        backend_type: str = "motrix",
    ) -> None:
        super().__init__(cfg, num_envs=num_envs, backend_type=backend_type)

        self._saved_grasping_states: list[list[np.ndarray]] = [
            list() for _ in range(self._num_scales)
        ]
        self._grasp_target_per_scale = max(1, int(cfg.grasp_collection_target // self._num_scales))
        self._grasp_cache_saved = False

        self._init_domain_randomization(SharpaInhandGraspDRProvider())

    def _collect_successful_grasps(self, env_ids: np.ndarray) -> None:
        if self.state is None or len(env_ids) == 0:
            return

        success_mask = self.state.truncated[env_ids] & ~self.state.terminated[env_ids]
        if not np.any(success_mask):
            return

        success_env_ids = env_ids[np.flatnonzero(success_mask)]
        hand_qpos = self.get_hand_dof_pos()[success_env_ids]
        object_pos = self.get_object_pos()[success_env_ids]
        object_quat = self.get_object_quat()[success_env_ids]
        all_states = np.concatenate([hand_qpos, object_pos, object_quat], axis=1).astype(np.float32)

        saved_scale_ids = self.scale_ids[success_env_ids]
        for i, scale_id in enumerate(saved_scale_ids):
            bucket = self._saved_grasping_states[int(scale_id)]
            if len(bucket) < self._grasp_target_per_scale:
                bucket.append(all_states[i : i + 1])

        if self._grasp_cache_saved:
            return

        finished_scales = sum(
            int(len(bucket) >= self._grasp_target_per_scale)
            for bucket in self._saved_grasping_states
        )
        if finished_scales < self._num_scales:
            return

        if not self._cfg.grasp_auto_save:
            self._grasp_cache_saved = True
            return

        output_file = resolve_grasp_cache_file(
            self._cfg.grasp_cache_path or "cache/sharpa_grasp_linspace",
            self._cfg.scale_range,
        )
        output_file.parent.mkdir(parents=True, exist_ok=True)

        by_scale = []
        for bucket in self._saved_grasping_states:
            if bucket:
                by_scale.append(np.concatenate(bucket, axis=0)[: self._grasp_target_per_scale])
            else:
                by_scale.append(np.zeros((0, 29), dtype=np.float32))

        save_data = np.concatenate(by_scale, axis=0)
        np.save(output_file, save_data)

        self._grasp_cache_saved = True
        if self.state is not None:
            log = self.state.info.get("log", {})
            log["grasp_cache/saved"] = 1.0
            log["grasp_cache/num_states"] = float(save_data.shape[0])
            self.state.info["log"] = log

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
        del info, dof_pos, dof_vel, object_pos, object_linvel, object_angvel, torques
        return np.zeros((self._num_envs,), dtype=self._np_dtype)

    def update_state(self, state: NpEnvState) -> NpEnvState:
        next_state = super().update_state(state)

        fingertip_pos = self.get_fingertip_pos()
        object_pos = self.get_object_pos()
        object_quat = self.get_object_quat()
        object_default_pose = np.asarray(
            next_state.info.get(
                "object_default_pose", np.zeros((self._num_envs, 7), dtype=self._np_dtype)
            ),
            dtype=self._np_dtype,
        )

        cond1 = np.all(
            np.linalg.norm(fingertip_pos - object_pos[:, None, :], axis=-1) < 0.1, axis=1
        )
        tactile = np.asarray(self.last_contacts, dtype=self._np_dtype)
        cond2 = np.sum(tactile > 0.5, axis=1) >= 3
        quat_error = np_quat_error_magnitude(object_default_pose[:, 3:7], object_quat)
        cond3 = quat_error < self._cfg.reset_angle_diff

        grasp_valid = cond1 & cond2 & cond3
        terminated = np.asarray(next_state.terminated | (~grasp_valid), dtype=bool)

        reward = np.zeros((self._num_envs,), dtype=self._np_dtype)

        step_count = next_state.info.get("steps", np.zeros((self._num_envs,), dtype=np.uint32))
        should_log = self._enable_reward_log and (int(step_count[0]) % 4 == 0)
        if should_log:
            log = next_state.info.get("log", {})
            log["grasp/cond1"] = float(np.mean(cond1.astype(np.float32)))
            log["grasp/cond2"] = float(np.mean(cond2.astype(np.float32)))
            log["grasp/cond3"] = float(np.mean(cond3.astype(np.float32)))
            log["grasp/valid"] = float(np.mean(grasp_valid.astype(np.float32)))
            collected = float(sum(len(bucket) for bucket in self._saved_grasping_states))
            log["grasp/cache_size"] = collected
            next_state.info["log"] = log

        return next_state.replace(reward=reward, terminated=terminated)


SharpaWaveGraspCfg = SharpaInhandGraspEnvCfg
