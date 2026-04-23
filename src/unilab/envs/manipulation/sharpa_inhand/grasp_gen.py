from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, cast

import numpy as np

from unilab.base import registry
from unilab.base.np_env import NpEnvState
from unilab.dr import ResetPlan
from unilab.dr.dr_utils import build_common_reset_randomization
from unilab.envs.manipulation.sharpa_inhand.base import (
    SOURCE_DEFAULT_HAND_JOINT_POS_DEG,
    resolve_grasp_cache_file,
)
from unilab.envs.manipulation.sharpa_inhand.rotation import (
    RewardConfig,
    SharpaInhandRotationCfg,
    SharpaInhandRotationDRProvider,
    SharpaInhandRotationEnv,
)
from unilab.utils.math_utils import np_quat_error_magnitude


@dataclass
class SharpaInhandRotationGraspCfg(SharpaInhandRotationCfg):
    max_episode_seconds: float = 3.0  # 12.0
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


class SharpaInhandGraspDRProvider(SharpaInhandRotationDRProvider):
    def build_reset_plan(self, env: Any, env_ids: np.ndarray) -> ResetPlan:
        # Keep original grasp task behavior: collect successful pre-reset states on each reset.
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
        hand_qpos = np.broadcast_to(env._grasp_default_angles, (num_reset, env._num_action)).copy()
        hand_qpos += 0.15 * rand
        hand_qpos = np.clip(hand_qpos, env._ctrl_lower, env._ctrl_upper)

        object_pos = np.broadcast_to(env._init_qpos[env._obj_pos_slice], (num_reset, 3)).copy()
        object_quat = np.broadcast_to(env._init_qpos[env._obj_quat_slice], (num_reset, 4)).copy()

        qpos = np.zeros((num_reset, env.nq), dtype=np.float64)
        qpos[:, : env._num_action] = hand_qpos
        qpos[:, env._obj_pos_slice] = object_pos
        qpos[:, env._obj_quat_slice] = object_quat

        qvel = np.zeros((num_reset, env.nv), dtype=np.float64)
        p_gain, d_gain = self._sample_reset_pd_gains(env, num_reset, dtype=env._np_dtype)

        info_updates = self._build_info_updates(
            env,
            env_ids=env_ids,
            hand_qpos=hand_qpos,
            object_pos=object_pos,
            object_quat=object_quat,
            reset_height_lower=np.full((num_reset,), env.cfg.reset_height_lower, dtype=np.float64),
            reset_height_upper=np.full((num_reset,), env.cfg.reset_height_upper, dtype=np.float64),
            rot_axis=np.broadcast_to(env._rot_axis, (num_reset, 3)).astype(np.float64),
            p_gain=p_gain,
            d_gain=d_gain,
            friction_scale=None,
            randomized_mass=None,
            randomized_com_offset=None,
        )

        return ResetPlan(
            env_ids=env_ids,
            qpos=qpos,
            qvel=qvel,
            info_updates=info_updates,
            randomization=build_common_reset_randomization(env, num_reset),
        )


@registry.env("SharpaInhandRotationGrasp", sim_backend="mujoco")
@registry.env("SharpaInhandRotationGrasp", sim_backend="motrix")
class SharpaInhandRotationGraspEnv(SharpaInhandRotationEnv):
    _cfg: SharpaInhandRotationGraspCfg

    def __init__(
        self,
        cfg: SharpaInhandRotationGraspCfg,
        num_envs: int = 1,
        backend_type: str = "motrix",
    ) -> None:
        super().__init__(
            cfg,
            num_envs=num_envs,
            backend_type=backend_type,
            dr_provider=SharpaInhandGraspDRProvider(),
        )

        self._saved_grasping_states: list[list[np.ndarray]] = [
            list() for _ in range(self._num_scales)
        ]
        if self._num_scales != 1:
            raise ValueError(
                "Sharpa grasp generation now collects exactly one object scale per run; "
                f"got scale_list={list(self.scale_values)}"
            )
        self._grasp_target_per_scale = max(1, int(cfg.grasp_collection_target))
        self._grasp_cache_saved = False
        self._grasp_target_reached_notified = False
        self._last_grasp_progress_step = -1
        self._last_grasp_progress_counts: tuple[int, ...] | None = None
        self._grasp_default_angles = np.asarray(
            np.deg2rad(np.asarray(SOURCE_DEFAULT_HAND_JOINT_POS_DEG, dtype=np.float64)),
            dtype=self._np_dtype,
        )
        if self._grasp_default_angles.shape[0] != self._num_action:
            raise ValueError(
                "Source grasp default angle count mismatch: "
                f"{self._grasp_default_angles.shape[0]} vs expected {self._num_action}"
            )
        if np.any(np.bincount(self.scale_ids, minlength=self._num_scales) == 0):
            raise ValueError(
                "Sharpa grasp generation requires at least one environment for the configured scale; "
                f"got num_envs={num_envs}, num_scales={self._num_scales}"
            )

    def apply_action(self, actions: np.ndarray, state: NpEnvState) -> np.ndarray:
        # Grasp-cache collection should not use policy/random actions.
        # Keep controls fixed at reset targets by forcing zero action input.
        zero_actions = np.zeros_like(actions, dtype=self._np_dtype)
        return super().apply_action(zero_actions, state)

    def _total_saved_grasps(self) -> int:
        return int(sum(len(bucket) for bucket in self._saved_grasping_states))

    def _collection_target_reached(self) -> bool:
        return all(
            len(bucket) >= self._grasp_target_per_scale for bucket in self._saved_grasping_states
        )

    def _get_per_scale_grasp_counts(self) -> tuple[int, ...]:
        """Return collected grasp counts for each scale bucket.

        Returns:
            Tuple where index is scale id and value is collected grasp count.
        """
        return tuple(len(bucket) for bucket in self._saved_grasping_states)

    def _maybe_print_grasp_progress(self, force: bool = False) -> None:
        """Print runtime grasp-collection progress grouped by scale.

        Args:
            force: When True, print even if throttling would normally skip.
        """
        if self.state is None:
            return

        counts = self._get_per_scale_grasp_counts()
        step_info = self.state.info.get("steps")
        step = int(step_info[0]) if isinstance(step_info, np.ndarray) and step_info.size > 0 else 0
        if not force:
            if counts == self._last_grasp_progress_counts:
                return
            if self._last_grasp_progress_step >= 0 and step - self._last_grasp_progress_step < 32:
                return

        total = int(sum(counts))
        per_scale = ", ".join(
            f"scale={float(self.scale_values[i]):g}:{count}" for i, count in enumerate(counts)
        )
        print(
            "[SharpaInhandRotationGrasp] "
            f"grasp progress total={total}/{int(self._cfg.grasp_collection_target)}, "
            f"per_scale=[{per_scale}]"
        )
        self._last_grasp_progress_step = step
        self._last_grasp_progress_counts = counts

    def _stop_collection(self) -> None:
        if self._grasp_target_reached_notified:
            return
        if not self._collection_target_reached():
            return

        self._maybe_print_grasp_progress(force=True)
        self._grasp_target_reached_notified = True
        collected = self._total_saved_grasps()
        target = int(self._cfg.grasp_collection_target)
        print(
            "[SharpaInhandRotationGrasp] Grasp collection target reached "
            f"(saved={collected}, configured_target={target}). Program stopped."
        )

        if self.state is not None:
            log = self.state.info.get("log", {})
            log["grasp/target_reached"] = 1.0
            self.state.info["log"] = log

        exit(0)

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

        self._maybe_print_grasp_progress()
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
            self._stop_collection()
            return

        output_file = resolve_grasp_cache_file(
            self._cfg.grasp_cache_path or "cache/sharpa_grasp_linspace",
            float(self.scale_values[0]),
        )
        output_file.parent.mkdir(parents=True, exist_ok=True)
        save_data = np.concatenate(self._saved_grasping_states[0], axis=0)[
            : self._grasp_target_per_scale
        ]
        np.save(output_file, save_data)

        self._grasp_cache_saved = True
        if self.state is not None:
            log = self.state.info.get("log", {})
            log["grasp_cache/saved"] = 1.0
            log["grasp_cache/num_states"] = float(save_data.shape[0])
            self.state.info["log"] = log

        self._stop_collection()

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
            per_scale_counts = self._get_per_scale_grasp_counts()
            log["grasp/target_cache_size"] = float(self._cfg.grasp_collection_target)
            for scale_idx, count in enumerate(per_scale_counts):
                scale_value = float(self.scale_values[scale_idx])
                log[f"grasp/cache_size_scale_{scale_value:g}"] = float(count)
            next_state.info["log"] = log

        return next_state.replace(reward=reward, terminated=terminated)


SharpaWaveGraspCfg = SharpaInhandGraspEnvCfg
