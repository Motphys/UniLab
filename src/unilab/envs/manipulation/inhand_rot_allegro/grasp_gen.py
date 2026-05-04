"""Allegro grasp-generation task built on top of rotation env."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from unilab.base import registry
from unilab.base.np_env import NpEnvState
from unilab.envs.manipulation.inhand_rot_allegro.rotation import (
    AllegroRotationPPO,
    AllegroRotationPPOCfg,
    RewardConfigPPO,
)


@registry.envcfg("AllegroInhandRotationGrasp")
@dataclass
class AllegroRotationGraspCfg(AllegroRotationPPOCfg):
    # These are fallback defaults. Hydra task env overrides (e.g.
    # conf/ppo/task/allegro_inhand_grasp/mujoco.yaml and CLI env.*)
    # are applied at env construction and take precedence.
    max_episode_seconds: float = 2.0
    reward_config: RewardConfigPPO = field(
        default_factory=lambda: RewardConfigPPO(
            scales={
                "rotate": 0.0,
                "obj_linvel": 0.0,
                "pose_diff": 0.0,
                "torque": 0.0,
                "work": 0.0,
                "drop": 0.0,
            },
            angvel_clip_min=-0.5,
            angvel_clip_max=0.5,
            reset_z_threshold=0.125,
        )
    )
    gen_grasp: bool = True
    grasp_collection_target: int = 50_000
    grasp_auto_save: bool = True
    grasp_quality_check: bool = True
    grasp_min_contacts: int = 2


@registry.env("AllegroInhandRotationGrasp", sim_backend="mujoco")
@registry.env("AllegroInhandRotationGrasp", sim_backend="motrix")
class AllegroRotationGrasp(AllegroRotationPPO):
    _cfg: AllegroRotationGraspCfg
    _CONTACT_SENSORS = ("ff_contact", "mf_contact", "rf_contact", "th_contact")

    def __init__(
        self, cfg: AllegroRotationGraspCfg, num_envs: int = 1, backend_type: str = "mujoco"
    ) -> None:
        super().__init__(cfg, num_envs=num_envs, backend_type=backend_type)
        self._saved_grasping_states: list[np.ndarray] = []
        self._grasp_cache_saved = False
        self._grasp_target_reached_notified = False

    def apply_action(self, actions: np.ndarray, state: NpEnvState) -> np.ndarray:
        del actions
        zero_actions = np.zeros((self._num_envs, self._NUM_HAND_DOF), dtype=self._np_dtype)
        return super().apply_action(zero_actions, state)

    @staticmethod
    def _sensor_scalar(sensor_data: np.ndarray) -> np.ndarray:
        sensor_data = np.asarray(sensor_data)
        if sensor_data.ndim == 1:
            return sensor_data
        return sensor_data.reshape(sensor_data.shape[0], -1)[:, 0]

    def _contact_count(self) -> np.ndarray:
        contacts = np.stack(
            [self._sensor_scalar(self.get_sensor_data(name)) for name in self._CONTACT_SENSORS],
            axis=1,
        )
        return np.asarray(np.sum(contacts > 0.5, axis=1), dtype=np.int32)

    def _compute_grasp_conditions(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        # Hora-style good grasp criteria from compute_reward:
        # 1) all fingertips near object, 2) enough fingertip contacts, 3) object not dropped.
        ball_pos = self.get_ball_pos()
        fingertip_pos = self.get_fingertip_pos()

        cond1 = np.all(np.linalg.norm(fingertip_pos - ball_pos[:, None, :], axis=-1) < 0.1, axis=1)
        cond2 = self._contact_count() >= int(self._cfg.grasp_min_contacts)
        cond3 = ball_pos[:, 2] > float(self._reward_cfg.reset_z_threshold)
        return (
            np.asarray(cond1, dtype=bool),
            np.asarray(cond2, dtype=bool),
            np.asarray(cond3, dtype=bool),
        )

    def _check_grasp_quality(self, env_ids: np.ndarray) -> np.ndarray:
        cond1, cond2, cond3 = self._compute_grasp_conditions()
        return np.asarray(cond1[env_ids] & cond2[env_ids] & cond3[env_ids], dtype=bool)

    def _total_saved_grasps(self) -> int:
        if not self._saved_grasping_states:
            return 0
        return int(sum(states.shape[0] for states in self._saved_grasping_states))

    def _stop_collection(self) -> None:
        if self._grasp_target_reached_notified:
            return

        target = int(self._cfg.grasp_collection_target)
        if target <= 0:
            return

        total = self._total_saved_grasps()
        if total < target:
            return

        self._grasp_target_reached_notified = True
        print(
            "[AllegroInhandRotationGrasp] Grasp collection target reached "
            f"({total}/{target}). Program stopped."
        )

        if self.state is not None:
            log = self.state.info.get("log", {})
            log["grasp/target_reached"] = 1.0
            self.state.info["log"] = log

        exit(0)

    def _save_grasp_cache(self, force: bool = False) -> None:
        if self._grasp_cache_saved and not force:
            return

        total = self._total_saved_grasps()
        target = int(self._cfg.grasp_collection_target)
        if not force and total < target:
            return

        if total == 0:
            return

        all_states = np.concatenate(self._saved_grasping_states, axis=0).astype(np.float32)
        if target > 0:
            all_states = all_states[:target]

        output_file = Path(self._cfg.grasp_cache_path or "cache/allegro_grasp_50k.npy")
        if not output_file.is_absolute():
            output_file = Path.cwd() / output_file
        output_file.parent.mkdir(parents=True, exist_ok=True)
        np.save(output_file, all_states)

        self._grasp_cache_saved = True
        if self.state is not None:
            log = self.state.info.get("log", {})
            log["grasp_cache/saved"] = 1.0
            log["grasp_cache/num_states"] = float(all_states.shape[0])
            self.state.info["log"] = log

    def _collect_successful_grasps(self, env_ids: np.ndarray) -> None:
        if self.state is None or env_ids.size == 0:
            return

        success_mask = self.state.truncated[env_ids] & ~self.state.terminated[env_ids]
        if not np.any(success_mask):
            return

        success_env_ids = env_ids[np.flatnonzero(success_mask)]
        if self._cfg.grasp_quality_check:
            quality_mask = self._check_grasp_quality(success_env_ids)
            success_env_ids = success_env_ids[np.flatnonzero(quality_mask)]

        if success_env_ids.size == 0:
            return

        curr_dof_pos = np.asarray(self.state.info.get("curr_dof_pos", self.get_hand_dof_pos()))
        curr_ball_pos = np.asarray(self.state.info.get("curr_ball_pos", self.get_ball_pos()))
        curr_ball_quat = np.asarray(self.state.info.get("curr_ball_quat", self.get_ball_quat()))

        hand_qpos = curr_dof_pos[success_env_ids, : self._NUM_HAND_DOF]
        ball_pos = curr_ball_pos[success_env_ids]
        ball_quat = curr_ball_quat[success_env_ids]
        states = np.concatenate([hand_qpos, ball_pos, ball_quat], axis=1).astype(np.float32)

        self._saved_grasping_states.append(states)
        self._save_grasp_cache()
        self._stop_collection()

        if self.state is not None:
            log = self.state.info.get("log", {})
            log["grasp/cache_size"] = float(self._total_saved_grasps())
            self.state.info["log"] = log

    def update_state(self, state: NpEnvState) -> NpEnvState:
        next_state = super().update_state(state)
        reward = np.zeros((self._num_envs,), dtype=self._np_dtype)

        cond1, cond2, cond3 = self._compute_grasp_conditions()
        if self._cfg.grasp_quality_check:
            grasp_valid = cond1 & cond2 & cond3
            terminated = np.asarray(next_state.terminated | (~grasp_valid), dtype=bool)
        else:
            grasp_valid = np.ones((self._num_envs,), dtype=bool)
            terminated = np.asarray(next_state.terminated, dtype=bool)

        step_count = next_state.info.get("steps", np.zeros((self._num_envs,), dtype=np.uint32))
        should_log = self._enable_reward_log and (int(step_count[0]) % 4 == 0)
        if should_log:
            log = next_state.info.get("log", {})
            log["grasp/cond1"] = float(np.mean(cond1.astype(np.float32)))
            log["grasp/cond2"] = float(np.mean(cond2.astype(np.float32)))
            log["grasp/cond3"] = float(np.mean(cond3.astype(np.float32)))
            log["grasp/valid"] = float(np.mean(grasp_valid.astype(np.float32)))
            log["grasp/cache_size"] = float(self._total_saved_grasps())
            next_state.info["log"] = log

        return next_state.replace(reward=reward, terminated=terminated)

    def _reset_done_envs(self) -> None:
        if self.state is not None:
            done = self.state.terminated | self.state.truncated
            if np.any(done):
                env_ids = np.flatnonzero(done).astype(np.int32)
                self._collect_successful_grasps(env_ids)
        super()._reset_done_envs()

    def close(self) -> None:
        self._save_grasp_cache(force=bool(self._cfg.grasp_auto_save))
        super().close()


AllegroRotationGraspEnv = AllegroRotationGrasp
AllegroRotationGraspCfgAlias = AllegroRotationGraspCfg
