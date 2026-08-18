"""Robot-agnostic motion-tracking engine.

Holds :class:`MotionTrackingEnv` (the imitation engine, inheriting the shared
``G1BaseEnv`` locomotion base) and :class:`MotionTrackingDeployEnv` (the
unitree_rl_lab mimic actor variant). Per-concern math lives in the owner
modules (``rewards`` / ``observations`` / ``terminations`` / ``transforms`` /
``reset`` / ``domain_randomization``); the engine keeps only the
thin polymorphic method surface and per-step orchestration.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from unilab.base.backend import create_backend, env_backend_kwargs
from unilab.base.np_env import NpEnvState
from unilab.dtype_config import get_global_dtype
from unilab.envs.locomotion.g1.base import G1BaseEnv
from unilab.envs.motion_tracking.common import observations
from unilab.envs.motion_tracking.common.rewards import (
    RewardContext,
    build_reward_functions,
    compute_reward,
)
from unilab.envs.motion_tracking.common.terminations import compute_terminations
from unilab.envs.motion_tracking.common.transforms import update_relative_transforms

from .config import MotionTrackingCfg, MotionTrackingDeployEnvCfg
from .domain_randomization import MotionTrackingDomainRandomizationProvider
from .motion_loader import MotionData, MotionLoader, MotionSampler
from .reset import build_motion_reference_state


class MotionTrackingEnv(G1BaseEnv):
    """Motion Tracking Environment (robot-agnostic imitation engine)."""

    _cfg: MotionTrackingCfg  # pyright: ignore[reportIncompatibleVariableOverride]

    def __init__(self, cfg: MotionTrackingCfg, num_envs=1, backend_type="mujoco"):
        if not cfg.motion_file:
            raise ValueError("motion_file must be specified in config")

        backend = create_backend(
            backend_type,
            cfg.scene,
            num_envs,
            cfg.sim_dt,
            base_name=cfg.asset.base_name,
            push_body_name=cfg.domain_rand.push_body_name,
            add_body_sensors=True,
            **env_backend_kwargs(cfg),
        )
        super().__init__(cfg, backend, num_envs)

        # Resolve body IDs for backend querying and motion-file indexing.
        self.body_ids = self._backend.get_body_ids(cfg.body_names)
        motion_body_ids = self._backend.get_motion_body_ids(cfg.body_names)

        self.anchor_body_idx = cfg.body_names.index(cfg.anchor_body_name)

        # Get end-effector body indices for termination
        self.ee_body_indices = np.array(
            [cfg.body_names.index(name) for name in cfg.ee_body_names], dtype=np.int32
        )
        self._has_ee_body_indices = bool(self.ee_body_indices.size)

        # Get non-EE body indices for undesired contact penalty
        ee_set = set(cfg.ee_body_names)
        self.undesired_contact_body_indices = np.array(
            [i for i, name in enumerate(cfg.body_names) if name not in ee_set],
            dtype=np.int32,
        )
        self._has_undesired_contact_body_indices = bool(self.undesired_contact_body_indices.size)

        # Load motion data
        self.motion_loader = MotionLoader(cfg.motion_file, body_indices=motion_body_ids)
        self.motion_sampler = MotionSampler(
            self.motion_loader,
            mode=cfg.sampling_mode,
            num_envs=num_envs,
            start_ratio=cfg.sampling_start_ratio,
        )
        needs_kp_kd = cfg.domain_rand.randomize_kp or cfg.domain_rand.randomize_kd
        needs_friction = getattr(cfg.domain_rand, "randomize_geom_friction", False)
        base_kp = base_kd = None
        if needs_kp_kd:
            base_kp, base_kd = backend.get_actuator_gains()
        base_geom_friction = None
        foot_geom_ids = None
        if needs_friction:
            import re as _re

            base_geom_friction = backend.get_geom_friction()
            geom_names = backend.get_geom_names()
            pattern = _re.compile(cfg.domain_rand.friction_geom_pattern)
            foot_geom_ids = np.asarray(
                [i for i, name in enumerate(geom_names) if name and pattern.match(name)],
                dtype=np.int64,
            )
            if foot_geom_ids.size == 0:
                raise ValueError(
                    "friction_geom_pattern "
                    f"'{cfg.domain_rand.friction_geom_pattern}' did not match any geom"
                )
        dr_provider = MotionTrackingDomainRandomizationProvider(
            base_kp=base_kp,
            base_kd=base_kd,
            base_geom_friction=base_geom_friction,
            foot_geom_ids=foot_geom_ids,
        )
        self._init_domain_randomization(dr_provider)

        dtype = get_global_dtype()
        n_body = len(cfg.body_names)
        self._n_motion_bodies = n_body
        self._actor_obs_width = self._actor_obs_dim(self._num_action)
        self._critic_base_obs_width = self._critic_base_obs_dim(self._num_action)
        self._critic_obs_width = self._critic_base_obs_width + n_body * 9
        self._copy_body_state_w = self._backend.copy_body_state_w

        # Buffers for relative body transforms
        self.body_pos_relative_w = np.zeros((num_envs, n_body, 3), dtype=dtype)
        self.body_quat_relative_w = np.zeros((num_envs, n_body, 4), dtype=dtype)
        self.body_quat_relative_w[:, :, 0] = 1.0  # Initialize to identity quaternion
        self._motion_data_buffer = (
            self.motion_loader.make_motion_data_buffer(num_envs)
            if hasattr(self.motion_loader, "make_motion_data_buffer")
            else None
        )
        self._zero_actions = np.zeros((num_envs, self._num_action), dtype=dtype)
        self._joint_range = self._backend.get_joint_range()
        if self._joint_range is not None:
            self._joint_range = np.asarray(self._joint_range, dtype=dtype)
            self._joint_lower = self._joint_range[:, 0]
            self._joint_upper = self._joint_range[:, 1]
        else:
            self._joint_lower = None
            self._joint_upper = None
        self._delta_pos_w = np.empty((num_envs, 3), dtype=dtype)
        self._delta_ori_w = np.empty((num_envs, 4), dtype=dtype)
        self._motion_anchor_pos_b = np.empty((num_envs, 3), dtype=dtype)
        self._motion_anchor_ori_b = np.empty((num_envs, 6), dtype=dtype)
        self._motion_command = np.empty((num_envs, self._num_action * 2), dtype=dtype)
        self._joint_pos_rel = np.empty((num_envs, self._num_action), dtype=dtype)
        self._robot_body_pos_w = np.empty((num_envs, n_body, 3), dtype=dtype)
        self._robot_body_quat_w = np.empty((num_envs, n_body, 4), dtype=dtype)
        self._robot_body_lin_vel_w = np.empty((num_envs, n_body, 3), dtype=dtype)
        self._robot_body_ang_vel_w = np.empty((num_envs, n_body, 3), dtype=dtype)
        self._quat_error_w = np.empty((num_envs, n_body), dtype=dtype)
        self._quat_error_x = np.empty((num_envs, n_body), dtype=dtype)
        self._body_vec_error = np.empty((num_envs, n_body, 3), dtype=dtype)
        self._body_vec_tmp = np.empty((num_envs, n_body, 3), dtype=dtype)
        self._joint_error = np.empty((num_envs, self._num_action), dtype=dtype)
        self._joint_error_upper = np.empty((num_envs, self._num_action), dtype=dtype)
        self._env_error = np.empty((num_envs,), dtype=dtype)
        self._env_error2 = np.empty((num_envs,), dtype=dtype)
        self._reward_term = np.empty((num_envs,), dtype=dtype)
        self._weighted_reward = np.empty((num_envs,), dtype=dtype)
        self._terminated = np.empty((num_envs,), dtype=bool)
        self._env_bool = np.empty((num_envs,), dtype=bool)
        self._ee_pos_error_z = np.empty((num_envs, self.ee_body_indices.size), dtype=dtype)
        self._ee_terminated = np.empty((num_envs, self.ee_body_indices.size), dtype=bool)
        self._undesired_contact_mask = np.empty(
            (num_envs, self.undesired_contact_body_indices.size), dtype=bool
        )

        self._enable_reward_log = True
        self._init_reward_functions()
        self._active_reward_fns = {
            name: reward_fn
            for name, reward_fn in self._reward_fns.items()
            if self._reward_term_is_active(name)
        }
        self._clip_end_truncated = np.zeros((num_envs,), dtype=bool)

    def _effective_default_angles(self, env_ids: np.ndarray | None = None) -> np.ndarray:
        """Return default_angles with per-episode joint-default-pos bias applied."""
        state = getattr(self, "_state", None)
        if state is not None:
            bias = state.info.get("default_dof_pos_bias")
            if bias is not None:
                if env_ids is not None:
                    return self.default_angles + bias[env_ids]
                return self.default_angles + bias
        return self.default_angles

    def apply_action(self, actions: np.ndarray, state: NpEnvState) -> np.ndarray:
        state.info["last_actions"] = state.info.get("current_actions", np.zeros_like(actions))
        state.info["current_actions"] = actions
        exec_actions = (
            state.info["last_actions"]
            if self._cfg.control_config.simulate_action_latency
            else actions
        )
        bias = state.info.get("default_dof_pos_bias")
        base = self.default_angles + bias if bias is not None else self.default_angles
        ctrl: np.ndarray = exec_actions * self._cfg.control_config.action_scale + base
        return ctrl

    def _resample_reference_state(self, env_ids: np.ndarray) -> None:
        motion_frames = self.motion_sampler.sample_frames(env_ids)
        motion_data = self.motion_loader.get_motion_at_frame(motion_frames)
        qpos, qvel = build_motion_reference_state(self, env_ids, motion_data)
        self._backend.set_state(env_ids, qpos, qvel)

    def _refresh_observation_rows(
        self, obs: dict[str, np.ndarray], info: dict, env_ids: np.ndarray
    ) -> None:
        motion_data = self.motion_loader.get_motion_at_frame(
            self.motion_sampler.current_frames[env_ids]
        )
        row_ids = np.asarray(env_ids, dtype=np.intp)
        linvel = self._backend.get_sensor_data_rows(self._cfg.sensor.local_linvel, row_ids)
        gyro = self._backend.get_sensor_data_rows(self._cfg.sensor.gyro, row_ids)
        dof_pos = self.get_dof_pos()[row_ids]
        dof_vel = self.get_dof_vel()[row_ids]
        robot_body_pos_w, robot_body_quat_w = self._backend.get_body_pose_w_rows(
            row_ids, self.body_ids
        )

        obs_info: dict[str, Any] = {}
        current_actions = info.get("current_actions")
        if isinstance(current_actions, np.ndarray):
            obs_info["current_actions"] = current_actions[env_ids]
        obs_info["env_ids"] = env_ids

        refreshed_obs = self._compute_obs(
            obs_info,
            motion_data,
            linvel,
            gyro,
            dof_pos,
            dof_vel,
            robot_body_pos_w,
            robot_body_quat_w,
        )
        for key, value in refreshed_obs.items():
            if value.shape[0] == len(env_ids):
                obs[key][env_ids] = value
            else:
                obs[key][env_ids] = value[env_ids]

    def _get_body_pose_w(self) -> tuple[np.ndarray, np.ndarray]:
        return self._backend.get_body_pose_w(self.body_ids)

    def _get_body_state_w(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        copy_body_state_w = self._copy_body_state_w
        if copy_body_state_w is not None:
            return copy_body_state_w(
                self.body_ids,
                self._robot_body_pos_w,
                self._robot_body_quat_w,
                self._robot_body_lin_vel_w,
                self._robot_body_ang_vel_w,
            )
        robot_body_pos_w, robot_body_quat_w = self._get_body_pose_w()
        robot_body_lin_vel_w, robot_body_ang_vel_w = self._backend.get_body_vel_w(self.body_ids)
        return (
            robot_body_pos_w,
            robot_body_quat_w,
            robot_body_lin_vel_w,
            robot_body_ang_vel_w,
        )

    def _get_joint_range(self) -> np.ndarray | None:
        return self._joint_range

    def _get_current_motion(self) -> MotionData:
        if self._motion_data_buffer is None:
            return self.motion_sampler.get_current_motion()
        return self.motion_sampler.get_current_motion(self._motion_data_buffer)

    @property
    def obs_groups_spec(self) -> dict[str, int]:
        return observations.obs_groups_spec(self)

    def _actor_obs_dim(self, n: int) -> int:
        return observations.actor_obs_dim(n)

    def _critic_base_obs_dim(self, n: int) -> int:
        return observations.critic_base_obs_dim(n)

    def _build_actor_obs(
        self,
        *,
        command: np.ndarray,
        motion_anchor_pos_b: np.ndarray,
        motion_anchor_ori_b: np.ndarray,
        noisy_linvel: np.ndarray,
        noisy_gyro: np.ndarray,
        noisy_joint_pos_rel: np.ndarray,
        noisy_dof_vel: np.ndarray,
        last_actions: np.ndarray,
    ) -> np.ndarray:
        n_action = noisy_joint_pos_rel.shape[1]
        return observations.build_actor_obs(
            actor_obs_dim=self._actor_obs_dim(n_action),
            command=command,
            motion_anchor_pos_b=motion_anchor_pos_b,
            motion_anchor_ori_b=motion_anchor_ori_b,
            noisy_linvel=noisy_linvel,
            noisy_gyro=noisy_gyro,
            noisy_joint_pos_rel=noisy_joint_pos_rel,
            noisy_dof_vel=noisy_dof_vel,
            last_actions=last_actions,
        )

    def _init_reward_functions(self):
        self._reward_fns = build_reward_functions()

    def _reward_term_is_active(self, name: str) -> bool:
        if name == "joint_limit":
            return self._joint_lower is not None and self._joint_upper is not None
        if name == "undesired_contacts":
            return self._has_undesired_contact_body_indices
        if name == "motion_ee_body_pos_z":
            return self._has_ee_body_indices
        return True

    def update_state(self, state: NpEnvState) -> NpEnvState:
        self._clip_end_truncated.fill(False)

        # Get current motion data
        motion_data = self._get_current_motion()

        # Get robot state
        linvel = self.get_local_linvel()
        gyro = self.get_gyro()
        dof_pos = self.get_dof_pos()
        dof_vel = self.get_dof_vel()

        # Get body states
        (
            robot_body_pos_w,
            robot_body_quat_w,
            robot_body_lin_vel_w,
            robot_body_ang_vel_w,
        ) = self._get_body_state_w()

        # Compute relative body transforms (for observations and rewards)
        self._update_relative_transforms(motion_data, robot_body_pos_w, robot_body_quat_w)

        # Compute terminations
        terminated = self._compute_terminations(motion_data, robot_body_pos_w, robot_body_quat_w)

        # Compute reward
        reward = self._compute_reward(
            state.info,
            motion_data,
            robot_body_pos_w,
            robot_body_quat_w,
            robot_body_lin_vel_w,
            robot_body_ang_vel_w,
            dof_pos,
            dof_vel,
        )

        # Compute observations
        obs = self._compute_obs(
            state.info,
            motion_data,
            linvel,
            gyro,
            dof_pos,
            dof_vel,
            robot_body_pos_w,
            robot_body_quat_w,
        )

        # Update failure statistics for adaptive sampling
        self.motion_sampler.update_failure_stats(terminated)

        # Advance motion frames
        done_env_ids = self.motion_sampler.step()
        if len(done_env_ids) > 0:
            if self._cfg.truncate_on_clip_end:
                self._clip_end_truncated[done_env_ids] = True
            else:
                # Match BeyondMimic: clip boundaries are command resampling points, not
                # episode boundaries; sync the simulated robot to the new reference.
                resample_env_ids = done_env_ids[~terminated[done_env_ids]]
                if len(resample_env_ids) > 0:
                    self._resample_reference_state(resample_env_ids)
                    self._refresh_observation_rows(obs, state.info, resample_env_ids)

        return state.replace(obs=obs, reward=reward, terminated=terminated)

    def _compute_truncated(self, state: NpEnvState) -> np.ndarray:
        truncated = super()._compute_truncated(state)
        clip_end_only = getattr(self, "_env_bool", None)
        if clip_end_only is None or clip_end_only.shape != (self._num_envs,):
            clip_end_only = np.empty((self._num_envs,), dtype=bool)
            self._env_bool = clip_end_only
        np.logical_not(state.terminated, out=clip_end_only)
        np.logical_and(self._clip_end_truncated, clip_end_only, out=clip_end_only)
        np.logical_or(truncated, clip_end_only, out=truncated)
        return truncated

    def _update_relative_transforms(
        self, motion_data, robot_body_pos_w: np.ndarray, robot_body_quat_w: np.ndarray
    ):
        """Update relative body transforms for tracking."""
        update_relative_transforms(self, motion_data, robot_body_pos_w, robot_body_quat_w)

    def _compute_terminations(
        self,
        motion_data,
        robot_body_pos_w: np.ndarray,
        robot_body_quat_w: np.ndarray,
    ) -> np.ndarray:
        """Compute termination conditions."""
        return compute_terminations(self, motion_data, robot_body_pos_w, robot_body_quat_w)

    def _write_body_pos_in_anchor_frame(
        self,
        anchor_pos: np.ndarray,
        anchor_quat: np.ndarray,
        body_pos: np.ndarray,
        out: np.ndarray,
    ) -> None:
        observations.write_body_pos_in_anchor_frame(
            anchor_pos, anchor_quat, body_pos, out, body_vec_error=self._body_vec_error
        )

    def _write_body_ori6_in_anchor_frame(
        self,
        anchor_quat: np.ndarray,
        body_quat: np.ndarray,
        out: np.ndarray,
    ) -> None:
        observations.write_body_ori6_in_anchor_frame(anchor_quat, body_quat, out)

    def _compute_obs(
        self,
        info: dict,
        motion_data,
        linvel: np.ndarray,
        gyro: np.ndarray,
        dof_pos: np.ndarray,
        dof_vel: np.ndarray,
        robot_body_pos_w: np.ndarray,
        robot_body_quat_w: np.ndarray,
    ) -> dict[str, np.ndarray]:
        """Compute observations as dict with actor and critic groups."""
        return observations.compute_obs(
            self,
            info,
            motion_data,
            linvel,
            gyro,
            dof_pos,
            dof_vel,
            robot_body_pos_w,
            robot_body_quat_w,
        )

    def _compute_reward(
        self,
        info: dict,
        motion_data,
        robot_body_pos_w: np.ndarray,
        robot_body_quat_w: np.ndarray,
        robot_body_lin_vel_w: np.ndarray,
        robot_body_ang_vel_w: np.ndarray,
        dof_pos: np.ndarray,
        dof_vel: np.ndarray,
    ) -> np.ndarray:
        """Compute reward."""
        ctx = RewardContext(
            info=info,
            motion_data=motion_data,
            robot_body_pos_w=robot_body_pos_w,
            robot_body_quat_w=robot_body_quat_w,
            robot_body_lin_vel_w=robot_body_lin_vel_w,
            robot_body_ang_vel_w=robot_body_ang_vel_w,
            ref_body_pos_w=self.body_pos_relative_w,
            ref_body_quat_w=self.body_quat_relative_w,
            dof_pos=dof_pos,
            dof_vel=dof_vel,
            reward_config=self._cfg.reward_config,
            anchor_body_idx=self.anchor_body_idx,
            ee_body_indices=self.ee_body_indices,
            undesired_contact_body_indices=self.undesired_contact_body_indices,
            joint_lower=self._joint_lower,
            joint_upper=self._joint_upper,
            undesired_contact_z_threshold=self._cfg.undesired_contact_z_threshold,
            num_envs=self._num_envs,
            body_vec_error=self._body_vec_error,
            joint_error=self._joint_error,
            joint_error_upper=self._joint_error_upper,
            env_error=self._env_error,
            env_error2=self._env_error2,
            reward_term=self._reward_term,
            weighted_reward=self._weighted_reward,
            quat_error_w=self._quat_error_w,
            quat_error_x=self._quat_error_x,
            ee_pos_error_z=self._ee_pos_error_z,
            undesired_contact_mask=self._undesired_contact_mask,
        )
        return compute_reward(
            ctx,
            active_reward_fns=self._active_reward_fns,
            all_reward_fns=self._reward_fns,
            scales=self._cfg.reward_config.scales,
            ctrl_dt=self._cfg.ctrl_dt,
            enable_log=self._enable_reward_log,
        )


class MotionTrackingDeployEnv(MotionTrackingEnv):
    """Deploy-oriented motion tracking env with unitree_rl_lab mimic actor inputs."""

    _cfg: MotionTrackingDeployEnvCfg  # pyright: ignore[reportIncompatibleVariableOverride]

    def _actor_obs_dim(self, n: int) -> int:
        return observations.mimic_actor_obs_dim(n)

    def _build_actor_obs(
        self,
        *,
        command: np.ndarray,
        motion_anchor_pos_b: np.ndarray,
        motion_anchor_ori_b: np.ndarray,
        noisy_linvel: np.ndarray,
        noisy_gyro: np.ndarray,
        noisy_joint_pos_rel: np.ndarray,
        noisy_dof_vel: np.ndarray,
        last_actions: np.ndarray,
    ) -> np.ndarray:
        return observations.build_mimic_actor_obs(
            command=command,
            motion_anchor_ori_b=motion_anchor_ori_b,
            noisy_gyro=noisy_gyro,
            noisy_joint_pos_rel=noisy_joint_pos_rel,
            noisy_dof_vel=noisy_dof_vel,
            last_actions=last_actions,
        )
