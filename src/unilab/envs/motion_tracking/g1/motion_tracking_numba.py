"""Optional Numba hot path for the G1 motion-tracking task.

This module is deliberately task-owned. It mirrors the reward and termination
math in ``tracking.py`` while keeping the env/backend contracts unchanged.
Importing it is safe when ``numba`` is not installed; constructing the
accelerator is not.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

try:  # pragma: no cover - exercised in environments with numba installed
    from numba import get_num_threads, get_thread_id, njit, prange, set_num_threads

    NUMBA_AVAILABLE = True
except Exception:  # pragma: no cover - default test env may not install numba
    get_num_threads = get_thread_id = njit = prange = set_num_threads = None  # type: ignore[assignment]
    NUMBA_AVAILABLE = False


TERM_ORDER: tuple[str, ...] = (
    "motion_global_root_pos",
    "motion_global_root_ori",
    "motion_body_pos",
    "motion_body_ori",
    "motion_body_lin_vel",
    "motion_body_ang_vel",
    "motion_ee_body_pos_z",
    "motion_joint_pos",
    "motion_joint_vel",
    "action_rate_l2",
    "joint_limit",
    "undesired_contacts",
)
TERM_INDEX = {name: i for i, name in enumerate(TERM_ORDER)}
SUPPORTED_TERMS = frozenset(TERM_ORDER)


@dataclass(frozen=True)
class G1MotionTrackingNumbaResult:
    reward: np.ndarray
    terminated: np.ndarray
    log: dict[str, float]


def _active_terms(scales: Mapping[str, float]) -> frozenset[str]:
    return frozenset(name for name, scale in scales.items() if scale != 0.0)


def unsupported_terms(scales: Mapping[str, float]) -> frozenset[str]:
    """Return nonzero reward terms this task-specific kernel cannot compute."""
    return _active_terms(scales) - SUPPORTED_TERMS


def is_available(scales: Mapping[str, float]) -> bool:
    return NUMBA_AVAILABLE and not unsupported_terms(scales)


if NUMBA_AVAILABLE:

    def _dev(fn):
        return njit(inline="always", fastmath=True, cache=True, nogil=True)(fn)

    @_dev
    def _exp_reward(error, std):
        return math.exp(error / -(std * std))

    @_dev
    def _quat_angle_sq_i(q1, q2, body_idx, i):
        dot = (
            q1[i, body_idx, 0] * q2[i, body_idx, 0]
            + q1[i, body_idx, 1] * q2[i, body_idx, 1]
            + q1[i, body_idx, 2] * q2[i, body_idx, 2]
            + q1[i, body_idx, 3] * q2[i, body_idx, 3]
        )
        dot = abs(dot)
        if dot > 1.0:
            dot = 1.0
        angle = 2.0 * math.acos(dot)
        return angle * angle

    @_dev
    def motion_global_root_pos_i(motion_pos, robot_pos, anchor, std, i):
        dx = motion_pos[i, anchor, 0] - robot_pos[i, anchor, 0]
        dy = motion_pos[i, anchor, 1] - robot_pos[i, anchor, 1]
        dz = motion_pos[i, anchor, 2] - robot_pos[i, anchor, 2]
        return _exp_reward(dx * dx + dy * dy + dz * dz, std)

    @_dev
    def motion_global_root_ori_i(motion_quat, robot_quat, anchor, std, i):
        return _exp_reward(_quat_angle_sq_i(motion_quat, robot_quat, anchor, i), std)

    @_dev
    def _mean_body_xyz_sq_error_i(reference, actual, n_body, i):
        acc = 0.0
        for body_idx in range(n_body):
            dx = reference[i, body_idx, 0] - actual[i, body_idx, 0]
            dy = reference[i, body_idx, 1] - actual[i, body_idx, 1]
            dz = reference[i, body_idx, 2] - actual[i, body_idx, 2]
            acc += dx * dx + dy * dy + dz * dz
        return acc / n_body

    @_dev
    def motion_body_pos_i(reference, actual, n_body, std, i):
        return _exp_reward(_mean_body_xyz_sq_error_i(reference, actual, n_body, i), std)

    @_dev
    def motion_body_ori_i(reference, actual, n_body, std, i):
        acc = 0.0
        for body_idx in range(n_body):
            acc += _quat_angle_sq_i(reference, actual, body_idx, i)
        return _exp_reward(acc / n_body, std)

    @_dev
    def motion_body_lin_vel_i(motion_vel, robot_vel, n_body, std, i):
        return _exp_reward(_mean_body_xyz_sq_error_i(motion_vel, robot_vel, n_body, i), std)

    @_dev
    def motion_body_ang_vel_i(motion_vel, robot_vel, n_body, std, i):
        return _exp_reward(_mean_body_xyz_sq_error_i(motion_vel, robot_vel, n_body, i), std)

    @_dev
    def motion_ee_body_pos_z_i(reference, actual, ee_indices, std, i):
        if ee_indices.shape[0] == 0:
            return 0.0
        acc = 0.0
        for idx in range(ee_indices.shape[0]):
            body_idx = ee_indices[idx]
            dz = reference[i, body_idx, 2] - actual[i, body_idx, 2]
            acc += dz * dz
        return _exp_reward(acc / ee_indices.shape[0], std)

    @_dev
    def _mean_joint_sq_error_i(reference, actual, n_action, i):
        acc = 0.0
        for j in range(n_action):
            d = reference[i, j] - actual[i, j]
            acc += d * d
        return acc / n_action

    @_dev
    def motion_joint_pos_i(motion_joint_pos, dof_pos, n_action, std, i):
        return _exp_reward(_mean_joint_sq_error_i(motion_joint_pos, dof_pos, n_action, i), std)

    @_dev
    def motion_joint_vel_i(motion_joint_vel, dof_vel, n_action, std, i):
        return _exp_reward(_mean_joint_sq_error_i(motion_joint_vel, dof_vel, n_action, i), std)

    @_dev
    def action_rate_l2_i(current_actions, last_actions, n_action, i):
        acc = 0.0
        for j in range(n_action):
            d = current_actions[i, j] - last_actions[i, j]
            acc += d * d
        return acc

    @_dev
    def joint_limit_i(dof_pos, joint_lower, joint_upper, n_action, has_joint_limits, i):
        if not has_joint_limits:
            return 0.0
        acc = 0.0
        for j in range(n_action):
            low = joint_lower[j] - dof_pos[i, j]
            if low < 0.0:
                low = 0.0
            high = dof_pos[i, j] - joint_upper[j]
            if high < 0.0:
                high = 0.0
            v = low + high
            acc += v * v
        return acc

    @_dev
    def undesired_contacts_i(robot_body_pos, undesired_indices, undesired_contact_z_threshold, i):
        acc = 0.0
        for idx in range(undesired_indices.shape[0]):
            if robot_body_pos[i, undesired_indices[idx], 2] < undesired_contact_z_threshold:
                acc += 1.0
        return acc

    @_dev
    def terminated_i(
        motion_pos,
        motion_quat,
        ref_pos,
        robot_pos,
        robot_quat,
        anchor,
        ee_indices,
        undesired_indices,
        anchor_pos_z_threshold,
        anchor_ori_threshold,
        ee_body_pos_z_threshold,
        undesired_contact_z_threshold,
        terminate_on_undesired_contacts,
        i,
    ):
        if abs(motion_pos[i, anchor, 2] - robot_pos[i, anchor, 2]) > anchor_pos_z_threshold:
            return True
        if anchor_ori_threshold < 2.0:
            motion_gravity_z = (
                2.0
                * (
                    motion_quat[i, anchor, 1] * motion_quat[i, anchor, 1]
                    + motion_quat[i, anchor, 2] * motion_quat[i, anchor, 2]
                )
                - 1.0
            )
            robot_gravity_z = (
                2.0
                * (
                    robot_quat[i, anchor, 1] * robot_quat[i, anchor, 1]
                    + robot_quat[i, anchor, 2] * robot_quat[i, anchor, 2]
                )
                - 1.0
            )
            if abs(motion_gravity_z - robot_gravity_z) > anchor_ori_threshold:
                return True
        for idx in range(ee_indices.shape[0]):
            body_idx = ee_indices[idx]
            if abs(ref_pos[i, body_idx, 2] - robot_pos[i, body_idx, 2]) > ee_body_pos_z_threshold:
                return True
        if terminate_on_undesired_contacts:
            for idx in range(undesired_indices.shape[0]):
                if robot_pos[i, undesired_indices[idx], 2] < undesired_contact_z_threshold:
                    return True
        return False

    @njit(parallel=True, fastmath=True, cache=True, nogil=True)  # type: ignore[misc]
    def _compute_reward_termination_kernel(
        motion_body_pos_w,
        motion_body_quat_w,
        motion_body_lin_vel_w,
        motion_body_ang_vel_w,
        motion_joint_pos,
        motion_joint_vel,
        ref_body_pos_w,
        ref_body_quat_w,
        robot_body_pos_w,
        robot_body_quat_w,
        robot_body_lin_vel_w,
        robot_body_ang_vel_w,
        dof_pos,
        dof_vel,
        current_actions,
        last_actions,
        joint_lower,
        joint_upper,
        scale,
        std,
        anchor,
        ee_indices,
        undesired_indices,
        ctrl_dt,
        anchor_pos_z_threshold,
        anchor_ori_threshold,
        ee_body_pos_z_threshold,
        undesired_contact_z_threshold,
        terminate_on_undesired_contacts,
        has_joint_limits,
        reward,
        terminated,
        log_scratch,
    ):
        n = reward.shape[0]
        n_body = robot_body_pos_w.shape[1]
        n_action = dof_pos.shape[1]
        for i in prange(n):
            tid = get_thread_id()
            total = 0.0

            w = motion_global_root_pos_i(
                motion_body_pos_w, robot_body_pos_w, anchor, std[0], i
            ) * scale[0]
            total += w
            log_scratch[tid, 0] += w

            w = motion_global_root_ori_i(
                motion_body_quat_w, robot_body_quat_w, anchor, std[1], i
            ) * scale[1]
            total += w
            log_scratch[tid, 1] += w

            w = motion_body_pos_i(ref_body_pos_w, robot_body_pos_w, n_body, std[2], i) * scale[2]
            total += w
            log_scratch[tid, 2] += w

            w = motion_body_ori_i(ref_body_quat_w, robot_body_quat_w, n_body, std[3], i) * scale[3]
            total += w
            log_scratch[tid, 3] += w

            w = motion_body_lin_vel_i(
                motion_body_lin_vel_w, robot_body_lin_vel_w, n_body, std[4], i
            ) * scale[4]
            total += w
            log_scratch[tid, 4] += w

            w = motion_body_ang_vel_i(
                motion_body_ang_vel_w, robot_body_ang_vel_w, n_body, std[5], i
            ) * scale[5]
            total += w
            log_scratch[tid, 5] += w

            w = motion_ee_body_pos_z_i(
                ref_body_pos_w, robot_body_pos_w, ee_indices, std[6], i
            ) * scale[6]
            total += w
            log_scratch[tid, 6] += w

            w = motion_joint_pos_i(motion_joint_pos, dof_pos, n_action, std[7], i) * scale[7]
            total += w
            log_scratch[tid, 7] += w

            w = motion_joint_vel_i(motion_joint_vel, dof_vel, n_action, std[8], i) * scale[8]
            total += w
            log_scratch[tid, 8] += w

            w = action_rate_l2_i(current_actions, last_actions, n_action, i) * scale[9]
            total += w
            log_scratch[tid, 9] += w

            w = joint_limit_i(
                dof_pos, joint_lower, joint_upper, n_action, has_joint_limits, i
            ) * scale[10]
            total += w
            log_scratch[tid, 10] += w

            w = (
                undesired_contacts_i(
                    robot_body_pos_w, undesired_indices, undesired_contact_z_threshold, i
                )
                * scale[11]
            )
            total += w
            log_scratch[tid, 11] += w

            reward[i] = total * ctrl_dt
            terminated[i] = terminated_i(
                motion_body_pos_w,
                motion_body_quat_w,
                ref_body_pos_w,
                robot_body_pos_w,
                robot_body_quat_w,
                anchor,
                ee_indices,
                undesired_indices,
                anchor_pos_z_threshold,
                anchor_ori_threshold,
                ee_body_pos_z_threshold,
                undesired_contact_z_threshold,
                terminate_on_undesired_contacts,
                i,
            )


class G1MotionTrackingNumbaAccelerator:
    """Driver that keeps config-derived arrays and calls the fused kernel."""

    def __init__(
        self,
        *,
        num_envs: int,
        num_action: int,
        ctrl_dt: float,
        reward_config: Any,
        anchor_body_idx: int,
        ee_body_indices: np.ndarray,
        undesired_contact_body_indices: np.ndarray,
        joint_lower: np.ndarray | None,
        joint_upper: np.ndarray | None,
        anchor_pos_z_threshold: float,
        anchor_ori_threshold: float,
        ee_body_pos_z_threshold: float,
        undesired_contact_z_threshold: float,
        terminate_on_undesired_contacts: bool,
        num_threads: int | None = None,
    ) -> None:
        self.num_envs = int(num_envs)
        self.num_action = int(num_action)
        self.ctrl_dt = float(ctrl_dt)
        self.anchor_body_idx = int(anchor_body_idx)
        self.ee_body_indices = np.asarray(ee_body_indices, dtype=np.int32)
        self.undesired_contact_body_indices = np.asarray(
            undesired_contact_body_indices, dtype=np.int32
        )
        self.anchor_pos_z_threshold = float(anchor_pos_z_threshold)
        self.anchor_ori_threshold = float(anchor_ori_threshold)
        self.ee_body_pos_z_threshold = float(ee_body_pos_z_threshold)
        self.undesired_contact_z_threshold = float(undesired_contact_z_threshold)
        self.terminate_on_undesired_contacts = bool(terminate_on_undesired_contacts)
        self.num_threads = num_threads
        self.has_joint_limits = joint_lower is not None and joint_upper is not None
        if self.has_joint_limits:
            self.joint_lower = np.asarray(joint_lower, dtype=np.float64)
            self.joint_upper = np.asarray(joint_upper, dtype=np.float64)
        else:
            self.joint_lower = np.zeros((self.num_action,), dtype=np.float64)
            self.joint_upper = np.zeros((self.num_action,), dtype=np.float64)
        self.scale = np.zeros((len(TERM_ORDER),), dtype=np.float64)
        self.std = self._build_std_vector(reward_config)
        self._zero_actions = np.zeros((self.num_envs, self.num_action), dtype=np.float64)

    @classmethod
    def from_env(
        cls, env: Any, num_threads: int | None = None
    ) -> "G1MotionTrackingNumbaAccelerator":
        if not NUMBA_AVAILABLE:
            raise RuntimeError(
                "G1MotionTracking numba_acceleration=True requires numba. Install it or run "
                "through `uv run --with numba ...`; disable numba_acceleration to use the "
                "numpy path."
            )
        unsupported = unsupported_terms(env._cfg.reward_config.scales)
        if unsupported:
            raise ValueError(
                "G1MotionTracking Numba accelerator does not support active reward terms "
                f"{sorted(unsupported)}. Disable numba_acceleration or add these terms to "
                "src/unilab/envs/motion_tracking/g1/motion_tracking_numba.py."
            )
        return cls(
            num_envs=env.num_envs,
            num_action=env._num_action,
            ctrl_dt=env._cfg.ctrl_dt,
            reward_config=env._cfg.reward_config,
            anchor_body_idx=env.anchor_body_idx,
            ee_body_indices=env.ee_body_indices,
            undesired_contact_body_indices=env.undesired_contact_body_indices,
            joint_lower=env._joint_lower,
            joint_upper=env._joint_upper,
            anchor_pos_z_threshold=env._cfg.anchor_pos_z_threshold,
            anchor_ori_threshold=env._cfg.anchor_ori_threshold,
            ee_body_pos_z_threshold=env._cfg.ee_body_pos_z_threshold,
            undesired_contact_z_threshold=env._cfg.undesired_contact_z_threshold,
            terminate_on_undesired_contacts=env._cfg.terminate_on_undesired_contacts,
            num_threads=num_threads,
        )

    def _build_std_vector(self, reward_config: Any) -> np.ndarray:
        per_term = {
            "motion_global_root_pos": reward_config.std_root_pos,
            "motion_global_root_ori": reward_config.std_root_ori,
            "motion_body_pos": reward_config.std_body_pos,
            "motion_body_ori": reward_config.std_body_ori,
            "motion_body_lin_vel": reward_config.std_body_lin_vel,
            "motion_body_ang_vel": reward_config.std_body_ang_vel,
            "motion_ee_body_pos_z": reward_config.std_body_pos,
            "motion_joint_pos": reward_config.std_joint_pos,
            "motion_joint_vel": reward_config.std_joint_vel,
            "action_rate_l2": 0.0,
            "joint_limit": 0.0,
            "undesired_contacts": 0.0,
        }
        return np.array([per_term[name] for name in TERM_ORDER], dtype=np.float64)

    def _sync_scales(self, scales: Mapping[str, float]) -> None:
        unsupported = unsupported_terms(scales)
        if unsupported:
            raise ValueError(
                "G1MotionTracking Numba accelerator does not support active reward terms "
                f"{sorted(unsupported)}. Disable numba_acceleration or add these terms to "
                "src/unilab/envs/motion_tracking/g1/motion_tracking_numba.py."
            )
        self.scale.fill(0.0)
        for name, value in scales.items():
            idx = TERM_INDEX.get(name)
            if idx is not None:
                self.scale[idx] = float(value)

    def compute(
        self,
        *,
        info: dict[str, Any],
        motion_data: Any,
        ref_body_pos_w: np.ndarray,
        ref_body_quat_w: np.ndarray,
        robot_body_pos_w: np.ndarray,
        robot_body_quat_w: np.ndarray,
        robot_body_lin_vel_w: np.ndarray,
        robot_body_ang_vel_w: np.ndarray,
        dof_pos: np.ndarray,
        dof_vel: np.ndarray,
        scales: Mapping[str, float],
        enable_log: bool,
    ) -> G1MotionTrackingNumbaResult:
        if not NUMBA_AVAILABLE:
            raise RuntimeError(
                "G1MotionTracking Numba accelerator was constructed while numba is "
                "unavailable; this indicates an invalid accelerator state."
            )
        self._sync_scales(scales)
        if self.num_threads is not None:
            set_num_threads(self.num_threads)

        dtype = dof_pos.dtype
        current_actions = np.asarray(info.get("current_actions", self._zero_actions), dtype=dtype)
        last_actions = np.asarray(info.get("last_actions", self._zero_actions), dtype=dtype)
        reward = np.empty((dof_pos.shape[0],), dtype=dtype)
        terminated = np.empty((dof_pos.shape[0],), dtype=np.bool_)
        log_scratch = np.zeros((get_num_threads(), len(TERM_ORDER)), dtype=np.float64)

        _compute_reward_termination_kernel(
            motion_data.body_pos_w,
            motion_data.body_quat_w,
            motion_data.body_lin_vel_w,
            motion_data.body_ang_vel_w,
            motion_data.joint_pos,
            motion_data.joint_vel,
            ref_body_pos_w,
            ref_body_quat_w,
            robot_body_pos_w,
            robot_body_quat_w,
            robot_body_lin_vel_w,
            robot_body_ang_vel_w,
            dof_pos,
            dof_vel,
            current_actions,
            last_actions,
            self.joint_lower,
            self.joint_upper,
            self.scale,
            self.std,
            self.anchor_body_idx,
            self.ee_body_indices,
            self.undesired_contact_body_indices,
            self.ctrl_dt,
            self.anchor_pos_z_threshold,
            self.anchor_ori_threshold,
            self.ee_body_pos_z_threshold,
            self.undesired_contact_z_threshold,
            self.terminate_on_undesired_contacts,
            self.has_joint_limits,
            reward,
            terminated,
            log_scratch,
        )

        step_count = info.get("steps")
        should_log = enable_log and (
            int(step_count[0]) % 4 == 0 if isinstance(step_count, np.ndarray) else True
        )
        log = {} if should_log else info.get("log", {})
        if should_log:
            term_sums = log_scratch.sum(axis=0)
            for idx, name in enumerate(TERM_ORDER):
                if self.scale[idx] != 0.0:
                    log[f"reward/{name}"] = float(term_sums[idx] / dof_pos.shape[0])
        return G1MotionTrackingNumbaResult(reward=reward, terminated=terminated, log=log)
