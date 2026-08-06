"""Optional Numba hot path for the G1 motion-tracking task.

This module is deliberately task-owned. It mirrors the reward and termination
math in ``tracking.py`` while keeping the env/backend contracts unchanged.
Importing it is safe when ``numba`` is not installed; constructing the
accelerator is not.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from unilab.dtype_config import get_global_dtype
from unilab.utils.numba_geometry import (
    quat_angle_sq_at,
    quat_gravity_z_at,
    write_relative_anchor_transform_at,
    write_yaw_aligned_body_transforms_at,
)

try:  # pragma: no cover - exercised in environments with numba installed
    from numba import get_num_threads, njit, set_num_threads

    NUMBA_AVAILABLE = True
except Exception:  # pragma: no cover - default test env may not install numba
    get_num_threads = njit = set_num_threads = None  # type: ignore[assignment]
    NUMBA_AVAILABLE = False


NUMBA_PREAMBLE_ITEM = None
NUMBA_REWARD_ITEMS: dict[str, Any] = {}
NUMBA_OBSERVATION_ITEM = None
NUMBA_TERMINATION_ITEM = None


@dataclass(frozen=True)
class G1MotionTrackingNumbaUpdateStateResult:
    obs: dict[str, np.ndarray]
    reward: np.ndarray
    terminated: np.ndarray
    log: dict[str, float]


def _active_terms(scales: Mapping[str, float]) -> frozenset[str]:
    return frozenset(name for name, scale in scales.items() if scale != 0.0)


def unsupported_terms(scales: Mapping[str, float]) -> frozenset[str]:
    """Return nonzero reward terms this task-specific kernel cannot compute."""
    from unilab.envs.motion_tracking.common.terms import REWARD_KEYS

    return _active_terms(scales) - REWARD_KEYS.keys()


def is_available(scales: Mapping[str, float]) -> bool:
    return NUMBA_AVAILABLE and not unsupported_terms(scales)


if NUMBA_AVAILABLE:

    def _dev(fn):
        return njit(inline="always", fastmath=True, cache=True, nogil=True)(fn)

    @_dev
    def _exp_reward(error, std):
        return math.exp(error / -(std * std))

    @_dev
    def motion_global_root_pos_i(motion_pos, robot_pos, anchor, std, i):
        dx = motion_pos[i, anchor, 0] - robot_pos[i, anchor, 0]
        dy = motion_pos[i, anchor, 1] - robot_pos[i, anchor, 1]
        dz = motion_pos[i, anchor, 2] - robot_pos[i, anchor, 2]
        return _exp_reward(dx * dx + dy * dy + dz * dz, std)

    @_dev
    def motion_global_root_ori_i(motion_quat, robot_quat, anchor, std, i):
        return _exp_reward(quat_angle_sq_at(motion_quat, robot_quat, anchor, i), std)

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
            acc += quat_angle_sq_at(reference, actual, body_idx, i)
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
            motion_gravity_z = quat_gravity_z_at(motion_quat, anchor, i)
            robot_gravity_z = quat_gravity_z_at(robot_quat, anchor, i)
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

    @_dev
    def _write_reference_transforms_i(
        motion_body_pos_w,
        motion_body_quat_w,
        robot_body_pos_w,
        robot_body_quat_w,
        anchor,
        n_body,
        ref_body_pos_w,
        ref_body_quat_w,
        i,
    ):
        write_yaw_aligned_body_transforms_at(
            motion_body_pos_w,
            motion_body_quat_w,
            robot_body_pos_w,
            robot_body_quat_w,
            anchor,
            n_body,
            ref_body_pos_w,
            ref_body_quat_w,
            i,
        )

    @_dev
    def _write_motion_anchor_i(
        motion_body_pos_w,
        motion_body_quat_w,
        robot_body_pos_w,
        robot_body_quat_w,
        anchor,
        motion_anchor_pos_b,
        motion_anchor_ori_b,
        i,
    ):
        write_relative_anchor_transform_at(
            motion_body_pos_w,
            motion_body_quat_w,
            robot_body_pos_w,
            robot_body_quat_w,
            anchor,
            motion_anchor_pos_b,
            motion_anchor_ori_b,
            i,
        )


if NUMBA_AVAILABLE:

    @_dev
    def _write_body_workspace_i(
        body_pos_w,
        body_quat_w,
        anchor,
        body_pos_b,
        body_ori_b,
        i,
    ):
        anchor_px = body_pos_w[i, anchor, 0]
        anchor_py = body_pos_w[i, anchor, 1]
        anchor_pz = body_pos_w[i, anchor, 2]
        aw = body_quat_w[i, anchor, 0]
        ax = body_quat_w[i, anchor, 1]
        ay = body_quat_w[i, anchor, 2]
        az = body_quat_w[i, anchor, 3]
        for body in range(body_pos_w.shape[1]):
            vx = body_pos_w[i, body, 0] - anchor_px
            vy = body_pos_w[i, body, 1] - anchor_py
            vz = body_pos_w[i, body, 2] - anchor_pz
            ix, iy, iz = -ax, -ay, -az
            tx = 2.0 * (iy * vz - iz * vy)
            ty = 2.0 * (iz * vx - ix * vz)
            tz = 2.0 * (ix * vy - iy * vx)
            body_pos_b[i, body, 0] = vx + aw * tx + iy * tz - iz * ty
            body_pos_b[i, body, 1] = vy + aw * ty + iz * tx - ix * tz
            body_pos_b[i, body, 2] = vz + aw * tz + ix * ty - iy * tx

            bw = body_quat_w[i, body, 0]
            bx = body_quat_w[i, body, 1]
            by = body_quat_w[i, body, 2]
            bz = body_quat_w[i, body, 3]
            rw = aw * bw + ax * bx + ay * by + az * bz
            rx = aw * bx - ax * bw - ay * bz + az * by
            ry = aw * by + ax * bz - ay * bw - az * bx
            rz = aw * bz - ax * by + ay * bx - az * bw
            body_ori_b[i, body, 0] = 1.0 - 2.0 * (ry * ry + rz * rz)
            body_ori_b[i, body, 1] = 2.0 * (rx * ry - rw * rz)
            body_ori_b[i, body, 2] = 2.0 * (rx * ry + rw * rz)
            body_ori_b[i, body, 3] = 1.0 - 2.0 * (rx * rx + rz * rz)
            body_ori_b[i, body, 4] = 2.0 * (rx * rz - rw * ry)
            body_ori_b[i, body, 5] = 2.0 * (ry * rz + rw * rx)

    @_dev
    def _plan_preamble_i(
        motion_body_pos_w,
        motion_body_quat_w,
        motion_joint_pos,
        motion_joint_vel,
        robot_body_pos_w,
        robot_body_quat_w,
        dof_pos,
        effective_default_angles,
        anchor,
        delta_pos_w,
        delta_ori_w,
        body_vec_error,
        env_error,
        term_scratch,
        ref_body_pos_w,
        ref_body_quat_w,
        motion_anchor_pos_b,
        motion_anchor_ori_b,
        motion_command,
        joint_pos_rel,
        robot_body_pos_b,
        robot_body_ori_b,
        i,
    ):
        anchor_idx = int(anchor)
        n_body = robot_body_pos_w.shape[1]
        _write_reference_transforms_i(
            motion_body_pos_w,
            motion_body_quat_w,
            robot_body_pos_w,
            robot_body_quat_w,
            anchor_idx,
            n_body,
            ref_body_pos_w,
            ref_body_quat_w,
            i,
        )
        _write_motion_anchor_i(
            motion_body_pos_w,
            motion_body_quat_w,
            robot_body_pos_w,
            robot_body_quat_w,
            anchor_idx,
            motion_anchor_pos_b,
            motion_anchor_ori_b,
            i,
        )
        for j in range(dof_pos.shape[1]):
            motion_command[i, j] = motion_joint_pos[i, j]
            motion_command[i, dof_pos.shape[1] + j] = motion_joint_vel[i, j]
            joint_pos_rel[i, j] = dof_pos[i, j] - effective_default_angles[i, j]
        _write_body_workspace_i(
            robot_body_pos_w,
            robot_body_quat_w,
            anchor_idx,
            robot_body_pos_b,
            robot_body_ori_b,
            i,
        )

    @_dev
    def _root_pos_plan_i(motion_pos, robot_pos, anchor, std, env_error, i):
        return motion_global_root_pos_i(motion_pos, robot_pos, int(anchor), std, i)

    @_dev
    def _root_ori_plan_i(motion_quat, robot_quat, anchor, std, env_error, i):
        return motion_global_root_ori_i(motion_quat, robot_quat, int(anchor), std, i)

    @_dev
    def _body_pos_plan_i(robot_pos, std, ref_pos, body_error, env_error, i):
        return motion_body_pos_i(ref_pos, robot_pos, robot_pos.shape[1], std, i)

    @_dev
    def _body_ori_plan_i(robot_quat, std, ref_quat, quat_w, quat_x, env_error, i):
        return motion_body_ori_i(ref_quat, robot_quat, robot_quat.shape[1], std, i)

    @_dev
    def _body_lin_vel_plan_i(motion_vel, robot_vel, std, body_error, env_error, i):
        return motion_body_lin_vel_i(motion_vel, robot_vel, robot_vel.shape[1], std, i)

    @_dev
    def _body_ang_vel_plan_i(motion_vel, robot_vel, std, body_error, env_error, i):
        return motion_body_ang_vel_i(motion_vel, robot_vel, robot_vel.shape[1], std, i)

    @_dev
    def _ee_pos_plan_i(robot_pos, indices, std, ref_pos, indexed_error, env_error, i):
        return motion_ee_body_pos_z_i(ref_pos, robot_pos, indices, std, i)

    @_dev
    def _joint_pos_plan_i(motion_pos, dof_pos, std, joint_error, env_error, i):
        return motion_joint_pos_i(motion_pos, dof_pos, dof_pos.shape[1], std, i)

    @_dev
    def _joint_vel_plan_i(motion_vel, dof_vel, std, joint_error, env_error, i):
        return motion_joint_vel_i(motion_vel, dof_vel, dof_vel.shape[1], std, i)

    @_dev
    def _action_rate_plan_i(current, previous, joint_error, env_error, i):
        return action_rate_l2_i(current, previous, current.shape[1], i)

    @_dev
    def _joint_limit_plan_i(lower, upper, dof_pos, joint_error, joint_upper, i):
        acc = 0.0
        for j in range(dof_pos.shape[1]):
            low = lower[i, j] - dof_pos[i, j]
            high = dof_pos[i, j] - upper[i, j]
            violation = (low if low > 0.0 else 0.0) + (high if high > 0.0 else 0.0)
            acc += violation * violation
        return acc

    @_dev
    def _undesired_plan_i(robot_pos, indices, threshold, indexed_mask, env_error, i):
        return undesired_contacts_i(robot_pos, indices, threshold, i)

    @_dev
    def _termination_plan_i(
        motion_pos,
        motion_quat,
        robot_pos,
        robot_quat,
        anchor,
        anchor_pos_threshold,
        check_anchor_ori,
        anchor_ori_threshold,
        ee_indices,
        ee_threshold,
        contact_indices,
        contact_threshold,
        ref_pos,
        env_error,
        indexed_error,
        indexed_mask,
        i,
    ):
        anchor_idx = int(anchor)
        if abs(motion_pos[i, anchor_idx, 2] - robot_pos[i, anchor_idx, 2]) > anchor_pos_threshold:
            return True
        if check_anchor_ori != 0.0:
            motion_gravity_z = quat_gravity_z_at(motion_quat, anchor_idx, i)
            robot_gravity_z = quat_gravity_z_at(robot_quat, anchor_idx, i)
            if abs(motion_gravity_z - robot_gravity_z) > anchor_ori_threshold:
                return True
        for index in range(ee_indices.shape[0]):
            body_idx = ee_indices[index]
            if abs(ref_pos[i, body_idx, 2] - robot_pos[i, body_idx, 2]) > ee_threshold:
                return True
        for index in range(contact_indices.shape[0]):
            if robot_pos[i, contact_indices[index], 2] < contact_threshold:
                return True
        return False

    @_dev
    def _copy_plan_i(source, output, offset, i):
        if source.ctypes.data == output.ctypes.data + offset * output.itemsize:
            return
        width = source.size // source.shape[0]
        for j in range(width):
            output[i, offset + j] = source.flat[i * width + j]

    NUMBA_PREAMBLE_ITEM = _plan_preamble_i
    NUMBA_REWARD_ITEMS = {
        "motion_global_root_pos": _root_pos_plan_i,
        "motion_global_root_ori": _root_ori_plan_i,
        "motion_body_pos": _body_pos_plan_i,
        "motion_body_ori": _body_ori_plan_i,
        "motion_body_lin_vel": _body_lin_vel_plan_i,
        "motion_body_ang_vel": _body_ang_vel_plan_i,
        "motion_ee_body_pos_z": _ee_pos_plan_i,
        "motion_joint_pos": _joint_pos_plan_i,
        "motion_joint_vel": _joint_vel_plan_i,
        "action_rate_l2": _action_rate_plan_i,
        "joint_limit": _joint_limit_plan_i,
        "undesired_contacts": _undesired_plan_i,
    }
    NUMBA_OBSERVATION_ITEM = _copy_plan_i
    NUMBA_TERMINATION_ITEM = _termination_plan_i


class MotionTrackingNumbaAccelerator:
    """Resolved-plan Numba driver for the G1 motion-tracking pilot."""

    def __init__(self, env: Any, *, num_threads: int | None = None) -> None:
        from unilab.envs.motion_tracking.common.terms import resolve_motion_term_plan
        from unilab.term.numba import FusedOutputLayout, materialize_numba_plan

        config = env._cfg.term_plan
        if config is None:
            raise ValueError(
                "MotionTracking fused Numba execution requires env.term_plan; disable "
                "numba_acceleration for profiles that retain the legacy NumPy path"
            )
        dtype = np.dtype(get_global_dtype())
        resolved = resolve_motion_term_plan(
            cfg=env,
            n_action=env._num_action,
            n_body=env._n_motion_bodies,
            anchor_body_idx=env.anchor_body_idx,
            ee_indices=env.ee_body_indices,
            undesired_indices=env.undesired_contact_body_indices,
            config=config,
            dtype=dtype,
        )
        inputs = {
            name: np.zeros((env._num_envs, *spec.shape), dtype=spec.numpy_dtype)
            for name, spec in resolved.plan.input_specs.items()
        }
        for name, value in (
            ("joint_lower", env._joint_lower),
            ("joint_upper", env._joint_upper),
            ("effective_default_angles", env.default_angles),
        ):
            if name in inputs and value is not None:
                np.copyto(inputs[name], np.broadcast_to(value, inputs[name].shape))
        observations = {
            group: np.empty((env._num_envs, width), dtype=dtype)
            for group, width in resolved.observation_dims.items()
        }
        layout = FusedOutputLayout(
            preambles=resolved.preamble_outputs,
            rewards=tuple(output for _, output in resolved.reward_outputs),
            observations=resolved.observation_outputs,
            terminations=resolved.termination_outputs,
        )
        self.runtime = materialize_numba_plan(
            resolved.plan,
            layout,
            num_envs=env._num_envs,
            inputs=inputs,
            observations=observations,
            reward=np.empty((env._num_envs,), dtype=dtype),
            terminated=np.empty((env._num_envs,), dtype=np.bool_),
        )
        self.env = env
        self.resolved = resolved
        self.inputs = inputs
        self._owned_inputs = inputs.copy()
        self.obs = observations
        self.num_envs = env._num_envs
        self.num_threads = num_threads
        self.ctrl_dt = float(env._cfg.ctrl_dt)
        self.first_jit_ms: float | None = None
        self._reward_outputs = dict(resolved.reward_outputs)
        self._scale_snapshot: tuple[float, ...] | None = None
        self._plan_indices = {term.name: index for index, term in enumerate(resolved.plan.terms)}
        self._log_scratch: np.ndarray | None = None
        claimed_workspace = set()
        for group, names in resolved.observation_outputs.items():
            offset = 0
            for term_name in names:
                term = resolved.plan.terms[self._plan_indices[term_name]]
                width = resolved.plan.output_specs[term_name].shape[0]
                if len(term.definition.workspace) == 1:
                    source = term.definition.workspace[0].name
                    spec = resolved.plan.workspace_specs[source]
                    if source not in claimed_workspace:
                        output = observations[group]
                        inner_strides = []
                        stride = output.itemsize
                        for dimension in reversed(spec.shape):
                            inner_strides.append(stride)
                            stride *= dimension
                        view = np.ndarray(
                            (env._num_envs, *spec.shape),
                            dtype=spec.numpy_dtype,
                            buffer=output,
                            offset=offset * output.itemsize,
                            strides=(output.strides[0], *reversed(inner_strides)),
                        )
                        self.runtime.bind_workspace(source, view)
                        claimed_workspace.add(source)
                offset += width
        workspace = self.runtime.workspace
        env.body_pos_relative_w = workspace["ref_body_pos_w"]
        env.body_quat_relative_w = workspace["ref_body_quat_w"]
        env._motion_anchor_pos_b = workspace["motion_anchor_pos_b"]
        env._motion_anchor_ori_b = workspace["motion_anchor_ori_b"]
        env._motion_command = workspace["motion_command"]
        env._joint_pos_rel = workspace["joint_pos_rel"]

    @property
    def compile_info(self):
        return self.runtime.compile_info

    @classmethod
    def from_env(cls, env: Any, num_threads: int | None = None) -> "MotionTrackingNumbaAccelerator":
        if not NUMBA_AVAILABLE:
            raise RuntimeError(
                "MotionTracking numba_acceleration=True requires numba. Install it or "
                "disable numba_acceleration to use the NumPy path."
            )
        return cls(env, num_threads=num_threads)

    def _sync_scales(self, scales: Mapping[str, float]) -> None:
        unsupported = _active_terms(scales) - self._reward_outputs.keys()
        if unsupported:
            raise ValueError(
                f"MotionTracking Numba plan does not support active reward terms {sorted(unsupported)}"
            )
        snapshot = tuple(float(scales.get(name, 0.0)) for name in self._reward_outputs)
        if snapshot == self._scale_snapshot:
            return
        for reward_name, output_name in self._reward_outputs.items():
            configured = scales.get(reward_name, 0.0)
            scale = configured if self.resolved.reward_available.get(reward_name, True) else 0.0
            self.runtime.set_scale(output_name, scale)
        self._scale_snapshot = snapshot

    def _bind_inputs(
        self,
        *,
        info: dict[str, Any],
        motion_data: Any,
        linvel: np.ndarray,
        gyro: np.ndarray,
        dof_pos: np.ndarray,
        dof_vel: np.ndarray,
        robot_body_pos_w: np.ndarray,
        robot_body_quat_w: np.ndarray,
        robot_body_lin_vel_w: np.ndarray,
        robot_body_ang_vel_w: np.ndarray,
    ) -> None:
        def bind(name: str, value: np.ndarray) -> None:
            if name in self.inputs:
                self.inputs[name] = np.asarray(value)

        for name, value in (
            ("motion_body_pos_w", motion_data.body_pos_w),
            ("motion_body_quat_w", motion_data.body_quat_w),
            ("motion_body_lin_vel_w", motion_data.body_lin_vel_w),
            ("motion_body_ang_vel_w", motion_data.body_ang_vel_w),
            ("motion_joint_pos", motion_data.joint_pos),
            ("motion_joint_vel", motion_data.joint_vel),
            ("robot_body_pos_w", robot_body_pos_w),
            ("robot_body_quat_w", robot_body_quat_w),
            ("robot_body_lin_vel_w", robot_body_lin_vel_w),
            ("robot_body_ang_vel_w", robot_body_ang_vel_w),
            ("dof_pos", dof_pos),
            ("dof_vel", dof_vel),
            ("linvel", linvel),
            ("gyro", gyro),
        ):
            bind(name, value)
        zero = self.env._zero_actions
        current = info.get("current_actions")
        current = current if isinstance(current, np.ndarray) else zero
        previous = info.get("last_actions")
        previous = previous if isinstance(previous, np.ndarray) else zero
        bind("current_actions", current)
        bind("last_actions", previous)
        bind("obs_actions", current)

        effective = self._owned_inputs["effective_default_angles"]
        bias = info.get("default_dof_pos_bias")
        if isinstance(bias, np.ndarray):
            np.add(self.env.default_angles, bias, out=effective)
        else:
            np.copyto(effective, np.broadcast_to(self.env.default_angles, effective.shape))
        bind("effective_default_angles", effective)

        noise = self.env._cfg.noise_config
        if "noisy_linvel" in self.inputs:
            bind("noisy_linvel", self.env._obs_noise(linvel, noise.scale_linvel))
        if "noisy_gyro" in self.inputs:
            bind("noisy_gyro", self.env._obs_noise(gyro, noise.scale_gyro))
        if "noisy_joint_pos_rel" in self.inputs:
            noisy_joint = self._owned_inputs["noisy_joint_pos_rel"]
            np.subtract(dof_pos, effective, out=noisy_joint)
            bind(
                "noisy_joint_pos_rel",
                self.env._obs_noise(noisy_joint, noise.scale_joint_angle),
            )
        if "noisy_dof_vel" in self.inputs:
            bind("noisy_dof_vel", self.env._obs_noise(dof_vel, noise.scale_joint_vel))
        self.runtime.bind_inputs(self.inputs)

    def compute_update_state(
        self,
        *,
        info: dict[str, Any],
        motion_data: Any,
        linvel: np.ndarray,
        gyro: np.ndarray,
        dof_pos: np.ndarray,
        dof_vel: np.ndarray,
        robot_body_pos_w: np.ndarray,
        robot_body_quat_w: np.ndarray,
        robot_body_lin_vel_w: np.ndarray,
        robot_body_ang_vel_w: np.ndarray,
        scales: Mapping[str, float],
        enable_log: bool,
    ) -> G1MotionTrackingNumbaUpdateStateResult:
        if dof_pos.shape[0] != self.num_envs:
            raise ValueError(
                "G1MotionTracking Numba update_state only supports full-batch updates; "
                f"got {dof_pos.shape[0]} rows for configured num_envs={self.num_envs}."
            )
        self._sync_scales(scales)
        self._bind_inputs(
            info=info,
            motion_data=motion_data,
            linvel=linvel,
            gyro=gyro,
            dof_pos=dof_pos,
            dof_vel=dof_vel,
            robot_body_pos_w=robot_body_pos_w,
            robot_body_quat_w=robot_body_quat_w,
            robot_body_lin_vel_w=robot_body_lin_vel_w,
            robot_body_ang_vel_w=robot_body_ang_vel_w,
        )
        if self.num_threads is not None:
            set_num_threads(self.num_threads)
        nthreads = get_num_threads()
        if self._log_scratch is None or self._log_scratch.shape[0] != nthreads:
            self._log_scratch = np.empty((nthreads, len(self.resolved.plan.terms)), np.float64)
        self._log_scratch.fill(0.0)
        started = time.perf_counter()
        self.runtime.execute(reward_multiplier=self.ctrl_dt, log_scratch=self._log_scratch)
        if self.first_jit_ms is None:
            self.first_jit_ms = (time.perf_counter() - started) * 1e3

        steps = info.get("steps")
        should_log = enable_log and (
            int(steps[0]) % 4 == 0 if isinstance(steps, np.ndarray) else True
        )
        log = {} if should_log else info.get("log", {})
        if should_log:
            sums = self._log_scratch.sum(axis=0)
            for reward_name, output_name in self._reward_outputs.items():
                if scales.get(reward_name, 0.0) != 0.0:
                    index = self._plan_indices[output_name]
                    log[f"reward/{reward_name}"] = float(sums[index] / self.num_envs)
        return G1MotionTrackingNumbaUpdateStateResult(
            obs=self.obs,
            reward=self.runtime.reward,
            terminated=self.runtime.terminated,
            log=log,
        )
