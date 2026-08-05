"""Optional Numba hot path for the G1 joystick locomotion task.

This module is task-owned on purpose: it mirrors the reward/termination math in
``joystick.py`` without changing the base env contract or the shared locomotion
reward dispatcher.  Importing this file must be cheap and safe when ``numba`` is
not installed.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from unilab.dtype_config import get_global_dtype

try:  # pragma: no cover - exercised in environments with numba installed
    from numba import get_num_threads, njit, set_num_threads

    NUMBA_AVAILABLE = True
except Exception:  # pragma: no cover - default test env may not install numba
    get_num_threads = njit = set_num_threads = None  # type: ignore[assignment]
    NUMBA_AVAILABLE = False


NUMBA_REWARD_ITEMS: dict[str, Any] = {}
NUMBA_OBSERVATION_ITEM = None
NUMBA_TERMINATION_ITEM = None


@dataclass(frozen=True)
class G1WalkNumbaResult:
    reward: np.ndarray
    terminated: np.ndarray
    log: dict[str, float]


@dataclass(frozen=True)
class G1WalkNumbaUpdateStateResult:
    obs: dict[str, np.ndarray]
    reward: np.ndarray
    terminated: np.ndarray
    log: dict[str, float]


def _active_terms(scales: Mapping[str, float]) -> frozenset[str]:
    return frozenset(name for name, scale in scales.items() if scale != 0.0)


def unsupported_terms(scales: Mapping[str, float]) -> frozenset[str]:
    """Return nonzero reward terms this task-specific kernel cannot compute."""
    from unilab.envs.locomotion.g1.terms import REWARD_TERM_KEYS

    return _active_terms(scales) - REWARD_TERM_KEYS.keys()


def is_available(scales: Mapping[str, float]) -> bool:
    return NUMBA_AVAILABLE and not unsupported_terms(scales)


if NUMBA_AVAILABLE:

    def _dev(fn):
        return njit(inline="always", fastmath=True, cache=True, nogil=True)(fn)

    @_dev
    def _positive(x):
        return x if x > 0.0 else 0.0

    @_dev
    def tracking_lin_vel_i(linvel, commands, tracking_sigma, i):
        dx = commands[i, 0] - linvel[i, 0]
        dy = commands[i, 1] - linvel[i, 1]
        return math.exp(-(dx * dx + dy * dy) / tracking_sigma)

    @_dev
    def tracking_ang_vel_i(gyro, commands, tracking_sigma, i):
        dz = commands[i, 2] - gyro[i, 2]
        return math.exp(-(dz * dz) / tracking_sigma)

    @_dev
    def forward_progress_i(linvel, commands, i):
        commanded_speed = commands[i, 0] if commands[i, 0] > 1.0e-6 else 1.0e-6
        progress = _positive(linvel[i, 0]) / commanded_speed
        return 1.0 if progress > 1.0 else progress

    @_dev
    def under_speed_i(linvel, commands, i):
        commanded_speed = commands[i, 0] if commands[i, 0] > 1.0e-6 else 1.0e-6
        gap = commands[i, 0] - _positive(linvel[i, 0])
        return _positive(gap) / commanded_speed

    @_dev
    def lin_vel_z_i(linvel, i):
        return linvel[i, 2] * linvel[i, 2]

    @_dev
    def orientation_i(gravity, i):
        return gravity[i, 0] * gravity[i, 0] + gravity[i, 1] * gravity[i, 1]

    @_dev
    def ang_vel_xy_i(gyro, i):
        return gyro[i, 0] * gyro[i, 0] + gyro[i, 1] * gyro[i, 1]

    @_dev
    def action_rate_i(current_actions, last_actions, n_action, i):
        acc = 0.0
        for j in range(n_action):
            d = current_actions[i, j] - last_actions[i, j]
            acc += d * d
        return acc

    @_dev
    def base_height_i(base_height, base_height_target, i):
        d = base_height[i] - base_height_target
        return d * d

    @_dev
    def feet_ori_i(left_foot_quat, right_foot_quat, i):
        return (
            left_foot_quat[i, 1] * left_foot_quat[i, 1]
            + left_foot_quat[i, 2] * left_foot_quat[i, 2]
            + right_foot_quat[i, 1] * right_foot_quat[i, 1]
            + right_foot_quat[i, 2] * right_foot_quat[i, 2]
        )

    @_dev
    def _bezier_height(phi, swing_height):
        phi_normalized = (phi + math.pi) % (2.0 * math.pi) - math.pi
        x = (phi_normalized + math.pi) / (2.0 * math.pi)
        if x <= 0.5:
            t = 2.0 * x
            bezier = t * t * t + 3.0 * (t * t * (1.0 - t))
            return swing_height * bezier
        t = 2.0 * x - 1.0
        bezier = t * t * t + 3.0 * (t * t * (1.0 - t))
        return swing_height * (1.0 - bezier)

    @_dev
    def _gait_gate(linvel, min_forward_speed_for_gait_reward, i):
        return 1.0 if _positive(linvel[i, 0]) >= min_forward_speed_for_gait_reward else 0.0

    @_dev
    def feet_phase_i(
        linvel,
        gait_phase,
        left_foot_pos,
        right_foot_pos,
        swing_height,
        tracking_sigma,
        min_forward_speed_for_gait_reward,
        i,
    ):
        left_target = _bezier_height(gait_phase[i, 0], swing_height)
        right_target = _bezier_height(gait_phase[i, 1], swing_height)
        left_err = left_foot_pos[i, 2] - left_target
        right_err = right_foot_pos[i, 2] - right_target
        reward = math.exp(-((left_err * left_err) + (right_err * right_err)) / tracking_sigma)
        return reward * _gait_gate(linvel, min_forward_speed_for_gait_reward, i)

    @_dev
    def feet_phase_contrast_i(
        linvel,
        gait_phase,
        left_foot_pos,
        right_foot_pos,
        swing_height,
        tracking_sigma,
        min_forward_speed_for_gait_reward,
        i,
    ):
        left_target = _bezier_height(gait_phase[i, 0], swing_height)
        right_target = _bezier_height(gait_phase[i, 1], swing_height)
        actual_delta = left_foot_pos[i, 2] - right_foot_pos[i, 2]
        target_delta = left_target - right_target
        err = actual_delta - target_delta
        reward = math.exp(-(err * err) / tracking_sigma)
        return reward * _gait_gate(linvel, min_forward_speed_for_gait_reward, i)

    @_dev
    def feet_phase_contact_i(
        linvel,
        gait_phase,
        left_contact,
        right_contact,
        swing_height,
        min_forward_speed_for_gait_reward,
        i,
    ):
        contact_height_threshold = swing_height * 0.5
        left_target_contact = (
            _bezier_height(gait_phase[i, 0], swing_height) <= contact_height_threshold
        )
        right_target_contact = (
            _bezier_height(gait_phase[i, 1], swing_height) <= contact_height_threshold
        )
        left_match = 1.0 if left_contact[i] == left_target_contact else 0.0
        right_match = 1.0 if right_contact[i] == right_target_contact else 0.0
        return (
            0.5
            * (left_match + right_match)
            * _gait_gate(linvel, min_forward_speed_for_gait_reward, i)
        )

    @_dev
    def feet_double_stance_i(commands, left_contact, right_contact, i):
        forward_command = 1.0 if _positive(commands[i, 0]) > 1.0e-6 else 0.0
        double_stance = 1.0 if left_contact[i] and right_contact[i] else 0.0
        return double_stance * forward_command

    @_dev
    def terminated_i(gravity, base_height, max_tilt_rad, min_base_height, i):
        gz = gravity[i, 2]
        if gz < -1.0:
            gz = -1.0
        elif gz > 1.0:
            gz = 1.0
        return math.acos(gz) > max_tilt_rad or base_height[i] < min_base_height

    @_dev
    def weighted_pose_plan_i(dof_pos, default_angles, weights, i):
        acc = 0.0
        for j in range(dof_pos.shape[1]):
            d = dof_pos[i, j] - default_angles[i, j]
            acc += weights[i, j] * d * d
        return acc

    @_dev
    def action_rate_plan_i(current_actions, last_actions, i):
        return action_rate_i(current_actions, last_actions, current_actions.shape[1], i)

    @_dev
    def alive_i(i):
        return 1.0

    @_dev
    def scaled_copy_i(source, multiplier, output, offset, i):
        for j in range(source.shape[1]):
            output[i, offset + j] = source[i, j] * multiplier

    NUMBA_REWARD_ITEMS = {
        "tracking_lin_vel": tracking_lin_vel_i,
        "tracking_ang_vel": tracking_ang_vel_i,
        "forward_progress": forward_progress_i,
        "under_speed": under_speed_i,
        "lin_vel_z": lin_vel_z_i,
        "orientation": orientation_i,
        "ang_vel_xy": ang_vel_xy_i,
        "action_rate": action_rate_plan_i,
        "base_height": base_height_i,
        "pose": weighted_pose_plan_i,
        "upper_body_pose": weighted_pose_plan_i,
        "feet_ori": feet_ori_i,
        "feet_phase": feet_phase_i,
        "feet_phase_contrast": feet_phase_contrast_i,
        "feet_phase_contact": feet_phase_contact_i,
        "feet_double_stance": feet_double_stance_i,
        "alive": alive_i,
    }
    NUMBA_OBSERVATION_ITEM = scaled_copy_i
    NUMBA_TERMINATION_ITEM = terminated_i


class G1WalkNumbaAccelerator:
    """Resolved-plan Numba driver for the G1WalkFlat pilot."""

    def __init__(self, env: Any, *, num_threads: int | None = None) -> None:
        from unilab.envs.locomotion.g1.terms import resolve_g1_walk_term_plan
        from unilab.term.numba import FusedOutputLayout, materialize_numba_plan

        config = env._cfg.term_plan
        if config is None:
            raise ValueError(
                "G1Walk Numba fused execution requires env.term_plan; disable "
                "numba_acceleration for tasks that retain the legacy NumPy path"
            )
        resolved = resolve_g1_walk_term_plan(
            num_action=env._num_action,
            reward_cfg=env._reward_cfg,
            observations=config.observations,
            terminations=config.terminations,
            walk_profile=env._uses_walk_observation_profile(),
        )
        inputs = {
            name: np.zeros((env._num_envs, *spec.shape), dtype=spec.numpy_dtype)
            for name, spec in resolved.plan.input_specs.items()
        }
        for name, value in (
            ("default_angles", env.default_angles),
            ("pose_weights", env._pose_weights),
            ("upper_body_pose_weights", env._upper_body_pose_weights),
        ):
            if name in inputs:
                np.copyto(inputs[name], np.broadcast_to(value, inputs[name].shape))
        observations = {
            group: np.empty((env._num_envs, width), dtype=get_global_dtype())
            for group, width in resolved.observation_dims.items()
        }
        reward = np.empty((env._num_envs,), dtype=get_global_dtype())
        terminated = np.empty((env._num_envs,), dtype=np.bool_)
        layout = FusedOutputLayout(
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
            reward=reward,
            terminated=terminated,
        )
        self.resolved = resolved
        self.inputs = inputs
        self._owned_inputs = inputs.copy()
        self.obs = observations
        self.num_envs = env._num_envs
        self.num_threads = num_threads
        self.ctrl_dt = float(env._cfg.ctrl_dt)
        self.first_jit_ms: float | None = None
        self._reward_outputs = dict(resolved.reward_outputs)
        self._plan_indices = {term.name: index for index, term in enumerate(resolved.plan.terms)}
        self._log_scratch: np.ndarray | None = None

    @property
    def compile_info(self):
        return self.runtime.compile_info

    @classmethod
    def from_env(cls, env: Any, num_threads: int | None = None) -> "G1WalkNumbaAccelerator":
        if not NUMBA_AVAILABLE:
            raise RuntimeError(
                "G1Walk numba_acceleration=True requires numba. Install it or disable "
                "numba_acceleration to use the NumPy path."
            )
        return cls(env, num_threads=num_threads)

    def _sync_scales(self, scales: Mapping[str, float]) -> None:
        unsupported = _active_terms(scales) - self._reward_outputs.keys()
        if unsupported:
            raise ValueError(
                f"G1Walk Numba plan does not support active reward terms {sorted(unsupported)}"
            )
        for reward_name, output_name in self._reward_outputs.items():
            self.runtime.set_scale(output_name, scales.get(reward_name, 0.0))

    def _bind_inputs(
        self,
        env: Any,
        info: dict[str, Any],
        linvel: np.ndarray,
        gyro: np.ndarray,
        gravity: np.ndarray,
        dof_pos: np.ndarray,
        dof_vel: np.ndarray,
    ) -> None:
        def bind(name: str, value: np.ndarray) -> None:
            if name in self.inputs:
                self.inputs[name] = np.asarray(value)

        for name, value in (
            ("linvel", linvel),
            ("gyro", gyro),
            ("gravity", gravity),
            ("dof_pos", dof_pos),
            ("dof_vel", dof_vel),
        ):
            bind(name, value)
        for name in ("commands", "current_actions", "last_actions", "gait_phase"):
            if name not in self.inputs:
                continue
            value = info.get(name)
            if value is None:
                value = self._owned_inputs[name]
                value.fill(0)
            bind(name, value)

        if "dof_pos_diff" in self.inputs:
            target = self._owned_inputs["dof_pos_diff"]
            np.subtract(dof_pos, env.default_angles, out=target)
            bind("dof_pos_diff", target)
        noisy_dof_pos = (
            self.inputs["dof_pos_diff"]
            if "dof_pos_diff" in self.inputs
            else dof_pos - env.default_angles
        )
        noisy_sources = (
            ("noisy_gyro", gyro, env._cfg.noise_config.scale_gyro),
            ("noisy_gravity", gravity, env._cfg.noise_config.scale_gravity),
            (
                "noisy_dof_pos",
                noisy_dof_pos,
                env._cfg.noise_config.scale_joint_angle,
            ),
            ("noisy_dof_vel", dof_vel, env._cfg.noise_config.scale_joint_vel),
        )
        for name, value, scale in noisy_sources:
            if name in self.inputs:
                bind(name, env._obs_noise(value, scale))

        if "base_height" in self.inputs:
            bind("base_height", env._backend.get_base_pos()[:, 2])
        for name in ("left_foot_pos", "right_foot_pos", "left_foot_quat", "right_foot_quat"):
            if name in self.inputs:
                bind(name, env._backend.get_sensor_data(name))
        if "left_contact" in self.inputs or "right_contact" in self.inputs:
            from unilab.envs.locomotion.g1.joystick import (
                LEFT_FOOT_CONTACT_SENSORS,
                RIGHT_FOOT_CONTACT_SENSORS,
                compute_aggregated_foot_contact,
            )

            bind(
                "left_contact",
                compute_aggregated_foot_contact(env._backend, LEFT_FOOT_CONTACT_SENSORS),
            )
            bind(
                "right_contact",
                compute_aggregated_foot_contact(env._backend, RIGHT_FOOT_CONTACT_SENSORS),
            )
        self.runtime.bind_inputs(self.inputs)

    def compute_update_state(
        self,
        *,
        env: Any,
        info: dict[str, Any],
        linvel: np.ndarray,
        gyro: np.ndarray,
        gravity: np.ndarray,
        dof_pos: np.ndarray,
        dof_vel: np.ndarray,
        scales: Mapping[str, float],
        enable_log: bool,
        noise_level: float,
    ) -> G1WalkNumbaUpdateStateResult:
        del noise_level  # RNG and noise application remain owned by the task input binder.
        if linvel.shape[0] != self.num_envs:
            raise ValueError(
                "G1Walk Numba update_state only supports full-batch updates; "
                f"got {linvel.shape[0]} rows for configured num_envs={self.num_envs}."
            )
        self._sync_scales(scales)
        self._bind_inputs(env, info, linvel, gyro, gravity, dof_pos, dof_vel)
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
                index = self._plan_indices[output_name]
                if self.runtime.scales[index] != 0.0:
                    log[f"reward/{reward_name}"] = float(sums[index] / self.num_envs)
        return G1WalkNumbaUpdateStateResult(
            obs=self.obs,
            reward=self.runtime.reward,
            terminated=self.runtime.terminated,
            log=log,
        )

    def compute(self, **kwargs: Any) -> G1WalkNumbaResult:
        out = self.compute_update_state(noise_level=0.0, **kwargs)
        return G1WalkNumbaResult(out.reward, out.terminated, out.log)
