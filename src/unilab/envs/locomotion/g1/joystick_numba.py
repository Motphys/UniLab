"""Optional Numba hot path for the G1 joystick locomotion task.

This module is task-owned on purpose: it mirrors the reward/termination math in
``joystick.py`` without changing the base env contract or the shared locomotion
reward dispatcher.  Importing this file must be cheap and safe when ``numba`` is
not installed.
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
    "tracking_lin_vel",
    "tracking_ang_vel",
    "forward_progress",
    "under_speed",
    "lin_vel_z",
    "orientation",
    "penalty_orientation",
    "ang_vel_xy",
    "penalty_ang_vel_xy",
    "action_rate",
    "penalty_action_rate",
    "base_height",
    "pose",
    "upper_body_pose",
    "penalty_close_feet_xy",
    "penalty_feet_ori",
    "feet_phase",
    "feet_phase_contrast",
    "feet_phase_contact",
    "feet_double_stance",
    "feet_air_time",
    "alive",
)
TERM_INDEX = {name: i for i, name in enumerate(TERM_ORDER)}
SUPPORTED_TERMS = frozenset(TERM_ORDER)

FOOT_POSITION_TERMS = frozenset(("penalty_close_feet_xy", "feet_phase", "feet_phase_contrast"))
FOOT_QUAT_TERMS = frozenset(("penalty_feet_ori",))
FOOT_CONTACT_TERMS = frozenset(("feet_phase_contact", "feet_double_stance"))
FEET_AIR_TIME_TERMS = frozenset(("feet_air_time",))


@dataclass(frozen=True)
class G1WalkNumbaResult:
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


def _scalar_sensor(sensor_values: np.ndarray) -> np.ndarray:
    arr = np.asarray(sensor_values)
    if arr.ndim == 1:
        return arr
    if arr.ndim == 2 and arr.shape[1] == 1:
        return arr[:, 0]
    raise ValueError(f"Expected scalar sensor values, got shape {arr.shape}")


def _aggregated_contact(backend: Any, sensor_names: tuple[str, ...]) -> np.ndarray:
    contacts = [_scalar_sensor(backend.get_sensor_data(name)) for name in sensor_names]
    return np.asarray(np.any(np.stack(contacts, axis=1) > 0.5, axis=1), dtype=np.bool_)


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
    def weighted_pose_i(dof_pos, default_angles, weights, n_action, i):
        acc = 0.0
        for j in range(n_action):
            d = dof_pos[i, j] - default_angles[j]
            acc += weights[j] * d * d
        return acc

    @_dev
    def base_height_i(base_height, base_height_target, i):
        d = base_height[i] - base_height_target
        return d * d

    @_dev
    def close_feet_xy_i(left_foot_pos, right_foot_pos, close_feet_threshold, i):
        dx = left_foot_pos[i, 0] - right_foot_pos[i, 0]
        dy = left_foot_pos[i, 1] - right_foot_pos[i, 1]
        feet_dist = math.sqrt(dx * dx + dy * dy)
        if feet_dist >= close_feet_threshold:
            return 0.0
        d = feet_dist - close_feet_threshold
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
    def feet_air_time_i(feet_air_time, i):
        acc = 0.0
        if feet_air_time[i, 0] > 0.05 and feet_air_time[i, 0] < 0.5:
            acc += 1.0
        if feet_air_time[i, 1] > 0.05 and feet_air_time[i, 1] < 0.5:
            acc += 1.0
        return acc

    @_dev
    def terminated_i(gravity, base_height, max_tilt_rad, min_base_height, i):
        gz = gravity[i, 2]
        if gz < -1.0:
            gz = -1.0
        elif gz > 1.0:
            gz = 1.0
        return math.acos(gz) > max_tilt_rad or base_height[i] < min_base_height

    @njit(parallel=True, fastmath=True, cache=True, nogil=True)  # type: ignore[misc]
    def _compute_reward_termination_kernel(
        linvel,
        gyro,
        gravity,
        dof_pos,
        dof_vel,
        base_height,
        commands,
        current_actions,
        last_actions,
        gait_phase,
        default_angles,
        pose_weights,
        upper_body_pose_weights,
        left_foot_pos,
        right_foot_pos,
        left_foot_quat,
        right_foot_quat,
        left_contact,
        right_contact,
        feet_air_time,
        scale,
        ctrl_dt,
        tracking_sigma,
        base_height_target,
        min_base_height,
        max_tilt_rad,
        feet_phase_swing_height,
        feet_phase_tracking_sigma,
        min_forward_speed_for_gait_reward,
        close_feet_threshold,
        reward,
        terminated,
        log_scratch,
    ):
        n = reward.shape[0]
        n_action = dof_pos.shape[1]
        for i in prange(n):
            tid = get_thread_id()
            r = 0.0

            w = tracking_lin_vel_i(linvel, commands, tracking_sigma, i) * scale[0]
            r += w
            log_scratch[tid, 0] += w

            w = tracking_ang_vel_i(gyro, commands, tracking_sigma, i) * scale[1]
            r += w
            log_scratch[tid, 1] += w

            w = forward_progress_i(linvel, commands, i) * scale[2]
            r += w
            log_scratch[tid, 2] += w

            w = under_speed_i(linvel, commands, i) * scale[3]
            r += w
            log_scratch[tid, 3] += w

            w = lin_vel_z_i(linvel, i) * scale[4]
            r += w
            log_scratch[tid, 4] += w

            orientation = orientation_i(gravity, i)
            w = orientation * scale[5]
            r += w
            log_scratch[tid, 5] += w
            w = orientation * scale[6]
            r += w
            log_scratch[tid, 6] += w

            ang_vel_xy = ang_vel_xy_i(gyro, i)
            w = ang_vel_xy * scale[7]
            r += w
            log_scratch[tid, 7] += w
            w = ang_vel_xy * scale[8]
            r += w
            log_scratch[tid, 8] += w

            action_rate = action_rate_i(current_actions, last_actions, n_action, i)
            w = action_rate * scale[9]
            r += w
            log_scratch[tid, 9] += w
            w = action_rate * scale[10]
            r += w
            log_scratch[tid, 10] += w

            w = base_height_i(base_height, base_height_target, i) * scale[11]
            r += w
            log_scratch[tid, 11] += w

            pose = weighted_pose_i(dof_pos, default_angles, pose_weights, n_action, i)
            w = pose * scale[12]
            r += w
            log_scratch[tid, 12] += w
            upper_body_pose = weighted_pose_i(
                dof_pos, default_angles, upper_body_pose_weights, n_action, i
            )
            w = upper_body_pose * scale[13]
            r += w
            log_scratch[tid, 13] += w

            w = close_feet_xy_i(left_foot_pos, right_foot_pos, close_feet_threshold, i) * scale[14]
            r += w
            log_scratch[tid, 14] += w

            w = feet_ori_i(left_foot_quat, right_foot_quat, i) * scale[15]
            r += w
            log_scratch[tid, 15] += w

            w = (
                feet_phase_i(
                    linvel,
                    gait_phase,
                    left_foot_pos,
                    right_foot_pos,
                    feet_phase_swing_height,
                    feet_phase_tracking_sigma,
                    min_forward_speed_for_gait_reward,
                    i,
                )
                * scale[16]
            )
            r += w
            log_scratch[tid, 16] += w

            w = (
                feet_phase_contrast_i(
                    linvel,
                    gait_phase,
                    left_foot_pos,
                    right_foot_pos,
                    feet_phase_swing_height,
                    feet_phase_tracking_sigma,
                    min_forward_speed_for_gait_reward,
                    i,
                )
                * scale[17]
            )
            r += w
            log_scratch[tid, 17] += w

            w = (
                feet_phase_contact_i(
                    linvel,
                    gait_phase,
                    left_contact,
                    right_contact,
                    feet_phase_swing_height,
                    min_forward_speed_for_gait_reward,
                    i,
                )
                * scale[18]
            )
            r += w
            log_scratch[tid, 18] += w

            w = feet_double_stance_i(commands, left_contact, right_contact, i) * scale[19]
            r += w
            log_scratch[tid, 19] += w

            w = feet_air_time_i(feet_air_time, i) * scale[20]
            r += w
            log_scratch[tid, 20] += w

            w = scale[21]
            r += w
            log_scratch[tid, 21] += w

            reward[i] = r * ctrl_dt
            terminated[i] = terminated_i(gravity, base_height, max_tilt_rad, min_base_height, i)


class G1WalkNumbaAccelerator:
    """Driver that keeps config-derived arrays and calls the fused kernel."""

    def __init__(
        self,
        *,
        num_envs: int,
        num_action: int,
        ctrl_dt: float,
        tracking_sigma: float,
        base_height_target: float,
        min_base_height: float,
        max_tilt_deg: float,
        feet_phase_swing_height: float,
        feet_phase_tracking_sigma: float,
        min_forward_speed_for_gait_reward: float,
        close_feet_threshold: float,
        default_angles: np.ndarray,
        pose_weights: np.ndarray,
        upper_body_pose_weights: np.ndarray,
        num_threads: int | None = None,
    ) -> None:
        self.num_envs = int(num_envs)
        self.num_action = int(num_action)
        self.ctrl_dt = float(ctrl_dt)
        self.tracking_sigma = float(tracking_sigma)
        self.base_height_target = float(base_height_target)
        self.min_base_height = float(min_base_height)
        self.max_tilt_rad = float(np.deg2rad(max_tilt_deg))
        self.feet_phase_swing_height = float(feet_phase_swing_height)
        self.feet_phase_tracking_sigma = float(feet_phase_tracking_sigma)
        self.min_forward_speed_for_gait_reward = float(min_forward_speed_for_gait_reward)
        self.close_feet_threshold = float(close_feet_threshold)
        self.default_angles = np.asarray(default_angles, dtype=np.float64)
        self.pose_weights = np.asarray(pose_weights, dtype=np.float64)
        self.upper_body_pose_weights = np.asarray(upper_body_pose_weights, dtype=np.float64)
        self.num_threads = num_threads
        self.scale = np.zeros((len(TERM_ORDER),), dtype=np.float64)
        self._zero_vec2 = np.zeros((self.num_envs, 2), dtype=np.float64)
        self._zero_vec3 = np.zeros((self.num_envs, 3), dtype=np.float64)
        self._zero_vec4 = np.zeros((self.num_envs, 4), dtype=np.float64)
        self._zero_bool = np.zeros((self.num_envs,), dtype=np.bool_)

    @classmethod
    def from_env(cls, env: Any, num_threads: int | None = None) -> "G1WalkNumbaAccelerator":
        if not NUMBA_AVAILABLE:
            raise RuntimeError(
                "G1Walk numba_acceleration=True requires numba. Install it or run through "
                "`uv run --with numba ...`; disable numba_acceleration to use the numpy path."
            )
        return cls(
            num_envs=env.num_envs,
            num_action=env._num_action,
            ctrl_dt=env._cfg.ctrl_dt,
            tracking_sigma=env._reward_cfg.tracking_sigma,
            base_height_target=env._reward_cfg.base_height_target,
            min_base_height=env._reward_cfg.min_base_height,
            max_tilt_deg=env._reward_cfg.max_tilt_deg,
            feet_phase_swing_height=env._reward_cfg.feet_phase_swing_height,
            feet_phase_tracking_sigma=env._reward_cfg.feet_phase_tracking_sigma,
            min_forward_speed_for_gait_reward=getattr(
                env._reward_cfg, "min_forward_speed_for_gait_reward", 0.0
            ),
            close_feet_threshold=getattr(env._reward_cfg, "close_feet_threshold", 0.15),
            default_angles=env.default_angles,
            pose_weights=env._pose_weights,
            upper_body_pose_weights=env._upper_body_pose_weights,
            num_threads=num_threads,
        )

    def _sync_scales(self, scales: Mapping[str, float]) -> None:
        unsupported = unsupported_terms(scales)
        if unsupported:
            raise ValueError(
                "G1Walk Numba accelerator does not support active reward terms "
                f"{sorted(unsupported)}. Disable numba_acceleration or add these terms "
                "to src/unilab/envs/locomotion/g1/joystick_numba.py."
            )
        self.scale.fill(0.0)
        for name, value in scales.items():
            idx = TERM_INDEX.get(name)
            if idx is not None:
                self.scale[idx] = float(value)

    def compute(
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
    ) -> G1WalkNumbaResult:
        if not NUMBA_AVAILABLE:
            raise RuntimeError(
                "G1Walk Numba accelerator was constructed while numba is unavailable; "
                "this indicates an invalid accelerator state."
            )
        self._sync_scales(scales)

        active = _active_terms(scales)
        backend = env._backend
        dtype = linvel.dtype
        base_height = np.asarray(backend.get_base_pos()[:, 2], dtype=dtype)
        commands = np.asarray(info["commands"], dtype=dtype)
        current_actions = np.asarray(
            info.get("current_actions", np.zeros((self.num_envs, self.num_action), dtype=dtype)),
            dtype=dtype,
        )
        last_actions = np.asarray(
            info.get("last_actions", np.zeros((self.num_envs, self.num_action), dtype=dtype)),
            dtype=dtype,
        )
        gait_phase = np.asarray(info.get("gait_phase", self._zero_vec2), dtype=dtype)
        feet_air_time = np.asarray(info.get("feet_air_time", self._zero_vec2), dtype=dtype)

        if active & FOOT_POSITION_TERMS:
            left_foot_pos = np.asarray(backend.get_sensor_data("left_foot_pos"), dtype=dtype)
            right_foot_pos = np.asarray(backend.get_sensor_data("right_foot_pos"), dtype=dtype)
        else:
            left_foot_pos = right_foot_pos = self._zero_vec3

        if active & FOOT_QUAT_TERMS:
            left_foot_quat = np.asarray(backend.get_sensor_data("left_foot_quat"), dtype=dtype)
            right_foot_quat = np.asarray(backend.get_sensor_data("right_foot_quat"), dtype=dtype)
        else:
            left_foot_quat = right_foot_quat = self._zero_vec4

        if active & FOOT_CONTACT_TERMS:
            left_contact = _aggregated_contact(
                backend,
                (
                    "left_foot_contact_0",
                    "left_foot_contact_1",
                    "left_foot_contact_2",
                    "left_foot_contact_3",
                ),
            )
            right_contact = _aggregated_contact(
                backend,
                (
                    "right_foot_contact_0",
                    "right_foot_contact_1",
                    "right_foot_contact_2",
                    "right_foot_contact_3",
                ),
            )
        else:
            left_contact = right_contact = self._zero_bool

        if self.num_threads is not None:
            set_num_threads(self.num_threads)
        nthreads = get_num_threads()
        reward = np.empty((linvel.shape[0],), dtype=dtype)
        terminated = np.empty((linvel.shape[0],), dtype=np.bool_)
        log_scratch = np.zeros((nthreads, len(TERM_ORDER)), dtype=np.float64)

        _compute_reward_termination_kernel(
            linvel,
            gyro,
            gravity,
            dof_pos,
            dof_vel,
            base_height,
            commands,
            current_actions,
            last_actions,
            gait_phase,
            self.default_angles,
            self.pose_weights,
            self.upper_body_pose_weights,
            left_foot_pos,
            right_foot_pos,
            left_foot_quat,
            right_foot_quat,
            left_contact,
            right_contact,
            feet_air_time,
            self.scale,
            self.ctrl_dt,
            self.tracking_sigma,
            self.base_height_target,
            self.min_base_height,
            self.max_tilt_rad,
            self.feet_phase_swing_height,
            self.feet_phase_tracking_sigma,
            self.min_forward_speed_for_gait_reward,
            self.close_feet_threshold,
            reward,
            terminated,
            log_scratch,
        )

        step_count = info.get("steps", np.zeros((linvel.shape[0],), dtype=np.uint32))
        should_log = enable_log and int(step_count[0]) % 4 == 0
        log = {} if should_log else info.get("log", {})
        if should_log:
            term_sums = log_scratch.sum(axis=0)
            for idx, name in enumerate(TERM_ORDER):
                if self.scale[idx] != 0.0:
                    log[f"reward/{name}"] = float(term_sums[idx] / linvel.shape[0])
        return G1WalkNumbaResult(reward=reward, terminated=terminated, log=log)
