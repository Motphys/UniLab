from __future__ import annotations

import numpy as np

LEG_JOINT_SENSOR_PREFIXES: tuple[str, ...] = (
    "FR_hip",
    "FR_thigh",
    "FR_calf",
    "FL_hip",
    "FL_thigh",
    "FL_calf",
    "RR_hip",
    "RR_thigh",
    "RR_calf",
    "RL_hip",
    "RL_thigh",
    "RL_calf",
)
WHEEL_JOINT_SENSOR_PREFIXES: tuple[str, ...] = ("FR_wheel", "FL_wheel", "RR_wheel", "RL_wheel")
JOINT_SENSOR_PREFIXES: tuple[str, ...] = LEG_JOINT_SENSOR_PREFIXES + WHEEL_JOINT_SENSOR_PREFIXES

NUM_LEG_ACTIONS = len(LEG_JOINT_SENSOR_PREFIXES)
NUM_WHEEL_ACTIONS = len(WHEEL_JOINT_SENSOR_PREFIXES)
NUM_GO2W_ACTIONS = len(JOINT_SENSOR_PREFIXES)


def compute_go2w_motor_ctrl(
    policy_ctrl: np.ndarray,
    joint_pos: np.ndarray,
    joint_vel: np.ndarray,
    leg_kp: np.ndarray,
    leg_kd: np.ndarray,
    wheel_kd: np.ndarray,
    ctrl_lower: np.ndarray,
    ctrl_upper: np.ndarray,
    out: np.ndarray,
) -> np.ndarray:
    """Convert Go2W owner-level controls into motor actuator torques.

    Hot path: shapes/dtypes are validated by the owning env at init/reset.
    """
    leg_out = out[:, :NUM_LEG_ACTIONS]
    np.subtract(policy_ctrl[:, :NUM_LEG_ACTIONS], joint_pos[:, :NUM_LEG_ACTIONS], out=leg_out)
    np.multiply(leg_out, leg_kp, out=leg_out)
    leg_out -= leg_kd * joint_vel[:, :NUM_LEG_ACTIONS]
    wheel_out = out[:, NUM_LEG_ACTIONS:]
    np.subtract(policy_ctrl[:, NUM_LEG_ACTIONS:], joint_vel[:, NUM_LEG_ACTIONS:], out=wheel_out)
    np.multiply(wheel_out, wheel_kd, out=wheel_out)
    np.clip(out, ctrl_lower, ctrl_upper, out=out)
    return out


__all__ = [
    "JOINT_SENSOR_PREFIXES",
    "LEG_JOINT_SENSOR_PREFIXES",
    "NUM_GO2W_ACTIONS",
    "NUM_LEG_ACTIONS",
    "NUM_WHEEL_ACTIONS",
    "WHEEL_JOINT_SENSOR_PREFIXES",
    "compute_go2w_motor_ctrl",
]
