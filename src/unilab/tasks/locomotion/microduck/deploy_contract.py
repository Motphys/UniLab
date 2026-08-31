"""MicroDuck observation contract for sim2real and ONNX export.

The segment order matches pollen-robotics/microduck_rl and the onboard
MicroDuck runtime, allowing exported policies to keep a stable I/O contract.
"""

from __future__ import annotations

MICRODUCK_ACTOR_OBS_DIM = 61
MICRODUCK_CRITIC_OBS_DIM = 64
MICRODUCK_NUM_ACTION = 14

MICRODUCK_OBS_SEGMENTS: tuple[tuple[str, int], ...] = (
    ("gyro", 3),
    ("gravity", 3),
    ("joint_pos", MICRODUCK_NUM_ACTION),
    ("joint_vel", MICRODUCK_NUM_ACTION),
    ("last_action", MICRODUCK_NUM_ACTION),
    ("twist_cmd", 3),
    ("head_pose_cmd", 4),
    ("body_pose_cmd", 6),
)

MICRODUCK_HEAD_JOINT_INDICES: tuple[int, ...] = (5, 6, 7, 8)
MICRODUCK_LEG_JOINT_INDICES: tuple[int, ...] = (0, 1, 2, 3, 4, 9, 10, 11, 12, 13)
