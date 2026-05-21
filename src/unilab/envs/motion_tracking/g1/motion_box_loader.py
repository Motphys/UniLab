"""Motion loading with object state support for box tracking tasks."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .motion_loader import MotionData, MotionLoader


@dataclass
class BoxMotionData(MotionData):
    """Motion data with optional object state."""

    object_pos_w: np.ndarray | None = None
    object_quat_w: np.ndarray | None = None
    object_lin_vel_w: np.ndarray | None = None
    object_ang_vel_w: np.ndarray | None = None


class BoxMotionLoader(MotionLoader):
    """Motion loader that also loads object state from NPZ files."""

    def __init__(self, motion_file, body_indices=None):
        super().__init__(motion_file, body_indices)

        object_pos_list: list[np.ndarray] = []
        object_quat_list: list[np.ndarray] = []
        object_lin_vel_list: list[np.ndarray] = []
        object_ang_vel_list: list[np.ndarray] = []
        has_object = False

        for clip_idx, motion_path in enumerate(self.motion_files):
            with np.load(motion_path) as data:
                if "object_pos_w" in data:
                    obj_pos = data["object_pos_w"].astype(np.float32)
                    obj_quat = data["object_quat_w"].astype(np.float32)
                    obj_lin_vel = data["object_lin_vel_w"].astype(np.float32)
                    obj_ang_vel = data["object_ang_vel_w"].astype(np.float32)

                    if clip_idx == 0:
                        has_object = True
                        self._obj_pos_dim = obj_pos.shape[1]
                    elif obj_pos.shape[1] != self._obj_pos_dim:
                        raise ValueError(
                            f"Motion file '{motion_path}' has incompatible object position dimensions"
                        )

                    object_pos_list.append(obj_pos)
                    object_quat_list.append(obj_quat)
                    object_lin_vel_list.append(obj_lin_vel)
                    object_ang_vel_list.append(obj_ang_vel)
                elif has_object:
                    raise ValueError(
                        f"Motion file '{motion_path}' is missing object data "
                        "but earlier clips had it; all clips must be consistent"
                    )

        self.has_object = has_object
        if has_object:
            self.object_pos_w = np.concatenate(object_pos_list, axis=0)
            self.object_quat_w = np.concatenate(object_quat_list, axis=0)
            self.object_lin_vel_w = np.concatenate(object_lin_vel_list, axis=0)
            self.object_ang_vel_w = np.concatenate(object_ang_vel_list, axis=0)

            with np.load(self.motion_files[0]) as data:
                if "joint_names" in data:
                    n_robot_joints = len(data["joint_names"])
                else:
                    n_robot_joints = self.joint_pos.shape[1] - 7

            self.num_joints = n_robot_joints
            self.joint_pos = self.joint_pos[:, :n_robot_joints]
            self.joint_vel = self.joint_vel[:, :n_robot_joints]

    def get_motion_at_frame(self, frame_idx: np.ndarray) -> BoxMotionData:
        base = super().get_motion_at_frame(frame_idx)
        if not self.has_object:
            return BoxMotionData(
                joint_pos=base.joint_pos,
                joint_vel=base.joint_vel,
                body_pos_w=base.body_pos_w,
                body_quat_w=base.body_quat_w,
                body_lin_vel_w=base.body_lin_vel_w,
                body_ang_vel_w=base.body_ang_vel_w,
            )
        return BoxMotionData(
            joint_pos=base.joint_pos,
            joint_vel=base.joint_vel,
            body_pos_w=base.body_pos_w,
            body_quat_w=base.body_quat_w,
            body_lin_vel_w=base.body_lin_vel_w,
            body_ang_vel_w=base.body_ang_vel_w,
            object_pos_w=self.object_pos_w[frame_idx],
            object_quat_w=self.object_quat_w[frame_idx],
            object_lin_vel_w=self.object_lin_vel_w[frame_idx],
            object_ang_vel_w=self.object_ang_vel_w[frame_idx],
        )
