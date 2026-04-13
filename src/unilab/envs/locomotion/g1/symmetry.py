"""MuJoCo-only G1 symmetry augmentation helpers.

This module builds symmetry mappings from MuJoCo actuator names and therefore
only applies to MuJoCo-backed G1 tooling.
"""

import torch


class G1SymmetryAugmentation:
    """MuJoCo-only symmetry helper derived from MuJoCo actuator ordering."""

    def __init__(self, model, obs_structure: dict, device: str = "cuda"):
        import mujoco

        # Build joint mapping
        actuator_names = [
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i) for i in range(model.nu)
        ]
        symmetry_pairs = {
            "left_hip_pitch_joint": "right_hip_pitch_joint",
            "left_hip_roll_joint": "right_hip_roll_joint",
            "left_hip_yaw_joint": "right_hip_yaw_joint",
            "left_knee_joint": "right_knee_joint",
            "left_ankle_pitch_joint": "right_ankle_pitch_joint",
            "left_ankle_roll_joint": "right_ankle_roll_joint",
            "left_shoulder_pitch_joint": "right_shoulder_pitch_joint",
            "left_shoulder_roll_joint": "right_shoulder_roll_joint",
            "left_shoulder_yaw_joint": "right_shoulder_yaw_joint",
            "left_elbow_joint": "right_elbow_joint",
            "left_wrist_roll_joint": "right_wrist_roll_joint",
            "left_wrist_pitch_joint": "right_wrist_pitch_joint",
            "left_wrist_yaw_joint": "right_wrist_yaw_joint",
        }
        name_to_idx = {name: i for i, name in enumerate(actuator_names)}
        joint_map = {}
        for left, right in symmetry_pairs.items():
            if left in name_to_idx and right in name_to_idx:
                joint_map[name_to_idx[left]] = name_to_idx[right]
                joint_map[name_to_idx[right]] = name_to_idx[left]
        for i in range(len(actuator_names)):
            if i not in joint_map:
                joint_map[i] = i
        self.joint_map = torch.tensor(
            [joint_map[i] for i in range(len(actuator_names))], device=device, dtype=torch.long
        )

        # Sign flip: which joints need negation after mirroring
        flip_names = {"roll", "yaw"}
        sign_mask = [1.0] * len(actuator_names)
        for i, name in enumerate(actuator_names):
            if any(flip in name for flip in flip_names):
                sign_mask[i] = -1.0
        self.sign_mask = torch.tensor(sign_mask, device=device)

        # Precompute obs flip mask and joint mapping indices
        idx = 0
        obs_dim = sum(
            obs_structure.get(k, 0)
            for k in [
                "linvel",
                "gyro",
                "gravity",
                "dof_pos",
                "dof_vel",
                "actions",
                "command",
                "gait_phase",
            ]
        )
        self.obs_flip_mask = torch.ones(obs_dim, device=device)
        self.obs_joint_map = torch.arange(obs_dim, device=device, dtype=torch.long)
        self.obs_joint_sign = torch.ones(obs_dim, device=device)

        for key in [
            "linvel",
            "gyro",
            "gravity",
            "dof_pos",
            "dof_vel",
            "actions",
            "command",
            "gait_phase",
        ]:
            if key not in obs_structure:
                continue
            dim = obs_structure[key]
            if key == "linvel":
                self.obs_flip_mask[idx + 1] = -1.0
            elif key == "gyro":
                self.obs_flip_mask[idx] = -1.0
                self.obs_flip_mask[idx + 2] = -1.0
            elif key == "gravity":
                self.obs_flip_mask[idx + 1] = -1.0
            elif key in ["dof_pos", "dof_vel", "actions"]:
                self.obs_joint_map[idx : idx + dim] = self.joint_map + idx
                self.obs_joint_sign[idx : idx + dim] = self.sign_mask
            elif key == "command":
                self.obs_flip_mask[idx + 1] = -1.0
                self.obs_flip_mask[idx + 2] = -1.0
            elif key == "gait_phase":
                # Swap left and right gait phases
                self.obs_joint_map[idx] = idx + 1
                self.obs_joint_map[idx + 1] = idx
            idx += dim

    def mirror_action(self, action: torch.Tensor) -> torch.Tensor:
        return action[..., self.joint_map] * self.sign_mask

    def mirror_obs(self, obs: torch.Tensor) -> torch.Tensor:
        return obs[..., self.obs_joint_map] * self.obs_flip_mask * self.obs_joint_sign

    def augment(self, obs: torch.Tensor, actions: torch.Tensor):
        return torch.cat([obs, self.mirror_obs(obs)], dim=0), torch.cat(
            [actions, self.mirror_action(actions)], dim=0
        )
