from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from unilab.base.backend import SimBackend
from unilab.envs.common.rotation import (
    np_matrix_from_quat,
    np_quat_canonicalize,
    np_quat_inv,
    np_quat_mul,
)
from unilab.envs.locomotion.common.base import (
    ControlConfigBase,
    LocomotionBaseCfg,
    LocomotionBaseEnv,
    Sensor,
)

DEFAULT_LEG_ANGLES = np.asarray(
    [
        0.1,
        0.8,
        -1.5,
        -0.1,
        0.8,
        -1.5,
        0.1,
        0.8,
        -1.5,
        -0.1,
        0.8,
        -1.5,
    ],
    dtype=np.float64,
)


@dataclass
class NoiseConfig:
    level: float = 0.0
    scale_joint_angle: float = 0.03
    scale_joint_vel: float = 0.5
    scale_gyro: float = 0.2
    scale_gravity: float = 0.05
    scale_linvel: float = 0.1
    scale_ee_pos: float = 0.02


@dataclass
class ControlConfig(ControlConfigBase):
    Kp: float = 35.0
    Kd: float = 0.5
    leg_kp: float | list[float] | None = 60.0
    leg_kd: float | list[float] | None = 2.0
    arm_kp: float | list[float] | None = field(
        default_factory=lambda: [95.0, 115.0, 100.0, 52.0, 54.0, 55.0]
    )
    arm_kd: float | list[float] | None = field(
        default_factory=lambda: [3.5, 3.8, 2.5, 1.5, 1.5, 1.5]
    )
    # 机械臂动作缩放（独立于腿部 action_scale），IK 已提供主要控制，
    # policy 仅做残差修正，设小值防止早期随机输出压过 IK
    arm_action_scale: float = 0.0


@dataclass
class IKConfig:
    damping: float = 0.05
    gain: float = 1.0
    dq_clip: float = 0.2
    use_orientation: bool = False
    # Used only when use_orientation is true:
    # - target: track goal_local_quat against curr_local_quat.
    # - zero_error: include the rotational Jacobian with zero orientation error,
    #   matching go2_arx's position IK with orientation-change regularization.
    orientation_mode: str = "target"


@dataclass
class Asset:
    base_name: str = "base"
    foot_name: str = "foot"
    ground: str = "floor"
    ee_site_name: str = "endpoint"
    ee_body_name: str = "link6"
    arm_joint_names: tuple[str, ...] = (
        "joint1",
        "joint2",
        "joint3",
        "joint4",
        "joint5",
        "joint6",
    )


@dataclass
class Go2ArmSensor(Sensor):
    feet_force: list[str] = field(
        default_factory=lambda: [
            "FL_foot_contact",
            "FR_foot_contact",
            "RL_foot_contact",
            "RR_foot_contact",
        ]
    )
    feet_pos: list[str] = field(default_factory=lambda: ["FL_pos", "FR_pos", "RL_pos", "RR_pos"])
    ee_local_pos: str = "endpoint_pos"
    ee_local_quat: str = "endpoint_quat"
    ee_local_vel: str = "endpoint_vel"
    ee_relative_pos: str = "endpoint_relative_pos"
    ee_relative_quat: str = "endpoint_relative_quat"
    arm_ref_world_quat: str = "armbasepoint_world_quat"


@dataclass
class Go2ArmBaseCfg(LocomotionBaseCfg):
    noise_config: NoiseConfig = field(default_factory=NoiseConfig)  # type: ignore[assignment]
    control_config: ControlConfig = field(default_factory=ControlConfig)  # type: ignore[assignment]
    ik: IKConfig = field(default_factory=IKConfig)
    asset: Asset = field(default_factory=Asset)
    sensor: Go2ArmSensor = field(default_factory=Go2ArmSensor)  # type: ignore[assignment]
    iterations: int | None = None
    sim_dt: float = 0.01
    ctrl_dt: float = 0.02


def _expand_gain(
    name: str, value: float | list[float] | None, fallback: float, size: int
) -> np.ndarray:
    raw_value = fallback if value is None else value
    gain = np.asarray(raw_value, dtype=np.float64)
    if gain.ndim == 0:
        return np.full((size,), float(gain), dtype=np.float64)
    if gain.shape != (size,):
        raise ValueError(f"{name} must be a scalar or have shape ({size},), got {gain.shape}")
    return gain


def build_go2_arm_position_gains(cfg: ControlConfig) -> dict[str, np.ndarray]:
    leg_kp = _expand_gain("control_config.leg_kp", cfg.leg_kp, cfg.Kp, 12)
    leg_kd = _expand_gain("control_config.leg_kd", cfg.leg_kd, cfg.Kd, 12)
    arm_kp = _expand_gain("control_config.arm_kp", cfg.arm_kp, cfg.Kp, 6)
    arm_kd = _expand_gain("control_config.arm_kd", cfg.arm_kd, cfg.Kd, 6)
    return {
        "kp": np.concatenate([leg_kp, arm_kp]),
        "kd": np.concatenate([leg_kd, arm_kd]),
    }


class Go2ArmBaseEnv(LocomotionBaseEnv):
    _cfg: Go2ArmBaseCfg

    def __init__(self, cfg: Go2ArmBaseCfg, backend: SimBackend, num_envs: int = 1):
        super().__init__(cfg, backend, num_envs)
        self._ee_site_id = int(self._backend.get_site_ids([cfg.asset.ee_site_name])[0])
        self._arm_jacobian_dof_indices = self._backend.get_joint_dof_indices(
            cfg.asset.arm_joint_names
        )
        self._arm_dof_pos_indices = self._backend.get_joint_dof_pos_indices(
            cfg.asset.arm_joint_names
        )
        self._arm_dof_vel_indices = self._backend.get_joint_dof_vel_indices(
            cfg.asset.arm_joint_names
        )

    @property
    def arm_dof_pos_indices(self) -> np.ndarray:
        return self._arm_dof_pos_indices

    @property
    def arm_jacobian_dof_indices(self) -> np.ndarray:
        return self._arm_jacobian_dof_indices

    def _obs_noise(self, data: np.ndarray, scale: float) -> np.ndarray:
        """Apply per-step uniform observation noise scaled by ``noise_config.level``."""
        noise_cfg = self._cfg.noise_config
        if noise_cfg.level > 0.0:
            return data + (
                np.random.uniform(-1.0, 1.0, data.shape).astype(data.dtype)
                * noise_cfg.level
                * scale
            )
        return data

    def get_foot_pos(self) -> np.ndarray:
        """Get foot positions. Returns shape (num_envs, 4, 3)."""
        foot_pos = [self._backend.get_sensor_data(name) for name in self._cfg.sensor.feet_pos]
        return np.stack(foot_pos, axis=1)

    def get_foot_contact(self) -> np.ndarray:
        """Get foot contact values. Returns shape (num_envs, 4)."""
        contacts = [
            self._backend.get_sensor_data(name)[:, 0] for name in self._cfg.sensor.feet_force
        ]
        return np.stack(contacts, axis=1)

    def get_ee_local_pose(self) -> tuple[np.ndarray, np.ndarray]:
        """Return end-effector pose expressed in the arm reference frame."""
        pos = self._backend.get_sensor_data(self._cfg.sensor.ee_local_pos)
        quat = self._backend.get_sensor_data(self._cfg.sensor.ee_local_quat)
        return pos, quat

    def get_arm_dof_pos(self) -> np.ndarray:
        return self.get_dof_pos()[:, self._arm_dof_pos_indices]

    def get_arm_dof_vel(self) -> np.ndarray:
        return self.get_dof_vel()[:, self._arm_dof_vel_indices]

    def compute_arm_ik_delta(
        self,
        goal_local_pos: np.ndarray,
        curr_local_pos: np.ndarray,
        goal_local_quat: np.ndarray | None = None,
        curr_local_quat: np.ndarray | None = None,
    ) -> np.ndarray:
        """Compute damped least-squares IK delta for the 6 arm joints.

        Position and optional orientation errors are expressed in the arm
        reference frame (``armbasepoint``). MuJoCo provides world-frame site
        Jacobians, so both translational and rotational blocks are rotated into
        that same frame before solving.
        """
        cfg = self._cfg.ik
        pos_err = np.asarray(goal_local_pos - curr_local_pos)
        jacp_w, jacr_w = self._backend.get_site_jacobian_w(
            self._ee_site_id,
            self._arm_jacobian_dof_indices,
        )
        ref_rot_w = np_matrix_from_quat(
            self._backend.get_sensor_data(self._cfg.sensor.arm_ref_world_quat)
        )
        rot_w_to_b = np.swapaxes(ref_rot_w, 1, 2)
        jacp_b = np.matmul(rot_w_to_b, jacp_w)

        if cfg.use_orientation:
            if cfg.orientation_mode == "target":
                if goal_local_quat is None or curr_local_quat is None:
                    raise ValueError(
                        "goal_local_quat and curr_local_quat are required when "
                        "ik.use_orientation=true and ik.orientation_mode='target'"
                    )
                orn_err = _orientation_error_local(goal_local_quat, curr_local_quat)
            elif cfg.orientation_mode == "zero_error":
                orn_err = np.zeros_like(pos_err)
            else:
                raise ValueError(
                    "ik.orientation_mode must be one of {'target', 'zero_error'}, "
                    f"got {cfg.orientation_mode!r}"
                )
            jacr_b = np.matmul(rot_w_to_b, jacr_w)
            jac = np.concatenate([jacp_b, jacr_b], axis=1)
            dpose = np.concatenate([pos_err, orn_err], axis=1)
        else:
            jac = jacp_b
            dpose = pos_err

        identity = np.eye(jac.shape[1], dtype=jac.dtype)[None, :, :]
        lhs = np.matmul(jac, np.swapaxes(jac, 1, 2)) + identity * (cfg.damping**2)
        rhs = dpose[:, :, None]
        solved = np.linalg.solve(lhs, rhs)
        dq = np.matmul(np.swapaxes(jac, 1, 2), solved)[:, :, 0]
        if cfg.dq_clip > 0.0:
            dq = np.clip(dq, -cfg.dq_clip, cfg.dq_clip)
        return dq


def _normalize_quat(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q)
    if q.ndim == 1:
        q = q[None, :]
    norm = np.linalg.norm(q, axis=1, keepdims=True)
    return q / np.clip(norm, 1.0e-8, None)


def _orientation_error_local(goal_quat: np.ndarray, curr_quat: np.ndarray) -> np.ndarray:
    goal = _normalize_quat(goal_quat)
    curr = _normalize_quat(curr_quat)
    rel = np_quat_mul(goal, np_quat_inv(curr))
    rel = np_quat_canonicalize(rel)
    sign = np.where(rel[:, 0:1] < 0.0, -1.0, 1.0)
    return rel[:, 1:] * sign
