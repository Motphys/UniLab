"""Stewart-platform ball-balancing task.

A 6-DOF parallel (Stewart) platform balances a free ball on its top plate. The
policy commands a 2-D platform tilt (roll, pitch); an inverse-kinematics step
converts the commanded plate pose into the six prismatic leg lengths that the
position actuators track. The objective is to bring the ball to the plate center
and hold it still. The platform base is welded to the world.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import gymnasium as gym
import numpy as np

from unilab.assets import ASSETS_ROOT_PATH
from unilab.base import registry
from unilab.base.backend import SimBackend, create_backend
from unilab.base.base import EnvCfg
from unilab.base.np_env import NpEnv, NpEnvState
from unilab.base.scene import SceneCfg
from unilab.dr.provider import DomainRandomizationProvider
from unilab.dr.types import DomainRandomizationCapabilities, ResetPlan
from unilab.dtype_config import get_global_dtype
from unilab.envs.common.rotation import (
    np_quat_apply_batched,
    np_quat_apply_inverse,
    np_quat_conjugate_batched,
    np_quat_from_euler_xyz,
    np_quat_mul_batched,
    np_quat_to_axis_angle,
)

# Leg base / top-connect bodies in actuator order (a0..a5 -> slide00,slide10,
# slide01,slide11,slide02,slide12), produced by the `replicate count=3` in the XML.
_LEG_BODY_NAMES = ["leg00", "leg10", "leg01", "leg11", "leg02", "leg12"]
_TOP_CONNECT_NAMES = [
    "top_connect00",
    "top_connect10",
    "top_connect01",
    "top_connect11",
    "top_connect02",
    "top_connect12",
]

_OBS_DIM = 15
_ACTION_DIM = 2


@dataclass
class StewartRewardConfig:
    """Reward shaping for the ball-balancing task (see `_compute_reward`)."""

    scales: dict[str, float] = field(
        default_factory=lambda: {"center": 0.7, "progress": 0.6, "still": 3.0}
    )
    fall_penalty: float = -6.0


@registry.envcfg("StewartBalance")
@dataclass
class StewartBalanceCfg(EnvCfg):
    scene: SceneCfg = field(
        default_factory=lambda: SceneCfg(
            model_file=str(ASSETS_ROOT_PATH / "robots" / "stewart" / "scene.xml")
        )
    )
    # The XML model is stiff; do not raise sim_dt above ~0.005.
    sim_dt: float = 0.004
    ctrl_dt: float = 0.02
    max_episode_seconds: float = 24.0
    render_spacing: float = 4.5

    # Body the backend treats as the kinematic base for its base-pose accessors.
    # The task reads explicit body ids instead, so any real body works; the moving
    # plate is the natural choice.
    base_name: str = "top"

    # Geometry (platform centered at the world origin, plate center at z=1).
    platform_radius: float = 0.8
    # The episode ends (ball "fallen") once it strays this far from the plate
    # center. Kept inside the physical rim (platform_radius) so the ball never
    # reaches the edge-contact regime that destabilizes the stiff closed-loop solver.
    fall_radius: float = 0.5
    top_center_z: float = 1.0
    top_surface_offset: float = 0.1
    ball_radius: float = 0.10
    init_ball_radius_ratio: float = 0.18

    # Control.
    target_rotation_limit_deg: float = 6.0
    action_smooth: float = 0.60
    center_control_radius: float = 0.25
    center_control_min_gain: float = 0.15
    vel_smooth: float = 0.25

    # Success / stillness window.
    still_xy: float = 0.12
    still_vel: float = 0.07
    still_xy_hysteresis: float = 1.15
    still_vel_hysteresis: float = 1.20
    zero_vel_thresh: float = 0.07
    still_steps_needed: int = 5

    reward_config: StewartRewardConfig = field(default_factory=StewartRewardConfig)

    def validate(self) -> None:
        super().validate()
        if not 0.0 <= self.init_ball_radius_ratio <= 1.0:
            raise ValueError("init_ball_radius_ratio must be in [0, 1]")
        if not 0.0 <= self.action_smooth <= 1.0:
            raise ValueError("action_smooth must be in [0, 1]")
        if not 0.0 <= self.center_control_min_gain <= 1.0:
            raise ValueError("center_control_min_gain must be in [0, 1]")


def _ball_home_z(cfg: StewartBalanceCfg) -> float:
    return cfg.top_center_z + cfg.top_surface_offset + cfg.ball_radius


def _roll_pitch_from_quat(quat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Extract roll/pitch (rad) from a wxyz quaternion via its rotation matrix."""
    w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
    r20 = 2.0 * (x * z - w * y)
    r21 = 2.0 * (y * z + w * x)
    r22 = 1.0 - 2.0 * (x * x + y * y)
    roll = np.arctan2(r21, r22)
    pitch = np.arctan2(-r20, np.sqrt(np.clip(r21 * r21 + r22 * r22, 0.0, None)))
    return roll.astype(np.float32), pitch.astype(np.float32)


# The mujoco backend constructs, resets, and steps correctly. Its closed-loop
# constraint solver is, however, less forgiving than motrix's under load (stiff
# parallel mechanism + ball contact), so a trained-grade policy is not yet stable
# there — closed-loop stability tuning for mujoco is a follow-up. The motrix
# backend is the validated training path.
@registry.env("StewartBalance", sim_backend="mujoco")
@registry.env("StewartBalance", sim_backend="motrix")
class StewartBalanceEnv(NpEnv):
    _cfg: StewartBalanceCfg

    def __init__(
        self,
        cfg: StewartBalanceCfg,
        num_envs: int = 1,
        backend_type: str = "motrix",
        dr_provider: DomainRandomizationProvider | None = None,
    ) -> None:
        # add_body_sensors=True injects body-pose tracking sensors the MuJoCo
        # backend needs for get_body_pos_w/quat_w on arbitrary bodies (the IK + obs
        # read top/ball/leg poses). The motrix backend reads poses natively and
        # ignores the flag.
        backend = create_backend(
            backend_type,
            cfg.scene,
            num_envs,
            cfg.sim_dt,
            base_name=cfg.base_name,
            add_body_sensors=True,
        )
        super().__init__(cfg, backend, num_envs)

        self._np_dtype = get_global_dtype()
        self._action_space = gym.spaces.Box(-1.0, 1.0, (_ACTION_DIM,), dtype=np.float32)

        if self._backend.num_actuators != 6:
            raise ValueError(f"Stewart model needs 6 actuators, got {self._backend.num_actuators}")
        ctrl_range = np.asarray(self._backend.get_actuator_ctrl_range(), dtype=np.float32)
        self._ctrl_lo = ctrl_range[:, 0]
        self._ctrl_hi = ctrl_range[:, 1]

        self._top_body_ids = self._backend.get_body_ids(["top"])
        self._ball_body_ids = self._backend.get_body_ids(["ball"])
        self._leg_body_ids = self._backend.get_body_ids(_LEG_BODY_NAMES)
        self._top_connect_ids = self._backend.get_body_ids(_TOP_CONNECT_NAMES)
        # The ball free joint is the first jointed body in the scene, so its
        # position occupies qpos[0:3] (validated against the default qpos in reset).
        self._ball_pos_qpos_idx = np.array([0, 1, 2], dtype=np.int64)

        # IK calibration (top home center, connect offsets, neutral leg lengths) is
        # resolved lazily on first use, once reset has placed the home state.
        self._ik_ready = False
        self._top_pos0 = np.zeros(3, dtype=np.float32)
        self._connect_offsets = np.zeros((6, 3), dtype=np.float32)
        self._leg0 = np.zeros(6, dtype=np.float32)

        self._init_domain_randomization(
            dr_provider if dr_provider is not None else StewartBalanceDRProvider()
        )

    @property
    def action_space(self) -> gym.spaces.Box:
        return self._action_space

    @property
    def obs_groups_spec(self) -> dict[str, int]:
        return {"obs": _OBS_DIM}

    def _ensure_ik_calibration(self) -> None:
        if self._ik_ready:
            return
        top_pos = np.asarray(
            self._backend.get_body_pos_w(self._top_body_ids)[:, 0, :], dtype=np.float32
        )
        connects = np.asarray(self._backend.get_body_pos_w(self._top_connect_ids), dtype=np.float32)
        legs = np.asarray(self._backend.get_body_pos_w(self._leg_body_ids), dtype=np.float32)
        # Use env 0 as the reference home configuration (all envs share the model).
        self._top_pos0 = top_pos[0]
        self._connect_offsets = (connects[0] - self._top_pos0).astype(np.float32)  # (6,3)
        self._leg0 = np.linalg.norm(connects[0] - legs[0], axis=-1).astype(np.float32)  # (6,)
        self._ik_ready = True

    def _leg_ctrl_for_tilt(self, target_tilt_rad: np.ndarray) -> np.ndarray:
        """Inverse kinematics: tilt command -> six prismatic leg targets.

        ``target_tilt_rad`` has shape (N, 2) = (roll, pitch). The plate stays at its
        home center; only its orientation tracks the commanded tilt.
        """
        self._ensure_ik_calibration()
        num = target_tilt_rad.shape[0]
        zeros = np.zeros((num,), dtype=np.float32)
        target_quat = np_quat_from_euler_xyz(
            target_tilt_rad[:, 0], target_tilt_rad[:, 1], zeros
        ).reshape(num, 4)
        # Rotate each connect offset by the per-env target quat: (N,1,4) x (1,6,3) -> (N,6,3).
        rotated = np_quat_apply_batched(target_quat[:, None, :], self._connect_offsets[None, :, :])
        expected = self._top_pos0[None, None, :] + rotated  # (N,6,3) connect targets
        bottoms = np.asarray(
            self._backend.get_body_pos_w(self._leg_body_ids), dtype=np.float32
        )  # (N,6,3)
        leg_len = np.linalg.norm(expected - bottoms, axis=-1) - self._leg0[None, :]
        return np.clip(leg_len, self._ctrl_lo, self._ctrl_hi).astype(np.float32)

    def apply_action(self, actions: np.ndarray, state: NpEnvState) -> np.ndarray:
        cfg = self._cfg
        info = state.info
        raw = np.clip(np.asarray(actions, dtype=np.float32), -1.0, 1.0).reshape(
            self._num_envs, _ACTION_DIM
        )

        # Exponential action smoothing.
        prev = info["prev_action_exec"]
        alpha = float(cfg.action_smooth)
        action_exec = (alpha * raw + (1.0 - alpha) * prev).astype(np.float32)
        info["prev_action_exec"] = action_exec
        info["action_exec"] = action_exec

        # Soften authority while the ball is already near the center.
        rel_xy = info["last_rel_xy"]
        if cfg.center_control_radius > 0.0 and cfg.center_control_min_gain < 1.0:
            ratio = np.clip(rel_xy / max(cfg.center_control_radius, 1e-6), 0.0, 1.0)
            gain = cfg.center_control_min_gain + (1.0 - cfg.center_control_min_gain) * ratio
        else:
            gain = np.ones((self._num_envs,), dtype=np.float32)
        effective = action_exec * gain[:, None]

        target_tilt_deg = effective * cfg.target_rotation_limit_deg
        info["target_tilt_cmd"] = target_tilt_deg.astype(np.float32)
        return self._leg_ctrl_for_tilt(np.deg2rad(target_tilt_deg).astype(np.float32))

    def _read_ball_rel(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        top_pos = np.asarray(
            self._backend.get_body_pos_w(self._top_body_ids)[:, 0, :], dtype=np.float32
        )
        top_quat = np.asarray(
            self._backend.get_body_quat_w(self._top_body_ids)[:, 0, :], dtype=np.float32
        )
        ball_pos = np.asarray(
            self._backend.get_body_pos_w(self._ball_body_ids)[:, 0, :], dtype=np.float32
        )
        rel = np_quat_apply_inverse(top_quat, (ball_pos - top_pos)).astype(np.float32)
        return rel, top_quat, ball_pos

    def update_state(self, state: NpEnvState) -> NpEnvState:
        cfg = self._cfg
        info = state.info
        dt = float(cfg.ctrl_dt)

        rel, top_quat, ball_pos = self._read_ball_rel()

        rel_vel = (rel - info["prev_rel"]) / dt
        filt_rel_vel = (
            cfg.vel_smooth * rel_vel + (1.0 - cfg.vel_smooth) * info["filtered_rel_vel"]
        ).astype(np.float32)
        info["prev_rel"] = rel
        info["filtered_rel_vel"] = filt_rel_vel

        quat_delta = np_quat_mul_batched(top_quat, np_quat_conjugate_batched(info["prev_top_quat"]))
        top_ang_vel = (np_quat_to_axis_angle(quat_delta) / dt).astype(np.float32)
        filt_ang_vel = (
            cfg.vel_smooth * top_ang_vel + (1.0 - cfg.vel_smooth) * info["filtered_top_ang_vel"]
        ).astype(np.float32)
        info["prev_top_quat"] = top_quat
        info["filtered_top_ang_vel"] = filt_ang_vel
        ang_vel_local = np_quat_apply_inverse(top_quat, filt_ang_vel).astype(np.float32)

        roll, pitch = _roll_pitch_from_quat(top_quat)
        rel_xy = np.linalg.norm(rel[:, :2], axis=-1).astype(np.float32)
        vel_xy = np.linalg.norm(filt_rel_vel[:, :2], axis=-1).astype(np.float32)
        info["last_rel_xy"] = rel_xy

        limit = max(cfg.target_rotation_limit_deg, 1e-6)
        obs = np.concatenate(
            [
                rel,
                filt_rel_vel,
                np.stack([np.rad2deg(roll) / limit, np.rad2deg(pitch) / limit], axis=-1).astype(
                    np.float32
                ),
                ang_vel_local,
                info["target_tilt_cmd"] / limit,
                info["action_exec"],
            ],
            axis=-1,
        ).astype(self._np_dtype)

        reward, terminated = self._compute_reward(cfg, info, rel_xy, vel_xy, ball_pos)
        return state.replace(obs={"obs": obs}, reward=reward, terminated=terminated)

    def _compute_reward(self, cfg, info, rel_xy, vel_xy, ball_pos):
        rc = cfg.reward_config
        scales = rc.scales

        fall_z = cfg.top_center_z - np.sin(np.deg2rad(30.0)) * cfg.platform_radius
        fallen = (rel_xy > cfg.fall_radius) | (ball_pos[:, 2] < fall_z)

        center_score = np.clip(1.0 - rel_xy / max(cfg.fall_radius, 1e-6), 0.0, 1.0)
        term_center = scales["center"] * center_score

        # Reward shrinking the stop-radius: progress toward the center between
        # near-zero-velocity moments (a stable "settled closer than before" event).
        prev_zero = info["prev_zero_vel_rel_xy"]
        zero_event = vel_xy <= cfg.zero_vel_thresh
        improve = np.maximum(prev_zero - rel_xy, 0.0)
        improve_norm = np.clip(improve / max(cfg.platform_radius, 1e-6), 0.0, 1.0)
        term_progress = np.where(
            zero_event & (rel_xy < prev_zero), scales["progress"] * improve_norm, 0.0
        )
        next_zero = prev_zero.copy()
        next_zero[zero_event] = rel_xy[zero_event]
        info["prev_zero_vel_rel_xy"] = next_zero.astype(np.float32)

        still_steps = self._update_stillness(cfg, info, rel_xy, vel_xy)
        success = still_steps >= cfg.still_steps_needed
        term_still = np.where(success, scales["still"], 0.0)

        reward = (term_center + term_progress + term_still).astype(self._np_dtype)
        reward = np.where(fallen, rc.fall_penalty, reward).astype(self._np_dtype)
        terminated = (fallen | success).astype(bool)
        return reward, terminated

    def _update_stillness(self, cfg, info, rel_xy, vel_xy) -> np.ndarray:
        xy_enter, vel_enter = cfg.still_xy, cfg.still_vel
        xy_exit = cfg.still_xy * cfg.still_xy_hysteresis
        vel_exit = cfg.still_vel * cfg.still_vel_hysteresis

        active = info["still_window_active"]
        steps = info["still_steps"]
        keep = active & (rel_xy <= xy_exit) & (vel_xy <= vel_exit)
        enter = (~active) & (rel_xy <= xy_enter) & (vel_xy <= vel_enter)
        steps = np.where(keep, steps + 1, np.where(enter, 1, 0)).astype(np.int32)
        active = keep | enter
        info["still_window_active"] = active
        info["still_steps"] = steps
        return steps


class StewartBalanceDRProvider(DomainRandomizationProvider):
    """Resets the plate to its level home and drops the ball near the center.

    No physical randomization terms are used; the reset variety comes from the
    randomized ball position, which is the balancing challenge.
    """

    def validate(self, env, capabilities: DomainRandomizationCapabilities) -> None:  # noqa: D102
        return None

    def build_reset_plan(self, env: StewartBalanceEnv, env_ids: np.ndarray) -> ResetPlan:
        cfg: StewartBalanceCfg = env._cfg
        n = int(env_ids.shape[0])
        default_qpos = np.asarray(env._backend.get_default_qpos(), dtype=np.float64)
        # Ball free joint is first in the scene -> position is qpos[0:3].
        if not np.allclose(default_qpos[0:2], 0.0, atol=1e-3):
            raise ValueError("Unexpected qpos layout: ball position is not at qpos[0:3]")
        qpos = np.broadcast_to(default_qpos, (n, default_qpos.shape[0])).copy()

        # Ball: uniform within a disk near the plate center, resting on the surface.
        radius = (
            cfg.platform_radius
            * cfg.init_ball_radius_ratio
            * np.sqrt(np.random.uniform(0.0, 1.0, size=n))
        )
        theta = np.random.uniform(0.0, 2.0 * np.pi, size=n)
        ball_xyz = np.stack(
            [radius * np.cos(theta), radius * np.sin(theta), np.full((n,), _ball_home_z(cfg))],
            axis=-1,
        )
        qpos[:, env._ball_pos_qpos_idx] = ball_xyz

        init_qvel = np.asarray(env._backend.get_init_qvel(), dtype=np.float64)
        qvel = np.broadcast_to(init_qvel, (n, init_qvel.shape[0])).copy()

        zeros3 = np.zeros((n, 3), dtype=np.float32)
        zeros2 = np.zeros((n, 2), dtype=np.float32)
        identity_quat = np.tile(np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32), (n, 1))
        info_updates: dict = {
            "prev_rel": zeros3.copy(),
            "filtered_rel_vel": zeros3.copy(),
            "prev_top_quat": identity_quat,
            "filtered_top_ang_vel": zeros3.copy(),
            "target_tilt_cmd": zeros2.copy(),
            "action_exec": zeros2.copy(),
            "prev_action_exec": zeros2.copy(),
            "last_rel_xy": np.zeros((n,), dtype=np.float32),
            "prev_zero_vel_rel_xy": np.full((n,), cfg.platform_radius, dtype=np.float32),
            "still_steps": np.zeros((n,), dtype=np.int32),
            "still_window_active": np.zeros((n,), dtype=bool),
        }
        return ResetPlan(env_ids=env_ids, qpos=qpos, qvel=qvel, info_updates=info_updates)

    def build_reset_observation(
        self, env: StewartBalanceEnv, env_ids: np.ndarray, info_updates: dict
    ) -> dict:
        rel, top_quat, _ = env._read_ball_rel()
        rel = rel[env_ids]
        top_quat = top_quat[env_ids]
        roll, pitch = _roll_pitch_from_quat(top_quat)
        limit = max(env._cfg.target_rotation_limit_deg, 1e-6)
        n = int(env_ids.shape[0])
        info_updates["prev_rel"] = rel.astype(np.float32)
        info_updates["prev_top_quat"] = top_quat.astype(np.float32)
        info_updates["last_rel_xy"] = np.linalg.norm(rel[:, :2], axis=-1).astype(np.float32)
        info_updates["prev_zero_vel_rel_xy"] = info_updates["last_rel_xy"].copy()
        obs = np.concatenate(
            [
                rel,
                np.zeros((n, 3), dtype=np.float32),
                np.stack([np.rad2deg(roll) / limit, np.rad2deg(pitch) / limit], axis=-1).astype(
                    np.float32
                ),
                np.zeros((n, 3), dtype=np.float32),
                np.zeros((n, 2), dtype=np.float32),
                np.zeros((n, 2), dtype=np.float32),
            ],
            axis=-1,
        ).astype(get_global_dtype())
        return {"obs": obs}
