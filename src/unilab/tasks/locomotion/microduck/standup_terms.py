# Ported from microduck_rl (anchor commit 29e887e) tasks/mdp.py +
# microduck_standup_env_cfg.py; adapted to the NumPy manager runtime and the
# base-owned reset transaction.
"""Stand-up task terms for MicroDuck (sit keyframe / ground poses -> stand).

Mirrors the upstream standup layer: the fixed-target goal-state rewards
(``pose_target_match`` / ``pose_l1_penalty`` / ``height_target_gaussian`` /
``height_l1_penalty`` / ``standing_composite_score``), the two-layer upright
pair (``body_upright_linear`` / ``upright_gaussian_at_height``), the gentle-rise
and arrival-damping penalties (``trunk_vertical_accel_penalty`` /
``body_ang_vel_at_height``), the locomotion-aware 6D body pose tracking reward
(``body_pose_tracking_locomotion``), the mixed ground-state reset event
(``set_random_ground_state``: face-down / face-up / sitting / standing
buckets), and the ``robot_state_is_nan`` termination with contact-force
sensor coverage.

Sign convention (upstream, kept here): ``*_penalty`` / ``*_l1`` functions are
self-negating (return <= 0) and take POSITIVE weights; plain cost functions
(``body_ang_vel_at_height``) return >= 0 and take negative weights.

Documented adaptation: upstream ``body_pose_tracking_locomotion`` reads the
left_foot/right_foot SITES for the x/y/yaw feet reference; the UniLab Entity
facade exposes no site state, so the port reads the ankle BODIES
(``feet_cfg`` body_names). The standup config tracks only z/roll/pitch
(``axis_weights=(0, 0, 1, 1, 1, 0)``), where the feet reference is unused, so
the trained reward is identical; the x/y/yaw axes would differ from upstream
if their weights were raised.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any, ClassVar, cast

import numpy as np

from unilab.dtype_config import get_global_dtype
from unilab.envs.mdp import resolve_env_ids
from unilab.managers import ManagerTermBase, ManagerTermBaseCfg, SceneEntityCfg
from unilab.tasks.locomotion.common.manager_terms import SensorTermBase, _real
from unilab.tasks.locomotion.microduck.manager_terms import _asset_selection, _command
from unilab.tasks.locomotion.microduck.recovery_terms import (
    _check_params,
    _cos_tilt,
    _entity,
    _fallen_mask,
    _root_z,
)
from unilab.utils.rotation import np_wrap_to_pi

if TYPE_CHECKING:
    from unilab.base.entity import Entity
    from unilab.managers._types import ManagerBasedRlEnv


_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")
_DEFAULT_FEET_CFG = SceneEntityCfg("robot", body_names=("ankle_left", "ankle_right"))


def _smoothstep(t: np.ndarray) -> np.ndarray:
    return t * t * (3.0 - 2.0 * t)


def _z_gate(z: np.ndarray, low: float, high: float) -> np.ndarray:
    t = np.clip((z - low) / max(high - low, 1e-6), 0.0, 1.0)
    return _smoothstep(t)


class _PoseTargetTerm(ManagerTermBase):
    """Fixed-target pose error against ``default_joint_pos`` (+ overrides).

    Upstream keys ``target_overrides`` by servo qpos index; the UniLab port
    keys it by joint NAME (the scene-declared joint order is the contract here)
    and resolves names to columns at term construction (cold path).
    """

    _allowed_params: ClassVar[frozenset[str]] = frozenset({"asset_cfg", "target_overrides"})

    def __init__(self, cfg: ManagerTermBaseCfg, env: ManagerBasedRlEnv):
        super().__init__(env)
        _check_params(cfg, self._allowed_params, self.name)
        self._entity, self._joint_ids = _asset_selection(cfg, env, term=self.name)
        overrides = cfg.params.get("target_overrides")
        self._override_columns: np.ndarray | None = None
        self._override_values: np.ndarray | None = None
        if overrides:
            if not isinstance(overrides, dict):
                raise TypeError(f"{self.name} target_overrides must be a joint-name mapping")
            columns: list[int] = []
            values: list[float] = []
            selected = {int(j): k for k, j in enumerate(self._joint_ids)}
            for joint_name, angle in overrides.items():
                if not isinstance(joint_name, str) or not joint_name:
                    raise ValueError(f"{self.name} target_overrides keys must be joint names")
                ids, _ = self._entity.find_joints(f"^{joint_name}$")
                column = int(ids[0])
                if column not in selected:
                    raise ValueError(
                        f"{self.name} target_overrides joint '{joint_name}' is not in the "
                        "asset_cfg selection"
                    )
                columns.append(selected[column])
                values.append(_real(self.name, f"target_overrides[{joint_name}]", angle))
            self._override_columns = np.asarray(columns, dtype=np.intp)
            self._override_values = np.asarray(values, dtype=get_global_dtype())

    def _pose_error(self, env: ManagerBasedRlEnv) -> np.ndarray:
        del env
        actual = self._entity.data.joint_pos[:, self._joint_ids]
        target = self._entity.data.default_joint_pos[:, self._joint_ids]
        if self._override_columns is not None:
            target = np.array(target, copy=True)
            target[:, self._override_columns] = self._override_values
        return np.asarray(actual - target, dtype=get_global_dtype())


class pose_target_match(_PoseTargetTerm):
    """Gaussian pose-match against a single fixed target, rewarded from t=0."""

    _allowed_params: ClassVar[frozenset[str]] = _PoseTargetTerm._allowed_params | {"std"}

    def __init__(self, cfg: ManagerTermBaseCfg, env: ManagerBasedRlEnv):
        super().__init__(cfg, env)
        self._std = _real(
            self.name, "std", cfg.params.get("std", 0.3), minimum=0.0, strict_minimum=True
        )

    def __call__(self, env: ManagerBasedRlEnv, **params: Any) -> np.ndarray:
        del params
        per_joint = np.exp(-np.square(self._pose_error(env) / self._std))
        return np.asarray(np.mean(per_joint, axis=1), dtype=get_global_dtype())


class pose_l1_penalty(_PoseTargetTerm):
    """L1 companion to ``pose_target_match`` (constant gradient toward target).

    Self-negating (returns <= 0): configure with a POSITIVE weight.
    """

    def __call__(self, env: ManagerBasedRlEnv, **params: Any) -> np.ndarray:
        del params
        return np.asarray(-np.mean(np.abs(self._pose_error(env)), axis=1), dtype=get_global_dtype())


def height_target_gaussian(
    env: ManagerBasedRlEnv,
    target_height: float,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    std: float = 0.02,
) -> np.ndarray:
    """Gaussian on trunk z against a single fixed target."""
    _real("height_target_gaussian", "target_height", target_height)
    scale = _real("height_target_gaussian", "std", std, minimum=0.0, strict_minimum=True)
    entity = _entity_simple(env, asset_cfg, "height_target_gaussian")
    z = _root_z(env, entity)
    return np.asarray(np.exp(-np.square((z - target_height) / scale)), dtype=get_global_dtype())


def height_l1_penalty(
    env: ManagerBasedRlEnv,
    target_height: float,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> np.ndarray:
    """L1 companion to ``height_target_gaussian`` (self-negating: positive weight)."""
    _real("height_l1_penalty", "target_height", target_height)
    entity = _entity_simple(env, asset_cfg, "height_l1_penalty")
    z = _root_z(env, entity)
    return np.asarray(-np.abs(z - target_height), dtype=get_global_dtype())


def _entity_simple(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg, term: str) -> Entity:
    if not isinstance(asset_cfg, SceneEntityCfg):
        raise TypeError(f"{term} asset_cfg must be SceneEntityCfg")
    return cast("Entity", env.scene[asset_cfg.name])


class trunk_vertical_accel_penalty(ManagerTermBase):
    """Penalty proportional to ``|a_z|`` of the trunk (finite-diff of v_z).

    Self-negating (returns ``-|a_z|``): configure with a POSITIVE weight.
    At episode reset the accel is zeroed so the previous episode's final
    velocity does not leak into the new one.
    """

    _allowed_params: ClassVar[frozenset[str]] = frozenset({"asset_cfg"})

    def __init__(self, cfg: ManagerTermBaseCfg, env: ManagerBasedRlEnv):
        super().__init__(env)
        _check_params(cfg, self._allowed_params, self.name)
        self._entity = _entity(cfg, env, term=self.name)
        self._prev: np.ndarray | None = None

    def __call__(self, env: ManagerBasedRlEnv, **params: Any) -> np.ndarray:
        del params
        vz = np.nan_to_num(self._entity.data.root_link_lin_vel_w[:, 2], nan=0.0)
        prev = vz.copy() if self._prev is None else self._prev
        step_dt = _real(self.name, "step_dt", env.step_dt, minimum=0.0, strict_minimum=True)
        a_z = (vz - prev) / step_dt
        # Zero out a_z at reset steps to suppress the cross-episode transient.
        fresh = env.episode_length_buf <= 1
        a_z = np.where(fresh, 0.0, a_z)
        self._prev = vz.copy()
        return np.asarray(-np.abs(a_z), dtype=get_global_dtype())


def body_ang_vel_at_height(
    env: ManagerBasedRlEnv,
    height_low: float,
    height_high: float,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    tilt_full_deg: float | None = None,
    tilt_zero_deg: float = 45.0,
) -> np.ndarray:
    """Trunk ``sum(w_xy^2)`` cost gated by trunk z (and optionally tilt).

    Returns the gated POSITIVE cost; configure with a negative weight. Zero
    below ``height_low``, full above ``height_high``; with ``tilt_full_deg``
    set the cost is additionally smoothstep-gated to zero beyond
    ``tilt_zero_deg`` so the approach to vertical stays untaxed.
    """
    low = _real("body_ang_vel_at_height", "height_low", height_low)
    high = _real("body_ang_vel_at_height", "height_high", height_high)
    entity = _entity_simple(env, asset_cfg, "body_ang_vel_at_height")
    ang_vel = np.asarray(entity.data.body_link_ang_vel_w[:, asset_cfg.body_ids, :])
    if ang_vel.ndim != 3 or ang_vel.shape[1] != 1:
        raise ValueError(
            "body_ang_vel_at_height asset_cfg must select exactly one body; "
            f"received shape {ang_vel.shape}"
        )
    cost = np.sum(np.square(ang_vel[:, 0, :2]), axis=1)
    gate = _z_gate(_root_z(env, entity), low, high)
    if tilt_full_deg is not None:
        full = _real("body_ang_vel_at_height", "tilt_full_deg", tilt_full_deg, minimum=0.0)
        zero = _real("body_ang_vel_at_height", "tilt_zero_deg", tilt_zero_deg, minimum=0.0)
        cos_tilt = np.clip(_cos_tilt(entity), -1.0, 1.0)
        tilt_deg = np.degrees(np.arccos(cos_tilt))
        s = np.clip((zero - tilt_deg) / max(zero - full, 1e-6), 0.0, 1.0)
        gate = gate * _smoothstep(s)
    return np.asarray(cost * gate, dtype=get_global_dtype())


def body_upright_linear(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    gate_z_below: float | None = None,
    gate_tilt_above_deg: float = 40.0,
) -> np.ndarray:
    """Linear uprightness reward: ``cos(tilt)`` in [-1, 1] with gradient everywhere.

    With ``gate_z_below`` set (velstand recovery variant), the reward is
    multiplied by the fallen mask so it cannot dilute tracking rewards during
    clean walking. Standup uses the ungated form.
    """
    entity = _entity_simple(env, asset_cfg, "body_upright_linear")
    reward = _cos_tilt(entity)
    if gate_z_below is not None:
        reward = reward * _fallen_mask(env, entity, gate_z_below, gate_tilt_above_deg)
    return np.asarray(reward, dtype=get_global_dtype())


def upright_gaussian_at_height(
    env: ManagerBasedRlEnv,
    std: float,
    height_low: float,
    height_high: float,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> np.ndarray:
    """Gaussian upright reward weighted by a smoothstep on trunk z.

    Full ``exp(-tilt^2/std^2)`` when ``z >= height_high``, zero when
    ``z <= height_low`` — the upright incentive only applies near the target
    standing height so the policy cannot farm it in a low vertical crouch.
    """
    scale = _real("upright_gaussian_at_height", "std", std, minimum=0.0, strict_minimum=True)
    low = _real("upright_gaussian_at_height", "height_low", height_low)
    high = _real("upright_gaussian_at_height", "height_high", height_high)
    entity = _entity_simple(env, asset_cfg, "upright_gaussian_at_height")
    quat = entity.data.root_link_quat_w
    tilt_sq = 2.0 * (np.square(quat[:, 1]) + np.square(quat[:, 2]))
    upright_g = np.exp(-tilt_sq / (scale * scale))
    gate = _z_gate(_root_z(env, entity), low, high)
    return np.asarray(upright_g * gate, dtype=get_global_dtype())


class standing_composite_score(_PoseTargetTerm):
    """Smooth multiplicative goal-state score (product of three Gaussians).

    ``height_score * upright_score * pose_score``, each in [0, 1]: a deficiency
    in any one factor collapses the whole reward, breaking additive-stack
    compromise basins.
    """

    _allowed_params: ClassVar[frozenset[str]] = _PoseTargetTerm._allowed_params | {
        "target_height",
        "height_std",
        "upright_std",
        "pose_std",
    }

    def __init__(self, cfg: ManagerTermBaseCfg, env: ManagerBasedRlEnv):
        super().__init__(cfg, env)
        self._target_height = _real(self.name, "target_height", cfg.params.get("target_height"))
        self._height_std = _real(
            self.name, "height_std", cfg.params.get("height_std"), minimum=0.0, strict_minimum=True
        )
        self._upright_std = _real(
            self.name,
            "upright_std",
            cfg.params.get("upright_std"),
            minimum=0.0,
            strict_minimum=True,
        )
        self._pose_std = _real(
            self.name, "pose_std", cfg.params.get("pose_std"), minimum=0.0, strict_minimum=True
        )

    def __call__(self, env: ManagerBasedRlEnv, **params: Any) -> np.ndarray:
        del params
        z = _root_z(env, self._entity)
        height_score = np.exp(-np.square((z - self._target_height) / self._height_std))
        quat = self._entity.data.root_link_quat_w
        tilt_sq = 2.0 * (np.square(quat[:, 1]) + np.square(quat[:, 2]))
        upright_score = np.exp(-tilt_sq / (self._upright_std * self._upright_std))
        pose_err_sq = np.mean(np.square(self._pose_error(env)), axis=1)
        pose_score = np.exp(-pose_err_sq / (self._pose_std * self._pose_std))
        return np.asarray(height_score * upright_score * pose_score, dtype=get_global_dtype())


def body_pose_tracking_locomotion(
    env: ManagerBasedRlEnv,
    command_name: str = "body_pose",
    nominal_height: float = 0.105,
    xy_std: float = 0.02,
    z_std: float = 0.03,
    angle_std: float = math.radians(30),
    axis_weights: tuple[float, float, float, float, float, float] = (1.0, 1.0, 1.0, 1.0, 1.0, 1.0),
    vel_gate_command_name: str | None = None,
    vel_gate_std: float = 0.1,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    feet_cfg: SceneEntityCfg = _DEFAULT_FEET_CFG,
) -> np.ndarray:
    """Locomotion-aware 6D body pose tracking (per-axis Gaussians, weighted mean).

    Command is ``[x, y, z, roll, pitch, yaw]`` deltas from the nominal stand.
    x/y are measured trunk-relative to the feet centroid (rotated into the
    trunk body frame) and yaw relative to the circular-mean feet yaw, so the
    reward stays meaningful away from the spawn origin; z/roll/pitch are
    world-frame trunk quantities relative to ``nominal_height`` / upright.

    Port note: upstream reads foot SITES; this port reads the ankle BODIES via
    ``feet_cfg`` (the Entity facade exposes no site state). With the standup
    ``axis_weights=(0, 0, 1, 1, 1, 0)`` the feet reference feeds only
    zero-weighted axes and the reward is upstream-identical.
    """
    term = "body_pose_tracking_locomotion"
    _real(term, "nominal_height", nominal_height)
    for label, value in (("xy_std", xy_std), ("z_std", z_std), ("angle_std", angle_std)):
        _real(term, label, value, minimum=0.0, strict_minimum=True)
    if len(axis_weights) != 6:
        raise ValueError(f"{term} axis_weights must have 6 entries")
    weights = np.asarray(
        [_real(term, f"axis_weights[{i}]", w, minimum=0.0) for i, w in enumerate(axis_weights)],
        dtype=get_global_dtype(),
    )
    entity = _entity_simple(env, asset_cfg, term)
    if not isinstance(feet_cfg, SceneEntityCfg):
        raise TypeError(f"{term} feet_cfg must be SceneEntityCfg")

    command = _command(env, term, command_name)
    if command.shape[1] != 6:
        raise ValueError(
            f"{term} command '{command_name}' must have width 6, received shape {command.shape}"
        )

    pos_w = entity.data.root_link_pos_w
    quat = entity.data.root_link_quat_w
    qw, qx, qy, qz = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
    trunk_yaw = np.arctan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))
    roll = np.arctan2(2.0 * (qw * qx + qy * qz), 1.0 - 2.0 * (qx * qx + qy * qy))
    pitch = np.arcsin(np.clip(2.0 * (qw * qy - qz * qx), -1.0, 1.0))

    # Feet reference (ankle bodies; see the port note above).
    foot_pos = entity.data.body_link_pos_w[:, feet_cfg.body_ids, :]  # (N, 2, 3)
    foot_quat = entity.data.body_link_quat_w[:, feet_cfg.body_ids, :]  # (N, 2, 4)
    feet_centroid = np.mean(foot_pos, axis=1)

    # Trunk xy in body frame relative to the feet centroid.
    dx_w = pos_w[:, 0] - feet_centroid[:, 0]
    dy_w = pos_w[:, 1] - feet_centroid[:, 1]
    cos_y = np.cos(trunk_yaw)
    sin_y = np.sin(trunk_yaw)
    x_body = cos_y * dx_w + sin_y * dy_w
    y_body = -sin_y * dx_w + cos_y * dy_w

    z_world = _root_z(env, entity)

    fqw, fqx, fqy, fqz = (
        foot_quat[..., 0],
        foot_quat[..., 1],
        foot_quat[..., 2],
        foot_quat[..., 3],
    )
    foot_yaws = np.arctan2(2.0 * (fqw * fqz + fqx * fqy), 1.0 - 2.0 * (fqy * fqy + fqz * fqz))
    mean_foot_yaw = np.arctan2(
        np.mean(np.sin(foot_yaws), axis=1), np.mean(np.cos(foot_yaws), axis=1)
    )

    errors = (
        (x_body - command[:, 0]) / xy_std,
        (y_body - command[:, 1]) / xy_std,
        (z_world - (nominal_height + command[:, 2])) / z_std,
        (roll - command[:, 3]) / angle_std,
        (pitch - command[:, 4]) / angle_std,
        np_wrap_to_pi(trunk_yaw - mean_foot_yaw - command[:, 5]) / angle_std,
    )
    per_axis = np.exp(-np.square(np.stack(errors, axis=1)))
    total_w = float(np.sum(weights))
    reward = np.sum(per_axis * weights, axis=1) / max(total_w, 1e-6)

    if vel_gate_command_name is not None:
        # Gate on commanded planar velocity: body tracking only contributes
        # when the robot is supposed to be standing still.
        gate_std = _real(term, "vel_gate_std", vel_gate_std, minimum=0.0, strict_minimum=True)
        vel_cmd = _command(env, term, vel_gate_command_name)
        vel_mag = np.linalg.norm(vel_cmd[:, :2], axis=1)
        reward = reward * np.exp(-np.square(vel_mag / gate_std))
    return np.asarray(reward, dtype=get_global_dtype())


class robot_state_is_nan(SensorTermBase):
    """Terminate envs whose physics state or named contact-sensor forces go non-finite.

    Mirrors upstream ``robot_state_is_nan``: covers joint pos/vel, root
    pos/quat/lin/ang vel, plus the forces of the named contact sensors — a
    degenerate contact produces a NaN impulse a step before the integrated
    state diverges, and standup lands/flips constantly.
    """

    _allowed_params: ClassVar[frozenset[str]] = frozenset({"asset_cfg", "sensor_names"})

    def __init__(self, cfg: ManagerTermBaseCfg, env: ManagerBasedRlEnv):
        super().__init__(cfg, env)
        asset_cfg = cfg.params.get("asset_cfg", _DEFAULT_ASSET_CFG)
        if not isinstance(asset_cfg, SceneEntityCfg):
            raise TypeError(f"{self.name} asset_cfg must be SceneEntityCfg")
        self._entity = cast("Entity", env.scene[asset_cfg.name])
        sensor_names = cfg.params.get("sensor_names", ())
        if isinstance(sensor_names, (str, bytes)) or not isinstance(sensor_names, (tuple, list)):
            raise TypeError(f"{self.name} sensor_names must be a sequence of sensor names")
        self._view = self._bind(tuple(sensor_names)) if sensor_names else None

    def __call__(self, env: ManagerBasedRlEnv, **params: Any) -> np.ndarray:
        del params
        data = self._entity.data
        states = (
            data.joint_pos,
            data.joint_vel,
            data.root_link_pos_w,
            data.root_link_quat_w,
            data.root_link_lin_vel_w,
            data.root_link_ang_vel_w,
        )
        invalid = np.zeros(env.num_envs, dtype=np.bool_)
        for state in states:
            invalid |= ~np.isfinite(state).reshape(env.num_envs, -1).all(axis=1)
        if self._view is not None:
            forces = self._view.read()
            invalid |= ~np.isfinite(forces).reshape(env.num_envs, -1).all(axis=1)
        return invalid


# Reset event (mode="reset"). Runs inside the env-owned reset transaction after
# reset_base, reusing its staged xy; z / orientation / velocities are fully
# overwritten per bucket, and the sitting bucket also rewrites the joints.


def set_random_ground_state(
    env: ManagerBasedRlEnv,
    env_ids: np.ndarray | None,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    face_down_prob: float = 0.4,
    face_up_prob: float = 0.4,
    sitting_prob: float = 0.2,
    standing_prob: float = 0.0,
    prone_z_min: float = 0.20,
    prone_z_max: float = 0.25,
    sitting_z_min: float = 0.07,
    sitting_z_max: float = 0.09,
    standing_z_min: float = 0.11,
    standing_z_max: float = 0.12,
    sitting_joint_overrides: dict[str, float] | None = None,
    sitting_joint_noise_std: float = 0.0,
    sitting_tilt_max: float = 0.0,
    face_up_roll_max: float = 0.0,
) -> None:
    """Reset to a random ground state: face-down, face-up, sitting, or standing.

    Bucket probabilities are normalized internally (they need not sum to 1).
    Face-down/up: +-90 deg pitch, random yaw, z in the prone band; face-up
    spawns get an additional uniform +-``face_up_roll_max`` roll about the
    body long axis (reverse-curriculum starts partway along the supine->prone
    roll). Sitting: upright with optional +-``sitting_tilt_max`` pitch/roll
    noise, z in the sitting band, joints set to ``sitting_joint_overrides``
    (keyed by joint NAME; upstream indexes servo qpos columns) plus Gaussian
    joint noise. Standing: upright, z in the standing band, joints left at the
    staged HOME keyframe.
    """
    term = "set_random_ground_state"
    probs = [
        _real(term, label, value, minimum=0.0)
        for label, value in (
            ("face_down_prob", face_down_prob),
            ("face_up_prob", face_up_prob),
            ("sitting_prob", sitting_prob),
            ("standing_prob", standing_prob),
        )
    ]
    total = sum(probs)
    if total <= 0.0:
        raise ValueError(f"{term} bucket probabilities must not all be zero")
    entity = _entity_simple(env, asset_cfg, term)
    ids = resolve_env_ids(env, env_ids)
    if ids.size == 0:
        return
    num = ids.size

    p_fd = probs[0] / total
    p_fu = (probs[0] + probs[1]) / total
    p_sit = (probs[0] + probs[1] + probs[2]) / total

    yaw = env.rng.uniform(-math.pi, math.pi, size=num)
    cy = np.cos(yaw * 0.5)
    sy = np.sin(yaw * 0.5)
    s = 2.0**-0.5  # sqrt(2)/2

    face_down = np.stack([s * cy, -s * sy, s * cy, s * sy], axis=1)
    face_up = np.stack([s * cy, s * sy, -s * cy, s * sy], axis=1)
    tilt_max = _real(term, "sitting_tilt_max", sitting_tilt_max, minimum=0.0)
    if tilt_max > 0.0:
        pitch = env.rng.uniform(-tilt_max, tilt_max, size=num)
        roll = env.rng.uniform(-tilt_max, tilt_max, size=num)
        cp = np.cos(pitch * 0.5)
        sp = np.sin(pitch * 0.5)
        cr = np.cos(roll * 0.5)
        sr = np.sin(roll * 0.5)
        # ZYX intrinsic Euler -> quaternion (yaw * pitch * roll).
        sitting = np.stack(
            [
                cr * cp * cy + sr * sp * sy,
                sr * cp * cy - cr * sp * sy,
                cr * sp * cy + sr * cp * sy,
                cr * cp * sy - sr * sp * cy,
            ],
            axis=1,
        )
    else:
        sitting = np.stack([cy, np.zeros_like(cy), np.zeros_like(cy), sy], axis=1)

    u = env.rng.random(num)
    is_fu = (u >= p_fd) & (u < p_fu)
    is_sit = (u >= p_fu) & (u < p_sit)
    is_stand = u >= p_sit

    # Face-up partial-roll noise about the body LONG axis (body z): starts a
    # fraction of face-up envs partway along the supine->prone roll.
    roll_max = _real(term, "face_up_roll_max", face_up_roll_max, minimum=0.0)
    if roll_max > 0.0:
        theta = env.rng.uniform(-roll_max, roll_max, size=num)
        ct = np.cos(theta * 0.5)
        st = np.sin(theta * 0.5)
        # Body-frame rotation -> right-multiply: q_fu (x) [ct, 0, 0, st].
        w, x, y, z_ = face_up[:, 0], face_up[:, 1], face_up[:, 2], face_up[:, 3]
        face_up = np.stack(
            [w * ct - z_ * st, x * ct + y * st, y * ct - x * st, w * st + z_ * ct],
            axis=1,
        )

    # Sitting and standing share the same upright orientation; they differ only
    # in trunk height and joint pose.
    new_quat = face_down.copy()
    new_quat[is_fu] = face_up[is_fu]
    new_quat[is_sit] = sitting[is_sit]
    new_quat[is_stand] = sitting[is_stand]

    z_prone = env.rng.uniform(
        _real(term, "prone_z_min", prone_z_min, minimum=0.0),
        _real(term, "prone_z_max", prone_z_max, minimum=0.0),
        size=num,
    )
    z_sit = env.rng.uniform(
        _real(term, "sitting_z_min", sitting_z_min, minimum=0.0),
        _real(term, "sitting_z_max", sitting_z_max, minimum=0.0),
        size=num,
    )
    z_stand = env.rng.uniform(
        _real(term, "standing_z_min", standing_z_min, minimum=0.0),
        _real(term, "standing_z_max", standing_z_max, minimum=0.0),
        size=num,
    )
    new_z = np.where(is_sit, z_sit, z_prone)
    new_z = np.where(is_stand, z_stand, new_z)

    pose = entity.read_reset_root_pose(env_ids=ids)
    pose[:, 2] = new_z
    pose[:, 3:7] = new_quat
    entity.write_root_link_pose_to_sim(pose, env_ids=ids)
    entity.write_root_link_velocity_to_sim(
        np.zeros((num, 6), dtype=get_global_dtype()), env_ids=ids
    )

    # Sitting bucket: joints = staged HOME + overrides, then Gaussian noise on
    # every joint (on the plain microduck model every declared joint is a
    # servo, matching the upstream servo-only noise mask).
    sit_ids = ids[is_sit]
    if sit_ids.size == 0:
        return
    joints = np.array(entity.data.default_joint_pos[sit_ids], copy=True)
    if sitting_joint_overrides:
        if not isinstance(sitting_joint_overrides, dict):
            raise TypeError(f"{term} sitting_joint_overrides must be a joint-name mapping")
        for joint_name, angle in sitting_joint_overrides.items():
            if not isinstance(joint_name, str) or not joint_name:
                raise ValueError(f"{term} sitting_joint_overrides keys must be joint names")
            joint_ids, _ = entity.find_joints(f"^{joint_name}$")
            joints[:, int(joint_ids[0])] = _real(
                term, f"sitting_joint_overrides[{joint_name}]", angle
            )
    noise_std = _real(term, "sitting_joint_noise_std", sitting_joint_noise_std, minimum=0.0)
    if noise_std > 0.0:
        joints += env.rng.normal(0.0, noise_std, size=joints.shape)
    entity.write_joint_state_to_sim(
        np.asarray(joints, dtype=get_global_dtype()),
        np.zeros_like(joints, dtype=get_global_dtype()),
        env_ids=sit_ids,
    )


__all__ = [
    "body_ang_vel_at_height",
    "body_pose_tracking_locomotion",
    "body_upright_linear",
    "height_l1_penalty",
    "height_target_gaussian",
    "pose_l1_penalty",
    "pose_target_match",
    "robot_state_is_nan",
    "set_random_ground_state",
    "standing_composite_score",
    "trunk_vertical_accel_penalty",
    "upright_gaussian_at_height",
]
