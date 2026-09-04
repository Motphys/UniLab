# Fall/recovery layer for the MicroDuck velstand task, ported from
# microduck_rl (anchor commit 29e887e) tasks/mdp.py +
# microduck_velstand_env_cfg.py; adapted to the NumPy manager runtime and the
# base-owned reset transaction.
"""Fall/recovery terms for the MicroDuck velstand task (walk + get-up).

Mirrors the upstream recovery layer: potential-based ``upright_progress`` /
``height_progress`` shaping, the tilt-gated ``com_upward_velocity``, the
hysteretic ``fallen_state_penalty`` tax, the one-shot ``recovery_success``
bounty, the ``joint_torque_rate_l2`` smoothness penalty, the
``fallen_too_long`` backstop termination, ``feet_air_time_upright`` (air-time
window zeroed while fallen), and the prone/crouch reset events.

NaN handling mirrors upstream ``nan_to_num`` semantics: the termination
manager computes before the reward manager within a step, so a NaN solver
state can reach these terms in the same step that ``nan_state`` fires.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any, ClassVar, cast

import numpy as np

from unilab.dtype_config import get_global_dtype
from unilab.envs.mdp import resolve_env_ids
from unilab.managers import ManagerTermBase, ManagerTermBaseCfg, SceneEntityCfg
from unilab.tasks.locomotion.common.gait_terms import feet_air_time
from unilab.tasks.locomotion.common.manager_terms import _real

if TYPE_CHECKING:
    from unilab.base.entity import Entity
    from unilab.managers._types import ManagerBasedRlEnv


_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def _entity(cfg: ManagerTermBaseCfg, env: ManagerBasedRlEnv, *, term: str) -> Entity:
    asset_cfg = cfg.params.get("asset_cfg", _DEFAULT_ASSET_CFG)
    if not isinstance(asset_cfg, SceneEntityCfg):
        raise TypeError(f"{term} asset_cfg must be SceneEntityCfg")
    return cast("Entity", env.scene[asset_cfg.name])


def _check_params(cfg: ManagerTermBaseCfg, allowed: frozenset[str], term: str) -> None:
    unexpected = set(cfg.params) - allowed
    if unexpected:
        raise TypeError(f"{term} received unsupported parameters: {sorted(unexpected)}")


def _fallen_mask(
    env: ManagerBasedRlEnv,
    entity: Entity,
    gate_z_below: float,
    gate_tilt_above_deg: float,
) -> np.ndarray:
    """1.0 where FALLEN: trunk height below ``gate_z_below`` OR tilt beyond the gate."""
    z = np.nan_to_num(
        entity.data.root_link_pos_w[:, 2] - np.asarray(env.scene.env_origins)[:, 2],
        nan=0.0,
    )
    quat = entity.data.root_link_quat_w
    # cos(tilt) = R22 = 1 - 2(qx^2 + qy^2)
    cos_tilt = 1.0 - 2.0 * (np.square(quat[:, 1]) + np.square(quat[:, 2]))
    fallen = (z < gate_z_below) | (cos_tilt < math.cos(math.radians(gate_tilt_above_deg)))
    return np.asarray(fallen, dtype=get_global_dtype())


def _cos_tilt(entity: Entity) -> np.ndarray:
    quat = entity.data.root_link_quat_w
    return 1.0 - 2.0 * (np.square(quat[:, 1]) + np.square(quat[:, 2]))


def _root_z(env: ManagerBasedRlEnv, entity: Entity) -> np.ndarray:
    return np.nan_to_num(
        entity.data.root_link_pos_w[:, 2] - np.asarray(env.scene.env_origins)[:, 2],
        nan=0.0,
    )


class feet_air_time_upright(feet_air_time):
    """velocity-template feet_air_time, zeroed while FALLEN (tilt > gate).

    A robot lying on its trunk can still tap its feet rhythmically through the
    air-time window (the upstream "lies there shaking a leg" exploit); air time
    is only meaningful upright.
    """

    _allowed_params: ClassVar[frozenset[str]] = feet_air_time._allowed_params | {
        "asset_cfg",
        "gate_tilt_above_deg",
    }

    def __init__(self, cfg: ManagerTermBaseCfg, env: ManagerBasedRlEnv):
        super().__init__(cfg, env)
        _check_params(cfg, self._allowed_params, self.name)
        self._gate_tilt_above_deg = _real(
            self.name,
            "gate_tilt_above_deg",
            cfg.params.get("gate_tilt_above_deg", 40.0),
            minimum=0.0,
        )
        self._entity = _entity(cfg, env, term=self.name)

    def __call__(self, env: ManagerBasedRlEnv, **params: Any) -> np.ndarray:
        del params
        reward = super().__call__(env)
        upright = 1.0 - _fallen_mask(env, self._entity, 0.0, self._gate_tilt_above_deg)
        return np.asarray(reward * upright, dtype=get_global_dtype())


class upright_progress(ManagerTermBase):
    """Potential-based upright shaping: delta cos(tilt) per step.

    Pays for progress toward upright, charges for progress toward fallen, and
    pays exactly zero for holding any pose (unfarmable). A full prone-to-stand
    recovery collects delta ~ +1 (times weight); a fall costs the same.
    """

    _allowed_params: ClassVar[frozenset[str]] = frozenset({"asset_cfg"})

    def __init__(self, cfg: ManagerTermBaseCfg, env: ManagerBasedRlEnv):
        super().__init__(env)
        _check_params(cfg, self._allowed_params, self.name)
        self._entity = _entity(cfg, env, term=self.name)
        self._prev: np.ndarray | None = None

    def __call__(self, env: ManagerBasedRlEnv, **params: Any) -> np.ndarray:
        del params
        cos_tilt = np.asarray(
            np.nan_to_num(_cos_tilt(self._entity), nan=1.0), dtype=get_global_dtype()
        )
        prev = cos_tilt.copy() if self._prev is None else self._prev
        # Freshly reset envs: no spurious delta from the previous episode's pose.
        fresh = env.episode_length_buf <= 1
        prev[fresh] = cos_tilt[fresh]
        delta = cos_tilt - prev
        self._prev = cos_tilt.copy()
        return np.asarray(delta, dtype=get_global_dtype())


class height_progress(ManagerTermBase):
    """Potential-based height shaping: delta min(trunk z, ceiling) per step.

    The z-axis companion to ``upright_progress``: the crouch-to-stand last
    mile is mostly a height change at modest tilt, where delta cos(tilt) is
    tiny. Capped at ``ceiling`` so hopping above stance height pays nothing.
    """

    _allowed_params: ClassVar[frozenset[str]] = frozenset({"asset_cfg", "ceiling"})

    def __init__(self, cfg: ManagerTermBaseCfg, env: ManagerBasedRlEnv):
        super().__init__(env)
        _check_params(cfg, self._allowed_params, self.name)
        self._entity = _entity(cfg, env, term=self.name)
        self._ceiling = _real(self.name, "ceiling", cfg.params.get("ceiling", 0.115), minimum=0.0)
        self._prev: np.ndarray | None = None

    def __call__(self, env: ManagerBasedRlEnv, **params: Any) -> np.ndarray:
        del params
        pot = np.asarray(
            np.minimum(_root_z(env, self._entity), self._ceiling), dtype=get_global_dtype()
        )
        prev = pot.copy() if self._prev is None else self._prev
        fresh = env.episode_length_buf <= 1
        prev[fresh] = pot[fresh]
        delta = pot - prev
        self._prev = pot.copy()
        return np.asarray(delta, dtype=get_global_dtype())


class fallen_state_penalty(ManagerTermBase):
    """1.0 while FALLEN (weight it negative): a flat per-step tax on staying down.

    With ``release_*`` set the tax has hysteresis: a fall arms it and it keeps
    paying until the robot is genuinely up (tilt < release_tilt AND z >
    release_z), not merely under the arming gate. Arms only on a genuine fall,
    so gait-cycle tilt wobble is never taxed.
    """

    _allowed_params: ClassVar[frozenset[str]] = frozenset(
        {"asset_cfg", "gate_tilt_above_deg", "release_tilt_below_deg", "release_z_above"}
    )

    def __init__(self, cfg: ManagerTermBaseCfg, env: ManagerBasedRlEnv):
        super().__init__(env)
        _check_params(cfg, self._allowed_params, self.name)
        self._entity = _entity(cfg, env, term=self.name)
        self._gate_tilt_above_deg = _real(
            self.name,
            "gate_tilt_above_deg",
            cfg.params.get("gate_tilt_above_deg", 40.0),
            minimum=0.0,
        )
        release_tilt = cfg.params.get("release_tilt_below_deg")
        self._release_tilt_below_deg = (
            None
            if release_tilt is None
            else _real(self.name, "release_tilt_below_deg", release_tilt, minimum=0.0)
        )
        release_z = cfg.params.get("release_z_above")
        self._release_z_above = (
            None if release_z is None else _real(self.name, "release_z_above", release_z)
        )
        self._armed = np.zeros(env.num_envs, dtype=np.bool_)

    def reset(self, env_ids: np.ndarray | slice | None) -> None:
        self._armed[slice(None) if env_ids is None else env_ids] = False

    def __call__(self, env: ManagerBasedRlEnv, **params: Any) -> np.ndarray:
        del params
        fallen = _fallen_mask(env, self._entity, 0.0, self._gate_tilt_above_deg) > 0.0
        if self._release_tilt_below_deg is None:
            return np.asarray(fallen, dtype=get_global_dtype())
        up = _cos_tilt(self._entity) > math.cos(math.radians(self._release_tilt_below_deg))
        if self._release_z_above is not None:
            up &= _root_z(env, self._entity) > self._release_z_above
        fresh = env.episode_length_buf <= 1
        self._armed[fresh] = False
        self._armed |= fallen
        self._armed &= ~up
        return np.asarray(self._armed, dtype=get_global_dtype())


class recovery_success(ManagerTermBase):
    """One-shot bounty on a COMPLETED recovery.

    Fires on the frame where an env that has been fallen (tilt >
    ``fallen_tilt_deg`` for >= ``min_fallen_s``) becomes genuinely upright
    (tilt < ``up_tilt_deg`` AND trunk z > ``up_z``). Hysteresis: re-arms only
    by being fallen again, so oscillating around the gate pays nothing.
    """

    _allowed_params: ClassVar[frozenset[str]] = frozenset(
        {"asset_cfg", "fallen_tilt_deg", "min_fallen_s", "up_tilt_deg", "up_z"}
    )

    def __init__(self, cfg: ManagerTermBaseCfg, env: ManagerBasedRlEnv):
        super().__init__(env)
        _check_params(cfg, self._allowed_params, self.name)
        self._entity = _entity(cfg, env, term=self.name)
        self._fallen_tilt_deg = _real(
            self.name, "fallen_tilt_deg", cfg.params.get("fallen_tilt_deg", 40.0), minimum=0.0
        )
        self._min_fallen_s = _real(
            self.name, "min_fallen_s", cfg.params.get("min_fallen_s", 0.5), minimum=0.0
        )
        self._up_tilt_deg = _real(
            self.name, "up_tilt_deg", cfg.params.get("up_tilt_deg", 25.0), minimum=0.0
        )
        self._up_z = _real(self.name, "up_z", cfg.params.get("up_z", 0.105))
        self._fallen_s = np.zeros(env.num_envs, dtype=get_global_dtype())
        self._armed = np.zeros(env.num_envs, dtype=np.bool_)

    def reset(self, env_ids: np.ndarray | slice | None) -> None:
        rows = slice(None) if env_ids is None else env_ids
        self._fallen_s[rows] = 0.0
        self._armed[rows] = False

    def __call__(self, env: ManagerBasedRlEnv, **params: Any) -> np.ndarray:
        del params
        cos_tilt = _cos_tilt(self._entity)
        fallen = cos_tilt < math.cos(math.radians(self._fallen_tilt_deg))
        up = (cos_tilt > math.cos(math.radians(self._up_tilt_deg))) & (
            _root_z(env, self._entity) > self._up_z
        )
        fresh = env.episode_length_buf <= 1
        self._fallen_s[fresh] = 0.0
        self._armed[fresh] = False
        step_dt = _real(self.name, "step_dt", env.step_dt, minimum=0.0, strict_minimum=True)
        self._fallen_s = np.where(fallen, self._fallen_s + step_dt, 0.0)
        self._armed |= self._fallen_s >= self._min_fallen_s
        fired = self._armed & up
        self._armed &= ~fired
        return np.asarray(fired, dtype=get_global_dtype())


class joint_torque_rate_l2(ManagerTermBase):
    """Penalize rate of change in actuator torques (proxy for gearbox shock).

    Returns the sum of squared torque differences from the previous step.
    Upstream reads ``asset.data.actuator_force``; the UniLab BAM port reads
    :attr:`BamVoltageAction.applied_torque` (the torque actually written to
    ctrl each control step) from the named action term, fail-closed.
    """

    _allowed_params: ClassVar[frozenset[str]] = frozenset({"action_name"})

    def __init__(self, cfg: ManagerTermBaseCfg, env: ManagerBasedRlEnv):
        super().__init__(env)
        _check_params(cfg, self._allowed_params, self.name)
        action_name = cfg.params.get("action_name", "joint_pos")
        if not isinstance(action_name, str) or not action_name:
            raise ValueError(f"{self.name} action_name must be a non-empty string")
        from unilab.tasks.locomotion.microduck.bam_action import BamVoltageAction

        term = env.action_manager.get_term(action_name)
        if not isinstance(term, BamVoltageAction):
            raise TypeError(
                f"{self.name} requires a BamVoltageAction action term, got "
                f"'{action_name}' of type {type(term).__name__}"
            )
        self._action_term = term
        self._prev: np.ndarray | None = None

    def __call__(self, env: ManagerBasedRlEnv, **params: Any) -> np.ndarray:
        del params
        current = np.asarray(self._action_term.applied_torque, dtype=get_global_dtype())
        if self._prev is None:
            self._prev = current.copy()
            return np.zeros(env.num_envs, dtype=get_global_dtype())
        rate = current - self._prev
        self._prev = current.copy()
        return np.asarray(np.sum(np.square(rate), axis=1), dtype=get_global_dtype())


class fallen_too_long(ManagerTermBase):
    """Terminate envs that have been continuously FALLEN for ``max_duration_s``.

    Backstop for envs that mix walking with fall recovery: the tilt
    termination gets disabled by curriculum so the policy can attempt
    recovery; this recycles failed recoveries instead of letting them farm
    recovery reward for the whole episode.
    """

    _allowed_params: ClassVar[frozenset[str]] = frozenset(
        {"asset_cfg", "gate_z_below", "gate_tilt_above_deg", "max_duration_s"}
    )

    def __init__(self, cfg: ManagerTermBaseCfg, env: ManagerBasedRlEnv):
        super().__init__(env)
        _check_params(cfg, self._allowed_params, self.name)
        self._entity = _entity(cfg, env, term=self.name)
        self._gate_z_below = _real(self.name, "gate_z_below", cfg.params.get("gate_z_below", 0.10))
        self._gate_tilt_above_deg = _real(
            self.name,
            "gate_tilt_above_deg",
            cfg.params.get("gate_tilt_above_deg", 40.0),
            minimum=0.0,
        )
        self._max_duration_s = _real(
            self.name,
            "max_duration_s",
            cfg.params.get("max_duration_s", 5.0),
            minimum=0.0,
            strict_minimum=True,
        )
        self._timer = np.zeros(env.num_envs, dtype=get_global_dtype())

    def reset(self, env_ids: np.ndarray | slice | None) -> None:
        self._timer[slice(None) if env_ids is None else env_ids] = 0.0

    def __call__(self, env: ManagerBasedRlEnv, **params: Any) -> np.ndarray:
        del params
        fallen = (
            _fallen_mask(env, self._entity, self._gate_z_below, self._gate_tilt_above_deg) > 0.0
        )
        # Freshly reset envs start with a clean timer.
        self._timer[env.episode_length_buf <= 1] = 0.0
        step_dt = _real(self.name, "step_dt", env.step_dt, minimum=0.0, strict_minimum=True)
        self._timer = np.where(fallen, self._timer + step_dt, 0.0)
        return self._timer >= self._max_duration_s


def com_upward_velocity(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    max_height: float = 0.08,
    gate_z_below: float | None = None,
    gate_tilt_above_deg: float = 40.0,
    max_vz: float | None = None,
) -> np.ndarray:
    """Reward upward CoM velocity while below ``max_height``.

    Once standing, the reward is zero so the robot has no incentive to keep
    squatting to farm upward-velocity reward. ``max_vz`` caps the rewarded
    velocity so the gentlest rise that reaches the cap is optimal. With
    ``gate_z_below`` set (velstand), the reward is additionally multiplied by
    the fallen mask so gait dip-and-rise cannot farm it.
    """
    if not isinstance(asset_cfg, SceneEntityCfg):
        raise TypeError("com_upward_velocity asset_cfg must be SceneEntityCfg")
    entity = cast("Entity", env.scene[asset_cfg.name])
    com_z = _root_z(env, entity)
    vz = np.nan_to_num(entity.data.root_link_lin_vel_w[:, 2], nan=0.0)
    below_target = np.asarray(com_z < max_height, dtype=get_global_dtype())
    reward = np.clip(vz, 0.0, max_vz) * below_target
    if gate_z_below is not None:
        reward = reward * _fallen_mask(env, entity, gate_z_below, gate_tilt_above_deg)
    return np.asarray(reward, dtype=get_global_dtype())


# Reset events (mode="reset"). These run inside the env-owned reset
# transaction after reset_base, so the staged root pose (x, y already
# randomized) is read back through Entity.read_reset_root_pose, modified, and
# written back; velocities are zeroed like the upstream qvel writes.


def set_random_prone_orientation(
    env: ManagerBasedRlEnv,
    env_ids: np.ndarray | None,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    face_down_prob: float = 0.5,
) -> None:
    """Randomly set each env face-down (belly) or face-up (back), random yaw.

    Face-down: +90 deg pitch -> quat = [s*cy, -s*sy,  s*cy,  s*sy]
    Face-up:   -90 deg pitch -> quat = [s*cy,  s*sy, -s*cy,  s*sy]
    """
    if not isinstance(asset_cfg, SceneEntityCfg):
        raise TypeError("set_random_prone_orientation asset_cfg must be SceneEntityCfg")
    ids = resolve_env_ids(env, env_ids)
    if ids.size == 0:
        return
    entity = cast("Entity", env.scene[asset_cfg.name])
    num = ids.size

    yaw = env.rng.uniform(-math.pi, math.pi, size=num)
    cy = np.cos(yaw * 0.5)
    sy = np.sin(yaw * 0.5)
    s = 2.0**-0.5  # sqrt(2)/2

    face_down = np.stack([s * cy, -s * sy, s * cy, s * sy], axis=1)
    face_up = np.stack([s * cy, s * sy, -s * cy, s * sy], axis=1)
    mask = env.rng.random(num) < face_down_prob  # True -> face-down
    new_quat = np.where(mask[:, None], face_down, face_up)

    pose = entity.read_reset_root_pose(env_ids=ids)
    pose[:, 3:7] = new_quat
    entity.write_root_link_pose_to_sim(pose, env_ids=ids)
    entity.write_root_link_velocity_to_sim(
        np.zeros((num, 6), dtype=get_global_dtype()), env_ids=ids
    )


# Deep-crouch anchor pose (velstand run-5): the "stuck" mid-recovery basin —
# knees folded under the body, trunk pitched forward, feet flat. Values chosen
# by extending the HOME zig-zag (hip fwd / knee back / ankle fwd, sign
# conventions per the SIT keyframe fold directions) to deep flexion, inside
# the +-1.57 joint limits. hip_yaw/hip_roll/neck stay at HOME.
_CROUCH_ANCHOR_BY_NAME = {
    "left_hip_pitch": -1.15,
    "left_knee": 1.25,
    "left_ankle": 1.05,
    "right_hip_pitch": 1.15,
    "right_knee": -1.25,
    "right_ankle": -1.05,
}


def set_random_crouch_state(
    env: ManagerBasedRlEnv,
    env_ids: np.ndarray | None,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    depth_min: float = 0.35,
    depth_max: float = 1.0,
    pitch_max_deg: float = 55.0,
    joint_noise: float = 0.12,
    z_stand: float = 0.115,
    z_deep: float = 0.06,
) -> None:
    """Reset selected envs into a random mid-recovery crouch.

    Reverse curriculum for the recovery last mile: prone-init episodes spend
    most of their fallen budget getting TO the deep crouch, so crouch-to-stand
    gets almost no on-policy data. Seeding resets across that mile (depth lam
    in [depth_min, depth_max] between standing and the deep-crouch anchor,
    trunk pitch and z scaled with lam) makes the frontier dense from step 0.
    """
    if not isinstance(asset_cfg, SceneEntityCfg):
        raise TypeError("set_random_crouch_state asset_cfg must be SceneEntityCfg")
    ids = resolve_env_ids(env, env_ids)
    if ids.size == 0:
        return
    entity = cast("Entity", env.scene[asset_cfg.name])
    num = ids.size

    lam = env.rng.uniform(depth_min, depth_max, size=num)

    # Joints: lerp HOME -> anchor on the leg pitch chain, uniform noise on the
    # servo joints. On the plain microduck model every declared joint is a
    # servo, so the upstream servo-only noise mask covers all of them.
    joints = np.array(entity.data.default_joint_pos[ids], copy=True)
    for joint_name, anchor in _CROUCH_ANCHOR_BY_NAME.items():
        joint_ids, _ = entity.find_joints(f"^{joint_name}$")
        column = joint_ids[0]
        joints[:, column] = joints[:, column] + lam * (anchor - joints[:, column])
    joints += env.rng.uniform(-1.0, 1.0, size=joints.shape) * joint_noise

    # Base orientation: forward pitch scaled with depth (the stuck basin is a
    # forward crouch from both fall directions), random yaw, small roll noise.
    pitch = lam * math.radians(pitch_max_deg) + env.rng.uniform(
        -math.radians(10.0), math.radians(10.0), size=num
    )
    pitch = np.clip(pitch, math.radians(5.0), None)
    roll = env.rng.uniform(-math.radians(8.0), math.radians(8.0), size=num)
    yaw = env.rng.uniform(-math.pi, math.pi, size=num)
    cy = np.cos(yaw * 0.5)
    sy = np.sin(yaw * 0.5)
    cp = np.cos(pitch * 0.5)
    sp = np.sin(pitch * 0.5)
    cr = np.cos(roll * 0.5)
    sr = np.sin(roll * 0.5)
    # ZYX intrinsic Euler -> quaternion (yaw * pitch * roll).
    quat = np.stack(
        [
            cr * cp * cy + sr * sp * sy,
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
        ],
        axis=1,
    )

    # Trunk height scaled with depth, small upward margin to settle cleanly.
    z = z_stand + lam * (z_deep - z_stand) + env.rng.random(num) * 0.01

    pose = entity.read_reset_root_pose(env_ids=ids)
    pose[:, 2] = z
    pose[:, 3:7] = quat
    entity.write_root_link_pose_to_sim(pose, env_ids=ids)
    entity.write_root_link_velocity_to_sim(
        np.zeros((num, 6), dtype=get_global_dtype()), env_ids=ids
    )
    entity.write_joint_state_to_sim(
        np.asarray(joints, dtype=get_global_dtype()),
        np.zeros_like(joints, dtype=get_global_dtype()),
        env_ids=ids,
    )


def maybe_set_random_prone_orientation(
    env: ManagerBasedRlEnv,
    env_ids: np.ndarray | None,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    prone_prob: float = 0.0,
    face_down_prob: float = 0.5,
    prone_z_min: float = 0.20,
    prone_z_max: float = 0.25,
    crouch_prob: float = 0.0,
) -> None:
    """Reset event that overrides orientation to prone with probability ``prone_prob``.

    With prob ``prone_prob``, replaces the upright orientation (already staged
    by reset_base) with a prone orientation; otherwise leaves it upright.
    Among the overridden envs, ``face_down_prob`` picks face-down vs face-up.
    Prone envs are also lifted to z in [prone_z_min, prone_z_max] so the
    head/neck clearance is sufficient — the vel-env reset z (~0.125) would
    clip the head through the ground at 90 deg pitch. With ``crouch_prob`` >
    0, an additional exclusive slice of envs is reset into a random
    mid-recovery crouch via ``set_random_crouch_state``.
    """
    if not isinstance(asset_cfg, SceneEntityCfg):
        raise TypeError("maybe_set_random_prone_orientation asset_cfg must be SceneEntityCfg")
    if prone_prob <= 0.0 and crouch_prob <= 0.0:
        return
    ids = resolve_env_ids(env, env_ids)
    if ids.size == 0:
        return
    # One draw partitions envs into exclusive prone / crouch / untouched slices.
    u = env.rng.random(ids.size)
    selected = ids[u < prone_prob]
    crouch_selected = ids[(u >= prone_prob) & (u < prone_prob + crouch_prob)]
    if selected.size > 0:
        set_random_prone_orientation(
            env, selected, asset_cfg=asset_cfg, face_down_prob=face_down_prob
        )
        # Override z so the prone body has head/neck clearance when settling.
        entity = cast("Entity", env.scene[asset_cfg.name])
        pose = entity.read_reset_root_pose(env_ids=selected)
        pose[:, 2] = env.rng.uniform(prone_z_min, prone_z_max, size=selected.size)
        entity.write_root_link_pose_to_sim(pose, env_ids=selected)
    if crouch_selected.size > 0:
        set_random_crouch_state(env, crouch_selected, asset_cfg=asset_cfg)


__all__ = [
    "com_upward_velocity",
    "fallen_state_penalty",
    "fallen_too_long",
    "feet_air_time_upright",
    "height_progress",
    "joint_torque_rate_l2",
    "maybe_set_random_prone_orientation",
    "recovery_success",
    "set_random_crouch_state",
    "set_random_prone_orientation",
    "upright_progress",
]
