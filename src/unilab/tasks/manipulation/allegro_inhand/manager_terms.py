"""Manager-Based terms for Allegro in-hand ball rotation.

Hydra owns the production task declaration.  These terms use only the public
Entity facade and the community manager lifecycle; they do not inspect backend
objects or physical state layouts.
"""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Real
from typing import TYPE_CHECKING, Any, Protocol, cast

import numpy as np
from etils import epath

from unilab.assets import ASSETS_ROOT_PATH
from unilab.dtype_config import get_global_dtype
from unilab.managers import ActionTerm, ActionTermCfg, ManagerTermBase, ManagerTermBaseCfg
from unilab.utils.geometry import np_normalize_axis, np_quat_angular_velocity_from_pair

if TYPE_CHECKING:
    from unilab.base.entity import Entity
    from unilab.managers._types import ManagerBasedRlEnv
    from unilab.managers.action_manager import ActionManager
    from unilab.managers.observation_manager import ObservationManager
    from unilab.managers.termination_manager import TerminationManager

    class _AllegroEnv(ManagerBasedRlEnv, Protocol):
        @property
        def common_step_counter(self) -> int: ...

        @property
        def action_manager(self) -> ActionManager: ...

        @property
        def observation_manager(self) -> ObservationManager: ...

        @property
        def termination_manager(self) -> TerminationManager: ...


def _real(
    term: str,
    name: str,
    value: Any,
    *,
    minimum: float | None = None,
    strict_minimum: bool = False,
) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise TypeError(f"{term} {name} must be a real number")
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{term} {name} must be finite")
    if minimum is not None and (result <= minimum if strict_minimum else result < minimum):
        relation = "greater than" if strict_minimum else "at least"
        raise ValueError(f"{term} {name} must be {relation} {minimum}")
    return result


def _name(term: str, name: str, value: Any) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{term} {name} must be a non-empty string")
    return value


def _pair(term: str, name: str, value: Any) -> tuple[float, float]:
    if isinstance(value, (str, bytes)) or not isinstance(value, (tuple, list)):
        raise TypeError(f"{term} {name} must be a numeric (min, max) pair")
    if len(value) != 2:
        raise ValueError(f"{term} {name} must contain two values")
    lower = _real(term, f"{name}[0]", value[0])
    upper = _real(term, f"{name}[1]", value[1])
    if lower > upper:
        raise ValueError(f"{term} {name} lower bound {lower} exceeds upper bound {upper}")
    return lower, upper


def _env_ids(env: ManagerBasedRlEnv, env_ids: np.ndarray | slice | None) -> np.ndarray:
    if env_ids is None:
        return np.arange(env.num_envs, dtype=np.int32)
    if isinstance(env_ids, slice):
        return np.arange(env.num_envs, dtype=np.int32)[env_ids]
    return env_ids


def _resolve_grasp_cache(cache_path: str) -> epath.Path:
    path = epath.Path(cache_path)
    if path.is_absolute() or path.exists():
        return path
    return epath.Path(ASSETS_ROOT_PATH / cache_path)


@dataclass(kw_only=True)
class AllegroIncrementalPositionActionCfg(ActionTermCfg):
    """Incremental position targets used by the original Allegro policy."""

    actuator_names: tuple[str, ...] | list[str]
    action_scale: float
    raw_action_clip: tuple[float, float] | list[float]

    def build(self, env: ManagerBasedRlEnv) -> AllegroIncrementalPositionAction:
        return AllegroIncrementalPositionAction(self, env)


class AllegroIncrementalPositionAction(ActionTerm):
    """Integrate clipped policy deltas into bounded hand-joint targets."""

    cfg: AllegroIncrementalPositionActionCfg
    _entity: Entity
    _raw_action: np.ndarray
    _clipped_action: np.ndarray
    _target: np.ndarray

    def __init__(self, cfg: AllegroIncrementalPositionActionCfg, env: ManagerBasedRlEnv):
        super().__init__(cfg, env)
        term = type(self).__name__
        if cfg.clip is not None:
            raise NotImplementedError(
                f"{term} does not support actuator-name clip; use raw_action_clip"
            )
        if isinstance(cfg.actuator_names, (str, bytes)) or not isinstance(
            cfg.actuator_names, (tuple, list)
        ):
            raise TypeError(f"{term} actuator_names must be a sequence of patterns")
        self._joint_ids, target_names = self._entity.find_joints_by_actuator_names(
            cfg.actuator_names
        )
        actuator_ids, actuator_names = self._entity.find_actuators(
            cfg.actuator_names, preserve_order=True
        )
        if len(self._joint_ids) != len(actuator_ids) or target_names != list(
            self._entity.joint_names[index] for index in self._joint_ids
        ):
            raise ValueError(f"{term} actuator-to-joint mapping is incomplete")
        if len(self._joint_ids) != 16:
            raise ValueError(f"{term} requires 16 hand actuators, got {len(self._joint_ids)}")
        if len(set(actuator_names)) != len(actuator_names):
            raise ValueError(f"{term} actuator selector resolved duplicate names")

        self._joint_ids_array = np.asarray(self._joint_ids, dtype=np.intp)
        self._joint_ids_array.setflags(write=False)
        local_actuator_ids = np.asarray(actuator_ids, dtype=np.intp)
        ranges = np.asarray(self._entity.data.actuator_ctrl_range, dtype=get_global_dtype())
        self._ctrl_lower = np.array(ranges[local_actuator_ids, 0], copy=True)
        self._ctrl_upper = np.array(ranges[local_actuator_ids, 1], copy=True)
        if np.any(self._ctrl_lower >= self._ctrl_upper):
            raise ValueError(f"{term} actuator control ranges must have lower < upper")

        self._scale = _real(term, "action_scale", cfg.action_scale, minimum=0.0)
        self._raw_clip = _pair(term, "raw_action_clip", cfg.raw_action_clip)
        dtype = get_global_dtype()
        self._raw_action = np.zeros((env.num_envs, len(self._joint_ids)), dtype=dtype)
        self._clipped_action = np.zeros_like(self._raw_action)
        self._target = np.asarray(
            self._entity.data.default_joint_pos[:, self._joint_ids_array], dtype=dtype
        ).copy()

    @property
    def action_dim(self) -> int:
        return int(self._raw_action.shape[1])

    @property
    def raw_action(self) -> np.ndarray:
        return self._raw_action

    @property
    def target(self) -> np.ndarray:
        return self._target

    @property
    def ctrl_lower(self) -> np.ndarray:
        return self._ctrl_lower

    @property
    def ctrl_upper(self) -> np.ndarray:
        return self._ctrl_upper

    @property
    def joint_ids(self) -> np.ndarray:
        return self._joint_ids_array

    def process_actions(self, actions: np.ndarray) -> None:
        if not isinstance(actions, np.ndarray):
            raise TypeError(f"expected np.ndarray actions, got {type(actions).__name__}")
        if actions.shape != self._raw_action.shape:
            raise ValueError(f"expected action shape {self._raw_action.shape}, got {actions.shape}")
        if not np.isfinite(actions).all():
            raise ValueError("received NaN or Inf actions")
        self._raw_action[:] = actions
        np.clip(actions, self._raw_clip[0], self._raw_clip[1], out=self._clipped_action)
        self._target += self._scale * self._clipped_action
        np.clip(self._target, self._ctrl_lower, self._ctrl_upper, out=self._target)

    def apply_actions(self) -> None:
        self._entity.set_joint_position_target(self._target, joint_ids=self._joint_ids_array)

    def reset(self, env_ids: np.ndarray | slice | None = None) -> None:
        ids = _env_ids(self._env, env_ids)
        self._raw_action[ids] = 0.0
        self._clipped_action[ids] = 0.0
        self._target[ids] = self._entity.data.joint_pos[ids][:, self._joint_ids_array]


class AllegroRotationObservation(ManagerTermBase):
    """One 35-D frame plus state shared by termination and reward terms."""

    _ALLOWED_PARAMS = frozenset(
        {
            "entity_name",
            "action_name",
            "joint_noise",
            "torque_estimate_kp",
            "torque_estimate_kd",
        }
    )

    def __init__(self, cfg: ManagerTermBaseCfg, env: _AllegroEnv):
        super().__init__(env)
        term = type(self).__name__
        unexpected = set(cfg.params) - self._ALLOWED_PARAMS
        if unexpected:
            raise TypeError(f"{term} received unsupported parameters: {sorted(unexpected)}")
        entity_name = _name(term, "entity_name", cfg.params.get("entity_name"))
        action_name = _name(term, "action_name", cfg.params.get("action_name"))
        self._entity = cast("Entity", env.scene[entity_name])
        action = env.action_manager.get_term(action_name)
        if not isinstance(action, AllegroIncrementalPositionAction):
            raise TypeError(
                f"{term} action {action_name!r} must be AllegroIncrementalPositionAction, "
                f"got {type(action).__name__}"
            )
        self._action = action
        self._joint_noise = _real(term, "joint_noise", cfg.params.get("joint_noise"), minimum=0.0)
        self._torque_kp = _real(
            term, "torque_estimate_kp", cfg.params.get("torque_estimate_kp"), minimum=0.0
        )
        self._torque_kd = _real(
            term, "torque_estimate_kd", cfg.params.get("torque_estimate_kd"), minimum=0.0
        )

        dtype = get_global_dtype()
        self.dof_pos = np.asarray(
            self._entity.data.joint_pos[:, self._action.joint_ids], dtype=dtype
        ).copy()
        self.dof_vel = np.zeros_like(self.dof_pos)
        self.ball_pos = np.asarray(self._entity.data.root_link_pos_w, dtype=dtype).copy()
        self.ball_quat = np.asarray(self._entity.data.root_link_quat_w, dtype=dtype).copy()
        self.ball_linvel = np.zeros_like(self.ball_pos)
        self.ball_angvel = np.zeros_like(self.ball_pos)
        self.torques = np.zeros_like(self.dof_pos)
        self.init_pose = self.dof_pos.copy()
        self._previous_dof_pos = self.dof_pos.copy()
        self._previous_ball_pos = self.ball_pos.copy()
        self._previous_ball_quat = self.ball_quat.copy()
        self._just_reset = np.ones(env.num_envs, dtype=np.bool_)
        self._last_counter = int(env.common_step_counter)

        self._dof_mid = (self._action.ctrl_upper + self._action.ctrl_lower) / 2.0
        self._dof_range = self._action.ctrl_upper - self._action.ctrl_lower

    @property
    def last_counter(self) -> int:
        return self._last_counter

    def reset(self, env_ids: np.ndarray | slice | None = None) -> None:
        ids = _env_ids(self._env, env_ids)
        dof_pos = np.asarray(
            self._entity.data.joint_pos[:, self._action.joint_ids], dtype=get_global_dtype()
        )
        ball_pos = np.asarray(self._entity.data.root_link_pos_w, dtype=get_global_dtype())
        ball_quat = np.asarray(self._entity.data.root_link_quat_w, dtype=get_global_dtype())
        self.dof_pos[ids] = dof_pos[ids]
        self.dof_vel[ids] = 0.0
        self.ball_pos[ids] = ball_pos[ids]
        self.ball_quat[ids] = ball_quat[ids]
        self.ball_linvel[ids] = 0.0
        self.ball_angvel[ids] = 0.0
        self.torques[ids] = 0.0
        self.init_pose[ids] = dof_pos[ids]
        self._previous_dof_pos[ids] = dof_pos[ids]
        self._previous_ball_pos[ids] = ball_pos[ids]
        self._previous_ball_quat[ids] = ball_quat[ids]
        self._just_reset[ids] = True
        self._last_counter = int(cast("_AllegroEnv", self._env).common_step_counter)

    def snapshot(self, env: _AllegroEnv) -> AllegroRotationObservation:
        counter = int(env.common_step_counter)
        if counter == self._last_counter:
            return self
        if counter != self._last_counter + 1:
            raise RuntimeError(
                f"AllegroRotationObservation missed a control-step update: "
                f"last={self._last_counter}, current={counter}"
            )

        dtype = get_global_dtype()
        dof_pos = np.asarray(self._entity.data.joint_pos[:, self._action.joint_ids], dtype=dtype)
        ball_pos = np.asarray(self._entity.data.root_link_pos_w, dtype=dtype)
        ball_quat = np.asarray(self._entity.data.root_link_quat_w, dtype=dtype)
        np.subtract(dof_pos, self._previous_dof_pos, out=self.dof_vel)
        self.dof_vel /= env.step_dt
        np.subtract(ball_pos, self._previous_ball_pos, out=self.ball_linvel)
        self.ball_linvel /= env.step_dt
        self.ball_angvel[:] = np_quat_angular_velocity_from_pair(
            ball_quat, self._previous_ball_quat, env.step_dt
        )
        self.dof_pos[:] = dof_pos
        self.ball_pos[:] = ball_pos
        self.ball_quat[:] = ball_quat
        self.torques[:] = self._torque_kp * (self._action.target - self.dof_pos)
        self.torques -= self._torque_kd * self.dof_vel
        np.clip(self.torques, -0.5, 0.5, out=self.torques)
        self._previous_dof_pos[:] = dof_pos
        self._previous_ball_pos[:] = ball_pos
        self._previous_ball_quat[:] = ball_quat
        self._last_counter = counter
        return self

    def __call__(self, env: _AllegroEnv, **params: Any) -> np.ndarray:
        del params
        self.snapshot(env)
        dof_pos_norm = 2.0 * (self.dof_pos - self._dof_mid) / (self._dof_range + 1.0e-8)
        if self._joint_noise > 0.0:
            active = ~self._just_reset
            if np.any(active):
                dof_pos_norm = dof_pos_norm.copy()
                dof_pos_norm[active] += env.rng.uniform(
                    -self._joint_noise,
                    self._joint_noise,
                    size=(int(np.count_nonzero(active)), self._action.action_dim),
                )
        self._just_reset[:] = False
        return np.concatenate(
            (dof_pos_norm, self._action.target, self.ball_pos),
            axis=1,
            dtype=get_global_dtype(),
        )


class AllegroDropTermination(ManagerTermBase):
    """Termination-owned drop state, computed before reward terms."""

    def __init__(self, cfg: ManagerTermBaseCfg, env: _AllegroEnv):
        super().__init__(env)
        term = type(self).__name__
        allowed = {"observation_group", "observation_term", "minimum_ball_height"}
        unexpected = set(cfg.params) - allowed
        if unexpected:
            raise TypeError(f"{term} received unsupported parameters: {sorted(unexpected)}")
        group = _name(term, "observation_group", cfg.params.get("observation_group"))
        name = _name(term, "observation_term", cfg.params.get("observation_term"))
        observation = env.observation_manager.get_term_cfg(group, name).func
        if not isinstance(observation, AllegroRotationObservation):
            raise TypeError(
                f"{term} observation {group}/{name} must be AllegroRotationObservation, "
                f"got {type(observation).__name__}"
            )
        self.observation = observation
        self._minimum_height = _real(
            term, "minimum_ball_height", cfg.params.get("minimum_ball_height")
        )
        self.dropped = np.zeros(env.num_envs, dtype=np.bool_)
        self._last_counter = int(env.common_step_counter)

    @property
    def last_counter(self) -> int:
        return self._last_counter

    def reset(self, env_ids: np.ndarray | slice | None = None) -> None:
        self.dropped[_env_ids(self._env, env_ids)] = False
        self._last_counter = int(cast("_AllegroEnv", self._env).common_step_counter)

    def __call__(self, env: _AllegroEnv, **params: Any) -> np.ndarray:
        del params
        self.observation.snapshot(env)
        self.dropped[:] = self.observation.ball_pos[:, 2] < self._minimum_height
        self._last_counter = int(env.common_step_counter)
        return self.dropped


def _rotation_state(
    env: _AllegroEnv,
    state_term_name: str,
) -> tuple[AllegroDropTermination, AllegroRotationObservation]:
    name = _name("Allegro reward", "state_term_name", state_term_name)
    state = env.termination_manager.get_term_cfg(name).func
    if not isinstance(state, AllegroDropTermination):
        raise TypeError(
            f"Allegro reward termination term {name!r} must be AllegroDropTermination, "
            f"got {type(state).__name__}"
        )
    if state.last_counter != int(env.common_step_counter):
        raise RuntimeError(
            f"Allegro reward state {name!r} was not computed for control step "
            f"{env.common_step_counter}"
        )
    return state, state.observation


class AllegroRotateReward(ManagerTermBase):
    """Reward angular velocity projected onto a cold-path-normalized axis."""

    def __init__(self, cfg: ManagerTermBaseCfg, env: _AllegroEnv):
        super().__init__(env)
        term = type(self).__name__
        allowed = {"state_term_name", "rotation_axis", "clip_min", "clip_max"}
        unexpected = set(cfg.params) - allowed
        if unexpected:
            raise TypeError(f"{term} received unsupported parameters: {sorted(unexpected)}")
        self._state_term_name = _name(term, "state_term_name", cfg.params.get("state_term_name"))
        try:
            axis = np.asarray(cfg.params.get("rotation_axis"), dtype=get_global_dtype())
        except (TypeError, ValueError) as exc:
            raise TypeError(f"{term} rotation_axis must contain three numeric values") from exc
        if axis.shape != (3,) or not np.isfinite(axis).all():
            raise ValueError(f"{term} rotation_axis must be a finite 3-D vector")
        self._axis = np.asarray(np_normalize_axis(axis), dtype=get_global_dtype())
        self._clip = _pair(
            term,
            "clip",
            (cfg.params.get("clip_min"), cfg.params.get("clip_max")),
        )

    def __call__(self, env: _AllegroEnv, **params: Any) -> np.ndarray:
        del params
        _, state = _rotation_state(env, self._state_term_name)
        return np.asarray(
            np.clip(state.ball_angvel @ self._axis, self._clip[0], self._clip[1]),
            dtype=get_global_dtype(),
        )


def object_linear_velocity_l1(env: _AllegroEnv, state_term_name: str) -> np.ndarray:
    _, state = _rotation_state(env, state_term_name)
    return np.asarray(np.sum(np.abs(state.ball_linvel), axis=1), dtype=get_global_dtype())


def hand_pose_deviation_l2(env: _AllegroEnv, state_term_name: str) -> np.ndarray:
    _, state = _rotation_state(env, state_term_name)
    return np.asarray(
        np.sum(np.square(state.dof_pos - state.init_pose), axis=1), dtype=get_global_dtype()
    )


def estimated_torque_l2(env: _AllegroEnv, state_term_name: str) -> np.ndarray:
    _, state = _rotation_state(env, state_term_name)
    return np.asarray(np.sum(np.square(state.torques), axis=1), dtype=get_global_dtype())


def estimated_work_l2(env: _AllegroEnv, state_term_name: str) -> np.ndarray:
    _, state = _rotation_state(env, state_term_name)
    work = np.sum(state.torques * state.dof_vel, axis=1)
    return np.asarray(np.square(work), dtype=get_global_dtype())


def dropped(env: _AllegroEnv, state_term_name: str) -> np.ndarray:
    state, _ = _rotation_state(env, state_term_name)
    return np.asarray(state.dropped, dtype=get_global_dtype())


class AllegroHandBallReset(ManagerTermBase):
    """Reset hand joints and the ball root without exposing qpos layout."""

    def __init__(self, cfg: ManagerTermBaseCfg, env: ManagerBasedRlEnv):
        super().__init__(env)
        term = type(self).__name__
        allowed = {
            "entity_name",
            "grasp_cache_path",
            "joint_noise",
            "ball_velocity_noise",
            "ball_z_offset",
        }
        unexpected = set(cfg.params) - allowed
        if unexpected:
            raise TypeError(f"{term} received unsupported parameters: {sorted(unexpected)}")
        entity_name = _name(term, "entity_name", cfg.params.get("entity_name"))
        self._entity = cast("Entity", env.scene[entity_name])
        if self._entity.num_joints != 16:
            raise ValueError(f"{term} requires 16 hand joints, got {self._entity.num_joints}")
        if self._entity.data.default_root_state.shape != (env.num_envs, 13):
            raise ValueError(f"{term} requires a 13-D floating ball root for every environment")
        self._joint_noise = _real(term, "joint_noise", cfg.params.get("joint_noise"), minimum=0.0)
        self._ball_velocity_noise = _real(
            term,
            "ball_velocity_noise",
            cfg.params.get("ball_velocity_noise"),
            minimum=0.0,
        )
        self._ball_z_offset = _real(term, "ball_z_offset", cfg.params.get("ball_z_offset"))

        cache_value = cfg.params.get("grasp_cache_path")
        self._grasp_cache: np.ndarray | None = None
        if cache_value is not None:
            cache_path = _resolve_grasp_cache(_name(term, "grasp_cache_path", cache_value))
            if not cache_path.exists():
                raise FileNotFoundError(
                    f"{term} configured grasp cache does not exist: {cache_path}. "
                    "Set grasp_cache_path to null to explicitly use the model home pose, "
                    "or generate a cache with `uv run train --algo ppo "
                    "--task allegro_inhand_grasp --sim mujoco training.no_play=true`."
                )
            cache = np.asarray(np.load(cache_path), dtype=np.float64)
            if cache.ndim != 2 or cache.shape[1] != 23 or cache.shape[0] == 0:
                raise ValueError(
                    f"{term} grasp cache {cache_path} must have shape (N, 23), got {cache.shape}"
                )
            if not np.isfinite(cache).all():
                raise ValueError(f"{term} grasp cache {cache_path} contains NaN or Inf")
            self._grasp_cache = cache

        ranges = np.asarray(self._entity.data.actuator_ctrl_range, dtype=np.float64)
        if ranges.shape != (16, 2):
            raise ValueError(f"{term} actuator control range must have shape (16, 2)")
        self._ctrl_lower = ranges[:, 0]
        self._ctrl_upper = ranges[:, 1]

    def __call__(
        self,
        env: ManagerBasedRlEnv,
        env_ids: np.ndarray | None,
        **params: Any,
    ) -> None:
        del params
        ids = _env_ids(env, env_ids)
        count = ids.size
        dtype = get_global_dtype()
        root_state = np.array(self._entity.data.default_root_state[ids], copy=True)
        if self._grasp_cache is not None:
            rows = self._grasp_cache[env.rng.integers(0, self._grasp_cache.shape[0], size=count)]
            joint_pos = np.array(rows[:, :16], copy=True)
            root_state[:, :3] = rows[:, 16:19]
            root_state[:, 3:7] = rows[:, 19:23]
        else:
            joint_pos = np.array(self._entity.data.default_joint_pos[ids], copy=True)
            if self._joint_noise > 0.0:
                joint_pos += env.rng.uniform(
                    -self._joint_noise, self._joint_noise, size=joint_pos.shape
                )
            root_state[:, 2] += self._ball_z_offset
        np.clip(joint_pos, self._ctrl_lower, self._ctrl_upper, out=joint_pos)
        joint_vel = np.zeros_like(joint_pos)
        root_state[:, 7:] = 0.0
        if self._ball_velocity_noise > 0.0:
            root_state[:, 7:10] = env.rng.uniform(
                -self._ball_velocity_noise,
                self._ball_velocity_noise,
                size=(count, 3),
            )
        self._entity.write_joint_state_to_sim(
            np.asarray(joint_pos, dtype=dtype),
            np.asarray(joint_vel, dtype=dtype),
            env_ids=ids,
        )
        self._entity.write_root_link_pose_to_sim(
            np.asarray(root_state[:, :7], dtype=dtype), env_ids=ids
        )
        self._entity.write_root_link_velocity_to_sim(
            np.asarray(root_state[:, 7:], dtype=dtype), env_ids=ids
        )


__all__ = [
    "AllegroDropTermination",
    "AllegroHandBallReset",
    "AllegroIncrementalPositionAction",
    "AllegroIncrementalPositionActionCfg",
    "AllegroRotateReward",
    "AllegroRotationObservation",
    "dropped",
    "estimated_torque_l2",
    "estimated_work_l2",
    "hand_pose_deviation_l2",
    "object_linear_velocity_l1",
]
