"""Manager-Based terms and registry owner for Allegro grasp generation."""

from __future__ import annotations

from numbers import Integral, Real
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, cast

import numpy as np

from unilab.assets import ASSETS_ROOT_PATH
from unilab.base import registry
from unilab.base.run_control import RunComplete
from unilab.dtype_config import get_global_dtype
from unilab.envs import ManagerBasedRlEnvCfg, make_manager_based_rl_env
from unilab.managers import ManagerTermBase, ManagerTermBaseCfg, RecorderTerm, RecorderTermCfg

from .manager_terms import AllegroRotationObservation

if TYPE_CHECKING:
    from unilab.base.entity import Entity
    from unilab.managers._types import ManagerBasedRlEnv
    from unilab.managers.action_manager import ActionManager
    from unilab.managers.observation_manager import ObservationManager
    from unilab.managers.termination_manager import TerminationManager

    class _GraspEnv(ManagerBasedRlEnv, Protocol):
        @property
        def common_step_counter(self) -> int: ...

        @property
        def action_manager(self) -> ActionManager: ...

        @property
        def observation_manager(self) -> ObservationManager: ...

        @property
        def termination_manager(self) -> TerminationManager: ...

        @property
        def reset_terminated(self) -> np.ndarray: ...

        @property
        def reset_time_outs(self) -> np.ndarray: ...

        @property
        def extras(self) -> dict[str, Any]: ...


def _name(term: str, field: str, value: Any) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{term} {field} must be a non-empty string")
    return value


def _names(term: str, field: str, value: Any) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, (tuple, list)):
        raise TypeError(f"{term} {field} must be a sequence of names")
    result = tuple(_name(term, field, item) for item in value)
    if not result:
        raise ValueError(f"{term} {field} must not be empty")
    if len(set(result)) != len(result):
        raise ValueError(f"{term} {field} must contain unique names")
    return result


def _real(term: str, field: str, value: Any, *, positive: bool = False) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise TypeError(f"{term} {field} must be a real number")
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{term} {field} must be finite")
    if positive and result <= 0.0:
        raise ValueError(f"{term} {field} must be positive")
    return result


def _positive_int(term: str, field: str, value: Any) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise TypeError(f"{term} {field} must be an integer")
    result = int(value)
    if result <= 0:
        raise ValueError(f"{term} {field} must be positive")
    return result


def _bool(term: str, field: str, value: Any) -> bool:
    if not isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{term} {field} must be boolean")
    return bool(value)


class AllegroGraspQualityTermination(ManagerTermBase):
    """Reject timeouts that are not stable multi-finger grasps."""

    _ALLOWED_PARAMS = frozenset(
        {
            "entity_name",
            "observation_group",
            "observation_term",
            "fingertip_body_names",
            "contact_sensor_names",
            "max_fingertip_distance",
            "minimum_contacts",
            "minimum_ball_height",
            "enabled",
        }
    )

    def __init__(self, cfg: ManagerTermBaseCfg, env: _GraspEnv):
        super().__init__(env)
        term = type(self).__name__
        unexpected = set(cfg.params) - self._ALLOWED_PARAMS
        if unexpected:
            raise TypeError(f"{term} received unsupported parameters: {sorted(unexpected)}")

        entity_name = _name(term, "entity_name", cfg.params.get("entity_name"))
        self._entity = cast("Entity", env.scene[entity_name])
        fingertip_names = _names(
            term, "fingertip_body_names", cfg.params.get("fingertip_body_names")
        )
        fingertip_ids, matched_names = self._entity.find_bodies(
            fingertip_names, preserve_order=True
        )
        if tuple(matched_names) != fingertip_names:
            raise ValueError(
                f"{term} fingertip body order mismatch: expected {fingertip_names}, "
                f"got {tuple(matched_names)}"
            )
        self._fingertip_ids = np.asarray(fingertip_ids, dtype=np.intp)
        self._fingertip_ids.setflags(write=False)

        contact_names = _names(term, "contact_sensor_names", cfg.params.get("contact_sensor_names"))
        self._contact_view = env.scene.bind_sensor_data(contact_names)
        if self._contact_view.dimensions != (1,) * len(contact_names):
            raise ValueError(
                f"{term} contact sensors must each be scalar, got "
                f"{dict(zip(contact_names, self._contact_view.dimensions, strict=True))}"
            )

        group = _name(term, "observation_group", cfg.params.get("observation_group"))
        name = _name(term, "observation_term", cfg.params.get("observation_term"))
        observation = env.observation_manager.get_term_cfg(group, name).func
        if not isinstance(observation, AllegroRotationObservation):
            raise TypeError(
                f"{term} observation {group}/{name} must be AllegroRotationObservation, "
                f"got {type(observation).__name__}"
            )
        self.observation = observation
        self._maximum_distance = _real(
            term,
            "max_fingertip_distance",
            cfg.params.get("max_fingertip_distance"),
            positive=True,
        )
        self._minimum_contacts = _positive_int(
            term, "minimum_contacts", cfg.params.get("minimum_contacts")
        )
        if self._minimum_contacts > len(contact_names):
            raise ValueError(
                f"{term} minimum_contacts={self._minimum_contacts} exceeds "
                f"{len(contact_names)} configured contact sensors"
            )
        self._minimum_height = _real(
            term, "minimum_ball_height", cfg.params.get("minimum_ball_height")
        )
        self._enabled = _bool(term, "enabled", cfg.params.get("enabled"))

        self.fingertips_close = np.zeros(env.num_envs, dtype=np.bool_)
        self.enough_contacts = np.zeros(env.num_envs, dtype=np.bool_)
        self.ball_held = np.zeros(env.num_envs, dtype=np.bool_)
        self.valid = np.zeros(env.num_envs, dtype=np.bool_)
        self._disabled = np.zeros(env.num_envs, dtype=np.bool_)
        self._last_counter = int(env.common_step_counter)

    @property
    def last_counter(self) -> int:
        return self._last_counter

    def reset(self, env_ids: np.ndarray | slice | None = None) -> None:
        ids = slice(None) if env_ids is None else env_ids
        self.fingertips_close[ids] = False
        self.enough_contacts[ids] = False
        self.ball_held[ids] = False
        self.valid[ids] = False
        self._last_counter = int(cast("_GraspEnv", self._env).common_step_counter)

    def __call__(self, env: _GraspEnv, **params: Any) -> np.ndarray:
        del params
        self.observation.snapshot(env)
        fingertip_pos = self._entity.data.body_link_pos_w[:, self._fingertip_ids]
        distance = np.linalg.norm(fingertip_pos - self.observation.ball_pos[:, None, :], axis=-1)
        self.fingertips_close[:] = np.all(distance < self._maximum_distance, axis=1)
        contacts = self._contact_view.read()
        self.enough_contacts[:] = np.count_nonzero(contacts > 0.5, axis=1) >= self._minimum_contacts
        self.ball_held[:] = self.observation.ball_pos[:, 2] > self._minimum_height
        np.logical_and(self.fingertips_close, self.enough_contacts, out=self.valid)
        np.logical_and(self.valid, self.ball_held, out=self.valid)
        self._last_counter = int(env.common_step_counter)
        if not self._enabled:
            return self._disabled
        return np.logical_not(self.valid)


class AllegroGraspQualityMetric(ManagerTermBase):
    """Expose one cached quality condition through the community MetricsManager."""

    def __init__(self, cfg: ManagerTermBaseCfg, env: _GraspEnv):
        super().__init__(env)
        term_name = type(self).__name__
        unexpected = set(cfg.params) - {"quality_term_name", "condition"}
        if unexpected:
            raise TypeError(f"{term_name} received unsupported parameters: {sorted(unexpected)}")
        quality_name = _name(term_name, "quality_term_name", cfg.params.get("quality_term_name"))
        quality = env.termination_manager.get_term_cfg(quality_name).func
        if not isinstance(quality, AllegroGraspQualityTermination):
            raise TypeError(
                f"{term_name} termination term {quality_name!r} must be "
                f"AllegroGraspQualityTermination, got {type(quality).__name__}"
            )
        condition = _name(term_name, "condition", cfg.params.get("condition"))
        values = {
            "fingertips_close": quality.fingertips_close,
            "enough_contacts": quality.enough_contacts,
            "ball_held": quality.ball_held,
            "valid": quality.valid,
        }
        try:
            self._value = values[condition]
        except KeyError:
            raise ValueError(
                f"{term_name} condition must be one of {sorted(values)}, got {condition!r}"
            ) from None
        self._quality = quality
        self._quality_name = quality_name

    def __call__(self, env: _GraspEnv, **params: Any) -> np.ndarray:
        del params
        if self._quality.last_counter != int(env.common_step_counter):
            raise RuntimeError(
                f"{type(self).__name__} term {self._quality_name!r} was not computed for "
                f"control step {env.common_step_counter}"
            )
        return np.asarray(self._value, dtype=get_global_dtype())


class AllegroGraspRecorder(RecorderTerm):
    """Collect successful timeout states and persist the canonical 23-D cache."""

    _ALLOWED_PARAMS = frozenset(
        {"quality_term_name", "output_path", "collection_target", "auto_save"}
    )

    def __init__(self, cfg: RecorderTermCfg, env: _GraspEnv):
        super().__init__(cfg, env)
        term = type(self).__name__
        unexpected = set(cfg.params) - self._ALLOWED_PARAMS
        if unexpected:
            raise TypeError(f"{term} received unsupported parameters: {sorted(unexpected)}")

        quality_name = _name(term, "quality_term_name", cfg.params.get("quality_term_name"))
        quality = env.termination_manager.get_term_cfg(quality_name).func
        if not isinstance(quality, AllegroGraspQualityTermination):
            raise TypeError(
                f"{term} termination term {quality_name!r} must be "
                f"AllegroGraspQualityTermination, got {type(quality).__name__}"
            )
        self._quality = quality
        output = Path(_name(term, "output_path", cfg.params.get("output_path")))
        self._output_path = output if output.is_absolute() else Path(ASSETS_ROOT_PATH) / output
        self._target = _positive_int(term, "collection_target", cfg.params.get("collection_target"))
        self._auto_save = _bool(term, "auto_save", cfg.params.get("auto_save"))
        self._saved_states: list[np.ndarray] = []
        self._cache_saved = False
        self._target_notified = False

    @property
    def total_saved_grasps(self) -> int:
        return int(sum(states.shape[0] for states in self._saved_states))

    @property
    def cache_saved(self) -> bool:
        return self._cache_saved

    @property
    def output_path(self) -> Path:
        return self._output_path

    def _log(self, name: str, value: float) -> None:
        env = cast("_GraspEnv", self._env)
        log = env.extras.setdefault("log", {})
        log[name] = value

    def _save_cache(self, *, force: bool = False) -> None:
        if self._cache_saved:
            return
        total = self.total_saved_grasps
        if not force and total < self._target:
            return
        if total == 0:
            return

        all_states = np.concatenate(self._saved_states, axis=0).astype(np.float32)
        all_states = all_states[: self._target]
        if all_states.ndim != 2 or all_states.shape[1] != 23:
            raise ValueError(
                f"{type(self).__name__} collected cache must have shape (N, 23), "
                f"got {all_states.shape}"
            )
        if not np.isfinite(all_states).all():
            raise ValueError(f"{type(self).__name__} collected cache contains NaN or Inf")
        self._output_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(self._output_path, all_states)
        self._cache_saved = True
        self._log("grasp_cache/saved", 1.0)
        self._log("grasp_cache/num_states", float(all_states.shape[0]))

    def _stop_collection(self) -> None:
        if self._target_notified or self.total_saved_grasps < self._target:
            return
        total = self.total_saved_grasps
        self._target_notified = True
        self._log("grasp/target_reached", 1.0)
        print(
            "[AllegroInhandRotationGrasp] Grasp collection target reached "
            f"({total}/{self._target}). Collection completed."
        )
        raise RunComplete(
            reason="grasp_collection_target_reached",
            summary={
                "collected_grasps": total,
                "saved_grasps": min(total, self._target),
                "grasp_collection_target": self._target,
            },
        )

    def record_pre_reset(self, env_ids: np.ndarray) -> None:
        env = cast("_GraspEnv", self._env)
        ids = np.asarray(env_ids, dtype=np.intp)
        success = env.reset_time_outs[ids] & ~env.reset_terminated[ids]
        success_ids = ids[np.flatnonzero(success)]
        if success_ids.size == 0:
            return
        if self._quality.last_counter != int(env.common_step_counter):
            raise RuntimeError(
                f"{type(self).__name__} quality state was not computed for control step "
                f"{env.common_step_counter}"
            )
        state = self._quality.observation
        rows = np.concatenate(
            (
                state.dof_pos[success_ids],
                state.ball_pos[success_ids],
                state.ball_quat[success_ids],
            ),
            axis=1,
            dtype=np.float32,
        )
        if rows.shape != (success_ids.size, 23):
            raise ValueError(
                f"{type(self).__name__} expected collected rows shape "
                f"({success_ids.size}, 23), got {rows.shape}"
            )
        self._saved_states.append(rows)
        self._save_cache()
        self._stop_collection()
        self._log("grasp/cache_size", float(self.total_saved_grasps))

    def close(self) -> None:
        self._save_cache(force=self._auto_save)


registry.register_env_config("AllegroInhandRotationGrasp", ManagerBasedRlEnvCfg)
registry.register_env("AllegroInhandRotationGrasp", make_manager_based_rl_env, sim_backend="mujoco")
registry.register_env("AllegroInhandRotationGrasp", make_manager_based_rl_env, sim_backend="motrix")


__all__ = [
    "AllegroGraspQualityMetric",
    "AllegroGraspQualityTermination",
    "AllegroGraspRecorder",
]
