"""Task-owned Manager-Based terms for the quadruped locomotion pilots.

The equations come from UniLab's existing Go1/Go2 joystick tasks.  The adaptation
uses community ``func + params`` terms, NumPy, and the base-owned sensor facade.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar, Protocol

import numpy as np

from unilab.dtype_config import get_global_dtype
from unilab.managers.manager_base import ManagerTermBase, ManagerTermBaseCfg

if TYPE_CHECKING:
    from unilab.managers._types import ManagerBasedRlEnv, ManagerSensorView

    class _GaitEnv(ManagerBasedRlEnv, Protocol):
        @property
        def common_step_counter(self) -> int: ...


_OFFSETS = (0.0, 0.5, 0.5, 0.0)


def _real(
    term: str,
    name: str,
    value: Any,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
    strict_minimum: bool = False,
) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.number)):
        raise TypeError(f"{term} {name} must be a real number")
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{term} {name} must be finite")
    if minimum is not None and (result <= minimum if strict_minimum else result < minimum):
        relation = "greater than" if strict_minimum else "at least"
        raise ValueError(f"{term} {name} must be {relation} {minimum}")
    if maximum is not None and result > maximum:
        raise ValueError(f"{term} {name} must be at most {maximum}")
    return result


def _offsets(term: str, value: Any) -> np.ndarray:
    if isinstance(value, (str, bytes)) or not isinstance(value, (tuple, list, np.ndarray)):
        raise TypeError(f"{term} phase_offsets must be a sequence of four real numbers")
    if isinstance(value, np.ndarray) and value.ndim != 1:
        raise ValueError(f"{term} phase_offsets must be one-dimensional, got {value.shape}")
    items = list(value)
    if len(items) != 4:
        raise ValueError(f"{term} phase_offsets must contain 4 values, got {len(items)}")
    result = np.asarray(
        [_real(term, f"phase_offsets[{index}]", item) for index, item in enumerate(items)],
        dtype=get_global_dtype(),
    )
    result.setflags(write=False)
    return result


def _names(term: str, value: Any) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, (tuple, list)):
        raise TypeError(f"{term} sensor_names must be a sequence of four strings")
    names = tuple(value)
    if len(names) != 4:
        raise ValueError(f"{term} sensor_names must contain 4 names, got {len(names)}")
    if any(not isinstance(name, str) or not name for name in names):
        raise ValueError(f"{term} sensor_names must contain non-empty strings")
    if len(set(names)) != 4:
        raise ValueError(f"{term} sensor_names must be unique: {names}")
    return names


class _GaitTerm(ManagerTermBase):
    _allowed_params: ClassVar[frozenset[str]] = frozenset({"frequency", "phase_offsets"})

    def __init__(self, cfg: ManagerTermBaseCfg, env: _GaitEnv):
        super().__init__(env)
        unexpected = set(cfg.params) - self._allowed_params
        if unexpected:
            raise TypeError(f"{self.name} received unsupported parameters: {sorted(unexpected)}")
        self._frequency = _real(
            self.name, "frequency", cfg.params.get("frequency", 2.0), minimum=0.0
        )
        self._offsets = _offsets(self.name, cfg.params.get("phase_offsets", _OFFSETS))
        self._step_dt = _real(self.name, "step_dt", env.step_dt, minimum=0.0, strict_minimum=True)
        self._phase_value = np.asarray(0.0, dtype=get_global_dtype())
        self._last_counter = 0
        self._advance_to(self._counter(env))

    def _counter(self, env: _GaitEnv) -> int:
        counter = env.common_step_counter
        if isinstance(counter, (bool, np.bool_)) or not isinstance(counter, (int, np.integer)):
            raise TypeError(f"{self.name} common_step_counter must be an integer")
        if counter < 0:
            raise ValueError(f"{self.name} common_step_counter must be non-negative")
        return int(counter)

    def _advance_to(self, counter: int) -> None:
        delta = counter - self._last_counter
        if delta < 0:
            raise ValueError(f"{self.name} common_step_counter cannot move backwards")
        increment = np.asarray(self._step_dt * self._frequency, dtype=get_global_dtype())
        if delta == 1:  # Hot path: preserve the legacy float32 iterative phase exactly.
            self._phase_value = np.fmod(self._phase_value + increment, 1.0)
        elif delta > 1:  # Cold catch-up for a term constructed or inspected between steps.
            for _ in range(delta):
                self._phase_value = np.fmod(self._phase_value + increment, 1.0)
        self._last_counter = counter

    def _phase(self, env: _GaitEnv) -> np.ndarray:
        self._advance_to(self._counter(env))
        phase = np.remainder(self._phase_value + self._offsets, 1.0).astype(
            get_global_dtype(), copy=False
        )
        return np.broadcast_to(phase, (env.num_envs, 4)).copy()


class quadruped_gait_phase(_GaitTerm):
    """Four-foot phase observation with the legacy diagonal-trot ordering."""

    def __call__(self, env: _GaitEnv, **params: Any) -> np.ndarray:
        del params
        return self._phase(env)


class _FootSensorTerm(_GaitTerm):
    def __init__(self, cfg: ManagerTermBaseCfg, env: _GaitEnv):
        super().__init__(cfg, env)
        sensor_names = _names(self.name, cfg.params.get("sensor_names"))
        try:
            self._view = env.scene.bind_sensor_data(sensor_names)
        except (KeyError, TypeError, ValueError, NotImplementedError) as exc:
            raise type(exc)(
                f"Manager term '{self.name}' named-foot-sensor capability could not be "
                f"materialized for {sensor_names}: {exc}"
            ) from exc

    def _read(self) -> np.ndarray:
        try:
            return self._view.read()
        except (KeyError, TypeError, ValueError, NotImplementedError) as exc:
            raise type(exc)(
                f"Manager term '{self.name}' named-foot-sensor capability failed on "
                f"backend '{self._view.backend_type}': {exc}"
            ) from exc


class feet_phase_contact(_FootSensorTerm):
    """Reward foot contact matching the configured stance portion of gait phase."""

    _allowed_params = _GaitTerm._allowed_params | {
        "sensor_names",
        "contact_threshold",
        "stance_threshold",
    }

    def __init__(self, cfg: ManagerTermBaseCfg, env: _GaitEnv):
        super().__init__(cfg, env)
        self._contact_threshold = _real(
            self.name, "contact_threshold", cfg.params.get("contact_threshold", 0.1), minimum=0.0
        )
        self._stance_threshold = _real(
            self.name,
            "stance_threshold",
            cfg.params.get("stance_threshold", 0.6),
            minimum=0.0,
            maximum=1.0,
        )
        if any(width not in (1, 3) for width in self._view.dimensions):
            raise ValueError(
                f"{self.name} contact sensors must each expose 1-D found or 3-D force; "
                f"received {self._view.dimensions} on backend '{self._view.backend_type}'"
            )
        starts = np.cumsum((0, *self._view.dimensions[:-1]), dtype=np.int64)
        self._columns = starts + [0 if width == 1 else 2 for width in self._view.dimensions]

    def __call__(self, env: _GaitEnv, **params: Any) -> np.ndarray:
        del params
        contact = self._read()[:, self._columns] > self._contact_threshold
        expected = self._phase(env) < self._stance_threshold
        if self._frequency < 1.0e-8:
            expected.fill(True)
        return np.mean(contact == expected, axis=1).astype(get_global_dtype(), copy=False)


class feet_phase_swing_height(_FootSensorTerm):
    """Reward foot height near a target during the configured swing phase."""

    _allowed_params = _GaitTerm._allowed_params | {
        "sensor_names",
        "target_height",
        "kernel",
        "swing_start",
    }

    def __init__(self, cfg: ManagerTermBaseCfg, env: _GaitEnv):
        super().__init__(cfg, env)
        self._target = _real(
            self.name, "target_height", cfg.params.get("target_height", 0.1), minimum=0.0
        )
        self._kernel = _real(
            self.name,
            "kernel",
            cfg.params.get("kernel", 0.01),
            minimum=0.0,
            strict_minimum=True,
        )
        self._swing_start = _real(
            self.name,
            "swing_start",
            cfg.params.get("swing_start", 0.6),
            minimum=0.0,
            maximum=1.0,
        )
        if self._view.dimensions != (3, 3, 3, 3):
            raise ValueError(
                f"{self.name} position sensors must each expose 3-D xyz; received "
                f"{self._view.dimensions} on backend '{self._view.backend_type}'"
            )

    def __call__(self, env: _GaitEnv, **params: Any) -> np.ndarray:
        del params
        heights = self._read()[:, (2, 5, 8, 11)]
        swing = self._phase(env) >= self._swing_start
        reward = np.exp(-np.square(heights - self._target) / self._kernel) * swing
        return np.mean(reward, axis=1).astype(get_global_dtype(), copy=False)


__all__ = ["feet_phase_contact", "feet_phase_swing_height", "quadruped_gait_phase"]
