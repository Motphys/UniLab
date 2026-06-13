from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field

import numpy as np

SensorPacket = dict[str, np.ndarray]


@dataclass(frozen=True)
class DrakePoolOutput:
    """Batched Drake rollout output at the UniLab backend boundary."""

    state: np.ndarray
    sensor: SensorPacket = field(default_factory=dict)
    timing: dict[str, float] = field(default_factory=dict)


StepImpl = Callable[
    [np.ndarray, np.ndarray, int, np.ndarray | None, bool],
    DrakePoolOutput,
]
ResetImpl = Callable[[np.ndarray, np.ndarray], DrakePoolOutput]


class DrakeEnvPool:
    """Dog-scoped Python oracle for the future native DrakeUni pool.

    The public contract is intentionally buffer based. Drake-specific handles
    stay behind the implementation callback so the native C++ pool can later
    replace this class without changing DrakeBackend or the Go1 task.
    """

    def __init__(
        self,
        *,
        nbatch: int,
        state_dim: int,
        control_dim: int,
        sensor_shapes: Mapping[str, tuple[int, ...]],
        step_impl: StepImpl,
        reset_impl: ResetImpl,
    ) -> None:
        self.nbatch = int(nbatch)
        self.state_dim = int(state_dim)
        self.control_dim = int(control_dim)
        if self.nbatch < 1:
            raise ValueError(f"DrakeEnvPool requires nbatch >= 1, got {nbatch}")
        if self.state_dim < 1:
            raise ValueError(f"DrakeEnvPool requires state_dim >= 1, got {state_dim}")
        if self.control_dim < 1:
            raise ValueError(f"DrakeEnvPool requires control_dim >= 1, got {control_dim}")
        self.sensor_shapes = dict(sensor_shapes)
        self._step_impl = step_impl
        self._reset_impl = reset_impl

    def step(
        self,
        state0: np.ndarray,
        *,
        nstep: int,
        control: np.ndarray,
        push_force: np.ndarray | None = None,
        return_sensor: bool = True,
    ) -> DrakePoolOutput:
        nstep_int = int(nstep)
        if nstep_int < 1:
            raise ValueError(f"nstep must be >= 1, got {nstep}")
        state = self._require_state_batch(state0, expected_rows=self.nbatch, name="state0")
        control_values = self._require_control(control, nstep=nstep_int)
        push_values = self._require_push_force(push_force)
        output = self._step_impl(
            state,
            control_values,
            nstep_int,
            push_values,
            bool(return_sensor),
        )
        self._validate_output(output, require_sensor=bool(return_sensor))
        return output

    def reset(self, env_ids: np.ndarray, initial_state: np.ndarray) -> DrakePoolOutput:
        ids = np.asarray(env_ids, dtype=np.int32)
        if ids.ndim != 1:
            raise ValueError(f"env_ids must be one-dimensional, got {ids.shape}")
        if np.any(ids < 0) or np.any(ids >= self.nbatch):
            raise IndexError(f"env_ids must be in [0, {self.nbatch - 1}], got {ids.tolist()}")
        state = self._require_state_batch(
            initial_state,
            expected_rows=int(ids.size),
            name="initial_state",
        )
        output = self._reset_impl(ids, state)
        self._validate_output(output, require_sensor=True)
        return output

    def _require_state_batch(
        self,
        state: np.ndarray,
        *,
        expected_rows: int,
        name: str,
    ) -> np.ndarray:
        values = np.asarray(state, dtype=np.float64)
        expected_shape = (expected_rows, self.state_dim)
        if values.shape != expected_shape:
            raise ValueError(f"{name} must have shape {expected_shape}, got {values.shape}")
        return values

    def _require_control(self, control: np.ndarray, *, nstep: int) -> np.ndarray:
        values = np.asarray(control, dtype=np.float64)
        if values.shape == (self.nbatch, self.control_dim):
            return values
        expected_traj_shape = (self.nbatch, nstep, self.control_dim)
        if values.shape != expected_traj_shape:
            raise ValueError(
                "control must have shape "
                f"({self.nbatch}, {self.control_dim}) or {expected_traj_shape}, "
                f"got {values.shape}"
            )
        return values

    def _require_push_force(self, push_force: np.ndarray | None) -> np.ndarray | None:
        if push_force is None:
            return None
        values = np.asarray(push_force, dtype=np.float64)
        expected_shape = (self.nbatch, 3)
        if values.shape != expected_shape:
            raise ValueError(f"push_force must have shape {expected_shape}, got {values.shape}")
        return values

    def _validate_output(self, output: DrakePoolOutput, *, require_sensor: bool) -> None:
        if not isinstance(output, DrakePoolOutput):
            raise TypeError(f"DrakeEnvPool implementation returned {type(output)!r}")
        expected_state_shape = (self.nbatch, self.state_dim)
        if output.state.shape != expected_state_shape:
            raise ValueError(
                f"pool state output must have shape {expected_state_shape}, "
                f"got {output.state.shape}"
            )
        if not require_sensor:
            return
        missing = sorted(set(self.sensor_shapes) - set(output.sensor))
        if missing:
            raise ValueError(f"pool sensor output missing keys: {missing}")
        for key, shape in self.sensor_shapes.items():
            if output.sensor[key].shape != shape:
                raise ValueError(
                    f"pool sensor {key!r} must have shape {shape}, "
                    f"got {output.sensor[key].shape}"
                )


__all__ = ["DrakeEnvPool", "DrakePoolOutput", "SensorPacket"]
