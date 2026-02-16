from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Union

import mujoco
from mujoco import rollout
import numpy as np
import mlx.core as mx


ArrayLikeControl = Union[np.ndarray, mx.array]
ModelLike = Union[mujoco.MjModel, Sequence[mujoco.MjModel]]
DataLike = Union[mujoco.MjData, Sequence[mujoco.MjData]]


@dataclass
class RolloutMLXResult:
    state_mx: mx.array
    sensordata_mx: mx.array


class RolloutMLXBridge:
    """Bridge MuJoCo rollout with optional MLX input/output.

    Notes:
    - MuJoCo Python rollout currently consumes/produces NumPy arrays.
    - This bridge keeps MuJoCo-side behavior unchanged and only adapts the edges.
    - The conversion boundary is centralized here, so MuJoCo updates are low-cost to adopt.
    """

    def __init__(self, *, nthread: Optional[int] = None):
        self._runner = rollout.Rollout(nthread=0 if nthread is None else int(nthread))

    def close(self) -> None:
        if self._runner is not None:
            self._runner.close()
            self._runner = None

    def __enter__(self) -> "RolloutMLXBridge":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    @staticmethod
    def _as_numpy_f64(array: ArrayLikeControl) -> np.ndarray:
        if isinstance(array, np.ndarray):
            # rollout expects contiguous float64; avoid redundant copies when possible.
            if array.dtype == np.float64 and array.flags.c_contiguous:
                return array
            return np.ascontiguousarray(array, dtype=np.float64)
        if isinstance(array, mx.array):
            return np.ascontiguousarray(np.asarray(array), dtype=np.float64)
        raise TypeError(f"Unsupported control type: {type(array)!r}")

    def rollout_numpy(
        self,
        model: ModelLike,
        data: DataLike,
        initial_state: np.ndarray,
        control: ArrayLikeControl,
        *,
        nstep: Optional[int] = None,
        control_spec: int = mujoco.mjtState.mjSTATE_CTRL.value,
        state: Optional[np.ndarray] = None,
        sensordata: Optional[np.ndarray] = None,
        chunk_size: Optional[int] = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        if self._runner is None:
            raise RuntimeError("RolloutMLXBridge has been closed.")

        control_np = self._as_numpy_f64(control)
        state_np, sensordata_np = self._runner.rollout(
            model=model,
            data=data,
            initial_state=initial_state,
            control=control_np,
            control_spec=control_spec,
            nstep=nstep,
            state=state,
            sensordata=sensordata,
            chunk_size=chunk_size,
        )
        return state_np, sensordata_np

    def rollout_mlx(
        self,
        model: ModelLike,
        data: DataLike,
        initial_state: np.ndarray,
        control: ArrayLikeControl,
        *,
        nstep: Optional[int] = None,
        control_spec: int = mujoco.mjtState.mjSTATE_CTRL.value,
        chunk_size: Optional[int] = None,
        out_dtype: mx.Dtype = mx.float32,
    ) -> RolloutMLXResult:
        state_np, sensordata_np = self.rollout_numpy(
            model=model,
            data=data,
            initial_state=initial_state,
            control=control,
            nstep=nstep,
            control_spec=control_spec,
            state=None,
            sensordata=None,
            chunk_size=chunk_size,
        )
        return RolloutMLXResult(
            state_mx=mx.array(state_np, dtype=out_dtype),
            sensordata_mx=mx.array(sensordata_np, dtype=out_dtype),
        )
