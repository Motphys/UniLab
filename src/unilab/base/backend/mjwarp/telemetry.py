"""Allocation-stable transfer telemetry owned by the ``mjwarp`` backend."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from unilab.base.backend.telemetry import (
    BackendTransferBarrier,
    BackendTransferCounters,
    BackendTransferDirection,
    BackendTransferEvent,
    BackendTransferProfile,
    BackendTransferTrace,
)

_BARRIER_NAMES = ("init", "step", "reset")
_BUFFER_NAMES = ("control", "reset_mask", "qpos", "qvel", "sensordata")
_DIRECTIONS = (
    BackendTransferDirection.HOST_TO_DEVICE,
    BackendTransferDirection.DEVICE_TO_HOST,
    BackendTransferDirection.SYNCHRONIZE,
)
_BARRIER_CODES = {name: index for index, name in enumerate(_BARRIER_NAMES)}
_BUFFER_CODES = {name: index for index, name in enumerate(_BUFFER_NAMES)}
_DIRECTION_CODES = {direction: index for index, direction in enumerate(_DIRECTIONS)}

MJWARP_HOST_TRANSFER_PROFILE = BackendTransferProfile(
    name="mjwarp-host-cache-v1",
    execution_profile="host_numpy",
    barriers=(
        BackendTransferBarrier(
            name="init",
            host_to_device_buffers=("qpos", "qvel"),
            device_to_host_buffers=("qpos", "qvel", "sensordata"),
            global_synchronizations=1,
        ),
        BackendTransferBarrier(
            name="step",
            host_to_device_buffers=("control",),
            device_to_host_buffers=("qpos", "qvel", "sensordata"),
            global_synchronizations=1,
        ),
        BackendTransferBarrier(
            name="reset",
            host_to_device_buffers=("reset_mask", "qpos", "qvel"),
            device_to_host_buffers=("qpos", "qvel", "sensordata"),
            global_synchronizations=1,
        ),
    ),
)


@dataclass
class MjwarpTransferTelemetry:
    """Backend-owned telemetry with preallocated primitive trace storage.

    Runtime writes only scalar counters and fixed NumPy arrays.  Immutable
    dataclass events are created only by the cold diagnostic query method.
    """

    capacity: int = 4096
    _barrier_codes: np.ndarray = field(init=False, repr=False)
    _direction_codes: np.ndarray = field(init=False, repr=False)
    _buffer_codes: np.ndarray = field(init=False, repr=False)
    _byte_counts: np.ndarray = field(init=False, repr=False)
    _trace_size: int = field(default=0, init=False, repr=False)
    _overflow_count: int = field(default=0, init=False, repr=False)
    _current_barrier_code: int = field(default=0, init=False, repr=False)
    _host_to_device_transfers: int = field(default=0, init=False, repr=False)
    _device_to_host_transfers: int = field(default=0, init=False, repr=False)
    _host_to_device_bytes: int = field(default=0, init=False, repr=False)
    _device_to_host_bytes: int = field(default=0, init=False, repr=False)
    _global_synchronizations: int = field(default=0, init=False, repr=False)

    def __post_init__(self) -> None:
        if (
            isinstance(self.capacity, bool)
            or not isinstance(self.capacity, int)
            or self.capacity <= 0
        ):
            raise ValueError("trace capacity must be a positive integer")
        self._barrier_codes = np.empty((self.capacity,), dtype=np.int8)
        self._direction_codes = np.empty((self.capacity,), dtype=np.int8)
        self._buffer_codes = np.empty((self.capacity,), dtype=np.int8)
        self._byte_counts = np.empty((self.capacity,), dtype=np.int64)
        self.reset()

    def reset(self) -> None:
        """Clear diagnostic counters/trace without mutating simulator state."""
        self._trace_size = 0
        self._overflow_count = 0
        self._current_barrier_code = _BARRIER_CODES["init"]
        self._host_to_device_transfers = 0
        self._device_to_host_transfers = 0
        self._host_to_device_bytes = 0
        self._device_to_host_bytes = 0
        self._global_synchronizations = 0

    def begin_barrier(self, name: str) -> None:
        try:
            self._current_barrier_code = _BARRIER_CODES[name]
        except KeyError as exc:
            raise ValueError(f"unknown mjwarp transfer barrier {name!r}") from exc

    def host_to_device(self, buffer_name: str, nbytes: int) -> None:
        self._host_to_device_transfers += 1
        self._host_to_device_bytes += self._record_transfer(
            BackendTransferDirection.HOST_TO_DEVICE,
            buffer_name,
            nbytes,
        )

    def device_to_host(self, buffer_name: str, nbytes: int) -> None:
        self._device_to_host_transfers += 1
        self._device_to_host_bytes += self._record_transfer(
            BackendTransferDirection.DEVICE_TO_HOST,
            buffer_name,
            nbytes,
        )

    def synchronize(self) -> None:
        self._global_synchronizations += 1
        self._record(
            direction=BackendTransferDirection.SYNCHRONIZE,
            buffer_code=-1,
            nbytes=0,
        )

    def _record_transfer(
        self, direction: BackendTransferDirection, buffer_name: str, nbytes: int
    ) -> int:
        if isinstance(nbytes, bool) or not isinstance(nbytes, int) or nbytes <= 0:
            raise ValueError("mjwarp transfer byte count must be a positive integer")
        try:
            buffer_code = _BUFFER_CODES[buffer_name]
        except KeyError as exc:
            raise ValueError(f"unknown mjwarp transfer buffer {buffer_name!r}") from exc
        self._record(direction=direction, buffer_code=buffer_code, nbytes=nbytes)
        return nbytes

    def _record(
        self, *, direction: BackendTransferDirection, buffer_code: int, nbytes: int
    ) -> None:
        if self._trace_size >= self.capacity:
            self._overflow_count += 1
            return
        index = self._trace_size
        self._barrier_codes[index] = self._current_barrier_code
        self._direction_codes[index] = _DIRECTION_CODES[direction]
        self._buffer_codes[index] = buffer_code
        self._byte_counts[index] = nbytes
        self._trace_size += 1

    def counters(self) -> BackendTransferCounters:
        return BackendTransferCounters(
            host_to_device_transfers=self._host_to_device_transfers,
            device_to_host_transfers=self._device_to_host_transfers,
            host_to_device_bytes=self._host_to_device_bytes,
            device_to_host_bytes=self._device_to_host_bytes,
            global_synchronizations=self._global_synchronizations,
        )

    def trace(self) -> BackendTransferTrace:
        events = tuple(
            BackendTransferEvent(
                sequence=index,
                barrier=_BARRIER_NAMES[int(self._barrier_codes[index])],
                direction=_DIRECTIONS[int(self._direction_codes[index])],
                buffer_name=(
                    None
                    if int(self._buffer_codes[index]) < 0
                    else _BUFFER_NAMES[int(self._buffer_codes[index])]
                ),
                nbytes=int(self._byte_counts[index]),
            )
            for index in range(self._trace_size)
        )
        return BackendTransferTrace(events=events, overflow_count=self._overflow_count)


__all__ = ["MJWARP_HOST_TRANSFER_PROFILE", "MjwarpTransferTelemetry"]
