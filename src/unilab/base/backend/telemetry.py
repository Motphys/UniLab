"""Public transfer/synchronization telemetry contracts for simulation backends.

The contract intentionally keeps transfer reporting separate from managed-batch
diagnostics: compatibility profiles may expose a bounded host-cache path before
they implement ``step_batch``.  A backend owns collection; callers only inspect
immutable snapshots on a cold diagnostic path.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class BackendTransferTelemetryError(ValueError):
    """Raised when a transfer profile or trace violates its public contract."""


class BackendTransferDirection(str, Enum):
    """One observable data-movement or explicit synchronization operation."""

    HOST_TO_DEVICE = "host_to_device"
    DEVICE_TO_HOST = "device_to_host"
    SYNCHRONIZE = "synchronize"


def _non_empty(value: str, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise BackendTransferTelemetryError(f"{name} must be a non-empty string")
    return value.strip()


def _count(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise BackendTransferTelemetryError(f"{name} must be an integer >= 0")
    return int(value)


@dataclass(frozen=True)
class BackendTransferEvent:
    """One event from a fixed-capacity backend-owned transfer trace."""

    sequence: int
    barrier: str
    direction: BackendTransferDirection
    buffer_name: str | None
    nbytes: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(self, "sequence", _count(self.sequence, "sequence"))
        object.__setattr__(self, "barrier", _non_empty(self.barrier, "barrier"))
        if not isinstance(self.direction, BackendTransferDirection):
            raise BackendTransferTelemetryError("direction must be a BackendTransferDirection")
        object.__setattr__(self, "nbytes", _count(self.nbytes, "nbytes"))
        if self.direction is BackendTransferDirection.SYNCHRONIZE:
            if self.buffer_name is not None or self.nbytes != 0:
                raise BackendTransferTelemetryError(
                    "synchronization events require buffer_name=None and nbytes=0"
                )
            return
        if self.buffer_name is None:
            raise BackendTransferTelemetryError("transfer events require a buffer_name")
        object.__setattr__(self, "buffer_name", _non_empty(self.buffer_name, "buffer_name"))
        if self.nbytes <= 0:
            raise BackendTransferTelemetryError("transfer events require nbytes > 0")


@dataclass(frozen=True)
class BackendTransferCounters:
    """Cumulative backend-owned transfer/synchronization counters."""

    host_to_device_transfers: int = 0
    device_to_host_transfers: int = 0
    host_to_device_bytes: int = 0
    device_to_host_bytes: int = 0
    global_synchronizations: int = 0

    def __post_init__(self) -> None:
        for name in (
            "host_to_device_transfers",
            "device_to_host_transfers",
            "host_to_device_bytes",
            "device_to_host_bytes",
            "global_synchronizations",
        ):
            object.__setattr__(self, name, _count(getattr(self, name), name))

    def delta(self, earlier: BackendTransferCounters) -> BackendTransferCounters:
        """Return a fail-closed monotonic delta from an earlier snapshot."""
        if not isinstance(earlier, BackendTransferCounters):
            raise BackendTransferTelemetryError("earlier must be BackendTransferCounters")
        values = {
            name: getattr(self, name) - getattr(earlier, name)
            for name in (
                "host_to_device_transfers",
                "device_to_host_transfers",
                "host_to_device_bytes",
                "device_to_host_bytes",
                "global_synchronizations",
            )
        }
        if any(value < 0 for value in values.values()):
            raise BackendTransferTelemetryError("transfer counters must be monotonic")
        return BackendTransferCounters(**values)


@dataclass(frozen=True)
class BackendTransferBuffer:
    """Stable host-visible byte size for one named transfer buffer."""

    name: str
    nbytes: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _non_empty(self.name, "buffer name"))
        nbytes = _count(self.nbytes, "buffer nbytes")
        if nbytes <= 0:
            raise BackendTransferTelemetryError("buffer nbytes must be > 0")
        object.__setattr__(self, "nbytes", nbytes)


@dataclass(frozen=True)
class BackendTransferBarrier:
    """Cold-path declaration of transfers allowed at one lifecycle barrier."""

    name: str
    host_to_device_buffers: tuple[str, ...] = ()
    device_to_host_buffers: tuple[str, ...] = ()
    global_synchronizations: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _non_empty(self.name, "barrier name"))
        for attribute in ("host_to_device_buffers", "device_to_host_buffers"):
            buffers = getattr(self, attribute)
            if not isinstance(buffers, tuple):
                raise BackendTransferTelemetryError(f"{attribute} must be a tuple")
            normalized = tuple(_non_empty(buffer, attribute) for buffer in buffers)
            if len(set(normalized)) != len(normalized):
                raise BackendTransferTelemetryError(f"{attribute} must not contain duplicates")
            object.__setattr__(self, attribute, normalized)
        object.__setattr__(
            self,
            "global_synchronizations",
            _count(self.global_synchronizations, "global_synchronizations"),
        )


@dataclass(frozen=True)
class BackendTransferProfile:
    """Immutable transfer declaration for an explicit execution profile."""

    name: str
    execution_profile: str
    barriers: tuple[BackendTransferBarrier, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _non_empty(self.name, "profile name"))
        object.__setattr__(
            self,
            "execution_profile",
            _non_empty(self.execution_profile, "execution_profile"),
        )
        if not isinstance(self.barriers, tuple) or not self.barriers:
            raise BackendTransferTelemetryError("barriers must be a non-empty tuple")
        if any(not isinstance(barrier, BackendTransferBarrier) for barrier in self.barriers):
            raise BackendTransferTelemetryError(
                "barriers must contain BackendTransferBarrier values"
            )
        names = tuple(barrier.name for barrier in self.barriers)
        if len(set(names)) != len(names):
            raise BackendTransferTelemetryError("barrier names must be unique")

    def barrier(self, name: str) -> BackendTransferBarrier:
        requested = _non_empty(name, "barrier name")
        for barrier in self.barriers:
            if barrier.name == requested:
                return barrier
        available = ", ".join(item.name for item in self.barriers)
        raise KeyError(
            f"transfer profile {self.name!r} has no barrier {requested!r}; available: {available}"
        )


@dataclass(frozen=True)
class BackendTransferTrace:
    """Immutable snapshot of a backend-owned fixed-capacity profiler trace."""

    events: tuple[BackendTransferEvent, ...]
    overflow_count: int = 0

    def __post_init__(self) -> None:
        if not isinstance(self.events, tuple) or any(
            not isinstance(event, BackendTransferEvent) for event in self.events
        ):
            raise BackendTransferTelemetryError("events must contain BackendTransferEvent values")
        object.__setattr__(self, "overflow_count", _count(self.overflow_count, "overflow_count"))
        sequences = tuple(event.sequence for event in self.events)
        if sequences != tuple(range(len(self.events))):
            raise BackendTransferTelemetryError(
                "trace event sequences must start at zero and be dense"
            )

    def counters(self) -> BackendTransferCounters:
        """Reconstruct aggregate counters independently from the event sequence."""
        h2d = tuple(
            event
            for event in self.events
            if event.direction is BackendTransferDirection.HOST_TO_DEVICE
        )
        d2h = tuple(
            event
            for event in self.events
            if event.direction is BackendTransferDirection.DEVICE_TO_HOST
        )
        sync = tuple(
            event
            for event in self.events
            if event.direction is BackendTransferDirection.SYNCHRONIZE
        )
        return BackendTransferCounters(
            host_to_device_transfers=len(h2d),
            device_to_host_transfers=len(d2h),
            host_to_device_bytes=sum(event.nbytes for event in h2d),
            device_to_host_bytes=sum(event.nbytes for event in d2h),
            global_synchronizations=len(sync),
        )


__all__ = [
    "BackendTransferBarrier",
    "BackendTransferBuffer",
    "BackendTransferCounters",
    "BackendTransferDirection",
    "BackendTransferEvent",
    "BackendTransferProfile",
    "BackendTransferTelemetryError",
    "BackendTransferTrace",
]
