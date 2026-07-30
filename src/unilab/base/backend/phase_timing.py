"""Allocation-stable CUDA timing for one typed reset transaction.

The session is a public backend contract shared by a device manager and its
physics backend.  CUDA events are allocated and primed before the measured
window.  Recording only enqueues event timestamps; host synchronization and
materialization happen once, after the window closes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import numpy as np
import torch

from .batch import BufferPlacement, MemorySpace

DEVICE_RESET_TIMING_PHASES = (
    "mutation_sample",
    "mutation_commit",
    "recompute_constants",
    "reset_forward",
    "reset_barrier",
)
_BACKEND_TIMING_PHASES = DEVICE_RESET_TIMING_PHASES[1:4]


class DevicePhaseTimingError(ValueError):
    """Raised when a device timing window violates its lifecycle contract."""


class DevicePhaseTimingOverflowError(DevicePhaseTimingError):
    """Raised before recording a sample beyond the frozen event capacity."""


def _create_timing_event() -> torch.cuda.Event:
    """Create one timing event through an observable cold-path boundary."""

    return cast(torch.cuda.Event, torch.cuda.Event(enable_timing=True))


def _synchronize_timing_event(event: torch.cuda.Event) -> None:
    """Synchronize one terminal event through an observable cold-path boundary."""

    event.synchronize()


def _non_empty(value: str, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise DevicePhaseTimingError(f"{name} must be a non-empty string")
    return value.strip()


def _count(value: int, name: str, *, positive: bool = False) -> int:
    minimum = 1 if positive else 0
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        comparator = "> 0" if positive else ">= 0"
        raise DevicePhaseTimingError(f"{name} must be an integer {comparator}")
    return int(value)


def _milliseconds(value: float, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise DevicePhaseTimingError(f"{name} must be numeric")
    result = float(value)
    if not np.isfinite(result) or result < 0.0:
        raise DevicePhaseTimingError(f"{name} must be finite and non-negative")
    return result


@dataclass(frozen=True)
class DevicePhaseTimingInterval:
    """One phase interval relative to the full reset-barrier start event."""

    phase: str
    start_ms: float
    end_ms: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "phase", _non_empty(self.phase, "phase"))
        start = _milliseconds(self.start_ms, "start_ms")
        end = _milliseconds(self.end_ms, "end_ms")
        if end < start:
            raise DevicePhaseTimingError("phase end_ms must not precede start_ms")
        object.__setattr__(self, "start_ms", start)
        object.__setattr__(self, "end_ms", end)

    @property
    def milliseconds(self) -> float:
        return self.end_ms - self.start_ms


@dataclass(frozen=True)
class DeviceResetPhaseTimingRecord:
    """Materialized phase intervals for one reset transaction."""

    sample_index: int
    intervals: tuple[DevicePhaseTimingInterval, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "sample_index", _count(self.sample_index, "sample_index"))
        if not isinstance(self.intervals, tuple) or any(
            not isinstance(interval, DevicePhaseTimingInterval) for interval in self.intervals
        ):
            raise DevicePhaseTimingError(
                "reset timing intervals must contain DevicePhaseTimingInterval values"
            )
        phases = tuple(interval.phase for interval in self.intervals)
        if phases != DEVICE_RESET_TIMING_PHASES:
            raise DevicePhaseTimingError("reset timing intervals are not in canonical phase order")
        by_phase = {interval.phase: interval for interval in self.intervals}
        sample = by_phase["mutation_sample"]
        commit = by_phase["mutation_commit"]
        recompute = by_phase["recompute_constants"]
        forward = by_phase["reset_forward"]
        barrier = by_phase["reset_barrier"]
        if barrier.start_ms != 0.0:
            raise DevicePhaseTimingError("reset_barrier must start at zero")
        if not (
            barrier.start_ms
            <= sample.start_ms
            <= sample.end_ms
            <= commit.start_ms
            <= commit.end_ms
            <= recompute.start_ms
            <= recompute.end_ms
            <= forward.start_ms
            <= forward.end_ms
            <= barrier.end_ms
        ):
            raise DevicePhaseTimingError("reset timing intervals violate stream dependency order")

    def interval(self, phase: str) -> DevicePhaseTimingInterval:
        requested = _non_empty(phase, "phase")
        for interval in self.intervals:
            if interval.phase == requested:
                return interval
        raise KeyError(f"unknown reset timing phase {requested!r}")


@dataclass(frozen=True)
class DeviceResetPhaseTimingDiagnostics:
    """Host-only lifecycle counters that never query CUDA event timestamps."""

    capacity: int
    recorded_samples: int
    events_preallocated: int
    overflow_attempts: int
    priming_synchronizations: int
    materializations: int
    materialization_synchronizations: int
    sample_open: bool

    def __post_init__(self) -> None:
        object.__setattr__(self, "capacity", _count(self.capacity, "capacity", positive=True))
        for name in (
            "recorded_samples",
            "events_preallocated",
            "overflow_attempts",
            "priming_synchronizations",
            "materializations",
            "materialization_synchronizations",
        ):
            object.__setattr__(self, name, _count(getattr(self, name), name))
        if self.recorded_samples > self.capacity:
            raise DevicePhaseTimingError("recorded_samples exceeds timing capacity")
        if not isinstance(self.sample_open, bool):
            raise DevicePhaseTimingError("sample_open must be a bool")


@dataclass(frozen=True)
class DeviceResetPhaseTimingTrace:
    """One immutable, post-window timing materialization."""

    backend_type: str
    backend_instance_id: str
    placement: BufferPlacement
    capacity: int
    samples: tuple[DeviceResetPhaseTimingRecord, ...]
    events_preallocated: int
    priming_synchronizations: int
    materialization_synchronizations: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "backend_type", _non_empty(self.backend_type, "backend_type"))
        object.__setattr__(
            self,
            "backend_instance_id",
            _non_empty(self.backend_instance_id, "backend_instance_id"),
        )
        if not isinstance(self.placement, BufferPlacement):
            raise DevicePhaseTimingError("timing trace placement is invalid")
        object.__setattr__(self, "capacity", _count(self.capacity, "capacity", positive=True))
        if not isinstance(self.samples, tuple) or any(
            not isinstance(sample, DeviceResetPhaseTimingRecord) for sample in self.samples
        ):
            raise DevicePhaseTimingError(
                "timing trace samples must contain DeviceResetPhaseTimingRecord values"
            )
        if tuple(sample.sample_index for sample in self.samples) != tuple(range(len(self.samples))):
            raise DevicePhaseTimingError("timing trace sample indices must be dense and ordered")
        if not self.samples or len(self.samples) > self.capacity:
            raise DevicePhaseTimingError("timing trace requires 1..capacity samples")
        for name in (
            "events_preallocated",
            "priming_synchronizations",
            "materialization_synchronizations",
        ):
            object.__setattr__(self, name, _count(getattr(self, name), name))


@dataclass(frozen=True)
class DeviceResetPhaseTimingSampleToken:
    """Preallocated capability token for one timing slot."""

    sample_index: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "sample_index", _count(self.sample_index, "sample_index"))


class DeviceResetPhaseTimingSession:
    """Fixed-capacity CUDA event storage shared across manager/backend streams."""

    def __init__(
        self,
        *,
        backend_type: str,
        backend_instance_id: str,
        placement: BufferPlacement,
        capacity: int,
    ) -> None:
        self._backend_type = _non_empty(backend_type, "backend_type")
        self._backend_instance_id = _non_empty(backend_instance_id, "backend_instance_id")
        if not isinstance(placement, BufferPlacement):
            raise DevicePhaseTimingError("timing session placement is invalid")
        if (
            placement.memory_space is not MemorySpace.DEVICE
            or placement.device_type != "cuda"
            or placement.device_index is None
        ):
            raise DevicePhaseTimingError("timing session requires a concrete CUDA placement")
        self._placement = placement
        self._capacity = _count(capacity, "capacity", positive=True)
        self._device = torch.device(f"cuda:{placement.device_index}")
        self._phase_indices = {
            phase: index for index, phase in enumerate(DEVICE_RESET_TIMING_PHASES)
        }
        self._events = tuple(
            tuple(
                (
                    _create_timing_event(),
                    _create_timing_event(),
                )
                for _ in DEVICE_RESET_TIMING_PHASES
            )
            for _ in range(self._capacity)
        )
        self._tokens = tuple(
            DeviceResetPhaseTimingSampleToken(index) for index in range(self._capacity)
        )
        self._recorded_samples = 0
        self._active_token: DeviceResetPhaseTimingSampleToken | None = None
        self._mutation_sample_ended = False
        self._backend_phase_index = 0
        self._backend_phase_open = False
        self._overflow_attempts = 0
        self._priming_synchronizations = 0
        self._materializations = 0
        self._materialization_synchronizations = 0
        self._materialized = False
        self._prime_events()

    @property
    def is_materialized(self) -> bool:
        return self._materialized

    @property
    def backend_type(self) -> str:
        return self._backend_type

    @property
    def backend_instance_id(self) -> str:
        return self._backend_instance_id

    @property
    def placement(self) -> BufferPlacement:
        return self._placement

    @property
    def capacity(self) -> int:
        return self._capacity

    @property
    def diagnostics(self) -> DeviceResetPhaseTimingDiagnostics:
        return DeviceResetPhaseTimingDiagnostics(
            capacity=self._capacity,
            recorded_samples=self._recorded_samples,
            events_preallocated=self._capacity * len(DEVICE_RESET_TIMING_PHASES) * 2,
            overflow_attempts=self._overflow_attempts,
            priming_synchronizations=self._priming_synchronizations,
            materializations=self._materializations,
            materialization_synchronizations=self._materialization_synchronizations,
            sample_open=self._active_token is not None,
        )

    def _require_stream(self, stream: torch.cuda.Stream) -> None:
        if not isinstance(stream, torch.cuda.Stream) or stream.device != self._device:
            raise DevicePhaseTimingError(
                "timing event stream differs from the session CUDA placement"
            )

    def _phase_events(
        self, token: DeviceResetPhaseTimingSampleToken, phase: str
    ) -> tuple[torch.cuda.Event, torch.cuda.Event]:
        self._require_token(token)
        try:
            phase_index = self._phase_indices[phase]
        except KeyError as exc:
            raise DevicePhaseTimingError(f"unknown reset timing phase {phase!r}") from exc
        return self._events[token.sample_index][phase_index]

    def _prime_events(self) -> None:
        stream = cast(torch.cuda.Stream, torch.cuda.current_stream(self._device))
        self._require_stream(stream)
        for sample_events in self._events:
            for start, end in sample_events:
                start.record(stream)
                end.record(stream)
        _synchronize_timing_event(self._events[-1][-1][1])
        self._priming_synchronizations += 1

    def _require_token(self, token: DeviceResetPhaseTimingSampleToken) -> None:
        if (
            not isinstance(token, DeviceResetPhaseTimingSampleToken)
            or token.sample_index >= self._capacity
            or self._tokens[token.sample_index] is not token
        ):
            raise DevicePhaseTimingError("foreign or forged reset timing token")

    def _require_active(self, token: DeviceResetPhaseTimingSampleToken) -> None:
        self._require_token(token)
        if self._active_token is not token:
            raise DevicePhaseTimingError("reset timing token is not the active sample")

    def require_backend_sample(
        self,
        token: DeviceResetPhaseTimingSampleToken,
        *,
        backend_type: str,
        backend_instance_id: str,
        placement: BufferPlacement,
    ) -> None:
        """Validate owner identity and manager completion before backend recording."""

        self._require_active(token)
        if (
            backend_type != self._backend_type
            or backend_instance_id != self._backend_instance_id
            or placement != self._placement
        ):
            raise DevicePhaseTimingError("reset timing session belongs to another backend owner")
        if self._backend_phase_index != 0 or self._backend_phase_open:
            raise DevicePhaseTimingError("reset timing backend phases already started")
        if not self._mutation_sample_ended:
            raise DevicePhaseTimingError("mutation_sample must end before backend phase recording")

    def begin_sample(self, stream: torch.cuda.Stream) -> DeviceResetPhaseTimingSampleToken:
        """Open one slot and record the full-barrier/sample start timestamps."""

        self._require_stream(stream)
        if self._materialized:
            raise DevicePhaseTimingError("materialized timing sessions cannot record new samples")
        if self._active_token is not None:
            raise DevicePhaseTimingError("a reset timing sample is already open")
        if self._recorded_samples >= self._capacity:
            self._overflow_attempts += 1
            raise DevicePhaseTimingOverflowError(
                f"reset timing capacity {self._capacity} is exhausted"
            )
        token = self._tokens[self._recorded_samples]
        barrier_start, _ = self._phase_events(token, "reset_barrier")
        sample_start, _ = self._phase_events(token, "mutation_sample")
        barrier_start.record(stream)
        sample_start.record(stream)
        self._active_token = token
        self._mutation_sample_ended = False
        self._backend_phase_index = 0
        self._backend_phase_open = False
        return token

    def end_mutation_sample(
        self,
        token: DeviceResetPhaseTimingSampleToken,
        stream: torch.cuda.Stream,
    ) -> None:
        self._require_stream(stream)
        self._require_active(token)
        if self._backend_phase_index != 0 or self._backend_phase_open:
            raise DevicePhaseTimingError("mutation_sample must end before backend phases")
        if self._mutation_sample_ended:
            raise DevicePhaseTimingError("mutation_sample timing already ended")
        _, sample_end = self._phase_events(token, "mutation_sample")
        sample_end.record(stream)
        self._mutation_sample_ended = True

    def begin_backend_phase(
        self,
        token: DeviceResetPhaseTimingSampleToken,
        phase: str,
        stream: torch.cuda.Stream,
    ) -> None:
        self._require_stream(stream)
        self._require_active(token)
        if self._backend_phase_open:
            raise DevicePhaseTimingError("a reset timing backend phase is already open")
        try:
            expected = _BACKEND_TIMING_PHASES[self._backend_phase_index]
        except IndexError as exc:
            raise DevicePhaseTimingError("all reset timing backend phases are complete") from exc
        if phase != expected:
            raise DevicePhaseTimingError(
                f"reset timing expected backend phase {expected!r}, got {phase!r}"
            )
        start, _ = self._phase_events(token, phase)
        start.record(stream)
        self._backend_phase_open = True

    def end_backend_phase(
        self,
        token: DeviceResetPhaseTimingSampleToken,
        phase: str,
        stream: torch.cuda.Stream,
    ) -> None:
        self._require_stream(stream)
        self._require_active(token)
        if not self._backend_phase_open:
            raise DevicePhaseTimingError("no reset timing backend phase is open")
        expected = _BACKEND_TIMING_PHASES[self._backend_phase_index]
        if phase != expected:
            raise DevicePhaseTimingError(
                f"reset timing expected backend phase {expected!r}, got {phase!r}"
            )
        _, end = self._phase_events(token, phase)
        end.record(stream)
        self._backend_phase_index += 1
        self._backend_phase_open = False

    def end_sample(
        self,
        token: DeviceResetPhaseTimingSampleToken,
        stream: torch.cuda.Stream,
    ) -> None:
        """Record the barrier end after every backend phase and state refresh."""

        self._require_stream(stream)
        self._require_active(token)
        if self._backend_phase_open or self._backend_phase_index != len(_BACKEND_TIMING_PHASES):
            raise DevicePhaseTimingError("reset timing sample ended before all backend phases")
        _, barrier_end = self._phase_events(token, "reset_barrier")
        barrier_end.record(stream)
        self._recorded_samples += 1
        self._active_token = None

    def materialize(self) -> DeviceResetPhaseTimingTrace:
        """Synchronize the final event once and materialize all phase intervals."""

        if self._materialized:
            raise DevicePhaseTimingError("reset timing session may only materialize once")
        if self._active_token is not None:
            raise DevicePhaseTimingError("cannot materialize an open reset timing sample")
        if self._recorded_samples == 0:
            raise DevicePhaseTimingError("cannot materialize an empty reset timing session")
        final_token = self._tokens[self._recorded_samples - 1]
        _, final_event = self._phase_events(final_token, "reset_barrier")
        _synchronize_timing_event(final_event)
        self._materialization_synchronizations += 1

        records: list[DeviceResetPhaseTimingRecord] = []
        for index in range(self._recorded_samples):
            token = self._tokens[index]
            barrier_start, _ = self._phase_events(token, "reset_barrier")
            intervals: list[DevicePhaseTimingInterval] = []
            for phase in DEVICE_RESET_TIMING_PHASES:
                start, end = self._phase_events(token, phase)
                start_ms = 0.0 if phase == "reset_barrier" else barrier_start.elapsed_time(start)
                end_ms = barrier_start.elapsed_time(end)
                intervals.append(
                    DevicePhaseTimingInterval(
                        phase=phase,
                        start_ms=start_ms,
                        end_ms=end_ms,
                    )
                )
            records.append(
                DeviceResetPhaseTimingRecord(sample_index=index, intervals=tuple(intervals))
            )
        self._materializations += 1
        self._materialized = True
        return DeviceResetPhaseTimingTrace(
            backend_type=self._backend_type,
            backend_instance_id=self._backend_instance_id,
            placement=self._placement,
            capacity=self._capacity,
            samples=tuple(records),
            events_preallocated=self._capacity * len(DEVICE_RESET_TIMING_PHASES) * 2,
            priming_synchronizations=self._priming_synchronizations,
            materialization_synchronizations=self._materialization_synchronizations,
        )


__all__ = [
    "DEVICE_RESET_TIMING_PHASES",
    "DevicePhaseTimingError",
    "DevicePhaseTimingInterval",
    "DevicePhaseTimingOverflowError",
    "DeviceResetPhaseTimingDiagnostics",
    "DeviceResetPhaseTimingRecord",
    "DeviceResetPhaseTimingSampleToken",
    "DeviceResetPhaseTimingSession",
    "DeviceResetPhaseTimingTrace",
]
