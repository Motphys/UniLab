"""Torch device aliases and shared device-runtime primitives for RL.

Merged from unilab_rl @ 3654e4b (unilab-rl 1.4.15), Apache-2.0; previously
``uni_rl.utils.device``. ``get_default_device`` is owned by
:mod:`unilab.utils.device` and re-exported here (module path, per repo rules).
Backend process-device resolution/binding is not owned by this layer: runners
and workers receive it through injected callables (see
:mod:`unilab.rl.runtime.offpolicy.double_buffer_runner` and
:mod:`unilab.rl.runtime.offpolicy.worker`).
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Protocol

import torch

from unilab.utils.device import get_default_device

__all__ = [
    "CudaDeviceTimer",
    "DeviceTimer",
    "SynchronizedDeviceTimer",
    "device_runtime_capabilities",
    "get_default_device",
    "build_device_timer",
    "sync_device",
]


def sync_device(device: str | torch.device, *, stream: bool = False) -> None:
    """Wait for device work using the platform's public synchronization API.

    ``stream=True`` waits on the device's current stream.  It is CUDA-specific:
    MPS exposes no stream-scoped synchronization API, so requesting it there
    fails closed rather than silently draining unrelated queued work.
    """
    resolved = torch.device(device)
    if resolved.type == "cuda":
        if stream:
            torch.cuda.current_stream(resolved).synchronize()
        else:
            torch.cuda.synchronize(resolved)
    elif resolved.type == "mps":
        if stream:
            raise ValueError("torch.mps does not expose stream synchronization")
        torch.mps.synchronize()
    elif resolved.type != "cpu":
        raise ValueError(f"Unsupported device synchronization type: {resolved.type!r}")


class DeviceTimer(Protocol):
    """Portable wall-time-compatible timing boundary for accelerator work."""

    def start(self) -> None: ...

    def stop(self) -> None: ...

    def elapsed_s(self) -> float: ...


class SynchronizedDeviceTimer:
    """Wall-clock timer whose elapsed read owns the device synchronization.

    ``stop`` only records wall time, allowing action publication or another
    host boundary to occur first.  ``elapsed_s`` synchronizes before returning.
    """

    def __init__(self, device: str | torch.device) -> None:
        self._device = torch.device(device)

    def start(self) -> None:
        self._start_ns = time.perf_counter_ns()

    def stop(self) -> None:
        self._end_ns = time.perf_counter_ns()

    def elapsed_s(self) -> float:
        sync_device(self._device)
        return (self._end_ns - self._start_ns) / 1e9


class CudaDeviceTimer:
    """CUDA event timer whose elapsed read owns the synchronization point."""

    def __init__(self, device: str | torch.device) -> None:
        self._device = torch.device(device)
        self._start_event = torch.cuda.Event(enable_timing=True)
        self._end_event = torch.cuda.Event(enable_timing=True)

    def start(self) -> None:
        self._start_event.record(torch.cuda.current_stream(self._device))

    def stop(self) -> None:
        self._end_event.record(torch.cuda.current_stream(self._device))

    def elapsed_s(self) -> float:
        self._end_event.synchronize()
        return float(self._start_event.elapsed_time(self._end_event) / 1e3)


def build_device_timer(device: str | torch.device) -> DeviceTimer:
    """Build the validated timing implementation for ``device``."""
    device_type = torch.device(device).type
    if device_type == "cuda":
        return CudaDeviceTimer(device)
    if device_type in {"mps", "cpu"}:
        return SynchronizedDeviceTimer(device)
    raise ValueError(f"Unsupported device timer type: {device_type!r}")


@dataclass(frozen=True)
class DeviceRuntimeCapabilities:
    """Explicit capability/policy labels recorded in runtime diagnostics."""

    device_type: str
    sync_api: str
    stream_sync_api: str | None
    has_timing_events: bool

    def manifest(self) -> dict[str, str | bool | None]:
        return {
            "device_type": self.device_type,
            "sync_api": self.sync_api,
            "stream_sync_api": self.stream_sync_api,
            "has_timing_events": self.has_timing_events,
            "sync_policy": "capability",
            "timing_policy": "capability",
        }


def device_runtime_capabilities(device: str | torch.device) -> DeviceRuntimeCapabilities:
    """Return the supported public synchronization/timing surface."""
    device_type = torch.device(device).type
    if device_type == "cuda":
        return DeviceRuntimeCapabilities(
            device_type=device_type,
            sync_api="torch.cuda.synchronize",
            stream_sync_api="torch.cuda.current_stream.synchronize",
            has_timing_events=True,
        )
    if device_type == "mps":
        return DeviceRuntimeCapabilities(
            device_type=device_type,
            sync_api="torch.mps.synchronize",
            stream_sync_api=None,
            has_timing_events=False,
        )
    if device_type != "cpu":
        raise ValueError(f"Unsupported device capability type: {device_type!r}")
    return DeviceRuntimeCapabilities(
        device_type=device_type,
        sync_api="none",
        stream_sync_api=None,
        has_timing_events=False,
    )
