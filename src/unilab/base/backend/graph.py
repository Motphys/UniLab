"""Backend-neutral device graph identity and diagnostics contracts."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

DEVICE_GRAPH_CONTRACT_VERSION = 1


class DeviceGraphContractError(ValueError):
    """Raised when graph identity or diagnostics are incomplete or ambiguous."""


def _non_empty(value: str, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise DeviceGraphContractError(f"{name} must be a non-empty string")
    return value


def _count(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise DeviceGraphContractError(f"{name} must be a non-negative integer")
    return value


class DeviceGraphExecutionMode(str, Enum):
    """Explicit backend execution mode for device physics."""

    CUDA_GRAPH = "cuda_graph"


@dataclass(frozen=True)
class DeviceGraphBufferAddress:
    """Cold-path identity metadata for one graph-captured device allocation."""

    name: str
    address: int
    shape: tuple[int, ...]
    dtype: str
    device: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _non_empty(self.name, "buffer name"))
        if isinstance(self.address, bool) or not isinstance(self.address, int) or self.address < 0:
            raise DeviceGraphContractError("buffer address must be a non-negative integer")
        if not isinstance(self.shape, tuple) or any(
            isinstance(dim, bool) or not isinstance(dim, int) or dim < 0 for dim in self.shape
        ):
            raise DeviceGraphContractError("buffer shape must contain non-negative dimensions")
        object.__setattr__(self, "dtype", _non_empty(self.dtype, "buffer dtype"))
        object.__setattr__(self, "device", _non_empty(self.device, "buffer device"))


@dataclass(frozen=True)
class DeviceGraphCaptureKey:
    """Complete identity of one captured device physics graph bundle.

    The key includes the compiled task plan and every physics storage identity
    that can make a graph replay unsafe.  A backend may reject a changed key or
    capture a new bundle, but must never silently reuse the previous graph.
    """

    backend_type: str
    plan_fingerprint: str
    num_envs: int
    state_dtype: str
    control_dtype: str
    physics_substeps: int
    storage_generation: int
    storage_fingerprint: str
    contract_version: int = DEVICE_GRAPH_CONTRACT_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(self, "backend_type", _non_empty(self.backend_type, "backend_type"))
        object.__setattr__(
            self,
            "plan_fingerprint",
            _non_empty(self.plan_fingerprint, "plan_fingerprint"),
        )
        if (
            isinstance(self.num_envs, bool)
            or not isinstance(self.num_envs, int)
            or self.num_envs <= 0
        ):
            raise DeviceGraphContractError("num_envs must be a positive integer")
        object.__setattr__(self, "state_dtype", _non_empty(self.state_dtype, "state_dtype"))
        object.__setattr__(
            self,
            "control_dtype",
            _non_empty(self.control_dtype, "control_dtype"),
        )
        if (
            isinstance(self.physics_substeps, bool)
            or not isinstance(self.physics_substeps, int)
            or self.physics_substeps <= 0
        ):
            raise DeviceGraphContractError("physics_substeps must be a positive integer")
        _count(self.storage_generation, "storage_generation")
        object.__setattr__(
            self,
            "storage_fingerprint",
            _non_empty(self.storage_fingerprint, "storage_fingerprint"),
        )
        if self.contract_version != DEVICE_GRAPH_CONTRACT_VERSION:
            raise DeviceGraphContractError(
                f"unsupported device graph contract version {self.contract_version}"
            )

    @property
    def canonical_order(self) -> tuple[str, int, str, str, int, int, str]:
        return (
            self.plan_fingerprint,
            self.num_envs,
            self.state_dtype,
            self.control_dtype,
            self.physics_substeps,
            self.storage_generation,
            self.storage_fingerprint,
        )


@dataclass(frozen=True)
class DeviceGraphDiagnostics:
    """Immutable cold-path graph execution and storage audit snapshot.

    ``capture_count`` counts successful graph-bundle capture transactions, not
    individual reset/forward/step graph objects. ``recapture_count`` is the
    subset caused by an explicit storage-generation transition.
    """

    backend_type: str
    execution_mode: DeviceGraphExecutionMode
    active_keys: tuple[DeviceGraphCaptureKey, ...]
    storage_buffers: tuple[DeviceGraphBufferAddress, ...]
    storage_generation: int
    storage_fingerprint: str
    capture_count: int
    launch_count: int
    recapture_count: int
    stale_rejection_count: int
    eager_fallback_count: int
    storage_verification_count: int
    instrumentation_complete: bool
    contract_version: int = DEVICE_GRAPH_CONTRACT_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(self, "backend_type", _non_empty(self.backend_type, "backend_type"))
        if not isinstance(self.execution_mode, DeviceGraphExecutionMode):
            raise DeviceGraphContractError("execution_mode must be a DeviceGraphExecutionMode")
        if not isinstance(self.active_keys, tuple) or any(
            not isinstance(key, DeviceGraphCaptureKey) for key in self.active_keys
        ):
            raise DeviceGraphContractError("active_keys must contain DeviceGraphCaptureKey values")
        key_order = tuple(key.canonical_order for key in self.active_keys)
        if key_order != tuple(sorted(key_order)) or len(set(self.active_keys)) != len(
            self.active_keys
        ):
            raise DeviceGraphContractError("active graph keys must be canonical and unique")
        if not isinstance(self.storage_buffers, tuple) or any(
            not isinstance(buffer, DeviceGraphBufferAddress) for buffer in self.storage_buffers
        ):
            raise DeviceGraphContractError(
                "storage_buffers must contain DeviceGraphBufferAddress values"
            )
        buffer_names = tuple(buffer.name for buffer in self.storage_buffers)
        if (
            not buffer_names
            or buffer_names != tuple(sorted(buffer_names))
            or len(set(buffer_names)) != len(buffer_names)
        ):
            raise DeviceGraphContractError("storage buffer names must be canonical and unique")
        _count(self.storage_generation, "storage_generation")
        object.__setattr__(
            self,
            "storage_fingerprint",
            _non_empty(self.storage_fingerprint, "storage_fingerprint"),
        )
        for name in (
            "capture_count",
            "launch_count",
            "recapture_count",
            "stale_rejection_count",
            "eager_fallback_count",
            "storage_verification_count",
        ):
            _count(getattr(self, name), name)
        if self.recapture_count > self.capture_count:
            raise DeviceGraphContractError("recapture_count cannot exceed capture_count")
        if not isinstance(self.instrumentation_complete, bool):
            raise DeviceGraphContractError("instrumentation_complete must be a bool")
        if self.contract_version != DEVICE_GRAPH_CONTRACT_VERSION:
            raise DeviceGraphContractError(
                f"unsupported device graph contract version {self.contract_version}"
            )
        for key in self.active_keys:
            if (
                key.backend_type != self.backend_type
                or key.storage_generation != self.storage_generation
                or key.storage_fingerprint != self.storage_fingerprint
            ):
                raise DeviceGraphContractError(
                    "active graph keys must match the diagnostics backend and storage identity"
                )


__all__ = [
    "DEVICE_GRAPH_CONTRACT_VERSION",
    "DeviceGraphBufferAddress",
    "DeviceGraphCaptureKey",
    "DeviceGraphContractError",
    "DeviceGraphDiagnostics",
    "DeviceGraphExecutionMode",
]
