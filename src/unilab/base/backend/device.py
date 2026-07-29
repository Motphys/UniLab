"""Explicit CUDA tensor ownership and stream handoff primitives.

The typed backend contract intentionally treats a device pointer as more than
an opaque array: its producing owner, lifetime epoch, placement and completion
event are part of the ABI.  This module is backend-neutral and only depends on
PyTorch, which is already a core UniLab dependency.  Optional physics backends
construct these views on their cold path; manager and runner code consume them
without learning backend-private storage details.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, TypeAlias, cast

import numpy as np
import torch

from .batch import (
    BackendBatchContractError,
    BufferContract,
    BufferPlacement,
    MemorySpace,
)

TorchCudaStream: TypeAlias = torch.cuda.Stream


class DeviceBufferContractError(BackendBatchContractError):
    """Raised when a device view, epoch, or stream handoff is invalid."""


class DeviceBufferLease:
    """Owner-controlled generation clock for device views.

    A device pointer can remain physically allocated after a physics mutation,
    but the semantic contents are no longer safe for an old consumer.  Device
    views therefore retain an epoch rather than pretending a raw Torch tensor
    has an eternal lifetime.
    """

    __slots__ = ("_epoch", "__weakref__", "owner_id")

    def __init__(self, owner_id: str) -> None:
        if not isinstance(owner_id, str) or not owner_id.strip():
            raise DeviceBufferContractError("device lease owner_id must be non-empty")
        self.owner_id = owner_id.strip()
        self._epoch = 0

    @property
    def epoch(self) -> int:
        return self._epoch

    def invalidate(self) -> None:
        self._epoch += 1

    def assert_valid(self, epoch: int) -> None:
        if epoch != self._epoch:
            raise DeviceBufferContractError(
                "device buffer view is stale because its owner crossed a mutation barrier"
            )


def _torch_dtype_name(dtype: torch.dtype) -> str:
    mapping = {
        torch.float16: "float16",
        torch.float32: "float32",
        torch.float64: "float64",
        torch.int8: "int8",
        torch.int16: "int16",
        torch.int32: "int32",
        torch.int64: "int64",
        torch.uint8: "uint8",
        torch.bool: "bool",
    }
    try:
        return mapping[dtype]
    except KeyError as exc:
        raise DeviceBufferContractError(f"unsupported Torch dtype {dtype!s}") from exc


@dataclass(frozen=True)
class DeviceCompletion:
    """One explicit CUDA completion event with provenance and epoch metadata.

    ``wait`` only establishes a stream dependency.  It deliberately never
    calls ``synchronize`` and therefore cannot hide a host-side barrier.
    """

    placement: BufferPlacement
    owner_id: str
    epoch: int
    event: torch.cuda.Event = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        if not isinstance(self.placement, BufferPlacement):
            raise DeviceBufferContractError("device completion placement is invalid")
        if self.placement.memory_space is not MemorySpace.DEVICE:
            raise DeviceBufferContractError("device completion requires device placement")
        if self.placement.device_type != "cuda":
            raise DeviceBufferContractError("device completion currently requires CUDA placement")
        if not isinstance(self.owner_id, str) or not self.owner_id.strip():
            raise DeviceBufferContractError("device completion owner_id must be non-empty")
        if isinstance(self.epoch, bool) or not isinstance(self.epoch, int) or self.epoch < 0:
            raise DeviceBufferContractError("device completion epoch must be non-negative")
        if not isinstance(self.event, torch.cuda.Event):
            raise DeviceBufferContractError("device completion event must be a torch.cuda.Event")

    @classmethod
    def record(
        cls,
        *,
        placement: BufferPlacement,
        owner_id: str,
        epoch: int,
        stream: torch.cuda.Stream | None = None,
        event: torch.cuda.Event | None = None,
    ) -> DeviceCompletion:
        """Record a CUDA event on ``stream`` without a host synchronization."""

        if placement.memory_space is not MemorySpace.DEVICE or placement.device_type != "cuda":
            raise DeviceBufferContractError("device completion record requires CUDA placement")
        device = torch.device(f"cuda:{placement.device_index}")
        if stream is None:
            stream = torch.cuda.current_stream(device)
        if stream.device != device:
            raise DeviceBufferContractError(
                "device completion stream placement differs from the declared buffer placement"
            )
        if event is None:
            event = torch.cuda.Event(enable_timing=False)
        event.record(stream)
        return cls(placement=placement, owner_id=owner_id, epoch=epoch, event=event)

    def wait(self, stream: torch.cuda.Stream | None = None) -> None:
        """Make a consumer stream wait for this event, without host blocking."""

        device = torch.device(f"cuda:{self.placement.device_index}")
        if stream is None:
            stream = torch.cuda.current_stream(device)
        if stream.device != device:
            raise DeviceBufferContractError(
                "consumer stream placement differs from the producer completion event"
            )
        stream.wait_event(self.event)


@dataclass(frozen=True)
class DeviceTensorView:
    """Lease-guarded zero-copy CUDA tensor and optional producer event.

    The wrapper, not a naked ``torch.Tensor``, is the public typed-batch
    handle.  It exposes DLPack only while its owner epoch is current.  Callers
    must explicitly wait for :attr:`completion` before consuming a view from a
    different stream; a missing completion is rejected by the backend/runner
    handoff helpers instead of being repaired with a global synchronization.
    """

    tensor_handle: torch.Tensor = field(repr=False, compare=False)
    contract: BufferContract
    lease: DeviceBufferLease = field(repr=False, compare=False)
    completion: DeviceCompletion | None = field(default=None, repr=False, compare=False)
    _epoch: int = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if not isinstance(self.tensor_handle, torch.Tensor):
            raise DeviceBufferContractError("device tensor view requires a torch.Tensor")
        if not isinstance(self.contract, BufferContract):
            raise DeviceBufferContractError("device tensor view contract is invalid")
        if not isinstance(self.lease, DeviceBufferLease):
            raise DeviceBufferContractError("device tensor view lease is invalid")
        placement = self.contract.placement
        tensor = self.tensor_handle
        if placement.memory_space is not MemorySpace.DEVICE or placement.device_type != "cuda":
            raise DeviceBufferContractError("device tensor views require CUDA buffer placement")
        expected_device = torch.device(f"cuda:{placement.device_index}")
        if tensor.device != expected_device:
            raise DeviceBufferContractError(
                f"device tensor is on {tensor.device}, expected {expected_device}"
            )
        if not tensor.is_contiguous():
            raise DeviceBufferContractError("device tensor view must be C-contiguous")
        if tensor.ndim < 1:
            raise DeviceBufferContractError("device tensor view must include a batch dimension")
        if tuple(tensor.shape[1:]) != self.contract.row_shape:
            raise DeviceBufferContractError(
                "device tensor row shape differs from its typed buffer contract"
            )
        if _torch_dtype_name(tensor.dtype) != np.dtype(self.contract.dtype).name:
            raise DeviceBufferContractError(
                "device tensor dtype differs from its typed buffer contract"
            )
        if self.completion is not None:
            if not isinstance(self.completion, DeviceCompletion):
                raise DeviceBufferContractError("device tensor completion is invalid")
            if self.completion.placement != placement:
                raise DeviceBufferContractError(
                    "device tensor completion placement differs from its buffer placement"
                )
            if self.completion.owner_id != self.lease.owner_id:
                raise DeviceBufferContractError(
                    "device tensor completion owner differs from its buffer lease owner"
                )
            if self.completion.epoch != self.lease.epoch:
                raise DeviceBufferContractError(
                    "device tensor completion epoch differs from the current buffer lease"
                )
        object.__setattr__(self, "_epoch", self.lease.epoch)

    @property
    def epoch(self) -> int:
        return self._epoch

    @property
    def owner_id(self) -> str:
        """Return the explicit owner identity carried by the view's lease."""

        self.assert_valid()
        return self.lease.owner_id

    @property
    def shape(self) -> tuple[int, ...]:
        self.assert_valid()
        return tuple(int(dim) for dim in self.tensor_handle.shape)

    @property
    def dtype(self) -> str:
        self.assert_valid()
        return _torch_dtype_name(self.tensor_handle.dtype)

    @property
    def data_ptr(self) -> int:
        self.assert_valid()
        return int(self.tensor_handle.data_ptr())

    def assert_valid(self) -> None:
        self.lease.assert_valid(self._epoch)

    def torch(self) -> torch.Tensor:
        """Return the zero-copy tensor after checking the owner epoch."""

        self.assert_valid()
        return self.tensor_handle

    def require_completion(self) -> DeviceCompletion:
        """Return an explicit producer event or fail closed."""

        self.assert_valid()
        if self.completion is None:
            raise DeviceBufferContractError(
                "device tensor view has no producer completion event; refusing implicit ordering"
            )
        return self.completion

    def wait(self, stream: TorchCudaStream | None = None) -> None:
        """Explicitly establish a consumer-stream dependency."""

        self.require_completion().wait(stream)

    def __dlpack_device__(self) -> tuple[int, int]:
        self.assert_valid()
        method = self.tensor_handle.__dlpack_device__
        return cast(tuple[int, int], method())

    def __dlpack__(
        self, stream: int | None = None, max_version: tuple[int, int] | None = None
    ) -> Any:
        """Export a fresh DLPack capsule while the source lease remains valid."""

        self.assert_valid()
        if not self.contract.dlpack_exportable:
            raise DeviceBufferContractError("this typed device buffer is not DLPack exportable")
        try:
            return self.tensor_handle.__dlpack__(stream=stream, max_version=max_version)
        except TypeError:
            # PyTorch versions before the max-version argument still support
            # the standardized stream argument.
            return self.tensor_handle.__dlpack__(stream=stream)


def require_device_tensor_view(
    value: object,
    *,
    contract: BufferContract,
    require_completion: bool,
) -> DeviceTensorView:
    """Validate one typed device handle at a runner/backend boundary."""

    if not isinstance(value, DeviceTensorView):
        raise DeviceBufferContractError(
            "device-resident batches require a DeviceTensorView, not a raw tensor or host array"
        )
    value.assert_valid()
    if value.contract != contract:
        raise DeviceBufferContractError("device tensor view contract differs from the bound plan")
    if require_completion:
        value.require_completion()
    return value


__all__ = [
    "DeviceBufferContractError",
    "DeviceBufferLease",
    "DeviceCompletion",
    "DeviceTensorView",
    "require_device_tensor_view",
]
