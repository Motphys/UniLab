"""Shared weight synchronization for actor networks."""

from __future__ import annotations

import multiprocessing as mp
import time
from multiprocessing import shared_memory
from typing import Any, Dict

import numpy as np

_SPAWN_CTX = mp.get_context("spawn")


class SharedWeightSync:
    """Synchronize actor weights between learner and collector."""

    def __init__(
        self, param_shapes: Dict, *, create: bool = True, shm_name: str | None = None, lock=None
    ):
        self._param_shapes = param_shapes
        self._param_names = list(param_shapes.keys())
        self.trace_recorder: Any | None = None
        self.trace_thread_time = False

        total_numel = sum(s.numel() for s in param_shapes.values())
        _f32 = np.dtype(np.float32).itemsize
        _i64 = np.dtype(np.int64).itemsize
        data_bytes = total_numel * _f32
        meta_bytes = _i64
        total_bytes = data_bytes + meta_bytes

        if create:
            self._shm = shared_memory.SharedMemory(create=True, size=max(total_bytes, 1))
            self._lock = _SPAWN_CTX.Lock()
        else:
            assert shm_name is not None
            self._shm = shared_memory.SharedMemory(name=shm_name, create=False)
            # lock must be passed in from the parent process when attaching
            self._lock = lock

        buf = self._shm.buf
        assert buf is not None
        self._buffer: np.ndarray = np.ndarray((total_numel,), dtype=np.float32, buffer=buf)
        self._version_arr: np.ndarray = np.ndarray((1,), dtype=np.int64, buffer=buf[data_bytes:])
        if create:
            self._version_arr[0] = 0

    @property
    def name(self) -> str:
        return self._shm.name

    @property
    def version(self) -> int:
        return int(self._version_arr[0])

    @classmethod
    def from_state_dict(cls, state_dict, **kwargs):
        param_shapes = {name: p.shape for name, p in state_dict.items()}
        obj = cls(param_shapes, **kwargs)
        obj.write_weights(state_dict)
        return obj

    def write_weights(self, state_dict) -> None:
        _trace_ns = time.perf_counter_ns()
        _thread_ns = time.thread_time_ns() if self.trace_thread_time else None
        if self._lock is not None:
            with self._lock:
                offset = 0
                for name in self._param_names:
                    param = state_dict[name]
                    arr = param.detach().cpu().numpy().ravel()
                    n = arr.size
                    self._buffer[offset : offset + n] = arr
                    offset += n
                self._version_arr[0] += 1
        else:
            # No lock - direct write
            offset = 0
            for name in self._param_names:
                param = state_dict[name]
                arr = param.detach().cpu().numpy().ravel()
                n = arr.size
                self._buffer[offset : offset + n] = arr
                offset += n
            self._version_arr[0] += 1
        if self.trace_recorder is not None:
            self.trace_recorder.add_slice(
                "weight_sync/write_weights_d2h",
                category="weight_sync",
                start_ns=_trace_ns,
                end_ns=time.perf_counter_ns(),
                args={"version": int(self._version_arr[0]), "mode": "sync"},
            )
            if _thread_ns is not None:
                self.trace_recorder.add_counter(
                    "weight_sync/write_thread_cpu_us",
                    (time.thread_time_ns() - _thread_ns) / 1000.0,
                    category="weight_sync",
                )

    def read_weights_into(self, state_dict) -> int:
        import torch

        _trace_ns = time.perf_counter_ns()
        _thread_ns = time.thread_time_ns() if self.trace_thread_time else None
        if self._lock is not None:
            with self._lock:
                offset = 0
                for name in self._param_names:
                    param = state_dict[name]
                    n = param.numel()
                    data = self._buffer[offset : offset + n].copy()
                    param.data.copy_(torch.from_numpy(data.reshape(param.shape)))
                    offset += n
                version = int(self._version_arr[0])
        else:
            # No lock - direct read (for subprocess)
            offset = 0
            for name in self._param_names:
                param = state_dict[name]
                n = param.numel()
                data = self._buffer[offset : offset + n].copy()
                param.data.copy_(torch.from_numpy(data.reshape(param.shape)))
                offset += n
            version = int(self._version_arr[0])
        if self.trace_recorder is not None:
            self.trace_recorder.add_slice(
                "weight_sync/read_weights_into_cpu_actor",
                category="weight_sync",
                start_ns=_trace_ns,
                end_ns=time.perf_counter_ns(),
                args={"version": version},
            )
            if _thread_ns is not None:
                self.trace_recorder.add_counter(
                    "weight_sync/read_thread_cpu_us",
                    (time.thread_time_ns() - _thread_ns) / 1000.0,
                    category="weight_sync",
                )
        return version

    def create_device_applier(self, state_dict, device) -> DeviceWeightApplier:
        """Build a DeviceWeightApplier bound to this sync's layout."""
        return DeviceWeightApplier(self, state_dict, device)

    def cleanup(self) -> None:
        try:
            self._shm.close()
            self._shm.unlink()
        except Exception:
            pass

    def close(self) -> None:
        try:
            self._shm.close()
        except Exception:
            pass


class DeviceWeightApplier:
    """Device-side fast apply path for one SharedWeightSync snapshot layout.

    One refresh is a single flat host snapshot copy under the lock, one flat
    H2D copy into persistent device staging, then on-device copies into the
    target tensors. Host/device staging and the layout are computed once and
    reused, so steady-state refreshes do not allocate or rebuild host arrays.

    The shared-memory snapshot, version and lock semantics are unchanged: the
    host copy runs entirely under the existing lock, so a refresh never sees a
    partially published state.
    """

    def __init__(self, weight_sync: SharedWeightSync, state_dict, device) -> None:
        import torch

        self._sync = weight_sync
        self._device = torch.device(device)

        names = weight_sync._param_names
        shapes = weight_sync._param_shapes
        self._dst_views = []
        self._offsets: list[tuple[int, int]] = []
        offset = 0
        for name in names:
            tensor = state_dict[name]
            if tuple(tensor.shape) != tuple(shapes[name]):
                raise ValueError(
                    f"DeviceWeightApplier shape mismatch for '{name}': "
                    f"{tuple(tensor.shape)} != {tuple(shapes[name])}"
                )
            n = tensor.numel()
            self._offsets.append((offset, n))
            self._dst_views.append(tensor.data.view(-1))
            offset += n

        total_numel = offset
        if total_numel != weight_sync._buffer.size:
            raise ValueError(
                "DeviceWeightApplier layout mismatch: state_dict total numel "
                f"{total_numel} != snapshot numel {weight_sync._buffer.size}"
            )

        self._host = torch.empty(
            total_numel, dtype=torch.float32, pin_memory=self._device.type == "cuda"
        )
        self._host_np = self._host.numpy()
        self._dev = torch.empty(total_numel, dtype=torch.float32, device=self._device)
        # Guards reuse of the (pinned) host staging buffer: the previous async
        # H2D must be finished before apply() overwrites the host buffer again.
        self._h2d_done: Any | None = torch.cuda.Event() if self._device.type == "cuda" else None

    def apply(self) -> int:
        """Apply the latest published snapshot to the bound tensors; return its version."""
        _trace_ns = time.perf_counter_ns()
        _thread_ns = time.thread_time_ns() if self._sync.trace_thread_time else None
        if self._h2d_done is not None:
            self._h2d_done.synchronize()

        if self._sync._lock is not None:
            with self._sync._lock:
                np.copyto(self._host_np, self._sync._buffer)
                version = int(self._sync._version_arr[0])
        else:
            np.copyto(self._host_np, self._sync._buffer)
            version = int(self._sync._version_arr[0])

        self._dev.copy_(self._host, non_blocking=True)
        if self._h2d_done is not None:
            self._h2d_done.record()
        for (offset, n), dst in zip(self._offsets, self._dst_views, strict=True):
            dst.copy_(self._dev[offset : offset + n])

        if self._sync.trace_recorder is not None:
            self._sync.trace_recorder.add_slice(
                "weight_sync/read_weights_into_device_actor",
                category="weight_sync",
                start_ns=_trace_ns,
                end_ns=time.perf_counter_ns(),
                args={"version": version, "device": self._device.type},
            )
            if _thread_ns is not None:
                self._sync.trace_recorder.add_counter(
                    "weight_sync/read_thread_cpu_us",
                    (time.thread_time_ns() - _thread_ns) / 1000.0,
                    category="weight_sync",
                )
        return version
