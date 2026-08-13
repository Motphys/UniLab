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
