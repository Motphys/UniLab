"""Shared weight synchronization for actor networks."""

from __future__ import annotations

import multiprocessing as mp
from multiprocessing import shared_memory
from typing import Dict

import numpy as np

_SPAWN_CTX = mp.get_context("spawn")


class SharedWeightSync:
    """Synchronize actor weights between learner and collector."""

    def __init__(
        self, param_shapes: Dict, *, create: bool = True, shm_name: str | None = None, lock=None
    ):
        self._param_shapes = param_shapes
        self._param_names = list(param_shapes.keys())

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

        self._buffer = np.ndarray((total_numel,), dtype=np.float32, buffer=self._shm.buf)
        self._version_arr = np.ndarray((1,), dtype=np.int64, buffer=self._shm.buf[data_bytes:])
        if create:
            self._version_arr[0] = 0
        self._total_numel = total_numel

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

    def read_weights_into(self, state_dict) -> int:
        import torch

        if self._lock is not None:
            with self._lock:
                offset = 0
                for name in self._param_names:
                    param = state_dict[name]
                    n = param.numel()
                    data = self._buffer[offset : offset + n].copy()
                    param.data.copy_(torch.from_numpy(data.reshape(param.shape)))
                    offset += n
                return int(self._version_arr[0])
        else:
            # No lock - direct read (for subprocess)
            offset = 0
            for name in self._param_names:
                param = state_dict[name]
                n = param.numel()
                data = self._buffer[offset : offset + n].copy()
                param.data.copy_(torch.from_numpy(data.reshape(param.shape)))
                offset += n
            return int(self._version_arr[0])

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
