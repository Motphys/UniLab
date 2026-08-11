"""Array-backend shim shared by the torch_env benchmarks.

The workload kernels in this directory are written once against this shim and
executed with either NumPy or Torch (cpu / cuda / mps) arrays, so every variant
runs the same sequence of math ops on identically shaped float32 tensors as the
real env code (`NpEnv.update_state` / `NpEnv._reset_done_envs` paths).

Scope notes:
- Backend physics (`backend.step`, `backend.set_state`) is excluded: it is not
  NumPy/Torch-level env computation and is identical across variants.
- NumPy RNG follows the real code (`np.random.uniform`, float64 draw + astype
  float32 where the real code does so). Torch draws float32 directly, which is
  the idiomatic torch equivalent.
"""

from __future__ import annotations

import numpy as np


class NumpyBackend:
    name = "numpy"
    device = "cpu"
    is_torch = False

    # --- creation / conversion -------------------------------------------------
    def convert(self, x):
        return np.asarray(x)

    def to_numpy(self, x):
        return np.asarray(x)

    def index(self, x):
        return np.asarray(x, dtype=np.int64)

    def zeros(self, shape):
        return np.zeros(shape, dtype=np.float32)

    def ones(self, shape):
        return np.ones(shape, dtype=np.float32)

    def empty(self, shape):
        return np.empty(shape, dtype=np.float32)

    def zeros_like(self, x):
        return np.zeros_like(x)

    def copy(self, x):
        return x.copy()

    def float_(self, x):
        return np.asarray(x, dtype=np.float32)

    def long(self, x):
        return np.asarray(x).astype(np.int64)

    # --- elementwise -----------------------------------------------------------
    def exp(self, x):
        return np.exp(x)

    def square(self, x):
        return np.square(x)

    def sqrt(self, x):
        return np.sqrt(x)

    def cos(self, x):
        return np.cos(x)

    def sin(self, x):
        return np.sin(x)

    def arccos(self, x):
        return np.arccos(x)

    def atan2(self, y, x):
        return np.arctan2(y, x)

    def fmod(self, x, y):
        return np.fmod(x, y)

    def clip(self, x, lo, hi):
        return np.clip(x, lo, hi)

    def abs(self, x):
        return np.abs(x)

    def maximum(self, a, b):
        return np.maximum(a, b)

    def minimum(self, a, b):
        return np.minimum(a, b)

    def where(self, c, a, b):
        return np.where(c, a, b)

    def logical_or(self, a, b):
        return np.logical_or(a, b)

    def logical_and(self, a, b):
        return np.logical_and(a, b)

    def logical_not(self, a):
        return np.logical_not(a)

    # --- reductions / joins ----------------------------------------------------
    def sum(self, x, axis=None):
        return np.sum(x, axis=axis)

    def mean(self, x, axis=None):
        return np.mean(x, axis=axis)

    def any(self, x, axis=None):
        return np.any(x, axis=axis)

    def concat(self, xs, axis=1):
        return np.concatenate(xs, axis=axis)

    def stack(self, xs, axis=-1):
        return np.stack(xs, axis=axis)

    def tile_batch(self, x, n):
        return np.tile(x, (n, 1))

    def take(self, x, idx):
        return np.take(x, idx, axis=0)

    def bincount(self, x, minlength):
        return np.bincount(x, minlength=minlength)

    def nonzero(self, x):
        return np.flatnonzero(x)

    def any_scalar(self, x) -> bool:
        return bool(np.any(x))

    def scalar(self, x) -> float:
        return float(x)

    def sync(self):
        pass


class TorchBackend:
    is_torch = True

    def __init__(self, device: str):
        import torch

        self.torch = torch
        self.device = device
        self.name = f"torch-{device}"

    # --- creation / conversion -------------------------------------------------
    def convert(self, x):
        return self.torch.as_tensor(np.asarray(x), device=self.device)

    def to_numpy(self, x):
        return x.detach().cpu().numpy()

    def index(self, x):
        return self.torch.as_tensor(np.asarray(x), dtype=self.torch.long, device=self.device)

    def zeros(self, shape):
        return self.torch.zeros(shape, device=self.device, dtype=self.torch.float32)

    def ones(self, shape):
        return self.torch.ones(shape, device=self.device, dtype=self.torch.float32)

    def empty(self, shape):
        return self.torch.empty(shape, device=self.device, dtype=self.torch.float32)

    def zeros_like(self, x):
        return self.torch.zeros_like(x)

    def copy(self, x):
        return x.clone()

    def float_(self, x):
        return x.to(self.torch.float32)

    def long(self, x):
        return x.to(self.torch.long)

    # --- elementwise -----------------------------------------------------------
    def exp(self, x):
        return self.torch.exp(x)

    def square(self, x):
        return self.torch.square(x)

    def sqrt(self, x):
        return self.torch.sqrt(x)

    def cos(self, x):
        return self.torch.cos(x)

    def sin(self, x):
        return self.torch.sin(x)

    def arccos(self, x):
        return self.torch.acos(x)

    def atan2(self, y, x):
        return self.torch.atan2(y, x)

    def fmod(self, x, y):
        return self.torch.fmod(x, y)

    def clip(self, x, lo, hi):
        return self.torch.clamp(x, lo, hi)

    def abs(self, x):
        return self.torch.abs(x)

    def maximum(self, a, b):
        if not self.torch.is_tensor(a):
            a = self.torch.as_tensor(a, device=self.device, dtype=self.torch.float32)
        if not self.torch.is_tensor(b):
            b = self.torch.as_tensor(b, device=self.device, dtype=self.torch.float32)
        return self.torch.maximum(a, b)

    def minimum(self, a, b):
        if not self.torch.is_tensor(a):
            a = self.torch.as_tensor(a, device=self.device, dtype=self.torch.float32)
        if not self.torch.is_tensor(b):
            b = self.torch.as_tensor(b, device=self.device, dtype=self.torch.float32)
        return self.torch.minimum(a, b)

    def where(self, c, a, b):
        return self.torch.where(c, a, b)

    def logical_or(self, a, b):
        return self.torch.logical_or(a, b)

    def logical_and(self, a, b):
        return self.torch.logical_and(a, b)

    def logical_not(self, a):
        return self.torch.logical_not(a)

    # --- reductions / joins ----------------------------------------------------
    def sum(self, x, axis=None):
        return self.torch.sum(x, dim=axis)

    def mean(self, x, axis=None):
        return self.torch.mean(x, dim=axis)

    def any(self, x, axis=None):
        return self.torch.any(x, dim=axis)

    def concat(self, xs, axis=1):
        return self.torch.cat(list(xs), dim=axis)

    def stack(self, xs, axis=-1):
        return self.torch.stack(list(xs), dim=axis)

    def tile_batch(self, x, n):
        return x.unsqueeze(0).repeat(n, 1)

    def take(self, x, idx):
        return x.index_select(0, idx)

    def bincount(self, x, minlength):
        return self.torch.bincount(x, minlength=minlength)

    def nonzero(self, x):
        return self.torch.nonzero(x, as_tuple=False).flatten()

    def any_scalar(self, x) -> bool:
        return bool(x.any())

    def scalar(self, x) -> float:
        return float(x)

    def sync(self):
        if self.device == "cuda":
            self.torch.cuda.synchronize()
        elif self.device == "mps":
            self.torch.mps.synchronize()


class NumpyRng:
    """Mirrors the real env code: np.random.uniform / np.random.choice."""

    def uniform(self, low, high, shape):
        return np.random.uniform(low, high, size=shape)

    def choice(self, a, size, p):
        # Normalize in float64 so np.random.choice's sum-to-1 check is stable
        # even when the kernel produced float32 probabilities.
        p64 = np.asarray(p, dtype=np.float64)
        p64 /= p64.sum()
        return np.random.choice(a, size=size, p=p64)


class TorchRng:
    """Idiomatic torch equivalent: float32 draws directly on device."""

    def __init__(self, device: str):
        import torch

        self.torch = torch
        self.device = device

    def uniform(self, low, high, shape):
        t = self.torch.rand(shape, device=self.device, dtype=self.torch.float32)
        if np.isscalar(low) and np.isscalar(high):
            return t * float(high - low) + float(low)
        low_t = self.torch.as_tensor(np.asarray(low), device=self.device)
        high_t = self.torch.as_tensor(np.asarray(high), device=self.device)
        return t * (high_t - low_t) + low_t

    def choice(self, a, size, p):
        return self.torch.multinomial(p, size, replacement=True)


class RecordingRng:
    """Records every draw as float32 numpy so torch variants can replay them."""

    def __init__(self):
        self.calls: list[np.ndarray] = []

    def uniform(self, low, high, shape):
        value = np.random.uniform(low, high, size=shape).astype(np.float32)
        self.calls.append(value)
        return value

    def choice(self, a, size, p):
        p64 = np.asarray(p, dtype=np.float64)
        p64 /= p64.sum()
        value = np.random.choice(a, size=size, p=p64)
        self.calls.append(value)
        return value


class ReplayRng:
    """Replays RecordingRng draws on any backend for cross-backend validation."""

    def __init__(self, backend, calls):
        self._b = backend
        self._calls = list(calls)
        self._i = 0

    def _next(self):
        value = self._calls[self._i]
        self._i += 1
        return self._b.convert(value)

    def uniform(self, low, high, shape):
        value = self._next()
        assert tuple(value.shape) == tuple(shape), (value.shape, shape)
        return value

    def choice(self, a, size, p):
        value = self._next()
        assert tuple(value.shape) == (size,), (value.shape, size)
        return self._b.index(self._b.to_numpy(value))
