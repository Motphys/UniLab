# Derived from mujocolab/mjlab v1.6.0 (0fb8a681), src/mjlab/utils/buffers/circular_buffer.py.
# Copyright 2025, The mjlab Developers.
# Modified by UniLab for NumPy and UniLab contracts; licensed under Apache-2.0.
"""Circular buffer for storing a history of batched tensor data.

Understanding Dimensions
========================

Internal storage shape: (max_len, batch_size, ...)
                         ↑         ↑
                         time      environments
                         axis      axis

External view (buffer property): (batch_size, max_len, ...)
                                  ↑           ↑
                                  environments time
                                  axis        axis

Why different?
  - Internal: time-first for clean API (self._buffer[pointer] = data)
  - External: batch-first for typical use (iterate over environments)

Backfill Behavior
=================

When you first append to a new or reset batch row, that first value is copied
to ALL history slots for that row:

  buffer = CircularBuffer(max_len=3, batch_size=2)
  buffer.append(np.array([[5.0], [10.0]]))

  # buffer.buffer contains (shape: 2, 3, 1):
  # Batch 0: [5.0, 5.0, 5.0]   <- all slots filled with first value
  # Batch 1: [10.0, 10.0, 10.0] <- all slots filled with first value

Why backfill?
  You always have valid data. If training expects 3 frames of history, you
  don't want garbage/zeros for the first 2 timesteps.

Per-Batch Reset
===============

Reset affects specific batch rows. The circular pointer advances globally, but
reset rows get "first-append" treatment on their next write:

  buffer = CircularBuffer(max_len=3, batch_size=3)

  buffer.append(np.array([[1.0], [10.0], [100.0]]))   # t0
  buffer.append(np.array([[2.0], [20.0], [200.0]]))   # t1
  buffer.append(np.array([[3.0], [30.0], [300.0]]))   # t2

  # buffer.buffer:
  # Batch 0: [1.0, 2.0, 3.0]
  # Batch 1: [10.0, 20.0, 30.0]
  # Batch 2: [100.0, 200.0, 300.0]

  buffer.reset(batch_ids=[1])  # Reset only batch 1

  # buffer.buffer after reset:
  # Batch 0: [1.0, 2.0, 3.0]       <- unchanged
  # Batch 1: [0.0, 0.0, 0.0]       <- zeroed
  # Batch 2: [100.0, 200.0, 300.0] <- unchanged

  # current_length: [3, 0, 3]  <- batch 1 has 0 valid frames

  buffer.append(np.array([[4.0], [99.0], [400.0]]))  # t3

  # buffer.buffer after append:
  # Batch 0: [2.0, 3.0, 4.0]        <- oldest overwritten (normal)
  # Batch 1: [99.0, 99.0, 99.0]     <- BACKFILLED with 99.0
  # Batch 2: [200.0, 300.0, 400.0]  <- oldest overwritten (normal)

Key insight:
  Reset only affects specific batch rows. The pointer keeps advancing for
  everyone, but reset rows get backfilled on their next append.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


class CircularBuffer:
    """Fixed-length circular buffer for batched tensor history.

    Stores history with shape (max_len, batch_size, ...) internally.
    The `buffer` property returns chronologically ordered data (oldest to newest)
    with shape (batch_size, max_len, ...).

    Storage and Retrieval
    ---------------------
    Internal storage (circular):
      ┌─────┬─────┬─────┐
      │  2  │  3  │  1  │  <- pointer at index 1 (newest = 3)
      └─────┴─────┴─────┘
       idx:0  idx:1  idx:2

    buffer property returns (chronological, oldest→newest):
      ┌─────┬─────┬─────┐
      │  1  │  2  │  3  │
      └─────┴─────┴─────┘

    LIFO Retrieval via __getitem__
    -------------------------------
    Given buffer with [1, 2, 3] (oldest to newest):
      buffer[lag=0] -> 3  (most recent)
      buffer[lag=1] -> 2  (one step back)
      buffer[lag=2] -> 1  (oldest)

    Per-Batch Reset
    ---------------
    After reset(batch_ids=[1]):
      Batch 0: [1, 2, 3]  (unchanged)
      Batch 1: [0, 0, 0]  (zeroed, current_length=0)
      Batch 2: [1, 2, 3]  (unchanged)

    Next append backfills reset rows with first value.

    Args:
      max_len: Maximum number of historical frames to retain.
      batch_size: Size of the batch dimension.
    """

    def __init__(self, max_len: int, batch_size: int) -> None:
        if max_len < 1:
            raise ValueError(f"Buffer size must be >= 1, got {max_len}")

        self._max_len = max_len
        self._batch_size = batch_size
        self._pointer: int = -1
        self._buffer: np.ndarray | None = None
        self._all_indices = np.arange(batch_size)
        self._num_pushes = np.zeros(batch_size, dtype=np.int64)
        self._max_len_array = np.full(batch_size, max_len, dtype=np.int64)

    @property
    def batch_size(self) -> int:
        return self._batch_size

    @property
    def max_length(self) -> int:
        return self._max_len

    @property
    def current_length(self) -> np.ndarray:
        """Per-batch count of valid frames. Shape: (batch_size,)."""
        return np.minimum(self._num_pushes, self._max_len_array)

    @property
    def is_initialized(self) -> bool:
        """Check if the buffer has been initialized with at least one append."""
        return self._buffer is not None

    @property
    def buffer(self) -> np.ndarray:
        """History in chronological order (oldest to newest).

        Returns:
          Tensor of shape (batch_size, max_len, ...) where index 0 is oldest
          and index -1 is newest.
        """
        if self._buffer is None:
            raise RuntimeError("Buffer not initialized. Call append() first.")

        start = (self._pointer + 1) % self._max_len
        idx = (np.arange(self._max_len) + start) % self._max_len
        buf = self._buffer[idx]  # (max_len, batch, ...)
        return np.swapaxes(buf, 0, 1)  # (batch, max_len, ...)

    def reset(self, batch_ids: Sequence[int] | np.ndarray | slice | None = None) -> None:
        """Zero out values and counters for specified batch rows.

        Args:
          batch_ids: Batch indices to reset, or None to reset all.
        """
        ids: Sequence[int] | np.ndarray | slice = slice(None) if batch_ids is None else batch_ids
        self._num_pushes[ids] = 0
        if self._buffer is not None:
            self._buffer[:, ids] = 0.0

    def backfill(self, data: np.ndarray, batch_ids: np.ndarray) -> None:
        """Fill the given rows' entire history with one frame, without advancing time.

        Unlike append, the global pointer does not move and other rows are
        untouched. Used after a partial reset: the reset rows get their first
        post-reset frame in every slot (the same backfill their next append would
        apply) while the remaining rows keep their history intact.

        Args:
          data: Tensor of shape (batch_size, ...); only rows at batch_ids are read.
          batch_ids: Batch indices to backfill.
        """
        if data.shape[0] != self._batch_size:
            raise ValueError(f"Expected batch size {self._batch_size}, got {data.shape[0]}")
        if self._buffer is None:
            raise RuntimeError("Buffer not initialized. Call append() first.")

        data = np.asarray(data)
        self._buffer[:, batch_ids] = np.expand_dims(data[batch_ids], axis=0)
        self._num_pushes[batch_ids] = 1

    def append(self, data: np.ndarray) -> None:
        """Append a new frame for all batch elements.

        Args:
          data: Tensor of shape (batch_size, ...).
        """
        if data.shape[0] != self._batch_size:
            raise ValueError(f"Expected batch size {self._batch_size}, got {data.shape[0]}")

        data = np.asarray(data)

        if self._buffer is None:
            self._pointer = -1
            self._buffer = np.empty((self._max_len, *data.shape), dtype=data.dtype)

        self._pointer = (self._pointer + 1) % self._max_len
        self._buffer[self._pointer] = data

        # Backfill only newly initialized rows. After warm-up this branch avoids
        # scanning the full history buffer on every hot-path append.
        is_first_push = self._num_pushes == 0
        if np.any(is_first_push):
            first_ids = np.flatnonzero(is_first_push)
            self._buffer[:, first_ids] = np.expand_dims(data[first_ids], axis=0)

        self._num_pushes += 1

    def __getitem__(self, key: np.ndarray | int) -> np.ndarray:
        """Retrieve lagged frames per batch (LIFO).

        Args:
          key: Per-batch lags (Tensor) or shared lag (int). Shape (batch_size,) or scalar.
        """
        if self._buffer is None:
            raise RuntimeError("Buffer not initialized. Call append() first.")

        if isinstance(key, int):
            key = np.full(self._batch_size, key, dtype=np.int64)
        else:
            key = np.asarray(key, dtype=np.int64)
            if key.ndim == 0:
                key = np.full(self._batch_size, key.item(), dtype=np.int64)

        if key.size != self._batch_size:
            raise ValueError(f"Expected {self._batch_size} lags, got {key.size}")

        # Clamp to the oldest retained frame: without the max_len bound, a lag
        # beyond the buffer length would wrap around to a newer frame once
        # num_pushes exceeds max_len.
        pushes = np.maximum(self._num_pushes, 1)
        max_lag = np.minimum(pushes, self._max_len_array) - 1
        valid = np.maximum(np.minimum(key, max_lag), 0)

        idx = np.remainder(self._pointer - valid, self._max_len)
        return self._buffer[idx, self._all_indices]
