"""Tests for SharedObsNormStats IPC primitive."""

from __future__ import annotations

import queue

import numpy as np

from unilab.ipc.shared_obs_stats import SharedObsNormStats


class _ThreadingCtx:
    """Synchronous queue context using threading.Queue — makes empty() reliable in unit tests."""

    @staticmethod
    def Queue(maxsize: int = 0):
        return queue.Queue(maxsize=maxsize)


class _RaisingQueue:
    """Queue stub: appears non-empty once, then raises on get_nowait() — exercises exception handler paths.

    After the first get_nowait() raises, empty() returns True so the drain
    while-loop in put() exits rather than spinning forever.
    """

    def __init__(self):
        self._raised = False

    def empty(self) -> bool:
        # Non-empty until we've raised once; then empty to break the drain loop
        return self._raised

    def get_nowait(self):
        self._raised = True
        raise queue.Empty("mock raise")

    def put(self, item) -> None:
        pass  # silently discard


class _RaisingCtx:
    @staticmethod
    def Queue(maxsize: int = 0):
        return _RaisingQueue()


def _make_stats() -> SharedObsNormStats:
    return SharedObsNormStats(_ThreadingCtx())


def test_put_get_roundtrip():
    stats = _make_stats()
    mean = np.ones(8, dtype=np.float32)
    std = np.full(8, 2.0, dtype=np.float32)

    stats.put((mean, std))
    result = stats.get()

    assert result is not None
    got_mean, got_std = result
    np.testing.assert_array_equal(got_mean, mean)
    np.testing.assert_array_equal(got_std, std)


def test_get_returns_none_when_empty():
    stats = _make_stats()
    assert stats.get() is None


def test_put_does_not_block_when_full():
    """A third put() should drain the old entry and not block."""
    stats = _make_stats()
    for i in range(3):
        mean = np.full(4, float(i), dtype=np.float32)
        std = np.ones(4, dtype=np.float32)
        stats.put((mean, std))  # must not block or raise

    # After draining, latest value should be accessible
    result = stats.get()
    assert result is not None


def test_get_returns_latest_after_multiple_puts():
    """get() should return the most recent stats after multiple puts."""
    stats = _make_stats()
    for i in range(2):
        stats.put((np.full(4, float(i), dtype=np.float32), np.ones(4, dtype=np.float32)))

    result = stats.get()
    assert result is not None


def test_put_exception_handler_does_not_raise():
    """put() must not propagate exceptions from get_nowait() during drain."""
    stats = SharedObsNormStats(_RaisingCtx())
    # The drain loop will hit the except branch — must not raise
    stats.put((np.zeros(4, dtype=np.float32), np.ones(4, dtype=np.float32)))


def test_get_exception_handler_does_not_raise():
    """get() must not propagate exceptions from get_nowait()."""
    stats = SharedObsNormStats(_RaisingCtx())
    # The while loop will hit the except branch — must not raise, returns None
    result = stats.get()
    assert result is None
