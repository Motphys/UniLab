"""Tests for AsyncRunner base class."""

from __future__ import annotations

import multiprocessing as mp
import signal
import time
from typing import Any
from unittest.mock import MagicMock

import pytest

from unilab.ipc.async_runner import AsyncRunner

_SPAWN_CTX = mp.get_context("spawn")


# ---------------------------------------------------------------------------
# Minimal concrete implementation for testing
# ---------------------------------------------------------------------------


class _StubRunner(AsyncRunner):
    """Minimal concrete AsyncRunner — used only for unit tests."""

    def _get_default_device(self) -> str:
        return "cpu"

    def _build_learner(self) -> Any:
        return None

    def _collector_fn(self, stop_event: Any, **kwargs) -> None:
        pass

    def learn(self, max_iterations: int, save_interval: int = 50, log_dir: str = "logs") -> None:
        pass


def _make_runner(rl_cfg=None, **kwargs) -> _StubRunner:
    return _StubRunner(
        env_name="DummyEnv",
        env_cfg_overrides={},
        rl_cfg=rl_cfg or {},
        **kwargs,
    )


# ---------------------------------------------------------------------------
# Initialisation
# ---------------------------------------------------------------------------


def test_init_stores_env_name():
    r = _make_runner()
    assert r.env_name == "DummyEnv"


def test_init_stores_rl_cfg():
    cfg = {"gamma": 0.99}
    r = _make_runner(rl_cfg=cfg)
    assert r.rl_cfg == cfg


def test_init_device_explicit():
    r = _make_runner(device="cpu")
    assert r.device == "cpu"


def test_init_device_default():
    """When device=None, uses _get_default_device() which returns 'cpu' for stub."""
    r = _make_runner()
    assert r.device == "cpu"


def test_init_collector_device_defaults_to_device():
    r = _make_runner(device="cpu")
    assert r.collector_device == "cpu"


def test_init_collector_device_explicit():
    r = _make_runner(device="cpu", collector_device="cpu")
    assert r.collector_device == "cpu"


def test_init_num_envs():
    r = _make_runner(num_envs=64)
    assert r.num_envs == 64


def test_init_shared_resources_empty():
    r = _make_runner()
    assert r._shared_resources == []


def test_init_collector_process_none():
    r = _make_runner()
    assert r._collector_process is None


# ---------------------------------------------------------------------------
# close() — no collector
# ---------------------------------------------------------------------------


def test_close_with_no_collector_does_not_raise():
    r = _make_runner()
    r.close()  # _collector_process is None → should be a no-op


def test_close_is_idempotent():
    r = _make_runner()
    r.close()
    r.close()


# ---------------------------------------------------------------------------
# close() — resource cleanup
# ---------------------------------------------------------------------------


def test_close_calls_cleanup_on_resources():
    """Resources with cleanup() method must be cleaned up on close()."""
    r = _make_runner()
    mock_res = MagicMock(spec=["cleanup"])
    r._shared_resources.append(mock_res)
    r.close()
    mock_res.cleanup.assert_called_once()


def test_close_calls_close_on_resources_without_cleanup():
    """Resources without cleanup() but with close() must have close() called."""
    r = _make_runner()
    mock_res = MagicMock(spec=["close"])
    r._shared_resources.append(mock_res)
    r.close()
    mock_res.close.assert_called_once()


def test_close_handles_multiple_resources():
    r = _make_runner()
    resources = [MagicMock(spec=["cleanup"]) for _ in range(3)]
    r._shared_resources.extend(resources)
    r.close()
    for res in resources:
        res.cleanup.assert_called_once()


# ---------------------------------------------------------------------------
# close() — with live collector process
# ---------------------------------------------------------------------------


def _worker_wait_for_stop(stop_event) -> None:
    """Cooperative worker: exits as soon as stop_event is set."""
    stop_event.wait(timeout=30)


def test_close_joins_running_collector():
    """close() must signal the stop event and reap the collector process.

    Under heavy CI load a spawned process may still be importing the test module
    when close() hits its timeout path, so SIGTERM is also an acceptable outcome
    as long as the collector does not leak.
    """
    r = _make_runner()
    r._collector_process = _SPAWN_CTX.Process(
        target=_worker_wait_for_stop,
        args=(r._stop_event,),
        daemon=True,
    )
    r._collector_process.start()
    assert r._collector_process.is_alive()

    r.close()

    assert r._stop_event.is_set()
    assert not r._collector_process.is_alive()
    assert r._collector_process.exitcode in (0, -signal.SIGTERM)


# ---------------------------------------------------------------------------
# _start_collector
# ---------------------------------------------------------------------------


def _noop_collector(stop_event) -> None:
    stop_event.wait(timeout=30)


def test_start_collector_spawns_process():
    """_start_collector() must create and start a subprocess."""
    r = _make_runner()
    r._start_collector(target_fn=_noop_collector, kwargs={"stop_event": r._stop_event})
    assert r._collector_process is not None
    assert r._collector_process.is_alive()
    r.close()


# ---------------------------------------------------------------------------
# __del__ exception handling
# ---------------------------------------------------------------------------


def test_del_does_not_raise_even_if_close_fails():
    """__del__ must swallow exceptions from close()."""
    r = _make_runner()
    # Force close() to fail by corrupting internal state
    r._shared_resources = None  # type: ignore[assignment]  # will raise in close()
    # __del__ calls close() and must not propagate the exception
    r.__del__()  # noqa: PLC2801  (explicit __del__ call for test)
