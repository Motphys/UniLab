"""Base async runner for multi-process RL training."""

from __future__ import annotations

import multiprocessing as mp
import os
import signal
import sys
from abc import ABC, abstractmethod
from contextlib import contextmanager
from typing import Any, Callable

from unilab.rl.runtime.ipc.collector_error import (
    ExceptionWrapper,
    build_error_pipe,
    collector_error_guard,
    format_collector_death,
)
from unilab.rl.runtime.ipc.parent_watchdog import (
    DEFAULT_GRACE_PERIOD_S,
    DEFAULT_POLL_INTERVAL_S,
    install_parent_watchdog,
)

_SPAWN_CTX = mp.get_context("spawn")


def _is_ctrl_c_exitcode(exitcode: int) -> bool:
    return exitcode in (-signal.SIGINT, 128 + signal.SIGINT)


def _collector_entry_wrapper(
    target_fn: Callable,
    error_conn: Any,
    kwargs: dict,
) -> None:
    """Top-level wrapper for collector subprocess entry point.

    Ensures ALL exceptions (including import errors and env creation
    failures) are captured and sent to the parent via the error pipe.

    Also installs the parent-death watchdog: ``daemon=True`` only covers
    parents that exit through the Python interpreter, so a SIGKILLed
    learner would otherwise leave the collector orphaned on a stop_event
    nobody sets.
    """
    label = kwargs.pop("_error_label", "collector")
    install_parent_watchdog(
        stop_event=kwargs.get("stop_event"),
        parent_pid=kwargs.pop("_parent_pid", None),
        poll_interval=kwargs.pop("_parent_watchdog_poll_interval", DEFAULT_POLL_INTERVAL_S),
        grace_period=kwargs.pop("_parent_watchdog_grace_period", DEFAULT_GRACE_PERIOD_S),
        label=label,
    )
    with _collector_ignores_ctrl_c():
        with collector_error_guard(
            error_conn=error_conn,
            metrics_queue=kwargs.get("metrics_queue"),
            stop_event=kwargs.get("stop_event"),
            label=label,
        ):
            target_fn(**kwargs)


@contextmanager
def _collector_ignores_ctrl_c():
    """Make the learner the sole owner of Ctrl+C cancellation.

    Terminal-generated SIGINT reaches every process in the foreground process
    group.  A collector interpreting it independently raises before the parent
    learner can set ``stop_event``, which surfaces as a collector crash during
    an ordinary user cancellation.  Ignoring SIGINT here keeps the normal close
    path authoritative; the parent's ``AsyncRunner.close()`` still sets the stop
    event and escalates to SIGTERM if cooperative shutdown exceeds its deadline.
    """
    try:
        previous_handler = signal.signal(signal.SIGINT, signal.SIG_IGN)
    except ValueError:
        # signal.signal is main-thread-only on some platforms. Collector entry
        # points are main-thread spawn targets, but inline test doubles need not
        # install this policy.
        yield
        return
    try:
        yield
    finally:
        signal.signal(signal.SIGINT, previous_handler)


class AsyncRunner(ABC):
    """Base class for async RL algorithms.

    Manages:
    - Shared memory allocation/cleanup
    - Collector process lifecycle
    - Error propagation from collector subprocess
    - Training loop skeleton
    """

    def __init__(
        self,
        env_name: str,
        env_cfg_overrides: dict,
        rl_cfg: dict,
        *,
        device: str | None = None,
        collector_device: str | None = None,
        sim_backend: str = "mujoco",
        num_envs: int = 4096,
    ):
        self.env_name = env_name
        self.env_cfg_overrides = env_cfg_overrides
        self.rl_cfg = rl_cfg
        self.device = device or self._get_default_device()
        self.collector_device = collector_device or self.device
        self.sim_backend = sim_backend
        self.num_envs = num_envs

        self._collector_process: Any = None
        self._stop_event = _SPAWN_CTX.Event()
        self._shared_resources: list = []
        self._error_recv: Any = None
        self._error_send: Any = None
        self._close_complete = False

    @abstractmethod
    def _get_default_device(self) -> str:
        """Get default device (backend-specific)."""
        ...

    @abstractmethod
    def _build_learner(self) -> Any: ...

    @abstractmethod
    def learn(
        self, max_iterations: int, save_interval: int = 50, log_dir: str = "logs"
    ) -> None: ...

    def _start_collector(self, target_fn: Callable, kwargs: dict) -> None:
        self._error_recv, self._error_send = build_error_pipe()

        # Hand the learner PID to the collector for its parent-death
        # watchdog. The child cannot read it back from getppid(): a learner
        # killed during the child's slow spawn boot leaves it reparented to
        # PID 1 before the entry point runs.
        kwargs = {**kwargs, "_parent_pid": os.getpid()}
        self._collector_process = _SPAWN_CTX.Process(
            target=_collector_entry_wrapper,
            args=(target_fn, self._error_send, kwargs),
            daemon=True,
        )
        self._collector_process.start()
        self._error_send.close()
        self._error_send = None

    def _check_collector_alive(self) -> bool:
        """Check if collector is alive. Prints full diagnostic if dead."""
        if self._collector_process is None:
            return True
        if self._collector_process.is_alive():
            return True

        death_info = self._read_collector_error()
        print(f"\n{death_info}\n", file=sys.stderr, flush=True)
        return False

    def _read_collector_error(self) -> str:
        """Read error info from dead collector — pipe first, then exit code."""
        traceback_text = None
        if self._error_recv is not None:
            try:
                if self._error_recv.poll(timeout=0.1):
                    obj = self._error_recv.recv()
                    if isinstance(obj, ExceptionWrapper):
                        traceback_text = obj.exc_msg
            except (EOFError, OSError):
                pass

        exitcode = getattr(self._collector_process, "exitcode", None)
        return format_collector_death(exitcode, traceback_text)

    def close(self) -> None:
        if self._close_complete:
            return
        self._stop_event.set()
        if self._collector_process is not None and self._collector_process.is_alive():
            self._collector_process.join(timeout=10)
            if self._collector_process.is_alive():
                self._collector_process.terminate()
                self._collector_process.join(timeout=5)

        if self._collector_process is not None:
            exitcode = getattr(self._collector_process, "exitcode", None)
            # -15 (SIGTERM) is expected during normal close(). A SIGINT exit is
            # likewise an externally cancelled hot loop, not a collector crash;
            # close() still releases all shared resources below.
            if (
                exitcode is not None
                and exitcode not in (0, -15)
                and not _is_ctrl_c_exitcode(exitcode)
            ):
                death_info = self._read_collector_error()
                print(
                    f"\n[AsyncRunner] Collector exited with code {exitcode}:\n{death_info}\n",
                    file=sys.stderr,
                    flush=True,
                )

        for resource in self._shared_resources:
            if hasattr(resource, "cleanup"):
                resource.cleanup()
            elif hasattr(resource, "close"):
                resource.close()
        self._shared_resources.clear()

        if self._error_recv is not None:
            try:
                self._error_recv.close()
            except Exception:
                pass
            self._error_recv = None
        if self._error_send is not None:
            try:
                self._error_send.close()
            except Exception:
                pass
            self._error_send = None
        self._close_complete = True

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
