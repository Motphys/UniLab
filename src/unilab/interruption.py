"""Interruption behavior shared by entrypoints and child-process routers."""

from __future__ import annotations

import signal
import subprocess
import sys
from collections.abc import Sequence
from functools import wraps
from typing import Any, Callable, TypeVar

_F = TypeVar("_F", bound=Callable[..., Any])


def graceful_keyboard_interrupt(entrypoint: _F) -> _F:
    """Let entrypoint cleanup finish, then exit Ctrl+C without a traceback.

    Runners own their ``finally`` blocks.  This outer boundary converts only the
    resulting ``KeyboardInterrupt`` into the conventional shell status after all
    collector, logger, and distributed-process cleanup has completed.
    """

    @wraps(entrypoint)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        try:
            return entrypoint(*args, **kwargs)
        except KeyboardInterrupt:
            print("Interrupted by Ctrl+C; shutdown completed.", file=sys.stderr, flush=True)
            raise SystemExit(128 + signal.SIGINT) from None

    return wrapper  # type: ignore[return-value]


def run_process_with_graceful_interrupt(command: Sequence[str]) -> int:
    """Run a child while treating the first Ctrl+C as cooperative shutdown.

    The first SIGINT is already delivered to the complete foreground process
    group. The child owns cooperative cleanup, so this routing process records
    the request and keeps waiting. A second SIGINT escalates to terminate/kill.
    """
    interrupted = False
    force_exit = False

    def handle_ctrl_c(signum: int, frame: Any) -> None:
        nonlocal interrupted, force_exit
        if not interrupted:
            interrupted = True
            return
        force_exit = True

    previous_handler = signal.signal(signal.SIGINT, handle_ctrl_c)
    try:
        try:
            process = subprocess.Popen(command)
        except KeyboardInterrupt:
            return 130
        while process.poll() is None:
            try:
                process.wait(timeout=0.1)
            except subprocess.TimeoutExpired:
                pass
            if force_exit:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    pass
                if process.poll() is None:
                    process.kill()
                    try:
                        process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        pass
                break
    finally:
        signal.signal(signal.SIGINT, previous_handler)

    returncode = process.returncode
    if returncode is None:
        return 1
    return 130 if interrupted else returncode
