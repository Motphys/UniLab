"""Cross-process error propagation for collector subprocesses.

Uses a Pipe + ExceptionWrapper pattern (proven by PyTorch DataLoader and
CPython's ProcessPoolExecutor) to ensure collector tracebacks always
reach the parent process — even when stderr is lost or interleaved.
"""

from __future__ import annotations

import signal
import sys
import traceback
from contextlib import contextmanager
from typing import Any

import multiprocessing as _mp

_SPAWN_CTX = _mp.get_context("spawn")


class ExceptionWrapper:
    """Picklable exception + traceback for cross-process propagation.

    Stores the exception type and a pre-formatted traceback string
    (not exc_info — that would create reference cycles preventing GC
    of objects in the exception scope).
    """

    def __init__(self, where: str = "in collector"):
        exc_info = sys.exc_info()
        self.exc_type = exc_info[0]
        self.exc_msg = "".join(traceback.format_exception(*exc_info))
        self.where = where

    def reraise(self) -> None:
        msg = (
            f"Caught {self.exc_type.__name__} {self.where}.\n"
            f"Original traceback:\n{self.exc_msg}"
        )
        try:
            raise self.exc_type(msg)
        except TypeError:
            raise RuntimeError(msg) from None


def create_error_pipe() -> tuple[Any, Any]:
    """Create a unidirectional pipe for error reporting.

    Returns (recv_conn, send_conn). Parent keeps recv, child gets send.
    """
    return _SPAWN_CTX.Pipe(duplex=False)


@contextmanager
def collector_error_guard(
    error_conn: Any | None = None,
    metrics_queue: Any | None = None,
    stop_event: Any | None = None,
    label: str = "collector",
):
    """Context manager that catches all exceptions in collector subprocess.

    Sends a picklable ExceptionWrapper through the error pipe so the
    parent process can surface the full traceback. Also pushes to
    metrics_queue for fast-path detection by the training loop.
    """
    try:
        yield
    except Exception:
        wrapper = ExceptionWrapper(where=f"in {label}")
        print(
            f"\n{'=' * 60}\n[{label.upper()} CRASH]\n"
            f"{wrapper.exc_msg}\n{'=' * 60}\n",
            file=sys.stderr,
            flush=True,
        )
        if error_conn is not None:
            try:
                error_conn.send(wrapper)
            except Exception:
                pass
        if metrics_queue is not None:
            try:
                metrics_queue.put_nowait({"error": wrapper.exc_msg})
            except Exception:
                pass
        if stop_event is not None:
            try:
                stop_event.set()
            except Exception:
                pass
        raise


def format_collector_death(exitcode: int | None, traceback_text: str | None = None) -> str:
    """Format a human-readable death report for a collector process."""
    parts = []

    if traceback_text:
        parts.append("Collector process crashed.")
        parts.append(f"\n{'─' * 50}")
        parts.append(traceback_text.rstrip())
        parts.append(f"{'─' * 50}")
    elif exitcode is not None and exitcode < 0:
        sig_num = -exitcode
        sig_name = _signal_name(sig_num)
        parts.append(f"Collector process killed by signal {sig_num} ({sig_name}).")
        parts.append("  No Python traceback available — killed externally.")
        if sig_num == 9:
            parts.append("  Common cause: OOM killer. Check: dmesg | grep -i oom")
        elif sig_num == 11:
            parts.append("  Common cause: segfault in native code (C++/CUDA).")
    elif exitcode is not None:
        parts.append(f"Collector process exited with code {exitcode}.")
    else:
        parts.append("Collector process died (exit code unknown).")

    return "\n".join(parts)


def _signal_name(sig_num: int) -> str:
    try:
        return signal.Signals(sig_num).name
    except (ValueError, AttributeError):
        return f"SIG{sig_num}"
