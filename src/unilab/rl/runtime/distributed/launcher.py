"""Single-node multi-GPU data-parallel launch and rank supervision.

Merged from unilab_rl @ 3654e4b (unilab-rl 1.4.15), Apache-2.0; previously
``uni_rl.ipc.dp_launcher``. Rank/device topology context lives in
:mod:`unilab.rl.runtime.distributed.context`; collector CPU partitioning lives
in :mod:`unilab.rl.runtime.distributed.affinity`.
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, Literal, Sequence

from unilab.rl.runtime.distributed.context import (
    UNILAB_DP_LOG_DIR,
    UNILAB_DP_RANK,
    UNILAB_DP_WORLD_SIZE,
    selected_visible_entries,
)

_WATCHDOG_INTERVAL_S = 0.5
_COOPERATIVE_EXIT_GRACE_S = 10.0
_TERMINATE_TIMEOUT_S = 10.0
# Grace window for sibling ranks to finish their own runs once rank 0
# completed normally. Ranks run the same config, so they should land within
# startup skew of each other; exceeding the grace is treated as a failure.
_NORMAL_EXIT_GRACE_S = 600.0


def launch_torchrun_workers(
    *,
    world_size: int,
    script_path: str | os.PathLike[str],
    argv: Sequence[str],
    log_dir: str,
) -> None:
    """Launch one single-GPU torchrun worker per selected visibility entry.

    PyTorch elastic owns worker supervision and failure propagation. Each
    worker re-enters the regular training script with the original Hydra
    arguments, while ``UNILAB_DP_LOG_DIR`` supplies one canonical rank-0-owned
    run directory.
    """
    selected = selected_visible_entries(world_size=world_size)
    launch_env = os.environ.copy()
    # torchrun assigns one ordinal per worker; the worker remaps this list to
    # its own single-entry visibility before constructing CUDA resources.
    launch_env["CUDA_VISIBLE_DEVICES"] = ",".join(selected)
    launch_env.setdefault("NCCL_P2P_DISABLE", "1")
    launch_env.setdefault("NCCL_SHM_DISABLE", "1")
    launch_env[UNILAB_DP_LOG_DIR] = str(log_dir)
    argv = [
        "--standalone",
        "--nnodes=1",
        f"--nproc_per_node={world_size}",
        str(Path(script_path).resolve()),
        *argv,
    ]
    print(
        f"Launching RSL-RL data parallel training on {world_size} rank-local "
        "CUDA_VISIBLE_DEVICES entries.",
        flush=True,
    )
    _run_torchrun_process(argv, launch_env)


def _run_torchrun_process(argv: list[str], env: dict[str, str]) -> None:
    """Run torch elastic in-process so Ctrl+C is a cancellation, not a crash."""
    missing = object()
    previous_env = {key: os.environ.get(key, missing) for key in env}
    try:
        os.environ.update(env)
        # Keep torch imports lazy: importing this launcher must not pull the
        # distributed runtime into lightweight package initialization paths.
        from torch.distributed.elastic.multiprocessing import SignalException

        try:
            _run_torchrun_elastic(argv)
        except SignalException as exc:
            if exc.sigval == signal.SIGINT:
                print("Training interrupted by Ctrl+C.", file=sys.stderr, flush=True)
                raise SystemExit(128 + signal.SIGINT) from None
            raise
    finally:
        for key, previous in previous_env.items():
            if previous is missing:
                os.environ.pop(key, None)
            else:
                os.environ[key] = str(previous)


def _run_torchrun_elastic(argv: list[str]) -> None:
    from torch.distributed.run import parse_args, run

    run(parse_args(argv))


def _current_entry_command() -> list[str]:
    """Build the command used to re-enter the current training entry point.

    The entry script belongs to the downstream owner (for example UniLab),
    while this package only owns the data-parallel supervisor.  Reusing
    ``sys.argv[0]`` keeps that ownership boundary intact and also preserves
    the original script path for source checkouts and installed consumers.
    """
    return [sys.executable, sys.argv[0], *sys.argv[1:]]


def _sigterm_system_exit(signum: int, _frame: Any) -> None:
    raise SystemExit(f"data-parallel rank 0 received signal {signum}")


def _is_interrupt_exit(exitcode: object) -> bool:
    return exitcode in (128 + signal.SIGINT, -signal.SIGINT)


def _signal_process_group(child: subprocess.Popen, signum: int) -> None:
    """Signal one rank and all of the worker processes it owns."""
    try:
        if hasattr(os, "killpg"):
            os.killpg(child.pid, signum)
        else:  # pragma: no cover - Windows fallback for non-CUDA test hosts.
            child.send_signal(signum)
    except ProcessLookupError:
        pass


def _process_group_exists(child: subprocess.Popen) -> bool:
    """Return whether a rank process group still owns any live processes."""
    # Reap an exited group leader before probing the group. Otherwise the
    # leader may remain a zombie and make ``killpg(..., 0)`` look alive for the
    # entire escalation timeout even though no rank resources remain.
    child.poll()
    if not hasattr(os, "killpg"):  # pragma: no cover - Windows fallback.
        return child.returncode is None
    try:
        os.killpg(child.pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


class DpRankSupervisor:
    """Rank-0 supervisor that spawns and watches data-parallel rank subprocesses.

    Ranks 1..N-1 re-run the current entry script (``sys.argv[0]``, e.g.
    ``unilab/scripts/train_sac.py``) with the same Hydra
    argv plus the ``UNILAB_DP_*`` environment; each spawned rank builds its
    own learner+collector pair through the regular runner path. If any rank
    subprocess dies with a non-zero exit code while active, the supervisor
    delivers SIGTERM to rank 0 so the runner lifecycle unwinds through the
    normal try/finally (``runner.close()``), and ``__exit__`` tears down the
    remaining ranks. Each spawned rank owns a separate process group containing
    its collector. Rank 0 forwards Ctrl+C to those groups for cooperative
    cleanup before escalating to SIGTERM/SIGKILL.
    """

    def __init__(self, world_size: int, log_dir: str) -> None:
        self._selected = selected_visible_entries(world_size=world_size)
        self._log_dir = log_dir
        self._world_size = len(self._selected)
        self._children: list[subprocess.Popen] = []
        self._watchdog_stop = threading.Event()
        self._watchdog: threading.Thread | None = None
        self._previous_signal_handlers: dict[int, Any] = {}
        self._shutdown_requested = False
        self._interrupt_requested = False

    def __enter__(self) -> DpRankSupervisor:
        if self._world_size <= 1:
            # Single-device degenerate case: nothing to spawn or watch.
            return self
        base_env = os.environ | {
            UNILAB_DP_WORLD_SIZE: str(self._world_size),
            UNILAB_DP_LOG_DIR: self._log_dir,
        }
        self._install_signal_handlers()
        try:
            for rank, entry in enumerate(self._selected[1:], start=1):
                env = base_env | {
                    UNILAB_DP_RANK: str(rank),
                    "CUDA_VISIBLE_DEVICES": entry,
                }
                self._children.append(
                    subprocess.Popen(
                        _current_entry_command(),
                        env=env,
                        # Rank-local collectors inherit this group. The terminal
                        # only interrupts rank 0; the supervisor then forwards
                        # one coordinated SIGINT to each complete rank tree.
                        start_new_session=os.name == "posix",
                    )
                )
        except BaseException:
            self._request_children_shutdown()
            try:
                self._wait_for_children(_COOPERATIVE_EXIT_GRACE_S)
            finally:
                try:
                    self._terminate_children()
                finally:
                    self._restore_signal_handlers()
            raise
        self._watchdog = threading.Thread(
            target=self._watchdog_loop,
            name="dp-rank-watchdog",
            daemon=True,
        )
        self._watchdog.start()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> Literal[False]:
        self._watchdog_stop.set()
        failed: list[tuple[int, object]] = []
        try:
            if self._watchdog is not None:
                self._watchdog.join(timeout=_TERMINATE_TIMEOUT_S)
                self._watchdog = None

            # Ranks that already exited on their own keep their exit code;
            # non-zero means the data-parallel run failed even if rank 0 is fine.
            failed = [
                (rank, child.returncode)
                for rank, child in enumerate(self._children, start=1)
                if child.returncode is not None
                and child.returncode != 0
                and not (self._interrupt_requested and _is_interrupt_exit(child.returncode))
            ]
            if exc_type is None and not failed:
                # Normal rank-0 completion: give sibling ranks a grace window to
                # finish their own runs instead of cutting them short.
                unfinished = self._wait_for_children(_NORMAL_EXIT_GRACE_S)
                for rank, child in unfinished:
                    print(
                        f"[launcher] rank {rank} subprocess pid={child.pid} did not "
                        f"exit within {_NORMAL_EXIT_GRACE_S}s of rank 0 completion",
                        file=sys.stderr,
                    )
                    failed.append((rank, "timeout"))
                for rank, child in enumerate(self._children, start=1):
                    if child.returncode not in (None, 0) and not any(
                        failed_rank == rank for failed_rank, _ in failed
                    ):
                        failed.append((rank, child.returncode))
            else:
                # For Ctrl+C this is normally a no-op because the signal handler
                # already forwarded SIGINT. It also makes arbitrary rank-0
                # failures unwind sibling runner lifecycles cooperatively.
                self._request_children_shutdown()
                self._wait_for_children(_COOPERATIVE_EXIT_GRACE_S)
        finally:
            # Always reap complete rank process groups, including collectors
            # orphaned by a rank that exited before finishing its own cleanup.
            try:
                self._terminate_children()
            finally:
                self._restore_signal_handlers()
        if failed:
            message = "data-parallel rank subprocess(es) failed: " + ", ".join(
                f"rank {rank} exit code {code}" for rank, code in failed
            )
            if exc_type is None:
                raise RuntimeError(message)
            print(f"[launcher] {message}", file=sys.stderr)
        return False

    def _install_signal_handlers(self) -> None:
        try:
            self._previous_signal_handlers[signal.SIGINT] = signal.signal(
                signal.SIGINT, self._handle_sigint
            )
            self._previous_signal_handlers[signal.SIGTERM] = signal.signal(
                signal.SIGTERM, self._handle_sigterm
            )
        except BaseException:
            self._restore_signal_handlers()
            raise

    def _restore_signal_handlers(self) -> None:
        for signum, previous in self._previous_signal_handlers.items():
            signal.signal(signum, previous)
        self._previous_signal_handlers.clear()

    def _handle_sigint(self, signum: int, frame: Any) -> None:
        self._interrupt_requested = True
        self._request_children_shutdown()
        previous = self._previous_signal_handlers.get(signal.SIGINT, signal.default_int_handler)
        if previous == signal.SIG_IGN:
            return
        if previous == signal.SIG_DFL:
            signal.default_int_handler(signum, frame)
            return
        previous(signum, frame)

    def _handle_sigterm(self, signum: int, frame: Any) -> None:
        # A child-rank watchdog failure uses SIGTERM to unwind rank 0. Sibling
        # rank trees still receive SIGINT so their Python finally blocks run.
        self._request_children_shutdown()
        _sigterm_system_exit(signum, frame)

    def _request_children_shutdown(self) -> None:
        if self._shutdown_requested:
            return
        self._shutdown_requested = True
        for child in self._children:
            if _process_group_exists(child):
                _signal_process_group(child, signal.SIGINT)

    def _wait_for_children(self, timeout: float) -> list[tuple[int, subprocess.Popen]]:
        deadline = time.monotonic() + max(float(timeout), 0.0)
        pending = list(enumerate(self._children, start=1))
        while pending:
            pending = [(rank, child) for rank, child in pending if child.poll() is None]
            if not pending:
                break
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            try:
                pending[0][1].wait(timeout=min(_WATCHDOG_INTERVAL_S, remaining))
            except subprocess.TimeoutExpired:
                pass
        return pending

    @staticmethod
    def _wait_for_process_groups(
        children: list[subprocess.Popen], timeout: float
    ) -> list[subprocess.Popen]:
        deadline = time.monotonic() + max(float(timeout), 0.0)
        pending = [child for child in children if _process_group_exists(child)]
        while pending:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            time.sleep(min(_WATCHDOG_INTERVAL_S, remaining))
            pending = [child for child in pending if _process_group_exists(child)]
        return pending

    def _terminate_children(self) -> None:
        groups = [child for child in self._children if _process_group_exists(child)]
        for child in groups:
            _signal_process_group(child, signal.SIGTERM)
        groups = self._wait_for_process_groups(groups, _TERMINATE_TIMEOUT_S)
        for child in groups:
            _signal_process_group(child, signal.SIGKILL)
        self._wait_for_process_groups(groups, _TERMINATE_TIMEOUT_S)

        # Reap every direct rank child after its complete process group has
        # stopped so no zombie or Popen handle survives supervisor teardown.
        for child in self._children:
            if child.poll() is not None:
                continue
            try:
                child.wait(timeout=_TERMINATE_TIMEOUT_S)
            except subprocess.TimeoutExpired:
                _signal_process_group(child, signal.SIGKILL)
                child.wait(timeout=_TERMINATE_TIMEOUT_S)

    def _watchdog_loop(self) -> None:
        while not self._watchdog_stop.wait(_WATCHDOG_INTERVAL_S):
            for rank, child in enumerate(self._children, start=1):
                exit_code = child.poll()
                if exit_code is None:
                    continue
                if exit_code == 0:
                    # A rank that finishes cleanly (e.g. while rank 0 is still
                    # in playback) is not a failure; keep watching the rest.
                    continue
                print(
                    f"[launcher] rank {rank} subprocess pid={child.pid} exited "
                    f"unexpectedly with code {exit_code}; shutting down rank 0",
                    file=sys.stderr,
                )
                os.kill(os.getpid(), signal.SIGTERM)
                return
