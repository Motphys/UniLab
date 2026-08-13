"""Multi-GPU data-parallel rank topology for off-policy training.

Topology rules (config parsing, rank/device mapping, subprocess supervision)
live here so that ``scripts/train_offpolicy.py`` only assembles the flow.
This module covers topology only; parameter sync between ranks is out of
scope for this stage.
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import threading
from pathlib import Path
from typing import Any

UNILAB_DP_RANK = "UNILAB_DP_RANK"
UNILAB_DP_WORLD_SIZE = "UNILAB_DP_WORLD_SIZE"
UNILAB_DP_DEVICES = "UNILAB_DP_DEVICES"
UNILAB_DP_LOG_DIR = "UNILAB_DP_LOG_DIR"

_OFFPOLICY_SCRIPT = Path(__file__).resolve().parents[3] / "scripts" / "train_offpolicy.py"

_WATCHDOG_INTERVAL_S = 0.5
_TERMINATE_TIMEOUT_S = 10.0


def resolve_dp_topology(devices_cfg: Any) -> tuple[int, ...] | None:
    """Normalize ``training.devices`` into an ordered CUDA-index tuple.

    Returns None for the single-card default (null / empty list). The user
    given order is preserved: rank i maps to ``cuda:{devices[i]}``.
    """
    if devices_cfg is None:
        return None
    devices = list(devices_cfg)
    if len(devices) == 0:
        return None
    normalized: list[int] = []
    for entry in devices:
        if isinstance(entry, bool) or not isinstance(entry, int):
            raise ValueError(
                f"training.devices entries must be integer CUDA indices, got {entry!r}"
            )
        if entry < 0:
            raise ValueError(f"training.devices entries must be non-negative, got {entry}")
        normalized.append(int(entry))
    if len(set(normalized)) != len(normalized):
        raise ValueError(f"training.devices must not contain duplicates, got {normalized}")
    return tuple(normalized)


def current_dp_rank() -> int:
    """Data-parallel rank of this process (0 when not spawned as a rank)."""
    return int(os.environ.get(UNILAB_DP_RANK, "0"))


def current_dp_world_size() -> int:
    """Data-parallel world size of this process (1 when not spawned as a rank)."""
    return int(os.environ.get(UNILAB_DP_WORLD_SIZE, "1"))


def validate_dp_launchable(devices: tuple[int, ...]) -> None:
    """Fail fast at launch time when the host lacks any requested CUDA device."""
    import torch

    device_count = torch.cuda.device_count()
    missing = [index for index in devices if index >= device_count]
    if missing:
        raise ValueError(
            f"training.devices={list(devices)} requires CUDA device index(es) {missing}, "
            f"but torch.cuda.device_count()={device_count}"
        )


def apply_dp_rank_config(cfg: Any, devices: tuple[int, ...] | None, rank: int) -> None:
    """Rewrite per-rank device and seed into the composed Hydra config.

    Rank 0 keeps the configured seed; rank i>0 trains with ``seed + i`` until
    init broadcast lands in a later stage.
    """
    if devices is None:
        return
    if cfg.training.device is not None:
        raise ValueError(
            "training.device and training.devices are mutually exclusive; "
            "use training.devices for multi-GPU data-parallel training and leave "
            "training.device null"
        )
    if rank < 0 or rank >= len(devices):
        raise ValueError(
            f"data-parallel rank {rank} is out of range for training.devices={list(devices)}"
        )
    from omegaconf import open_dict

    with open_dict(cfg):
        cfg.training.device = f"cuda:{devices[rank]}"
        cfg.algo.seed = int(cfg.algo.seed) + rank


def _sigterm_system_exit(signum: int, _frame: Any) -> None:
    raise SystemExit(f"data-parallel rank 0 received signal {signum}")


class DpRankSupervisor:
    """Rank-0 supervisor that spawns and watches data-parallel rank subprocesses.

    Ranks 1..N-1 re-run ``scripts/train_offpolicy.py`` with the same Hydra
    argv plus the ``UNILAB_DP_*`` environment; each spawned rank builds its
    own learner+collector pair through the regular runner path. If any rank
    subprocess dies with a non-zero exit code while active, the supervisor
    delivers SIGTERM to rank 0 so the runner lifecycle unwinds through the
    normal try/finally (``runner.close()``), and ``__exit__`` tears down the
    remaining ranks.
    """

    def __init__(self, devices: tuple[int, ...], log_dir: str) -> None:
        self._devices = tuple(devices)
        self._log_dir = log_dir
        self._world_size = len(self._devices)
        self._children: list[subprocess.Popen] = []
        self._watchdog_stop = threading.Event()
        self._watchdog: threading.Thread | None = None
        self._previous_sigterm_handler: Any = None

    def __enter__(self) -> DpRankSupervisor:
        if self._world_size <= 1:
            # Single-device degenerate case: nothing to spawn or watch.
            return self
        base_env = os.environ | {
            UNILAB_DP_WORLD_SIZE: str(self._world_size),
            UNILAB_DP_DEVICES: ",".join(str(index) for index in self._devices),
            UNILAB_DP_LOG_DIR: self._log_dir,
        }
        try:
            for rank in range(1, self._world_size):
                env = base_env | {UNILAB_DP_RANK: str(rank)}
                self._children.append(
                    subprocess.Popen(
                        [sys.executable, str(_OFFPOLICY_SCRIPT), *sys.argv[1:]],
                        env=env,
                    )
                )
        except BaseException:
            self._terminate_children()
            raise
        self._previous_sigterm_handler = signal.signal(signal.SIGTERM, _sigterm_system_exit)
        self._watchdog = threading.Thread(
            target=self._watchdog_loop,
            name="dp-rank-watchdog",
            daemon=True,
        )
        self._watchdog.start()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> bool:
        self._watchdog_stop.set()
        if self._watchdog is not None:
            self._watchdog.join(timeout=_TERMINATE_TIMEOUT_S)
            self._watchdog = None
        if self._previous_sigterm_handler is not None:
            signal.signal(signal.SIGTERM, self._previous_sigterm_handler)
            self._previous_sigterm_handler = None

        # Ranks that already exited on their own keep their exit code;
        # non-zero means the data-parallel run failed even if rank 0 is fine.
        failed = [
            (rank, child.returncode)
            for rank, child in enumerate(self._children, start=1)
            if child.returncode is not None and child.returncode != 0
        ]
        self._terminate_children()
        if failed:
            message = (
                "data-parallel rank subprocess(es) failed: "
                + ", ".join(f"rank {rank} exit code {code}" for rank, code in failed)
            )
            if exc_type is None:
                raise RuntimeError(message)
            print(f"[dp_launcher] {message}", file=sys.stderr)
        return False

    def _terminate_children(self) -> None:
        alive = [child for child in self._children if child.poll() is None]
        for child in alive:
            child.terminate()
        for child in alive:
            try:
                child.wait(timeout=_TERMINATE_TIMEOUT_S)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait()

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
                    f"[dp_launcher] rank {rank} subprocess pid={child.pid} exited "
                    f"unexpectedly with code {exit_code}; shutting down rank 0",
                    file=sys.stderr,
                )
                os.kill(os.getpid(), signal.SIGTERM)
                return
