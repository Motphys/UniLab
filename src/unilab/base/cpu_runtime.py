"""Process-local CPU confinement for envs that own an explicit CPU block.

Multi-rank off-policy data-parallel runs partition host CPUs so each rank's
collector owns one contiguous block (``training.dp_collector_cpu_ids``, routed
into ``EnvCfg.cpu_ids``). The MuJoCo BatchEnvPool already pins its physics
workers to that block, but the collector's host-side compute did not follow:
Numba parallel kernels (``unisim.backend.body_state`` and the
motion-tracking kernels) size their pool from the host CPU count and leave
placement to the OS, so they drift across rank boundaries and compete with
sibling ranks' pinned physics workers.

``apply_env_cpu_runtime`` is the generic env-level counterpart, applied once on
the env-construction cold path — before managers, backend materialization, or
the first Numba parallel call exist:

- ``os.sched_setaffinity`` pins the calling thread to the block, and every
  already-running thread (e.g. BLAS pools spawned at ``import numpy``) is
  pinned individually via ``/proc/self/task``; threads spawned later
  (including Numba's lazily-launched pool) inherit the mask.
- ``numba.set_num_threads(len(cpu_ids))`` sizes Numba's pool to the block
  instead of the host, unless the operator pinned ``NUMBA_NUM_THREADS``
  (mirroring the motion-kernel runtime policy).

The function is backend-agnostic: ``EnvCfg.cpu_ids`` is the single source of
truth, so any backend whose env declares a block gets the same confinement.
"""

from __future__ import annotations

import os
import warnings
from collections.abc import Sequence

_PROC_TASK_DIR = "/proc/self/task"


def _confine_existing_threads(ids: set[int]) -> None:
    """Pin already-running threads; later threads inherit the caller's mask.

    Native pools spawned before env construction (OpenBLAS spawns its workers
    at ``import numpy``) would otherwise keep the host-wide mask and compete
    with sibling ranks' pinned CPUs. Threads may exit between listing and
    pinning; those races are ignored.
    """
    # Resolved at call time (not import) so tests can monkeypatch the seam;
    # ``getattr`` keeps this checkable on platforms where typeshed hides the
    # Linux-only symbol (mypy ``attr-defined`` on darwin).
    sched_setaffinity = getattr(os, "sched_setaffinity", None)
    if sched_setaffinity is None or not os.path.isdir(_PROC_TASK_DIR):
        return
    for entry in os.listdir(_PROC_TASK_DIR):
        try:
            sched_setaffinity(int(entry), ids)
        except OSError:
            continue


def apply_env_cpu_runtime(cpu_ids: Sequence[int] | None) -> None:
    """Confine this process's host-side compute to the env-owned CPU block.

    Cold path only: call from env construction, before the backend pool,
    managers, or any Numba parallel kernel exist. ``None`` (the single-rank
    default) is a no-op so the default path stays bit-identical.

    Structural validation (non-empty, unique, non-negative ints) is owned by
    ``EnvCfg.validate``; this function fails closed on CPU ids that are not
    available to the process.
    """
    if cpu_ids is None:
        return
    ids = {int(cpu_id) for cpu_id in cpu_ids}

    sched_setaffinity = getattr(os, "sched_setaffinity", None)
    sched_getaffinity = getattr(os, "sched_getaffinity", None)
    if sched_setaffinity is not None and sched_getaffinity is not None:
        available = set(sched_getaffinity(0))
        missing = sorted(ids - available)
        if missing:
            raise ValueError(
                f"EnvCfg.cpu_ids entries {missing} are not available to this process "
                f"(sched_getaffinity={sorted(available)})"
            )
        sched_setaffinity(0, ids)
        _confine_existing_threads(ids)
    else:
        warnings.warn(
            "EnvCfg.cpu_ids process confinement requires os.sched_setaffinity "
            "(Linux); only the Numba thread cap is applied",
            stacklevel=2,
        )

    if "NUMBA_NUM_THREADS" in os.environ:
        return
    from numba import set_num_threads

    try:
        set_num_threads(len(ids))
    except ValueError as exc:
        # len(ids) <= NUMBA_NUM_THREADS holds whenever the pool default came
        # from this host's CPU count; warn instead of failing env construction.
        warnings.warn(
            f"EnvCfg.cpu_ids Numba thread cap to {len(ids)} rejected: {exc}",
            stacklevel=2,
        )
