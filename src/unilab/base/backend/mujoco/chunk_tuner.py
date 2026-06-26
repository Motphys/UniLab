"""Adaptive thread-pool ``chunk_size`` selection for the MuJoCo BatchEnvPool.

All probing/benchmarking happens on the cold path (``materialize()``), never on
``step``/``reset``. Results are cached on disk keyed by model signature + device
fingerprint + (num_envs, nthread, dtype, ...) so repeated trainings reuse them.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import logging
import math
import os
import platform
import socket
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

MAX_CANDIDATES = 16


def _emit(msg: str) -> None:
    """Surface a one-time chunk_size decision on the terminal.

    Collector envs materialize inside ``spawn`` subprocesses whose root logger is
    unconfigured (default level WARNING), so ``logger.info`` would be dropped and
    the benchmark table would never reach the terminal. When INFO is not enabled,
    fall back to a direct stderr write (the same channel the collector uses for
    its own diagnostics); otherwise log normally so the main process sees no
    duplicate line.
    """
    if logger.isEnabledFor(logging.INFO):
        logger.info(msg)
    else:
        print(f"[unilab.chunk_size] {msg}", file=sys.stderr, flush=True)


def _native_default_chunk(num_envs: int, nthread: int) -> int:
    """The chunk_size BatchEnvPool uses when ``chunk_size=None`` -- mujoco's own
    default ``max(1, nbatch // (10 * nthread))`` (see mujoco ``batch_env`` /
    ``rollout``). Shown next to ``default``/``None`` so the log is unambiguous."""
    return max(1, int(num_envs) // (10 * max(1, int(nthread))))


def _chosen_label(chosen: int | None, default_chunk: int) -> str:
    """Spell out the effective chunk_size for the native default (``None``)."""
    return f"None(={default_chunk})" if chosen is None else str(chosen)


def _format_candidate_table(per_candidate_ms: dict, default_chunk: int) -> str:
    """Render a stored ``per_candidate_ms`` mapping (keys ``"None"``/``"4"``/...,
    values in ms) as ``default(=5)=9.38ms, 1=25.43ms, ...`` -- native default first
    (annotated with its actual chunk_size), then candidates in ascending order."""

    def sort_key(k: str):
        return (0, -1) if k == "None" else (1, int(k))

    def label(k: str) -> str:
        return f"default(={default_chunk})" if k == "None" else k

    return ", ".join(
        f"{label(k)}={per_candidate_ms[k]:.2f}ms" for k in sorted(per_candidate_ms, key=sort_key)
    )


def _emit_cache_hit(value: int | None, per_candidate_ms: dict | None, default_chunk: int) -> None:
    """Report a cache hit. Include the stored candidate breakdown when present so the
    full benchmark stays visible on every run, not only the first (cold) one."""
    chosen = _chosen_label(value, default_chunk)
    if per_candidate_ms:
        _emit(
            f"chunk_size: cache hit -> chosen={chosen} | "
            f"{_format_candidate_table(per_candidate_ms, default_chunk)}"
        )
    else:
        _emit(f"chunk_size: cache hit -> {chosen}")


def make_candidates(num_envs: int, nthread: int) -> list[int]:
    """Candidate chunk_sizes within ``[1, upper]``, densified at the sweet spot.

    ``upper = ceil(num_envs / nthread)`` is a hard ceiling: a chunk_size beyond it
    produces fewer chunks than threads, leaving threads idle -> always slower. The
    optimum sits near ``heur = num_envs / (10 * nthread)`` (~10 chunks per thread),
    so the band around it is sampled densely while the always-bad large chunks are
    never generated.
    """
    if num_envs < 1:
        raise ValueError(f"num_envs must be >= 1, got {num_envs}")
    nthread = max(1, int(nthread))
    upper = max(1, math.ceil(num_envs / nthread))
    if upper == 1:
        return [1]
    heur = min(upper, max(1, round(num_envs / (10 * nthread))))
    cands: set[int] = {1, upper, heur}
    for m in (0.5, 1.5, 2.0, 3.0):  # densify around the expected optimum
        cands.add(min(upper, max(1, round(heur * m))))
    c = 1  # geometric coverage of [1, upper]
    while c < upper:
        cands.add(c)
        c = max(c + 1, round(c * 1.8))
    cands.add(upper)
    return sorted(cands)


def filter_candidates(
    candidates: list[int],
    num_envs: int,
    *,
    max_candidates: int = MAX_CANDIDATES,
) -> list[int]:
    """Clamp to ``[1, num_envs]``, dedup, sort. If over the cap, trim the coarse
    middle while keeping the low/optimum band (smallest values) plus the coarsest
    anchor -- never log-spaced (which used to drop the optimum)."""
    valid = sorted({c for c in candidates if 1 <= c <= num_envs})
    if len(valid) <= max_candidates:
        return valid
    return sorted(set(valid[: max_candidates - 1]) | {valid[-1]})


def device_fingerprint() -> dict[str, Any]:
    """Coarse, deterministic local-device descriptor for the cache key."""
    return {
        "system": platform.system(),
        "machine": platform.machine(),
        "cpu_count": int(os.cpu_count() or 1),
    }


def model_signature(model: Any, n_variants: int) -> dict[str, int]:
    """Structural determinants of per-step cost (proxy for 'task')."""
    return {
        "nq": int(model.nq),
        "nv": int(model.nv),
        "nbody": int(model.nbody),
        "njnt": int(model.njnt),
        "nu": int(model.nu),
        "ngeom": int(model.ngeom),
        "nsensordata": int(model.nsensordata),
        "n_variants": int(n_variants),
    }


def make_cache_key(
    *,
    backend_type: str,
    model_sig: dict,
    device: dict,
    num_envs: int,
    nthread: int,
    dtype: Any,
    post_step_forward_sensor: bool,
    bench_nsteps: int,
) -> str:
    payload = {
        "backend_type": backend_type,
        "model": model_sig,
        "device": device,
        "num_envs": int(num_envs),
        "nthread": int(nthread),
        "dtype": np.dtype(dtype).name,
        "post_step_forward_sensor": bool(post_step_forward_sensor),
        "bench_nsteps": int(bench_nsteps),
    }
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha1(blob.encode("utf-8")).hexdigest()


def cache_path() -> Path:
    override = os.environ.get("UNILAB_CHUNK_SIZE_CACHE")
    if override:
        return Path(override)
    xdg = os.environ.get("XDG_CACHE_HOME")
    root = Path(xdg) if xdg else Path.home() / ".cache"
    return root / "unilab" / "chunk_size.json"


def load_cache(path: Path) -> dict:
    try:
        with Path(path).open("r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return {}


def store_cache(path: Path, key: str, value: dict) -> None:
    path = Path(path)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        data = load_cache(path)
        data[key] = value
        tmp = path.with_name(f"{path.name}.tmp.{os.getpid()}")
        with tmp.open("w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, sort_keys=True)
        os.replace(tmp, path)
    except OSError as e:
        logger.warning("chunk_size cache write failed: %s", e)


@contextlib.contextmanager
def file_lock(lock_path: Path):
    """Best-effort inter-process exclusive lock; no-op where fcntl is absent."""
    try:
        import fcntl
    except ImportError:
        yield
        return
    lock_path = Path(lock_path)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    f = lock_path.open("w")
    try:
        fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        yield
    finally:
        try:
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)
        finally:
            f.close()


def _aggregate_times(times) -> float:
    """Reduce repeated samples to one estimate. Use ``min``: background load only
    inflates a sample (never deflates it below the compute-bound floor), so the
    minimum is the most stable proxy for the uncontended cost a dedicated training
    box sees -- unlike the median, which a loaded machine biases upward."""
    return float(min(times))


def _robust_step_time(
    pool, state, nstep, control, chunk_size, post_step_forward_sensor, reps
) -> float:
    times = []
    for _ in range(reps):
        s = state.copy()
        t0 = time.perf_counter()
        pool.step(
            s,
            nstep=nstep,
            control=control,
            chunk_size=chunk_size,
            return_sensor=True,
            post_step_forward_sensor=post_step_forward_sensor,
        )
        times.append(time.perf_counter() - t0)
    return _aggregate_times(times)


def benchmark_chunk_sizes(
    pool,
    state,
    nstep,
    candidates,
    *,
    control,
    post_step_forward_sensor,
    warmup=2,
    reps=20,
    time_budget_s=25.0,  # headroom for the heaviest envs (~16k) to sweep all candidates
) -> dict[int | None, float]:
    """Min wall-clock of a representative ``pool.step`` per candidate.

    Each candidate is sampled ``reps`` times and reduced via ``min`` (uncontended
    cost). ``None`` (native default) is always measured as the baseline anchor.
    """
    results: dict[int | None, float] = {}
    start = time.perf_counter()
    for cs in [None, *candidates]:
        for _ in range(warmup):
            s = state.copy()
            pool.step(
                s,
                nstep=nstep,
                control=control,
                chunk_size=cs,
                return_sensor=True,
                post_step_forward_sensor=post_step_forward_sensor,
            )
        results[cs] = _robust_step_time(
            pool, state, nstep, control, cs, post_step_forward_sensor, reps
        )
        if time.perf_counter() - start > time_budget_s:
            logger.warning("chunk_size benchmark time budget exceeded; using partial results")
            break
    return results


def select_chunk_size(
    timings: dict[int | None, float], base: int, *, margin=0.03, tie_tol=0.02
) -> int | None:
    baseline = timings.get(None)
    cand: dict[int, float] = {k: v for k, v in timings.items() if k is not None}
    if not cand:
        return None
    best_cs = min(cand, key=lambda k: cand[k])
    best_t = cand[best_cs]
    if baseline is not None and best_t > baseline * (1.0 - margin):
        return None  # not worth leaving the native default
    band: list[int] = [k for k, v in cand.items() if v <= best_t * (1.0 + tie_tol)]
    return min(band, key=lambda c: (abs(c - base), c))


def _log_benchmark_table(
    timings: dict[int | None, float], chosen: int | None, default_chunk: int
) -> None:
    def label(k: int | None) -> str:
        return f"default(={default_chunk})" if k is None else str(k)

    rows = ", ".join(
        f"{label(k)}={v * 1000:.2f}ms"
        for k, v in sorted(timings.items(), key=lambda kv: (kv[0] is not None, kv[0]))
    )
    _emit(f"chunk_size benchmark: {rows} -> chosen={_chosen_label(chosen, default_chunk)}")


def resolve_chunk_size(
    *,
    pool,
    state,
    model,
    n_variants: int,
    num_envs: int,
    nthread: int,
    dtype,
    post_step_forward_sensor: bool,
    bench_nsteps: int,
    manual_chunk_size: int | None,
    adaptive: bool,
    backend_type: str = "mujoco",
    model_file: str | None = None,
) -> int | None:
    # 1. manual override always wins (highest priority).
    if manual_chunk_size is not None:
        _emit(f"chunk_size: manual override = {int(manual_chunk_size)}")
        return int(manual_chunk_size)
    # 2. adaptive disabled -> native default.
    if not adaptive:
        return None
    # 2b. Nothing to tune: num_envs <= nthread means at most one work-chunk, so
    #     chunk_size cannot change anything. Skip benchmark/cache/log entirely -- this
    #     silences the num_envs=1 setup envs APPO/off-policy build just to read dims.
    if math.ceil(num_envs / max(1, nthread)) <= 1:
        return None
    default_chunk = _native_default_chunk(num_envs, nthread)  # what chunk_size=None uses
    # 3. cache lookup.
    device = device_fingerprint()
    model_sig = model_signature(model, n_variants)
    key = make_cache_key(
        backend_type=backend_type,
        model_sig=model_sig,
        device=device,
        num_envs=num_envs,
        nthread=nthread,
        dtype=dtype,
        post_step_forward_sensor=post_step_forward_sensor,
        bench_nsteps=bench_nsteps,
    )
    path = cache_path()
    hit = load_cache(path).get(key)
    if hit is not None:
        value = hit.get("chunk_size")
        _emit_cache_hit(value, hit.get("per_candidate_ms"), default_chunk)
        return value if value is None else int(value)
    # 4. miss -> lock -> re-check -> benchmark -> store.
    with file_lock(path.with_name(path.name + ".lock")):
        hit = load_cache(path).get(key)
        if hit is not None:
            value = hit.get("chunk_size")
            _emit_cache_hit(value, hit.get("per_candidate_ms"), default_chunk)
            return value if value is None else int(value)
        base = math.ceil(num_envs / max(1, nthread))  # upper anchor for the select tie-break
        candidates = filter_candidates(make_candidates(num_envs, nthread), num_envs)
        try:
            control = np.zeros((num_envs, bench_nsteps, int(model.nu)), dtype=np.float64)
            timings = benchmark_chunk_sizes(
                pool,
                state,
                bench_nsteps,
                candidates,
                control=control,
                post_step_forward_sensor=post_step_forward_sensor,
            )
        except Exception as e:  # benchmark must never crash training
            logger.warning("chunk_size benchmark failed (%s); using native default", e)
            return None
        chosen = select_chunk_size(timings, base)
        _log_benchmark_table(timings, chosen, default_chunk)
        try:
            store_cache(
                path,
                key,
                {
                    "chunk_size": chosen,
                    "per_candidate_ms": {str(k): v * 1000 for k, v in timings.items()},
                    "model_file": model_file,
                    "hostname": socket.gethostname(),
                },
            )
        except Exception as e:
            logger.warning(
                "chunk_size cache store failed (%s); continuing with chosen=%s", e, chosen
            )
        return chosen
