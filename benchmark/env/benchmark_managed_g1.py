"""Fail-closed paired host benchmark for Issue #705's managed G1 executors.

The benchmark deliberately uses a fresh ``uv run`` process for every
``(batch, repeat, mode)`` sample and rotates the execution order in each
three-way pair.  It measures the full public ``step`` call, rather than a
reward/observation micro-kernel, so manager lifecycle and typed state
materialization remain inside the candidate cost.

Typical use after this benchmark implementation has been committed and the
worktree is clean::

    uv run benchmark/env/benchmark_managed_g1.py --execute \
      --out /tmp/issue705-g1-host-fused.json
    uv run benchmark/env/benchmark_managed_g1.py \
      --validate-artifact /tmp/issue705-g1-host-fused.json

``--execute`` is intentionally rejected for a dirty source tree.  A later
evidence PR must capture from a clean committed implementation; it must not
measure uncommitted code and subsequently claim provenance for a different
commit.
"""

from __future__ import annotations

import argparse
import dataclasses
import functools
import hashlib
import json
import math
import os
import subprocess
import sys
import tempfile
import time
import uuid
from pathlib import Path
from statistics import median
from typing import Any, Mapping, Sequence, cast

import numpy as np
from omegaconf import OmegaConf

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from benchmark.core.mem_profile import build_memory_summary, memory_snapshot
from benchmark.env.benchmark_env_step import _g1_flat_cfg, _g1_walk_env_cls
from benchmark.issue705.benchmark_g1_phase0 import (
    _hardware_payload,
    _json_safe,
    _preflight_payload,
    _run_subprocess,
)
from unilab.base.backend import SimBackend, create_backend, env_backend_kwargs
from unilab.envs.locomotion.g1.managed_fused import (
    G1_MANAGED_FUSED_EXECUTOR_KEY,
    create_g1_managed_fused_runtime,
)
from unilab.envs.locomotion.g1.managed_reference import (
    G1_MANAGED_REFERENCE_EXECUTOR_KEY,
    create_g1_managed_reference_runtime,
)
from unilab.manager import ManagedReferenceRuntime
from unilab.tools.g1_baseline_provenance import (
    G1BaselinePlan,
    assert_clean_affinity,
    canonical_sha256,
    load_g1_baseline_plan,
    numeric_stats,
    sha256_file,
    source_tree_sha256,
    source_tree_sha256_at_commit,
)

ISSUE = 705
SCHEMA_VERSION = 1
PROFILE = "host_numpy"
MODES = ("hand_written", "managed_reference", "managed_fused")
EXPECTED_EXECUTOR_KEYS = {
    "hand_written": "legacy.g1-walk-flat.hand-written.v1",
    "managed_reference": G1_MANAGED_REFERENCE_EXECUTOR_KEY,
    "managed_fused": G1_MANAGED_FUSED_EXECUTOR_KEY,
}
DEFAULT_BASELINE_PLAN = Path("tests/acceptance/issue_705/g1_mujoco_baseline_plan.yaml")
DEFAULT_THRESHOLD_MANIFEST = Path("tests/acceptance/issue_705/g1_threshold_manifest.yaml")
DEFAULT_THRESHOLD_RECEIPT = Path("tests/acceptance/issue_705/g1_threshold_freeze_receipt.yaml")
DEFAULT_OUTPUT = Path("/tmp/unilab_issue705_g1_host_fused.json")

# Keep this list deliberately narrow: it contains every input that can change
# this benchmark's measured managed host lifecycle, but not the output
# artifact itself.  This lets a later evidence PR validate a clean source
# commit that predates the committed JSON artifact.
SOURCE_INPUTS = (
    "benchmark/env/benchmark_managed_g1.py",
    "benchmark/env/benchmark_env_step.py",
    "benchmark/issue705/benchmark_g1_phase0.py",
    "benchmark/core/mem_profile.py",
    "src/unilab/base/backend/base.py",
    "src/unilab/base/backend/mujoco",
    "src/unilab/base/backend/batch.py",
    "src/unilab/base/np_env.py",
    "src/unilab/dr",
    "src/unilab/manager",
    "src/unilab/dtype_config.py",
    "src/unilab/envs/locomotion/common",
    "src/unilab/envs/locomotion/g1",
    "src/unilab/tools/g1_baseline_provenance.py",
    "conf/ppo/task/g1_walk_flat/mujoco.yaml",
    "tests/acceptance/issue_705/g1_mujoco_baseline_plan.yaml",
    "tests/acceptance/issue_705/g1_threshold_manifest.yaml",
    "tests/acceptance/issue_705/g1_threshold_freeze_receipt.yaml",
    "tests/benchmark/test_managed_g1_host_benchmark.py",
    "uv.lock",
)


class HostBenchmarkError(RuntimeError):
    """Raised when a benchmark input, worker, artifact, or gate is invalid."""


@dataclasses.dataclass(frozen=True)
class ThresholdBinding:
    """The small, immutable Phase-4 subset of the Phase-0 threshold freeze."""

    threshold_set_id: str
    manifest_path: Path
    manifest_sha256: str
    freeze_commit: str
    batch_sizes: tuple[int, ...]
    process_repeats: int
    warmup_steps: int
    measure_steps: int
    p50_latency_ratio_max: float
    p95_latency_ratio_max: float
    host_preferred_memory_ratio_max: float
    max_population_cv_by_batch: tuple[tuple[int, float], ...]

    @property
    def cv_by_batch(self) -> dict[int, float]:
        return dict(self.max_population_cv_by_batch)


def _load_yaml(path: Path) -> dict[str, Any]:
    try:
        payload = OmegaConf.to_container(OmegaConf.load(path), resolve=False)
    except Exception as exc:  # noqa: BLE001 - normalize external YAML errors.
        raise HostBenchmarkError(f"cannot load {path}: {type(exc).__name__}: {exc}") from exc
    if not isinstance(payload, dict):
        raise HostBenchmarkError(f"{path} must contain a mapping")
    return cast(dict[str, Any], payload)


def _require_mapping(value: object, *, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise HostBenchmarkError(f"{label} must be a mapping")
    return cast(dict[str, Any], value)


def _require_string(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise HostBenchmarkError(f"{label} must be a non-empty string")
    return value


def _require_positive_int(value: object, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise HostBenchmarkError(f"{label} must be a positive integer")
    return int(value)


def _require_positive_float(value: object, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise HostBenchmarkError(f"{label} must be a positive finite number")
    result = float(value)
    if not np.isfinite(result) or result <= 0.0:
        raise HostBenchmarkError(f"{label} must be a positive finite number")
    return result


def _git(*args: str) -> str:
    completed = subprocess.run(
        ["git", *args],
        cwd=ROOT_DIR,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _is_commit(value: object) -> bool:
    if not isinstance(value, str) or len(value) != 40:
        return False
    return all(character in "0123456789abcdef" for character in value)


def load_threshold_binding(
    *,
    threshold_manifest: Path = DEFAULT_THRESHOLD_MANIFEST,
    threshold_receipt: Path = DEFAULT_THRESHOLD_RECEIPT,
) -> ThresholdBinding:
    """Load the frozen performance threshold without accepting local edits."""

    manifest_path = (
        threshold_manifest if threshold_manifest.is_absolute() else (ROOT_DIR / threshold_manifest)
    )
    receipt_path = (
        threshold_receipt if threshold_receipt.is_absolute() else (ROOT_DIR / threshold_receipt)
    )
    manifest = _load_yaml(manifest_path)
    receipt = _load_yaml(receipt_path)
    if manifest.get("schema_version") != 1 or manifest.get("issue") != ISSUE:
        raise HostBenchmarkError("threshold manifest does not belong to Issue #705 schema v1")
    if manifest.get("state") != "frozen":
        raise HostBenchmarkError("host benchmark requires a frozen threshold manifest")
    threshold_set_id = _require_string(manifest.get("threshold_set_id"), label="threshold_set_id")
    measurement = _require_mapping(manifest.get("measurement"), label="measurement")
    gates = _require_mapping(manifest.get("gates"), label="gates")
    performance = _require_mapping(gates.get("performance"), label="gates.performance")
    memory = _require_mapping(gates.get("memory"), label="gates.memory")
    batch_values = measurement.get("batch_sizes")
    if not isinstance(batch_values, list) or not batch_values:
        raise HostBenchmarkError("measurement.batch_sizes must be a non-empty list")
    batch_sizes = tuple(
        _require_positive_int(item, label=f"measurement.batch_sizes[{index}]")
        for index, item in enumerate(batch_values)
    )
    if batch_sizes != (128, 1024, 4096):
        raise HostBenchmarkError("host benchmark batch matrix must remain [128, 1024, 4096]")
    process_repeats = _require_positive_int(
        measurement.get("env_process_repeats"), label="measurement.env_process_repeats"
    )
    if process_repeats < 5:
        raise HostBenchmarkError("host benchmark requires at least five isolated process repeats")
    cv_raw = _require_mapping(
        performance.get("max_population_cv_by_batch"),
        label="gates.performance.max_population_cv_by_batch",
    )
    cv_by_batch: list[tuple[int, float]] = []
    for batch_size in batch_sizes:
        value = cv_raw.get(str(batch_size))
        cv_by_batch.append(
            (
                batch_size,
                _require_positive_float(value, label=f"max_population_cv_by_batch.{batch_size}"),
            )
        )
    expected_manifest = receipt.get("manifest_path")
    if expected_manifest != threshold_manifest.as_posix():
        raise HostBenchmarkError("threshold receipt manifest path differs from benchmark manifest")
    manifest_sha256 = sha256_file(manifest_path)
    if receipt.get("manifest_sha256") != manifest_sha256:
        raise HostBenchmarkError("threshold receipt SHA does not match threshold manifest")
    freeze_commit = receipt.get("freeze_commit")
    if not _is_commit(freeze_commit):
        raise HostBenchmarkError("threshold receipt freeze_commit must be a full Git SHA")
    if receipt.get("threshold_set_id") != threshold_set_id:
        raise HostBenchmarkError("threshold receipt threshold_set_id does not match manifest")
    return ThresholdBinding(
        threshold_set_id=threshold_set_id,
        manifest_path=threshold_manifest,
        manifest_sha256=manifest_sha256,
        freeze_commit=cast(str, freeze_commit),
        batch_sizes=batch_sizes,
        process_repeats=process_repeats,
        # These are the Phase-0 frozen env-lane values.  The benchmark plan is
        # the canonical source rather than an editable CLI default.
        warmup_steps=10,
        measure_steps=50,
        p50_latency_ratio_max=_require_positive_float(
            performance.get("p50_latency_ratio_max"), label="p50_latency_ratio_max"
        ),
        p95_latency_ratio_max=_require_positive_float(
            performance.get("p95_latency_ratio_max"), label="p95_latency_ratio_max"
        ),
        host_preferred_memory_ratio_max=_require_positive_float(
            memory.get("host_preferred_metric_ratio_max"),
            label="host_preferred_metric_ratio_max",
        ),
        max_population_cv_by_batch=tuple(cv_by_batch),
    )


def _load_plan(path: Path) -> G1BaselinePlan:
    resolved = path if path.is_absolute() else ROOT_DIR / path
    plan = load_g1_baseline_plan(resolved)
    relative = resolved.relative_to(ROOT_DIR)
    return dataclasses.replace(plan, source_path=relative)


def _validate_plan_binding(plan: G1BaselinePlan, binding: ThresholdBinding) -> None:
    """Refuse a changed Phase-0 warmup/sample plan behind an unchanged receipt."""

    if plan.env_lane.batch_sizes != binding.batch_sizes:
        raise HostBenchmarkError("baseline plan batch matrix differs from frozen threshold matrix")
    if plan.env_lane.process_repeats != binding.process_repeats:
        raise HostBenchmarkError(
            "baseline plan process repeats differ from frozen threshold matrix"
        )
    if plan.env_lane.warmup_steps != binding.warmup_steps:
        raise HostBenchmarkError("baseline plan warmup steps differ from frozen host profile")
    if plan.env_lane.measure_steps != binding.measure_steps:
        raise HostBenchmarkError("baseline plan measure steps differ from frozen host profile")


def _host_cfg() -> Any:
    """Compose the owner YAML, then freeze the currently supported manager profile."""

    cfg = _g1_flat_cfg("mujoco")
    # These are not silent behavior changes: the manager pilot does not yet
    # implement typed DR/Event, and this explicit compatibility profile is
    # included in the artifact's resolved-config hash.
    cfg.adaptive_chunk_size = False
    cfg.chunk_size = None
    cfg.domain_rand.randomize_kp = False
    cfg.domain_rand.randomize_kd = False
    cfg.noise_config.level = 0.0
    cfg.noise_config.seed = None
    cfg.max_episode_seconds = None
    cfg.curriculum.enabled = False
    cfg.commands.resampling_time = 0.0
    cfg.commands.heading_command = False
    cfg.numba_acceleration = False
    cfg.validate()
    return cfg


def _backend(cfg: Any, *, num_envs: int) -> SimBackend:
    if cfg.scene is None:
        raise HostBenchmarkError("G1 host benchmark requires a materialized scene")
    return create_backend(
        "mujoco",
        cfg.scene,
        num_envs,
        cfg.sim_dt,
        base_name=cfg.asset.base_name,
        push_body_name=cfg.domain_rand.push_body_name,
        **env_backend_kwargs(cfg),
    )


def _actions(*, batch_size: int, count: int, seed: int) -> np.ndarray:
    if count <= 0:
        raise HostBenchmarkError("action count must be positive")
    generator = np.random.default_rng(seed)
    values = generator.uniform(-1.0, 1.0, size=(count, batch_size, 29))
    return np.ascontiguousarray(values, dtype=np.float32)


def _array_sha256(values: np.ndarray) -> str:
    return f"sha256:{hashlib.sha256(values.tobytes(order='C')).hexdigest()}"


def _action_seed(*, plan: G1BaselinePlan, batch_size: int, repeat_index: int) -> int:
    return plan.env_lane.action_seed_base + batch_size * 100 + repeat_index


@functools.lru_cache(maxsize=None)
def _expected_action_sha256(*, batch_size: int, sample_count: int, seed: int) -> str:
    """Hash the frozen deterministic worker action stream.

    Keeping this verifier-side generator independent from a worker receipt
    closes the otherwise easy loophole where all three paired processes use
    the *same*, but hand-picked, action schedule.
    """

    return _array_sha256(_actions(batch_size=batch_size, count=sample_count, seed=seed))


def _record_backend_timings(
    runtime: ManagedReferenceRuntime,
    records: dict[str, list[float]],
) -> None:
    diagnostics = runtime.last_step_diagnostics
    if diagnostics is None:
        raise HostBenchmarkError("managed benchmark step produced no backend diagnostics")
    for timing in diagnostics.timings:
        records.setdefault(f"backend_{timing.phase}_ms", []).append(float(timing.milliseconds))


def _run_worker(
    *,
    plan: G1BaselinePlan,
    binding: ThresholdBinding,
    mode: str,
    batch_size: int,
    repeat_index: int,
) -> dict[str, Any]:
    if mode not in MODES:
        raise HostBenchmarkError(f"unknown host benchmark mode {mode!r}")
    if batch_size not in binding.batch_sizes:
        raise HostBenchmarkError("worker batch_size is not in frozen matrix")
    if repeat_index not in range(binding.process_repeats):
        raise HostBenchmarkError("worker repeat_index is outside frozen matrix")
    assert_clean_affinity(plan)
    os.sched_setaffinity(0, set(plan.hardware.affinity_cpus))
    for key, value in plan.environment.env_vars:
        os.environ[key] = value

    cfg = _host_cfg()
    config = cast(dict[str, Any], _json_safe(cfg))
    action_seed = _action_seed(plan=plan, batch_size=batch_size, repeat_index=repeat_index)
    # The legacy compatibility path still samples its reset payload through
    # NumPy's process-global stream.  Each worker is isolated, so setting it
    # here gives all three modes the same frozen initial/reset schedule
    # without introducing a mutable global into the measured hot path.
    np.random.seed(action_seed)
    actions = _actions(
        batch_size=batch_size,
        count=binding.warmup_steps + binding.measure_steps,
        seed=action_seed,
    )
    # A legacy environment may retain and later reset a view of a past action
    # row.  Receipt the immutable schedule *before* stepping so the benchmark
    # compares the inputs delivered to every mode, rather than any permitted
    # post-step mutation of an old caller-owned view.
    action_sha256 = _array_sha256(actions)
    timing_records: dict[str, list[float]] = {"env_step_total_ms": []}
    memory_samples: dict[str, dict[str, Any]] = {}
    runner: Any = None
    backend: SimBackend | None = None
    plan_fingerprint = "legacy.g1-walk-flat.hand-written.v1"
    executor_key = "legacy.g1-walk-flat.hand-written.v1"
    try:
        memory_samples["before_env"] = memory_snapshot("before_env")
        if mode == "hand_written":
            runner = _g1_walk_env_cls()(cfg, num_envs=batch_size, backend_type="mujoco")
        else:
            backend = _backend(cfg, num_envs=batch_size)
            factory = (
                create_g1_managed_reference_runtime
                if mode == "managed_reference"
                else create_g1_managed_fused_runtime
            )
            runner = factory(backend=backend, cfg=cfg, reset_seed=action_seed)
            plan_fingerprint = runner.plan.fingerprint
            executor_key = runner.plan.executor_key
        runner.init_state()
        for action in actions[: binding.warmup_steps]:
            runner.step(action)
        for action in actions[binding.warmup_steps :]:
            started = time.perf_counter_ns()
            state = runner.step(action)
            elapsed_ms = (time.perf_counter_ns() - started) / 1_000_000.0
            timing_records["env_step_total_ms"].append(elapsed_ms)
            if mode == "hand_written":
                timing = state.info.get("timing", {})
                if not isinstance(timing, dict):
                    raise HostBenchmarkError("hand-written G1 step produced no timing mapping")
                for key, value in timing.items():
                    timing_records.setdefault(f"legacy_{key}", []).append(float(value))
            else:
                _record_backend_timings(cast(ManagedReferenceRuntime, runner), timing_records)
        if mode != "hand_written":
            backend_materialize = timing_records.get("backend_state_materialize_ms", [])
            if len(backend_materialize) != binding.measure_steps:
                raise HostBenchmarkError(
                    "managed worker is missing typed state materialization timings"
                )
            residual: list[float] = []
            for total, materialize in zip(
                timing_records["env_step_total_ms"], backend_materialize, strict=True
            ):
                # The timing is nested in the public step interval.  Negative
                # residual would mean the benchmark's component accounting is
                # internally inconsistent, so fail rather than clamp it.
                value = total - materialize
                if value < 0.0:
                    raise HostBenchmarkError(
                        "backend component exceeds full managed step wall time"
                    )
                residual.append(value)
            timing_records["runtime_non_materialize_ms"] = residual
        memory_samples["after_benchmark"] = memory_snapshot("after_benchmark")
    finally:
        if mode == "hand_written" and runner is not None:
            runner.close()
            memory_samples["after_close"] = memory_snapshot("after_close")
        elif backend is not None:
            backend.cleanup_scene_assets()
            memory_samples["after_close"] = memory_snapshot("after_close")
    raw = {
        "timing_records": timing_records,
        "memory": build_memory_summary(memory_samples, batch_size),
        "resolved_env_config": config,
        "resolved_config_sha256": canonical_sha256(config),
        "action_seed": action_seed,
        "reset_seed": action_seed,
        "action_sha256": action_sha256,
        "executor_key": executor_key,
        "plan_fingerprint": plan_fingerprint,
        "backend_identity": "mujoco",
    }
    return cast(dict[str, Any], _json_safe(raw))


def summarize_worker_raw(raw: Mapping[str, Any], *, batch_size: int) -> dict[str, Any]:
    timing_records = raw.get("timing_records")
    if not isinstance(timing_records, dict):
        raise HostBenchmarkError("worker raw timing_records must be a mapping")
    total = timing_records.get("env_step_total_ms")
    if not isinstance(total, list) or not total:
        raise HostBenchmarkError("worker raw must include env_step_total_ms samples")
    timings: dict[str, dict[str, float | int]] = {}
    for name, values in sorted(timing_records.items()):
        if not isinstance(values, list):
            raise HostBenchmarkError(f"worker timing {name!r} must be a list")
        timings[str(name)] = numeric_stats([float(value) for value in values])
    total_ms = [float(value) for value in total]
    memory = raw.get("memory")
    if not isinstance(memory, dict):
        raise HostBenchmarkError("worker raw memory must be a mapping")
    preferred_metric = memory.get("preferred_metric")
    if preferred_metric not in {"rss", "uss", "pss"}:
        raise HostBenchmarkError("worker memory preferred_metric must be rss, uss, or pss")
    total_preferred_delta = _require_positive_float(
        memory.get(f"total_{preferred_metric}_delta_bytes"),
        label="worker total preferred memory delta",
    )
    after_preferred = _require_positive_float(
        memory.get(f"after_benchmark_{preferred_metric}_bytes"),
        label="worker after benchmark preferred memory",
    )
    return {
        "timing_stats_ms": timings,
        "throughput_env_steps_per_sec": float(
            batch_size * len(total_ms) / (sum(total_ms) / 1000.0)
        ),
        "memory": {
            "preferred_metric": preferred_metric,
            "total_preferred_delta_bytes": int(total_preferred_delta),
            "after_benchmark_preferred_bytes": int(after_preferred),
        },
    }


def _summary_matches_recomputed(recorded: object, expected: Mapping[str, Any]) -> bool:
    if not isinstance(recorded, dict) or set(recorded) != set(expected):
        return False
    throughput = recorded.get("throughput_env_steps_per_sec")
    if isinstance(throughput, bool) or not isinstance(throughput, (int, float)):
        return False
    return (
        recorded.get("timing_stats_ms") == expected.get("timing_stats_ms")
        and recorded.get("memory") == expected.get("memory")
        # Python 3.11 and 3.13 can differ by a few ULPs when summing the same samples.
        and math.isclose(
            float(throughput),
            float(expected["throughput_env_steps_per_sec"]),
            rel_tol=1e-15,
            abs_tol=0.0,
        )
    )


def _case_id(*, batch_size: int, repeat_index: int, mode: str) -> str:
    return f"host-b{batch_size}-r{repeat_index}-{mode}"


def _mode_order(repeat_index: int) -> tuple[str, ...]:
    rotation = repeat_index % len(MODES)
    return (*MODES[rotation:], *MODES[:rotation])


def expected_case_ids(binding: ThresholdBinding) -> set[str]:
    return {
        _case_id(batch_size=batch_size, repeat_index=repeat_index, mode=mode)
        for batch_size in binding.batch_sizes
        for repeat_index in range(binding.process_repeats)
        for mode in MODES
    }


def _run_case_subprocess(
    *,
    plan: G1BaselinePlan,
    binding: ThresholdBinding,
    mode: str,
    batch_size: int,
    repeat_index: int,
    sequence_index: int,
) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="unilab_issue705_host_worker_") as directory:
        output = Path(directory) / "raw.json"
        command = [
            "uv",
            "run",
            "benchmark/env/benchmark_managed_g1.py",
            "--baseline-plan",
            plan.source_path.as_posix(),
            "--threshold-manifest",
            binding.manifest_path.as_posix(),
            "--worker",
            "--mode",
            mode,
            "--batch-size",
            str(batch_size),
            "--repeat-index",
            str(repeat_index),
            "--worker-output",
            str(output),
        ]
        process, _, stdout, stderr = _run_subprocess(command, plan)
        if process["return_code"] != 0 or not output.is_file():
            raise HostBenchmarkError(
                "host benchmark worker failed: "
                f"command={command!r}\nstdout:\n{stdout[-4000:]}\nstderr:\n{stderr[-4000:]}"
            )
        raw = json.loads(output.read_text(encoding="utf-8"))
    return {
        "case_id": _case_id(batch_size=batch_size, repeat_index=repeat_index, mode=mode),
        "mode": mode,
        "batch_size": batch_size,
        "repeat_index": repeat_index,
        "sequence_index": sequence_index,
        "process": process,
        "raw": raw,
        "summary": summarize_worker_raw(raw, batch_size=batch_size),
    }


def _population_cv(values: Sequence[float]) -> float:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or array.size < 2 or not np.all(np.isfinite(array)):
        raise HostBenchmarkError("population CV requires at least two finite process summaries")
    mean = float(array.mean())
    if mean <= 0.0:
        raise HostBenchmarkError("population CV requires a positive mean")
    return float(array.std(ddof=0) / mean)


def build_aggregates(
    cases: Sequence[Mapping[str, Any]], binding: ThresholdBinding
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for batch_size in binding.batch_sizes:
        by_mode: dict[str, Any] = {}
        for mode in MODES:
            selected = [
                case
                for case in cases
                if case.get("batch_size") == batch_size and case.get("mode") == mode
            ]
            if len(selected) != binding.process_repeats:
                raise HostBenchmarkError("aggregate requires every mode/repeat process sample")
            summaries = [
                _require_mapping(case.get("summary"), label="case.summary")
                for case in sorted(selected, key=lambda item: int(item["repeat_index"]))
            ]
            p50 = [
                float(
                    _require_mapping(summary["timing_stats_ms"], label="timings")[
                        "env_step_total_ms"
                    ]["p50"]
                )
                for summary in summaries
            ]
            p95 = [
                float(
                    _require_mapping(summary["timing_stats_ms"], label="timings")[
                        "env_step_total_ms"
                    ]["p95"]
                )
                for summary in summaries
            ]
            throughput = [float(summary["throughput_env_steps_per_sec"]) for summary in summaries]
            memories = [
                _require_mapping(summary["memory"], label="memory") for summary in summaries
            ]
            preferred_metrics = {memory.get("preferred_metric") for memory in memories}
            if len(preferred_metrics) != 1 or next(iter(preferred_metrics)) not in {
                "rss",
                "uss",
                "pss",
            }:
                raise HostBenchmarkError("aggregate requires one valid preferred memory metric")
            preferred_memory_deltas = [
                _require_positive_float(
                    memory.get("total_preferred_delta_bytes"),
                    label="total_preferred_delta_bytes",
                )
                for memory in memories
            ]
            by_mode[mode] = {
                "process_count": len(summaries),
                "env_step_total_ms": {
                    "p50_median_of_process_summaries": float(median(p50)),
                    "p95_median_of_process_summaries": float(median(p95)),
                    "p50_population_cv": _population_cv(p50),
                },
                "throughput_median_env_steps_per_sec": float(median(throughput)),
                "memory": {
                    "preferred_metric": next(iter(preferred_metrics)),
                    "total_preferred_delta_median_bytes": float(median(preferred_memory_deltas)),
                },
            }
        result[str(batch_size)] = by_mode
    return result


def build_gate(aggregates: Mapping[str, Any], binding: ThresholdBinding) -> dict[str, Any]:
    """Evaluate the frozen p50/p95/CV gate from recomputed case summaries."""

    by_batch: dict[str, Any] = {}
    passed = True
    for batch_size in binding.batch_sizes:
        batch = _require_mapping(aggregates.get(str(batch_size)), label=f"aggregates.{batch_size}")
        baseline = _require_mapping(batch.get("hand_written"), label="hand_written aggregate")
        fused = _require_mapping(batch.get("managed_fused"), label="managed_fused aggregate")
        reference = _require_mapping(
            batch.get("managed_reference"), label="managed_reference aggregate"
        )
        baseline_timing = _require_mapping(
            baseline.get("env_step_total_ms"), label="baseline timing"
        )
        fused_timing = _require_mapping(fused.get("env_step_total_ms"), label="fused timing")
        reference_timing = _require_mapping(
            reference.get("env_step_total_ms"), label="reference timing"
        )
        baseline_memory = _require_mapping(baseline.get("memory"), label="baseline memory")
        fused_memory = _require_mapping(fused.get("memory"), label="fused memory")
        baseline_p50 = _require_positive_float(
            baseline_timing.get("p50_median_of_process_summaries"), label="baseline p50"
        )
        baseline_p95 = _require_positive_float(
            baseline_timing.get("p95_median_of_process_summaries"), label="baseline p95"
        )
        p50_ratio = (
            _require_positive_float(
                fused_timing.get("p50_median_of_process_summaries"), label="fused p50"
            )
            / baseline_p50
        )
        p95_ratio = (
            _require_positive_float(
                fused_timing.get("p95_median_of_process_summaries"), label="fused p95"
            )
            / baseline_p95
        )
        baseline_memory_metric = baseline_memory.get("preferred_metric")
        fused_memory_metric = fused_memory.get("preferred_metric")
        if baseline_memory_metric != fused_memory_metric or baseline_memory_metric not in {
            "rss",
            "uss",
            "pss",
        }:
            raise HostBenchmarkError(
                "host memory gate requires the same valid metric in both modes"
            )
        memory_ratio = _require_positive_float(
            fused_memory.get("total_preferred_delta_median_bytes"), label="fused memory"
        ) / _require_positive_float(
            baseline_memory.get("total_preferred_delta_median_bytes"),
            label="baseline memory",
        )
        cv_limit = binding.cv_by_batch[batch_size]
        # ``0.0`` is the ideal CV, so it needs a finite non-negative rule
        # instead of the positive-latency helper used above.
        cvs = {}
        for mode in MODES:
            value = _require_mapping(batch[mode], label=f"{mode} aggregate")["env_step_total_ms"][
                "p50_population_cv"
            ]
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise HostBenchmarkError(f"{mode} p50 population CV must be numeric")
            numeric = float(value)
            if not np.isfinite(numeric) or numeric < 0.0:
                raise HostBenchmarkError(
                    f"{mode} p50 population CV must be finite and non-negative"
                )
            cvs[mode] = numeric
        row_passed = (
            p50_ratio <= binding.p50_latency_ratio_max
            and p95_ratio <= binding.p95_latency_ratio_max
            and memory_ratio <= binding.host_preferred_memory_ratio_max
            and all(value <= cv_limit for value in cvs.values())
        )
        passed = passed and row_passed
        by_batch[str(batch_size)] = {
            "hand_written_p50_ms": baseline_p50,
            "hand_written_p95_ms": baseline_p95,
            "managed_reference_p50_ms": float(reference_timing["p50_median_of_process_summaries"]),
            "managed_reference_p95_ms": float(reference_timing["p95_median_of_process_summaries"]),
            "managed_fused_p50_ms": float(fused_timing["p50_median_of_process_summaries"]),
            "managed_fused_p95_ms": float(fused_timing["p95_median_of_process_summaries"]),
            "fused_to_hand_written_p50_ratio": p50_ratio,
            "fused_to_hand_written_p95_ratio": p95_ratio,
            "p50_latency_ratio_max": binding.p50_latency_ratio_max,
            "p95_latency_ratio_max": binding.p95_latency_ratio_max,
            "preferred_memory_metric": baseline_memory_metric,
            "fused_to_hand_written_preferred_memory_ratio": memory_ratio,
            "preferred_memory_ratio_max": binding.host_preferred_memory_ratio_max,
            "population_cv_limit": cv_limit,
            "population_cv_by_mode": cvs,
            "passed": row_passed,
        }
    return {"by_batch": by_batch, "passed": passed}


def _source_payload(plan: G1BaselinePlan, binding: ThresholdBinding) -> dict[str, Any]:
    dirty = bool(_git("status", "--short"))
    if dirty:
        raise HostBenchmarkError("host benchmark execution requires a clean git worktree")
    commit = _git("rev-parse", "HEAD")
    if commit == binding.freeze_commit:
        raise HostBenchmarkError("candidate benchmark cannot run at the threshold-freeze commit")
    return {
        "candidate_commit": commit,
        "branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
        "source_dirty": False,
        "source_tree_sha256": source_tree_sha256(ROOT_DIR, SOURCE_INPUTS),
        "uv_lock_sha256": sha256_file(ROOT_DIR / "uv.lock"),
        "owner_yaml_sha256": sha256_file(ROOT_DIR / plan.owner_yaml),
    }


def _assert_capture_source_unchanged(source: Mapping[str, Any]) -> None:
    """Fail if a long benchmark modified or moved its candidate source."""

    if _git("status", "--short"):
        raise HostBenchmarkError("host benchmark changed the candidate source during capture")
    if _git("rev-parse", "HEAD") != source.get("candidate_commit"):
        raise HostBenchmarkError("candidate HEAD changed during host benchmark capture")
    if source_tree_sha256(ROOT_DIR, SOURCE_INPUTS) != source.get("source_tree_sha256"):
        raise HostBenchmarkError("benchmark source inputs changed during host benchmark capture")
    if sha256_file(ROOT_DIR / "uv.lock") != source.get("uv_lock_sha256"):
        raise HostBenchmarkError("uv.lock changed during host benchmark capture")


def _collect(
    plan: G1BaselinePlan,
    binding: ThresholdBinding,
    *,
    allow_gate_failure: bool = False,
) -> dict[str, Any]:
    source = _source_payload(plan, binding)
    hardware = _hardware_payload(plan)
    preflight_before = _preflight_payload(plan)
    cases: list[dict[str, Any]] = []
    total = len(expected_case_ids(binding))
    for batch_size in binding.batch_sizes:
        for repeat_index in range(binding.process_repeats):
            for sequence_index, mode in enumerate(_mode_order(repeat_index)):
                case = _run_case_subprocess(
                    plan=plan,
                    binding=binding,
                    mode=mode,
                    batch_size=batch_size,
                    repeat_index=repeat_index,
                    sequence_index=sequence_index,
                )
                cases.append(case)
                print(f"[{len(cases):02d}/{total:02d}] {case['case_id']} PASS", flush=True)
    _assert_capture_source_unchanged(source)
    # Linux's one-minute load average includes the just-finished CPU workers.
    # Keep it in the receipt for diagnosis, but do not mistake the benchmark's
    # own work for a foreign-load preflight failure.  GPU-idleness checks stay
    # fail-closed in both samples.
    preflight_after = _preflight_payload(plan, enforce_cpu_load=False)
    aggregates = build_aggregates(cases, binding)
    artifact = {
        "schema_version": SCHEMA_VERSION,
        "issue": ISSUE,
        "profile": PROFILE,
        "threshold": {
            "threshold_set_id": binding.threshold_set_id,
            "manifest_path": binding.manifest_path.as_posix(),
            "manifest_sha256": binding.manifest_sha256,
            "freeze_commit": binding.freeze_commit,
        },
        "candidate": source,
        "hardware": hardware,
        "execution": {
            "process_isolation": True,
            "affinity_cpus": list(plan.hardware.affinity_cpus),
            "env_vars": dict(plan.environment.env_vars),
            "warmup_steps": binding.warmup_steps,
            "measure_steps": binding.measure_steps,
            "mode_order_policy": "repeat-index cyclic rotation of hand_written, managed_reference, managed_fused",
            "preflight_before": preflight_before,
            "preflight_after": preflight_after,
        },
        "cases": cases,
        "aggregates": aggregates,
        "gate": build_gate(aggregates, binding),
    }
    errors = validate_artifact(
        artifact,
        binding=binding,
        plan=plan,
        repo_root=ROOT_DIR,
        require_passing_gate=not allow_gate_failure,
    )
    if errors:
        detail = "\n".join(f"- {error}" for error in errors)
        raise HostBenchmarkError(f"generated host benchmark artifact failed validation:\n{detail}")
    if artifact["gate"]["passed"] is not True and not allow_gate_failure:
        raise HostBenchmarkError("host fused performance gate did not pass")
    return artifact


def _exact_keys(
    value: object, expected: set[str], *, label: str, errors: list[str]
) -> dict[str, Any]:
    if not isinstance(value, dict):
        errors.append(f"{label}: expected mapping")
        return {}
    actual = set(value)
    missing = sorted(expected - actual)
    unknown = sorted(actual - expected)
    if missing:
        errors.append(f"{label}: missing keys {missing!r}")
    if unknown:
        errors.append(f"{label}: unknown keys {unknown!r}")
    return cast(dict[str, Any], value)


def _finite_samples(
    value: object,
    *,
    count: int,
    label: str,
    errors: list[str],
    lower_bound: float | None = None,
    strict_lower_bound: bool = False,
) -> list[float]:
    if not isinstance(value, list) or len(value) != count:
        errors.append(f"{label}: expected exactly {count} samples")
        return []
    result: list[float] = []
    for index, item in enumerate(value):
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            errors.append(f"{label}[{index}]: expected numeric value")
            continue
        numeric = float(item)
        below_bound = lower_bound is not None and (
            numeric <= lower_bound if strict_lower_bound else numeric < lower_bound
        )
        if not np.isfinite(numeric) or below_bound:
            expectation = "finite value"
            if lower_bound == 0.0 and strict_lower_bound:
                expectation = "positive finite value"
            elif lower_bound == 0.0:
                expectation = "non-negative finite value"
            errors.append(f"{label}[{index}]: expected {expectation}")
            continue
        result.append(numeric)
    return result


def _validate_case(
    case: object,
    *,
    binding: ThresholdBinding,
    plan: G1BaselinePlan,
    errors: list[str],
    index: int,
) -> dict[str, Any] | None:
    parsed = _exact_keys(
        case,
        {
            "case_id",
            "mode",
            "batch_size",
            "repeat_index",
            "sequence_index",
            "process",
            "raw",
            "summary",
        },
        label=f"cases[{index}]",
        errors=errors,
    )
    if not parsed:
        return None
    mode = parsed.get("mode")
    batch_size = parsed.get("batch_size")
    repeat_index = parsed.get("repeat_index")
    sequence_index = parsed.get("sequence_index")
    if mode not in MODES:
        errors.append(f"cases[{index}].mode: unsupported mode")
        return None
    if batch_size not in binding.batch_sizes:
        errors.append(f"cases[{index}].batch_size: outside frozen matrix")
        return None
    if not isinstance(repeat_index, int) or repeat_index not in range(binding.process_repeats):
        errors.append(f"cases[{index}].repeat_index: outside frozen matrix")
    if not isinstance(sequence_index, int) or sequence_index not in range(len(MODES)):
        errors.append(f"cases[{index}].sequence_index: must be in [0, 2]")
    expected_id = _case_id(
        batch_size=cast(int, batch_size), repeat_index=cast(int, repeat_index), mode=cast(str, mode)
    )
    if parsed.get("case_id") != expected_id:
        errors.append(f"cases[{index}].case_id: expected {expected_id!r}")
    process = _exact_keys(
        parsed.get("process"),
        {
            "run_id",
            "pid",
            "started_at",
            "duration_sec",
            "return_code",
            "command",
            "affinity_cpus",
            "env_vars",
            "stdout_sha256",
            "stderr_sha256",
        },
        label=f"cases[{index}].process",
        errors=errors,
    )
    if process:
        if process.get("return_code") != 0:
            errors.append(f"cases[{index}].process.return_code: expected 0")
        try:
            uuid.UUID(str(process.get("run_id")))
        except (ValueError, AttributeError):
            errors.append(f"cases[{index}].process.run_id: expected UUID")
        if isinstance(process.get("pid"), bool) or not isinstance(process.get("pid"), int):
            errors.append(f"cases[{index}].process.pid: expected integer")
        duration = process.get("duration_sec")
        if (
            isinstance(duration, bool)
            or not isinstance(duration, (int, float))
            or not np.isfinite(float(duration))
            or float(duration) <= 0.0
        ):
            errors.append(f"cases[{index}].process.duration_sec: expected positive finite value")
        command = process.get("command")
        if not isinstance(command, list) or command[:2] != ["uv", "run"]:
            errors.append(f"cases[{index}].process.command: requires uv run subprocess")
        elif (
            len(command) < 3
            or command[2] != "benchmark/env/benchmark_managed_g1.py"
            or "--worker" not in command
            or "--mode" not in command
            or "--batch-size" not in command
            or "--repeat-index" not in command
        ):
            errors.append(f"cases[{index}].process.command: is not the registered host worker")
        else:
            try:
                command_mode = command[command.index("--mode") + 1]
                command_batch = command[command.index("--batch-size") + 1]
                command_repeat = command[command.index("--repeat-index") + 1]
            except IndexError:
                errors.append(f"cases[{index}].process.command: missing worker argument value")
            else:
                if (
                    command_mode != mode
                    or command_batch != str(batch_size)
                    or command_repeat != str(repeat_index)
                ):
                    errors.append(
                        f"cases[{index}].process.command: worker arguments differ from case"
                    )
        if process.get("affinity_cpus") != list(plan.hardware.affinity_cpus):
            errors.append(f"cases[{index}].process.affinity_cpus: differs from frozen plan")
        if process.get("env_vars") != dict(plan.environment.env_vars):
            errors.append(f"cases[{index}].process.env_vars: differs from frozen plan")
        for key in ("stdout_sha256", "stderr_sha256"):
            value = process.get(key)
            if (
                not isinstance(value, str)
                or len(value) != len("sha256:") + 64
                or not value.startswith("sha256:")
            ):
                errors.append(f"cases[{index}].process.{key}: expected SHA-256 receipt")
    raw = _exact_keys(
        parsed.get("raw"),
        {
            "timing_records",
            "memory",
            "resolved_env_config",
            "resolved_config_sha256",
            "action_seed",
            "reset_seed",
            "action_sha256",
            "executor_key",
            "plan_fingerprint",
            "backend_identity",
        },
        label=f"cases[{index}].raw",
        errors=errors,
    )
    if raw:
        timing = raw.get("timing_records")
        if not isinstance(timing, dict):
            errors.append(f"cases[{index}].raw.timing_records: expected mapping")
        else:
            total = _finite_samples(
                timing.get("env_step_total_ms"),
                count=binding.measure_steps,
                label=f"cases[{index}].raw.timing_records.env_step_total_ms",
                errors=errors,
                lower_bound=0.0,
                strict_lower_bound=True,
            )
            for name, values in timing.items():
                _finite_samples(
                    values,
                    count=binding.measure_steps,
                    label=f"cases[{index}].raw.timing_records.{name}",
                    errors=errors,
                )
            if total:
                try:
                    expected_summary = summarize_worker_raw(raw, batch_size=cast(int, batch_size))
                except HostBenchmarkError as exc:
                    errors.append(f"cases[{index}].summary: {exc}")
                else:
                    if not _summary_matches_recomputed(parsed.get("summary"), expected_summary):
                        errors.append(
                            f"cases[{index}].summary: does not recompute from raw samples"
                        )
        config = raw.get("resolved_env_config")
        if not isinstance(config, dict):
            errors.append(f"cases[{index}].raw.resolved_env_config: expected mapping")
        elif raw.get("resolved_config_sha256") != canonical_sha256(config):
            errors.append(f"cases[{index}].raw.resolved_config_sha256: does not match config")
        elif (
            config.get("adaptive_chunk_size") is not False
            or config.get("chunk_size") is not None
            or _nested(config, "domain_rand", "randomize_kp") is not False
            or _nested(config, "domain_rand", "randomize_kd") is not False
            or _nested(config, "noise_config", "level") != 0.0
            or config.get("max_episode_seconds") is not None
        ):
            errors.append(f"cases[{index}].raw.resolved_env_config: unsupported host profile")
        if raw.get("backend_identity") != "mujoco":
            errors.append(f"cases[{index}].raw.backend_identity: must remain mujoco")
        expected_seed = _action_seed(
            plan=plan,
            batch_size=cast(int, batch_size),
            repeat_index=cast(int, repeat_index),
        )
        if raw.get("action_seed") != expected_seed:
            errors.append(f"cases[{index}].raw.action_seed: differs from frozen action schedule")
        if raw.get("reset_seed") != expected_seed:
            errors.append(f"cases[{index}].raw.reset_seed: differs from frozen reset schedule")
        expected_action_hash = _expected_action_sha256(
            batch_size=cast(int, batch_size),
            sample_count=binding.warmup_steps + binding.measure_steps,
            seed=expected_seed,
        )
        if raw.get("action_sha256") != expected_action_hash:
            errors.append(f"cases[{index}].raw.action_sha256: differs from frozen action schedule")
        if raw.get("executor_key") != EXPECTED_EXECUTOR_KEYS[cast(str, mode)]:
            errors.append(f"cases[{index}].raw.executor_key: does not match benchmark mode")
        if not isinstance(raw.get("plan_fingerprint"), str) or not raw["plan_fingerprint"]:
            errors.append(f"cases[{index}].raw.plan_fingerprint: invalid")
    return parsed


def _nested(value: Mapping[str, Any], *keys: str) -> Any:
    current: Any = value
    for key in keys:
        if not isinstance(current, Mapping):
            return None
        current = current.get(key)
    return current


def _validate_hardware(value: object, *, plan: G1BaselinePlan, errors: list[str]) -> None:
    hardware = _exact_keys(
        value,
        {
            "platform_system",
            "platform_release",
            "cpu_model",
            "cpu_physical_cores",
            "cpu_logical_cores",
            "affinity_cpus",
            "gpu_name",
            "gpu_uuid",
            "gpu_memory_mib",
            "driver_version",
            "cuda_runtime",
            "torch_version",
            "hostname",
        },
        label="artifact.hardware",
        errors=errors,
    )
    if not hardware:
        return
    frozen = {
        "platform_system": plan.hardware.platform_system,
        "cpu_model": plan.hardware.cpu_model,
        "cpu_physical_cores": plan.hardware.cpu_physical_cores,
        "cpu_logical_cores": plan.hardware.cpu_logical_cores,
        "affinity_cpus": list(plan.hardware.affinity_cpus),
        "gpu_name": plan.hardware.gpu_name,
        "gpu_uuid": plan.hardware.gpu_uuid,
        "gpu_memory_mib": plan.hardware.gpu_memory_mib,
        "driver_version": plan.hardware.driver_version,
    }
    for key, expected in frozen.items():
        if hardware.get(key) != expected:
            errors.append(f"artifact.hardware.{key}: differs from frozen benchmark host")
    for key in ("platform_release", "cuda_runtime", "torch_version", "hostname"):
        if not isinstance(hardware.get(key), str) or not hardware[key]:
            errors.append(f"artifact.hardware.{key}: expected non-empty string")


def _validate_preflight(
    value: object,
    *,
    plan: G1BaselinePlan,
    errors: list[str],
    enforce_cpu_load: bool,
) -> None:
    preflight = _exact_keys(
        value,
        {
            "timestamp",
            "load_average_1m",
            "load_per_physical_core",
            "gpu_compute_processes",
            "gpu_samples",
        },
        label="artifact.execution.preflight",
        errors=errors,
    )
    if not preflight:
        return
    load = preflight.get("load_per_physical_core")
    if isinstance(load, bool) or not isinstance(load, (int, float)):
        errors.append("artifact.execution.preflight.load_per_physical_core: expected number")
    elif enforce_cpu_load and float(load) > plan.preflight.max_load_per_physical_core:
        errors.append("artifact.execution.preflight: CPU load exceeds frozen limit")
    processes = preflight.get("gpu_compute_processes")
    if not isinstance(processes, list):
        errors.append("artifact.execution.preflight.gpu_compute_processes: expected list")
    elif len(processes) > plan.preflight.max_gpu_compute_processes:
        errors.append("artifact.execution.preflight: foreign GPU compute process detected")
    samples = preflight.get("gpu_samples")
    if not isinstance(samples, list) or len(samples) != plan.preflight.gpu_samples:
        errors.append("artifact.execution.preflight.gpu_samples: wrong frozen sample count")
        return
    for index, sample in enumerate(samples):
        parsed = _exact_keys(
            sample,
            {"utilization_percent", "memory_used_mib", "temperature_c", "pstate"},
            label=f"artifact.execution.preflight.gpu_samples[{index}]",
            errors=errors,
        )
        utilization = parsed.get("utilization_percent")
        if isinstance(utilization, bool) or not isinstance(utilization, (int, float)):
            errors.append(
                f"artifact.execution.preflight.gpu_samples[{index}].utilization_percent: invalid"
            )
        elif float(utilization) > plan.preflight.max_gpu_utilization_percent:
            errors.append("artifact.execution.preflight: GPU utilization exceeds frozen limit")


def _verify_candidate_source(
    candidate: Mapping[str, Any],
    *,
    binding: ThresholdBinding,
    plan: G1BaselinePlan,
    errors: list[str],
) -> None:
    commit = candidate.get("candidate_commit")
    if not _is_commit(commit):
        errors.append("candidate.candidate_commit: expected full Git SHA")
        return
    if commit == binding.freeze_commit:
        errors.append("candidate.candidate_commit: cannot equal threshold freeze commit")
    try:
        exists = subprocess.run(
            ["git", "cat-file", "-e", f"{commit}^{{commit}}"],
            cwd=ROOT_DIR,
            check=False,
            capture_output=True,
        )
        if exists.returncode != 0:
            errors.append("candidate.candidate_commit: Git object is unavailable")
            return
        ancestor = subprocess.run(
            ["git", "merge-base", "--is-ancestor", cast(str, commit), "HEAD"],
            cwd=ROOT_DIR,
            check=False,
            capture_output=True,
        )
        if ancestor.returncode != 0:
            errors.append("candidate.candidate_commit: is not an ancestor of HEAD")
            return
        frozen_ancestor = subprocess.run(
            ["git", "merge-base", "--is-ancestor", binding.freeze_commit, cast(str, commit)],
            cwd=ROOT_DIR,
            check=False,
            capture_output=True,
        )
        if frozen_ancestor.returncode != 0:
            errors.append("candidate.candidate_commit: does not descend from threshold freeze")
            return
        commit_string = cast(str, commit)
        expected = {
            "source_tree_sha256": source_tree_sha256_at_commit(
                ROOT_DIR, SOURCE_INPUTS, cast(str, commit)
            ),
            "uv_lock_sha256": f"sha256:{hashlib.sha256(_git_file_bytes(commit_string, 'uv.lock')).hexdigest()}",
            "owner_yaml_sha256": f"sha256:{hashlib.sha256(_git_file_bytes(commit_string, plan.owner_yaml)).hexdigest()}",
        }
        for key, expected_value in expected.items():
            if candidate.get(key) != expected_value:
                errors.append(f"candidate.{key}: does not match candidate commit")
    except (OSError, subprocess.CalledProcessError) as exc:
        errors.append(f"candidate source verification failed: {exc}")


def _git_file_bytes(commit: str, path: str) -> bytes:
    completed = subprocess.run(
        ["git", "show", f"{commit}:{path}"],
        cwd=ROOT_DIR,
        check=True,
        capture_output=True,
    )
    return completed.stdout


def validate_artifact(
    artifact: object,
    *,
    binding: ThresholdBinding,
    plan: G1BaselinePlan,
    repo_root: Path | None = None,
    require_passing_gate: bool = True,
) -> tuple[str, ...]:
    """Return all contract errors; an empty tuple means a genuine PASS artifact."""

    try:
        _validate_plan_binding(plan, binding)
    except HostBenchmarkError as exc:
        return (str(exc),)
    errors: list[str] = []
    root = _exact_keys(
        artifact,
        {
            "schema_version",
            "issue",
            "profile",
            "threshold",
            "candidate",
            "hardware",
            "execution",
            "cases",
            "aggregates",
            "gate",
        },
        label="artifact",
        errors=errors,
    )
    if not root:
        return tuple(errors)
    if root.get("schema_version") != SCHEMA_VERSION:
        errors.append("artifact.schema_version: unsupported")
    if root.get("issue") != ISSUE:
        errors.append("artifact.issue: must be 705")
    if root.get("profile") != PROFILE:
        errors.append("artifact.profile: must be host_numpy")
    _validate_hardware(root.get("hardware"), plan=plan, errors=errors)
    threshold = _exact_keys(
        root.get("threshold"),
        {"threshold_set_id", "manifest_path", "manifest_sha256", "freeze_commit"},
        label="artifact.threshold",
        errors=errors,
    )
    expected_threshold = {
        "threshold_set_id": binding.threshold_set_id,
        "manifest_path": binding.manifest_path.as_posix(),
        "manifest_sha256": binding.manifest_sha256,
        "freeze_commit": binding.freeze_commit,
    }
    if threshold and threshold != expected_threshold:
        errors.append("artifact.threshold: differs from frozen receipt")
    candidate = _exact_keys(
        root.get("candidate"),
        {
            "candidate_commit",
            "branch",
            "source_dirty",
            "source_tree_sha256",
            "uv_lock_sha256",
            "owner_yaml_sha256",
        },
        label="artifact.candidate",
        errors=errors,
    )
    if candidate and candidate.get("source_dirty") is not False:
        errors.append("artifact.candidate.source_dirty: must be false")
    execution = _exact_keys(
        root.get("execution"),
        {
            "process_isolation",
            "affinity_cpus",
            "env_vars",
            "warmup_steps",
            "measure_steps",
            "mode_order_policy",
            "preflight_before",
            "preflight_after",
        },
        label="artifact.execution",
        errors=errors,
    )
    if execution:
        if execution.get("process_isolation") is not True:
            errors.append("artifact.execution.process_isolation: must be true")
        if execution.get("affinity_cpus") != list(plan.hardware.affinity_cpus):
            errors.append("artifact.execution.affinity_cpus: differs from frozen plan")
        if execution.get("env_vars") != dict(plan.environment.env_vars):
            errors.append("artifact.execution.env_vars: differs from frozen plan")
        if execution.get("warmup_steps") != binding.warmup_steps:
            errors.append("artifact.execution.warmup_steps: differs from frozen plan")
        if execution.get("measure_steps") != binding.measure_steps:
            errors.append("artifact.execution.measure_steps: differs from frozen plan")
        if (
            execution.get("mode_order_policy")
            != "repeat-index cyclic rotation of hand_written, managed_reference, managed_fused"
        ):
            errors.append("artifact.execution.mode_order_policy: differs from frozen protocol")
        _validate_preflight(
            execution.get("preflight_before"),
            plan=plan,
            errors=errors,
            enforce_cpu_load=True,
        )
        _validate_preflight(
            execution.get("preflight_after"),
            plan=plan,
            errors=errors,
            enforce_cpu_load=False,
        )
    cases_raw = root.get("cases")
    parsed_cases: list[dict[str, Any]] = []
    if not isinstance(cases_raw, list):
        errors.append("artifact.cases: expected list")
    else:
        for index, case in enumerate(cases_raw):
            parsed = _validate_case(case, binding=binding, plan=plan, errors=errors, index=index)
            if parsed is not None:
                parsed_cases.append(parsed)
    actual_ids = [str(case.get("case_id")) for case in parsed_cases]
    expected_ids = expected_case_ids(binding)
    if set(actual_ids) != expected_ids or len(actual_ids) != len(expected_ids):
        errors.append("artifact.cases: incomplete, duplicate, or unexpected process matrix")
    run_ids = [
        str(case["process"].get("run_id"))
        for case in parsed_cases
        if isinstance(case.get("process"), dict)
    ]
    if len(set(run_ids)) != len(run_ids):
        errors.append("artifact.cases: duplicate isolated-process run_id")
    for batch_size in binding.batch_sizes:
        for repeat_index in range(binding.process_repeats):
            pair = [
                case
                for case in parsed_cases
                if case.get("batch_size") == batch_size and case.get("repeat_index") == repeat_index
            ]
            if len(pair) != len(MODES):
                continue
            sequence_indices = tuple(sorted(int(case["sequence_index"]) for case in pair))
            if sequence_indices != tuple(range(len(MODES))):
                errors.append(
                    f"artifact.cases: pair b{batch_size}/r{repeat_index} has duplicate sequence index"
                )
            actual_order = tuple(
                case["mode"] for case in sorted(pair, key=lambda item: int(item["sequence_index"]))
            )
            if actual_order != _mode_order(repeat_index):
                errors.append(
                    f"artifact.cases: pair b{batch_size}/r{repeat_index} is not cyclically interleaved"
                )
            hashes = {
                case["raw"].get("action_sha256")
                for case in pair
                if isinstance(case.get("raw"), dict)
            }
            if len(hashes) != 1:
                errors.append(
                    f"artifact.cases: pair b{batch_size}/r{repeat_index} did not use identical actions"
                )
    if parsed_cases:
        try:
            expected_aggregates = build_aggregates(parsed_cases, binding)
        except HostBenchmarkError as exc:
            errors.append(f"artifact.aggregates: {exc}")
            expected_aggregates = None
        if expected_aggregates is not None and root.get("aggregates") != expected_aggregates:
            errors.append("artifact.aggregates: does not recompute from raw process summaries")
        if expected_aggregates is not None:
            try:
                expected_gate = build_gate(expected_aggregates, binding)
            except HostBenchmarkError as exc:
                errors.append(f"artifact.gate: {exc}")
            else:
                if root.get("gate") != expected_gate:
                    errors.append("artifact.gate: does not recompute from frozen thresholds")
                elif require_passing_gate and expected_gate.get("passed") is not True:
                    errors.append("artifact.gate: host fused performance threshold failed")
    if candidate and not errors and repo_root is not None:
        _verify_candidate_source(candidate, binding=binding, plan=plan, errors=errors)
    return tuple(errors)


def _write_worker_output(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-plan", type=Path, default=DEFAULT_BASELINE_PLAN)
    parser.add_argument("--threshold-manifest", type=Path, default=DEFAULT_THRESHOLD_MANIFEST)
    parser.add_argument("--threshold-receipt", type=Path, default=DEFAULT_THRESHOLD_RECEIPT)
    parser.add_argument("--profile", choices=(PROFILE,), default=PROFILE)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument(
        "--allow-gate-failure",
        action="store_true",
        help=(
            "write a structurally validated diagnostic artifact even when the frozen "
            "performance gate fails; it remains invalid for evidence validation"
        ),
    )
    parser.add_argument("--list-cases", action="store_true")
    parser.add_argument("--validate-artifact", type=Path)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--mode", choices=MODES)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--repeat-index", type=int)
    parser.add_argument("--worker-output", type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.profile != PROFILE:  # pragma: no cover - argparse choices enforce this boundary.
        raise HostBenchmarkError(f"unsupported host benchmark profile {args.profile!r}")
    if args.allow_gate_failure and not args.execute:
        raise HostBenchmarkError("--allow-gate-failure is valid only with --execute")
    plan = _load_plan(args.baseline_plan)
    binding = load_threshold_binding(
        threshold_manifest=args.threshold_manifest,
        threshold_receipt=args.threshold_receipt,
    )
    _validate_plan_binding(plan, binding)
    if args.worker:
        if (
            args.mode is None
            or args.batch_size is None
            or args.repeat_index is None
            or args.worker_output is None
        ):
            raise HostBenchmarkError(
                "worker requires --mode, --batch-size, --repeat-index, --worker-output"
            )
        _write_worker_output(
            args.worker_output,
            _run_worker(
                plan=plan,
                binding=binding,
                mode=args.mode,
                batch_size=args.batch_size,
                repeat_index=args.repeat_index,
            ),
        )
        return 0
    if args.list_cases:
        for case_id in sorted(expected_case_ids(binding)):
            print(case_id)
        return 0
    if args.validate_artifact is not None:
        artifact = json.loads(args.validate_artifact.read_text(encoding="utf-8"))
        errors = validate_artifact(artifact, binding=binding, plan=plan, repo_root=ROOT_DIR)
        if errors:
            raise HostBenchmarkError(
                "artifact validation failed:\n" + "\n".join(f"- {item}" for item in errors)
            )
        print(f"PASS validated {args.validate_artifact}")
        return 0
    if not args.execute:
        raise SystemExit(
            "Refusing to run implicitly; pass --execute, --list-cases, or --validate-artifact"
        )
    artifact = _collect(plan, binding, allow_gate_failure=args.allow_gate_failure)
    output = args.out if args.out.is_absolute() else ROOT_DIR / args.out
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(artifact, indent=2, sort_keys=True), encoding="utf-8")
    if artifact["gate"]["passed"] is not True:
        print(f"DIAGNOSTIC gate failed; wrote non-evidence artifact to {output}")
        return 2
    print(f"PASS wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
