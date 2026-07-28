"""Strict plan and artifact contracts for the Issue #705 G1 baseline."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence, cast

import numpy as np
from omegaconf import OmegaConf

SCHEMA_VERSION = 1
ISSUE = 705
BASELINE_ID = "g1-mujoco-phase0-v1"
BACKEND = "mujoco"
TASK = "g1_walk_flat"

_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
_SHA256_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
_MISSING = object()


@dataclass(frozen=True)
class HardwarePlan:
    platform_system: str
    cpu_model: str
    cpu_physical_cores: int
    cpu_logical_cores: int
    affinity_cpus: tuple[int, ...]
    gpu_name: str
    gpu_uuid: str
    gpu_memory_mib: int
    driver_version: str


@dataclass(frozen=True)
class PreflightPlan:
    max_load_per_physical_core: float
    max_gpu_compute_processes: int
    max_gpu_utilization_percent: int
    gpu_samples: int
    sample_interval_sec: float


@dataclass(frozen=True)
class EnvironmentPlan:
    dtype: str
    hydra_overrides: tuple[str, ...]
    env_vars: tuple[tuple[str, str], ...]


@dataclass(frozen=True)
class EnvLanePlan:
    batch_sizes: tuple[int, ...]
    process_repeats: int
    warmup_steps: int
    measure_steps: int
    action_seed_base: int


@dataclass(frozen=True)
class DrLanePlan:
    num_envs: int
    modes: tuple[str, ...]
    reset_densities: tuple[float, ...]
    process_repeats: int
    warmup_resets: int
    measure_resets: int
    reset_seed_base: int


@dataclass(frozen=True)
class PpoLanePlan:
    num_envs: int
    num_steps_per_env: int
    max_iterations: int
    warmup_iterations: int
    seeds: tuple[int, ...]
    save_interval: int
    memory_poll_interval_sec: float
    required_scalar_tags: tuple[str, ...]


@dataclass(frozen=True)
class G1BaselinePlan:
    schema_version: int
    issue: int
    baseline_id: str
    backend: str
    task: str
    owner_yaml: str
    source_inputs: tuple[str, ...]
    hardware: HardwarePlan
    preflight: PreflightPlan
    environment: EnvironmentPlan
    env_lane: EnvLanePlan
    dr_lane: DrLanePlan
    ppo_lane: PpoLanePlan
    source_path: Path


class BaselineValidationError(ValueError):
    def __init__(self, source: Path, errors: Iterable[str]) -> None:
        self.source = source
        self.errors = tuple(errors)
        detail = "\n".join(f"- {error}" for error in self.errors)
        super().__init__(f"invalid Issue #705 baseline data {source}:\n{detail}")


class _Parser:
    def __init__(self) -> None:
        self.errors: list[str] = []

    def mapping(self, value: Any, path: str, keys: Sequence[str]) -> dict[str, Any]:
        if not isinstance(value, dict):
            self.errors.append(f"{path}: expected mapping")
            return {}
        expected = set(keys)
        actual = set(value)
        for key in sorted(expected - actual):
            self.errors.append(f"{path}: missing key `{key}`")
        for key in sorted(actual - expected, key=str):
            self.errors.append(f"{path}: unknown key `{key}`")
        return value

    def string(self, value: Any, path: str) -> str:
        if not isinstance(value, str) or not value.strip():
            self.errors.append(f"{path}: expected non-empty string")
            return ""
        if "${" in value:
            self.errors.append(f"{path}: interpolation is not allowed")
        return value

    def integer(self, value: Any, path: str, *, minimum: int | None = None) -> int:
        if isinstance(value, bool) or not isinstance(value, int):
            self.errors.append(f"{path}: expected integer")
            return 0
        if minimum is not None and value < minimum:
            self.errors.append(f"{path}: must be >= {minimum}")
        return int(value)

    def number(self, value: Any, path: str, *, minimum: float | None = None) -> float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            self.errors.append(f"{path}: expected number")
            return 0.0
        result = float(value)
        if not math.isfinite(result):
            self.errors.append(f"{path}: must be finite")
        if minimum is not None and result < minimum:
            self.errors.append(f"{path}: must be >= {minimum}")
        return result

    def string_list(self, value: Any, path: str) -> tuple[str, ...]:
        if not isinstance(value, list) or not value:
            self.errors.append(f"{path}: expected non-empty string list")
            return ()
        result = tuple(self.string(item, f"{path}[{index}]") for index, item in enumerate(value))
        self._duplicates(result, path)
        return result

    def integer_list(self, value: Any, path: str, *, minimum: int = 0) -> tuple[int, ...]:
        if not isinstance(value, list) or not value:
            self.errors.append(f"{path}: expected non-empty integer list")
            return ()
        result = tuple(
            self.integer(item, f"{path}[{index}]", minimum=minimum)
            for index, item in enumerate(value)
        )
        self._duplicates(result, path)
        return result

    def number_list(self, value: Any, path: str, *, minimum: float = 0.0) -> tuple[float, ...]:
        if not isinstance(value, list) or not value:
            self.errors.append(f"{path}: expected non-empty number list")
            return ()
        result = tuple(
            self.number(item, f"{path}[{index}]", minimum=minimum)
            for index, item in enumerate(value)
        )
        self._duplicates(result, path)
        return result

    def _duplicates(self, values: Sequence[Any], path: str) -> None:
        duplicates = sorted({value for value in values if values.count(value) > 1}, key=str)
        if duplicates:
            self.errors.append(f"{path}: duplicate values {duplicates!r}")


_ROOT_KEYS = (
    "schema_version",
    "issue",
    "baseline_id",
    "backend",
    "task",
    "owner_yaml",
    "source_inputs",
    "hardware",
    "preflight",
    "environment",
    "env_lane",
    "dr_lane",
    "ppo_lane",
)
_HARDWARE_PLAN_KEYS = (
    "platform_system",
    "cpu_model",
    "cpu_physical_cores",
    "cpu_logical_cores",
    "affinity_cpus",
    "gpu_name",
    "gpu_uuid",
    "gpu_memory_mib",
    "driver_version",
)
_PREFLIGHT_KEYS = (
    "max_load_per_physical_core",
    "max_gpu_compute_processes",
    "max_gpu_utilization_percent",
    "gpu_samples",
    "sample_interval_sec",
)
_ENVIRONMENT_KEYS = ("dtype", "hydra_overrides", "env_vars")
_ENV_LANE_KEYS = (
    "batch_sizes",
    "process_repeats",
    "warmup_steps",
    "measure_steps",
    "action_seed_base",
)
_DR_LANE_KEYS = (
    "num_envs",
    "modes",
    "reset_densities",
    "process_repeats",
    "warmup_resets",
    "measure_resets",
    "reset_seed_base",
)
_PPO_LANE_KEYS = (
    "num_envs",
    "num_steps_per_env",
    "max_iterations",
    "warmup_iterations",
    "seeds",
    "save_interval",
    "memory_poll_interval_sec",
    "required_scalar_tags",
)


def parse_g1_baseline_plan(raw: Any, *, source: Path = Path("<memory>")) -> G1BaselinePlan:
    parser = _Parser()
    root = parser.mapping(raw, "plan", _ROOT_KEYS)
    hardware_raw = parser.mapping(root.get("hardware"), "hardware", _HARDWARE_PLAN_KEYS)
    preflight_raw = parser.mapping(root.get("preflight"), "preflight", _PREFLIGHT_KEYS)
    environment_raw = parser.mapping(root.get("environment"), "environment", _ENVIRONMENT_KEYS)
    env_raw = parser.mapping(root.get("env_lane"), "env_lane", _ENV_LANE_KEYS)
    dr_raw = parser.mapping(root.get("dr_lane"), "dr_lane", _DR_LANE_KEYS)
    ppo_raw = parser.mapping(root.get("ppo_lane"), "ppo_lane", _PPO_LANE_KEYS)

    env_vars_raw = environment_raw.get("env_vars")
    env_vars: list[tuple[str, str]] = []
    if not isinstance(env_vars_raw, dict) or not env_vars_raw:
        parser.errors.append("environment.env_vars: expected non-empty mapping")
    else:
        for key, value in env_vars_raw.items():
            env_vars.append(
                (
                    parser.string(key, "environment.env_vars.<key>"),
                    parser.string(value, f"environment.env_vars.{key}"),
                )
            )

    plan = G1BaselinePlan(
        schema_version=parser.integer(root.get("schema_version"), "schema_version"),
        issue=parser.integer(root.get("issue"), "issue"),
        baseline_id=parser.string(root.get("baseline_id"), "baseline_id"),
        backend=parser.string(root.get("backend"), "backend"),
        task=parser.string(root.get("task"), "task"),
        owner_yaml=parser.string(root.get("owner_yaml"), "owner_yaml"),
        source_inputs=parser.string_list(root.get("source_inputs"), "source_inputs"),
        hardware=HardwarePlan(
            platform_system=parser.string(
                hardware_raw.get("platform_system"), "hardware.platform_system"
            ),
            cpu_model=parser.string(hardware_raw.get("cpu_model"), "hardware.cpu_model"),
            cpu_physical_cores=parser.integer(
                hardware_raw.get("cpu_physical_cores"),
                "hardware.cpu_physical_cores",
                minimum=1,
            ),
            cpu_logical_cores=parser.integer(
                hardware_raw.get("cpu_logical_cores"),
                "hardware.cpu_logical_cores",
                minimum=1,
            ),
            affinity_cpus=parser.integer_list(
                hardware_raw.get("affinity_cpus"), "hardware.affinity_cpus"
            ),
            gpu_name=parser.string(hardware_raw.get("gpu_name"), "hardware.gpu_name"),
            gpu_uuid=parser.string(hardware_raw.get("gpu_uuid"), "hardware.gpu_uuid"),
            gpu_memory_mib=parser.integer(
                hardware_raw.get("gpu_memory_mib"), "hardware.gpu_memory_mib", minimum=1
            ),
            driver_version=parser.string(
                hardware_raw.get("driver_version"), "hardware.driver_version"
            ),
        ),
        preflight=PreflightPlan(
            max_load_per_physical_core=parser.number(
                preflight_raw.get("max_load_per_physical_core"),
                "preflight.max_load_per_physical_core",
                minimum=0.0,
            ),
            max_gpu_compute_processes=parser.integer(
                preflight_raw.get("max_gpu_compute_processes"),
                "preflight.max_gpu_compute_processes",
                minimum=0,
            ),
            max_gpu_utilization_percent=parser.integer(
                preflight_raw.get("max_gpu_utilization_percent"),
                "preflight.max_gpu_utilization_percent",
                minimum=0,
            ),
            gpu_samples=parser.integer(
                preflight_raw.get("gpu_samples"), "preflight.gpu_samples", minimum=1
            ),
            sample_interval_sec=parser.number(
                preflight_raw.get("sample_interval_sec"),
                "preflight.sample_interval_sec",
                minimum=0.05,
            ),
        ),
        environment=EnvironmentPlan(
            dtype=parser.string(environment_raw.get("dtype"), "environment.dtype"),
            hydra_overrides=parser.string_list(
                environment_raw.get("hydra_overrides"), "environment.hydra_overrides"
            ),
            env_vars=tuple(sorted(env_vars)),
        ),
        env_lane=EnvLanePlan(
            batch_sizes=parser.integer_list(
                env_raw.get("batch_sizes"), "env_lane.batch_sizes", minimum=1
            ),
            process_repeats=parser.integer(
                env_raw.get("process_repeats"), "env_lane.process_repeats", minimum=1
            ),
            warmup_steps=parser.integer(
                env_raw.get("warmup_steps"), "env_lane.warmup_steps", minimum=1
            ),
            measure_steps=parser.integer(
                env_raw.get("measure_steps"), "env_lane.measure_steps", minimum=1
            ),
            action_seed_base=parser.integer(
                env_raw.get("action_seed_base"), "env_lane.action_seed_base", minimum=0
            ),
        ),
        dr_lane=DrLanePlan(
            num_envs=parser.integer(dr_raw.get("num_envs"), "dr_lane.num_envs", minimum=1),
            modes=parser.string_list(dr_raw.get("modes"), "dr_lane.modes"),
            reset_densities=parser.number_list(
                dr_raw.get("reset_densities"), "dr_lane.reset_densities", minimum=0.0
            ),
            process_repeats=parser.integer(
                dr_raw.get("process_repeats"), "dr_lane.process_repeats", minimum=1
            ),
            warmup_resets=parser.integer(
                dr_raw.get("warmup_resets"), "dr_lane.warmup_resets", minimum=1
            ),
            measure_resets=parser.integer(
                dr_raw.get("measure_resets"), "dr_lane.measure_resets", minimum=1
            ),
            reset_seed_base=parser.integer(
                dr_raw.get("reset_seed_base"), "dr_lane.reset_seed_base", minimum=0
            ),
        ),
        ppo_lane=PpoLanePlan(
            num_envs=parser.integer(ppo_raw.get("num_envs"), "ppo_lane.num_envs", minimum=1),
            num_steps_per_env=parser.integer(
                ppo_raw.get("num_steps_per_env"),
                "ppo_lane.num_steps_per_env",
                minimum=1,
            ),
            max_iterations=parser.integer(
                ppo_raw.get("max_iterations"), "ppo_lane.max_iterations", minimum=2
            ),
            warmup_iterations=parser.integer(
                ppo_raw.get("warmup_iterations"),
                "ppo_lane.warmup_iterations",
                minimum=1,
            ),
            seeds=parser.integer_list(ppo_raw.get("seeds"), "ppo_lane.seeds"),
            save_interval=parser.integer(
                ppo_raw.get("save_interval"), "ppo_lane.save_interval", minimum=1
            ),
            memory_poll_interval_sec=parser.number(
                ppo_raw.get("memory_poll_interval_sec"),
                "ppo_lane.memory_poll_interval_sec",
                minimum=0.05,
            ),
            required_scalar_tags=parser.string_list(
                ppo_raw.get("required_scalar_tags"), "ppo_lane.required_scalar_tags"
            ),
        ),
        source_path=source,
    )
    parser.errors.extend(_plan_semantic_errors(plan))
    if parser.errors:
        raise BaselineValidationError(source, parser.errors)
    return plan


def _plan_semantic_errors(plan: G1BaselinePlan) -> list[str]:
    errors: list[str] = []
    expected = {
        "schema_version": SCHEMA_VERSION,
        "issue": ISSUE,
        "baseline_id": BASELINE_ID,
        "backend": BACKEND,
        "task": TASK,
    }
    for field, value in expected.items():
        actual = getattr(plan, field)
        if actual != value:
            errors.append(f"{field}: expected {value!r}, got {actual!r}")
    if plan.owner_yaml != "conf/ppo/task/g1_walk_flat/mujoco.yaml":
        errors.append("owner_yaml: must select the G1 MuJoCo owner")
    if plan.env_lane.batch_sizes != (128, 1024, 4096):
        errors.append("env_lane.batch_sizes: must remain [128, 1024, 4096]")
    if plan.env_lane.process_repeats < 5 or plan.dr_lane.process_repeats < 5:
        errors.append("process repeats: env and DR lanes require at least five processes")
    if plan.ppo_lane.seeds != (0, 1, 2, 3, 4):
        errors.append("ppo_lane.seeds: must remain [0, 1, 2, 3, 4]")
    if plan.ppo_lane.max_iterations <= plan.ppo_lane.warmup_iterations:
        errors.append("ppo_lane: max_iterations must exceed warmup_iterations")
    if set(plan.dr_lane.modes) != {"disabled", "default_kp_kd"}:
        errors.append("dr_lane.modes: must cover disabled and default_kp_kd")
    if plan.preflight.max_gpu_compute_processes != 0:
        errors.append("preflight.max_gpu_compute_processes: baseline host must be exclusive")
    if plan.preflight.max_gpu_utilization_percent > 50:
        errors.append("preflight.max_gpu_utilization_percent: must remain <= 50")
    if plan.environment.hydra_overrides != (
        "env.adaptive_chunk_size=false",
        "env.chunk_size=null",
    ):
        errors.append("environment.hydra_overrides: does not match the frozen benchmark profile")
    if any(density <= 0.0 or density > 1.0 for density in plan.dr_lane.reset_densities):
        errors.append("dr_lane.reset_densities: each density must be in (0, 1]")
    required_tags = {
        "Perf/total_fps",
        "Perf/collection_time",
        "Perf/learning_time",
        "Train/mean_reward",
        "Train/mean_episode_length",
    }
    if set(plan.ppo_lane.required_scalar_tags) != required_tags:
        errors.append("ppo_lane.required_scalar_tags: does not match the frozen PPO evidence set")
    for source_input in plan.source_inputs:
        candidate = Path(source_input)
        if candidate.is_absolute() or ".." in candidate.parts:
            errors.append(f"source_inputs: invalid repository-relative path {source_input!r}")
    return errors


def load_g1_baseline_plan(path: Path) -> G1BaselinePlan:
    try:
        raw = OmegaConf.to_container(OmegaConf.load(path), resolve=False)
    except Exception as exc:  # noqa: BLE001 - normalize malformed plan errors
        raise BaselineValidationError(
            path, [f"cannot load YAML: {type(exc).__name__}: {exc}"]
        ) from exc
    return parse_g1_baseline_plan(raw, source=path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return f"sha256:{hashlib.sha256(payload.encode('utf-8')).hexdigest()}"


def source_tree_sha256(root: Path, source_inputs: Sequence[str]) -> str:
    files: list[Path] = []
    for item in source_inputs:
        candidate = root / item
        if candidate.is_file():
            files.append(candidate)
        elif candidate.is_dir():
            files.extend(
                path
                for path in candidate.rglob("*")
                if path.is_file() and "__pycache__" not in path.parts
            )
        else:
            raise FileNotFoundError(f"source input does not exist: {item}")
    digest = hashlib.sha256()
    for path in sorted(set(files), key=lambda item: item.relative_to(root).as_posix()):
        relative = path.relative_to(root).as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return f"sha256:{digest.hexdigest()}"


def numeric_stats(values: Sequence[float]) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or array.size == 0 or not np.all(np.isfinite(array)):
        raise ValueError("numeric samples must be a non-empty finite vector")
    return {
        "count": int(array.size),
        "mean": float(array.mean()),
        "p50": float(np.percentile(array, 50)),
        "p95": float(np.percentile(array, 95)),
        "min": float(array.min()),
        "max": float(array.max()),
    }


def _reward_auc(points: Sequence[Mapping[str, Any]], warmup: int) -> float:
    selected = sorted(
        (int(point["step"]), float(point["value"]))
        for point in points
        if int(point["step"]) >= warmup
    )
    if len(selected) < 2:
        raise ValueError("reward AUC requires at least two post-warmup points")
    steps = np.asarray([point[0] for point in selected], dtype=np.float64)
    values = np.asarray([point[1] for point in selected], dtype=np.float64)
    return float(np.trapezoid(values, steps))


def summarize_env_raw(raw: Mapping[str, Any], batch_size: int) -> dict[str, Any]:
    timing_records = raw.get("timing_records")
    if not isinstance(timing_records, dict):
        raise ValueError("env raw timing_records must be a mapping")
    total = _float_samples(timing_records.get("env_step_total_ms"), "env_step_total_ms")
    timing_stats = {
        str(key): numeric_stats(_float_samples(values, f"timing_records.{key}"))
        for key, values in sorted(timing_records.items())
    }
    total_sec = sum(total) / 1000.0
    return {
        "timing_stats_ms": timing_stats,
        "throughput_env_steps_per_sec": float(batch_size * len(total) / total_sec),
        "memory": _env_memory_summary(raw.get("memory")),
    }


def summarize_dr_raw(raw: Mapping[str, Any]) -> dict[str, Any]:
    samples = raw.get("reset_samples")
    if not isinstance(samples, list) or not samples:
        raise ValueError("DR raw reset_samples must be a non-empty list")
    by_key: dict[str, list[float]] = {}
    row_counts: list[float] = []
    for index, sample in enumerate(samples):
        if not isinstance(sample, dict):
            raise ValueError(f"reset_samples[{index}] must be a mapping")
        row_counts.append(_finite_float(sample.get("actual_rows"), "actual_rows"))
        timing = sample.get("timing")
        if not isinstance(timing, dict) or not timing:
            raise ValueError(f"reset_samples[{index}].timing must be non-empty")
        for key, value in timing.items():
            by_key.setdefault(str(key), []).append(_finite_float(value, f"timing.{key}"))
    return {
        "row_count": numeric_stats(row_counts),
        "timing_stats_ms": {key: numeric_stats(values) for key, values in sorted(by_key.items())},
        "memory": _env_memory_summary(raw.get("memory")),
    }


def summarize_ppo_raw(raw: Mapping[str, Any], plan: PpoLanePlan) -> dict[str, Any]:
    scalars = raw.get("scalars")
    if not isinstance(scalars, dict):
        raise ValueError("PPO raw scalars must be a mapping")
    scalar_stats: dict[str, dict[str, float | int]] = {}
    for tag in plan.required_scalar_tags:
        points = scalars.get(tag)
        if not isinstance(points, list):
            raise ValueError(f"missing scalar tag {tag}")
        values = [
            _finite_float(point.get("value"), f"scalars.{tag}.value")
            for point in points
            if isinstance(point, dict) and int(point.get("step", -1)) >= plan.warmup_iterations
        ]
        scalar_stats[tag] = numeric_stats(values)
    memory_samples = raw.get("memory_samples")
    if not isinstance(memory_samples, list) or not memory_samples:
        raise ValueError("PPO raw memory_samples must be non-empty")
    rss = [_finite_float(sample.get("rss_bytes"), "memory.rss_bytes") for sample in memory_samples]
    run_summary = raw.get("run_summary")
    if not isinstance(run_summary, dict):
        raise ValueError("PPO raw run_summary must be a mapping")
    peak_gpu_allocated = _finite_float(
        run_summary.get("peak_gpu_memory_allocated_bytes"),
        "run_summary.peak_gpu_memory_allocated_bytes",
    )
    peak_gpu_reserved = _finite_float(
        run_summary.get("peak_gpu_memory_reserved_bytes"),
        "run_summary.peak_gpu_memory_reserved_bytes",
    )
    reward_points = scalars["Train/mean_reward"]
    return {
        "scalar_stats": scalar_stats,
        "reward_auc": _reward_auc(reward_points, plan.warmup_iterations),
        "peak_rss_bytes": int(max(rss)),
        "peak_gpu_memory_allocated_bytes": int(peak_gpu_allocated),
        "peak_gpu_memory_reserved_bytes": int(peak_gpu_reserved),
    }


def _env_memory_summary(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError("memory must be a mapping")
    return {
        "preferred_metric": str(value.get("preferred_metric")),
        "total_rss_delta_bytes": _optional_int(value.get("total_rss_delta_bytes")),
        "total_uss_delta_bytes": _optional_int(value.get("total_uss_delta_bytes")),
        "after_benchmark_rss_bytes": _optional_int(value.get("after_benchmark_rss_bytes")),
        "after_benchmark_uss_bytes": _optional_int(value.get("after_benchmark_uss_bytes")),
    }


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("memory value must be numeric or null")
    return int(value)


def _finite_float(value: Any, path: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{path} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{path} must be finite")
    return result


def _float_samples(value: Any, path: str) -> list[float]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{path} must be a non-empty list")
    return [_finite_float(item, path) for item in value]


_ARTIFACT_ROOT_KEYS = (
    "schema_version",
    "issue",
    "baseline_id",
    "generated_at",
    "plan",
    "source",
    "hardware",
    "execution",
    "cases",
    "aggregates",
)
_ARTIFACT_PLAN_KEYS = ("path", "sha256")
_SOURCE_KEYS = (
    "commit",
    "branch",
    "dirty",
    "tree_sha256",
    "uv_lock_sha256",
    "owner_yaml_sha256",
)
_HARDWARE_KEYS = (
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
)
_EXECUTION_KEYS = (
    "process_isolation",
    "affinity_cpus",
    "env_vars",
    "hydra_overrides",
    "preflight",
)
_PREFLIGHT_ARTIFACT_KEYS = (
    "timestamp",
    "load_average_1m",
    "load_per_physical_core",
    "gpu_compute_processes",
    "gpu_samples",
)
_PREFLIGHT_GPU_SAMPLE_KEYS = (
    "utilization_percent",
    "memory_used_mib",
    "temperature_c",
    "pstate",
)
_CASE_KEYS = (
    "case_id",
    "lane",
    "repeat_index",
    "seed",
    "batch_size",
    "dr_mode",
    "reset_density",
    "process",
    "raw",
    "summary",
)
_PROCESS_KEYS = (
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
)
_ENV_RAW_KEYS = (
    "timing_records",
    "memory",
    "resolved_env_config",
    "resolved_config_sha256",
)
_DR_RAW_KEYS = (
    "reset_samples",
    "memory",
    "resolved_env_config",
    "resolved_config_sha256",
)
_PPO_RAW_KEYS = (
    "scalars",
    "memory_samples",
    "run_config",
    "run_config_sha256",
    "run_summary",
)


def expected_case_ids(plan: G1BaselinePlan) -> set[str]:
    result = {
        f"env-b{batch_size}-r{repeat}"
        for batch_size in plan.env_lane.batch_sizes
        for repeat in range(plan.env_lane.process_repeats)
    }
    result.update(
        f"dr-{mode}-d{_density_id(density)}-r{repeat}"
        for mode in plan.dr_lane.modes
        for density in plan.dr_lane.reset_densities
        for repeat in range(plan.dr_lane.process_repeats)
    )
    result.update(f"ppo-seed-{seed}" for seed in plan.ppo_lane.seeds)
    return result


def _density_id(density: float) -> str:
    return f"{density:.4f}".replace(".", "p")


def build_aggregates(cases: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    lanes: dict[str, dict[str, Any]] = {"env": {}, "dr": {}, "ppo": {}}
    for case in sorted(cases, key=lambda item: str(item["case_id"])):
        lanes[str(case["lane"])][str(case["case_id"])] = case["summary"]
    return {
        lane: {"case_count": len(summaries), "cases": summaries}
        for lane, summaries in lanes.items()
    }


def validate_g1_baseline_artifact(
    raw: Any,
    plan: G1BaselinePlan,
    *,
    source: Path = Path("<memory>"),
    repo_root: Path | None = None,
) -> tuple[str, ...]:
    parser = _Parser()
    root = parser.mapping(raw, "artifact", _ARTIFACT_ROOT_KEYS)
    plan_raw = parser.mapping(root.get("plan"), "artifact.plan", _ARTIFACT_PLAN_KEYS)
    source_raw = parser.mapping(root.get("source"), "artifact.source", _SOURCE_KEYS)
    hardware = parser.mapping(root.get("hardware"), "artifact.hardware", _HARDWARE_KEYS)
    execution = parser.mapping(root.get("execution"), "artifact.execution", _EXECUTION_KEYS)
    cases_raw = root.get("cases")
    if not isinstance(cases_raw, list) or not cases_raw:
        parser.errors.append("artifact.cases: expected non-empty list")
        cases_raw = []

    if root.get("schema_version") != SCHEMA_VERSION:
        parser.errors.append(f"schema_version: expected {SCHEMA_VERSION}")
    if root.get("issue") != ISSUE:
        parser.errors.append(f"issue: expected {ISSUE}")
    if root.get("baseline_id") != BASELINE_ID:
        parser.errors.append(f"baseline_id: expected {BASELINE_ID!r}")
    parser.string(root.get("generated_at"), "generated_at")
    if plan_raw.get("path") != plan.source_path.as_posix():
        parser.errors.append("artifact.plan.path: does not match loaded plan path")
    _validate_sha(plan_raw.get("sha256"), "artifact.plan.sha256", parser.errors)
    _validate_sha(source_raw.get("tree_sha256"), "artifact.source.tree_sha256", parser.errors)
    _validate_sha(source_raw.get("uv_lock_sha256"), "artifact.source.uv_lock_sha256", parser.errors)
    _validate_sha(
        source_raw.get("owner_yaml_sha256"),
        "artifact.source.owner_yaml_sha256",
        parser.errors,
    )
    commit = source_raw.get("commit")
    if not isinstance(commit, str) or _COMMIT_RE.fullmatch(commit) is None:
        parser.errors.append("artifact.source.commit: expected full 40-character SHA")
    if source_raw.get("dirty") is not False:
        parser.errors.append("artifact.source.dirty: baseline must start from a clean tree")

    _validate_hardware(hardware, plan, parser.errors)
    if execution.get("process_isolation") is not True:
        parser.errors.append("artifact.execution.process_isolation: must be true")
    if execution.get("affinity_cpus") != list(plan.hardware.affinity_cpus):
        parser.errors.append("artifact.execution.affinity_cpus: does not match plan")
    if execution.get("env_vars") != dict(plan.environment.env_vars):
        parser.errors.append("artifact.execution.env_vars: does not match plan")
    if execution.get("hydra_overrides") != list(plan.environment.hydra_overrides):
        parser.errors.append("artifact.execution.hydra_overrides: does not match plan")
    _validate_preflight(execution.get("preflight"), plan, parser.errors)

    case_ids: list[str] = []
    run_ids: list[str] = []
    parsed_cases: list[dict[str, Any]] = []
    for index, case_raw in enumerate(cases_raw):
        case = parser.mapping(case_raw, f"cases[{index}]", _CASE_KEYS)
        process = parser.mapping(case.get("process"), f"cases[{index}].process", _PROCESS_KEYS)
        case_id = parser.string(case.get("case_id"), f"cases[{index}].case_id")
        case_ids.append(case_id)
        run_id = parser.string(process.get("run_id"), f"cases[{index}].process.run_id")
        run_ids.append(run_id)
        _validate_process(process, plan, f"cases[{index}].process", parser.errors)
        _validate_case(case, plan, f"cases[{index}]", parser.errors)
        parsed_cases.append(case)

    duplicate_case_ids = sorted({item for item in case_ids if case_ids.count(item) > 1})
    if duplicate_case_ids:
        parser.errors.append(f"artifact.cases: duplicate case IDs {duplicate_case_ids!r}")
    duplicate_run_ids = sorted({item for item in run_ids if run_ids.count(item) > 1})
    if duplicate_run_ids:
        parser.errors.append(f"artifact.cases: duplicate process run IDs {duplicate_run_ids!r}")
    expected = expected_case_ids(plan)
    actual = set(case_ids)
    if actual != expected:
        parser.errors.append(
            "artifact.cases: matrix mismatch; "
            f"missing={sorted(expected - actual)!r}, extra={sorted(actual - expected)!r}"
        )
    aggregates = root.get("aggregates")
    expected_aggregates = build_aggregates(parsed_cases) if parsed_cases else None
    if expected_aggregates is not None and not _equivalent(aggregates, expected_aggregates):
        parser.errors.append("artifact.aggregates: does not recompute from raw case summaries")

    if repo_root is not None:
        _validate_current_source(root, plan, repo_root, parser.errors)
    return tuple(parser.errors)


def load_g1_baseline_artifact(
    path: Path, plan: G1BaselinePlan, *, repo_root: Path | None = None
) -> dict[str, Any]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001 - normalize malformed artifact errors
        raise BaselineValidationError(
            path, [f"cannot load JSON: {type(exc).__name__}: {exc}"]
        ) from exc
    errors = validate_g1_baseline_artifact(raw, plan, source=path, repo_root=repo_root)
    if errors:
        raise BaselineValidationError(path, errors)
    return cast(dict[str, Any], raw)


def _validate_current_source(
    root: Mapping[str, Any],
    plan: G1BaselinePlan,
    repo_root: Path,
    errors: list[str],
) -> None:
    plan_payload = root.get("plan", {})
    source_payload = root.get("source", {})
    checks = {
        "artifact.plan.sha256": (
            plan_payload.get("sha256"),
            sha256_file(repo_root / plan.source_path),
        ),
        "artifact.source.tree_sha256": (
            source_payload.get("tree_sha256"),
            source_tree_sha256(repo_root, plan.source_inputs),
        ),
        "artifact.source.uv_lock_sha256": (
            source_payload.get("uv_lock_sha256"),
            sha256_file(repo_root / "uv.lock"),
        ),
        "artifact.source.owner_yaml_sha256": (
            source_payload.get("owner_yaml_sha256"),
            sha256_file(repo_root / plan.owner_yaml),
        ),
    }
    for path, (actual, expected) in checks.items():
        if actual != expected:
            errors.append(f"{path}: stale; expected current fingerprint {expected!r}")


def _validate_hardware(
    hardware: Mapping[str, Any], plan: G1BaselinePlan, errors: list[str]
) -> None:
    expected = {
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
    for key, value in expected.items():
        if hardware.get(key) != value:
            errors.append(
                f"artifact.hardware.{key}: expected frozen value {value!r}, "
                f"got {hardware.get(key)!r}"
            )
    for key in ("platform_release", "cuda_runtime", "torch_version", "hostname"):
        if not isinstance(hardware.get(key), str) or not hardware.get(key):
            errors.append(f"artifact.hardware.{key}: expected non-empty string")


def _validate_process(
    process: Mapping[str, Any], plan: G1BaselinePlan, path: str, errors: list[str]
) -> None:
    if isinstance(process.get("pid"), bool) or not isinstance(process.get("pid"), int):
        errors.append(f"{path}.pid: expected integer")
    run_id = process.get("run_id")
    try:
        uuid.UUID(str(run_id))
    except (ValueError, AttributeError):
        errors.append(f"{path}.run_id: expected UUID")
    if process.get("return_code") != 0:
        errors.append(f"{path}.return_code: expected 0")
    duration = process.get("duration_sec")
    if isinstance(duration, bool) or not isinstance(duration, (int, float)) or duration <= 0:
        errors.append(f"{path}.duration_sec: expected positive number")
    command = process.get("command")
    if (
        not isinstance(command, list)
        or not command
        or not all(isinstance(item, str) and item for item in command)
    ):
        errors.append(f"{path}.command: expected non-empty argv list")
    elif command[:2] != ["uv", "run"]:
        errors.append(f"{path}.command: benchmark subprocess must use `uv run`")
    if process.get("affinity_cpus") != list(plan.hardware.affinity_cpus):
        errors.append(f"{path}.affinity_cpus: does not match plan")
    if process.get("env_vars") != dict(plan.environment.env_vars):
        errors.append(f"{path}.env_vars: does not match plan")
    for key in ("stdout_sha256", "stderr_sha256"):
        _validate_sha(process.get(key), f"{path}.{key}", errors)


def _validate_preflight(value: Any, plan: G1BaselinePlan, errors: list[str]) -> None:
    if not isinstance(value, dict):
        errors.append("artifact.execution.preflight: expected mapping")
        return
    _validate_exact_keys(value, _PREFLIGHT_ARTIFACT_KEYS, "artifact.execution.preflight", errors)
    load = value.get("load_per_physical_core")
    if isinstance(load, bool) or not isinstance(load, (int, float)):
        errors.append("artifact.execution.preflight.load_per_physical_core: expected number")
    elif load > plan.preflight.max_load_per_physical_core:
        errors.append("artifact.execution.preflight: CPU load exceeds frozen limit")
    processes = value.get("gpu_compute_processes")
    if not isinstance(processes, list):
        errors.append("artifact.execution.preflight.gpu_compute_processes: expected list")
    elif len(processes) > plan.preflight.max_gpu_compute_processes:
        errors.append("artifact.execution.preflight: foreign GPU compute process detected")
    else:
        for index, process in enumerate(processes):
            if not isinstance(process, dict):
                errors.append(
                    f"artifact.execution.preflight.gpu_compute_processes[{index}]: expected mapping"
                )
                continue
            _validate_exact_keys(
                process,
                ("pid", "process_name", "used_memory_mib"),
                f"artifact.execution.preflight.gpu_compute_processes[{index}]",
                errors,
            )
    samples = value.get("gpu_samples")
    if not isinstance(samples, list) or len(samples) != plan.preflight.gpu_samples:
        errors.append(
            f"artifact.execution.preflight.gpu_samples: expected {plan.preflight.gpu_samples} samples"
        )
    elif any(
        not isinstance(sample, dict)
        or sample.get("utilization_percent", plan.preflight.max_gpu_utilization_percent + 1)
        > plan.preflight.max_gpu_utilization_percent
        for sample in samples
    ):
        errors.append("artifact.execution.preflight: GPU utilization exceeds frozen limit")
    if isinstance(samples, list):
        for index, sample in enumerate(samples):
            if not isinstance(sample, dict):
                continue
            _validate_exact_keys(
                sample,
                _PREFLIGHT_GPU_SAMPLE_KEYS,
                f"artifact.execution.preflight.gpu_samples[{index}]",
                errors,
            )


def _validate_case(
    case: Mapping[str, Any], plan: G1BaselinePlan, path: str, errors: list[str]
) -> None:
    lane = case.get("lane")
    raw = case.get("raw")
    summary = case.get("summary")
    if not isinstance(raw, dict):
        errors.append(f"{path}.raw: expected mapping")
        return
    try:
        if lane == "env":
            batch_size = case.get("batch_size")
            if isinstance(batch_size, bool) or not isinstance(batch_size, int):
                errors.append(f"{path}.batch_size: expected integer")
                return
            expected_summary = summarize_env_raw(raw, batch_size)
            _validate_env_case(case, raw, plan, path, errors)
        elif lane == "dr":
            expected_summary = summarize_dr_raw(raw)
            _validate_dr_case(case, raw, plan, path, errors)
        elif lane == "ppo":
            expected_summary = summarize_ppo_raw(raw, plan.ppo_lane)
            _validate_ppo_case(case, raw, plan, path, errors)
        else:
            errors.append(f"{path}.lane: expected env, dr, or ppo")
            return
    except (TypeError, ValueError) as exc:
        errors.append(f"{path}.raw: {exc}")
        return
    if not _equivalent(summary, expected_summary):
        errors.append(f"{path}.summary: does not recompute from raw samples")


def _validate_env_case(
    case: Mapping[str, Any],
    raw: Mapping[str, Any],
    plan: G1BaselinePlan,
    path: str,
    errors: list[str],
) -> None:
    _validate_exact_keys(raw, _ENV_RAW_KEYS, f"{path}.raw", errors)
    batch_size = case.get("batch_size")
    repeat = case.get("repeat_index")
    expected_id = f"env-b{batch_size}-r{repeat}"
    if case.get("case_id") != expected_id:
        errors.append(f"{path}.case_id: expected {expected_id!r}")
    if batch_size not in plan.env_lane.batch_sizes:
        errors.append(f"{path}.batch_size: not in frozen matrix")
    if not isinstance(repeat, int) or repeat not in range(plan.env_lane.process_repeats):
        errors.append(f"{path}.repeat_index: out of range")
    for key in ("seed", "dr_mode", "reset_density"):
        if case.get(key) is not None:
            errors.append(f"{path}.{key}: must be null for env lane")
    timing = raw.get("timing_records", {})
    if isinstance(timing, dict):
        for key, values in timing.items():
            if not isinstance(values, list) or len(values) != plan.env_lane.measure_steps:
                errors.append(
                    f"{path}.raw.timing_records.{key}: expected "
                    f"{plan.env_lane.measure_steps} samples"
                )
    _validate_config_payload(raw, path, errors)


def _validate_dr_case(
    case: Mapping[str, Any],
    raw: Mapping[str, Any],
    plan: G1BaselinePlan,
    path: str,
    errors: list[str],
) -> None:
    _validate_exact_keys(raw, _DR_RAW_KEYS, f"{path}.raw", errors)
    mode = case.get("dr_mode")
    density = case.get("reset_density")
    repeat = case.get("repeat_index")
    if isinstance(density, bool) or not isinstance(density, (int, float)):
        errors.append(f"{path}.reset_density: expected number")
        return
    density_value = float(density)
    expected_id = f"dr-{mode}-d{_density_id(density_value)}-r{repeat}"
    if case.get("case_id") != expected_id:
        errors.append(f"{path}.case_id: expected {expected_id!r}")
    if case.get("batch_size") != plan.dr_lane.num_envs:
        errors.append(f"{path}.batch_size: expected DR num_envs")
    if mode not in plan.dr_lane.modes or density_value not in plan.dr_lane.reset_densities:
        errors.append(f"{path}: DR mode or density is outside frozen matrix")
    if case.get("seed") is not None:
        errors.append(f"{path}.seed: must be null for DR lane")
    samples = raw.get("reset_samples")
    if isinstance(samples, list):
        if len(samples) != plan.dr_lane.measure_resets:
            errors.append(
                f"{path}.raw.reset_samples: expected {plan.dr_lane.measure_resets} samples"
            )
        expected_rows = max(1, round(plan.dr_lane.num_envs * density_value))
        for index, sample in enumerate(samples):
            if not isinstance(sample, dict):
                continue
            _validate_exact_keys(
                sample,
                ("requested_rows", "actual_rows", "timing"),
                f"{path}.raw.reset_samples[{index}]",
                errors,
            )
            if sample.get("requested_rows") != expected_rows:
                errors.append(f"{path}.raw.reset_samples[{index}]: wrong requested row count")
            if sample.get("actual_rows") != expected_rows:
                errors.append(f"{path}.raw.reset_samples[{index}]: row isolation failed")
    _validate_config_payload(raw, path, errors)


def _validate_ppo_case(
    case: Mapping[str, Any],
    raw: Mapping[str, Any],
    plan: G1BaselinePlan,
    path: str,
    errors: list[str],
) -> None:
    _validate_exact_keys(raw, _PPO_RAW_KEYS, f"{path}.raw", errors)
    seed = case.get("seed")
    expected_id = f"ppo-seed-{seed}"
    if case.get("case_id") != expected_id:
        errors.append(f"{path}.case_id: expected {expected_id!r}")
    if seed not in plan.ppo_lane.seeds:
        errors.append(f"{path}.seed: not in frozen matrix")
    if case.get("batch_size") != plan.ppo_lane.num_envs:
        errors.append(f"{path}.batch_size: expected PPO num_envs")
    for key in ("repeat_index", "dr_mode", "reset_density"):
        if case.get(key) is not None:
            errors.append(f"{path}.{key}: must be null for PPO lane")
    run_config = raw.get("run_config")
    run_summary = raw.get("run_summary")
    if not isinstance(run_config, dict) or not isinstance(run_summary, dict):
        errors.append(f"{path}.raw: run_config and run_summary must be mappings")
        return
    if raw.get("run_config_sha256") != canonical_sha256(run_config):
        errors.append(f"{path}.raw.run_config_sha256: does not match run_config")
    config = run_config.get("config", {})
    checks = {
        "training.sim_backend": _nested(config, "training", "sim_backend"),
        "algo.seed": _nested(config, "algo", "seed"),
        "algo.num_envs": _nested(config, "algo", "num_envs"),
        "algo.num_steps_per_env": _nested(config, "algo", "num_steps_per_env"),
        "algo.max_iterations": _nested(config, "algo", "max_iterations"),
    }
    expected = {
        "training.sim_backend": BACKEND,
        "algo.seed": seed,
        "algo.num_envs": plan.ppo_lane.num_envs,
        "algo.num_steps_per_env": plan.ppo_lane.num_steps_per_env,
        "algo.max_iterations": plan.ppo_lane.max_iterations,
    }
    for key, actual in checks.items():
        if actual != expected[key]:
            errors.append(f"{path}.raw.run_config.{key}: expected {expected[key]!r}")
    if run_summary.get("status") != "completed":
        errors.append(f"{path}.raw.run_summary.status: expected completed")
    if run_summary.get("completed_iterations") != plan.ppo_lane.max_iterations:
        errors.append(f"{path}.raw.run_summary.completed_iterations: incomplete training run")
    scalars = raw.get("scalars")
    if isinstance(scalars, dict):
        for tag, points in scalars.items():
            if not isinstance(points, list):
                continue
            steps: list[int] = []
            for index, point in enumerate(points):
                if not isinstance(point, dict):
                    errors.append(f"{path}.raw.scalars.{tag}[{index}]: expected mapping")
                    continue
                _validate_exact_keys(
                    point,
                    ("step", "wall_time", "value"),
                    f"{path}.raw.scalars.{tag}[{index}]",
                    errors,
                )
                if isinstance(point.get("step"), int):
                    steps.append(int(point["step"]))
            if steps != sorted(set(steps)):
                errors.append(f"{path}.raw.scalars.{tag}: steps must be unique and ordered")
    memory_samples = raw.get("memory_samples")
    if isinstance(memory_samples, list):
        for index, sample in enumerate(memory_samples):
            if not isinstance(sample, dict):
                errors.append(f"{path}.raw.memory_samples[{index}]: expected mapping")
                continue
            _validate_exact_keys(
                sample,
                ("elapsed_sec", "rss_bytes"),
                f"{path}.raw.memory_samples[{index}]",
                errors,
            )


def _validate_config_payload(raw: Mapping[str, Any], path: str, errors: list[str]) -> None:
    config = raw.get("resolved_env_config")
    if not isinstance(config, dict):
        errors.append(f"{path}.raw.resolved_env_config: expected mapping")
    elif raw.get("resolved_config_sha256") != canonical_sha256(config):
        errors.append(f"{path}.raw.resolved_config_sha256: does not match config")


def _nested(value: Any, *keys: str) -> Any:
    current = value
    for key in keys:
        if not isinstance(current, dict):
            return _MISSING
        current = current.get(key, _MISSING)
    return current


def _validate_exact_keys(
    value: Mapping[str, Any], expected_keys: Sequence[str], path: str, errors: list[str]
) -> None:
    expected = set(expected_keys)
    actual = set(value)
    for key in sorted(expected - actual):
        errors.append(f"{path}: missing key `{key}`")
    for key in sorted(actual - expected):
        errors.append(f"{path}: unknown key `{key}`")


def _validate_sha(value: Any, path: str, errors: list[str]) -> None:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        errors.append(f"{path}: expected sha256:<64 lowercase hex>")


def _equivalent(left: Any, right: Any) -> bool:
    if isinstance(left, dict) and isinstance(right, dict):
        return set(left) == set(right) and all(_equivalent(left[key], right[key]) for key in left)
    if isinstance(left, list) and isinstance(right, list):
        return len(left) == len(right) and all(
            _equivalent(left_item, right_item)
            for left_item, right_item in zip(left, right, strict=True)
        )
    if isinstance(left, bool) or isinstance(right, bool):
        return left is right
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return math.isclose(float(left), float(right), rel_tol=1e-12, abs_tol=1e-12)
    return cast(bool, left == right)


def assert_clean_affinity(plan: G1BaselinePlan) -> None:
    if not hasattr(os, "sched_getaffinity"):
        raise RuntimeError("Issue #705 baseline requires Linux CPU affinity support")
    available = set(os.sched_getaffinity(0))
    required = set(plan.hardware.affinity_cpus)
    if not required.issubset(available):
        raise RuntimeError(
            f"planned CPUs are unavailable: missing={sorted(required - available)!r}, "
            f"available={sorted(available)!r}"
        )
