#!/usr/bin/env python3
"""Benchmark SAC G1 replay-buffer sampling placement.

This standalone benchmark composes the same off-policy owner config used by:

    uv run train --algo sac --task g1_walk_flat --sim mujoco

It does not run training.  It measures only the replay sampling boundary:

* CPU replay storage: random sample into a host batch, then host-to-device copy.
* GPU replay storage: random sample directly from a device-resident replay tensor.

The default matrix tests 1/2/4/8 visible CUDA devices and several replay
capacities derived from the configured replay size.  GPU counts that are not
available are recorded as skipped.

Usage:
    uv run benchmark/benchmark_sac_replay_buffer_sampling.py
    uv run benchmark/benchmark_sac_replay_buffer_sampling.py --gpu-counts 1,2
    uv run benchmark/benchmark_sac_replay_buffer_sampling.py --capacity-multipliers 0.25,0.5,1,2
    uv run benchmark/benchmark_sac_replay_buffer_sampling.py --warmup 3 --repeat 10
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, median, pstdev
from typing import Any, Callable, Iterable, cast

import torch
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig, OmegaConf

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from benchmark.core.device_info import get_device_info_dict, get_device_info_line

plt: Any | None = None
try:
    import matplotlib as _matplotlib

    _matplotlib.use("Agg")
    import matplotlib.pyplot as _plt

    plt = _plt
except Exception:
    plt = None

DEFAULT_OUTPUT_JSON = (
    ROOT_DIR / "benchmark" / "outputs" / "sac_replay_buffer_sampling" / "results.json"
)
DEFAULT_SIM = "mujoco"
DEFAULT_GPU_COUNTS = "1,2,4,8"
DEFAULT_CAPACITY_MULTIPLIERS = "0.25,0.5,1.0"
FLOAT_BYTES = 4


@dataclass(frozen=True)
class ReplayShape:
    obs_dim: int
    action_dim: int
    critic_dim: int

    @property
    def packed_width(self) -> int:
        return 2 * self.obs_dim + self.action_dim + 3 + 2 * self.critic_dim


@dataclass(frozen=True)
class BenchmarkCase:
    algo: str
    task: str
    sim: str
    command: str
    training_task_name: str
    num_envs: int
    replay_buffer_n: int
    config_capacity_rows: int
    configured_batch_size: int
    learner_batch_size: int
    symmetry_batch_multiplier: int
    updates_per_step: int
    sample_count_per_rank: int
    learning_starts: int
    shape: ReplayShape

    @property
    def sample_bytes_per_rank(self) -> int:
        return self.sample_count_per_rank * self.shape.packed_width * FLOAT_BYTES


@dataclass
class TimingStats:
    samples_ms: list[float]
    mean_ms: float
    median_ms: float
    std_ms: float
    min_ms: float
    max_ms: float
    warmup: int
    repeat: int


@dataclass
class CapacityResult:
    world_size: int
    capacity_rows: int
    torch_threads: int
    replay_bytes_per_rank: int
    sample_bytes_per_rank: int
    global_sample_bytes: int
    timings: dict[str, TimingStats]
    notes: list[str]


@dataclass
class SkippedCase:
    world_size: int
    capacity_rows: int | None
    reason: str


def _fmt_bytes(num_bytes: int) -> str:
    value = float(num_bytes)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if value < 1024.0 or unit == "TiB":
            return f"{value:.2f} {unit}" if unit != "B" else f"{int(value)} B"
        value /= 1024.0
    return f"{value:.2f} TiB"


def _stats(samples_ms: list[float], *, warmup: int, repeat: int) -> TimingStats:
    if not samples_ms:
        raise ValueError("no timing samples collected")
    return TimingStats(
        samples_ms=samples_ms,
        mean_ms=mean(samples_ms),
        median_ms=median(samples_ms),
        std_ms=pstdev(samples_ms) if len(samples_ms) > 1 else 0.0,
        min_ms=min(samples_ms),
        max_ms=max(samples_ms),
        warmup=warmup,
        repeat=repeat,
    )


def _measure_ms(
    fn: Callable[[int], None],
    *,
    warmup: int,
    repeat: int,
    sync_before: Callable[[], None] | None = None,
    sync_after: Callable[[], None] | None = None,
) -> TimingStats:
    samples: list[float] = []
    for idx in range(warmup + repeat):
        if sync_before is not None:
            sync_before()
        start_ns = time.perf_counter_ns()
        fn(idx)
        if sync_after is not None:
            sync_after()
        elapsed_ms = (time.perf_counter_ns() - start_ns) / 1e6
        if idx >= warmup:
            samples.append(elapsed_ms)
    return _stats(samples, warmup=warmup, repeat=repeat)


def _sync_devices(devices: Iterable[torch.device]) -> None:
    seen: set[str] = set()
    for device in devices:
        key = str(device)
        if key in seen:
            continue
        seen.add(key)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elif device.type == "xpu":
            xpu = getattr(torch, "xpu", None)
            synchronize = getattr(xpu, "synchronize", None)
            if callable(synchronize):
                try:
                    synchronize(device)
                except TypeError:
                    synchronize()
        elif device.type == "mps" and hasattr(torch, "mps"):
            torch.mps.synchronize()


def _cleanup_device() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if hasattr(torch, "mps") and torch.backends.mps.is_available():
        torch.mps.empty_cache()


def _compose_offpolicy_cfg(sim: str = DEFAULT_SIM) -> DictConfig:
    config_dir = str(ROOT_DIR / "conf" / "offpolicy")
    overrides = [
        "algo=sac",
        f"task=sac/g1_walk_flat/{sim}",
        "hydra.run.dir=.",
        "hydra.output_subdir=null",
        "hydra/job_logging=disabled",
        "hydra/hydra_logging=disabled",
    ]
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=config_dir, version_base="1.3"):
        return compose(config_name="config", overrides=overrides)


def _resolve_env_shape_and_symmetry(cfg: DictConfig) -> tuple[ReplayShape, int]:
    from unilab.base.observations import get_obs_dims
    from unilab.training import BackendAdapter, create_env, ensure_registries

    ensure_registries()
    env_cfg_override = BackendAdapter(
        cfg,
        root_dir=ROOT_DIR,
        algo_name="sac",
    ).build_task_env_cfg_override()
    env = create_env(cfg, num_envs=1, env_cfg_override=env_cfg_override)
    try:
        obs_dim, critic_dim = get_obs_dims(env.obs_groups_spec)
        action_shape = env.action_space.shape
        if action_shape is None:
            raise ValueError("env.action_space.shape must be defined")
        action_dim = int(action_shape[0])

        symmetry_batch_multiplier = 1
        if bool(OmegaConf.select(cfg, "algo.use_symmetry", default=False)):
            symmetry_builder = getattr(env, "build_symmetry_augmentation", None)
            if not callable(symmetry_builder):
                raise ValueError(f"{cfg.training.task_name} does not provide symmetry augmentation")
            symmetry = cast(Any, symmetry_builder(device="cpu"))
            if symmetry is None:
                raise ValueError(f"{cfg.training.task_name} does not provide symmetry augmentation")
            symmetry_batch_multiplier = int(symmetry.batch_multiplier)
    finally:
        env.close()

    return (
        ReplayShape(obs_dim=int(obs_dim), action_dim=action_dim, critic_dim=int(critic_dim)),
        symmetry_batch_multiplier,
    )


def _build_case(
    cfg: DictConfig,
    *,
    sim: str,
    shape: ReplayShape,
    symmetry_batch_multiplier: int,
) -> BenchmarkCase:
    num_envs = int(cfg.algo.num_envs)
    replay_buffer_n = int(cfg.algo.replay_buffer_n)
    configured_batch_size = int(cfg.algo.batch_size)
    learner_batch_size = configured_batch_size
    if bool(OmegaConf.select(cfg, "algo.use_symmetry", default=False)):
        if configured_batch_size % symmetry_batch_multiplier != 0:
            raise ValueError(
                "SAC symmetry requires batch_size divisible by "
                f"{symmetry_batch_multiplier}, got {configured_batch_size}"
            )
        learner_batch_size = configured_batch_size // symmetry_batch_multiplier

    updates_per_step = int(cfg.algo.updates_per_step)
    return BenchmarkCase(
        algo="sac",
        task="g1_walk_flat",
        sim=sim,
        command=f"uv run train --algo sac --task g1_walk_flat --sim {sim}",
        training_task_name=str(cfg.training.task_name),
        num_envs=num_envs,
        replay_buffer_n=replay_buffer_n,
        config_capacity_rows=num_envs * replay_buffer_n,
        configured_batch_size=configured_batch_size,
        learner_batch_size=learner_batch_size,
        symmetry_batch_multiplier=int(symmetry_batch_multiplier),
        updates_per_step=updates_per_step,
        sample_count_per_rank=learner_batch_size * updates_per_step,
        learning_starts=int(cfg.algo.learning_starts),
        shape=shape,
    )


def _parse_int_list(value: str, *, name: str) -> list[int]:
    parsed: list[int] = []
    for part in value.split(","):
        item = part.strip()
        if not item:
            continue
        number = int(item)
        if number <= 0:
            raise ValueError(f"{name} values must be positive, got {number}")
        if number not in parsed:
            parsed.append(number)
    if not parsed:
        raise ValueError(f"{name} must contain at least one positive integer")
    return parsed


def _parse_float_list(value: str, *, name: str) -> list[float]:
    parsed: list[float] = []
    for part in value.split(","):
        item = part.strip()
        if not item:
            continue
        number = float(item)
        if number <= 0.0:
            raise ValueError(f"{name} values must be positive, got {number}")
        if number not in parsed:
            parsed.append(number)
    if not parsed:
        raise ValueError(f"{name} must contain at least one positive value")
    return parsed


def _resolve_capacity_rows(
    *,
    config_capacity_rows: int,
    capacity_rows_arg: str,
    capacity_multipliers_arg: str,
    capacity_exponents_arg: str | None = None,
) -> list[int]:
    if capacity_exponents_arg is not None:
        return [
            2**exponent
            for exponent in _parse_int_list(capacity_exponents_arg, name="capacity exponents")
        ]

    if capacity_rows_arg.strip().lower() != "auto":
        return _parse_int_list(capacity_rows_arg, name="capacity rows")

    capacities: list[int] = []
    for multiplier in _parse_float_list(capacity_multipliers_arg, name="capacity multipliers"):
        rows = max(1, int(round(config_capacity_rows * multiplier)))
        if rows not in capacities:
            capacities.append(rows)
    return capacities


def _parse_device_ids(value: str) -> list[int]:
    if value.strip().lower() == "auto":
        return list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
    return _parse_int_list(value, name="device ids")


def _resolve_torch_threads(setting: str, *, world_size: int) -> int:
    if setting != "auto":
        threads = int(setting)
        if threads <= 0:
            raise ValueError("--torch-threads must be 'auto' or a positive integer")
        return threads
    cpu_count = os.cpu_count() or 1
    return max(1, cpu_count // max(int(world_size), 1))


def _allocate_cpu_tensor(
    *,
    rows: int,
    width: int,
    pin_memory: bool,
    prefill: str,
    notes: list[str],
) -> torch.Tensor:
    try:
        tensor = torch.empty((rows, width), dtype=torch.float32, pin_memory=pin_memory)
    except RuntimeError as exc:
        if not pin_memory:
            raise
        notes.append(f"pinned CPU allocation failed; fell back to pageable CPU: {exc}")
        tensor = torch.empty((rows, width), dtype=torch.float32)

    if prefill == "zeros":
        tensor.zero_()
    elif prefill != "none":
        raise ValueError(f"Unsupported prefill={prefill!r}")
    return tensor


def _allocate_device_tensor(
    *,
    rows: int,
    width: int,
    device: torch.device,
    prefill: str,
) -> torch.Tensor:
    tensor = torch.empty((rows, width), dtype=torch.float32, device=device)
    if prefill == "zeros":
        tensor.zero_()
        _sync_devices([device])
    elif prefill != "none":
        raise ValueError(f"Unsupported prefill={prefill!r}")
    return tensor


def _make_cpu_generators(world_size: int, seed: int) -> list[torch.Generator]:
    generators = []
    for rank in range(world_size):
        gen = torch.Generator(device="cpu")
        gen.manual_seed(int(seed) + rank * 1009)
        generators.append(gen)
    return generators


def _make_device_generators(devices: list[torch.device], seed: int) -> list[torch.Generator]:
    generators = []
    for rank, device in enumerate(devices):
        gen = torch.Generator(device=device)
        gen.manual_seed(int(seed) + rank * 1009)
        generators.append(gen)
    return generators


def _run_parallel(
    executor: ThreadPoolExecutor | None,
    world_size: int,
    worker: Callable[[int], None],
) -> None:
    if world_size == 1:
        worker(0)
        return
    assert executor is not None
    futures = [executor.submit(worker, rank) for rank in range(world_size)]
    for future in futures:
        future.result()


def _bench_cpu_sample_and_h2d(
    case: BenchmarkCase,
    *,
    capacity_rows: int,
    devices: list[torch.device],
    warmup: int,
    repeat: int,
    prefill: str,
    pinned_host_batch: bool,
    index_mode: str,
    seed: int,
    notes: list[str],
) -> dict[str, TimingStats]:
    world_size = len(devices)
    width = case.shape.packed_width
    sample_count = case.sample_count_per_rank
    storage = _allocate_cpu_tensor(
        rows=capacity_rows,
        width=width,
        pin_memory=False,
        prefill=prefill,
        notes=notes,
    )
    host_batches = [
        _allocate_cpu_tensor(
            rows=sample_count,
            width=width,
            pin_memory=pinned_host_batch and any(d.type == "cuda" for d in devices),
            prefill="none",
            notes=notes,
        )
        for _ in range(world_size)
    ]
    device_batches = [
        _allocate_device_tensor(rows=sample_count, width=width, device=device, prefill="none")
        for device in devices
    ]
    cpu_indices = [torch.empty(sample_count, dtype=torch.int64) for _ in range(world_size)]
    cpu_generators = _make_cpu_generators(world_size, seed)
    if index_mode == "pregenerated":
        for rank in range(world_size):
            torch.randint(
                capacity_rows,
                (sample_count,),
                generator=cpu_generators[rank],
                out=cpu_indices[rank],
            )
    elif index_mode != "timed":
        raise ValueError(f"Unsupported index_mode={index_mode!r}")

    streams: list[torch.cuda.Stream | None] = []
    for device in devices:
        if device.type == "cuda":
            with torch.cuda.device(device):
                streams.append(torch.cuda.Stream(device=device))
        else:
            streams.append(None)

    executor: ThreadPoolExecutor | None = None
    if world_size > 1:
        executor = ThreadPoolExecutor(max_workers=world_size, thread_name_prefix="cpu_replay_rank")

    def sample_worker(rank: int) -> None:
        indices = cpu_indices[rank]
        if index_mode == "timed":
            torch.randint(
                capacity_rows,
                (sample_count,),
                generator=cpu_generators[rank],
                out=indices,
            )
        torch.index_select(storage, 0, indices, out=host_batches[rank])

    def h2d_worker(rank: int) -> None:
        device = devices[rank]
        host = host_batches[rank]
        target = device_batches[rank]
        non_blocking = bool(host.is_pinned() and device.type == "cuda")
        if device.type == "cuda":
            stream = streams[rank]
            assert stream is not None
            with torch.cuda.device(device), torch.cuda.stream(stream):
                target.copy_(host, non_blocking=non_blocking)
        else:
            target.copy_(host)

    def sample_all(_: int) -> None:
        _run_parallel(executor, world_size, sample_worker)

    def h2d_all(_: int) -> None:
        _run_parallel(executor, world_size, h2d_worker)

    def sample_then_h2d_all(_: int) -> None:
        _run_parallel(executor, world_size, sample_worker)
        _run_parallel(executor, world_size, h2d_worker)

    def sync() -> None:
        _sync_devices(devices)

    try:
        sample_stats = _measure_ms(sample_all, warmup=warmup, repeat=repeat)
        h2d_stats = _measure_ms(
            h2d_all,
            warmup=warmup,
            repeat=repeat,
            sync_before=sync,
            sync_after=sync,
        )
        combined_stats = _measure_ms(
            sample_then_h2d_all,
            warmup=warmup,
            repeat=repeat,
            sync_before=sync,
            sync_after=sync,
        )
        return {
            "cpu_sample_wall": sample_stats,
            "cpu_sample_h2d_wall": h2d_stats,
            "cpu_sample_then_h2d_wall": combined_stats,
        }
    finally:
        if executor is not None:
            executor.shutdown(wait=True)
        del cpu_indices, device_batches, host_batches, storage
        _cleanup_device()


def _bench_gpu_sample(
    case: BenchmarkCase,
    *,
    capacity_rows: int,
    devices: list[torch.device],
    warmup: int,
    repeat: int,
    prefill: str,
    index_mode: str,
    seed: int,
) -> TimingStats:
    world_size = len(devices)
    width = case.shape.packed_width
    sample_count = case.sample_count_per_rank
    storages = [
        _allocate_device_tensor(rows=capacity_rows, width=width, device=device, prefill=prefill)
        for device in devices
    ]
    outputs = [
        _allocate_device_tensor(rows=sample_count, width=width, device=device, prefill="none")
        for device in devices
    ]
    indices = [torch.empty(sample_count, dtype=torch.int64, device=device) for device in devices]
    generators = _make_device_generators(devices, seed)
    if index_mode == "pregenerated":
        for rank, device in enumerate(devices):
            with torch.cuda.device(device) if device.type == "cuda" else _nullcontext():
                torch.randint(
                    capacity_rows,
                    (sample_count,),
                    generator=generators[rank],
                    device=device,
                    out=indices[rank],
                )
        _sync_devices(devices)
    elif index_mode != "timed":
        raise ValueError(f"Unsupported index_mode={index_mode!r}")

    streams: list[torch.cuda.Stream | None] = []
    for device in devices:
        if device.type == "cuda":
            with torch.cuda.device(device):
                streams.append(torch.cuda.Stream(device=device))
        else:
            streams.append(None)

    executor: ThreadPoolExecutor | None = None
    if world_size > 1:
        executor = ThreadPoolExecutor(max_workers=world_size, thread_name_prefix="gpu_replay_rank")

    def sample_worker(rank: int) -> None:
        device = devices[rank]
        if device.type == "cuda":
            stream = streams[rank]
            assert stream is not None
            with torch.cuda.device(device), torch.cuda.stream(stream):
                if index_mode == "timed":
                    torch.randint(
                        capacity_rows,
                        (sample_count,),
                        generator=generators[rank],
                        device=device,
                        out=indices[rank],
                    )
                torch.index_select(storages[rank], 0, indices[rank], out=outputs[rank])
        else:
            if index_mode == "timed":
                torch.randint(
                    capacity_rows,
                    (sample_count,),
                    generator=generators[rank],
                    device=device,
                    out=indices[rank],
                )
            torch.index_select(storages[rank], 0, indices[rank], out=outputs[rank])

    def sample_all(_: int) -> None:
        _run_parallel(executor, world_size, sample_worker)

    def sync() -> None:
        _sync_devices(devices)

    try:
        return _measure_ms(
            sample_all,
            warmup=warmup,
            repeat=repeat,
            sync_before=sync,
            sync_after=sync,
        )
    finally:
        if executor is not None:
            executor.shutdown(wait=True)
        del generators, indices, outputs, storages
        _cleanup_device()


class _nullcontext:
    def __enter__(self):
        return None

    def __exit__(self, exc_type, exc, tb):
        return False


def _estimate_device_bytes_per_gpu(case: BenchmarkCase, capacity_rows: int) -> int:
    width = case.shape.packed_width
    replay = capacity_rows * width * FLOAT_BYTES
    sample_batch = case.sample_count_per_rank * width * FLOAT_BYTES
    # GPU mode uses replay + output + indices; CPU/H2D mode uses a device batch.
    indices = case.sample_count_per_rank * 8
    return replay + (2 * sample_batch) + indices


def _cuda_memory_skip_reason(
    devices: list[torch.device],
    *,
    required_bytes_per_gpu: int,
    safety_factor: float,
) -> str | None:
    for device in devices:
        if device.type != "cuda":
            continue
        free_bytes, _total_bytes = torch.cuda.mem_get_info(device)
        needed = int(required_bytes_per_gpu * safety_factor)
        if free_bytes < needed:
            return (
                f"{device} free memory {_fmt_bytes(free_bytes)} < "
                f"estimated required {_fmt_bytes(needed)}"
            )
    return None


def _run_capacity_case(
    case: BenchmarkCase,
    *,
    capacity_rows: int,
    devices: list[torch.device],
    warmup: int,
    repeat: int,
    prefill: str,
    pinned_host_batch: bool,
    index_mode: str,
    seed: int,
    torch_threads: int,
) -> CapacityResult:
    notes: list[str] = []
    old_threads = torch.get_num_threads()
    torch.set_num_threads(torch_threads)
    try:
        cpu_timings = _bench_cpu_sample_and_h2d(
            case,
            capacity_rows=capacity_rows,
            devices=devices,
            warmup=warmup,
            repeat=repeat,
            prefill=prefill,
            pinned_host_batch=pinned_host_batch,
            index_mode=index_mode,
            seed=seed,
            notes=notes,
        )
        gpu_sample = _bench_gpu_sample(
            case,
            capacity_rows=capacity_rows,
            devices=devices,
            warmup=warmup,
            repeat=repeat,
            prefill=prefill,
            index_mode=index_mode,
            seed=seed + 100_000,
        )
    finally:
        torch.set_num_threads(old_threads)

    timings = dict(cpu_timings)
    timings["gpu_sample_wall"] = gpu_sample
    return CapacityResult(
        world_size=len(devices),
        capacity_rows=capacity_rows,
        torch_threads=torch_threads,
        replay_bytes_per_rank=capacity_rows * case.shape.packed_width * FLOAT_BYTES,
        sample_bytes_per_rank=case.sample_bytes_per_rank,
        global_sample_bytes=case.sample_bytes_per_rank * len(devices),
        timings=timings,
        notes=notes,
    )


def _print_timing(label: str, stat: TimingStats) -> None:
    print(
        f"    {label:<28}"
        f" mean={stat.mean_ms:8.3f} ms"
        f" median={stat.median_ms:8.3f} ms"
        f" std={stat.std_ms:7.3f} ms"
    )


def _print_result(result: CapacityResult) -> None:
    print(
        f"  {result.world_size} GPU(s), capacity={result.capacity_rows:,} "
        f"({_fmt_bytes(result.replay_bytes_per_rank)} per replay copy), "
        f"sample={_fmt_bytes(result.sample_bytes_per_rank)} per rank, "
        f"torch_threads={result.torch_threads}"
    )
    _print_timing("CPU sample", result.timings["cpu_sample_wall"])
    _print_timing("CPU sampled batch H2D", result.timings["cpu_sample_h2d_wall"])
    _print_timing("CPU sample then H2D", result.timings["cpu_sample_then_h2d_wall"])
    _print_timing("GPU replay sample", result.timings["gpu_sample_wall"])
    for note in result.notes:
        print(f"    note: {note}")


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "__dataclass_fields__"):
        return asdict(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _write_results(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=_json_default), encoding="utf-8")


def _json_ready(value: Any) -> Any:
    return json.loads(json.dumps(value, default=_json_default))


def _result_timing_mean(result: dict[str, Any], key: str) -> float:
    return float(result["timings"][key]["mean_ms"])


def _capacity_label(capacity_rows: int, config_capacity_rows: int) -> str:
    multiplier = capacity_rows / max(config_capacity_rows, 1)
    if capacity_rows % 1_048_576 == 0:
        rows_label = f"{capacity_rows // 1_048_576}M"
    elif capacity_rows % 1024 == 0:
        rows_label = f"{capacity_rows // 1024}K"
    else:
        rows_label = f"{capacity_rows:,}"
    return f"{multiplier:g}x\n{rows_label}"


def _annotate(ax: Any, x: float, y: float, text: str, *, fontsize: int = 8) -> None:
    if y <= 0:
        return
    ax.annotate(
        text,
        xy=(x, y),
        xytext=(0, 3),
        textcoords="offset points",
        ha="center",
        va="bottom",
        fontsize=fontsize,
    )


def _save_component_breakdown_plot(payload: dict[str, Any], output_path: Path) -> str | None:
    if plt is None:
        return None
    results = list(payload.get("results", []))
    if not results:
        return None

    config_capacity_rows = int(payload["case"]["config_capacity_rows"])
    world_sizes = sorted({int(result["world_size"]) for result in results})
    ncols = 2 if len(world_sizes) > 1 else 1
    nrows = (len(world_sizes) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(8.0 * ncols, 5.1 * nrows), squeeze=False)
    colors = {
        "cpu_sample": "#4C78A8",
        "h2d": "#F58518",
        "gpu": "#54A24B",
    }
    max_y = 0.0

    for index, world_size in enumerate(world_sizes):
        ax = axes[index // ncols][index % ncols]
        rows = sorted(
            (result for result in results if int(result["world_size"]) == world_size),
            key=lambda item: int(item["capacity_rows"]),
        )
        labels = [
            _capacity_label(int(result["capacity_rows"]), config_capacity_rows) for result in rows
        ]
        x_positions = list(range(len(rows)))
        bar_width = 0.34
        cpu_sample = [_result_timing_mean(result, "cpu_sample_wall") for result in rows]
        h2d = [_result_timing_mean(result, "cpu_sample_h2d_wall") for result in rows]
        cpu_total = [_result_timing_mean(result, "cpu_sample_then_h2d_wall") for result in rows]
        gpu_sample = [_result_timing_mean(result, "gpu_sample_wall") for result in rows]
        component_total = [sample + copy for sample, copy in zip(cpu_sample, h2d)]
        max_y = max(max_y, *(component_total or [0.0]), *(gpu_sample or [0.0]))

        cpu_x = [x - bar_width / 2 for x in x_positions]
        gpu_x = [x + bar_width / 2 for x in x_positions]
        ax.bar(cpu_x, cpu_sample, bar_width, color=colors["cpu_sample"], label="CPU sample")
        ax.bar(cpu_x, h2d, bar_width, bottom=cpu_sample, color=colors["h2d"], label="H2D")
        ax.bar(gpu_x, gpu_sample, bar_width, color=colors["gpu"], label="GPU sample")

        for x, total, measured_total in zip(cpu_x, component_total, cpu_total):
            _annotate(ax, x, total, f"{measured_total:.2f}", fontsize=7)
        for x, value in zip(gpu_x, gpu_sample):
            _annotate(ax, x, value, f"{value:.3f}", fontsize=7)

        ax.set_title(f"{world_size} GPU(s)")
        ax.set_xticks(x_positions)
        ax.set_xticklabels(labels)
        ax.set_ylabel("Mean wall time (ms)")
        ax.grid(True, axis="y", alpha=0.25)

    for index in range(len(world_sizes), nrows * ncols):
        axes[index // ncols][index % ncols].axis("off")

    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.suptitle(
        "SAC replay sampling placement: CPU sample + H2D vs GPU-resident sample",
        y=0.985,
    )
    fig.legend(handles, labels, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 0.955))
    if max_y > 0:
        for ax_row in axes:
            for ax in ax_row:
                if ax.has_data():
                    ax.set_ylim(0, max_y * 1.28)
    fig.tight_layout(rect=(0, 0, 1, 0.90), h_pad=2.0, w_pad=1.5)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=170, bbox_inches="tight")
    plt.close(fig)
    return str(output_path.resolve())


def _save_speedup_plot(payload: dict[str, Any], output_path: Path) -> str | None:
    if plt is None:
        return None
    results = list(payload.get("results", []))
    if not results:
        return None

    config_capacity_rows = int(payload["case"]["config_capacity_rows"])
    world_sizes = sorted({int(result["world_size"]) for result in results})
    capacities = sorted({int(result["capacity_rows"]) for result in results})
    x_positions = list(range(len(world_sizes)))
    width = min(0.8 / max(len(capacities), 1), 0.22)

    fig, ax = plt.subplots(figsize=(10.5, 5.6))
    for cap_index, capacity_rows in enumerate(capacities):
        values: list[float] = []
        for world_size in world_sizes:
            match = next(
                (
                    result
                    for result in results
                    if int(result["world_size"]) == world_size
                    and int(result["capacity_rows"]) == capacity_rows
                ),
                None,
            )
            if match is None:
                values.append(float("nan"))
                continue
            cpu_total = _result_timing_mean(match, "cpu_sample_then_h2d_wall")
            gpu_sample = _result_timing_mean(match, "gpu_sample_wall")
            values.append(cpu_total / gpu_sample if gpu_sample > 0 else float("nan"))
        offset = (cap_index - (len(capacities) - 1) / 2) * width
        bar_x = [x + offset for x in x_positions]
        bars = ax.bar(
            bar_x,
            values,
            width,
            label=_capacity_label(capacity_rows, config_capacity_rows).replace("\n", " "),
        )
        for bar, value in zip(bars, values):
            if value == value and value > 0:
                _annotate(ax, float(bar.get_x() + bar.get_width() / 2), value, f"{value:.1f}x")

    ax.set_title("CPU sample+H2D wall time divided by GPU-resident sample time")
    ax.set_ylabel("Speedup (x)")
    ax.set_xticks(x_positions)
    ax.set_xticklabels([str(world_size) for world_size in world_sizes])
    ax.set_xlabel("GPU count")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(title="Replay capacity", loc="best")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=170, bbox_inches="tight")
    plt.close(fig)
    return str(output_path.resolve())


def _save_gpu_sample_plot(payload: dict[str, Any], output_path: Path) -> str | None:
    if plt is None:
        return None
    results = list(payload.get("results", []))
    if not results:
        return None

    config_capacity_rows = int(payload["case"]["config_capacity_rows"])
    world_sizes = sorted({int(result["world_size"]) for result in results})
    capacities = sorted({int(result["capacity_rows"]) for result in results})
    x_positions = list(range(len(world_sizes)))
    width = min(0.8 / max(len(capacities), 1), 0.22)

    fig, ax = plt.subplots(figsize=(10.5, 5.6))
    for cap_index, capacity_rows in enumerate(capacities):
        values: list[float] = []
        for world_size in world_sizes:
            match = next(
                (
                    result
                    for result in results
                    if int(result["world_size"]) == world_size
                    and int(result["capacity_rows"]) == capacity_rows
                ),
                None,
            )
            values.append(
                float("nan") if match is None else _result_timing_mean(match, "gpu_sample_wall")
            )
        offset = (cap_index - (len(capacities) - 1) / 2) * width
        bar_x = [x + offset for x in x_positions]
        bars = ax.bar(
            bar_x,
            values,
            width,
            label=_capacity_label(capacity_rows, config_capacity_rows).replace("\n", " "),
        )
        for bar, value in zip(bars, values):
            if value == value and value > 0:
                _annotate(ax, float(bar.get_x() + bar.get_width() / 2), value, f"{value:.3f}")

    ax.set_title("GPU-resident replay sample wall time")
    ax.set_ylabel("Mean wall time (ms)")
    ax.set_xticks(x_positions)
    ax.set_xticklabels([str(world_size) for world_size in world_sizes])
    ax.set_xlabel("GPU count")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(title="Replay capacity", loc="best")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=170, bbox_inches="tight")
    plt.close(fig)
    return str(output_path.resolve())


def _save_plots(payload: dict[str, Any], plot_dir: Path) -> list[str]:
    if plt is None:
        print("Plotting skipped: matplotlib is not available.")
        return []

    saved: list[str] = []
    for maybe_path in (
        _save_component_breakdown_plot(
            payload, plot_dir / "sac_replay_sampling_component_breakdown.png"
        ),
        _save_speedup_plot(payload, plot_dir / "sac_replay_sampling_speedup.png"),
        _save_gpu_sample_plot(payload, plot_dir / "sac_replay_gpu_sample_wall_time.png"),
    ):
        if maybe_path is not None:
            saved.append(maybe_path)
    return saved


def _write_analysis_markdown(payload: dict[str, Any], output_path: Path) -> str:
    results = sorted(
        list(payload.get("results", [])),
        key=lambda result: (int(result["world_size"]), int(result["capacity_rows"])),
    )
    case = payload.get("case", {})
    shape = case.get("shape", {})
    packed_width = shape.get("packed_width")
    if packed_width is None and {"obs_dim", "action_dim", "critic_dim"} <= set(shape):
        packed_width = (
            2 * int(shape["obs_dim"]) + int(shape["action_dim"]) + 3 + 2 * int(shape["critic_dim"])
        )
    lines = [
        "# SAC Replay Buffer Sampling Time Analysis",
        "",
        f"- Command: `{case.get('command', 'unknown')}`",
        f"- Sample count per rank: `{case.get('sample_count_per_rank', 'unknown')}`",
        f"- Packed row width: `{packed_width if packed_width is not None else 'unknown'}` float32 columns",
        "",
        "Measured `CPU sample then H2D` is timed as its own end-to-end pass, so it can differ "
        "slightly from `CPU sample + H2D` due to cache effects and run-to-run noise.",
        "",
        "| GPUs | Capacity rows | CPU sample ms | H2D ms | CPU sample+H2D ms | GPU sample ms | Speedup | CPU sample share | H2D share |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for result in results:
        cpu_sample = _result_timing_mean(result, "cpu_sample_wall")
        h2d = _result_timing_mean(result, "cpu_sample_h2d_wall")
        cpu_total = _result_timing_mean(result, "cpu_sample_then_h2d_wall")
        gpu_sample = _result_timing_mean(result, "gpu_sample_wall")
        component_sum = cpu_sample + h2d
        sample_share = cpu_sample / component_sum if component_sum > 0 else 0.0
        h2d_share = h2d / component_sum if component_sum > 0 else 0.0
        speedup = cpu_total / gpu_sample if gpu_sample > 0 else float("nan")
        lines.append(
            "| "
            f"{int(result['world_size'])} | "
            f"{int(result['capacity_rows']):,} | "
            f"{cpu_sample:.3f} | "
            f"{h2d:.3f} | "
            f"{cpu_total:.3f} | "
            f"{gpu_sample:.3f} | "
            f"{speedup:.1f}x | "
            f"{sample_share * 100:.1f}% | "
            f"{h2d_share * 100:.1f}% |"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return str(output_path.resolve())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sim", default=DEFAULT_SIM)
    parser.add_argument("--gpu-counts", default=DEFAULT_GPU_COUNTS)
    parser.add_argument("--device-ids", default="auto")
    parser.add_argument(
        "--capacity-rows",
        default="auto",
        help="Comma-separated explicit replay row counts, or 'auto'.",
    )
    parser.add_argument(
        "--capacity-multipliers",
        default=DEFAULT_CAPACITY_MULTIPLIERS,
        help="Used when --capacity-rows=auto; relative to config replay capacity.",
    )
    parser.add_argument(
        "--capacity-exponents",
        default=None,
        help="Comma-separated powers of two for replay rows, e.g. '20,22,24,26,28,30'.",
    )
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeat", type=int, default=20)
    parser.add_argument("--prefill", choices=("zeros", "none"), default="zeros")
    parser.add_argument(
        "--index-mode",
        choices=("timed", "pregenerated"),
        default="timed",
        help="timed includes random index generation with preallocated index tensors.",
    )
    parser.add_argument(
        "--host-batch-memory",
        choices=("pinned", "pageable"),
        default="pinned",
        help="CPU sampled-batch memory before H2D.",
    )
    parser.add_argument(
        "--torch-threads",
        default="auto",
        help="PyTorch intra-op threads per run. 'auto' uses cpu_count/world_size.",
    )
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument(
        "--device-memory-safety-factor",
        type=float,
        default=1.15,
        help="Skip a CUDA case when free memory is below estimated bytes times this factor.",
    )
    parser.add_argument("--out-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--obs-dim", type=int, default=0)
    parser.add_argument("--action-dim", type=int, default=0)
    parser.add_argument("--critic-dim", type=int, default=0)
    parser.add_argument(
        "--symmetry-batch-multiplier",
        type=int,
        default=0,
        help="Only used with manual --obs-dim/--action-dim/--critic-dim.",
    )
    parser.add_argument("--plot-dir", type=Path, default=None)
    parser.add_argument("--analysis-md", type=Path, default=None)
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument(
        "--plot-only",
        type=Path,
        default=None,
        help="Load an existing results JSON and regenerate plots/analysis without benchmarking.",
    )
    args = parser.parse_args(argv)

    if args.plot_only is not None:
        payload = json.loads(args.plot_only.read_text(encoding="utf-8"))
        plot_dir = args.plot_dir or args.plot_only.parent
        plot_paths = [] if args.no_plots else _save_plots(payload, plot_dir)
        analysis_path = _write_analysis_markdown(
            payload,
            args.analysis_md or plot_dir / "analysis.md",
        )
        for path in plot_paths:
            print(f"Saved plot: {path}")
        print(f"Saved analysis: {analysis_path}")
        return 0

    if args.warmup < 0 or args.repeat <= 0:
        raise ValueError("--warmup must be >= 0 and --repeat must be > 0")
    if args.device_memory_safety_factor < 1.0:
        raise ValueError("--device-memory-safety-factor must be >= 1.0")

    cfg = _compose_offpolicy_cfg(args.sim)
    manual_shape = args.obs_dim > 0 or args.action_dim > 0 or args.critic_dim > 0
    if manual_shape:
        if args.obs_dim <= 0 or args.action_dim <= 0 or args.critic_dim < 0:
            raise ValueError(
                "Manual shape requires --obs-dim > 0, --action-dim > 0, --critic-dim >= 0"
            )
        shape = ReplayShape(
            obs_dim=int(args.obs_dim),
            action_dim=int(args.action_dim),
            critic_dim=int(args.critic_dim),
        )
        if bool(OmegaConf.select(cfg, "algo.use_symmetry", default=False)):
            if args.symmetry_batch_multiplier <= 0:
                raise ValueError(
                    "Manual shape with SAC symmetry requires --symmetry-batch-multiplier"
                )
            symmetry_batch_multiplier = int(args.symmetry_batch_multiplier)
        else:
            symmetry_batch_multiplier = 1
    else:
        shape, symmetry_batch_multiplier = _resolve_env_shape_and_symmetry(cfg)

    case = _build_case(
        cfg,
        sim=args.sim,
        shape=shape,
        symmetry_batch_multiplier=symmetry_batch_multiplier,
    )
    capacity_rows = _resolve_capacity_rows(
        config_capacity_rows=case.config_capacity_rows,
        capacity_rows_arg=args.capacity_rows,
        capacity_multipliers_arg=args.capacity_multipliers,
        capacity_exponents_arg=args.capacity_exponents,
    )
    gpu_counts = _parse_int_list(args.gpu_counts, name="gpu counts")
    device_ids = _parse_device_ids(args.device_ids)

    print("SAC Replay Buffer Sampling Benchmark")
    print(f"Config command: {case.command}")
    print(f"Device info: {get_device_info_line()}")
    print(
        "Shape: "
        f"obs={case.shape.obs_dim}, action={case.shape.action_dim}, "
        f"critic={case.shape.critic_dim}, packed_width={case.shape.packed_width}"
    )
    print(
        "Config: "
        f"num_envs={case.num_envs:,}, replay_buffer_n={case.replay_buffer_n:,}, "
        f"capacity={case.config_capacity_rows:,}, configured_batch={case.configured_batch_size:,}, "
        f"learner_batch={case.learner_batch_size:,}, updates={case.updates_per_step}, "
        f"sample_count/rank={case.sample_count_per_rank:,}"
    )
    print(f"Capacity rows: {', '.join(f'{rows:,}' for rows in capacity_rows)}")
    print(f"Requested GPU counts: {gpu_counts}; visible CUDA device ids: {device_ids}")

    results: list[CapacityResult] = []
    skipped: list[SkippedCase] = []

    if not torch.cuda.is_available() or not device_ids:
        for world_size in gpu_counts:
            skipped.append(
                SkippedCase(
                    world_size=world_size,
                    capacity_rows=None,
                    reason="CUDA is not available or no CUDA device ids were selected",
                )
            )
    else:
        for world_size in gpu_counts:
            if world_size > len(device_ids):
                skipped.append(
                    SkippedCase(
                        world_size=world_size,
                        capacity_rows=None,
                        reason=f"requested {world_size} GPUs but only {len(device_ids)} are visible",
                    )
                )
                continue

            devices = [torch.device(f"cuda:{device_id}") for device_id in device_ids[:world_size]]
            torch_threads = _resolve_torch_threads(args.torch_threads, world_size=world_size)
            for rows in capacity_rows:
                required = _estimate_device_bytes_per_gpu(case, rows)
                skip_reason = _cuda_memory_skip_reason(
                    devices,
                    required_bytes_per_gpu=required,
                    safety_factor=float(args.device_memory_safety_factor),
                )
                if skip_reason is not None:
                    skipped.append(
                        SkippedCase(
                            world_size=world_size,
                            capacity_rows=rows,
                            reason=skip_reason,
                        )
                    )
                    print(f"\nSkipping {world_size} GPU(s), capacity={rows:,}: {skip_reason}")
                    continue
                print(f"\nRunning {world_size} GPU(s), capacity={rows:,}")
                try:
                    result = _run_capacity_case(
                        case,
                        capacity_rows=rows,
                        devices=devices,
                        warmup=args.warmup,
                        repeat=args.repeat,
                        prefill=args.prefill,
                        pinned_host_batch=args.host_batch_memory == "pinned",
                        index_mode=args.index_mode,
                        seed=args.seed,
                        torch_threads=torch_threads,
                    )
                except RuntimeError as exc:
                    if "out of memory" not in str(exc).lower():
                        raise
                    _cleanup_device()
                    skipped.append(
                        SkippedCase(
                            world_size=world_size,
                            capacity_rows=rows,
                            reason=f"CUDA OOM during benchmark allocation/run: {exc}",
                        )
                    )
                    print(f"  skipped: CUDA OOM during benchmark allocation/run: {exc}")
                    continue
                results.append(result)
                _print_result(result)

    payload = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "torch_version": torch.__version__,
        "device_info": get_device_info_dict(),
        "args": {
            "sim": args.sim,
            "gpu_counts": gpu_counts,
            "device_ids": device_ids,
            "capacity_rows": capacity_rows,
            "capacity_rows_arg": args.capacity_rows,
            "capacity_multipliers": args.capacity_multipliers,
            "capacity_exponents": args.capacity_exponents,
            "warmup": args.warmup,
            "repeat": args.repeat,
            "prefill": args.prefill,
            "index_mode": args.index_mode,
            "host_batch_memory": args.host_batch_memory,
            "torch_threads": args.torch_threads,
            "seed": args.seed,
            "device_memory_safety_factor": args.device_memory_safety_factor,
            "manual_shape": manual_shape,
        },
        "case": case,
        "results": results,
        "skipped": skipped,
    }
    serializable_payload = _json_ready(payload)
    plot_dir = args.plot_dir or args.out_json.parent
    plot_paths = [] if args.no_plots else _save_plots(serializable_payload, plot_dir)
    analysis_path = _write_analysis_markdown(
        serializable_payload,
        args.analysis_md or plot_dir / "analysis.md",
    )
    serializable_payload["plots"] = plot_paths
    serializable_payload["analysis_markdown"] = analysis_path
    _write_results(args.out_json, serializable_payload)
    for item in skipped:
        suffix = "" if item.capacity_rows is None else f", capacity={item.capacity_rows:,}"
        print(f"Skipped {item.world_size} GPU(s){suffix}: {item.reason}")
    for path in plot_paths:
        print(f"Saved plot: {path}")
    print(f"Saved analysis: {analysis_path}")
    print(f"\nSaved JSON: {args.out_json}")
    if not results:
        print("No CUDA benchmark cases were run.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
