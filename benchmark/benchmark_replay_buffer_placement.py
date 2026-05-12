#!/usr/bin/env python3
"""Benchmark replay-buffer placement costs for off-policy G1 defaults.

This benchmark uses the default Hydra configs behind:

    uv run train --algo sac --task g1_walk_flat --sim mujoco
    uv run train --algo flashsac --task g1_walk_flat --sim mujoco

It does not run training.  It composes the task config, builds a single env to
read the obs/action/critic dimensions, and then times synthetic replay tensors
with the same capacity and learner sampling shape.

Usage:
    uv run benchmark/benchmark_replay_buffer_placement.py
    uv run benchmark/benchmark_replay_buffer_placement.py --tasks auto
    uv run benchmark/benchmark_replay_buffer_placement.py --warmup 3 --repeat 10
    uv run benchmark/benchmark_replay_buffer_placement.py --max-capacity-rows 1048576
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, median, pstdev
from typing import Any, cast

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
    ROOT_DIR / "benchmark" / "outputs" / "replay_buffer_placement" / "results.json"
)
DEFAULT_ALGOS = ("sac", "flashsac", "td3")
DEFAULT_TASKS_ARG = "auto"
DEFAULT_SIM = "mujoco"
FLOAT_BYTES = 4
TIMING_LABELS = {
    "cpu_full_random_sample": "CPU full sample",
    "gpu_full_random_sample": "Device full sample",
    "current_ipc_incremental_h2d": "IPC incremental transfer",
    "cpu_full_presample": "CPU pre-sample",
    "cpu_sampled_batch_h2d": "Sampled batch transfer",
    "cpu_presample_plus_h2d": "CPU pre-sample + transfer",
}
SCHEME_SEGMENTS = {
    "current": (
        ("current_ipc_incremental_h2d", "Incremental transfer", "#4C78A8"),
        ("gpu_full_random_sample", "Device random sample", "#72B7B2"),
    ),
    "cpu_presample": (
        ("cpu_full_presample", "CPU pre-sample", "#F58518"),
        ("cpu_sampled_batch_h2d", "Sampled batch transfer", "#E45756"),
    ),
}
SCHEME_LABELS = {
    "current": "Current",
    "cpu_presample": "CPU pre-sample",
}


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
    replay_pipeline: str
    num_envs: int
    env_steps_per_sync: int
    replay_buffer_n: int
    config_capacity_rows: int
    benchmark_capacity_rows: int
    configured_batch_size: int
    learner_batch_size: int
    symmetry_batch_multiplier: int
    updates_per_step: int
    sample_count: int
    learning_starts: int
    incremental_rows: int
    shape: ReplayShape

    @property
    def packed_replay_bytes(self) -> int:
        return self.benchmark_capacity_rows * self.shape.packed_width * FLOAT_BYTES

    @property
    def sampled_batch_bytes(self) -> int:
        return self.sample_count * self.shape.packed_width * FLOAT_BYTES

    @property
    def incremental_bytes(self) -> int:
        return self.incremental_rows * self.shape.packed_width * FLOAT_BYTES


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
class CaseResult:
    case: BenchmarkCase
    timings: dict[str, TimingStats]
    notes: list[str]


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


def _sync_device(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps" and hasattr(torch, "mps"):
        torch.mps.synchronize()


def _measure_ms(
    fn,
    *,
    warmup: int,
    repeat: int,
    device: torch.device | None = None,
) -> TimingStats:
    samples_ms: list[float] = []
    for idx in range(warmup + repeat):
        if device is not None:
            _sync_device(device)
        t0 = time.perf_counter_ns()
        fn()
        if device is not None:
            _sync_device(device)
        elapsed_ms = (time.perf_counter_ns() - t0) / 1e6
        if idx >= warmup:
            samples_ms.append(elapsed_ms)
    return _stats(samples_ms, warmup=warmup, repeat=repeat)


def _cleanup_device() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if hasattr(torch, "mps") and torch.backends.mps.is_available():
        torch.mps.empty_cache()


def _compose_offpolicy_cfg(algo: str, task: str, sim: str) -> DictConfig:
    config_dir = str(ROOT_DIR / "conf" / "offpolicy")
    overrides = [
        f"algo={algo}",
        f"task={algo}/{task}/{sim}",
        "hydra.run.dir=.",
        "hydra.output_subdir=null",
        "hydra/job_logging=disabled",
        "hydra/hydra_logging=disabled",
    ]
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=config_dir, version_base="1.3"):
        return compose(config_name="config", overrides=overrides)


def _owner_config_path(algo: str, task: str, sim: str) -> Path:
    return ROOT_DIR / "conf" / "offpolicy" / "task" / algo / task / f"{sim}.yaml"


def _owner_config_exists(algo: str, task: str, sim: str) -> bool:
    return _owner_config_path(algo, task, sim).is_file()


def _discover_supported_tasks(algo: str, sim: str) -> list[str]:
    task_root = ROOT_DIR / "conf" / "offpolicy" / "task" / algo
    if not task_root.is_dir():
        return []
    return sorted(path.parent.name for path in task_root.glob(f"*/{sim}.yaml") if path.is_file())


def _resolve_targets(
    *,
    algos: list[str],
    tasks: list[str],
    sim: str,
) -> tuple[list[tuple[str, str]], list[dict[str, str]]]:
    targets: list[tuple[str, str]] = []
    skipped: list[dict[str, str]] = []

    if tasks == ["auto"]:
        for algo in algos:
            for task in _discover_supported_tasks(algo, sim):
                targets.append((algo, task))
        return targets, skipped

    for algo in algos:
        for task in tasks:
            owner_config = _owner_config_path(algo, task, sim)
            if owner_config.is_file():
                targets.append((algo, task))
                continue
            skipped.append(
                {
                    "algo": algo,
                    "task": task,
                    "sim": sim,
                    "reason": "missing_owner_config",
                    "path": str(owner_config),
                }
            )
    return targets, skipped


def _resolve_replay_pipeline(algo: str, cfg: DictConfig) -> str:
    replay_pipeline = str(OmegaConf.select(cfg, "training.replay_pipeline") or "auto")
    if replay_pipeline == "auto":
        return "cpu_pinned_double_buffer" if algo == "sac" else "gpu_cache"
    return replay_pipeline


def _resolve_env_shape_and_symmetry(cfg: DictConfig, algo: str) -> tuple[ReplayShape, int]:
    from unilab.base.observations import get_obs_dims
    from unilab.training import BackendAdapter, create_env, ensure_registries

    ensure_registries()
    env_cfg_override = BackendAdapter(
        cfg,
        root_dir=ROOT_DIR,
        algo_name=algo,
    ).build_task_env_cfg_override()
    env = create_env(cfg, num_envs=1, env_cfg_override=env_cfg_override)
    try:
        obs_dim, critic_dim = get_obs_dims(env.obs_groups_spec)
        action_shape = env.action_space.shape
        if action_shape is None:
            raise ValueError("env.action_space.shape must be defined")
        action_dim = int(action_shape[0])

        symmetry_batch_multiplier = 1
        use_symmetry = bool(OmegaConf.select(cfg, "algo.use_symmetry", default=False))
        if algo == "sac" and use_symmetry:
            symmetry_builder = getattr(env, "build_symmetry_augmentation", None)
            if not callable(symmetry_builder):
                raise ValueError(f"{cfg.training.task_name} does not provide symmetry augmentation")
            symmetry = cast(Any, symmetry_builder(device="cpu"))
            if symmetry is None:
                raise ValueError(f"{cfg.training.task_name} does not provide symmetry augmentation")
            symmetry_batch_multiplier = int(symmetry.batch_multiplier)
    finally:
        env.close()

    return ReplayShape(
        obs_dim=int(obs_dim),
        action_dim=action_dim,
        critic_dim=int(critic_dim),
    ), symmetry_batch_multiplier


def _build_case(
    cfg: DictConfig,
    *,
    algo: str,
    task: str,
    sim: str,
    shape: ReplayShape,
    symmetry_batch_multiplier: int,
    max_capacity_rows: int | None,
) -> BenchmarkCase:
    num_envs = int(cfg.algo.num_envs)
    replay_buffer_n = int(cfg.algo.replay_buffer_n)
    config_capacity_rows = replay_buffer_n * num_envs
    benchmark_capacity_rows = config_capacity_rows
    if max_capacity_rows is not None and max_capacity_rows > 0:
        benchmark_capacity_rows = min(config_capacity_rows, int(max_capacity_rows))

    configured_batch_size = int(cfg.algo.batch_size)
    learner_batch_size = configured_batch_size
    if algo == "sac" and bool(OmegaConf.select(cfg, "algo.use_symmetry", default=False)):
        if configured_batch_size % symmetry_batch_multiplier != 0:
            raise ValueError(
                "SAC symmetry requires batch_size divisible by "
                f"{symmetry_batch_multiplier}, got {configured_batch_size}"
            )
        learner_batch_size = configured_batch_size // symmetry_batch_multiplier

    updates_per_step = int(cfg.algo.updates_per_step)
    env_steps_per_sync = int(cfg.training.env_steps_per_sync)
    command = f"uv run train --algo {algo} --task {task} --sim {sim}"

    return BenchmarkCase(
        algo=algo,
        task=task,
        sim=sim,
        command=command,
        training_task_name=str(cfg.training.task_name),
        replay_pipeline=_resolve_replay_pipeline(algo, cfg),
        num_envs=num_envs,
        env_steps_per_sync=env_steps_per_sync,
        replay_buffer_n=replay_buffer_n,
        config_capacity_rows=config_capacity_rows,
        benchmark_capacity_rows=benchmark_capacity_rows,
        configured_batch_size=configured_batch_size,
        learner_batch_size=learner_batch_size,
        symmetry_batch_multiplier=symmetry_batch_multiplier,
        updates_per_step=updates_per_step,
        sample_count=learner_batch_size * updates_per_step,
        learning_starts=int(cfg.algo.learning_starts),
        incremental_rows=num_envs * env_steps_per_sync,
        shape=shape,
    )


def _allocate_packed(
    *,
    rows: int,
    width: int,
    device: torch.device | str,
    pin_memory: bool = False,
    prefill: str = "zeros",
) -> torch.Tensor:
    tensor = torch.empty(
        (rows, width),
        dtype=torch.float32,
        device=device,
        pin_memory=pin_memory,
    )
    if prefill == "zeros":
        tensor.zero_()
    elif prefill != "none":
        raise ValueError(f"Unsupported prefill={prefill!r}")
    return tensor


def _allocate_cpu_batch(
    *,
    rows: int,
    width: int,
    prefer_pinned: bool,
    prefill: str,
    notes: list[str],
) -> torch.Tensor:
    if not prefer_pinned:
        return _allocate_packed(
            rows=rows,
            width=width,
            device="cpu",
            pin_memory=False,
            prefill=prefill,
        )
    try:
        return _allocate_packed(
            rows=rows,
            width=width,
            device="cpu",
            pin_memory=True,
            prefill=prefill,
        )
    except RuntimeError as exc:
        notes.append(f"pinned sampled-batch allocation failed; fell back to pageable CPU: {exc}")
        return _allocate_packed(
            rows=rows,
            width=width,
            device="cpu",
            pin_memory=False,
            prefill=prefill,
        )


def _bench_cpu_full_random_sample(
    case: BenchmarkCase,
    *,
    warmup: int,
    repeat: int,
    prefill: str,
) -> TimingStats:
    width = case.shape.packed_width
    storage = _allocate_packed(
        rows=case.benchmark_capacity_rows,
        width=width,
        device="cpu",
        prefill=prefill,
    )
    out = torch.empty((case.sample_count, width), dtype=torch.float32)

    def sample_cpu() -> None:
        indices = torch.randint(0, case.benchmark_capacity_rows, (case.sample_count,))
        torch.index_select(storage, 0, indices, out=out)

    try:
        return _measure_ms(sample_cpu, warmup=warmup, repeat=repeat)
    finally:
        del out, storage
        _cleanup_device()


def _bench_gpu_full_random_sample(
    case: BenchmarkCase,
    *,
    device: torch.device,
    warmup: int,
    repeat: int,
    prefill: str,
) -> TimingStats:
    width = case.shape.packed_width
    storage = _allocate_packed(
        rows=case.benchmark_capacity_rows,
        width=width,
        device=device,
        prefill=prefill,
    )
    out = torch.empty((case.sample_count, width), dtype=torch.float32, device=device)

    def sample_gpu() -> None:
        indices = torch.randint(
            0,
            case.benchmark_capacity_rows,
            (case.sample_count,),
            device=device,
        )
        torch.index_select(storage, 0, indices, out=out)

    try:
        return _measure_ms(sample_gpu, warmup=warmup, repeat=repeat, device=device)
    finally:
        del out, storage
        _cleanup_device()


def _bench_current_ipc_incremental_h2d(
    case: BenchmarkCase,
    *,
    device: torch.device,
    warmup: int,
    repeat: int,
    prefill: str,
    source_pinned: bool,
    notes: list[str],
) -> TimingStats:
    width = case.shape.packed_width
    source = _allocate_cpu_batch(
        rows=case.incremental_rows,
        width=width,
        prefer_pinned=source_pinned,
        prefill=prefill,
        notes=notes,
    )
    target = _allocate_packed(
        rows=case.benchmark_capacity_rows,
        width=width,
        device=device,
        prefill=prefill,
    )
    offset = 0

    def copy_increment() -> None:
        nonlocal offset
        remaining = case.incremental_rows
        source_offset = 0
        target_offset = offset
        non_blocking = source.is_pinned() and device.type == "cuda"
        while remaining > 0:
            chunk_rows = min(remaining, case.benchmark_capacity_rows - target_offset)
            target[target_offset : target_offset + chunk_rows].copy_(
                source[source_offset : source_offset + chunk_rows],
                non_blocking=non_blocking,
            )
            remaining -= chunk_rows
            source_offset += chunk_rows
            target_offset = (target_offset + chunk_rows) % case.benchmark_capacity_rows
        offset = target_offset

    try:
        return _measure_ms(copy_increment, warmup=warmup, repeat=repeat, device=device)
    finally:
        del source, target
        _cleanup_device()


def _bench_cpu_presample(
    case: BenchmarkCase,
    *,
    warmup: int,
    repeat: int,
    prefill: str,
    sampled_prefer_pinned: bool,
    notes: list[str],
) -> tuple[TimingStats, torch.Tensor, torch.Tensor]:
    width = case.shape.packed_width
    storage = _allocate_packed(
        rows=case.benchmark_capacity_rows,
        width=width,
        device="cpu",
        prefill=prefill,
    )
    sampled = _allocate_cpu_batch(
        rows=case.sample_count,
        width=width,
        prefer_pinned=sampled_prefer_pinned,
        prefill=prefill,
        notes=notes,
    )

    def sample_into_cpu_batch() -> None:
        indices = torch.randint(0, case.benchmark_capacity_rows, (case.sample_count,))
        torch.index_select(storage, 0, indices, out=sampled)

    stats = _measure_ms(sample_into_cpu_batch, warmup=warmup, repeat=repeat)
    return stats, storage, sampled


def _bench_cpu_sampled_h2d(
    sampled: torch.Tensor,
    *,
    device: torch.device,
    warmup: int,
    repeat: int,
) -> TimingStats:
    target = torch.empty_like(sampled, device=device)

    def copy_sampled_batch() -> None:
        target.copy_(sampled, non_blocking=sampled.is_pinned() and device.type == "cuda")

    try:
        return _measure_ms(copy_sampled_batch, warmup=warmup, repeat=repeat, device=device)
    finally:
        del target
        _cleanup_device()


def _run_case(
    case: BenchmarkCase,
    *,
    device: torch.device,
    warmup: int,
    repeat: int,
    prefill: str,
    incremental_source_pinned: bool,
    sampled_batch_pinned: bool,
) -> CaseResult:
    notes: list[str] = []
    timings: dict[str, TimingStats] = {}
    effective_incremental_source_pinned = incremental_source_pinned
    effective_sampled_batch_pinned = sampled_batch_pinned
    if device.type != "cuda":
        if incremental_source_pinned:
            notes.append(f"pinned incremental source is CUDA-only; using pageable CPU for {device}")
        if sampled_batch_pinned:
            notes.append(f"pinned sampled batch is CUDA-only; using pageable CPU for {device}")
        effective_incremental_source_pinned = False
        effective_sampled_batch_pinned = False

    print(f"\n[{case.algo}] {case.command}")
    print(
        "  "
        f"capacity={case.config_capacity_rows:,}"
        f" benchmark_capacity={case.benchmark_capacity_rows:,}"
        f" packed_width={case.shape.packed_width}"
    )
    print(
        "  "
        f"batch={case.learner_batch_size:,}"
        f" updates={case.updates_per_step}"
        f" sample_count={case.sample_count:,}"
        f" increment_rows={case.incremental_rows:,}"
    )
    print(
        "  "
        f"full replay={_fmt_bytes(case.packed_replay_bytes)}"
        f" sampled batch={_fmt_bytes(case.sampled_batch_bytes)}"
        f" increment={_fmt_bytes(case.incremental_bytes)}"
    )

    timings["cpu_full_random_sample"] = _bench_cpu_full_random_sample(
        case,
        warmup=warmup,
        repeat=repeat,
        prefill=prefill,
    )
    _print_timing("CPU full replay random sample", timings["cpu_full_random_sample"])

    timings["gpu_full_random_sample"] = _bench_gpu_full_random_sample(
        case,
        device=device,
        warmup=warmup,
        repeat=repeat,
        prefill=prefill,
    )
    _print_timing("Device full replay random sample", timings["gpu_full_random_sample"])

    timings["current_ipc_incremental_h2d"] = _bench_current_ipc_incremental_h2d(
        case,
        device=device,
        warmup=warmup,
        repeat=repeat,
        prefill=prefill,
        source_pinned=effective_incremental_source_pinned,
        notes=notes,
    )
    _print_timing(
        "Current IPC incremental transfer",
        timings["current_ipc_incremental_h2d"],
    )

    cpu_presample_stats, storage, sampled = _bench_cpu_presample(
        case,
        warmup=warmup,
        repeat=repeat,
        prefill=prefill,
        sampled_prefer_pinned=effective_sampled_batch_pinned,
        notes=notes,
    )
    timings["cpu_full_presample"] = cpu_presample_stats
    _print_timing("CPU full replay pre-sample", timings["cpu_full_presample"])
    try:
        timings["cpu_sampled_batch_h2d"] = _bench_cpu_sampled_h2d(
            sampled,
            device=device,
            warmup=warmup,
            repeat=repeat,
        )
        _print_timing("CPU sampled batch transfer", timings["cpu_sampled_batch_h2d"])
    finally:
        del sampled, storage
        _cleanup_device()

    return CaseResult(case=case, timings=timings, notes=notes)


def _print_timing(label: str, stat: TimingStats) -> None:
    print(
        f"  {label:<34}"
        f" mean={stat.mean_ms:8.3f} ms"
        f" median={stat.median_ms:8.3f} ms"
        f" std={stat.std_ms:7.3f} ms"
    )


def _timing_mean(result: CaseResult, key: str) -> float | None:
    if key == "cpu_presample_plus_h2d":
        pre = result.timings.get("cpu_full_presample")
        h2d = result.timings.get("cpu_sampled_batch_h2d")
        if pre is None or h2d is None:
            return None
        return pre.mean_ms + h2d.mean_ms
    stat = result.timings.get(key)
    return None if stat is None else stat.mean_ms


def _bar_labels(ax: Any, bars: Any) -> None:
    for bar in bars:
        height = float(bar.get_height())
        if height <= 0:
            continue
        ax.annotate(
            f"{height:.2f}",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8,
            rotation=0,
        )


def _save_grouped_bar_plot(
    results: list[CaseResult],
    *,
    keys: list[str],
    title: str,
    ylabel: str,
    output_path: Path,
    device_info: str,
) -> str | None:
    if plt is None or not results:
        return None

    labels = [result.case.algo for result in results]
    x_positions = list(range(len(labels)))
    width = min(0.8 / max(len(keys), 1), 0.22)

    fig_width = max(9.0, 2.4 * len(labels) + 1.2 * len(keys))
    fig, ax = plt.subplots(figsize=(fig_width, 5.8))
    positive_values: list[float] = []

    for key_index, key in enumerate(keys):
        offset = (key_index - (len(keys) - 1) / 2) * width
        values: list[float] = []
        for result in results:
            value = _timing_mean(result, key)
            values.append(float("nan") if value is None else value)
            if value is not None and value > 0:
                positive_values.append(value)
        bars = ax.bar(
            [x + offset for x in x_positions],
            values,
            width,
            label=TIMING_LABELS.get(key, key),
        )
        _bar_labels(ax, bars)

    full_title = f"{title}\n{device_info}" if device_info else title
    ax.set_title(full_title)
    ax.set_ylabel(ylabel)
    ax.set_xticks(x_positions)
    ax.set_xticklabels(labels)
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(loc="best", fontsize=9)
    if positive_values and max(positive_values) / max(min(positive_values), 1e-9) > 20:
        ax.set_yscale("log")
        ax.set_ylabel(f"{ylabel} (log scale)")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return str(output_path.resolve())


def _save_scheme_stacked_plot(
    results: list[CaseResult],
    *,
    output_path: Path,
    device_info: str,
) -> str | None:
    if plt is None or not results:
        return None

    bar_width = 0.46
    group_gap = 0.28
    group_step = 2 * bar_width + group_gap
    bar_specs: list[tuple[CaseResult, str, float]] = []
    group_centers: list[float] = []
    group_labels: list[str] = []
    for group_index, result in enumerate(results):
        center = group_index * group_step
        group_centers.append(center)
        group_labels.append(f"{result.case.algo}/{result.case.task}")
        bar_specs.append((result, "current", center - bar_width / 2))
        bar_specs.append((result, "cpu_presample", center + bar_width / 2))

    fig_width = max(13.0, 1.25 * len(results) + 4.0)
    fig, ax = plt.subplots(figsize=(fig_width, 6.4))
    legend_labels: set[str] = set()
    totals: list[tuple[float, float]] = []

    for result, scheme, x_pos in bar_specs:
        bottom = 0.0
        for key, label, color in SCHEME_SEGMENTS[scheme]:
            value = _timing_mean(result, key)
            height = 0.0 if value is None else float(value)
            show_label = label if label not in legend_labels else None
            ax.bar(
                x_pos,
                height,
                bar_width,
                bottom=bottom,
                label=show_label,
                color=color,
                edgecolor="white",
                linewidth=0.8,
            )
            legend_labels.add(label)
            if height > 0.12:
                ax.text(
                    x_pos,
                    bottom + height / 2,
                    f"{height:.2f}",
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="white",
                )
            bottom += height
        totals.append((x_pos, bottom))

    for x_pos, total in totals:
        if total <= 0:
            continue
        ax.annotate(
            f"{total:.2f}",
            xy=(x_pos, total),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    full_title = (
        "Replay transfer schemes by task\n"
        "Within each task pair: left bar = Current, right bar = CPU pre-sample"
    )
    if device_info:
        full_title = f"{full_title}\n{device_info}"
    ax.set_title(full_title)
    ax.set_ylabel("Mean time per learner tick (ms)")
    ax.set_xticks(group_centers)
    ax.set_xticklabels(group_labels, fontsize=8)
    if bar_specs:
        ax.set_xlim(
            min(x_pos for _, _, x_pos in bar_specs) - bar_width,
            max(x_pos for _, _, x_pos in bar_specs) + bar_width,
        )
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(loc="upper right", fontsize=9)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=170, bbox_inches="tight")
    plt.close(fig)
    return str(output_path.resolve())


def _save_plots(results: list[CaseResult], plot_dir: Path, *, device_info: str) -> list[str]:
    if plt is None:
        print("Plotting skipped: matplotlib is not available.")
        return []

    saved: list[str] = []
    stacked_path = _save_scheme_stacked_plot(
        results,
        output_path=plot_dir / "replay_buffer_scheme_stacked.png",
        device_info=device_info,
    )
    if stacked_path is not None:
        saved.append(stacked_path)
    return saved


def _parse_algos(value: str) -> list[str]:
    algos = [part.strip() for part in value.split(",") if part.strip()]
    if not algos:
        raise ValueError("at least one algo is required")
    unsupported = [algo for algo in algos if algo not in DEFAULT_ALGOS]
    if unsupported:
        raise ValueError(f"unsupported algos for this benchmark: {unsupported}")
    return algos


def _parse_tasks(value: str) -> list[str]:
    tasks = [part.strip() for part in value.split(",") if part.strip()]
    if not tasks:
        raise ValueError("at least one task is required")
    if "auto" in tasks:
        if tasks != ["auto"]:
            raise ValueError("--tasks=auto cannot be combined with explicit task names")
        return tasks
    deduped: list[str] = []
    for task in tasks:
        if task not in deduped:
            deduped.append(task)
    return deduped


def _resolve_device(value: str) -> torch.device:
    if value == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but torch.cuda.is_available() is false")
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise ValueError("MPS was requested but torch.backends.mps.is_available() is false")
    return device


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "__dataclass_fields__"):
        return asdict(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _write_results(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=_json_default), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--algos", default=",".join(DEFAULT_ALGOS))
    parser.add_argument(
        "--tasks",
        default=DEFAULT_TASKS_ARG,
        help="Comma-separated tasks, or 'auto' to discover existing owner configs.",
    )
    parser.add_argument(
        "--task",
        default=None,
        help="Single-task compatibility alias. Overrides --tasks when set.",
    )
    parser.add_argument("--sim", default=DEFAULT_SIM)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeat", type=int, default=20)
    parser.add_argument(
        "--max-capacity-rows",
        type=int,
        default=0,
        help="Optional cap for faster dry runs. Default 0 keeps config capacity.",
    )
    parser.add_argument("--prefill", choices=("zeros", "none"), default="zeros")
    parser.add_argument(
        "--incremental-source",
        choices=("pageable", "pinned"),
        default="pageable",
        help="CPU source memory for the incremental full-replay device transfer.",
    )
    parser.add_argument(
        "--sampled-batch-memory",
        choices=("pinned", "pageable"),
        default="pinned",
        help="CPU memory for the pre-sampled batch before device transfer.",
    )
    parser.add_argument("--out-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument(
        "--plot-dir",
        type=Path,
        default=None,
        help="Directory for PNG plots. Default: same directory as --out-json.",
    )
    args = parser.parse_args(argv)

    if args.warmup < 0 or args.repeat <= 0:
        raise ValueError("--warmup must be >= 0 and --repeat must be > 0")

    algos = _parse_algos(args.algos)
    tasks = _parse_tasks(args.task if args.task is not None else args.tasks)
    targets, skipped_targets = _resolve_targets(algos=algos, tasks=tasks, sim=args.sim)
    device = _resolve_device(args.device)
    max_capacity_rows = args.max_capacity_rows if args.max_capacity_rows > 0 else None
    plot_dir = args.plot_dir or args.out_json.parent

    print("Replay Buffer Placement Benchmark")
    print(f"Device request: {args.device} -> {device}")
    print(f"PyTorch: {torch.__version__}")
    device_info_line = get_device_info_line()
    print(device_info_line)
    if device.type == "cuda":
        print("Transfer path: CUDA pinned/native fast path where pinned memory is available.")
    else:
        print(f"Transfer path: portable torch_copy path on {device}.")

    results: list[CaseResult] = []
    for skipped in skipped_targets:
        print(f"Skipping missing owner config: {Path(skipped['path']).relative_to(ROOT_DIR)}")
    for algo, task in targets:
        cfg = _compose_offpolicy_cfg(algo, task, args.sim)
        shape, symmetry_batch_multiplier = _resolve_env_shape_and_symmetry(cfg, algo)
        case = _build_case(
            cfg,
            algo=algo,
            task=task,
            sim=args.sim,
            shape=shape,
            symmetry_batch_multiplier=symmetry_batch_multiplier,
            max_capacity_rows=max_capacity_rows,
        )
        results.append(
            _run_case(
                case,
                device=device,
                warmup=args.warmup,
                repeat=args.repeat,
                prefill=args.prefill,
                incremental_source_pinned=args.incremental_source == "pinned",
                sampled_batch_pinned=args.sampled_batch_memory == "pinned",
            )
        )

    if not results:
        raise ValueError("No benchmark targets matched existing owner configs")

    plot_paths = _save_plots(results, plot_dir, device_info=device_info_line)
    payload = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "torch_version": torch.__version__,
        "device": str(device),
        "device_info": get_device_info_dict(),
        "args": {
            "algos": algos,
            "tasks": tasks,
            "resolved_targets": [
                {"algo": algo, "task": task, "sim": args.sim} for algo, task in targets
            ],
            "sim": args.sim,
            "warmup": args.warmup,
            "repeat": args.repeat,
            "max_capacity_rows": max_capacity_rows,
            "prefill": args.prefill,
            "incremental_source": args.incremental_source,
            "sampled_batch_memory": args.sampled_batch_memory,
            "plot_dir": str(plot_dir),
        },
        "plots": plot_paths,
        "skipped_targets": skipped_targets,
        "results": results,
    }
    _write_results(args.out_json, payload)
    if plot_paths:
        for path in plot_paths:
            print(f"Saved plot: {path}")
    print(f"\nSaved JSON: {args.out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
