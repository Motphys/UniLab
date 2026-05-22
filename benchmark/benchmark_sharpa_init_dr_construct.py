#!/usr/bin/env python3
"""Benchmark Sharpa-hand init-DR construction cost vs variant count.

What this benchmark measures:
- `construct_only`: time spent in `create_env(...)` only.
- `construct_plus_pool`: time spent in `create_env(...)` plus the first lazy
  `BatchEnvPool` materialization via `env.init_state()`.

Why variant count matters:
- Sharpa init-DR compiles `variant_count` scale-specific MuJoCo models.
- UniLab then expands env-to-variant assignments into a per-env model sequence
  before constructing `BatchEnvPool`.

Relevant `mujoco-uni` constraint from source:
- `BatchEnvPool(model=...)` accepts either
  - one `MjModel`, or
  - a sequence of `MjModel` with length `1` or `nbatch`.

See:
- `mujoco_uni/python/mujoco/batch_env.py`
- `mujoco_uni/python/mujoco/batch_env.cc`

Usage:
    uv run benchmark/benchmark_sharpa_init_dr_construct.py
    uv run benchmark/benchmark_sharpa_init_dr_construct.py --env-nums 256,512,1024
    uv run benchmark/benchmark_sharpa_init_dr_construct.py --variant-counts 1,2,4,8
    uv run benchmark/benchmark_sharpa_init_dr_construct.py --measure construct_plus_pool
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import tempfile
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, median, pstdev
from typing import Any, Iterator

import numpy as np
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "unilab-matplotlib"))

plt: Any | None = None
try:
    import matplotlib as _matplotlib

    _matplotlib.use("Agg")
    import matplotlib.pyplot as _plt

    plt = _plt
except Exception:
    plt = None

from benchmark.core.device_info import get_device_info_dict, get_device_info_line

DEFAULT_ENV_NUMS = [2**power for power in range(8, 14)]
DEFAULT_VARIANT_COUNTS = [128, 256]
DEFAULT_SCALE_LOWER = 0.5
DEFAULT_SCALE_UPPER = 0.8
DEFAULT_OUTPUT_DIR = ROOT_DIR / "benchmark" / "outputs" / "sharpa_init_dr_construct"


@dataclass
class ConstructRecord:
    measure: str
    mode: str
    variant_count: int
    num_envs: int
    scale_list: list[float]
    repeats: int
    samples_sec: list[float]
    mean_sec: float
    median_sec: float
    std_sec: float
    min_sec: float
    max_sec: float
    init_randomization_applied: bool
    model_variant_count: int
    pool_built: bool


def _parse_env_nums(value: str | None) -> list[int]:
    if value:
        env_nums = [int(part.strip()) for part in value.split(",") if part.strip()]
    else:
        env_nums = list(DEFAULT_ENV_NUMS)
    if not env_nums:
        raise ValueError("env nums cannot be empty")
    if any(num_envs <= 0 for num_envs in env_nums):
        raise ValueError(f"env nums must be positive, got {env_nums}")
    return env_nums


def _parse_variant_counts(value: str | None) -> list[int]:
    if value:
        counts = [int(part.strip()) for part in value.split(",") if part.strip()]
    else:
        counts = list(DEFAULT_VARIANT_COUNTS)
    if not counts:
        raise ValueError("variant counts cannot be empty")
    if any(count <= 0 for count in counts):
        raise ValueError(f"variant counts must be positive, got {counts}")
    deduped: list[int] = []
    for count in counts:
        if count not in deduped:
            deduped.append(count)
    return deduped


def _compose_cfg(task: str, *, lower: float, upper: float, variant_count: int):
    config_dir = str(ROOT_DIR / "conf" / "ppo")
    scale_list = np.linspace(lower, upper, variant_count, dtype=np.float64)
    scale_override = ",".join(f"{float(scale):g}" for scale in scale_list)
    overrides = [
        f"task={task}",
        f"env.domain_rand.scale_list=[{scale_override}]",
        "hydra.run.dir=.",
        "hydra.output_subdir=null",
        "hydra/job_logging=disabled",
        "hydra/hydra_logging=disabled",
    ]

    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=config_dir, version_base="1.3"):
        return compose(config_name="config", overrides=overrides)


@contextmanager
def _init_dr_mode(enabled: bool) -> Iterator[None]:
    from unilab.envs.manipulation.sharpa_inhand.rotation import SharpaInhandRotationDRProvider

    original = SharpaInhandRotationDRProvider.build_init_randomization_plan
    if enabled:
        yield
        return

    def disabled_build_init_randomization_plan(self: Any, env: Any) -> None:
        del self, env
        return None

    SharpaInhandRotationDRProvider.build_init_randomization_plan = (
        disabled_build_init_randomization_plan
    )
    try:
        yield
    finally:
        SharpaInhandRotationDRProvider.build_init_randomization_plan = original


@contextmanager
def _synthetic_grasp_cache_mode(enabled: bool) -> Iterator[None]:
    from unilab.envs.manipulation.sharpa_inhand.rotation import SharpaInhandRotationDRProvider

    original = SharpaInhandRotationDRProvider._load_grasp_cache
    if not enabled:
        yield
        return

    def synthetic_load_grasp_cache(self: Any, env: Any) -> np.ndarray:
        del self
        num_scales = int(env._num_scales)
        hand_qpos = np.asarray(env.default_angles, dtype=np.float64)
        object_height = 0.5 * (
            float(env.cfg.reset_height_lower) + float(env.cfg.reset_height_upper)
        )
        object_pose = np.array([0.0, 0.0, object_height, 1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        row = np.concatenate([hand_qpos, object_pose], axis=0)
        return np.repeat(row[None, :], num_scales, axis=0)

    SharpaInhandRotationDRProvider._load_grasp_cache = synthetic_load_grasp_cache
    try:
        yield
    finally:
        SharpaInhandRotationDRProvider._load_grasp_cache = original


def _cleanup_env(env: Any) -> None:
    backend = getattr(env, "_backend", None)
    pool = getattr(backend, "_pool", None)
    if pool is not None:
        pool.close()
        if backend is not None:
            backend._pool = None
    close = getattr(env, "close", None)
    if close is not None:
        close()
    del env
    gc.collect()


def _construct_once(
    cfg: Any,
    *,
    num_envs: int,
    init_dr_enabled: bool,
    force_pool: bool,
) -> tuple[float, dict[str, Any]]:
    from unilab.training import BackendAdapter, create_env, ensure_registries

    ensure_registries()

    adapter = BackendAdapter(cfg, root_dir=ROOT_DIR, algo_name="ppo")
    env_cfg_override = adapter.build_task_env_cfg_override()
    task_name = str(cfg.training.task_name)
    sim_backend = str(cfg.training.sim_backend)

    with _init_dr_mode(init_dr_enabled), _synthetic_grasp_cache_mode(force_pool):
        t0 = time.perf_counter()
        env = create_env(
            cfg,
            num_envs=num_envs,
            env_cfg_override=env_cfg_override,
            sim_backend=sim_backend,
            task_name=task_name,
        )
        if force_pool:
            env.init_state()
        elapsed = time.perf_counter() - t0

    backend = getattr(env, "_backend", None)
    meta = {
        "init_randomization_applied": bool(getattr(env, "_init_randomization_applied", False)),
        "model_variant_count": len(getattr(backend, "_model_variants", ())),
        "pool_built": getattr(backend, "_pool", None) is not None,
    }
    _cleanup_env(env)
    return elapsed, meta


def _summarize_record(
    *,
    measure: str,
    mode: str,
    variant_count: int,
    num_envs: int,
    lower: float,
    upper: float,
    samples: list[float],
    meta: dict[str, Any],
) -> ConstructRecord:
    return ConstructRecord(
        measure=measure,
        mode=mode,
        variant_count=variant_count,
        num_envs=num_envs,
        scale_list=np.linspace(lower, upper, variant_count, dtype=np.float64).tolist(),
        repeats=len(samples),
        samples_sec=[float(sample) for sample in samples],
        mean_sec=float(mean(samples)),
        median_sec=float(median(samples)),
        std_sec=float(pstdev(samples)) if len(samples) > 1 else 0.0,
        min_sec=float(min(samples)),
        max_sec=float(max(samples)),
        init_randomization_applied=bool(meta["init_randomization_applied"]),
        model_variant_count=int(meta["model_variant_count"]),
        pool_built=bool(meta["pool_built"]),
    )


def run_benchmark(
    *,
    task: str,
    measure: str,
    env_nums: list[int],
    variant_counts: list[int],
    scale_lower: float,
    scale_upper: float,
    repeats: int,
    warmup: int,
) -> list[ConstructRecord]:
    if repeats <= 0:
        raise ValueError(f"repeats must be positive, got {repeats}")
    if warmup < 0:
        raise ValueError(f"warmup must be non-negative, got {warmup}")
    if scale_lower <= 0.0 or scale_upper <= 0.0:
        raise ValueError(
            f"scale bounds must be positive, got lower={scale_lower}, upper={scale_upper}"
        )

    for variant_count in variant_counts:
        indivisible = [num_envs for num_envs in env_nums if num_envs % variant_count != 0]
        if indivisible:
            raise ValueError(
                f"All num_envs must be divisible by variant_count={variant_count}, got {indivisible}"
            )

    force_pool = measure == "construct_plus_pool"
    records: list[ConstructRecord] = []
    baseline_cfg = _compose_cfg(task, lower=scale_lower, upper=scale_lower, variant_count=1)
    variant_cfgs = {
        variant_count: _compose_cfg(
            task,
            lower=scale_lower,
            upper=scale_upper,
            variant_count=variant_count,
        )
        for variant_count in variant_counts
    }

    for num_envs in env_nums:
        print(f"\nnum_envs={num_envs}", flush=True)

        for warmup_idx in range(warmup):
            elapsed, _ = _construct_once(
                baseline_cfg,
                num_envs=num_envs,
                init_dr_enabled=False,
                force_pool=force_pool,
            )
            print(
                f"  [init_dr_off] warmup {warmup_idx + 1}/{warmup}: {elapsed:.3f}s",
                flush=True,
            )

        off_samples: list[float] = []
        off_meta: dict[str, Any] = {}
        for repeat_idx in range(repeats):
            elapsed, off_meta = _construct_once(
                baseline_cfg,
                num_envs=num_envs,
                init_dr_enabled=False,
                force_pool=force_pool,
            )
            off_samples.append(elapsed)
            print(
                f"  [init_dr_off] repeat {repeat_idx + 1}/{repeats}: {elapsed:.3f}s",
                flush=True,
            )
        records.append(
            _summarize_record(
                measure=measure,
                mode="init_dr_off",
                variant_count=1,
                num_envs=num_envs,
                lower=scale_lower,
                upper=scale_lower,
                samples=off_samples,
                meta=off_meta,
            )
        )

        for variant_count in variant_counts:
            cfg = variant_cfgs[variant_count]
            for warmup_idx in range(warmup):
                elapsed, _ = _construct_once(
                    cfg,
                    num_envs=num_envs,
                    init_dr_enabled=True,
                    force_pool=force_pool,
                )
                print(
                    f"  [init_dr_on:v{variant_count}] warmup {warmup_idx + 1}/{warmup}: {elapsed:.3f}s",
                    flush=True,
                )

            on_samples: list[float] = []
            on_meta: dict[str, Any] = {}
            for repeat_idx in range(repeats):
                elapsed, on_meta = _construct_once(
                    cfg,
                    num_envs=num_envs,
                    init_dr_enabled=True,
                    force_pool=force_pool,
                )
                on_samples.append(elapsed)
                print(
                    f"  [init_dr_on:v{variant_count}] repeat {repeat_idx + 1}/{repeats}: {elapsed:.3f}s",
                    flush=True,
                )
            records.append(
                _summarize_record(
                    measure=measure,
                    mode="init_dr_on",
                    variant_count=variant_count,
                    num_envs=num_envs,
                    lower=scale_lower,
                    upper=scale_upper,
                    samples=on_samples,
                    meta=on_meta,
                )
            )

    return records


def _record_map(records: list[ConstructRecord]) -> dict[tuple[str, int, int], ConstructRecord]:
    return {(record.mode, record.variant_count, record.num_envs): record for record in records}


def print_table(records: list[ConstructRecord]) -> None:
    env_nums = sorted({record.num_envs for record in records})
    variant_counts = sorted(
        {record.variant_count for record in records if record.mode == "init_dr_on"}
    )
    by_key = _record_map(records)

    print()
    header = f"{'num_envs':>8} | {'off_mean(s)':>11}"
    for variant_count in variant_counts:
        header += f" | {f'on_v{variant_count}(s)':>11}"
    print(header)
    print("-" * len(header))
    for num_envs in env_nums:
        off = by_key[("init_dr_off", 1, num_envs)]
        row = f"{num_envs:8d} | {off.mean_sec:11.3f}"
        for variant_count in variant_counts:
            on = by_key[("init_dr_on", variant_count, num_envs)]
            row += f" | {on.mean_sec:11.3f}"
        print(row)

    print()
    print(
        f"{'num_envs':>8} | {'variant':>7} | {'delta(s)':>9} | {'ratio':>7} | {'variants(on)':>12}"
    )
    print("-" * 64)
    for num_envs in env_nums:
        off = by_key[("init_dr_off", 1, num_envs)]
        for variant_count in variant_counts:
            on = by_key[("init_dr_on", variant_count, num_envs)]
            delta = on.mean_sec - off.mean_sec
            ratio = on.mean_sec / max(off.mean_sec, 1e-12)
            print(
                f"{num_envs:8d} | {variant_count:7d} | {delta:9.3f} | {ratio:7.2f} | {on.model_variant_count:12d}"
            )


def save_json(path: Path, records: list[ConstructRecord], meta: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "meta": {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "device_info": get_device_info_dict(),
            **meta,
        },
        "results": [asdict(record) for record in records],
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Saved: {path.resolve()}")


def save_plot(path: Path, records: list[ConstructRecord]) -> bool:
    if plt is None or not records:
        return False

    path.parent.mkdir(parents=True, exist_ok=True)
    env_nums = sorted({record.num_envs for record in records})
    variant_counts = sorted(
        {record.variant_count for record in records if record.mode == "init_dr_on"}
    )

    fig, (ax_env, ax_variant) = plt.subplots(1, 2, figsize=(15, 5))
    cmap = plt.get_cmap("tab10")

    off_subset = sorted(
        [record for record in records if record.mode == "init_dr_off"],
        key=lambda record: record.num_envs,
    )
    ax_env.plot(
        [record.num_envs for record in off_subset],
        [record.mean_sec for record in off_subset],
        marker="o",
        color="#2563eb",
        label="init DR off",
    )

    for color_idx, variant_count in enumerate(variant_counts):
        subset = sorted(
            [
                record
                for record in records
                if record.mode == "init_dr_on" and record.variant_count == variant_count
            ],
            key=lambda record: record.num_envs,
        )
        ax_env.plot(
            [record.num_envs for record in subset],
            [record.mean_sec for record in subset],
            marker="o",
            color=cmap(color_idx % 10),
            label=f"init DR on (variants={variant_count})",
        )

    for color_idx, num_envs in enumerate(env_nums):
        subset = sorted(
            [
                record
                for record in records
                if record.mode == "init_dr_on" and record.num_envs == num_envs
            ],
            key=lambda record: record.variant_count,
        )
        ax_variant.plot(
            [record.variant_count for record in subset],
            [record.mean_sec for record in subset],
            marker="s",
            color=cmap(color_idx % 10),
            label=f"num_envs={num_envs}",
        )

    ax_env.set_xscale("log", base=2)
    ax_env.set_yscale("log")
    ax_env.set_xticks(env_nums)
    ax_env.set_xticklabels([str(num_envs) for num_envs in env_nums], rotation=30)
    ax_env.set_xlabel("num_envs")
    ax_env.set_ylabel("construction time mean (s)")
    ax_env.set_title("Construction Time vs num_envs")
    ax_env.grid(True, alpha=0.3)
    ax_env.legend(fontsize=8)

    ax_variant.set_xscale("log", base=2)
    ax_variant.set_yscale("log")
    ax_variant.set_xticks(variant_counts)
    ax_variant.set_xticklabels([str(variant_count) for variant_count in variant_counts])
    ax_variant.set_xlabel("variant_count")
    ax_variant.set_ylabel("construction time mean (s)")
    ax_variant.set_title("Construction Time vs variant_count")
    ax_variant.grid(True, alpha=0.3)
    ax_variant.legend(fontsize=8)

    measure = records[0].measure
    title = f"Sharpa-hand init-DR construction benchmark ({measure})"
    device_info = get_device_info_line()
    fig.suptitle(f"{title}\n{device_info}" if device_info else title)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    print(f"Saved: {path.resolve()}")
    return True


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark Sharpa-hand construction time vs init-DR variant count"
    )
    parser.add_argument("--task", type=str, default="sharpa_inhand/mujoco")
    parser.add_argument("--measure", type=str, default="construct_only")
    parser.add_argument("--env-nums", type=str, default=None, help="Comma-separated env counts")
    parser.add_argument("--variant-counts", type=str, default=None)
    parser.add_argument("--scale-lower", type=float, default=DEFAULT_SCALE_LOWER)
    parser.add_argument("--scale-upper", type=float, default=DEFAULT_SCALE_UPPER)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=0)
    parser.add_argument(
        "--out-json",
        type=Path,
        default=DEFAULT_OUTPUT_DIR / "results.json",
    )
    parser.add_argument(
        "--out-png",
        type=Path,
        default=DEFAULT_OUTPUT_DIR / "construct_time.png",
    )
    args = parser.parse_args()

    if args.measure not in {"construct_only", "construct_plus_pool"}:
        raise ValueError(
            f"--measure must be one of {{'construct_only','construct_plus_pool'}}, got {args.measure}"
        )

    env_nums = _parse_env_nums(args.env_nums)
    variant_counts = _parse_variant_counts(args.variant_counts)

    records = run_benchmark(
        task=args.task,
        measure=args.measure,
        env_nums=env_nums,
        variant_counts=variant_counts,
        scale_lower=float(args.scale_lower),
        scale_upper=float(args.scale_upper),
        repeats=args.repeats,
        warmup=args.warmup,
    )
    print_table(records)
    save_json(
        args.out_json,
        records,
        meta={
            "task": args.task,
            "measure": args.measure,
            "env_nums": env_nums,
            "variant_counts": variant_counts,
            "scale_lower": float(args.scale_lower),
            "scale_upper": float(args.scale_upper),
            "repeats": args.repeats,
            "warmup": args.warmup,
            "batch_env_contract": {
                "source_py": "mujoco_uni/python/mujoco/batch_env.py",
                "source_cc": "mujoco_uni/python/mujoco/batch_env.cc",
                "accepted_model_arity": "single MjModel, or sequence length 1 or nbatch",
            },
            "init_dr_off_mode": "local monkeypatch: provider returns no InitRandomizationPlan",
        },
    )
    save_plot(args.out_png, records)


if __name__ == "__main__":
    main()
