#!/usr/bin/env python3
"""
Benchmark MuJoCo rollout throughput for two batching strategies:

1. UniLab backend style: one shared MjModel broadcast across the batch.
2. Per-env model style: one distinct MjModel per env in the batch.

Single-model runs can optionally compare the same workload before/after
`compiler.discardvisual=true`. Multi-robot batch runs default to
`discardvisual` only to keep memory usage bounded.

This isolates rollout throughput only. Model construction is done before timing.
"""

from __future__ import annotations

import argparse
import gc
import importlib.util
import json
import os
import resource
import subprocess
import sys
import tempfile
import time
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from multiprocessing import cpu_count
from pathlib import Path
from typing import Sequence

import numpy as np

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
except Exception:
    plt = None
    Patch = None

try:
    import mujoco
    import mujoco.rollout
except ImportError:
    mujoco = None


DEFAULT_ROBOT_XMLS = {
    "go1": Path("src/unilab/assets/robots/go1/scene_flat.xml"),
    "go2": Path("src/unilab/assets/robots/go2/scene_flat.xml"),
    "g1": Path("src/unilab/assets/robots/g1/scene_flat.xml"),
}
DEFAULT_OUTPUT_DIR = Path("benchmark/outputs/mujoco_rollout_model_sharing")
MODE_COLORS = {
    "shared_single_model": "#2563eb",
    "distinct_batch_models": "#f97316",
}
MODE_LABELS = {
    "shared_single_model": "shared single model",
    "distinct_batch_models": "distinct batch models",
}
PLOT_BG = "#f8fafc"
PLOT_GRID = "#cbd5e1"
PLOT_TEXT = "#0f172a"


def _load_device_info_helpers():
    device_info_path = Path(__file__).resolve().parent / "core" / "device_info.py"
    spec = importlib.util.spec_from_file_location("_local_benchmark_device_info", device_info_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load device info helpers from {device_info_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.get_device_info_dict, module.get_device_info_line


_get_device_info_dict, _get_device_info_line = _load_device_info_helpers()


def _resolve_xml_path(xml_arg: str) -> str:
    xml_path = Path(xml_arg)
    if xml_path.is_absolute():
        return str(xml_path)
    return str((Path.cwd() / xml_path).resolve())


def _save_json(
    path: Path, results: Sequence[dict[str, int | float | str]], meta: dict[str, object]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "meta": {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            **meta,
        },
        "results": list(results),
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Saved: {path.resolve()}")


def _resolve_benchmark_targets(xml_arg: str | None, robots_arg: str) -> list[tuple[str, str]]:
    if xml_arg:
        xml_path = _resolve_xml_path(xml_arg)
        label = Path(xml_path).parent.name or Path(xml_path).stem
        return [(label, xml_path)]

    robot_names = [name.strip() for name in robots_arg.split(",") if name.strip()]
    if not robot_names:
        raise ValueError("No robots specified")

    targets: list[tuple[str, str]] = []
    available = {
        name: _resolve_xml_path(str(relative_path))
        for name, relative_path in DEFAULT_ROBOT_XMLS.items()
    }
    for name in robot_names:
        if name not in available:
            raise ValueError(f"Unknown robot '{name}'. Available: {sorted(available)}")
        targets.append((name, available[name]))
    return targets


def _default_nthread(batch_size: int) -> int:
    return min(batch_size, cpu_count() * 2)


def _rss_bytes() -> int:
    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":
        return int(usage)
    return int(usage) * 1024


def _mb_str(value_bytes: int) -> str:
    return f"{value_bytes / (1024 * 1024):.1f}"


def _initial_state_and_ctrl(
    model: "mujoco.MjModel", batch_size: int
) -> tuple[np.ndarray, np.ndarray]:
    data = mujoco.MjData(model)
    if model.nkey > 0:
        mujoco.mj_resetDataKeyframe(model, data, 0)
    else:
        mujoco.mj_resetData(model, data)

    nstate = mujoco.mj_stateSize(model, mujoco.mjtState.mjSTATE_FULLPHYSICS)
    state0 = np.empty((nstate,), dtype=np.float64)
    mujoco.mj_getState(model, data, state0, mujoco.mjtState.mjSTATE_FULLPHYSICS)

    initial_state = np.empty((batch_size, nstate), dtype=np.float64)
    initial_state[:] = state0

    ctrl0 = np.zeros((model.nu,), dtype=np.float64)
    if model.nkey > 0 and model.nu > 0:
        ctrl0[:] = np.asarray(model.key_ctrl[0], dtype=np.float64)

    control = np.empty((batch_size, 1, model.nu), dtype=np.float64)
    control[:] = ctrl0.reshape((1, 1, model.nu))
    return initial_state, control


def _build_model_batch(
    xml_path: str, batch_size: int, distinct_models: bool
) -> tuple["mujoco.MjModel | Sequence[mujoco.MjModel]", "mujoco.MjModel"]:
    base_model = mujoco.MjModel.from_xml_path(xml_path)
    if not distinct_models:
        return [base_model] * batch_size, base_model

    fd, mjb_path = tempfile.mkstemp(suffix=".mjb", dir=os.path.dirname(os.path.abspath(xml_path)))
    os.close(fd)
    try:
        mujoco.mj_saveModel(base_model, mjb_path)
        model_batch = [mujoco.MjModel.from_binary_path(mjb_path) for _ in range(batch_size)]
    finally:
        os.remove(mjb_path)
    return model_batch, model_batch[0]


def _materialize_discardvisual_xml(xml_path: str) -> tuple[str, str]:
    """Create an include-expanded XML variant with compiler.discardvisual enabled."""
    out_dir = os.path.dirname(os.path.abspath(xml_path))
    base_model = mujoco.MjModel.from_xml_path(xml_path)

    fd, expanded_path = tempfile.mkstemp(suffix=".xml", dir=out_dir)
    os.close(fd)
    mujoco.mj_saveLastXML(expanded_path, base_model)

    root = ET.parse(expanded_path).getroot()
    compiler = root.find("compiler")
    if compiler is None:
        compiler = ET.Element("compiler")
        root.insert(0, compiler)
    compiler.set("discardvisual", "true")

    fd, discardvisual_path = tempfile.mkstemp(suffix=".xml", dir=out_dir)
    os.close(fd)
    ET.indent(root, space="  ")
    ET.ElementTree(root).write(discardvisual_path)
    return discardvisual_path, expanded_path


def run_rollout_benchmark(
    xml_path: str,
    batch_size: int,
    steps: int,
    nthread: int,
    warmup: int,
    distinct_models: bool,
) -> tuple[float, float]:
    if mujoco is None or not hasattr(mujoco, "rollout"):
        raise RuntimeError("MuJoCo rollout is unavailable in the current environment")

    model_batch, worker_model = _build_model_batch(
        xml_path=xml_path, batch_size=batch_size, distinct_models=distinct_models
    )
    worker_data = [mujoco.MjData(worker_model) for _ in range(nthread)]
    initial_state, control = _initial_state_and_ctrl(worker_model, batch_size)
    nstate = initial_state.shape[-1]
    nsensordata = worker_model.nsensordata

    state_buf = np.empty((batch_size, 1, nstate), dtype=np.float64)
    sensor_buf = np.empty((batch_size, 1, nsensordata), dtype=np.float64)

    with mujoco.rollout.Rollout(nthread=nthread) as runner:
        for _ in range(warmup):
            state_traj, _ = runner.rollout(
                model_batch,
                worker_data,
                initial_state,
                control,
                skip_checks=True,
                nstep=1,
                state=state_buf,
                sensordata=sensor_buf,
            )
            initial_state[:] = state_traj[:, -1, :]

        start = time.perf_counter()
        for _ in range(steps):
            state_traj, _ = runner.rollout(
                model_batch,
                worker_data,
                initial_state,
                control,
                skip_checks=True,
                nstep=1,
                state=state_buf,
                sensordata=sensor_buf,
            )
            initial_state[:] = state_traj[:, -1, :]
        elapsed = max(time.perf_counter() - start, 1e-9)

    sps = (batch_size * steps) / elapsed
    return elapsed, sps


def collect_mode_metrics(
    xml_path: str,
    batch_size: int,
    steps: int,
    nthread: int,
    warmup: int,
    distinct_models: bool,
    variant: str,
) -> dict[str, int | float | str]:
    if mujoco is None or not hasattr(mujoco, "rollout"):
        raise RuntimeError("MuJoCo rollout is unavailable in the current environment")

    variant_xml_path = xml_path
    cleanup_paths: list[str] = []
    if variant == "discardvisual":
        variant_xml_path, expanded_path = _materialize_discardvisual_xml(xml_path)
        cleanup_paths.extend([variant_xml_path, expanded_path])

    try:
        gc.collect()
        rss_before = _rss_bytes()

        build_start = time.perf_counter()
        model_batch, worker_model = _build_model_batch(
            xml_path=variant_xml_path,
            batch_size=batch_size,
            distinct_models=distinct_models,
        )
        build_elapsed = max(time.perf_counter() - build_start, 1e-9)
        rss_after_model_build = _rss_bytes()

        setup_start = time.perf_counter()
        worker_data = [mujoco.MjData(worker_model) for _ in range(nthread)]
        initial_state, control = _initial_state_and_ctrl(worker_model, batch_size)
        nstate = initial_state.shape[-1]
        nsensordata = worker_model.nsensordata
        state_buf = np.empty((batch_size, 1, nstate), dtype=np.float64)
        sensor_buf = np.empty((batch_size, 1, nsensordata), dtype=np.float64)
        setup_elapsed = max(time.perf_counter() - setup_start, 1e-9)
        rss_after_setup = _rss_bytes()
        peak_rss = max(rss_before, rss_after_model_build, rss_after_setup)

        with mujoco.rollout.Rollout(nthread=nthread) as runner:
            for _ in range(warmup):
                state_traj, _ = runner.rollout(
                    model_batch,
                    worker_data,
                    initial_state,
                    control,
                    skip_checks=True,
                    nstep=1,
                    state=state_buf,
                    sensordata=sensor_buf,
                )
                initial_state[:] = state_traj[:, -1, :]
            peak_rss = max(peak_rss, _rss_bytes())

            start = time.perf_counter()
            for _ in range(steps):
                state_traj, _ = runner.rollout(
                    model_batch,
                    worker_data,
                    initial_state,
                    control,
                    skip_checks=True,
                    nstep=1,
                    state=state_buf,
                    sensordata=sensor_buf,
                )
                initial_state[:] = state_traj[:, -1, :]
            elapsed = max(time.perf_counter() - start, 1e-9)
            peak_rss = max(peak_rss, _rss_bytes())
    finally:
        for path in cleanup_paths:
            if os.path.exists(path):
                os.remove(path)

    sps = (batch_size * steps) / elapsed
    return {
        "variant": variant,
        "mode": "distinct_batch_models" if distinct_models else "shared_single_model",
        "models": batch_size if distinct_models else 1,
        "batch": batch_size,
        "threads": nthread,
        "build_sec": build_elapsed,
        "setup_sec": setup_elapsed,
        "rollout_sec": elapsed,
        "sps": sps,
        "rss_before_bytes": rss_before,
        "rss_after_model_build_bytes": rss_after_model_build,
        "rss_after_setup_bytes": rss_after_setup,
        "peak_rss_bytes": peak_rss,
        "rss_model_delta_bytes": rss_after_model_build - rss_before,
        "rss_setup_delta_bytes": rss_after_setup - rss_before,
    }


def _run_isolated_mode(
    script_path: Path,
    xml_path: str,
    batch_size: int,
    steps: int,
    warmup: int,
    nthread: int,
    mode: str,
    variant: str,
) -> dict[str, int | float | str]:
    cmd = [
        sys.executable,
        str(script_path),
        "--xml",
        xml_path,
        "--num-envs",
        str(batch_size),
        "--steps",
        str(steps),
        "--warmup",
        str(warmup),
        "--nthread",
        str(nthread),
        "--mode",
        mode,
        "--variant",
        variant,
        "--emit-json",
    ]
    result = subprocess.run(cmd, check=True, capture_output=True, text=True)
    return json.loads(result.stdout)


def _run_suite_for_target(
    script_path: Path,
    target_name: str,
    xml_path: str,
    batch_size: int,
    steps: int,
    warmup: int,
    nthread: int,
    mode: str,
    variant: str,
) -> list[dict[str, int | float | str]]:
    if mode != "both":
        if variant == "both":
            raise ValueError("--variant=both requires --mode=both")
        results = [
            collect_mode_metrics(
                xml_path=xml_path,
                batch_size=batch_size,
                steps=steps,
                nthread=nthread,
                warmup=warmup,
                distinct_models=mode == "distinct_batch_models",
                variant=variant,
            )
        ]
    else:
        variants = ["original", "discardvisual"] if variant == "both" else [variant]
        results = [
            _run_isolated_mode(
                script_path=script_path,
                xml_path=xml_path,
                batch_size=batch_size,
                steps=steps,
                warmup=warmup,
                nthread=nthread,
                mode=run_mode,
                variant=run_variant,
            )
            for run_variant in variants
            for run_mode in ("shared_single_model", "distinct_batch_models")
        ]

    for result in results:
        result["robot"] = target_name
        result["xml_path"] = xml_path
    return results


def _print_results_for_target(
    target_name: str,
    xml_path: str,
    results: Sequence[dict[str, int | float | str]],
    batch_size: int,
    warmup: int,
    steps: int,
    nthread: int,
) -> None:
    variants = list(dict.fromkeys(str(result["variant"]) for result in results))

    print(f"=== {target_name} ===")
    print(f"Model file: {xml_path}")
    print(f"Number of environments: {batch_size}")
    print(f"Warmup iterations: {warmup}")
    print(f"Timed iterations: {steps}")
    print(f"Worker threads: {nthread}")
    print(f"XML variant(s): {', '.join(variants)}")
    print()
    print("UniLab MuJoCo backend parallelism:")
    print("- one shared MjModel for all envs")
    print("- nthread worker MjData objects, not num_envs worker objects")
    print("- batched full-physics states passed into mujoco.rollout.Rollout")
    if "discardvisual" in variants:
        print("- discardvisual variant uses include-expanded XML with compiler.discardvisual=true")
    print()
    print(
        f"{'Variant':<14} | {'Mode':<24} | {'Models':<8} | {'Build(s)':<8} | {'RSS+Model':<10} | {'RSS+Setup':<10} | {'PeakRSS':<8} | {'SPS':<12} | {'Time(s)':<8}"
    )
    print("-" * 133)

    for result in results:
        print(
            f"{result['variant']:<14} | "
            f"{result['mode']:<24} | "
            f"{result['models']:<8} | "
            f"{result['build_sec']:<8.4f} | "
            f"{_mb_str(int(result['rss_model_delta_bytes'])):<10} | "
            f"{_mb_str(int(result['rss_setup_delta_bytes'])):<10} | "
            f"{_mb_str(int(result['peak_rss_bytes'])):<8} | "
            f"{result['sps']:<12.1f} | "
            f"{result['rollout_sec']:<8.4f}"
        )

    print()
    modes = {str(result["mode"]) for result in results}
    for variant in variants:
        if modes != {"shared_single_model", "distinct_batch_models"}:
            continue
        shared = next(
            r for r in results if r["variant"] == variant and r["mode"] == "shared_single_model"
        )
        distinct = next(
            r for r in results if r["variant"] == variant and r["mode"] == "distinct_batch_models"
        )
        print(
            f"[{target_name}][{variant}] Relative throughput: "
            f"shared/distinct = {float(shared['sps']) / float(distinct['sps']):.3f}x, "
            f"distinct/shared time = "
            f"{float(distinct['rollout_sec']) / float(shared['rollout_sec']):.3f}x"
        )
        print(
            f"[{target_name}][{variant}] Model build time ratio: "
            f"distinct/shared = {float(distinct['build_sec']) / float(shared['build_sec']):.3f}x"
        )
        print(
            f"[{target_name}][{variant}] Setup memory delta ratio: "
            f"distinct/shared = "
            f"{float(distinct['rss_setup_delta_bytes']) / max(float(shared['rss_setup_delta_bytes']), 1.0):.3f}x"
        )

    if len(variants) == 2 and modes == {"shared_single_model", "distinct_batch_models"}:
        print()
        for mode_name in ("shared_single_model", "distinct_batch_models"):
            original = next(
                r for r in results if r["variant"] == "original" and r["mode"] == mode_name
            )
            discardvisual = next(
                r for r in results if r["variant"] == "discardvisual" and r["mode"] == mode_name
            )
            print(
                f"[{target_name}][{mode_name}] discardvisual/original SPS = "
                f"{float(discardvisual['sps']) / float(original['sps']):.3f}x, "
                f"PeakRSS = "
                f"{float(discardvisual['peak_rss_bytes']) / max(float(original['peak_rss_bytes']), 1.0):.3f}x"
            )

    print("RSS uses per-process peak resident memory (`ru_maxrss`) inside each isolated mode.")
    print()


def _record_label(record: dict[str, int | float | str]) -> str:
    robot = str(record["robot"])
    variant = str(record["variant"])
    return robot if variant == "discardvisual" else f"{robot}\n{variant}"


def _style_axis(ax) -> None:
    ax.set_facecolor(PLOT_BG)
    ax.grid(axis="y", color=PLOT_GRID, alpha=0.6, linewidth=0.8)
    ax.tick_params(colors=PLOT_TEXT)
    ax.yaxis.label.set_color(PLOT_TEXT)
    ax.xaxis.label.set_color(PLOT_TEXT)
    ax.title.set_color(PLOT_TEXT)
    for spine in ax.spines.values():
        spine.set_color(PLOT_GRID)


def _plot_summary_plots(
    results: Sequence[dict[str, int | float | str]],
    out_dir: Path,
    batch_size: int,
    warmup: int,
    steps: int,
    nthread: int,
    variant: str,
) -> None:
    if plt is None or not results:
        print("Skipping plots: matplotlib unavailable or no results.")
        return

    mode_order = ["shared_single_model", "distinct_batch_models"]
    groups = list(dict.fromkeys(_record_label(record) for record in results))
    group_x = np.arange(len(groups), dtype=np.float64) * 1.5
    bar_width = 0.22
    offsets = {
        "shared_single_model": -bar_width * 0.55,
        "distinct_batch_models": bar_width * 0.55,
    }
    legend_handles = (
        [Patch(facecolor=MODE_COLORS[mode], label=label) for mode, label in MODE_LABELS.items()]
        if Patch is not None
        else []
    )
    run_info = (
        f"Batch size={batch_size} envs | Warmup={warmup} | Timed iterations={steps} | "
        f"Threads={nthread} | Variant={variant}"
    )

    throughput_fig, throughput_ax = plt.subplots(
        figsize=(max(8, len(groups) * 2.8), 5.2),
        facecolor=PLOT_BG,
    )
    for mode in mode_order:
        mode_records = {
            _record_label(record): record for record in results if record["mode"] == mode
        }
        x = np.array([gx + offsets[mode] for gx in group_x], dtype=np.float64)
        y = np.array([float(mode_records[label]["sps"]) for label in groups], dtype=np.float64)
        throughput_ax.bar(
            x,
            y,
            color=MODE_COLORS[mode],
            width=bar_width,
            label=MODE_LABELS[mode],
        )
    throughput_ax.set_xticks(group_x)
    throughput_ax.set_xticklabels(groups)
    throughput_ax.set_ylabel("Steps Per Second")
    throughput_ax.set_xlabel("Robot")
    throughput_ax.set_title(f"Throughput\n{run_info}", fontsize=10)
    _style_axis(throughput_ax)
    if legend_handles:
        throughput_ax.legend(handles=legend_handles, frameon=False, loc="upper right")
    throughput_fig.suptitle(
        f"MuJoCo Rollout Model Sharing\n{_get_device_info_line()}",
        y=0.98,
        fontsize=13,
    )
    throughput_fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.9])
    throughput_path = out_dir / "throughput_sps.png"
    throughput_fig.savefig(throughput_path, dpi=160)
    plt.close(throughput_fig)
    print(f"Saved: {throughput_path.resolve()}")

    memory_fig, axes = plt.subplots(
        2,
        1,
        figsize=(max(8, len(groups) * 2.8), 8.6),
        facecolor=PLOT_BG,
        sharex=True,
    )
    for mode in mode_order:
        mode_records = {
            _record_label(record): record for record in results if record["mode"] == mode
        }
        x = np.array([gx + offsets[mode] for gx in group_x], dtype=np.float64)
        setup_rss_mb = np.array(
            [
                float(mode_records[label]["rss_setup_delta_bytes"]) / (1024.0 * 1024.0)
                for label in groups
            ],
            dtype=np.float64,
        )
        peak_rss_mb = np.array(
            [float(mode_records[label]["peak_rss_bytes"]) / (1024.0 * 1024.0) for label in groups],
            dtype=np.float64,
        )
        axes[0].bar(
            x, setup_rss_mb, color=MODE_COLORS[mode], width=bar_width, label=MODE_LABELS[mode]
        )
        axes[1].bar(
            x, peak_rss_mb, color=MODE_COLORS[mode], width=bar_width, label=MODE_LABELS[mode]
        )

    axes[0].set_ylabel("MB")
    axes[0].set_title(f"RSS Delta After Setup\n{run_info}", fontsize=10)
    _style_axis(axes[0])
    if legend_handles:
        axes[0].legend(handles=legend_handles, frameon=False, loc="upper right")

    axes[1].set_xticks(group_x)
    axes[1].set_xticklabels(groups)
    axes[1].set_xlabel("Robot")
    axes[1].set_ylabel("MB")
    axes[1].set_title("Peak RSS")
    _style_axis(axes[1])

    memory_fig.suptitle(
        f"MuJoCo Rollout Model Sharing Memory\n{_get_device_info_line()}",
        y=0.985,
        fontsize=13,
    )
    memory_fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.92])
    memory_path = out_dir / "memory_mb.png"
    memory_fig.savefig(memory_path, dpi=160)
    plt.close(memory_fig)
    print(f"Saved: {memory_path.resolve()}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark MuJoCo rollout on go1/go2/g1: shared single model vs "
            "per-env distinct models, optionally before/after discardvisual"
        )
    )
    parser.add_argument(
        "--xml",
        type=str,
        default=None,
        help="Optional single XML model path. If omitted, benchmark the default robot set.",
    )
    parser.add_argument(
        "--robots",
        type=str,
        default="go1,go2,g1",
        help="Comma separated robot names to benchmark from src/unilab/assets/robots.",
    )
    parser.add_argument(
        "--num-envs",
        type=int,
        default=4096,
        help="Number of environments (batch size).",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=50,
        help="Timed rollout iterations.",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=5,
        help="Warmup rollout iterations.",
    )
    parser.add_argument(
        "--nthread",
        type=int,
        default=None,
        help="MuJoCo rollout worker threads. Defaults to UniLab backend policy: min(num_envs, cpu_count * 2).",
    )
    parser.add_argument(
        "--mode",
        choices=["both", "shared_single_model", "distinct_batch_models"],
        default="both",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--variant",
        choices=["both", "original", "discardvisual"],
        default="discardvisual",
        help=(
            "XML variant to benchmark. Multi-robot batch runs only allow "
            "`discardvisual`; use --xml for single-model `original` or `both`."
        ),
    )
    parser.add_argument(
        "--emit-json",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory for saved benchmark results.",
    )
    args = parser.parse_args()

    if mujoco is None or not hasattr(mujoco, "rollout"):
        raise RuntimeError("MuJoCo rollout is unavailable in the current environment")

    nthread = args.nthread if args.nthread is not None else _default_nthread(args.num_envs)
    targets = _resolve_benchmark_targets(args.xml, args.robots)
    for _, xml_path in targets:
        if not os.path.exists(xml_path):
            raise FileNotFoundError(f"Model file not found: {xml_path}")
    if args.xml is None and args.variant != "discardvisual":
        raise ValueError(
            "Multi-robot batch runs only support --variant=discardvisual. "
            "Use --xml for single-model original/both benchmarks."
        )

    if args.emit_json:
        if len(targets) != 1:
            raise ValueError("--emit-json requires exactly one target; pass --xml or one robot")
        if args.mode == "both" or args.variant == "both":
            raise ValueError("--emit-json requires a single mode and a single variant")
        target_name, xml_path = targets[0]
        result = collect_mode_metrics(
            xml_path=xml_path,
            batch_size=args.num_envs,
            steps=args.steps,
            nthread=nthread,
            warmup=args.warmup,
            distinct_models=args.mode == "distinct_batch_models",
            variant=args.variant,
        )
        result["robot"] = target_name
        result["xml_path"] = xml_path
        print(json.dumps(result))
        return

    script_path = Path(__file__).resolve()
    all_results: list[dict[str, int | float | str]] = []
    per_target_results: dict[str, list[dict[str, int | float | str]]] = {}

    for target_name, xml_path in targets:
        results = _run_suite_for_target(
            script_path=script_path,
            target_name=target_name,
            xml_path=xml_path,
            batch_size=args.num_envs,
            steps=args.steps,
            warmup=args.warmup,
            nthread=nthread,
            mode=args.mode,
            variant=args.variant,
        )
        per_target_results[target_name] = results
        all_results.extend(results)
        _print_results_for_target(
            target_name=target_name,
            xml_path=xml_path,
            results=results,
            batch_size=args.num_envs,
            warmup=args.warmup,
            steps=args.steps,
            nthread=nthread,
        )

    out_dir = Path(args.out_dir)
    _save_json(
        out_dir / "results.json",
        all_results,
        {
            "device_info": _get_device_info_dict(),
            "targets": [{"robot": name, "xml_path": xml_path} for name, xml_path in targets],
            "num_envs": args.num_envs,
            "steps": args.steps,
            "warmup": args.warmup,
            "nthread": nthread,
            "mode": args.mode,
            "variant": args.variant,
        },
    )
    for target_name, results in per_target_results.items():
        xml_path = next(path for name, path in targets if name == target_name)
        _save_json(
            out_dir / f"{target_name}_results.json",
            results,
            {
                "device_info": _get_device_info_dict(),
                "robot": target_name,
                "xml_path": xml_path,
                "num_envs": args.num_envs,
                "steps": args.steps,
                "warmup": args.warmup,
                "nthread": nthread,
                "mode": args.mode,
                "variant": args.variant,
            },
        )

    _plot_summary_plots(
        all_results,
        out_dir,
        batch_size=args.num_envs,
        warmup=args.warmup,
        steps=args.steps,
        nthread=nthread,
        variant=args.variant,
    )
    print(f"Saved results to {out_dir.resolve()}")


if __name__ == "__main__":
    main()
