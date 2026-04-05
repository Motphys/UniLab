#!/usr/bin/env python3
"""
Benchmark MuJoCo MjModel memory cost.

The script isolates each measurement in a subprocess and reports:
- `MjModel.nbuffer` for one model and for N models
- process peak RSS deltas after constructing models
- the largest ndarray-backed model attributes for one representative model

This is intended to explain why creating per-env model copies can explode
memory usage when `num_envs` reaches the thousands.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import resource
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

try:
    import mujoco
except ImportError:
    mujoco = None


def _default_xml() -> str:
    candidates = [
        Path("src/unilab/assets/robots/go2/scene_flat.xml"),
        Path("benchmark/xmls/humanoid/humanoid.xml"),
    ]
    for path in candidates:
        resolved = (Path.cwd() / path).resolve()
        if resolved.exists():
            return str(resolved)
    return str((Path.cwd() / candidates[-1]).resolve())


def _resolve_xml_path(xml_arg: str) -> str:
    xml_path = Path(xml_arg)
    if xml_path.is_absolute():
        return str(xml_path)
    return str((Path.cwd() / xml_path).resolve())


def _rss_bytes() -> int:
    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":
        return int(usage)
    return int(usage) * 1024


def _mb(value_bytes: float) -> float:
    return float(value_bytes) / (1024.0 * 1024.0)


def _top_array_attrs(model: "mujoco.MjModel", limit: int) -> list[dict[str, object]]:
    items: list[dict[str, object]] = []
    for name in dir(model):
        if name.startswith("_"):
            continue
        try:
            value = getattr(model, name)
        except Exception:
            continue
        if isinstance(value, np.ndarray):
            items.append(
                {
                    "name": name,
                    "nbytes": int(value.nbytes),
                    "shape": list(value.shape),
                    "dtype": str(value.dtype),
                }
            )
    items.sort(key=lambda item: int(item["nbytes"]), reverse=True)
    return items[:limit]


def _collect_metrics(xml_path: str, num_models: int, top_k: int) -> dict[str, object]:
    if mujoco is None:
        raise RuntimeError("MuJoCo is unavailable in the current environment")

    gc.collect()
    rss_before = _rss_bytes()

    build_start = time.perf_counter()
    models = [mujoco.MjModel.from_xml_path(xml_path) for _ in range(num_models)]
    build_sec = max(time.perf_counter() - build_start, 1e-9)
    rss_after_build = _rss_bytes()

    first_model = models[0]
    nbuffer_values = [int(model.nbuffer) for model in models]
    nbuffer_total = sum(nbuffer_values)

    return {
        "xml_path": xml_path,
        "num_models": num_models,
        "build_sec": build_sec,
        "rss_before_bytes": rss_before,
        "rss_after_build_bytes": rss_after_build,
        "rss_delta_bytes": rss_after_build - rss_before,
        "peak_rss_bytes": rss_after_build,
        "nbuffer_bytes_per_model": int(first_model.nbuffer),
        "nbuffer_total_bytes": int(nbuffer_total),
        "rss_delta_bytes_per_model": float(rss_after_build - rss_before) / float(num_models),
        "model_dims": {
            "nq": int(first_model.nq),
            "nv": int(first_model.nv),
            "nu": int(first_model.nu),
            "nbody": int(first_model.nbody),
            "ngeom": int(first_model.ngeom),
            "njnt": int(first_model.njnt),
            "nsensordata": int(first_model.nsensordata),
        },
        "top_array_attrs": _top_array_attrs(first_model, top_k),
    }


def _run_isolated(
    script_path: Path, xml_path: str, num_models: int, top_k: int
) -> dict[str, object]:
    cmd = [
        sys.executable,
        str(script_path),
        "--xml",
        xml_path,
        "--num-models",
        str(num_models),
        "--top-k",
        str(top_k),
        "--emit-json",
    ]
    result = subprocess.run(cmd, check=True, capture_output=True, text=True)
    return json.loads(result.stdout)


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark MuJoCo MjModel memory usage")
    parser.add_argument(
        "--xml",
        type=str,
        default=_default_xml(),
        help="Path to XML model.",
    )
    parser.add_argument(
        "--num-models",
        type=int,
        default=512,
        help="Number of model copies to analyze in the batch-model case.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=12,
        help="How many largest ndarray attributes to report for a representative model.",
    )
    parser.add_argument(
        "--emit-json",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    args = parser.parse_args()

    xml_path = _resolve_xml_path(args.xml)
    if not os.path.exists(xml_path):
        raise FileNotFoundError(f"Model file not found: {xml_path}")
    if mujoco is None:
        raise RuntimeError("MuJoCo is unavailable in the current environment")

    if args.emit_json:
        result = _collect_metrics(xml_path, args.num_models, args.top_k)
        print(json.dumps(result))
        return

    script_path = Path(__file__).resolve()
    single = _run_isolated(script_path, xml_path, 1, args.top_k)
    many = _run_isolated(script_path, xml_path, args.num_models, args.top_k)

    single_rss_per_model = float(single["rss_delta_bytes_per_model"])
    many_rss_per_model = float(many["rss_delta_bytes_per_model"])
    nbuffer_per_model = int(single["nbuffer_bytes_per_model"])

    print(f"Model file: {xml_path}")
    print(f"Batch model copies: {args.num_models}")
    print(
        "Representative model dims: "
        f"nq={single['model_dims']['nq']}, "
        f"nv={single['model_dims']['nv']}, "
        f"nu={single['model_dims']['nu']}, "
        f"nbody={single['model_dims']['nbody']}, "
        f"ngeom={single['model_dims']['ngeom']}, "
        f"nsensordata={single['model_dims']['nsensordata']}"
    )
    print()
    print(
        f"{'Case':<18} | {'Models':<8} | {'Build(s)':<9} | {'nBuffer(MB)':<12} | {'RSS Delta(MB)':<14} | {'Per Model RSS(MB)':<17}"
    )
    print("-" * 92)
    for label, record in (("single_model", single), ("batch_models", many)):
        print(
            f"{label:<18} | "
            f"{record['num_models']:<8} | "
            f"{float(record['build_sec']):<9.4f} | "
            f"{_mb(int(record['nbuffer_total_bytes'])):<12.1f} | "
            f"{_mb(int(record['rss_delta_bytes'])):<14.1f} | "
            f"{_mb(float(record['rss_delta_bytes_per_model'])):<17.1f}"
        )

    print()
    print("Interpretation:")
    print(
        f"- `MjModel.nbuffer` per model: {_mb(nbuffer_per_model):.1f} MB "
        "(MuJoCo main model buffer only)"
    )
    print(
        f"- Observed RSS delta per model: "
        f"single={_mb(single_rss_per_model):.1f} MB, "
        f"batch={_mb(many_rss_per_model):.1f} MB"
    )
    print(
        f"- Extrapolated total for {args.num_models} distinct models: "
        f"nBuffer={_mb(nbuffer_per_model * args.num_models):.1f} MB, "
        f"RSS~={_mb(many_rss_per_model * args.num_models):.1f} MB"
    )

    print()
    print("Largest ndarray-backed fields in one representative `MjModel`:")
    print(f"{'Field':<30} | {'Size(MB)':<10} | {'Shape':<20} | {'Dtype':<10}")
    print("-" * 82)
    for item in single["top_array_attrs"]:
        print(
            f"{str(item['name']):<30} | "
            f"{_mb(int(item['nbytes'])):<10.1f} | "
            f"{str(tuple(item['shape'])):<20} | "
            f"{str(item['dtype']):<10}"
        )

    print()
    print(
        "Note: `nBuffer` is the core MuJoCo model buffer. RSS delta is larger because "
        "Python objects, allocator behavior, and ancillary native allocations are included."
    )


if __name__ == "__main__":
    main()
