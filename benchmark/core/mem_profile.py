"""Process memory profiling helpers for benchmark scripts."""

from __future__ import annotations

import os
import subprocess
import sys
from typing import Any

BYTES_PER_MB = 1024.0 * 1024.0
MEMORY_METRICS = ("rss", "uss", "pss")


def current_memory_bytes() -> dict[str, Any]:
    try:
        import psutil

        process = psutil.Process()
        info = process.memory_info()
        result: dict[str, Any] = {
            "rss_bytes": int(info.rss),
            "memory_source": "psutil",
        }
        try:
            full_info = process.memory_full_info()
            uss = getattr(full_info, "uss", None)
            pss = getattr(full_info, "pss", None)
            if uss is not None:
                result["uss_bytes"] = int(uss)
            if pss is not None:
                result["pss_bytes"] = int(pss)
        except Exception:
            pass
        return result
    except Exception:
        pass

    if sys.platform.startswith("linux"):
        try:
            page_size = int(os.sysconf("SC_PAGE_SIZE"))
            with open("/proc/self/statm", encoding="utf-8") as f:
                parts = f.read().split()
            if len(parts) >= 2:
                return {
                    "rss_bytes": int(parts[1]) * page_size,
                    "memory_source": "procfs",
                }
        except Exception:
            pass

    try:
        result = subprocess.run(
            ["ps", "-o", "rss=", "-p", str(os.getpid())],
            check=True,
            capture_output=True,
            text=True,
        )
        return {
            "rss_bytes": int(result.stdout.strip()) * 1024,
            "memory_source": "ps",
        }
    except Exception:
        return {
            "rss_bytes": None,
            "memory_source": "unavailable",
        }


def peak_rss_bytes() -> int | None:
    try:
        import resource

        usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        if sys.platform == "darwin":
            return int(usage)
        return int(usage) * 1024
    except Exception:
        return None


def memory_snapshot(label: str) -> dict[str, Any]:
    memory = current_memory_bytes()
    source = memory.get("memory_source", "unavailable")
    return {
        "label": label,
        **memory,
        "rss_source": source,
        "peak_rss_bytes": peak_rss_bytes(),
    }


def bytes_delta(after: int | None, before: int | None) -> int | None:
    if after is None or before is None:
        return None
    return int(after) - int(before)


def bytes_per_env(value_bytes: int | None, num_envs: int) -> float | None:
    if value_bytes is None or num_envs <= 0:
        return None
    return float(value_bytes) / float(num_envs)


def mb(value_bytes: int | float | None) -> float | None:
    if value_bytes is None:
        return None
    return float(value_bytes) / BYTES_PER_MB


def format_mb(value_bytes: int | float | None, *, signed: bool = True) -> str:
    value_mb = mb(value_bytes)
    if value_mb is None:
        return "-"
    sign = "+" if signed else ""
    return f"{value_mb:{sign}.1f} MB"


def first_available_metric(samples: dict[str, dict[str, Any]]) -> str:
    for metric in ("uss", "pss", "rss"):
        if any(sample.get(f"{metric}_bytes") is not None for sample in samples.values()):
            return metric
    return "rss"


def build_memory_summary(
    samples: dict[str, dict[str, Any]],
    num_envs: int,
) -> dict[str, Any]:
    before_env = samples.get("before_env", {})
    after_benchmark = samples.get("after_benchmark", {})
    after_close = samples.get("after_close", {})
    preferred_metric = first_available_metric(samples)

    summary: dict[str, Any] = {
        "rss_source": after_benchmark.get("rss_source", "unavailable"),
        "memory_source": after_benchmark.get("memory_source", "unavailable"),
        "preferred_metric": preferred_metric,
        "before_env_rss_bytes": before_env.get("rss_bytes"),
        "after_benchmark_rss_bytes": after_benchmark.get("rss_bytes"),
        "after_close_rss_bytes": after_close.get("rss_bytes"),
        "after_benchmark_peak_rss_bytes": after_benchmark.get("peak_rss_bytes"),
        "samples": samples,
    }
    for metric in MEMORY_METRICS:
        key = f"{metric}_bytes"
        total_delta = bytes_delta(after_benchmark.get(key), before_env.get(key))
        close_delta = bytes_delta(after_close.get(key), before_env.get(key))

        summary[f"before_env_{metric}_bytes"] = before_env.get(key)
        summary[f"after_benchmark_{metric}_bytes"] = after_benchmark.get(key)
        summary[f"after_close_{metric}_bytes"] = after_close.get(key)
        summary[f"total_{metric}_delta_bytes"] = total_delta
        summary[f"retained_{metric}_delta_after_close_bytes"] = close_delta
        summary[f"total_{metric}_delta_bytes_per_env"] = bytes_per_env(total_delta, num_envs)
    return summary
