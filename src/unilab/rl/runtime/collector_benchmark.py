"""Versioned schema for the off-policy collector active-window benchmark."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

SCHEMA_NAME = "unilab.offpolicy-collector-active"
SCHEMA_VERSION = 2
SCHEMA_ID = f"{SCHEMA_NAME}/{SCHEMA_VERSION}"
ADDITIVE_PHASES = frozenset({"env_step_ms", "replay_ms", "bookkeeping_ms"})
LIFECYCLE_PHASES = frozenset(
    {
        "construction_ms",
        "collector_setup_ms",
        "warmup_ms",
        "total_active_ms",
        "teardown_ms",
    }
)
TIMING_STATISTICS = frozenset({"mean_ms", "p50_ms", "p95_ms", "min_ms", "max_ms"})


def _finite_number(value: Any, field: str) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be a number, got {value!r}")
    if not math.isfinite(float(value)):
        raise ValueError(f"{field} must be finite, got {value!r}")


def validate_result(result: Mapping[str, Any]) -> None:
    """Validate one serialized :class:`CollectorResult` without timing it."""

    case = result.get("case")
    if not isinstance(case, Mapping):
        raise ValueError("collector result requires a case mapping")
    for field in (
        "algo",
        "task",
        "sim",
        "runtime_sim_backend",
        "case_profile",
        "num_envs_source",
        "num_envs",
        "seed",
        "resolved_config_sha256",
    ):
        if field not in case:
            raise ValueError(f"collector case is missing {field}")
    if case["case_profile"] == "production":
        if (
            case["algo"] != "flashsac"
            or case["task"] != "g1_motion_tracking"
            or case["runtime_sim_backend"] != "genesis"
            or int(case["num_envs"]) != 2048
            or case["num_envs_source"] != "owner"
        ):
            raise ValueError("production Genesis collector profile is not frozen")

    phases = result.get("phase_ms_per_vector_step", {})
    if not isinstance(phases, Mapping) or set(phases) != set(ADDITIVE_PHASES):
        raise ValueError(
            f"additive collector phases must be exactly {sorted(ADDITIVE_PHASES)}, "
            f"got {sorted(phases) if isinstance(phases, Mapping) else phases!r}"
        )
    for name, stat in phases.items():
        if not isinstance(stat, Mapping) or not TIMING_STATISTICS.issubset(stat):
            raise ValueError(f"collector phase {name} lacks stable timing statistics")
        for statistic in TIMING_STATISTICS:
            _finite_number(stat[statistic], f"phase {name}.{statistic}")

    for field in LIFECYCLE_PHASES:
        _finite_number(result.get(field), f"lifecycle {field}")
    if result["total_active_ms"] <= 0.0:
        raise ValueError("measured active window must be positive")

    reset_rows = result.get("reset_rows_total")
    reset_steps = result.get("reset_vector_step_count")
    if isinstance(reset_rows, bool) or not isinstance(reset_rows, int) or reset_rows < 0:
        raise ValueError("reset_rows_total must be a non-negative integer")
    if isinstance(reset_steps, bool) or not isinstance(reset_steps, int) or reset_steps < 0:
        raise ValueError("reset_vector_step_count must be a non-negative integer")

    profile = result.get("backend_tensor_profile")
    if not isinstance(profile, Mapping):
        raise ValueError("collector result requires backend tensor-profile evidence")
    if case["case_profile"] == "production" and (
        profile.get("backend_type") != "genesis"
        or profile.get("execution") != "device_resident"
        or profile.get("data_plane") != "direct"
        or profile.get("process_topology") != "in_process"
    ):
        raise ValueError("production collector tensor profile is not exact CUDA DIRECT")


def validate_envelope(payload: Mapping[str, Any]) -> None:
    """Validate the complete machine-readable benchmark artifact."""

    if payload.get("schema") != SCHEMA_ID:
        raise ValueError(f"unsupported collector benchmark schema: {payload.get('schema')!r}")
    for field in ("benchmark", "command", "device", "fixation", "results", "errors"):
        if field not in payload:
            raise ValueError(f"collector benchmark envelope is missing {field}")
    results = payload["results"]
    if not isinstance(results, Sequence) or isinstance(results, (str, bytes)):
        raise ValueError("collector benchmark results must be a sequence")
    for result in results:
        validate_result(result)


__all__ = (
    "ADDITIVE_PHASES",
    "LIFECYCLE_PHASES",
    "SCHEMA_ID",
    "SCHEMA_NAME",
    "SCHEMA_VERSION",
    "TIMING_STATISTICS",
    "validate_envelope",
    "validate_result",
)
