"""Shared collector timing helpers."""

from __future__ import annotations

import math
from numbers import Real
from typing import Any

ENV_STEP_BREAKDOWN_TIMING_MAP = {
    "step_core_ms": "env_step_backend_ms",
    "update_state_ms": "env_step_update_state_ms",
    "reset_done_ms": "env_step_reset_done_ms",
}


def extract_env_step_breakdown_timing_ms(info: dict[str, Any] | None) -> dict[str, float]:
    """Extract env-owned step sub-timings for collector metrics."""
    if not isinstance(info, dict):
        return {}
    timing = info.get("timing")
    if not isinstance(timing, dict):
        return {}

    out: dict[str, float] = {}
    for source_key, collector_key in ENV_STEP_BREAKDOWN_TIMING_MAP.items():
        value = timing.get(source_key)
        if not isinstance(value, Real):
            continue
        value_float = float(value)
        if math.isfinite(value_float):
            out[collector_key] = value_float
    return out
