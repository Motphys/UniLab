"""Owner-controlled state-update graph mode contract."""

from __future__ import annotations

from collections.abc import Mapping
from enum import Enum
from typing import Any


class EnvUpdateGraphMode(Enum):
    """Cold-path modes for the no-reset Manager state-update boundary."""

    DISABLED = "disabled"
    EAGER = "eager"
    GRAPH = "graph"


ENV_UPDATE_GRAPH_MODES = frozenset({mode.value for mode in EnvUpdateGraphMode})
ENV_UPDATE_GRAPH_MANIFEST_VERSION = 1
ENV_UPDATE_GRAPH_INVALIDATION_REASONS = frozenset(
    {"view", "capacity", "stream_event", "allocator", "capture_failed"}
)


def resolve_env_update_graph_mode(value: Any) -> EnvUpdateGraphMode:
    """Resolve and validate one owner-selected state-update execution mode."""
    if isinstance(value, bool):
        raise TypeError(
            "env update graph mode must be one of 'disabled', 'eager', or 'graph'; "
            f"got boolean {value!r}"
        )
    if not isinstance(value, str):
        raise TypeError(
            "env update graph mode must be one of 'disabled', 'eager', or 'graph'; "
            f"got {type(value).__name__}"
        )
    try:
        return EnvUpdateGraphMode(value.strip().lower())
    except ValueError as exc:
        choices = ", ".join(sorted(ENV_UPDATE_GRAPH_MODES))
        raise ValueError(f"env update graph mode must be one of {choices}; got {value!r}") from exc


def validate_env_update_graph_manifest(manifest: Any) -> None:
    """Fail closed on malformed state-update graph evidence."""
    if not isinstance(manifest, Mapping):
        raise TypeError("env update graph manifest must be a mapping")
    required = {
        "contract_version",
        "mode",
        "requested",
        "enabled",
        "state",
        "capture_id",
        "invalidation_reason",
        "disable_reason",
        "fallback_count",
        "replay_count",
        "backend_step_receipt",
        "update_completion",
        "rng_state",
        "carrier_layout",
    }
    missing = sorted(required - set(manifest))
    if missing:
        raise ValueError(f"env update graph manifest is missing fields: {missing}")
    if manifest["contract_version"] != ENV_UPDATE_GRAPH_MANIFEST_VERSION:
        raise ValueError(
            f"unsupported env update graph manifest contract {manifest['contract_version']!r}"
        )
    mode = manifest["mode"]
    if not isinstance(mode, str) or mode not in ENV_UPDATE_GRAPH_MODES:
        raise ValueError(f"env update graph manifest has unsupported mode {mode!r}")
    for name in ("requested", "enabled"):
        if not isinstance(manifest[name], bool):
            raise ValueError(f"env update graph manifest {name} must be a bool")
    for name in ("fallback_count", "replay_count"):
        value = manifest[name]
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"env update graph manifest {name} must be a non-negative integer")
    state = manifest["state"]
    if not isinstance(state, str) or state not in {"unavailable", "eager", "ready", "invalidated"}:
        raise ValueError(f"env update graph manifest has unsupported state {state!r}")
    capture_id = manifest["capture_id"]
    reason = manifest["disable_reason"]
    invalidation_reason = manifest["invalidation_reason"]
    if invalidation_reason is not None and (
        not isinstance(invalidation_reason, str)
        or invalidation_reason not in ENV_UPDATE_GRAPH_INVALIDATION_REASONS
    ):
        raise ValueError(
            f"env update graph manifest has unsupported invalidation_reason {invalidation_reason!r}"
        )
    if not isinstance(manifest["backend_step_receipt"], Mapping):
        raise ValueError("env update graph backend_step_receipt must be a mapping")
    if not isinstance(manifest["update_completion"], Mapping):
        raise ValueError("env update graph update_completion must be a mapping")
    if not isinstance(manifest["rng_state"], Mapping):
        raise ValueError("env update graph rng_state must be a mapping")
    if not isinstance(manifest["carrier_layout"], Mapping):
        raise ValueError("env update graph carrier_layout must be a mapping")
    if manifest["enabled"]:
        if mode != "graph" or not manifest["requested"] or state != "ready":
            raise ValueError("an enabled env update graph must be a requested, READY graph mode")
        if not isinstance(capture_id, str) or not capture_id.strip():
            raise ValueError("an enabled env update graph requires a non-empty capture_id")
        if reason is not None:
            raise ValueError("an enabled env update graph must not declare a disable reason")
        if invalidation_reason is not None:
            raise ValueError("an enabled env update graph must not declare an invalidation reason")
        if manifest["replay_count"] <= 0:
            raise ValueError("an enabled env update graph requires evidence of a real replay")
    else:
        if capture_id is not None:
            raise ValueError("a disabled env update graph must not publish a capture_id")
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError("a disabled env update graph requires a non-empty disable reason")
        if state == "invalidated" and invalidation_reason is None:
            raise ValueError("an invalidated env update graph requires an invalidation reason")
        if state != "invalidated" and invalidation_reason is not None:
            raise ValueError(
                "only an invalidated env update graph may declare an invalidation reason"
            )
        if manifest["replay_count"] != 0:
            raise ValueError("a disabled env update graph must not claim graph replays")


__all__ = [
    "ENV_UPDATE_GRAPH_INVALIDATION_REASONS",
    "ENV_UPDATE_GRAPH_MANIFEST_VERSION",
    "ENV_UPDATE_GRAPH_MODES",
    "EnvUpdateGraphMode",
    "resolve_env_update_graph_mode",
    "validate_env_update_graph_manifest",
]
