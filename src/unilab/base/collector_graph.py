"""Owner-controlled collector graph mode contract."""

from __future__ import annotations

from collections.abc import Mapping
from enum import Enum
from typing import Any


class CollectorGraphMode(Enum):
    """Cold-path execution modes for the bounded K=1 collector boundary."""

    DISABLED = "disabled"
    EAGER = "eager"
    GRAPH = "graph"


COLLECTOR_GRAPH_MODES = frozenset({mode.value for mode in CollectorGraphMode})
COLLECTOR_GRAPH_MANIFEST_VERSION = 1
COLLECTOR_GRAPH_INVALIDATION_REASONS = frozenset(
    {"model", "data", "view", "capacity", "policy_io", "stream_event", "capture_failed"}
)


def resolve_collector_graph_mode(value: Any) -> CollectorGraphMode:
    """Resolve and validate one owner-selected collector execution mode."""
    if isinstance(value, bool):
        raise TypeError(
            "collector graph mode must be one of 'disabled', 'eager', or 'graph'; "
            f"got boolean {value!r}"
        )
    if not isinstance(value, str):
        raise TypeError(
            "collector graph mode must be one of 'disabled', 'eager', or 'graph'; "
            f"got {type(value).__name__}"
        )
    try:
        return CollectorGraphMode(value.strip().lower())
    except ValueError as exc:
        choices = ", ".join(sorted(COLLECTOR_GRAPH_MODES))
        raise ValueError(f"collector graph mode must be one of {choices}; got {value!r}") from exc


def validate_collector_graph_manifest(manifest: Any) -> None:
    """Fail closed on malformed collector graph mode evidence."""
    if not isinstance(manifest, Mapping):
        raise TypeError("collector graph manifest must be a mapping")
    required = {
        "contract_version",
        "mode",
        "invalidation_reason",
        "requested",
        "enabled",
        "state",
        "capture_id",
        "disable_reason",
        "fallback_count",
        "replay_count",
        "replay_publish_count",
        "replay_release_count",
    }
    missing = sorted(required - set(manifest))
    if missing:
        raise ValueError(f"collector graph manifest is missing fields: {missing}")
    if manifest["contract_version"] != COLLECTOR_GRAPH_MANIFEST_VERSION:
        raise ValueError(
            f"unsupported collector graph manifest contract {manifest['contract_version']!r}"
        )
    mode = manifest["mode"]
    if not isinstance(mode, str) or mode not in COLLECTOR_GRAPH_MODES:
        raise ValueError(f"collector graph manifest has unsupported mode {mode!r}")
    for name in ("requested", "enabled"):
        if not isinstance(manifest[name], bool):
            raise ValueError(f"collector graph manifest {name} must be a bool")
    for name in (
        "fallback_count",
        "replay_count",
        "replay_publish_count",
        "replay_release_count",
    ):
        value = manifest[name]
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"collector graph manifest {name} must be a non-negative integer")
    state = manifest["state"]
    if not isinstance(state, str) or state not in {"unavailable", "eager", "ready", "invalidated"}:
        raise ValueError(f"collector graph manifest has unsupported state {state!r}")
    capture_id = manifest["capture_id"]
    reason = manifest["disable_reason"]
    invalidation_reason = manifest["invalidation_reason"]
    if invalidation_reason is not None and (
        not isinstance(invalidation_reason, str)
        or invalidation_reason not in COLLECTOR_GRAPH_INVALIDATION_REASONS
    ):
        raise ValueError(
            f"collector graph manifest has unsupported invalidation_reason {invalidation_reason!r}"
        )
    if manifest["enabled"]:
        if mode != "graph" or not manifest["requested"] or state != "ready":
            raise ValueError("an enabled collector graph must be a requested, READY graph mode")
        if not isinstance(capture_id, str) or not capture_id.strip():
            raise ValueError("an enabled collector graph requires a non-empty capture_id")
        if reason is not None:
            raise ValueError("an enabled collector graph must not declare a disable reason")
        if invalidation_reason is not None:
            raise ValueError("an enabled collector graph must not declare an invalidation reason")
    else:
        if capture_id is not None:
            raise ValueError("a disabled collector graph must not publish a capture_id")
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError("a disabled collector graph requires a non-empty disable reason")
        if state == "invalidated" and invalidation_reason is None:
            raise ValueError("an invalidated collector graph requires an invalidation reason")
        if state != "invalidated" and invalidation_reason is not None:
            raise ValueError(
                "only an invalidated collector graph may declare an invalidation reason"
            )
    if manifest["replay_release_count"] > manifest["replay_publish_count"]:
        raise ValueError("collector graph replay release count exceeds publication count")


__all__ = [
    "COLLECTOR_GRAPH_MANIFEST_VERSION",
    "COLLECTOR_GRAPH_MODES",
    "CollectorGraphMode",
    "resolve_collector_graph_mode",
    "validate_collector_graph_manifest",
]
