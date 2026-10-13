from __future__ import annotations

import pytest

from unilab.base.collector_graph import (
    CollectorGraphMode,
    resolve_collector_graph_mode,
    validate_collector_graph_manifest,
)


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("disabled", CollectorGraphMode.DISABLED),
        (" eager ", CollectorGraphMode.EAGER),
        ("GRAPH", CollectorGraphMode.GRAPH),
    ],
)
def test_collector_graph_mode_resolution(value: str, expected: CollectorGraphMode) -> None:
    assert resolve_collector_graph_mode(value) is expected


def test_collector_graph_mode_rejects_boolean_and_unknown_values() -> None:
    with pytest.raises(TypeError, match="boolean"):
        resolve_collector_graph_mode(True)
    with pytest.raises(ValueError, match="disabled, eager, graph"):
        resolve_collector_graph_mode("capture")


def _manifest(**updates: object) -> dict[str, object]:
    value: dict[str, object] = {
        "contract_version": 1,
        "mode": "eager",
        "requested": False,
        "enabled": False,
        "state": "eager",
        "capture_id": None,
        "disable_reason": "explicit eager collector execution",
        "fallback_count": 0,
        "replay_count": 0,
        "replay_publish_count": 0,
        "replay_release_count": 0,
    }
    value.update(updates)
    return value


def test_collector_graph_manifest_accepts_strict_eager_evidence() -> None:
    validate_collector_graph_manifest(_manifest())


@pytest.mark.parametrize(
    "mutation",
    [
        lambda value: value.pop("capture_id"),
        lambda value: value.update(contract_version=2),
        lambda value: value.update(mode="capture"),
        lambda value: value.update(requested=1),
        lambda value: value.update(fallback_count=-1),
        lambda value: value.update(replay_release_count=1),
        lambda value: value.update(enabled=True),
    ],
)
def test_collector_graph_manifest_fails_closed(mutation) -> None:
    value = _manifest()
    mutation(value)
    with pytest.raises((TypeError, ValueError), match="collector graph"):
        validate_collector_graph_manifest(value)


def test_enabled_graph_manifest_requires_ready_capture_identity() -> None:
    validate_collector_graph_manifest(
        _manifest(
            mode="graph",
            requested=True,
            enabled=True,
            state="ready",
            capture_id="capture-a",
            disable_reason=None,
        )
    )
