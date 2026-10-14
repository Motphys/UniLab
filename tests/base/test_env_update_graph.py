from __future__ import annotations

import pytest

from unilab.base.env_update_graph import (
    ENV_UPDATE_GRAPH_MANIFEST_VERSION,
    EnvUpdateGraphMode,
    resolve_env_update_graph_mode,
    validate_env_update_graph_manifest,
)


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("disabled", EnvUpdateGraphMode.DISABLED),
        (" EAGER ", EnvUpdateGraphMode.EAGER),
        ("Graph", EnvUpdateGraphMode.GRAPH),
    ],
)
def test_env_update_graph_mode_resolution(value: str, expected: EnvUpdateGraphMode) -> None:
    assert resolve_env_update_graph_mode(value) is expected


def test_env_update_graph_mode_rejects_boolean_and_unknown_values() -> None:
    with pytest.raises(TypeError, match="boolean"):
        resolve_env_update_graph_mode(False)
    with pytest.raises(ValueError, match="disabled, eager, graph"):
        resolve_env_update_graph_mode("collector")


def _manifest(**updates: object) -> dict[str, object]:
    value: dict[str, object] = {
        "contract_version": ENV_UPDATE_GRAPH_MANIFEST_VERSION,
        "mode": "eager",
        "requested": False,
        "enabled": False,
        "state": "eager",
        "capture_id": None,
        "invalidation_reason": None,
        "disable_reason": "explicit eager env update execution",
        "fallback_count": 0,
        "replay_count": 0,
        "backend_step_receipt": {"operation": None, "sequence": None, "capture_id": None},
        "update_completion": {
            "operation": "env_update",
            "capture_id": None,
            "sequence": 0,
        },
        "rng_state": {"device": "cpu", "state_version": 1, "draw_version": 2},
        "carrier_layout": {"num_envs": 2},
    }
    value.update(updates)
    return value


def test_env_update_graph_manifest_accepts_strict_eager_evidence() -> None:
    validate_env_update_graph_manifest(_manifest())


@pytest.mark.parametrize(
    "mutation",
    [
        lambda value: value.pop("backend_step_receipt"),
        lambda value: value.update(contract_version=2),
        lambda value: value.update(mode="collector"),
        lambda value: value.update(requested=1),
        lambda value: value.update(fallback_count=-1),
        lambda value: value.update(replay_count=1),
        lambda value: value.update(enabled=True),
        lambda value: value.update(update_completion=None),
        lambda value: value.update(rng_state=[]),
        lambda value: value.update(carrier_layout="unstable"),
    ],
)
def test_env_update_graph_manifest_fails_closed(mutation) -> None:
    value = _manifest()
    mutation(value)
    with pytest.raises((TypeError, ValueError), match="env update graph"):
        validate_env_update_graph_manifest(value)


def test_enabled_update_graph_requires_real_replay_evidence() -> None:
    with pytest.raises(ValueError, match="real replay"):
        validate_env_update_graph_manifest(
            _manifest(
                mode="graph",
                requested=True,
                enabled=True,
                state="ready",
                capture_id="update-capture-a",
                disable_reason=None,
            )
        )

    validate_env_update_graph_manifest(
        _manifest(
            mode="graph",
            requested=True,
            enabled=True,
            state="ready",
            capture_id="update-capture-a",
            disable_reason=None,
            replay_count=1,
        )
    )
