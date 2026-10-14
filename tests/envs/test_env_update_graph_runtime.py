"""Controller tests for the owner-owned env-update graph runtime."""

from __future__ import annotations

from typing import Any, cast

import pytest
import torch
from unisim import GraphCompletionToken, GraphLifecycleState

from tests.envs.test_manager_based_rl_env import _make_cfg, _make_env
from unilab.base.env_update_graph import validate_env_update_graph_manifest


class FakeEnvUpdateGraphRuntime:
    """Bounded runtime double; production CUDA capture lands in a later phase."""

    def __init__(self, *, fail_capture: bool = False, fail_replay: bool = False):
        self.stream_id = 17
        self.layout: dict[str, Any] | None = None
        self.capture_id = "env-update-capture"
        self.capture_calls = 0
        self.replay_calls = 0
        self.fail_capture = fail_capture
        self.fail_replay = fail_replay

    def capture(self, verdict, body, *, current_layout):
        del verdict
        if self.fail_capture:
            raise RuntimeError("simulated capture failure")
        self.capture_calls += 1
        body()
        self.layout = current_layout
        return self.capture_id, current_layout

    def replay(self, receipt, body):
        del receipt, body
        if self.fail_replay:
            self.fail_replay = False
            raise RuntimeError("simulated replay failure")
        self.replay_calls += 1
        return GraphCompletionToken(
            contract_version=1,
            lifecycle_state=GraphLifecycleState.READY,
            operation="env_update",
            sequence=self.replay_calls,
            stream_id=self.stream_id,
            capture_id=self.capture_id,
        )


def _make_graph_env():
    cfg = _make_cfg(include_optional_managers=False)
    cfg.events = {}
    env, backend = _make_env(cfg, torch_stream_id=17)
    env.init_state()
    env.set_deferred_validation_boundaries(True)
    return env, backend


def _state_steps(env: Any) -> torch.Tensor:
    assert env._state is not None
    return env._state.info["steps"]


def test_env_update_runtime_rejects_cpu_before_capture() -> None:
    env, _backend = _make_graph_env()
    runtime = FakeEnvUpdateGraphRuntime()
    env._env_update_graph_runtime = cast(Any, runtime)

    with pytest.raises(RuntimeError, match="environment carriers must be CUDA-resident"):
        env.configure_env_update_graph("graph")
    assert runtime.capture_calls == 0
    manifest = env.get_env_update_graph_manifest()
    assert manifest["enabled"] is False
    validate_env_update_graph_manifest(manifest)
    env.close()


def test_env_update_graph_requires_runtime_before_capture() -> None:
    env, _backend = _make_graph_env()
    # Runtime absence is checked after the owner's fail-closed carrier checks.
    with pytest.raises(RuntimeError, match="CUDA-resident"):
        env.configure_env_update_graph("graph")
    env.close()


def test_env_update_runtime_requires_backend_receipt_for_replay() -> None:
    env, _backend = _make_graph_env()
    runtime = FakeEnvUpdateGraphRuntime()
    env._env_update_graph_runtime = cast(Any, runtime)
    with pytest.raises(RuntimeError, match="requires the backend step receipt"):
        env._replay_env_update_graph_body(_state_steps(env))
    env.close()


def _force_graph_ready(env) -> None:
    runtime = FakeEnvUpdateGraphRuntime()
    env._env_update_graph_runtime = runtime
    env._env_update_graph_state = "ready"
    env._env_update_graph_enabled = True
    env._env_update_graph_requested = True
    env._env_update_graph_capture_id = runtime.capture_id
    env._env_update_graph_disable_reason = None
    runtime.layout = env.get_env_update_carrier_layout()
    env._env_update_graph_backend_receipt = GraphCompletionToken(
        contract_version=1,
        lifecycle_state=GraphLifecycleState.READY,
        operation="step",
        sequence=1,
        stream_id=runtime.stream_id,
        capture_id="backend-capture",
    )


def test_env_update_runtime_rejects_stale_or_wrong_stream_receipt() -> None:
    env, _backend = _make_graph_env()
    _force_graph_ready(env)
    stale = GraphCompletionToken(
        contract_version=1,
        lifecycle_state=GraphLifecycleState.READY,
        operation="mask_reset",
        sequence=1,
        stream_id=env._env_update_graph_runtime.stream_id or 0,
        capture_id="backend-capture",
    )
    env._env_update_graph_backend_receipt = stale
    with pytest.raises(RuntimeError, match="receipt is not READY"):
        env._replay_env_update_graph_body(_state_steps(env))

    wrong_stream = GraphCompletionToken(
        contract_version=1,
        lifecycle_state=GraphLifecycleState.READY,
        operation="step",
        sequence=1,
        stream_id=99,
        capture_id="backend-capture",
    )
    env._env_update_graph_backend_receipt = wrong_stream
    with pytest.raises(RuntimeError, match="stream does not match"):
        env._replay_env_update_graph_body(_state_steps(env))
    assert cast(Any, env._env_update_graph_runtime).replay_calls == 0
    env.close()


def test_env_update_runtime_rejects_carrier_layout_drift() -> None:
    env, _backend = _make_graph_env()
    _force_graph_ready(env)
    runtime = env._env_update_graph_runtime
    assert runtime.layout is not None
    runtime.layout = {**runtime.layout, "device": "cuda:99"}

    with pytest.raises(RuntimeError, match="carrier layout changed"):
        env._replay_env_update_graph_body(_state_steps(env))
    manifest = env.get_env_update_graph_manifest()
    assert manifest["enabled"] is False
    assert manifest["state"] == "invalidated"
    assert manifest["invalidation_reason"] == "view"
    validate_env_update_graph_manifest(manifest)
    env.close()


def test_env_update_runtime_replay_failure_preserves_transition_boundary() -> None:
    env, backend = _make_graph_env()
    _force_graph_ready(env)
    runtime = env._env_update_graph_runtime
    cast(Any, runtime).fail_replay = True
    actions = torch.zeros((env.num_envs, 1), dtype=torch.float32, device=env.device)

    with pytest.raises(RuntimeError, match="simulated replay failure"):
        env.step(actions)
    assert env._state is not None
    assert "graph_completion" not in env._state.info
    assert env._env_update_graph_replay_count == 0
    assert env._pending_backend_step_completion is None
    assert len(backend.tensor_controls) == 1
    env.close()
