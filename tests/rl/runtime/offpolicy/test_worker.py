from __future__ import annotations

import queue
import statistics
import threading
from collections import deque
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch

import unilab.rl.runtime.offpolicy.worker as worker_module
from unilab.rl.logging.metrics_drain import RewardComponentWindow, drain_collector_metrics
from unilab.rl.logging.offpolicy import OffPolicyLogger
from unilab.rl.runtime.collector_timing import extract_env_step_breakdown_timing_ms
from unilab.rl.runtime.ipc.inference_ring import SharedInferenceRing
from unilab.rl.runtime.offpolicy.metrics import CollectorMetrics
from unilab.rl.runtime.offpolicy.worker import (
    _publish_collector_ready,
    _publish_inference_tick,
    _wait_for_inference_tick,
    sample_offpolicy_actions,
)


def test_tensor_collector_metrics_resets_done_episodes_and_flushes_one_transfer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    metrics = CollectorMetrics(num_envs=3, interval=4, device="cpu")
    calls = 0
    original_cpu = torch.Tensor.cpu

    def counting_cpu(tensor: torch.Tensor) -> torch.Tensor:
        nonlocal calls
        calls += 1
        return original_cpu(tensor)

    monkeypatch.setattr(torch.Tensor, "cpu", counting_cpu)

    metrics.update(
        rewards=torch.tensor([1.0, 2.0, 3.0]),
        done=torch.tensor([False, True, False]),
        timeout=torch.tensor([False, False, False]),
    )
    metrics.update(
        rewards=torch.tensor([4.0, 5.0, 6.0]),
        done=torch.tensor([True, False, True]),
        timeout=torch.tensor([False, False, False]),
    )
    metrics.update(
        rewards=torch.tensor([7.0, 8.0, 9.0]),
        done=torch.tensor([False, False, False]),
        timeout=torch.tensor([False, False, False]),
    )
    assert calls == 0
    metrics.update(
        rewards=torch.tensor([10.0, 11.0, 12.0]),
        done=torch.tensor([True, True, True]),
        timeout=torch.tensor([True, False, True]),
    )

    flushed = metrics.flush()

    assert flushed.rewards == [2.0, 5.0, 9.0, 17.0, 24.0, 21.0]
    assert flushed.lengths == [1, 2, 2, 2, 3, 2]
    assert flushed.done_count == 6
    assert flushed.timeout_count == 2
    assert calls == 1
    assert metrics.ready is False
    assert torch.count_nonzero(metrics.current_rewards) == 0
    assert torch.count_nonzero(metrics.current_lengths) == 0


def test_tensor_collector_metrics_packs_reward_components_in_same_transfer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    metrics = CollectorMetrics(
        num_envs=2,
        interval=2,
        device="cpu",
        reward_component_names=("left", "right"),
    )
    calls = 0
    original_cpu = torch.Tensor.cpu

    def counting_cpu(tensor: torch.Tensor) -> torch.Tensor:
        nonlocal calls
        calls += 1
        return original_cpu(tensor)

    monkeypatch.setattr(torch.Tensor, "cpu", counting_cpu)
    metrics.update(
        rewards=torch.tensor([1.0, 2.0]),
        done=torch.tensor([False, True]),
        timeout=torch.tensor([False, False]),
        reward_components=torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
    )
    metrics.update(
        rewards=torch.tensor([3.0, 4.0]),
        done=torch.tensor([True, False]),
        timeout=torch.tensor([True, False]),
        reward_components=torch.tensor([[5.0, 6.0], [7.0, 8.0]]),
    )

    flushed = metrics.flush()

    assert flushed.reward_component_means == {"left": 4.0, "right": 5.0}
    assert calls == 1
    assert torch.count_nonzero(metrics._reward_component_sum) == 0


def test_tensor_collector_metrics_supports_no_done_window() -> None:
    metrics = CollectorMetrics(num_envs=2, interval=2, device="cpu")
    metrics.update(
        rewards=torch.tensor([1.0, 2.0]),
        done=torch.tensor([False, False]),
        timeout=torch.tensor([False, False]),
    )
    metrics.update(
        rewards=torch.tensor([3.0, 4.0]),
        done=torch.tensor([False, False]),
        timeout=torch.tensor([False, False]),
    )

    flushed = metrics.flush()

    assert flushed.rewards == []
    assert flushed.lengths == []
    assert flushed.done_count == 0
    assert flushed.timeout_count == 0
    assert metrics.current_rewards.tolist() == [4.0, 6.0]
    assert metrics.current_lengths.tolist() == [2, 2]


def test_tensor_collector_final_flush_handles_empty_and_partial_windows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    metrics = CollectorMetrics(num_envs=2, interval=3, device="cpu")
    calls = 0
    original_cpu = torch.Tensor.cpu

    def counting_cpu(tensor: torch.Tensor) -> torch.Tensor:
        nonlocal calls
        calls += 1
        return original_cpu(tensor)

    monkeypatch.setattr(torch.Tensor, "cpu", counting_cpu)

    empty_before_updates = metrics.final_flush()
    empty_after_finalize = metrics.final_flush()

    assert empty_before_updates == empty_after_finalize
    assert calls == 0

    metrics = CollectorMetrics(num_envs=2, interval=3, device="cpu")
    metrics.update(
        rewards=torch.tensor([1.0, 10.0]),
        done=torch.tensor([False, False]),
        timeout=torch.tensor([False, False]),
    )
    no_completed = metrics.final_flush()

    assert no_completed.rewards == []
    assert no_completed.lengths == []
    assert no_completed.done_count == 0
    assert no_completed.timeout_count == 0
    assert calls == 1
    assert metrics.current_rewards.tolist() == [1.0, 10.0]
    assert metrics.current_lengths.tolist() == [1, 1]


def test_tensor_collector_final_flush_emits_only_completed_episodes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    metrics = CollectorMetrics(num_envs=2, interval=3, device="cpu")
    calls = 0
    original_cpu = torch.Tensor.cpu

    def counting_cpu(tensor: torch.Tensor) -> torch.Tensor:
        nonlocal calls
        calls += 1
        return original_cpu(tensor)

    monkeypatch.setattr(torch.Tensor, "cpu", counting_cpu)
    metrics.update(
        rewards=torch.tensor([1.0, 10.0]),
        done=torch.tensor([True, False]),
        timeout=torch.tensor([False, False]),
    )
    metrics.update(
        rewards=torch.tensor([2.0, 20.0]),
        done=torch.tensor([False, False]),
        timeout=torch.tensor([False, False]),
    )

    flushed = metrics.final_flush()

    assert flushed.rewards == [1.0]
    assert flushed.lengths == [1]
    assert flushed.done_count == 1
    assert flushed.timeout_count == 0
    assert calls == 1
    assert metrics.current_rewards.tolist() == [2.0, 30.0]
    assert metrics.current_lengths.tolist() == [1, 2]


def test_tensor_collector_final_flush_emits_multiple_completed_episodes() -> None:
    metrics = CollectorMetrics(num_envs=2, interval=4, device="cpu")
    metrics.update(
        rewards=torch.tensor([1.0, 10.0]),
        done=torch.tensor([True, False]),
        timeout=torch.tensor([False, False]),
    )
    metrics.update(
        rewards=torch.tensor([2.0, 20.0]),
        done=torch.tensor([True, True]),
        timeout=torch.tensor([False, True]),
    )

    flushed = metrics.final_flush()

    assert flushed.rewards == [1.0, 2.0, 30.0]
    assert flushed.lengths == [1, 1, 2]
    assert flushed.done_count == 3
    assert flushed.timeout_count == 1


def test_tensor_collector_final_flush_after_normal_flush_is_idempotent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    metrics = CollectorMetrics(num_envs=1, interval=2, device="cpu")
    calls = 0
    original_cpu = torch.Tensor.cpu

    def counting_cpu(tensor: torch.Tensor) -> torch.Tensor:
        nonlocal calls
        calls += 1
        return original_cpu(tensor)

    monkeypatch.setattr(torch.Tensor, "cpu", counting_cpu)
    metrics.update(
        rewards=torch.tensor([1.0]),
        done=torch.tensor([True]),
        timeout=torch.tensor([False]),
    )
    metrics.update(
        rewards=torch.tensor([2.0]),
        done=torch.tensor([True]),
        timeout=torch.tensor([True]),
    )
    normal_flush = metrics.flush()
    final_flush = metrics.final_flush()

    assert normal_flush.rewards == [1.0, 2.0]
    assert normal_flush.lengths == [1, 1]
    assert final_flush.rewards == []
    assert final_flush.lengths == []
    assert final_flush.done_count == 0
    assert final_flush.timeout_count == 0
    assert calls == 1


def test_tensor_collector_metrics_preserves_new_episode_after_done() -> None:
    metrics = CollectorMetrics(num_envs=1, interval=3, device="cpu")
    metrics.update(
        rewards=torch.tensor([1.0]),
        done=torch.tensor([True]),
        timeout=torch.tensor([False]),
    )
    metrics.update(
        rewards=torch.tensor([2.0]),
        done=torch.tensor([False]),
        timeout=torch.tensor([False]),
    )
    metrics.update(
        rewards=torch.tensor([3.0]),
        done=torch.tensor([False]),
        timeout=torch.tensor([False]),
    )

    first_flush = metrics.flush()

    assert first_flush.rewards == [1.0]
    assert first_flush.lengths == [1]
    assert metrics.current_rewards.tolist() == [5.0]
    assert metrics.current_lengths.tolist() == [2]

    metrics.update(
        rewards=torch.tensor([4.0]),
        done=torch.tensor([True]),
        timeout=torch.tensor([True]),
    )
    metrics.update(
        rewards=torch.tensor([0.0]),
        done=torch.tensor([False]),
        timeout=torch.tensor([False]),
    )
    metrics.update(
        rewards=torch.tensor([0.0]),
        done=torch.tensor([False]),
        timeout=torch.tensor([False]),
    )
    second_flush = metrics.flush()

    assert second_flush.rewards == [9.0]
    assert second_flush.lengths == [3]


def test_tensor_collector_metrics_enforces_window_and_input_contract() -> None:
    metrics = CollectorMetrics(num_envs=2, interval=1, device="cpu")

    with pytest.raises(RuntimeError, match="window is not ready"):
        metrics.flush()
    with pytest.raises(ValueError, match="positive environment count"):
        CollectorMetrics(num_envs=0, interval=1, device="cpu")
    with pytest.raises(ValueError, match="positive interval"):
        CollectorMetrics(num_envs=1, interval=True, device="cpu")
    with pytest.raises(TypeError, match="rewards must be Torch tensors"):
        metrics.update(
            rewards=np.asarray([1.0, 2.0]),
            done=torch.tensor([False, False]),
            timeout=torch.tensor([False, False]),
        )
    with pytest.raises(TypeError, match="rewards must have a floating dtype"):
        metrics.update(
            rewards=torch.tensor([1, 2]),
            done=torch.tensor([False, False]),
            timeout=torch.tensor([False, False]),
        )
    with pytest.raises(TypeError, match="done and timeout values must be Torch tensors"):
        metrics.update(
            rewards=torch.tensor([1.0, 2.0]),
            done=np.asarray([False, False]),
            timeout=torch.tensor([False, False]),
        )
    with pytest.raises(ValueError, match="environment shape"):
        metrics.update(
            rewards=torch.tensor([1.0]),
            done=torch.tensor([False, False]),
            timeout=torch.tensor([False, False]),
        )
    with pytest.raises(ValueError, match="timeout values must have environment shape"):
        metrics.update(
            rewards=torch.tensor([1.0, 2.0]),
            done=torch.tensor([False, False]),
            timeout=torch.tensor([False]),
        )
    with pytest.raises(TypeError, match="boolean tensors"):
        metrics.update(
            rewards=torch.tensor([1.0, 2.0]),
            done=torch.tensor([0, 0]),
            timeout=torch.tensor([False, False]),
        )

    metrics.update(
        rewards=torch.tensor([1.0, 2.0]),
        done=torch.tensor([False, False]),
        timeout=torch.tensor([False, False]),
    )
    with pytest.raises(RuntimeError, match="window was not flushed"):
        metrics.update(
            rewards=torch.tensor([3.0, 4.0]),
            done=torch.tensor([False, False]),
            timeout=torch.tensor([False, False]),
        )


def test_collector_publishes_ready_after_initialization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    stop_event = threading.Event()
    inference_request_queue: queue.Queue[int] = queue.Queue()
    metrics_queue: queue.Queue[dict] = queue.Queue()

    class _Env:
        state = None

        def init_state(self) -> None:
            self.state = SimpleNamespace(obs={"obs": torch.zeros((1, 2))}, info={})
            events.append("env_init")

    class _ReplayBuffer:
        trace_recorder = None
        trace_thread_time = False

        def attach_stop_event(self, stop) -> None:
            del stop

        def add_batch(self, *args, **kwargs) -> bool:
            del args, kwargs
            return True

    def publish_ready(coordination_queue, event) -> bool:
        assert not metrics_queue.empty()
        events.append("ready")
        event.set()
        assert _publish_collector_ready(coordination_queue, None)

    monkeypatch.setattr(worker_module, "apply_torch_thread_runtime", lambda *args, **kwargs: None)
    monkeypatch.setattr(worker_module, "apply_worker_seed", lambda *args, **kwargs: None)
    monkeypatch.setattr(worker_module, "_publish_collector_ready", publish_ready)
    monkeypatch.setattr(
        worker_module,
        "_publish_inference_tick",
        lambda *args, **kwargs: events.append("inference_tick"),
    )

    worker_module._run_collector(
        stop_event=stop_event,
        env_factory=lambda num_envs, env_cfg_override=None: _Env(),
        num_envs=1,
        replay_buffer=_ReplayBuffer(),
        inference_slot=None,
        inference_request_queue=inference_request_queue,
        inference_response_queue=queue.Queue(),
        algo_type="sac",
        actor_adapter_modules=None,
        metrics_queue=metrics_queue,
        inference_transport="host",
        sim_backend="mujoco",
        backend_device=None,
        env_cfg_override=None,
        seed=None,
        trace_enabled=False,
        trace_thread_time=False,
    )

    assert events == ["env_init", "ready"]
    assert metrics_queue.get_nowait()["runtime_manifest"]["inference_owner"] == "learner"
    assert inference_request_queue.get_nowait() == worker_module.COLLECTOR_READY_TICK


def test_collector_rejects_enabled_diagnostics_without_injected_guard_factory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _Env:
        num_envs = 1
        play_capabilities = SimpleNamespace(supports_physics_state_playback=False)

        def set_nan_guard(self, guard: object) -> None:
            del guard
            raise AssertionError("no guard should be constructed without a factory")

    monkeypatch.setattr(worker_module, "apply_torch_thread_runtime", lambda *args, **kwargs: None)
    monkeypatch.setattr(worker_module, "apply_worker_seed", lambda *args, **kwargs: None)

    with pytest.raises(ValueError, match="injected nan_guard_factory"):
        worker_module._run_collector(
            stop_event=threading.Event(),
            env_factory=lambda num_envs, env_cfg_override=None: _Env(),
            num_envs=1,
            replay_buffer=SimpleNamespace(),
            inference_slot=None,
            inference_request_queue=queue.Queue(),
            inference_response_queue=queue.Queue(),
            algo_type="sac",
            actor_adapter_modules=None,
            metrics_queue=queue.Queue(),
            inference_transport="host",
            sim_backend="mujoco",
            backend_device=None,
            env_cfg_override=None,
            seed=None,
            trace_enabled=False,
            trace_thread_time=False,
            nan_guard_cfg=SimpleNamespace(enabled=True),
        )


def _run_ready_collector(
    monkeypatch: pytest.MonkeyPatch,
    env: Any,
    metrics_queue: queue.Queue,
) -> None:
    stop_event = threading.Event()

    class _ReplayBuffer:
        trace_recorder = None
        trace_thread_time = False

        def attach_stop_event(self, stop) -> None:
            del stop

        def add_batch(self, *args, **kwargs) -> bool:
            del args, kwargs
            return True

    def publish_ready(coordination_queue, event) -> bool:
        del coordination_queue, event
        # Returning False exits right after the manifest publish, before the
        # collection loop and its teardown path.
        return False

    monkeypatch.setattr(worker_module, "apply_torch_thread_runtime", lambda *args, **kwargs: None)
    monkeypatch.setattr(worker_module, "apply_worker_seed", lambda *args, **kwargs: None)
    monkeypatch.setattr(worker_module, "_publish_collector_ready", publish_ready)

    worker_module._run_collector(
        stop_event=stop_event,
        env_factory=lambda num_envs, env_cfg_override=None: env,
        num_envs=1,
        replay_buffer=_ReplayBuffer(),
        inference_slot=None,
        inference_request_queue=queue.Queue(),
        inference_response_queue=queue.Queue(),
        algo_type="sac",
        actor_adapter_modules=None,
        metrics_queue=metrics_queue,
        inference_transport="host",
        sim_backend="mujoco",
        backend_device=None,
        env_cfg_override=None,
        seed=None,
        trace_enabled=False,
        trace_thread_time=False,
    )


def test_collector_manifest_publishes_backend_runtime_diagnostics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    metrics_queue: queue.Queue[dict] = queue.Queue()

    class _Env:
        state = None

        def init_state(self) -> None:
            self.state = SimpleNamespace(obs={"obs": torch.zeros((1, 2))}, info={})

        def close(self) -> None:
            pass

        def get_backend_runtime_diagnostics(self) -> dict:
            return {
                "available": True,
                "backend_type": "mjwarp",
                "diagnostics": {
                    "cuda_graph": {
                        "requested": True,
                        "enabled": False,
                        "disable_reason": "graph capture failed",
                    }
                },
            }

    _run_ready_collector(monkeypatch, _Env(), metrics_queue)

    manifest = metrics_queue.get_nowait()["runtime_manifest"]
    assert manifest["backend_diagnostics"] == {
        "available": True,
        "backend_type": "mjwarp",
        "diagnostics": {
            "cuda_graph": {
                "requested": True,
                "enabled": False,
                "disable_reason": "graph capture failed",
            }
        },
    }


def test_collector_manifest_records_unavailable_backend_diagnostics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    metrics_queue: queue.Queue[dict] = queue.Queue()

    class _Env:
        state = None

        def init_state(self) -> None:
            self.state = SimpleNamespace(obs={"obs": torch.zeros((1, 2))}, info={})

        def close(self) -> None:
            pass

    _run_ready_collector(monkeypatch, _Env(), metrics_queue)

    manifest = metrics_queue.get_nowait()["runtime_manifest"]
    assert manifest["backend_diagnostics"] == {
        "available": False,
        "unavailable_reason": "env does not expose backend runtime diagnostics",
    }


def test_collector_requires_batched_replay_publication(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stop_event = threading.Event()

    class _Env:
        state = None

        def init_state(self) -> None:
            self.state = SimpleNamespace(obs={"obs": torch.zeros((1, 2))}, info={})

    class _ReplayBuffer:
        trace_recorder = None
        trace_thread_time = False

        def attach_stop_event(self, stop) -> None:
            del stop

    monkeypatch.setattr(worker_module, "apply_torch_thread_runtime", lambda *args, **kwargs: None)
    monkeypatch.setattr(worker_module, "apply_worker_seed", lambda *args, **kwargs: None)

    with pytest.raises(AttributeError, match="add_batch"):
        worker_module._run_collector(
            stop_event=stop_event,
            env_factory=lambda num_envs, env_cfg_override=None: _Env(),
            num_envs=1,
            replay_buffer=_ReplayBuffer(),
            inference_slot=None,
            inference_request_queue=queue.Queue(),
            inference_response_queue=queue.Queue(),
            algo_type="sac",
            actor_adapter_modules=None,
            metrics_queue=queue.Queue(),
            inference_transport="host",
            sim_backend="mujoco",
            backend_device=None,
            env_cfg_override=None,
            seed=None,
            trace_enabled=False,
            trace_thread_time=False,
        )


def test_collector_binds_backend_device_before_env_materialization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[tuple[str, ...]] = []

    class _EnvMaterializedError(RuntimeError):
        pass

    def bind_device(backend: str, device: str) -> str:
        del backend
        events.append(("bind", device))
        return device

    def env_factory(num_envs, env_cfg_override=None):
        del env_cfg_override
        events.append(("make", str(num_envs)))
        raise _EnvMaterializedError

    monkeypatch.setattr(worker_module, "apply_torch_thread_runtime", lambda *args, **kwargs: None)
    monkeypatch.setattr(worker_module, "apply_worker_seed", lambda *args, **kwargs: None)

    with pytest.raises(_EnvMaterializedError):
        worker_module._run_collector(
            stop_event=None,
            env_factory=env_factory,
            num_envs=2,
            replay_buffer=None,
            inference_slot=None,
            inference_request_queue=None,
            inference_response_queue=None,
            algo_type="sac",
            actor_adapter_modules=None,
            metrics_queue=None,
            inference_transport="host",
            sim_backend="mjwarp",
            backend_device="cuda:3",
            env_cfg_override=None,
            seed=None,
            trace_enabled=False,
            trace_thread_time=False,
            backend_device_configurer=bind_device,
        )

    # The real configure_backend_process_device resolves mjwarp -> cuda:3 and
    # must bind it before the env factory runs.
    assert events == [
        ("bind", "cuda:3"),
        ("make", "2"),
    ]


def test_tensor_collector_reports_bounded_inference_scheduling_diagnostics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stop_event = threading.Event()
    inference_request_queue: queue.Queue[int] = queue.Queue()
    inference_response_queue: queue.Queue[int] = queue.Queue()
    metrics_queue: queue.Queue[dict] = queue.Queue(maxsize=4)
    events: list[str] = []
    replay_calls: list[tuple[tuple[torch.Tensor, ...], dict[str, object]]] = []

    initial_obs = {"obs": torch.zeros((2, 2), dtype=torch.float32)}
    next_obs = {"obs": torch.ones((2, 2), dtype=torch.float32)}

    class _State:
        obs = initial_obs
        reward = torch.ones(2)
        terminated = torch.zeros(2, dtype=torch.bool)
        truncated = torch.zeros(2, dtype=torch.bool)
        final_observation = None
        info = {"timing": {}}

    class _Env:
        state = _State()

        def step(self, actions):
            torch.testing.assert_close(actions, torch.zeros((2, 2)))
            events.append("step")
            stop_event.set()
            self.state = SimpleNamespace(
                obs=next_obs,
                reward=torch.ones(2),
                terminated=torch.zeros(2, dtype=torch.bool),
                truncated=torch.zeros(2, dtype=torch.bool),
                final_observation=None,
                info={"timing": {}},
            )
            return self.state

        def close(self):
            events.append("close")

    class _ReplayBuffer:
        trace_recorder = None
        trace_thread_time = False
        size = torch.zeros(1, dtype=torch.int64)

        def attach_stop_event(self, stop) -> None:
            del stop

        def add(self, *args, **kwargs) -> None:
            del args, kwargs
            events.append("replay_add")

        def add_batch(self, *args, **kwargs) -> bool:
            replay_calls.append((args, kwargs))
            del args, kwargs
            events.append("replay_add_batch")
            return True

        def release_ipc(self) -> None:
            events.append("replay_release")

    monkeypatch.setattr(worker_module, "apply_torch_thread_runtime", lambda *a, **kw: None)
    monkeypatch.setattr(worker_module, "apply_worker_seed", lambda *a, **kw: None)
    ring = SharedInferenceRing(2, 2, 2, capacity=4)

    def serve_learner_action(
        coordination_queue,
        tick_id,
        stop,
        **kwargs,
    ):
        del coordination_queue, stop, kwargs
        ring.copy_observation_to(
            tick_id=tick_id,
            observations=torch.empty((2, 2)),
            dones=torch.empty(2),
            epoch=0,
        )
        ring.publish_action(
            tick_id=tick_id,
            policy_version=0,
            actions=torch.zeros((2, 2)),
            epoch=0,
        )
        return True

    monkeypatch.setattr(worker_module, "_wait_for_inference_tick", serve_learner_action)

    worker_module._run_collector(
        stop_event=stop_event,
        env_factory=lambda num_envs, env_cfg_override=None: _Env(),
        num_envs=2,
        replay_buffer=_ReplayBuffer(),
        inference_slot=ring,
        inference_request_queue=inference_request_queue,
        inference_response_queue=inference_response_queue,
        algo_type="sac",
        actor_adapter_modules=None,
        metrics_queue=metrics_queue,
        inference_transport="host",
        sim_backend="mujoco",
        backend_device=None,
        env_cfg_override=None,
        inference_epoch=0,
        collector_metrics_interval=1,
        seed=1,
        trace_enabled=False,
        trace_thread_time=False,
    )

    manifest_message = metrics_queue.get_nowait()
    manifest = manifest_message["runtime_manifest"]
    assert manifest["inference_queue_capacity"] == 4
    assert manifest["inference_scheduling_policy"] == "sequential_transition_dependency"
    assert manifest["inference_legal_max_in_flight"] == 1
    assert manifest["inference_dependency_graph"]["transition_to_next_observation"] == (
        "env.step(action[t]) -> observation[t+1]"
    )

    report_message = metrics_queue.get_nowait()
    report = report_message["collector_inference"]
    assert report["queue_depth"] == 0
    assert report["action_backlog"] == 0
    assert report["max_action_backlog"] == 0
    assert report["in_flight"] == 0
    assert report["max_in_flight"] == 1
    assert report["wait_time_ms"] >= 0.0
    assert report["publication_lag"] == 0
    assert report["max_publication_lag"] == 1
    manifest_flight = report_message["runtime_manifest"]["inference_flight"]
    assert manifest_flight == report
    assert events == ["step", "replay_add_batch", "close", "replay_release"]
    assert len(replay_calls) == 1
    replay_args, replay_kwargs = replay_calls[0]
    assert len(replay_args) == 6
    assert tuple(tensor.shape[0] for tensor in replay_args) == (2, 2, 2, 2, 2, 2)
    assert replay_args[1].dtype == torch.float32
    for key in (
        "terminal_mask",
        "terminal_next_obs",
        "critic",
        "next_critic",
        "terminal_next_critic",
    ):
        assert key in replay_kwargs
    terminal_mask = replay_kwargs["terminal_mask"]
    critic = replay_kwargs["critic"]
    next_critic = replay_kwargs["next_critic"]
    assert isinstance(terminal_mask, torch.Tensor)
    assert isinstance(critic, torch.Tensor)
    assert isinstance(next_critic, torch.Tensor)
    assert terminal_mask.shape == (2,)
    assert critic.shape == (2, 2)
    assert next_critic.shape == (2, 2)
    assert inference_request_queue.get_nowait() == worker_module.COLLECTOR_READY_TICK
    assert inference_request_queue.get_nowait() == 0


def test_tensor_collector_passes_env_observations_to_replay_without_snapshot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Replay receives the env-provided observation tensors directly.

    The env contract guarantees caller-owned public observations (no in-place
    reuse across steps), so the collector hands them to replay without a
    defensive staging copy; the stored rows are the identical tensor objects.
    """
    stop_event = threading.Event()
    metrics_queue: queue.Queue[dict] = queue.Queue(maxsize=4)
    replay_calls: list[tuple[tuple[torch.Tensor, ...], dict[str, object]]] = []
    initial_obs = torch.zeros((2, 2), dtype=torch.float32)
    initial_critic = torch.full((2, 3), 5.0, dtype=torch.float32)
    next_obs = torch.full((2, 2), 7.0, dtype=torch.float32)
    next_critic = torch.full((2, 3), 11.0, dtype=torch.float32)

    class _Env:
        state = SimpleNamespace(
            obs={"obs": initial_obs, "critic": initial_critic},
            reward=torch.ones(2),
            terminated=torch.tensor([False, True]),
            truncated=torch.zeros(2, dtype=torch.bool),
            final_observation=None,
            info={"timing": {}},
        )

        def step(self, actions):
            torch.testing.assert_close(actions, torch.zeros((2, 2)))
            stop_event.set()
            self.state = SimpleNamespace(
                obs={"obs": next_obs, "critic": next_critic},
                reward=torch.ones(2),
                terminated=self.state.terminated,
                truncated=self.state.truncated,
                final_observation=None,
                info={"timing": {}},
            )
            return self.state

        def close(self):
            return None

    class _ReplayBuffer:
        trace_recorder = None
        trace_thread_time = False
        size = torch.zeros(1, dtype=torch.int64)

        def attach_stop_event(self, stop) -> None:
            del stop

        def add(self, *args, **kwargs) -> None:
            del args, kwargs

        def add_batch(self, *args, **kwargs) -> bool:
            replay_calls.append((args, kwargs))
            return True

        def release_ipc(self) -> None:
            return None

    monkeypatch.setattr(worker_module, "apply_torch_thread_runtime", lambda *a, **kw: None)
    monkeypatch.setattr(worker_module, "apply_worker_seed", lambda *a, **kw: None)
    ring = SharedInferenceRing(2, 2, 2, capacity=2)

    def serve_learner_action(
        coordination_queue,
        tick_id,
        stop,
        **kwargs,
    ):
        del coordination_queue, stop, kwargs
        ring.copy_observation_to(
            tick_id=tick_id,
            observations=torch.empty((2, 2)),
            dones=torch.empty(2),
            epoch=0,
        )
        ring.publish_action(
            tick_id=tick_id,
            policy_version=0,
            actions=torch.zeros((2, 2)),
            epoch=0,
        )
        return True

    monkeypatch.setattr(worker_module, "_wait_for_inference_tick", serve_learner_action)

    worker_module._run_collector(
        stop_event=stop_event,
        env_factory=lambda num_envs, env_cfg_override=None: _Env(),
        num_envs=2,
        replay_buffer=_ReplayBuffer(),
        inference_slot=ring,
        inference_request_queue=queue.Queue(),
        inference_response_queue=queue.Queue(),
        algo_type="sac",
        actor_adapter_modules=None,
        metrics_queue=metrics_queue,
        inference_transport="host",
        sim_backend="mujoco",
        backend_device=None,
        env_cfg_override=None,
        inference_epoch=0,
        collector_metrics_interval=1,
        seed=1,
        trace_enabled=False,
        trace_thread_time=False,
    )

    assert len(replay_calls) == 1
    replay_args, replay_kwargs = replay_calls[0]
    assert replay_args[0] is initial_obs
    assert replay_kwargs["critic"] is initial_critic
    assert replay_args[3] is next_obs
    assert replay_kwargs["next_critic"] is next_critic
    assert replay_args[0].data_ptr() == initial_obs.data_ptr()


def test_tensor_collector_steps_env_with_ring_tensor_actions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """env.step receives the ring's cloned Torch action tensor without restaging."""
    stop_event = threading.Event()
    metrics_queue: queue.Queue[dict] = queue.Queue(maxsize=4)
    replay_calls: list[tuple[tuple[torch.Tensor, ...], dict[str, object]]] = []
    step_actions: list[torch.Tensor] = []
    consumed_actions: list[torch.Tensor] = []

    class _Env:
        state = SimpleNamespace(
            obs={"obs": torch.zeros((2, 2), dtype=torch.float32)},
            reward=torch.ones(2),
            terminated=torch.zeros(2, dtype=torch.bool),
            truncated=torch.zeros(2, dtype=torch.bool),
            final_observation=None,
            info={"timing": {}},
        )

        def step(self, actions):
            step_actions.append(actions)
            assert isinstance(actions, torch.Tensor)
            assert actions.dtype == torch.float32
            torch.testing.assert_close(actions, torch.zeros((2, 2)))
            stop_event.set()
            self.state = SimpleNamespace(
                obs={"obs": torch.ones((2, 2), dtype=torch.float32)},
                reward=torch.ones(2),
                terminated=torch.zeros(2, dtype=torch.bool),
                truncated=torch.zeros(2, dtype=torch.bool),
                final_observation=None,
                info={"timing": {}},
            )
            return self.state

        def close(self):
            return None

    class _ReplayBuffer:
        trace_recorder = None
        trace_thread_time = False
        size = torch.zeros(1, dtype=torch.int64)

        def attach_stop_event(self, stop) -> None:
            del stop

        def add_batch(self, *args, **kwargs) -> bool:
            replay_calls.append((args, kwargs))
            return True

        def release_ipc(self) -> None:
            return None

    monkeypatch.setattr(worker_module, "apply_torch_thread_runtime", lambda *a, **kw: None)
    monkeypatch.setattr(worker_module, "apply_worker_seed", lambda *a, **kw: None)
    ring = SharedInferenceRing(2, 2, 2, capacity=2)

    def serve_learner_action(coordination_queue, tick_id, stop, **kwargs):
        del coordination_queue, stop, kwargs
        ring.copy_observation_to(
            tick_id=tick_id,
            observations=torch.empty((2, 2), dtype=torch.float32),
            dones=torch.empty(2, dtype=torch.float32),
            epoch=0,
        )
        ring.publish_action(
            tick_id=tick_id,
            policy_version=0,
            actions=torch.zeros((2, 2), dtype=torch.float32),
            epoch=0,
        )
        return True

    monkeypatch.setattr(worker_module, "_wait_for_inference_tick", serve_learner_action)
    original_consume_action = ring.consume_action

    def capture_consumed_action(**kwargs):
        actions, policy_version = original_consume_action(**kwargs)
        consumed_actions.append(actions)
        return actions, policy_version

    ring.consume_action = capture_consumed_action

    worker_module._run_collector(
        stop_event=stop_event,
        env_factory=lambda num_envs, env_cfg_override=None: _Env(),
        num_envs=2,
        replay_buffer=_ReplayBuffer(),
        inference_slot=ring,
        inference_request_queue=queue.Queue(),
        inference_response_queue=queue.Queue(),
        algo_type="sac",
        actor_adapter_modules=None,
        metrics_queue=metrics_queue,
        inference_transport="host",
        sim_backend="mujoco",
        backend_device=None,
        env_cfg_override=None,
        inference_epoch=0,
        collector_metrics_interval=1,
        seed=1,
        trace_enabled=False,
        trace_thread_time=False,
    )

    assert len(step_actions) == 1
    assert step_actions[0] is consumed_actions[0]
    assert len(replay_calls) == 1
    replay_args, _ = replay_calls[0]
    assert isinstance(replay_args[1], torch.Tensor)
    assert replay_args[1].dtype == torch.float32


def test_collector_fails_closed_on_numpy_actor_observation() -> None:
    """NumPy actor observations are rejected at init with no conversion."""
    stop_event = threading.Event()

    class _Env:
        state = SimpleNamespace(
            obs={"obs": np.zeros((2, 2), dtype=np.float32)},
            info={},
        )

    with pytest.raises(TypeError, match="requires Torch tensor observations"):
        worker_module._run_collector(
            stop_event=stop_event,
            env_factory=lambda num_envs, env_cfg_override=None: _Env(),
            num_envs=2,
            replay_buffer=SimpleNamespace(
                trace_recorder=None,
                trace_thread_time=False,
                attach_stop_event=lambda stop: None,
            ),
            inference_slot=None,
            inference_request_queue=queue.Queue(),
            inference_response_queue=queue.Queue(),
            algo_type="sac",
            actor_adapter_modules=None,
            metrics_queue=queue.Queue(),
            inference_transport="host",
            sim_backend="mujoco",
            backend_device=None,
            env_cfg_override=None,
            seed=None,
            trace_enabled=False,
            trace_thread_time=False,
        )


def test_collector_fails_closed_on_numpy_critic_observation() -> None:
    """A NumPy critic group is rejected even when the actor group is a tensor."""
    stop_event = threading.Event()

    class _Env:
        state = SimpleNamespace(
            obs={
                "obs": torch.zeros((2, 2), dtype=torch.float32),
                "critic": np.zeros((2, 3), dtype=np.float32),
            },
            info={},
        )

    with pytest.raises(TypeError, match="requires Torch tensor observations"):
        worker_module._run_collector(
            stop_event=stop_event,
            env_factory=lambda num_envs, env_cfg_override=None: _Env(),
            num_envs=2,
            replay_buffer=SimpleNamespace(
                trace_recorder=None,
                trace_thread_time=False,
                attach_stop_event=lambda stop: None,
            ),
            inference_slot=None,
            inference_request_queue=queue.Queue(),
            inference_response_queue=queue.Queue(),
            algo_type="sac",
            actor_adapter_modules=None,
            metrics_queue=queue.Queue(),
            inference_transport="host",
            sim_backend="mujoco",
            backend_device=None,
            env_cfg_override=None,
            seed=None,
            trace_enabled=False,
            trace_thread_time=False,
        )


def test_tensor_collector_preserves_interval_reward_component_means(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stop_event = threading.Event()
    metrics_queue: queue.Queue[dict] = queue.Queue(maxsize=4)

    class _Capabilities:
        reward_component_names = ("left", "right")

    class _Env:
        state = None
        algo_capabilities = _Capabilities()

        def init_state(self) -> None:
            self.state = SimpleNamespace(obs={"obs": torch.zeros((2, 2))}, info={})

        def step(self, actions):
            del actions
            stop_event.set()
            self.state = SimpleNamespace(
                obs={"obs": torch.ones((2, 2), dtype=torch.float32)},
                reward=torch.ones(2),
                terminated=torch.zeros(2, dtype=torch.bool),
                truncated=torch.zeros(2, dtype=torch.bool),
                final_observation=None,
                info={
                    "timing": {},
                    "reward_components": torch.tensor(
                        [[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32
                    ),
                },
            )
            return self.state

        def close(self) -> None:
            pass

    class _ReplayBuffer:
        trace_recorder = None
        trace_thread_time = False
        size = torch.zeros(1, dtype=torch.int64)

        def attach_stop_event(self, stop) -> None:
            del stop

        def add_batch(self, *args, **kwargs) -> bool:
            del args, kwargs
            return True

        def release_ipc(self) -> None:
            pass

    ring = SharedInferenceRing(2, 2, 2, capacity=1)

    def serve_learner_action(coordination_queue, tick_id, stop, **kwargs):
        del coordination_queue, stop, kwargs
        ring.copy_observation_to(
            tick_id=tick_id,
            observations=torch.empty((2, 2)),
            dones=torch.empty(2),
            epoch=0,
        )
        ring.publish_action(
            tick_id=tick_id,
            policy_version=0,
            actions=torch.zeros((2, 2)),
            epoch=0,
        )
        return True

    monkeypatch.setattr(worker_module, "apply_torch_thread_runtime", lambda *a, **kw: None)
    monkeypatch.setattr(worker_module, "apply_worker_seed", lambda *a, **kw: None)
    monkeypatch.setattr(worker_module, "_wait_for_inference_tick", serve_learner_action)

    worker_module._run_collector(
        stop_event=stop_event,
        env_factory=lambda num_envs, env_cfg_override=None: _Env(),
        num_envs=2,
        replay_buffer=_ReplayBuffer(),
        inference_slot=ring,
        inference_request_queue=queue.Queue(),
        inference_response_queue=queue.Queue(),
        algo_type="sac",
        actor_adapter_modules=None,
        metrics_queue=metrics_queue,
        inference_transport="host",
        sim_backend="mujoco",
        backend_device=None,
        env_cfg_override=None,
        inference_epoch=0,
        collector_metrics_interval=1,
        seed=1,
        trace_enabled=False,
        trace_thread_time=False,
    )

    metrics_queue.get_nowait()  # runtime manifest
    report = metrics_queue.get_nowait()
    assert report["reward_components"] == {"left": 2.0, "right": 3.0}
    assert report["reward_component_means"] is True


def test_worker_shutdown_flushes_partial_metrics_before_abnormal_cleanup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stop_event = threading.Event()
    inference_request_queue: queue.Queue[int] = queue.Queue()
    inference_response_queue: queue.Queue[dict] = queue.Queue()
    metrics_queue: queue.Queue[dict] = queue.Queue(maxsize=4)
    events: list[str] = []

    class _State:
        obs = {"obs": torch.zeros((2, 2), dtype=torch.float32)}
        reward = torch.tensor([4.0, 2.0])
        terminated = torch.tensor([True, False])
        truncated = torch.tensor([False, False])
        final_observation = None
        info = {"timing": {}}

    class _Env:
        state = _State()

        def step(self, actions):
            torch.testing.assert_close(actions, torch.zeros((2, 2)))
            events.append("step")
            stop_event.set()
            self.state = SimpleNamespace(
                obs={"obs": torch.ones((2, 2), dtype=torch.float32)},
                reward=torch.tensor([4.0, 2.0]),
                terminated=torch.tensor([True, False]),
                truncated=torch.tensor([False, False]),
                final_observation=None,
                info={"timing": {}},
            )
            return self.state

        def cleanup(self):
            events.append("cleanup")
            raise RuntimeError("abnormal shutdown after partial window")

    class _ReplayBuffer:
        trace_recorder = None
        trace_thread_time = False
        size = torch.tensor([2], dtype=torch.int64)

        def attach_stop_event(self, stop) -> None:
            del stop

        def add(self, *args, **kwargs) -> None:
            del args, kwargs
            events.append("replay_add")

        def add_batch(self, *args, **kwargs) -> bool:
            del args, kwargs
            events.append("replay_add")
            return True

        def release_ipc(self) -> None:
            events.append("replay_release")

    monkeypatch.setattr(worker_module, "apply_torch_thread_runtime", lambda *a, **kw: None)
    monkeypatch.setattr(worker_module, "apply_worker_seed", lambda *a, **kw: None)
    ring = SharedInferenceRing(2, 2, 2, capacity=4)

    def serve_learner_action(
        coordination_queue,
        tick_id,
        stop,
        **kwargs,
    ):
        del coordination_queue, stop, kwargs
        ring.copy_observation_to(
            tick_id=tick_id,
            observations=torch.empty((2, 2)),
            dones=torch.empty(2),
            epoch=0,
        )
        ring.publish_action(
            tick_id=tick_id,
            policy_version=0,
            actions=torch.zeros((2, 2)),
            epoch=0,
        )
        return True

    monkeypatch.setattr(worker_module, "_wait_for_inference_tick", serve_learner_action)

    with pytest.raises(RuntimeError, match="abnormal shutdown after partial window"):
        worker_module._run_collector(
            stop_event=stop_event,
            env_factory=lambda num_envs, env_cfg_override=None: _Env(),
            num_envs=2,
            replay_buffer=_ReplayBuffer(),
            inference_slot=ring,
            inference_request_queue=inference_request_queue,
            inference_response_queue=inference_response_queue,
            algo_type="sac",
            actor_adapter_modules=None,
            metrics_queue=metrics_queue,
            inference_transport="host",
            sim_backend="mujoco",
            backend_device=None,
            env_cfg_override=None,
            inference_epoch=0,
            collector_metrics_interval=3,
            seed=1,
            trace_enabled=False,
            trace_thread_time=False,
        )

    manifest = metrics_queue.get_nowait()["runtime_manifest"]
    assert "tensor_native_env" not in manifest
    final_message = metrics_queue.get_nowait()
    assert final_message["metric_flush"] == "final"
    assert final_message["total_steps"] == 2
    assert final_message["return_mean_ep100"] == 4.0
    assert final_message["mean_episode_length"] == 1.0
    assert final_message["timeout_rate"] == 0.0
    assert metrics_queue.empty()

    reward_history: deque[float] = deque(maxlen=100)
    logger = OffPolicyLogger(log_backend="none")
    # Requeue the captured messages so the shared drain under test sees the
    # exact worker publication sequence.
    metrics_queue.put_nowait({"runtime_manifest": {}})
    metrics_queue.put_nowait(final_message)
    drain_collector_metrics(
        metrics_queue,
        reward_history,
        RewardComponentWindow(),
        logger,
        runner_label="test",
        raise_on_collector_error=True,
        require_buffer_size=True,
    )

    scalars = logger._build_backend_scalars(
        iteration=1,
        metrics=None,
        return_mean_ep100=statistics.mean(reward_history),
        reward_components={},
    )
    assert list(reward_history) == [4.0]
    assert logger._total_steps == 2
    assert scalars["Train/mean_reward"] == 4.0
    assert scalars["Train/mean_episode_length"] == 1.0
    assert scalars["Episode/timeout_rate"] == 0.0
    assert events == ["step", "replay_add", "cleanup", "replay_release"]


def test_inference_request_publish_timeout_is_explicit() -> None:
    requests: queue.Queue[int] = queue.Queue(maxsize=1)
    requests.put_nowait(0)

    with pytest.raises(TimeoutError, match="publishing off-policy inference tick 1"):
        _publish_inference_tick(requests, 1, threading.Event(), timeout=0.01)


def test_inference_response_wait_timeout_is_explicit() -> None:
    with pytest.raises(TimeoutError, match="waiting for off-policy inference tick 0"):
        _wait_for_inference_tick(queue.Queue(), 0, threading.Event(), timeout=0.01)


def test_inference_response_wait_allows_healthy_busy_learner_beyond_deadline() -> None:
    from unilab.rl.runtime.offpolicy.coordination import LearnerCoordinationState

    state = LearnerCoordinationState()
    state.mark_busy()
    response_queue: queue.Queue[int] = queue.Queue()
    threading.Timer(0.04, response_queue.put, args=(0,)).start()

    assert _wait_for_inference_tick(
        response_queue,
        0,
        threading.Event(),
        learner_coordination=state,
        learner_pid=None,
        timeout=0.01,
    )


def test_inference_response_wait_detects_stopped_or_dead_learner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from unilab.rl.runtime.offpolicy.coordination import LearnerCoordinationState

    stopped = LearnerCoordinationState()
    stopped.mark_stopped()
    with pytest.raises(RuntimeError, match="Learner stopped before inference tick 0"):
        _wait_for_inference_tick(
            queue.Queue(),
            0,
            threading.Event(),
            learner_coordination=stopped,
            learner_pid=None,
            timeout=0.01,
        )

    alive_phase = LearnerCoordinationState()
    alive_phase.mark_busy()
    monkeypatch.setattr(worker_module, "learner_pid_is_alive", lambda pid: False)
    with pytest.raises(RuntimeError, match="Learner process died before inference tick 0"):
        _wait_for_inference_tick(
            queue.Queue(),
            0,
            threading.Event(),
            learner_coordination=alive_phase,
            learner_pid=12345,
            timeout=0.01,
        )


def test_inference_response_wait_uses_waiting_progress_not_latency() -> None:
    from unilab.rl.runtime.offpolicy.coordination import LearnerCoordinationState

    state = LearnerCoordinationState()
    state.mark_waiting()
    response_queue: queue.Queue[int] = queue.Queue()

    def progress_then_publish() -> None:
        state.mark_progress()
        response_queue.put(0)

    threading.Timer(0.02, progress_then_publish).start()

    assert _wait_for_inference_tick(
        response_queue,
        0,
        threading.Event(),
        learner_coordination=state,
        learner_pid=None,
        timeout=0.01,
    )


def test_learner_stop_releases_inference_response_wait() -> None:
    from unilab.rl.runtime.offpolicy.coordination import LearnerCoordinationState

    state = LearnerCoordinationState()
    state.mark_busy()
    stop_event = threading.Event()
    threading.Timer(0.01, stop_event.set).start()

    assert not _wait_for_inference_tick(
        queue.Queue(),
        0,
        stop_event,
        learner_coordination=state,
        learner_pid=None,
        timeout=10.0,
    )


def test_stopped_phase_with_stop_request_releases_wait_gracefully() -> None:
    """Normal training completion sets the stop event before STOPPED.

    A collector already inside the wait loop must observe the completed run as
    a graceful release, not as a "Learner stopped" crash.
    """
    from unilab.rl.runtime.offpolicy.coordination import LearnerCoordinationState

    state = LearnerCoordinationState()
    state.mark_busy()
    stop_event = threading.Event()

    def shutdown() -> None:
        # Mirrors DoubleBufferRunner._shutdown_collector ordering.
        stop_event.set()
        state.mark_stopped()

    threading.Timer(0.01, shutdown).start()

    assert not _wait_for_inference_tick(
        queue.Queue(),
        0,
        stop_event,
        learner_coordination=state,
        learner_pid=None,
        timeout=10.0,
    )


class _DummyActor:
    def __init__(self) -> None:
        self.calls: list[tuple[torch.Tensor, torch.Tensor, bool]] = []

    def explore(
        self,
        obs: torch.Tensor,
        dones: torch.Tensor | None = None,
        deterministic: bool = False,
    ) -> torch.Tensor:
        assert dones is not None
        self.calls.append((obs.clone(), dones.clone(), deterministic))
        return torch.ones(obs.shape[0], 3, dtype=obs.dtype)


def test_extract_env_step_breakdown_timing_ms_maps_env_owned_keys_only() -> None:
    timing = extract_env_step_breakdown_timing_ms(
        {
            "timing": {
                "step_core_ms": 1.5,
                "update_state_ms": 2.5,
                "reset_done_ms": 0.25,
                "action_validate_ms": 0.1,
                "apply_action_ms": 9.0,
            }
        }
    )

    assert timing == {
        "env_step_backend_ms": 1.5,
        "env_step_update_state_ms": 2.5,
        "env_step_reset_done_ms": 0.25,
        "env_step_action_validate_ms": 0.1,
        "env_step_apply_action_ms": 9.0,
    }


@pytest.mark.parametrize("algo_type", ["sac", "flashsac"])
def test_sample_offpolicy_actions_uses_actor_explore(algo_type: str) -> None:
    actor = _DummyActor()
    obs = torch.zeros(4, 5)
    dones = torch.zeros(4)

    actions = sample_offpolicy_actions(
        actor=actor,
        algo_type=algo_type,
        obs=obs,
        prev_dones=dones,
    )

    assert len(actor.calls) == 1
    assert actor.calls[0][2] is False
    assert actions.shape == (4, 3)


def test_sample_offpolicy_actions_rejects_unknown_algo() -> None:
    actor = _DummyActor()

    with pytest.raises(ValueError, match="learner action sampling"):
        sample_offpolicy_actions(
            actor=actor,
            algo_type="unknown",
            obs=torch.zeros(2, 4),
            prev_dones=torch.zeros(2),
        )


class _DummyPrivInfoActor:
    def __init__(self) -> None:
        self.calls: list[tuple[torch.Tensor, torch.Tensor, bool]] = []

    def explore(
        self,
        obs: torch.Tensor,
        priv_info: torch.Tensor,
        deterministic: bool = False,
    ) -> torch.Tensor:
        self.calls.append((obs.clone(), priv_info.clone(), deterministic))
        return torch.ones(obs.shape[0], 3, dtype=obs.dtype)


def _register_dummy_priv_adapter(monkeypatch: pytest.MonkeyPatch) -> None:
    import numpy as np

    from unilab.rl.runtime.offpolicy import actor_adapter as actor_adapter_module

    def resolve_priv_info(obs_np, critic_np, info):
        if info is not None and info.get("critic_info") is not None:
            return np.asarray(info["critic_info"], dtype=np.float32)
        return np.asarray(critic_np[:, obs_np.shape[1] :], dtype=np.float32)

    monkeypatch.setitem(
        actor_adapter_module._ADAPTERS,
        "dummy_priv_sac",
        actor_adapter_module.OffPolicyActorAdapter(
            algo_type="dummy_priv_sac",
            sample_actions=lambda actor, obs, dones, priv: actor.explore(
                obs, priv, deterministic=False
            ),
            resolve_priv_info=resolve_priv_info,
        ),
    )


def test_sample_offpolicy_actions_uses_adapter_sample_actions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _register_dummy_priv_adapter(monkeypatch)
    actor = _DummyPrivInfoActor()
    obs = torch.zeros(4, 5)
    priv_info = torch.randn(4, 2)

    actions = sample_offpolicy_actions(
        actor=actor,
        algo_type="dummy_priv_sac",
        obs=obs,
        prev_dones=torch.zeros(4),
        priv_info=priv_info,
    )

    assert actions.shape == (4, 3)
    assert len(actor.calls) == 1
    torch.testing.assert_close(actor.calls[0][1], priv_info)


def test_collector_rejects_transport_metadata_in_env_override() -> None:
    stop_event = threading.Event()
    ring = SharedInferenceRing(1, 2, 2, device="cpu")

    with pytest.raises(ValueError, match="Collector env override must not set"):
        worker_module._run_collector(
            stop_event=stop_event,
            env_factory=lambda num_envs, env_cfg_override=None: SimpleNamespace(),
            num_envs=1,
            replay_buffer=SimpleNamespace(),
            inference_slot=ring,
            inference_request_queue=queue.Queue(),
            inference_response_queue=queue.Queue(),
            algo_type="sac",
            actor_adapter_modules=None,
            metrics_queue=queue.Queue(),
            inference_transport="host",
            sim_backend="mujoco",
            backend_device=None,
            env_cfg_override={"inference_transport": "gpu"},
            seed=None,
            trace_enabled=False,
            trace_thread_time=False,
        )


def test_collector_rejects_removed_legacy_tensor_transport() -> None:
    stop_event = threading.Event()
    ring = SharedInferenceRing(1, 2, 2, device="cpu")

    with pytest.raises(ValueError, match="must be one of \\{'host', 'cuda'\\}"):
        worker_module._run_collector(
            stop_event=stop_event,
            env_factory=lambda num_envs, env_cfg_override=None: SimpleNamespace(),
            num_envs=1,
            replay_buffer=SimpleNamespace(),
            inference_slot=ring,
            inference_request_queue=queue.Queue(),
            inference_response_queue=queue.Queue(),
            algo_type="sac",
            actor_adapter_modules=None,
            metrics_queue=queue.Queue(),
            inference_transport="legacy_tensor",
            sim_backend="mujoco",
            backend_device=None,
            env_cfg_override={},
            seed=None,
            trace_enabled=False,
            trace_thread_time=False,
        )


def test_collector_rejects_cuda_transport_with_cpu_ring() -> None:
    stop_event = threading.Event()
    ring = SharedInferenceRing(1, 2, 2, device="cpu")

    with pytest.raises(ValueError, match="CUDA collector inference requires a CUDA inference ring"):
        worker_module._run_collector(
            stop_event=stop_event,
            env_factory=lambda num_envs, env_cfg_override=None: SimpleNamespace(),
            num_envs=1,
            replay_buffer=SimpleNamespace(),
            inference_slot=ring,
            inference_request_queue=queue.Queue(),
            inference_response_queue=queue.Queue(),
            algo_type="sac",
            actor_adapter_modules=None,
            metrics_queue=queue.Queue(),
            sim_backend="mujoco",
            backend_device=None,
            env_cfg_override={},
            inference_transport="cuda",
            seed=None,
            trace_enabled=False,
            trace_thread_time=False,
        )


class _CudaStubDeviceTensor(torch.Tensor):
    """CPU tensor shaped enough for early CUDA-device validation on CPU CI."""

    @property
    def device(self) -> torch.device:
        return torch.device("cuda")


def test_explicit_host_transport_rejects_cuda_tensor_observations() -> None:
    stop_event = threading.Event()

    class _State:
        obs = {
            "obs": torch.zeros((1, 2), device="cuda")
            if torch.cuda.is_available()
            else _CudaStubDeviceTensor()
        }
        info = {}

    class _Env:
        state = _State()

    ring = SharedInferenceRing(1, 2, 2, device="cpu")

    with pytest.raises(
        ValueError, match="Host inference transport received CUDA tensor observations"
    ):
        worker_module._run_collector(
            stop_event=stop_event,
            env_factory=lambda num_envs, env_cfg_override=None: _Env(),
            num_envs=1,
            replay_buffer=SimpleNamespace(
                trace_recorder=None,
                trace_thread_time=False,
                attach_stop_event=lambda stop: None,
            ),
            inference_slot=ring,
            inference_request_queue=queue.Queue(),
            inference_response_queue=queue.Queue(),
            algo_type="sac",
            actor_adapter_modules=None,
            metrics_queue=queue.Queue(),
            inference_transport="host",
            sim_backend="mujoco",
            backend_device=None,
            env_cfg_override={},
            seed=None,
            trace_enabled=False,
            trace_thread_time=False,
        )
