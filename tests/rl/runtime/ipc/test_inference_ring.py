"""Contracts for the bounded learner-owned inference ring."""

from __future__ import annotations

import multiprocessing as mp
from typing import Any

import numpy as np
import pytest
import torch

from unilab.rl.runtime.ipc import inference_ring as inference_ring_module
from unilab.rl.runtime.ipc.inference_ring import CudaRingTransport, SharedInferenceRing


def _observations(tick: int) -> torch.Tensor:
    return torch.arange(6, dtype=torch.float32).reshape(2, 3) + tick


def _dones(tick: int) -> torch.Tensor:
    return torch.tensor([tick % 2, (tick + 1) % 2], dtype=torch.float32)


def _actions(tick: int) -> torch.Tensor:
    return torch.arange(4, dtype=torch.float32).reshape(2, 2) + 10 * tick


def _roundtrip_one(ring: SharedInferenceRing, tick: int, policy_version: int) -> None:
    observations = _observations(tick)
    dones = _dones(tick)
    actions = _actions(tick)
    observation_destination = torch.empty((2, 3))
    done_destination = torch.empty(2)

    ring.publish_observation(
        tick_id=tick,
        observations=observations,
        dones=dones,
        epoch=0,
    )
    ring.copy_observation_to(
        tick_id=tick,
        observations=observation_destination,
        dones=done_destination,
        epoch=0,
    )
    ring.publish_action(
        tick_id=tick,
        policy_version=policy_version,
        actions=actions,
        epoch=0,
    )
    received_actions, received_version = ring.consume_action(
        tick_id=tick,
        epoch=0,
    )

    torch.testing.assert_close(observation_destination, observations)
    torch.testing.assert_close(done_destination, dones)
    torch.testing.assert_close(received_actions, actions)
    assert received_version == policy_version


@pytest.mark.parametrize(
    ("hip_version", "event_wait"),
    [(None, "stream"), ("7.2.53211", "synchronize")],
)
def test_cuda_ring_transport_selects_event_wait_by_runtime(
    monkeypatch: pytest.MonkeyPatch, hip_version: str | None, event_wait: str
) -> None:
    monkeypatch.setattr(torch.version, "hip", hip_version, raising=False)

    transport = inference_ring_module._build_ring_transport(torch.device("cuda"))

    assert isinstance(transport, CudaRingTransport)
    assert transport.event_wait == event_wait


def test_cuda_ring_transport_rejects_unknown_event_wait() -> None:
    with pytest.raises(ValueError, match="event_wait must be 'stream' or 'synchronize'"):
        CudaRingTransport(event_wait="busy-wait")  # type: ignore[arg-type]


def test_cuda_ring_transport_synchronize_waits_on_event_not_stream() -> None:
    class Event:
        def __init__(self) -> None:
            self.synchronize_count = 0

        def synchronize(self) -> None:
            self.synchronize_count += 1

        def wait(self, stream) -> None:
            raise AssertionError("compatibility fallback must not use stream event wait")

    class Ring:
        device = torch.device("cuda")

        def __init__(self, event: Event) -> None:
            self.event = event

        def _observation_event(self, slot: int) -> Event:
            assert slot == 0
            return self.event

        def _action_event(self, slot: int) -> Event:
            assert slot == 0
            return self.event

        def _consumer_done_event(self, slot: int) -> Event:
            assert slot == 0
            return self.event

    event = Event()
    ring = Ring(event)
    transport = CudaRingTransport(event_wait="synchronize")

    transport.wait_observation_ready(ring, 0)
    transport.wait_action_ready(ring, 0)
    transport.wait_consumer_done(ring, 0)

    assert event.synchronize_count == 3


@pytest.mark.parametrize("capacity", [1, 2, 4])
def test_inference_ring_sequences_contiguous_ticks_and_wraps(capacity: int) -> None:
    ring = SharedInferenceRing(
        num_envs=2,
        obs_dim=3,
        action_dim=2,
        capacity=capacity,
        epoch=0,
    )

    for tick in range(12):
        _roundtrip_one(ring, tick, policy_version=tick + 7)

    assert ring.diagnostics == {
        "capacity": capacity,
        "device": "cpu",
        "epoch": 0,
        "published_tick": 11,
        "observation_tick": 11,
        "action_tick": 11,
        "consumed_tick": 11,
        "closed": False,
    }
    assert ring.nbytes == (2 * 3 + 2 + 2 * 2) * capacity * 4


def test_inference_ring_starts_at_tick_zero() -> None:
    ring = SharedInferenceRing(num_envs=1, obs_dim=1, action_dim=1)

    with pytest.raises(RuntimeError, match="publication tick mismatch: expected 0, got 1"):
        ring.publish_observation(
            tick_id=1,
            observations=torch.zeros((1, 1)),
            dones=torch.zeros(1),
        )


def test_inference_ring_rejects_duplicate_and_skipped_publication() -> None:
    ring = SharedInferenceRing(num_envs=1, obs_dim=1, action_dim=1, capacity=2)
    observations = torch.zeros((1, 1))
    dones = torch.zeros(1)
    ring.publish_observation(tick_id=0, observations=observations, dones=dones)

    with pytest.raises(RuntimeError, match="publication tick mismatch: expected 1, got 0"):
        ring.publish_observation(tick_id=0, observations=observations, dones=dones)
    with pytest.raises(RuntimeError, match="publication tick mismatch: expected 1, got 2"):
        ring.publish_observation(tick_id=2, observations=observations, dones=dones)


def test_inference_ring_rejects_response_before_observation_and_stale_actions() -> None:
    ring = SharedInferenceRing(num_envs=1, obs_dim=1, action_dim=1, capacity=2)
    observations = torch.zeros((1, 1))
    dones = torch.zeros(1)
    ring.publish_observation(tick_id=0, observations=observations, dones=dones)

    with pytest.raises(RuntimeError, match="action publication tick mismatch"):
        ring.publish_action(
            tick_id=1,
            policy_version=1,
            actions=torch.ones((1, 1)),
        )

    ring.copy_observation_to(
        tick_id=0,
        observations=torch.empty((1, 1)),
        dones=torch.empty(1),
    )
    ring.publish_action(tick_id=0, policy_version=3, actions=torch.ones((1, 1)))
    ring.publish_observation(tick_id=1, observations=observations, dones=dones)
    ring.copy_observation_to(
        tick_id=1,
        observations=torch.empty((1, 1)),
        dones=torch.empty(1),
    )
    actions, policy_version = ring.consume_action(tick_id=0)
    torch.testing.assert_close(actions, torch.ones((1, 1)))
    assert policy_version == 3

    with pytest.raises(RuntimeError, match="action consumption tick mismatch: expected 1, got 0"):
        ring.consume_action(tick_id=0)
    assert ring.try_consume_action(tick_id=1, timeout_sec=0.0) is None


def test_inference_ring_rejects_duplicate_action_publication() -> None:
    ring = SharedInferenceRing(num_envs=1, obs_dim=1, action_dim=1)
    observations = torch.zeros((1, 1))
    dones = torch.zeros(1)
    ring.publish_observation(tick_id=0, observations=observations, dones=dones)
    ring.copy_observation_to(
        tick_id=0,
        observations=torch.empty((1, 1)),
        dones=torch.empty(1),
    )
    ring.publish_action(
        tick_id=0,
        policy_version=4,
        actions=torch.full((1, 1), 7.0),
    )

    with pytest.raises(RuntimeError, match="action response already published"):
        ring.publish_action(
            tick_id=0,
            policy_version=5,
            actions=torch.full((1, 1), 9.0),
        )

    actions, policy_version = ring.consume_action(tick_id=0)
    torch.testing.assert_close(actions, torch.full((1, 1), 7.0))
    assert policy_version == 4
    ring.publish_observation(tick_id=1, observations=observations, dones=dones)
    ring.copy_observation_to(
        tick_id=1,
        observations=torch.empty((1, 1)),
        dones=torch.empty(1),
    )
    ring.publish_action(
        tick_id=1,
        policy_version=6,
        actions=torch.full((1, 1), 11.0),
    )
    actions, policy_version = ring.consume_action(tick_id=1)
    torch.testing.assert_close(actions, torch.full((1, 1), 11.0))
    assert policy_version == 6


def test_inference_ring_rejects_out_of_order_observation_and_action_headers() -> None:
    ring = SharedInferenceRing(num_envs=1, obs_dim=1, action_dim=1, capacity=2)
    observations = torch.zeros((1, 1))
    dones = torch.zeros(1)
    ring.publish_observation(tick_id=0, observations=observations, dones=dones)

    with pytest.raises(
        RuntimeError, match="observation consumption tick mismatch: expected 0, got 1"
    ):
        ring.copy_observation_to(
            tick_id=1,
            observations=torch.empty((1, 1)),
            dones=torch.empty(1),
        )

    ring.copy_observation_to(
        tick_id=0,
        observations=torch.empty((1, 1)),
        dones=torch.empty(1),
    )
    ring.publish_action(tick_id=0, policy_version=1, actions=torch.ones((1, 1)))
    ring.publish_observation(tick_id=1, observations=observations, dones=dones)
    ring.copy_observation_to(
        tick_id=1,
        observations=torch.empty((1, 1)),
        dones=torch.empty(1),
    )


def test_inference_ring_epoch_is_fail_closed() -> None:
    ring = SharedInferenceRing(num_envs=1, obs_dim=1, action_dim=1, epoch=7)

    with pytest.raises(RuntimeError, match="Inference epoch mismatch: expected 7, got 6"):
        ring.publish_observation(
            tick_id=0,
            observations=torch.zeros((1, 1)),
            dones=torch.zeros(1),
            epoch=6,
        )
    ring.publish_observation(
        tick_id=0,
        observations=torch.zeros((1, 1)),
        dones=torch.zeros(1),
        epoch=7,
    )
    with pytest.raises(RuntimeError, match="Inference epoch mismatch: expected 7, got 8"):
        ring.copy_observation_to(
            tick_id=0,
            observations=torch.empty((1, 1)),
            dones=torch.empty(1),
            epoch=8,
        )


def test_inference_ring_backpressure_releases_after_action_consumption() -> None:
    ring = SharedInferenceRing(num_envs=1, obs_dim=1, action_dim=1, capacity=1)
    observations = torch.zeros((1, 1))
    dones = torch.zeros(1)
    ring.publish_observation(tick_id=0, observations=observations, dones=dones)

    assert (
        ring.try_publish_observation(
            tick_id=1,
            observations=observations,
            dones=dones,
            timeout_sec=0.0,
        )
        is False
    )

    ring.copy_observation_to(
        tick_id=0,
        observations=torch.empty((1, 1)),
        dones=torch.empty(1),
    )
    ring.publish_action(tick_id=0, policy_version=2, actions=torch.ones((1, 1)))
    ring.consume_action(tick_id=0)

    assert (
        ring.try_publish_observation(
            tick_id=1,
            observations=observations,
            dones=dones,
            timeout_sec=0.0,
        )
        is True
    )


def test_inference_ring_supports_bounded_multi_tick_observation_flight() -> None:
    ring = SharedInferenceRing(
        num_envs=2,
        obs_dim=3,
        action_dim=2,
        capacity=4,
    )
    observations = torch.empty((2, 3))
    dones = torch.empty(2)

    for tick in range(4):
        ring.publish_observation(
            tick_id=tick,
            observations=_observations(tick),
            dones=_dones(tick),
        )
        assert ring.diagnostics["published_tick"] == tick
        assert ring.diagnostics["observation_tick"] == -1
        assert ring.diagnostics["action_tick"] == -1
        assert ring.diagnostics["consumed_tick"] == -1

    assert (
        ring.try_publish_observation(
            tick_id=4,
            observations=_observations(4),
            dones=_dones(4),
            timeout_sec=0.0,
        )
        is False
    )

    for tick in range(4):
        ring.copy_observation_to(
            tick_id=tick,
            observations=observations,
            dones=dones,
        )
        torch.testing.assert_close(observations, _observations(tick))
        ring.publish_action(
            tick_id=tick,
            policy_version=tick + 1,
            actions=_actions(tick),
        )

    for tick in range(4):
        actions, policy_version = ring.consume_action(tick_id=tick)
        torch.testing.assert_close(actions, _actions(tick))
        assert policy_version == tick + 1

    assert ring.diagnostics["published_tick"] == 3
    assert ring.diagnostics["action_tick"] == 3
    assert ring.diagnostics["consumed_tick"] == 3


def test_inference_ring_constructor_failure_cleans_partial_resources(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    closed_semaphores: list[int] = []

    class TrackingSemaphore:
        def __init__(self, initial_value: int) -> None:
            self.initial_value = initial_value

        def close(self) -> None:
            closed_semaphores.append(self.initial_value)

    original_lock = inference_ring_module._SPAWN_CTX.Lock

    def fail_lock() -> None:
        raise RuntimeError("injected lock allocation failure")

    monkeypatch.setattr(
        inference_ring_module._SPAWN_CTX, "Semaphore", TrackingSemaphore, raising=False
    )
    monkeypatch.setattr(inference_ring_module._SPAWN_CTX, "Lock", fail_lock, raising=False)

    with pytest.raises(RuntimeError, match="injected lock allocation failure"):
        SharedInferenceRing(num_envs=1, obs_dim=1, action_dim=1, capacity=2)

    monkeypatch.setattr(inference_ring_module._SPAWN_CTX, "Lock", original_lock, raising=False)
    assert closed_semaphores == [2, 0, 0]


def test_inference_ring_close_is_idempotent_and_keeps_diagnostics() -> None:
    ring = SharedInferenceRing(num_envs=2, obs_dim=3, action_dim=2, capacity=3)
    expected_bytes = ring.nbytes

    ring.close()
    ring.cleanup()

    assert ring.diagnostics["closed"] is True
    assert ring.nbytes == expected_bytes
    with pytest.raises(RuntimeError, match="inference ring is closed"):
        ring.publish_observation(
            tick_id=0,
            observations=torch.zeros((2, 3)),
            dones=torch.zeros(2),
        )


def test_inference_ring_validates_constructor_and_payload_contract() -> None:
    with pytest.raises(ValueError, match="dimensions and capacity must be positive"):
        SharedInferenceRing(num_envs=0, obs_dim=1, action_dim=1)
    with pytest.raises(TypeError, match="capacity must be an integer"):
        SharedInferenceRing(num_envs=1, obs_dim=1, action_dim=1, capacity=True)
    with pytest.raises(ValueError, match="epoch must be non-negative"):
        SharedInferenceRing(num_envs=1, obs_dim=1, action_dim=1, epoch=-1)
    with pytest.raises(ValueError, match="CPU and CUDA only"):
        SharedInferenceRing(num_envs=1, obs_dim=1, action_dim=1, device="mps")

    ring = SharedInferenceRing(num_envs=1, obs_dim=2, action_dim=1)
    with pytest.raises(ValueError, match="observation shape must be"):
        ring.publish_observation(
            tick_id=0,
            observations=torch.zeros((1, 3)),
            dones=torch.zeros(1),
        )
    with pytest.raises(TypeError, match="observations and dones must be Torch tensors"):
        ring.publish_observation(tick_id=0, observations=None, dones=torch.zeros(1))
    with pytest.raises(TypeError, match="observations and dones must be Torch tensors"):
        ring.publish_observation(
            tick_id=0,
            observations=np.zeros((1, 2), dtype=np.float32),
            dones=np.zeros(1, dtype=np.float32),
        )
    with pytest.raises(TypeError, match="observation destinations must be Torch tensors"):
        ring.copy_observation_to(
            tick_id=0,
            observations=np.zeros((1, 2), dtype=np.float32),
            dones=torch.empty(1),
        )
    with pytest.raises(TypeError, match="destinations must have float32 dtype"):
        ring.copy_observation_to(
            tick_id=0,
            observations=torch.empty((1, 2), dtype=torch.float64),
            dones=torch.empty(1),
        )
    with pytest.raises(ValueError, match="policy version must be non-negative"):
        ring.publish_action(
            tick_id=0,
            policy_version=-1,
            actions=torch.empty((1, 1)),
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_cuda_inference_ring_waits_for_delayed_consumer_clone() -> None:
    ring = SharedInferenceRing(
        num_envs=2,
        obs_dim=3,
        action_dim=2,
        device="cuda",
        capacity=1,
    )
    observations = torch.zeros((2, 3), device="cuda")
    dones = torch.zeros(2, device="cuda")

    ring.publish_observation(tick_id=0, observations=observations, dones=dones, epoch=0)
    ring.copy_observation_to(
        tick_id=0,
        observations=torch.empty_like(observations),
        dones=torch.empty_like(dones),
        epoch=0,
    )
    ring.publish_action(
        tick_id=0,
        policy_version=1,
        actions=torch.zeros((2, 2), device="cuda"),
        epoch=0,
    )

    with torch.cuda.Stream() as delayed_stream:
        with torch.cuda.stream(delayed_stream):
            torch.cuda._sleep(50_000_000)
            received, _ = ring.consume_action(tick_id=0, epoch=0)

    ring.publish_observation(tick_id=1, observations=observations, dones=dones, epoch=0)
    ring.copy_observation_to(
        tick_id=1,
        observations=torch.empty_like(observations),
        dones=torch.empty_like(dones),
        epoch=0,
    )
    ring.publish_action(
        tick_id=1,
        policy_version=2,
        actions=torch.ones((2, 2), device="cuda"),
        epoch=0,
    )
    torch.cuda.synchronize()

    torch.testing.assert_close(received, torch.zeros((2, 2), device="cuda"))


def _cuda_ring_child(ring: SharedInferenceRing, result_queue: Any, tick_count: int = 3) -> None:
    try:
        for tick in range(tick_count):
            ring.publish_observation(
                tick_id=tick,
                observations=(
                    torch.arange(6, dtype=torch.float32, device=ring.device).reshape(2, 3) + tick
                ),
                dones=torch.tensor([0.0, 1.0], device=ring.device),
                epoch=0,
            )
            actions, policy_version = ring.consume_action(tick_id=tick, epoch=0)
            expected_action = float(10 * tick)
            if not bool(torch.all(actions == expected_action).item()):
                raise AssertionError(f"action {tick} was overwritten: {actions.cpu()}")
        ring.close()
        result_queue.put(ring.diagnostics)
    except BaseException as error:  # pragma: no cover - propagated to parent
        result_queue.put(error)


@pytest.mark.slow
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_cuda_inference_ring_ipc_events_cross_process() -> None:
    spawn_context = mp.get_context("spawn")
    result_queue = spawn_context.Queue()
    ring = SharedInferenceRing(
        num_envs=2,
        obs_dim=3,
        action_dim=2,
        device="cuda",
        capacity=2,
        epoch=0,
    )
    process = spawn_context.Process(target=_cuda_ring_child, args=(ring, result_queue))

    process.start()
    try:
        observations = torch.empty((2, 3), device="cuda")
        dones = torch.empty(2, device="cuda")
        for tick in range(3):
            ring.copy_observation_to(
                tick_id=tick,
                observations=observations,
                dones=dones,
                epoch=0,
            )
            torch.testing.assert_close(
                observations,
                torch.arange(6, dtype=torch.float32, device="cuda").reshape(2, 3) + tick,
            )
            ring.publish_action(
                tick_id=tick,
                policy_version=tick + 10,
                actions=torch.full((2, 2), float(10 * tick), dtype=torch.float32, device="cuda"),
                epoch=0,
            )
        result: Any | BaseException = result_queue.get(timeout=30)
    finally:
        process.join(timeout=30)
        if process.is_alive():  # pragma: no cover - failure path
            process.terminate()
            process.join()
        ring.close()

    assert not process.exitcode, f"child exited with {process.exitcode}"
    assert not isinstance(result, BaseException), result
    diagnostics = result
    assert diagnostics["published_tick"] == 2
    assert diagnostics["consumed_tick"] == 2
    assert diagnostics["closed"] is True
    assert not hasattr(ring, "observations")


@pytest.mark.slow
@pytest.mark.skipif(
    not torch.cuda.is_available() or not torch.version.hip,
    reason="ROCm/HIP CUDA runtime not available",
)
def test_hip_inference_ring_survives_rocm_ipc_event_slot_reuse() -> None:
    """Cover the 33rd IPC-event stream wait that breaks unfixed ROCm CLR 7.2."""
    spawn_context = mp.get_context("spawn")
    result_queue = spawn_context.Queue()
    ring = SharedInferenceRing(
        num_envs=2,
        obs_dim=3,
        action_dim=2,
        device="cuda",
        capacity=1,
        epoch=0,
    )
    process = spawn_context.Process(
        target=_cuda_ring_child,
        args=(ring, result_queue, 40),
    )
    process.start()
    try:
        observations = torch.empty((2, 3), device=ring.device)
        dones = torch.empty(2, device=ring.device)
        for tick in range(40):
            ring.copy_observation_to(
                tick_id=tick,
                observations=observations,
                dones=dones,
                epoch=0,
            )
            ring.publish_action(
                tick_id=tick,
                policy_version=tick + 10,
                actions=torch.full((2, 2), float(10 * tick), device=ring.device),
                epoch=0,
            )
        result: Any | BaseException = result_queue.get(timeout=30)
    finally:
        process.join(timeout=30)
        if process.is_alive():  # pragma: no cover - failure path
            process.terminate()
            process.join()
        ring.close()
        result_queue.close()
        result_queue.join_thread()

    assert not process.exitcode, f"child exited with {process.exitcode}"
    assert not isinstance(result, BaseException), result
    assert result["published_tick"] == 39
    assert result["consumed_tick"] == 39
    assert result["closed"] is True


def _delayed_consumer_reuse_child(
    ring: SharedInferenceRing,
    result_queue: Any,
    clone_enqueued_queue: Any,
) -> None:
    try:
        # Publish two observations before either action is consumed.  The learner
        # can therefore publish two distinct actions while the collector-side
        # consumer is deliberately delayed on another CUDA stream.
        for tick in range(2):
            ring.publish_observation(
                tick_id=tick,
                observations=(
                    torch.arange(6, dtype=torch.float32, device=ring.device).reshape(2, 3) + tick
                ),
                dones=torch.tensor([0.0, 1.0], device=ring.device),
                epoch=0,
            )

        with torch.cuda.Stream() as delayed_stream:
            with torch.cuda.stream(delayed_stream):
                # The enqueue must return before this long-running stream work
                # completes.  Returning is what makes immediate slot release a
                # real race rather than a hidden device synchronization.
                torch.cuda._sleep(200_000_000)
                action0, _ = ring.consume_action(tick_id=0, epoch=0)
                action1, _ = ring.consume_action(tick_id=1, epoch=0)

        # This host barrier lets the producer know that both stream-ordered
        # consume/clone operations are already enqueued before tick 2 can exist.
        clone_enqueued_queue.put(1)

        # Reusing physical slot 0 requires publishing tick 2.  This happens
        # after both consume calls return but potentially before either delayed
        # clone has executed.
        ring.publish_observation(
            tick_id=2,
            observations=torch.full((2, 3), 2.0, device=ring.device),
            dones=torch.tensor([0.0, 1.0], device=ring.device),
            epoch=0,
        )
        action2, _ = ring.consume_action(tick_id=2, epoch=0)
        torch.cuda.synchronize(ring.device)
        values = [
            float(action0.flatten()[0].item()),
            float(action1.flatten()[0].item()),
            float(action2.flatten()[0].item()),
        ]
        ring.close()
        result_queue.put({"action_values": values, "diagnostics": ring.diagnostics})
    except BaseException as error:  # pragma: no cover - propagated to parent
        result_queue.put(error)


@pytest.mark.slow
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_cuda_inference_ring_cross_process_reuse_waits_for_delayed_clone() -> None:
    """Combine cross-process IPC, delayed clones, wraparound, and slot reuse.

    Host queues prove only that stream-ordered clones were enqueued. The ring's
    interprocess consumer-done event must keep the next producer copy ordered
    behind those delayed clone kernels.
    """
    spawn_context = mp.get_context("spawn")
    result_queue = spawn_context.Queue()
    clone_enqueued_queue = spawn_context.Queue()
    ring = SharedInferenceRing(
        num_envs=2,
        obs_dim=3,
        action_dim=2,
        device="cuda",
        capacity=2,
        epoch=0,
    )
    process = spawn_context.Process(
        target=_delayed_consumer_reuse_child,
        args=(ring, result_queue, clone_enqueued_queue),
    )

    process.start()
    try:
        observations = torch.empty((2, 3), device=ring.device)
        dones = torch.empty(2, device=ring.device)
        for tick in range(2):
            ring.copy_observation_to(
                tick_id=tick,
                observations=observations,
                dones=dones,
                epoch=0,
            )
            ring.publish_action(
                tick_id=tick,
                policy_version=tick + 10,
                actions=torch.full(
                    (2, 2), float(10 * tick), dtype=torch.float32, device=ring.device
                ),
                epoch=0,
            )

        assert clone_enqueued_queue.get(timeout=30) == 1

        # This call cannot proceed until the child has consumed ticks 0/1 and
        # published tick 2.  At that point the shared consumer-done headers prove
        # both producer/consumer event chains have been recorded in order.
        ring.copy_observation_to(
            tick_id=2,
            observations=observations,
            dones=dones,
            epoch=0,
            timeout_sec=30,
        )
        assert list(ring._consumer_done_ticks) == [0, 1]
        assert ring.diagnostics["published_tick"] == 2
        assert ring.diagnostics["observation_tick"] == 2
        assert ring.diagnostics["action_tick"] == 1
        assert ring.diagnostics["consumed_tick"] == 1

        ring.publish_action(
            tick_id=2,
            policy_version=12,
            actions=torch.full((2, 2), 20.0, dtype=torch.float32, device=ring.device),
            epoch=0,
        )
        result: Any | BaseException = result_queue.get(timeout=30)
    finally:
        process.join(timeout=30)
        if process.is_alive():  # pragma: no cover - failure path
            process.terminate()
            process.join()
        ring.close()
        result_queue.close()
        result_queue.join_thread()
        clone_enqueued_queue.close()
        clone_enqueued_queue.join_thread()

    assert not process.exitcode, f"child exited with {process.exitcode}"
    assert not isinstance(result, BaseException), result
    values = result["action_values"]
    assert values == [0.0, 10.0, 20.0]
    assert result["diagnostics"]["consumed_tick"] == 2
    assert result["diagnostics"]["action_tick"] == 2
    assert result["diagnostics"]["closed"] is True
    assert ring.diagnostics["consumed_tick"] == 2
    assert ring.diagnostics["action_tick"] == 2


def _consumer_done_negative_child(
    ring: SharedInferenceRing,
    result_queue: Any,
    clone_staged_queue: Any,
    release_clone_event: Any,
) -> None:
    try:
        if not ring._acquire(ring._action_ready, 30.0):
            raise TimeoutError("negative-control action was not ready")
        stream = torch.cuda.current_stream(ring.device)
        ring._action_event(0).wait(stream)

        # Simulate consume_action returning after the action event is observed:
        # release the physical slot and publish consumer-done metadata before the
        # deliberately blocked clone executes. The real CUDA event is not recorded
        # yet, which is exactly the producer-side guard under test.
        ring._response_ticks[0] = -1
        ring._policy_versions[0] = -1
        ring._consumed_tick.value = 0
        ring._consumer_done_ticks[0] = 0
        ring._free_slots.release()
        clone_staged_queue.put(1)
        if not release_clone_event.wait(30.0):
            raise TimeoutError("negative-control clone was not released")

        old_actions = ring._clone_action(0)
        ring._consumer_done_event(0).record(stream)
        overwritten_actions, overwritten_version = ring.consume_action(
            tick_id=1,
            epoch=0,
        )
        torch.cuda.synchronize(ring.device)
        ring.close()
        result_queue.put(
            {
                "old_action": float(old_actions.flatten()[0].item()),
                "overwritten_action": float(overwritten_actions.flatten()[0].item()),
                "overwritten_version": int(overwritten_version),
                "diagnostics": ring.diagnostics,
            }
        )
    except BaseException as error:  # pragma: no cover - propagated to parent
        result_queue.put(error)


@pytest.mark.slow
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_cuda_inference_ring_consumer_done_guard_has_deterministic_negative_control(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Prove that ignoring consumer-done permits a stale clone to be corrupted."""
    spawn_context = mp.get_context("spawn")
    result_queue = spawn_context.Queue()
    clone_staged_queue = spawn_context.Queue()
    release_clone_event = spawn_context.Event()
    ring = SharedInferenceRing(
        num_envs=2,
        obs_dim=3,
        action_dim=2,
        device="cuda",
        capacity=1,
        epoch=0,
    )
    process = spawn_context.Process(
        target=_consumer_done_negative_child,
        args=(ring, result_queue, clone_staged_queue, release_clone_event),
    )
    process.start()

    class _NoWaitEvent:
        def wait(self, stream) -> None:
            del stream

    try:
        ring.publish_observation(
            tick_id=0,
            observations=torch.zeros((2, 3), device=ring.device),
            dones=torch.zeros(2, device=ring.device),
            epoch=0,
        )
        ring.copy_observation_to(
            tick_id=0,
            observations=torch.empty((2, 3), device=ring.device),
            dones=torch.empty(2, device=ring.device),
            epoch=0,
        )
        ring.publish_action(
            tick_id=0,
            policy_version=10,
            actions=torch.zeros((2, 2), device=ring.device),
            epoch=0,
        )
        assert clone_staged_queue.get(timeout=30) == 1

        # The producer observes the metadata but skips the actual interprocess
        # event wait. Patch after spawn so only the producer carries this defect.
        monkeypatch.setattr(
            ring,
            "_consumer_done_event",
            lambda slot: _NoWaitEvent(),
            raising=False,
        )
        ring.publish_observation(
            tick_id=1,
            observations=torch.ones((2, 3), device=ring.device),
            dones=torch.zeros(2, device=ring.device),
            epoch=0,
        )
        ring.copy_observation_to(
            tick_id=1,
            observations=torch.empty((2, 3), device=ring.device),
            dones=torch.empty(2, device=ring.device),
            epoch=0,
        )
        ring.publish_action(
            tick_id=1,
            policy_version=11,
            actions=torch.full((2, 2), 20.0, device=ring.device),
            epoch=0,
        )
        release_clone_event.set()
        result: Any | BaseException = result_queue.get(timeout=30)
    finally:
        process.join(timeout=30)
        if process.is_alive():  # pragma: no cover - failure path
            process.terminate()
            process.join()
        ring.close()
        result_queue.close()
        result_queue.join_thread()
        clone_staged_queue.close()
        clone_staged_queue.join_thread()

    assert not process.exitcode, f"child exited with {process.exitcode}"
    assert not isinstance(result, BaseException), result
    assert result["old_action"] == 20.0
    assert result["overwritten_action"] == 20.0
    assert result["overwritten_version"] == 11
    assert result["diagnostics"]["consumed_tick"] == 1
    assert result["diagnostics"]["closed"] is True
