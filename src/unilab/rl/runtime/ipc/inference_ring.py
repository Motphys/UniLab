"""Bounded, ordered inference exchange for learner-owned actors.

The ring keeps the successful single-slot ownership model but adds explicit
capacity, epochs, contiguous tick validation, and fail-closed backpressure.
The collector/learner boundary is Torch-tensor-only on every platform: both
CPU and CUDA rings return cloned tensor actions, and observation publication
rejects non-tensor inputs. CUDA publications use per-slot interprocess events
rather than synchronizing the complete producer stream before every CPU-ready
flag.
"""

from __future__ import annotations

import multiprocessing as mp
from typing import Any, Literal, Protocol, cast

import torch

from unilab.rl.runtime.device import sync_device

_SPAWN_CTX = mp.get_context("spawn")

CudaEventWait = Literal["stream", "synchronize"]


class RingTransport(Protocol):
    """Device-specific staging and event mechanics for an inference ring.

    Placement/topology policy remains in :mod:`unilab.rl.runtime.placement`;
    implementations here only consume an already allocated ring.
    """

    def validate_observation_inputs(
        self, ring: "SharedInferenceRing", observations: torch.Tensor, dones: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]: ...

    def validate_action_input(
        self, ring: "SharedInferenceRing", actions: torch.Tensor
    ) -> torch.Tensor: ...

    def stage_observation(
        self,
        ring: "SharedInferenceRing",
        slot: int,
        observations: torch.Tensor,
        dones: torch.Tensor,
    ) -> None: ...

    def record_observation_ready(self, ring: "SharedInferenceRing", slot: int) -> None: ...

    def wait_observation_ready(self, ring: "SharedInferenceRing", slot: int) -> None: ...

    def publish_action(
        self,
        ring: "SharedInferenceRing",
        slot: int,
        actions: torch.Tensor,
        *,
        non_blocking: bool,
    ) -> None: ...

    def clone_action(self, ring: "SharedInferenceRing", slot: int) -> torch.Tensor: ...

    def wait_action_ready(self, ring: "SharedInferenceRing", slot: int) -> None: ...

    def wait_consumer_done(self, ring: "SharedInferenceRing", slot: int) -> None: ...

    def record_action_ready(self, ring: "SharedInferenceRing", slot: int) -> None: ...

    def record_consumer_done(self, ring: "SharedInferenceRing", slot: int) -> None: ...

    def record_consumer_done_tick(
        self, ring: "SharedInferenceRing", slot: int, tick_id: int
    ) -> None: ...


class CpuRingTransport:
    """Synchronous staging transport for a CPU inference ring."""

    def validate_observation_inputs(
        self, ring: "SharedInferenceRing", observations: torch.Tensor, dones: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if observations.device.type != "cpu" or dones.device.type != "cpu":
            raise ValueError("CPU inference rings require CPU tensor observations")
        return observations, dones

    def validate_action_input(
        self, ring: "SharedInferenceRing", actions: torch.Tensor
    ) -> torch.Tensor:
        if actions.device.type not in {"cpu", "cuda"}:
            raise ValueError("CPU inference actions must live on CPU or CUDA")
        return actions

    def stage_observation(
        self,
        ring: "SharedInferenceRing",
        slot: int,
        observations: torch.Tensor,
        dones: torch.Tensor,
    ) -> None:
        ring.observations[slot].copy_(observations)
        ring.dones[slot].copy_(dones)

    def record_observation_ready(self, ring: "SharedInferenceRing", slot: int) -> None:
        del ring, slot

    def wait_observation_ready(self, ring: "SharedInferenceRing", slot: int) -> None:
        del ring, slot

    def publish_action(
        self,
        ring: "SharedInferenceRing",
        slot: int,
        actions: torch.Tensor,
        *,
        non_blocking: bool,
    ) -> None:
        ring.actions[slot].copy_(actions, non_blocking=non_blocking)
        if actions.device.type == "cuda":
            # The shared CPU storage cannot be released to a new producer until
            # the async device-to-host copy has completed.
            sync_device(actions.device, stream=True)

    def clone_action(self, ring: "SharedInferenceRing", slot: int) -> torch.Tensor:
        return ring.actions[slot].clone()

    def wait_action_ready(self, ring: "SharedInferenceRing", slot: int) -> None:
        del ring, slot

    def wait_consumer_done(self, ring: "SharedInferenceRing", slot: int) -> None:
        del ring, slot

    def record_action_ready(self, ring: "SharedInferenceRing", slot: int) -> None:
        del ring, slot

    def record_consumer_done(self, ring: "SharedInferenceRing", slot: int) -> None:
        del ring, slot

    def record_consumer_done_tick(
        self, ring: "SharedInferenceRing", slot: int, tick_id: int
    ) -> None:
        del ring, slot, tick_id


class CudaRingTransport:
    """Per-slot IPC-event staging transport for a CUDA inference ring."""

    def __init__(self, *, event_wait: CudaEventWait = "stream") -> None:
        if event_wait not in ("stream", "synchronize"):
            raise ValueError(
                f"CUDA inference event_wait must be 'stream' or 'synchronize', got {event_wait!r}"
            )
        self.event_wait = event_wait

    def validate_observation_inputs(
        self, ring: "SharedInferenceRing", observations: torch.Tensor, dones: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if observations.device != ring.device or dones.device != ring.device:
            raise ValueError("CUDA inference rings require tensors on the ring device")
        return observations, dones

    def validate_action_input(
        self, ring: "SharedInferenceRing", actions: torch.Tensor
    ) -> torch.Tensor:
        if actions.device != ring.device:
            raise ValueError("CUDA inference actions must live on the ring device")
        return actions

    def stage_observation(
        self,
        ring: "SharedInferenceRing",
        slot: int,
        observations: torch.Tensor,
        dones: torch.Tensor,
    ) -> None:
        ring.observations[slot].copy_(observations)
        ring.dones[slot].copy_(dones)

    def record_observation_ready(self, ring: "SharedInferenceRing", slot: int) -> None:
        ring._observation_event(slot).record(torch.cuda.current_stream(ring.device))

    def wait_observation_ready(self, ring: "SharedInferenceRing", slot: int) -> None:
        self._wait_event(ring._observation_event(slot), ring.device)

    def publish_action(
        self,
        ring: "SharedInferenceRing",
        slot: int,
        actions: torch.Tensor,
        *,
        non_blocking: bool,
    ) -> None:
        ring.actions[slot].copy_(actions, non_blocking=non_blocking)

    def clone_action(self, ring: "SharedInferenceRing", slot: int) -> torch.Tensor:
        return ring.actions[slot].clone()

    def wait_action_ready(self, ring: "SharedInferenceRing", slot: int) -> None:
        self._wait_event(ring._action_event(slot), ring.device)

    def wait_consumer_done(self, ring: "SharedInferenceRing", slot: int) -> None:
        self._wait_event(ring._consumer_done_event(slot), ring.device)

    def record_action_ready(self, ring: "SharedInferenceRing", slot: int) -> None:
        ring._action_event(slot).record(torch.cuda.current_stream(ring.device))

    def record_consumer_done(self, ring: "SharedInferenceRing", slot: int) -> None:
        del ring, slot

    def record_consumer_done_tick(
        self, ring: "SharedInferenceRing", slot: int, tick_id: int
    ) -> None:
        ring._consumer_done_event(slot).record(torch.cuda.current_stream(ring.device))
        ring._consumer_done_ticks[slot] = int(tick_id)

    def _wait_event(self, event: torch.cuda.Event, device: torch.device) -> None:
        if self.event_wait == "synchronize":
            event.synchronize()
            return
        event.wait(torch.cuda.current_stream(device))


def _build_ring_transport(device: torch.device) -> RingTransport:
    if device.type == "cpu":
        return CpuRingTransport()
    if device.type == "cuda":
        # PERFORMANCE-COMPATIBILITY FALLBACK: ROCm/HIP 7.2 emulated IPC events
        # omit the signal-ring modulo in hipStreamWaitEvent (ROCm CLR a44f100
        # fixes it), so the 33rd reuse of an event passes signal[32] and fails
        # with hipErrorInvalidValue. Event.synchronize uses the correctly
        # modulo-indexed ROCm path but blocks the host and reduces overlap; use
        # it only until every supported HIP runtime includes the CLR fix.
        return CudaRingTransport(
            event_wait="synchronize" if bool(getattr(torch.version, "hip", None)) else "stream"
        )
    raise ValueError(f"inference rings support CPU and CUDA only, got {device.type!r}")


def estimate_inference_ring_bytes(
    num_envs: int, obs_dim: int, action_dim: int, *, capacity: int
) -> int:
    """Return the exact float32 storage bytes for one inference ring."""
    values = (num_envs, obs_dim, action_dim, capacity)
    if any(isinstance(value, bool) or not isinstance(value, int) for value in values):
        raise TypeError("inference ring byte dimensions must be integers")
    if min(values) <= 0:
        raise ValueError("inference ring byte dimensions and capacity must be positive")
    columns = obs_dim + 1 + action_dim
    return num_envs * capacity * columns * 4


class SharedInferenceRing:
    """Fixed-capacity observation/action ring used by one collector and learner."""

    def __init__(
        self,
        num_envs: int,
        obs_dim: int,
        action_dim: int,
        *,
        device: str | torch.device = "cpu",
        capacity: int = 1,
        epoch: int = 0,
    ) -> None:
        if isinstance(num_envs, bool) or not isinstance(num_envs, int):
            raise TypeError("inference ring num_envs must be an integer")
        if isinstance(obs_dim, bool) or not isinstance(obs_dim, int):
            raise TypeError("inference ring obs_dim must be an integer")
        if isinstance(action_dim, bool) or not isinstance(action_dim, int):
            raise TypeError("inference ring action_dim must be an integer")
        if isinstance(capacity, bool) or not isinstance(capacity, int):
            raise TypeError("inference ring capacity must be an integer")
        if isinstance(epoch, bool) or not isinstance(epoch, int):
            raise TypeError("inference ring epoch must be an integer")
        if min(num_envs, obs_dim, action_dim, capacity) <= 0:
            raise ValueError("inference ring dimensions and capacity must be positive")
        if epoch < 0:
            raise ValueError("inference ring epoch must be non-negative")

        target = torch.device(device)
        if target.type not in {"cpu", "cuda"}:
            raise ValueError(f"inference rings support CPU and CUDA only, got {target!s}")
        if target.type == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError(f"CUDA inference ring requested on unavailable {target}")
            if target.index is None:
                target = torch.device("cuda", index=torch.cuda.current_device())
            if not hasattr(torch.cuda.Event, "from_ipc_handle"):
                raise RuntimeError("PyTorch CUDA events do not support interprocess handles")

        self.num_envs = int(num_envs)
        self.obs_dim = int(obs_dim)
        self.action_dim = int(action_dim)
        self.capacity = int(capacity)
        self.device = target
        self._epoch = _SPAWN_CTX.Value("q", int(epoch))
        self._closed = _SPAWN_CTX.Value("i", 0)
        self._published_tick = _SPAWN_CTX.Value("q", -1)
        self._observation_tick = _SPAWN_CTX.Value("q", -1)
        self._action_tick = _SPAWN_CTX.Value("q", -1)
        self._consumed_tick = _SPAWN_CTX.Value("q", -1)
        self._request_ticks = _SPAWN_CTX.RawArray("q", self.capacity)
        self._request_epochs = _SPAWN_CTX.RawArray("q", self.capacity)
        self._response_ticks = _SPAWN_CTX.RawArray("q", self.capacity)
        self._policy_versions = _SPAWN_CTX.RawArray("q", self.capacity)
        self._consumer_done_ticks = _SPAWN_CTX.RawArray("q", self.capacity)
        for index in range(self.capacity):
            self._request_ticks[index] = -1
            self._request_epochs[index] = -1
            self._response_ticks[index] = -1
            self._policy_versions[index] = -1
            self._consumer_done_ticks[index] = -1

        self._shared_semaphores: list = []
        self._shared_locks: list = []
        self._transport = _build_ring_transport(target)
        self._local_observation_events: dict[int, torch.cuda.Event] = {}
        self._local_action_events: dict[int, torch.cuda.Event] = {}
        self._local_consumer_done_events: dict[int, torch.cuda.Event] = {}
        self._observation_event_handles: tuple[bytes, ...] = ()
        self._action_event_handles: tuple[bytes, ...] = ()
        self._consumer_done_event_handles: tuple[bytes, ...] = ()
        self._local_closed = False
        try:
            self.observations = torch.empty(
                (self.capacity, self.num_envs, self.obs_dim),
                dtype=torch.float32,
                device=target,
            ).share_memory_()
            self.dones = torch.empty(
                (self.capacity, self.num_envs), dtype=torch.float32, device=target
            ).share_memory_()
            self.actions = torch.empty(
                (self.capacity, self.num_envs, self.action_dim),
                dtype=torch.float32,
                device=target,
            ).share_memory_()

            self._free_slots = _SPAWN_CTX.Semaphore(self.capacity)
            self._observation_ready = _SPAWN_CTX.Semaphore(0)
            self._action_ready = _SPAWN_CTX.Semaphore(0)
            self._shared_semaphores = [
                self._free_slots,
                self._observation_ready,
                self._action_ready,
            ]
            self._producer_lock = _SPAWN_CTX.Lock()
            self._observation_lock = _SPAWN_CTX.Lock()
            self._response_lock = _SPAWN_CTX.Lock()
            self._action_lock = _SPAWN_CTX.Lock()
            self._shared_locks = [
                self._producer_lock,
                self._observation_lock,
                self._response_lock,
                self._action_lock,
            ]

            if target.type == "cuda":
                event_groups = (
                    ("observation", self._local_observation_events),
                    ("action", self._local_action_events),
                    ("consumer_done", self._local_consumer_done_events),
                )
                for event_kind, local_events in event_groups:
                    events = []
                    handles = []
                    try:
                        for slot in range(self.capacity):
                            event = torch.cuda.Event(interprocess=True)
                            events.append(event)
                            local_events[slot] = event
                            handles.append(event.ipc_handle())
                    finally:
                        if len(handles) != self.capacity:
                            local_events.clear()
                            if event_kind == "observation":
                                self._observation_event_handles = ()
                            elif event_kind == "action":
                                self._action_event_handles = ()
                            else:
                                self._consumer_done_event_handles = ()
                        elif event_kind == "observation":
                            self._observation_event_handles = tuple(handles)
                        elif event_kind == "action":
                            self._action_event_handles = tuple(handles)
                        else:
                            self._consumer_done_event_handles = tuple(handles)
        except BaseException:
            # Keep constructor failures deterministic: no partially allocated
            # tensor, local CUDA event, IPC handle, semaphore, or lock is left
            # for garbage collection to discover later.
            self._closed.value = 1
            self._local_closed = True
            self._release_local_resources()
            raise

    def __getstate__(self) -> dict[str, Any]:
        # CUDA event wrappers are process-local. Their IPC handles are copied to
        # the spawn process and reconstructed lazily after CUDA initialization.
        state = self.__dict__.copy()
        state["_transport"] = _build_ring_transport(self.device)
        state["_local_observation_events"] = {}
        state["_local_action_events"] = {}
        state["_local_consumer_done_events"] = {}
        state["_local_closed"] = False
        return state

    @property
    def nbytes(self) -> int:
        values = (
            self.capacity * self.num_envs * self.obs_dim,
            self.capacity * self.num_envs,
            self.capacity * self.num_envs * self.action_dim,
        )
        element_size = torch.empty((), dtype=torch.float32).element_size()
        return sum(value * element_size for value in values)

    @property
    def diagnostics(self) -> dict[str, int | bool | str]:
        return {
            "capacity": self.capacity,
            "device": str(self.device),
            "epoch": int(self._epoch.value),
            "published_tick": int(self._published_tick.value),
            "observation_tick": int(self._observation_tick.value),
            "action_tick": int(self._action_tick.value),
            "consumed_tick": int(self._consumed_tick.value),
            "closed": bool(self._closed.value),
        }

    def publish_observation(
        self,
        *,
        tick_id: int,
        observations: torch.Tensor,
        dones: torch.Tensor,
        epoch: int | None = None,
        timeout_sec: float | None = None,
    ) -> None:
        if not self.try_publish_observation(
            tick_id=tick_id,
            observations=observations,
            dones=dones,
            epoch=epoch,
            timeout_sec=timeout_sec,
        ):
            raise TimeoutError(f"inference ring backpressure timed out at tick {tick_id}")

    def try_publish_observation(
        self,
        *,
        tick_id: int,
        observations: torch.Tensor,
        dones: torch.Tensor,
        epoch: int | None = None,
        timeout_sec: float | None = None,
    ) -> bool:
        observation_tensor, done_tensor = self._observation_inputs(observations, dones)
        with self._producer_lock:
            self._require_open()
            self._require_epoch(epoch)
            expected = int(self._published_tick.value) + 1
            self._require_tick(tick_id, expected, "publication")
            if not self._acquire(self._free_slots, timeout_sec):
                return False

            slot = self._slot_index(tick_id)
            self._request_ticks[slot] = -1
            self._request_epochs[slot] = -1
            self._transport.stage_observation(self, slot, observation_tensor, done_tensor)
            self._transport.record_observation_ready(self, slot)
            self._request_ticks[slot] = int(tick_id)
            self._request_epochs[slot] = int(self._epoch.value)
            self._published_tick.value = int(tick_id)
            self._observation_ready.release()
            return True

    def copy_observation_to(
        self,
        *,
        tick_id: int,
        observations: torch.Tensor,
        dones: torch.Tensor,
        non_blocking: bool = False,
        epoch: int | None = None,
        timeout_sec: float | None = None,
    ) -> None:
        if not self.try_copy_observation_to(
            tick_id=tick_id,
            observations=observations,
            dones=dones,
            non_blocking=non_blocking,
            epoch=epoch,
            timeout_sec=timeout_sec,
        ):
            raise TimeoutError(f"inference observation wait timed out at tick {tick_id}")

    def try_copy_observation_to(
        self,
        *,
        tick_id: int,
        observations: torch.Tensor,
        dones: torch.Tensor,
        non_blocking: bool = False,
        epoch: int | None = None,
        timeout_sec: float | None = None,
    ) -> bool:
        if not isinstance(observations, torch.Tensor) or not isinstance(dones, torch.Tensor):
            raise TypeError("Learner observation destinations must be Torch tensors")
        if tuple(observations.shape) != (self.num_envs, self.obs_dim):
            raise ValueError(
                f"Learner observation destination shape must be "
                f"{(self.num_envs, self.obs_dim)}, got {tuple(observations.shape)}"
            )
        if tuple(dones.shape) != (self.num_envs,):
            raise ValueError(
                f"Learner dones destination shape must be {(self.num_envs,)}, "
                f"got {tuple(dones.shape)}"
            )
        if observations.dtype != torch.float32 or dones.dtype != torch.float32:
            raise TypeError("Learner observation destinations must have float32 dtype")
        if observations.device != self.device or dones.device != self.device:
            raise ValueError("Learner observation destinations must live on the ring device")
        with self._observation_lock:
            self._require_open()
            self._require_epoch(epoch)
            expected = int(self._observation_tick.value) + 1
            self._require_tick(tick_id, expected, "observation consumption")
            if not self._acquire(self._observation_ready, timeout_sec):
                return False

            slot = self._slot_index(tick_id)
            self._require_header(
                int(self._request_ticks[slot]),
                int(self._request_epochs[slot]),
                int(tick_id),
                int(self._epoch.value),
                "observation",
            )
            self._transport.wait_observation_ready(self, slot)
            observations.copy_(self.observations[slot], non_blocking=non_blocking)
            dones.copy_(self.dones[slot], non_blocking=non_blocking)
            self._observation_tick.value = int(tick_id)
            return True

    def publish_action(
        self,
        *,
        tick_id: int,
        policy_version: int,
        actions: torch.Tensor,
        non_blocking: bool = False,
        epoch: int | None = None,
    ) -> None:
        if isinstance(policy_version, bool) or not isinstance(policy_version, int):
            raise TypeError(f"Inference policy version must be an integer, got {policy_version!r}")
        if policy_version < 0:
            raise ValueError("Inference policy version must be non-negative")
        action_tensor = self._action_input(actions)
        with self._response_lock:
            self._require_open()
            self._require_epoch(epoch)
            expected = int(self._observation_tick.value)
            self._require_tick(tick_id, expected, "action publication")
            slot = self._slot_index(tick_id)
            self._require_header(
                int(self._request_ticks[slot]),
                int(self._request_epochs[slot]),
                int(tick_id),
                int(self._epoch.value),
                "observation",
            )
            if int(self._response_ticks[slot]) != -1:
                raise RuntimeError(
                    f"Inference action response already published for tick {tick_id}"
                )

            if int(self._consumer_done_ticks[slot]) != -1:
                self._transport.wait_consumer_done(self, slot)
            self._response_ticks[slot] = -1
            self._policy_versions[slot] = -1
            self._transport.publish_action(
                self,
                slot,
                action_tensor,
                non_blocking=non_blocking,
            )
            self._transport.record_action_ready(self, slot)
            self._response_ticks[slot] = int(tick_id)
            self._policy_versions[slot] = int(policy_version)
            self._action_tick.value = int(tick_id)
            self._action_ready.release()

    def consume_action(
        self,
        *,
        tick_id: int,
        epoch: int | None = None,
        timeout_sec: float | None = None,
    ) -> tuple[torch.Tensor, int]:
        result = self.try_consume_action(tick_id=tick_id, epoch=epoch, timeout_sec=timeout_sec)
        if result is None:
            raise TimeoutError(f"inference action wait timed out at tick {tick_id}")
        return result

    def try_consume_action(
        self,
        *,
        tick_id: int,
        epoch: int | None = None,
        timeout_sec: float | None = None,
    ) -> tuple[torch.Tensor, int] | None:
        with self._action_lock:
            self._require_open()
            self._require_epoch(epoch)
            expected = int(self._consumed_tick.value) + 1
            self._require_tick(tick_id, expected, "action consumption")
            if not self._acquire(self._action_ready, timeout_sec):
                return None

            slot = self._slot_index(tick_id)
            actual_tick = int(self._response_ticks[slot])
            if actual_tick != int(tick_id):
                raise RuntimeError(
                    f"Inference action tick mismatch: expected {tick_id}, got {actual_tick}"
                )
            self._transport.wait_action_ready(self, slot)
            actions = self._transport.clone_action(self, slot)
            self._transport.record_consumer_done(self, slot)
            self._transport.record_consumer_done_tick(self, slot, tick_id)
            policy_version = int(self._policy_versions[slot])
            self._response_ticks[slot] = -1
            self._policy_versions[slot] = -1
            self._consumed_tick.value = int(tick_id)
            self._free_slots.release()
            return actions, policy_version

    def close(self) -> None:
        """Release process-local ring resources and mark the ring closed.

        This is fail-closed teardown, not remote cancellation. The runner remains
        responsible for stopping/joining peer processes before either endpoint
        closes the ring; a peer already blocked on a semaphore is not awakened by
        this call.
        """
        if self._local_closed:
            return
        self._closed.value = 1
        self._local_closed = True
        for name in ("observations", "dones", "actions"):
            if hasattr(self, name):
                delattr(self, name)
        self._release_local_resources()

    def _release_local_resources(self) -> None:
        """Best-effort local cleanup that never masks the original failure."""
        for name in ("observations", "dones", "actions"):
            if hasattr(self, name):
                delattr(self, name)
        self._local_observation_events.clear()
        self._local_action_events.clear()
        self._local_consumer_done_events.clear()
        self._observation_event_handles = ()
        self._action_event_handles = ()
        self._consumer_done_event_handles = ()
        for primitive in (*self._shared_semaphores, *self._shared_locks):
            close = getattr(primitive, "close", None)
            if callable(close):
                try:
                    close()
                except Exception:
                    # Teardown must preserve the constructor/runtime exception.
                    pass

    cleanup = close

    def _slot_index(self, tick_id: int) -> int:
        return int(tick_id) % self.capacity

    def _clone_action(self, slot: int) -> torch.Tensor:
        """Clone the slot's action storage (on the consumer's stream for CUDA)."""
        return self._transport.clone_action(self, slot)

    def _observation_inputs(
        self, observations: torch.Tensor, dones: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not isinstance(observations, torch.Tensor) or not isinstance(dones, torch.Tensor):
            raise TypeError("inference observations and dones must be Torch tensors")
        observation_tensor = observations.to(dtype=torch.float32)
        done_tensor = dones.to(dtype=torch.float32).reshape(-1)
        observation_tensor, done_tensor = self._transport.validate_observation_inputs(
            self, observation_tensor, done_tensor
        )
        if tuple(observation_tensor.shape) != (self.num_envs, self.obs_dim):
            raise ValueError(
                f"Inference observation shape must be {(self.num_envs, self.obs_dim)}, "
                f"got {tuple(observation_tensor.shape)}"
            )
        if tuple(done_tensor.shape) != (self.num_envs,):
            raise ValueError(
                f"Inference dones shape must be {(self.num_envs,)}, got {tuple(done_tensor.shape)}"
            )
        return observation_tensor, done_tensor

    def _action_input(self, actions: torch.Tensor) -> torch.Tensor:
        if not isinstance(actions, torch.Tensor):
            raise TypeError("inference actions must be Torch tensors")
        action_tensor = actions.to(dtype=torch.float32)
        if tuple(action_tensor.shape) != (self.num_envs, self.action_dim):
            raise ValueError(
                f"Inference action shape must be {(self.num_envs, self.action_dim)}, "
                f"got {tuple(action_tensor.shape)}"
            )
        action_tensor = self._transport.validate_action_input(self, action_tensor)
        return action_tensor

    def _acquire(self, semaphore: Any, timeout_sec: float | None) -> bool:
        if timeout_sec is None:
            semaphore.acquire()
            return True
        if isinstance(timeout_sec, bool) or not isinstance(timeout_sec, (int, float)):
            raise TypeError("inference ring timeout must be a number or None")
        if float(timeout_sec) < 0:
            raise ValueError("inference ring timeout must be non-negative")
        return bool(semaphore.acquire(timeout=float(timeout_sec)))

    def _require_open(self) -> None:
        if bool(self._closed.value):
            raise RuntimeError(f"inference ring is closed: {self.diagnostics}")

    def _require_epoch(self, epoch: int | None) -> None:
        if epoch is not None:
            if isinstance(epoch, bool) or not isinstance(epoch, int):
                raise TypeError("inference ring epoch must be an integer")
            if int(epoch) != int(self._epoch.value):
                raise RuntimeError(
                    f"Inference epoch mismatch: expected {int(self._epoch.value)}, got {epoch}"
                )

    @staticmethod
    def _require_tick(tick_id: int, expected: int, label: str) -> None:
        if isinstance(tick_id, bool) or not isinstance(tick_id, int):
            raise TypeError("inference tick IDs must be integers")
        if tick_id < 0:
            raise ValueError("inference tick IDs must be non-negative")
        if tick_id != expected:
            raise RuntimeError(
                f"Inference {label} tick mismatch: expected {expected}, got {tick_id}"
            )

    @staticmethod
    def _require_header(
        actual_tick: int, actual_epoch: int, tick_id: int, epoch: int, label: str
    ) -> None:
        if actual_tick != tick_id or actual_epoch != epoch:
            raise RuntimeError(
                f"Inference {label} header mismatch: expected tick={tick_id}, epoch={epoch}; "
                f"got tick={actual_tick}, epoch={actual_epoch}"
            )

    def _observation_event(self, slot: int) -> torch.cuda.Event:
        event = self._local_observation_events.get(slot)
        if event is None:
            if not self._observation_event_handles:
                raise RuntimeError("CUDA observation event handles are unavailable")
            event = cast(
                torch.cuda.Event,
                torch.cuda.Event.from_ipc_handle(
                    self.device, self._observation_event_handles[slot]
                ),
            )
            self._local_observation_events[slot] = event
        return event

    def _action_event(self, slot: int) -> torch.cuda.Event:
        event = self._local_action_events.get(slot)
        if event is None:
            if not self._action_event_handles:
                raise RuntimeError("CUDA action event handles are unavailable")
            event = cast(
                torch.cuda.Event,
                torch.cuda.Event.from_ipc_handle(self.device, self._action_event_handles[slot]),
            )
            self._local_action_events[slot] = event
        return event

    def _consumer_done_event(self, slot: int) -> torch.cuda.Event:
        event = self._local_consumer_done_events.get(slot)
        if event is None:
            if not self._consumer_done_event_handles:
                raise RuntimeError("CUDA consumer-done event handles are unavailable")
            event = cast(
                torch.cuda.Event,
                torch.cuda.Event.from_ipc_handle(
                    self.device, self._consumer_done_event_handles[slot]
                ),
            )
            self._local_consumer_done_events[slot] = event
        return event


__all__ = [
    "CpuRingTransport",
    "CudaRingTransport",
    "RingTransport",
    "SharedInferenceRing",
    "estimate_inference_ring_bytes",
]
