"""Packed replay storage and bounded collector ingress for off-policy RL."""

import multiprocessing as mp
import time
from typing import Any

import numpy as np
import torch

from unilab.rl.runtime.ipc.completion import (
    COMPLETION_CONTRACT_VERSION,
    CompletionTokenProtocol,
    EventCompletion,
    publication_receipt,
    record_ingress_publication,
    wait_completion,
)
from unilab.rl.runtime.settings import DEFAULT_REPLAY_INGRESS_DEPTH


class ReplayBuffer:
    """Bounded host ingress for an authoritative device replay ring.

    The collector publishes fixed-depth shared ingress slots. The replay
    pipeline advances ``ptr`` and ``size`` only after the device copy commits.
    """

    DEFAULT_INGRESS_DEPTH = DEFAULT_REPLAY_INGRESS_DEPTH

    def __init__(
        self,
        capacity: int,
        obs_dim: int,
        action_dim: int,
        device: str,
        *,
        ingress_slot_rows: int,
        critic_dim: int = 0,
        ingress_depth: int = DEFAULT_INGRESS_DEPTH,
        ingress_device: str | torch.device = "cpu",
        require_completion_receipt: bool = True,
    ):
        self.capacity = capacity
        self.device = device
        self.ptr = torch.zeros(1, dtype=torch.int64).share_memory_()
        self._obs_dim = obs_dim
        self._action_dim = action_dim
        self._critic_dim = critic_dim
        self._require_completion_receipt = bool(require_completion_receipt)
        self.last_incremental_h2d_time_s = 0.0
        self.trace_recorder: Any | None = None
        self.trace_thread_time = False
        self.trace_cuda_events = True
        self._stop_event: Any | None = None

        self.size = torch.zeros(1, dtype=torch.int64).share_memory_()
        self._init_packed_layout(obs_dim, action_dim, critic_dim)
        self._ingress_slots: list[torch.Tensor] = []
        self._init_bounded_ingress(
            slot_rows=ingress_slot_rows,
            depth=int(ingress_depth),
            ingress_device=ingress_device,
        )

    def _init_packed_layout(self, obs_dim: int, action_dim: int, critic_dim: int) -> None:
        self._storage_width = 2 * obs_dim + action_dim + 3 + 2 * critic_dim

        c = 0
        self._obs_sl = slice(c, c + obs_dim)
        c += obs_dim
        self._nobs_sl = slice(c, c + obs_dim)
        c += obs_dim
        self._act_sl = slice(c, c + action_dim)
        c += action_dim
        self._rew_col = c
        c += 1
        self._done_col = c
        c += 1
        self._trunc_col = c
        c += 1

        if critic_dim > 0:
            self._critic_sl = slice(c, c + critic_dim)
            c += critic_dim
            self._ncritic_sl = slice(c, c + critic_dim)
            c += critic_dim

        # Explicit per-column storage-class declaration consumed by the
        # device replay pipeline when it quantizes the ring to fp16. Any new
        # packed column must be registered here: undeclared columns fail
        # closed at pipeline construction instead of being silently
        # quantized (issue #2651).
        self._fp16_safe_slices: tuple[slice, ...] = (
            self._obs_sl,
            self._nobs_sl,
            self._act_sl,
        )
        if critic_dim > 0:
            self._fp16_safe_slices += (self._critic_sl, self._ncritic_sl)
        self._scalar_columns: tuple[int, ...] = (
            self._rew_col,
            self._done_col,
            self._trunc_col,
        )
        scalar_slice = slice(min(self._scalar_columns), max(self._scalar_columns) + 1)
        if scalar_slice.stop - scalar_slice.start != len(self._scalar_columns):
            raise RuntimeError(
                "Replay scalar columns (rewards/dones/truncated) must stay contiguous "
                "so the fp32-authoritative block has one packed slice"
            )
        self._scalar_sl = scalar_slice

    @property
    def obs_dim(self) -> int:
        """Packed per-row observation width."""
        return self._obs_dim

    @property
    def action_dim(self) -> int:
        """Packed per-row action width."""
        return self._action_dim

    @property
    def critic_dim(self) -> int:
        """Packed per-row critic-observation width; zero when the ring has none."""
        return self._critic_dim

    @property
    def fp16_safe_slices(self) -> tuple[slice, ...]:
        """Column slices whose values are bounded and fp16-quantizable.

        This is the layout layer's explicit declaration of which packed
        columns may be stored as fp16 by a device replay pipeline. Columns
        not covered here or by ``scalar_column_slice`` fail closed at
        pipeline construction when fp16 storage is requested.
        """
        return self._fp16_safe_slices

    @property
    def scalar_column_slice(self) -> slice:
        """Contiguous packed slice of the fp32-authoritative scalar columns.

        Rewards/dones/truncated keep full fp32 resolution in mixed-storage
        rings; the layout guarantees they form one contiguous block.
        """
        return self._scalar_sl

    def _init_bounded_ingress(
        self,
        *,
        slot_rows: int,
        depth: int,
        ingress_device: str | torch.device = "cpu",
    ) -> None:
        if slot_rows <= 0:
            raise ValueError("ingress_slot_rows must be positive")
        if slot_rows > self.capacity:
            raise ValueError("ingress_slot_rows cannot exceed replay capacity")
        if depth <= 0:
            raise ValueError("ingress_depth must be positive")
        self._ingress_slot_rows = slot_rows
        self._ingress_depth = depth
        self._ingress_device = torch.device(ingress_device)
        if self._ingress_device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(
                f"CUDA replay ingress requested on unavailable {self._ingress_device}"
            )
        self._ingress_slots = [
            torch.empty(
                (slot_rows, self._storage_width),
                dtype=torch.float32,
                device=self._ingress_device,
            ).share_memory_()
            for _ in range(depth)
        ]
        self._ingress_starts = torch.zeros(depth, dtype=torch.int64).share_memory_()
        self._ingress_counts = torch.zeros(depth, dtype=torch.int64).share_memory_()
        self._published_ptr = torch.zeros(1, dtype=torch.int64).share_memory_()
        self._ingress_publish_seq = torch.zeros(1, dtype=torch.int64).share_memory_()
        self._ingress_release_seq_tensor = torch.zeros(1, dtype=torch.int64).share_memory_()
        self._ingress_high_water = torch.zeros(1, dtype=torch.int64).share_memory_()
        self._ingress_backpressure_waits = torch.zeros(1, dtype=torch.int64).share_memory_()
        self._ingress_backpressure_wait_ns = torch.zeros(1, dtype=torch.int64).share_memory_()
        self._ingress_early_returns = torch.zeros(1, dtype=torch.int64).share_memory_()
        self._ingress_dropped_batches = torch.zeros(1, dtype=torch.int64).share_memory_()
        self._ingress_closed_returns = torch.zeros(1, dtype=torch.int64).share_memory_()
        self._ingress_stop_returns = torch.zeros(1, dtype=torch.int64).share_memory_()
        self._ingress_closed = torch.zeros(1, dtype=torch.bool).share_memory_()
        spawn_context = mp.get_context("spawn")
        self._ingress_free = spawn_context.Semaphore(depth)
        self._ingress_ready = spawn_context.Semaphore(0)
        self._ingress_consume_seq = 0
        self._ingress_release_seq = 0
        self._completion_sequence = 0
        self._completion_producer_id: str | None = None
        self._last_completion: CompletionTokenProtocol | None = None
        self._completion_metadata_version = torch.zeros(
            1, dtype=torch.int64, device="cpu"
        ).share_memory_()
        self._completion_metadata_contract = torch.zeros(
            1, dtype=torch.int64, device="cpu"
        ).share_memory_()
        self._completion_metadata_sequence = torch.zeros(
            1, dtype=torch.int64, device="cpu"
        ).share_memory_()
        self._completion_metadata_available = torch.zeros(
            1, dtype=torch.bool, device="cpu"
        ).share_memory_()
        self._completion_metadata_handle = torch.zeros(
            (1, 256), dtype=torch.uint8, device="cpu"
        ).share_memory_()
        self._completion_metadata_handle_size = torch.zeros(
            1, dtype=torch.int64, device="cpu"
        ).share_memory_()
        self._completion_metadata_producer = torch.zeros(
            (1, 256), dtype=torch.uint8, device="cpu"
        ).share_memory_()
        self._completion_metadata_producer_size = torch.zeros(
            1, dtype=torch.int64, device="cpu"
        ).share_memory_()
        self._completion_metadata_local_version = 0

    def _ingress_read_stream(self) -> Any | None:
        """Return this process's current ordered CUDA ingress stream."""
        return (
            torch.cuda.current_stream(self._ingress_device)
            if self._ingress_device.type == "cuda"
            else None
        )

    @property
    def storage_width(self) -> int:
        return self._storage_width

    @property
    def ingress_slot_rows(self) -> int:
        return self._ingress_slot_rows

    @property
    def host_storage_bytes(self) -> int:
        if self._ingress_device.type == "cuda":
            return 0
        return sum(slot.numel() * slot.element_size() for slot in self._ingress_slots)

    @property
    def ingress_device(self) -> torch.device:
        return self._ingress_device

    @property
    def published_ptr(self) -> int:
        return int(self._published_ptr[0])

    def _record_ingress_backpressure(self, *, start_ns: int) -> None:
        waited_ns = time.perf_counter_ns() - start_ns
        self._ingress_backpressure_waits[0] += 1
        self._ingress_backpressure_wait_ns[0] += waited_ns

    def _record_ingress_drop(
        self,
        *,
        reason: str,
        wait_start_ns: int,
        waited: bool,
    ) -> None:
        if waited:
            self._record_ingress_backpressure(start_ns=wait_start_ns)
        self._ingress_early_returns[0] += 1
        self._ingress_dropped_batches[0] += 1
        if reason == "closed":
            self._ingress_closed_returns[0] += 1
        elif reason == "stop":
            self._ingress_stop_returns[0] += 1

    def ingress_diagnostics(self) -> dict[str, int | float]:
        """Snapshot bounded-ingress state without touching ingress tensors.

        Every value comes from process-shared host metadata. In particular, a
        CUDA-resident ingress slot is never read, copied to host, or synchronized
        to produce this snapshot.
        Sequence and drop counters describe ingress publication chunks. A
        collector vector larger than ``ingress_slot_rows`` is split into
        multiple independently publishable chunks by ``add_batch()``.
        """
        published_sequence = int(self._ingress_publish_seq[0])
        release_sequence = int(self._ingress_release_seq_tensor[0])
        occupancy = max(0, published_sequence - release_sequence)
        return {
            "ingress_depth": self._ingress_depth,
            "ingress_slot_rows": self._ingress_slot_rows,
            "published_sequence": published_sequence,
            "release_sequence": release_sequence,
            "occupancy": occupancy,
            "high_water_occupancy": int(self._ingress_high_water[0]),
            "backpressure_waits": int(self._ingress_backpressure_waits[0]),
            "backpressure_wait_s": (int(self._ingress_backpressure_wait_ns[0]) / 1_000_000_000),
            "early_returns": int(self._ingress_early_returns[0]),
            "dropped_batches": int(self._ingress_dropped_batches[0]),
            "closed_returns": int(self._ingress_closed_returns[0]),
            "stop_returns": int(self._ingress_stop_returns[0]),
        }

    def attach_stop_event(self, stop_event: Any) -> None:
        self._stop_event = stop_event

    def take_published_ingress(self) -> tuple[int, int, int, torch.Tensor] | None:
        if not self._ingress_ready.acquire(block=False):
            return None
        slot = self._ingress_consume_seq % self._ingress_depth
        self._ingress_consume_seq += 1
        start = int(self._ingress_starts[slot])
        count = int(self._ingress_counts[slot])
        return slot, start, count, self._ingress_slots[slot][:count]

    @property
    def last_completion(self) -> CompletionTokenProtocol | None:
        """Return the replay-owned receipt guarding the newest published ingress slot."""
        return self._last_completion

    @property
    def completion_sequence(self) -> int:
        """Return the latest completion sequence accepted by this buffer."""
        return self._completion_sequence

    @property
    def completion_producer_id(self) -> str | None:
        """Return the producer identity accepted by this ingress producer."""
        return self._completion_producer_id

    def _publish_completion_metadata(self, completion: Any) -> None:
        """Publish the replay-owned ingress publication event for the learner copy.

        Only ``EventCompletion`` receipts recorded by ``record_ingress_publication``
        after the ingress writes carry an interprocess event. Host-path receipts
        have no device event to export; host slot ordering is carried by the
        ingress semaphores. Driver and IPC export failures must fail closed
        instead of becoming a missing receipt.
        """
        if not isinstance(completion, EventCompletion):
            return
        handle = bytes(completion.event.ipc_handle())
        producer = completion.producer_id.encode("utf-8")
        if (
            len(handle) > self._completion_metadata_handle.shape[1]
            or len(producer) > (self._completion_metadata_producer.shape[1])
        ):
            raise ValueError("completion IPC metadata exceeds ingress limits")
        self._completion_metadata_contract[0] = int(completion.contract_version)
        self._completion_metadata_sequence[0] = int(completion.sequence)
        self._completion_metadata_handle_size[0] = len(handle)
        self._completion_metadata_producer_size[0] = len(producer)
        self._completion_metadata_handle.zero_()
        self._completion_metadata_handle[0, : len(handle)] = torch.from_numpy(
            np.frombuffer(handle, dtype=np.uint8).copy()
        )
        self._completion_metadata_producer.zero_()
        self._completion_metadata_producer[0, : len(producer)] = torch.from_numpy(
            np.frombuffer(producer, dtype=np.uint8).copy()
        )
        self._completion_metadata_available[0] = True
        self._completion_metadata_version[0] += 1

    def _import_ingress_publication(self) -> CompletionTokenProtocol | None:
        """Import the newest replay-owned ingress publication event on the learner copy."""
        if not bool(self._completion_metadata_available[0]):
            return None
        handle_size = int(self._completion_metadata_handle_size[0])
        producer_size = int(self._completion_metadata_producer_size[0])
        if not 0 < handle_size <= self._completion_metadata_handle.shape[1]:
            raise TypeError("ingress publication event handle must be non-empty bounded bytes")
        handle_bytes = bytes(self._completion_metadata_handle[0, :handle_size].numpy().tobytes())
        producer_id = bytes(
            self._completion_metadata_producer[0, :producer_size].numpy().tobytes()
        ).decode("utf-8")
        contract_version = int(self._completion_metadata_contract[0])
        sequence = int(self._completion_metadata_sequence[0])
        if contract_version != COMPLETION_CONTRACT_VERSION:
            raise ValueError(f"unsupported completion contract {contract_version}")
        if not producer_id.strip():
            raise TypeError("completion producer_id must be a non-empty string")
        event = torch.cuda.Event.from_ipc_handle(self._ingress_device, handle_bytes)
        imported = EventCompletion(
            contract_version=contract_version,
            operation="step",
            sequence=sequence,
            producer_id=producer_id,
            event=event,
        )
        self._completion_sequence = imported.sequence
        self._completion_producer_id = imported.producer_id
        self._last_completion = imported
        return imported

    def import_ingress_publication(self) -> CompletionTokenProtocol | None:
        """Import the replay-owned publication receipt once per new shared version."""
        version = int(self._completion_metadata_version[0])
        if version == self._completion_metadata_local_version:
            return self._last_completion
        imported = self._import_ingress_publication()
        self._completion_metadata_local_version = version
        return imported

    def commit_ingress(self, *, slot: int, start: int, count: int) -> None:
        expected_slot = self._ingress_release_seq % self._ingress_depth
        if slot != expected_slot:
            raise RuntimeError(
                f"Ingress slots must commit in publication order: expected {expected_slot}, got {slot}"
            )
        if start != int(self.ptr[0]):
            raise RuntimeError(
                f"Ingress commit is not contiguous: committed ptr {int(self.ptr[0])}, start {start}"
            )
        self.ptr[0] = start + count
        self.size[0] = min(start + count, self.capacity)
        self._ingress_release_seq += 1
        self._ingress_release_seq_tensor[0] = self._ingress_release_seq
        self._ingress_free.release()

    def close(self) -> None:
        if bool(self._ingress_closed[0]):
            return
        self._ingress_closed[0] = True
        for _ in range(self._ingress_depth):
            self._ingress_free.release()
        self._ingress_slots.clear()

    def release_ipc(self) -> None:
        """Release this process's ingress IPC views without closing the buffer."""
        self._ingress_slots.clear()

    def __getstate__(self) -> dict:
        """Custom pickle support.

        The collector subprocess only publishes transitions through
        ``add_batch()``. Trace and stop-event handles are process-local; replay
        storage, ingress metadata, and semaphores remain shared.
        """
        state = self.__dict__.copy()
        state["trace_recorder"] = None
        state["_stop_event"] = None
        state["_last_completion"] = None
        return state

    def add(
        self,
        obs,
        actions,
        rewards,
        next_obs,
        dones,
        truncated,
        terminal_mask=None,
        terminal_next_obs=None,
        critic=None,
        next_critic=None,
        terminal_next_critic=None,
        completion: CompletionTokenProtocol | None = None,
    ) -> bool:
        """Add batch (called by collector).

        `dones` follows the UniLab env lifecycle contract:
        done = terminated | truncated. `truncated` is not the raw env
        truncation flag: the collector encodes the timeout-bootstrap
        eligibility mask (raw_truncated & ~terminated) before writing, so
        learners compute ``bootstrap = clamp(1 - dones + truncated, 0, 1)``
        and never bootstrap a terminated row — including rows where
        termination and truncation coincide. Timeout-eligible rows bootstrap
        from the terminal observations patched into the next-observation
        columns by ``terminal_mask``/``terminal_next_obs``.
        """
        _trace_ns = time.perf_counter_ns() if self.trace_recorder is not None else 0
        return self._add_to_ingress(
            obs,
            actions,
            rewards,
            next_obs,
            dones,
            truncated,
            terminal_mask,
            terminal_next_obs,
            critic,
            next_critic,
            terminal_next_critic,
            trace_start_ns=_trace_ns,
            completion=completion,
        )

    def add_batch(
        self,
        obs,
        actions,
        rewards,
        next_obs,
        dones,
        truncated,
        terminal_mask=None,
        terminal_next_obs=None,
        critic=None,
        next_critic=None,
        terminal_next_critic=None,
        completion: CompletionTokenProtocol | None = None,
    ) -> bool:
        """Publish a collector vector in bounded ingress-sized chunks.

        Each chunk is an independently publishable transition batch. Shutdown can
        retain a valid prefix of a vector, but never an invalid transition row.
        """
        count = int(obs.shape[0])
        if count <= self._ingress_slot_rows:
            return self.add(
                obs,
                actions,
                rewards,
                next_obs,
                dones,
                truncated,
                terminal_mask=terminal_mask,
                terminal_next_obs=terminal_next_obs,
                critic=critic,
                next_critic=next_critic,
                terminal_next_critic=terminal_next_critic,
                completion=completion,
            )

        def optional_slice(
            tensor: torch.Tensor | None, start: int, end: int
        ) -> torch.Tensor | None:
            return None if tensor is None else tensor[start:end]

        for start in range(0, count, self._ingress_slot_rows):
            end = min(start + self._ingress_slot_rows, count)
            published = self.add(
                obs[start:end],
                actions[start:end],
                rewards[start:end],
                next_obs[start:end],
                dones[start:end],
                truncated[start:end],
                terminal_mask=optional_slice(terminal_mask, start, end),
                terminal_next_obs=optional_slice(terminal_next_obs, start, end),
                critic=optional_slice(critic, start, end),
                next_critic=optional_slice(next_critic, start, end),
                terminal_next_critic=optional_slice(terminal_next_critic, start, end),
                completion=completion,
            )
            if not published:
                return False
        return True

    def _add_to_ingress(
        self,
        obs,
        actions,
        rewards,
        next_obs,
        dones,
        truncated,
        terminal_mask,
        terminal_next_obs,
        critic,
        next_critic,
        terminal_next_critic,
        *,
        trace_start_ns: int,
        completion: CompletionTokenProtocol | None = None,
    ) -> bool:
        count = int(obs.shape[0])
        if count > self._ingress_slot_rows:
            raise ValueError(
                f"Transition batch has {count} rows but bounded ingress slots hold "
                f"{self._ingress_slot_rows}"
            )
        has_critic = self._critic_dim > 0 and critic is not None
        if self._critic_dim > 0 and (critic is None or next_critic is None):
            raise ValueError("ReplayBuffer with critic_dim > 0 requires critic and next_critic")

        wait_start_ns = time.perf_counter_ns()
        waited_for_free_slot = False
        while not self._ingress_free.acquire(timeout=0.05):
            waited_for_free_slot = True
            if bool(self._ingress_closed[0]):
                self._record_ingress_drop(
                    reason="closed",
                    wait_start_ns=wait_start_ns,
                    waited=waited_for_free_slot,
                )
                return False
            if self._stop_event is not None and self._stop_event.is_set():
                self._record_ingress_drop(
                    reason="stop",
                    wait_start_ns=wait_start_ns,
                    waited=waited_for_free_slot,
                )
                return False
        wait_end_ns = time.perf_counter_ns()
        if bool(self._ingress_closed[0]):
            self._record_ingress_drop(
                reason="closed",
                wait_start_ns=wait_start_ns,
                waited=waited_for_free_slot,
            )
            return False
        if waited_for_free_slot:
            self._record_ingress_backpressure(start_ns=wait_start_ns)

        sequence = int(self._ingress_publish_seq[0])
        slot = sequence % self._ingress_depth
        target = self._ingress_slots[slot][:count]
        producer_sequence = int(completion.sequence) if completion is not None else 0
        try:
            # Gate the ingress stream before its first source-tensor read or
            # copy. Enqueueing a wait after earlier copies cannot order them.
            completion = wait_completion(
                completion,
                stream=self._ingress_read_stream(),
                operation=completion.operation if completion is not None else "step",
                producer_id=self._completion_producer_id,
                last_sequence=self._completion_sequence,
            )
            self._write_transition_rows(
                target,
                obs,
                actions,
                rewards,
                next_obs,
                dones,
                truncated,
                critic,
                next_critic,
                has_critic=has_critic,
            )
            self._patch_terminal_next_observations(
                target[:, self._nobs_sl],
                terminal_mask,
                terminal_next_obs,
                target[:, self._ncritic_sl] if has_critic else None,
                terminal_next_critic,
            )
            if completion is not None:
                completion = record_ingress_publication(
                    completion,
                    stream=self._ingress_read_stream(),
                )
                completion = publication_receipt(
                    completion,
                    sequence=producer_sequence,
                )
                self._completion_sequence = completion.sequence
                self._completion_producer_id = completion.producer_id
                self._last_completion = completion
                self._publish_completion_metadata(completion)
            if self._ingress_device.type == "cuda":
                # A CUDA slot is unreadable without a negotiated completion
                # receipt unless the producer explicitly declared synchronous
                # no-receipt steps (``unilab.rl`` env contract
                # ``produces_completion_receipts = False``). Fail closed
                # instead of synchronizing the producer stream or publishing
                # a race against the learner's copy.
                if completion is None and self._require_completion_receipt:
                    raise RuntimeError(
                        "CUDA replay ingress publication requires a completion receipt"
                    )
            start = int(self._published_ptr[0])
            self._ingress_starts[slot] = start
            self._ingress_counts[slot] = count
            self._published_ptr[0] = start + count
            self._ingress_publish_seq[0] = sequence + 1
            published_sequence = sequence + 1
            released_sequence = int(self._ingress_release_seq_tensor[0])
            occupancy = max(0, published_sequence - released_sequence)
            if occupancy > int(self._ingress_high_water[0]):
                self._ingress_high_water[0] = occupancy
            self._ingress_ready.release()
        except BaseException:
            self._ingress_free.release()
            raise

        if self.trace_recorder is not None:
            end_ns = time.perf_counter_ns()
            self.trace_recorder.add_slice(
                "replay/ingress_backpressure",
                category="replay",
                start_ns=wait_start_ns,
                end_ns=wait_end_ns,
                args={"batch_size": count, "slot": slot, "depth": self._ingress_depth},
            )
            self.trace_recorder.add_slice(
                "replay/add",
                category="replay",
                start_ns=trace_start_ns,
                end_ns=end_ns,
                args={
                    "batch_size": count,
                    "device": self.device,
                    "ingress_slot": slot,
                    "published_ptr": start + count,
                },
            )
        return True

    def _write_transition_rows(
        self,
        target,
        obs,
        actions,
        rewards,
        next_obs,
        dones,
        truncated,
        critic,
        next_critic,
        *,
        has_critic: bool,
    ) -> None:
        target[:, self._obs_sl] = obs
        target[:, self._nobs_sl] = next_obs
        target[:, self._act_sl] = actions
        target[:, self._rew_col] = rewards
        target[:, self._done_col] = dones
        target[:, self._trunc_col] = truncated
        if has_critic:
            assert critic is not None
            assert next_critic is not None
            target[:, self._critic_sl] = critic
            target[:, self._ncritic_sl] = next_critic

    @staticmethod
    def _patch_terminal_next_observations(
        target_next_obs,
        terminal_mask,
        terminal_next_obs,
        target_next_critic=None,
        terminal_next_critic=None,
    ) -> None:
        if terminal_mask is None or terminal_next_obs is None:
            return
        if terminal_mask.ndim != 1 or terminal_mask.shape[0] != target_next_obs.shape[0]:
            return
        expanded = terminal_mask[:, None]
        target_next_obs.copy_(torch.where(expanded, terminal_next_obs, target_next_obs))
        if target_next_critic is not None and terminal_next_critic is not None:
            target_next_critic.copy_(
                torch.where(expanded, terminal_next_critic, target_next_critic)
            )

    def field_view(self, packed: torch.Tensor, field_name: str) -> torch.Tensor:
        field = {
            "obs": self._obs_sl,
            "next_obs": self._nobs_sl,
            "actions": self._act_sl,
            "rewards": self._rew_col,
            "dones": self._done_col,
            "truncated": self._trunc_col,
            "critic": getattr(self, "_critic_sl", None),
            "next_critic": getattr(self, "_ncritic_sl", None),
        }.get(field_name)
        if field is None:
            raise KeyError(f"Replay field {field_name!r} is unavailable")
        return packed[:, field]
