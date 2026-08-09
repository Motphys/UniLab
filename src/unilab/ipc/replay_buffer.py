"""Packed replay storage and bounded collector ingress for off-policy RL."""

import multiprocessing as mp
import time
from typing import Any, Dict, Iterable

import torch

from unilab.ipc.shared_buffer import SharedBufferBase

DEFAULT_REPLAY_INGRESS_DEPTH = 2


class ReplayBuffer(SharedBufferBase):
    """Packed replay owner for CPU-authoritative and device-authoritative modes.

    The default mode owns a full shared-memory CPU ring. Device-authoritative
    replay instead owns only fixed-depth shared ingress slots; its replay
    pipeline commits ``ptr`` and ``size`` after the device copy completes.
    """

    DEFAULT_INGRESS_DEPTH = DEFAULT_REPLAY_INGRESS_DEPTH

    def __init__(
        self,
        capacity: int,
        obs_dim: int,
        action_dim: int,
        device: str,
        defer_gpu: bool = False,
        critic_dim: int = 0,
        packed_cpu_storage: bool = False,
        ingress_slot_rows: int | None = None,
        ingress_depth: int = DEFAULT_INGRESS_DEPTH,
    ):
        super().__init__(capacity, device, defer_gpu=defer_gpu)
        del packed_cpu_storage
        self._obs_dim = obs_dim
        self._action_dim = action_dim
        self._critic_dim = critic_dim
        self.last_incremental_h2d_time_s = 0.0
        self._packed_cpu_storage = True
        self.trace_recorder: Any | None = None
        self.trace_thread_time = False
        self.trace_cuda_events = True
        self._stop_event: Any | None = None

        self.size = torch.zeros(1, dtype=torch.int64).share_memory_()
        self._init_packed_layout(obs_dim, action_dim, critic_dim)
        self._device_authoritative = ingress_slot_rows is not None
        self._ingress_slots: list[torch.Tensor] = []
        if ingress_slot_rows is not None:
            self._init_bounded_ingress(
                slot_rows=ingress_slot_rows,
                depth=int(ingress_depth),
            )
        else:
            self._storage = torch.zeros(capacity, self._storage_width).share_memory_()

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

    def _init_bounded_ingress(self, *, slot_rows: int, depth: int) -> None:
        if slot_rows <= 0:
            raise ValueError("ingress_slot_rows must be positive")
        if slot_rows > self.capacity:
            raise ValueError("ingress_slot_rows cannot exceed replay capacity")
        if depth <= 0:
            raise ValueError("ingress_depth must be positive")
        self._ingress_slot_rows = slot_rows
        self._ingress_depth = depth
        self._ingress_slots = [
            torch.empty((slot_rows, self._storage_width), dtype=torch.float32).share_memory_()
            for _ in range(depth)
        ]
        self._ingress_starts = torch.zeros(depth, dtype=torch.int64).share_memory_()
        self._ingress_counts = torch.zeros(depth, dtype=torch.int64).share_memory_()
        self._published_ptr = torch.zeros(1, dtype=torch.int64).share_memory_()
        self._ingress_publish_seq = torch.zeros(1, dtype=torch.int64).share_memory_()
        self._ingress_closed = torch.zeros(1, dtype=torch.bool).share_memory_()
        spawn_context = mp.get_context("spawn")
        self._ingress_free = spawn_context.Semaphore(depth)
        self._ingress_ready = spawn_context.Semaphore(0)
        self._ingress_consume_seq = 0
        self._ingress_release_seq = 0

    @property
    def device_authoritative(self) -> bool:
        return self._device_authoritative

    @property
    def storage_width(self) -> int:
        return self._storage_width

    @property
    def host_storage_bytes(self) -> int:
        if self._device_authoritative:
            return sum(slot.numel() * slot.element_size() for slot in self._ingress_slots)
        return int(self._storage.numel() * self._storage.element_size())

    @property
    def published_ptr(self) -> int:
        if self._device_authoritative:
            return int(self._published_ptr[0])
        return int(self.ptr[0])

    def attach_stop_event(self, stop_event: Any) -> None:
        self._stop_event = stop_event

    def take_published_ingress(self) -> tuple[int, int, int, torch.Tensor] | None:
        if not self._device_authoritative:
            raise RuntimeError("Full CPU replay does not expose bounded ingress slots")
        if not self._ingress_ready.acquire(block=False):
            return None
        slot = self._ingress_consume_seq % self._ingress_depth
        self._ingress_consume_seq += 1
        start = int(self._ingress_starts[slot])
        count = int(self._ingress_counts[slot])
        return slot, start, count, self._ingress_slots[slot][:count]

    def commit_ingress(self, *, slot: int, start: int, count: int) -> None:
        if not self._device_authoritative:
            raise RuntimeError("Full CPU replay has no bounded ingress commit")
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
        self._ingress_free.release()

    def close(self) -> None:
        if not self._device_authoritative:
            return
        self._ingress_closed[0] = True
        for _ in range(self._ingress_depth):
            self._ingress_free.release()

    def critic_graph_packed_width(self) -> int:
        """Return graph-order packed width for FastSAC critic graph inputs."""
        if self._critic_dim <= 0:
            raise RuntimeError("critic_graph_packed_source requires critic replay storage")
        return self._critic_dim + self._action_dim + 1 + self._obs_dim + self._critic_dim + 1 + 1

    def sac_graph_packed_width(self) -> int:
        """Return graph-union packed width for one-H2D FastSAC graph staging."""
        if self._critic_dim <= 0:
            raise RuntimeError("sac_graph_packed_source requires critic replay storage")
        return self._storage_width

    def pack_critic_graph_source(
        self,
        packed_batch: torch.Tensor,
        *,
        out: torch.Tensor,
    ) -> torch.Tensor:
        """Pack replay rows in the exact input order consumed by critic CUDA graph."""
        if self._critic_dim <= 0:
            raise RuntimeError("critic_graph_packed_source requires critic replay storage")
        expected_shape = (packed_batch.shape[0], self.critic_graph_packed_width())
        if tuple(out.shape) != expected_shape:
            raise ValueError(
                "critic_graph_packed_source output shape mismatch: "
                f"expected {expected_shape}, got {tuple(out.shape)}"
            )
        offset = 0
        fields = (
            packed_batch[:, self._critic_sl],
            packed_batch[:, self._act_sl],
            packed_batch[:, self._rew_col : self._rew_col + 1],
            packed_batch[:, self._nobs_sl],
            packed_batch[:, self._ncritic_sl],
            packed_batch[:, self._done_col : self._done_col + 1],
            packed_batch[:, self._trunc_col : self._trunc_col + 1],
        )
        for field in fields:
            width = int(field.shape[1])
            out[:, offset : offset + width].copy_(field)
            offset += width
        return out

    def pack_sac_graph_source(
        self,
        packed_batch: torch.Tensor,
        *,
        out: torch.Tensor,
    ) -> torch.Tensor:
        """Pack replay rows once in a layout friendly to SAC actor/critic graphs."""
        if self._critic_dim <= 0:
            raise RuntimeError("sac_graph_packed_source requires critic replay storage")
        expected_shape = (packed_batch.shape[0], self.sac_graph_packed_width())
        if tuple(out.shape) != expected_shape:
            raise ValueError(
                "sac_graph_packed_source output shape mismatch: "
                f"expected {expected_shape}, got {tuple(out.shape)}"
            )
        offset = 0
        fields = (
            packed_batch[:, self._obs_sl],
            packed_batch[:, self._critic_sl],
            packed_batch[:, self._act_sl],
            packed_batch[:, self._rew_col : self._rew_col + 1],
            packed_batch[:, self._nobs_sl],
            packed_batch[:, self._ncritic_sl],
            packed_batch[:, self._done_col : self._done_col + 1],
            packed_batch[:, self._trunc_col : self._trunc_col + 1],
        )
        for field in fields:
            width = int(field.shape[1])
            out[:, offset : offset + width].copy_(field)
            offset += width
        return out

    def __getstate__(self) -> dict:
        """Custom pickle support.

        The collector subprocess only calls ``add()``. Trace and stop-event
        handles are process-local; replay storage, ingress metadata, and
        semaphores remain shared.
        """
        state = self.__dict__.copy()
        state["trace_recorder"] = None
        state["_stop_event"] = None
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
    ):
        """Add batch (called by collector).

        `dones` follows the UniLab env lifecycle contract:
        done = terminated | truncated. Learners must pair it with
        `truncated` when computing bootstrap masks.
        """
        _trace_ns = time.perf_counter_ns() if self.trace_recorder is not None else 0
        n = obs.shape[0]
        if self._device_authoritative:
            self._add_to_ingress(
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
            )
            return
        idx = int(self.ptr[0]) % self.capacity
        has_critic = self._critic_dim > 0 and critic is not None
        if self._critic_dim > 0 and (critic is None or next_critic is None):
            raise ValueError("ReplayBuffer with critic_dim > 0 requires critic and next_critic")

        if idx + n <= self.capacity:
            target = self._storage[idx : idx + n]
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
        else:
            split = self.capacity - idx
            first = self._storage[idx:]
            second = self._storage[: n - split]
            self._write_transition_rows(
                first,
                obs[:split],
                actions[:split],
                rewards[:split],
                next_obs[:split],
                dones[:split],
                truncated[:split],
                critic[:split] if critic is not None else None,
                next_critic[:split] if next_critic is not None else None,
                has_critic=has_critic,
            )
            self._write_transition_rows(
                second,
                obs[split:],
                actions[split:],
                rewards[split:],
                next_obs[split:],
                dones[split:],
                truncated[split:],
                critic[split:] if critic is not None else None,
                next_critic[split:] if next_critic is not None else None,
                has_critic=has_critic,
            )
            self._patch_terminal_next_observations(
                first[:, self._nobs_sl],
                terminal_mask[:split] if terminal_mask is not None else None,
                terminal_next_obs[:split] if terminal_next_obs is not None else None,
                first[:, self._ncritic_sl] if has_critic else None,
                terminal_next_critic[:split] if terminal_next_critic is not None else None,
            )
            self._patch_terminal_next_observations(
                second[:, self._nobs_sl],
                terminal_mask[split:] if terminal_mask is not None else None,
                terminal_next_obs[split:] if terminal_next_obs is not None else None,
                second[:, self._ncritic_sl] if has_critic else None,
                terminal_next_critic[split:] if terminal_next_critic is not None else None,
            )

        self.ptr[0] += n
        self.size[0] = min(int(self.size[0]) + n, self.capacity)
        if self.trace_recorder is not None:
            self.trace_recorder.add_slice(
                "replay/add",
                category="replay",
                start_ns=_trace_ns,
                end_ns=time.perf_counter_ns(),
                args={"batch_size": int(n), "device": self.device},
            )

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
    ) -> None:
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
        while not self._ingress_free.acquire(timeout=0.05):
            if bool(self._ingress_closed[0]):
                return
            if self._stop_event is not None and self._stop_event.is_set():
                return
        wait_end_ns = time.perf_counter_ns()
        if bool(self._ingress_closed[0]):
            return

        sequence = int(self._ingress_publish_seq[0])
        slot = sequence % self._ingress_depth
        target = self._ingress_slots[slot][:count]
        try:
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
            start = int(self._published_ptr[0])
            self._ingress_starts[slot] = start
            self._ingress_counts[slot] = count
            self._published_ptr[0] = start + count
            self._ingress_publish_seq[0] = sequence + 1
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
        if not torch.any(terminal_mask):
            return

        target_next_obs[terminal_mask] = terminal_next_obs[terminal_mask]

        if target_next_critic is not None and terminal_next_critic is not None:
            target_next_critic[terminal_mask] = terminal_next_critic[terminal_mask]

    def sample(self, batch_size: int) -> Dict[str, torch.Tensor]:
        """Sample batch (called by learner)."""
        if self._device_authoritative:
            raise RuntimeError("Device-authoritative replay must be sampled through its pipeline")
        self.last_incremental_h2d_time_s = 0.0
        _trace_ns = time.perf_counter_ns() if self.trace_recorder is not None else 0
        size = int(self.size[0])
        _indices_ns = time.perf_counter_ns() if self.trace_recorder is not None else 0
        indices = torch.randint(0, size, (batch_size,))
        if self.trace_recorder is not None:
            self.trace_recorder.add_slice(
                "replay/sample_indices",
                category="replay",
                start_ns=_indices_ns,
                end_ns=time.perf_counter_ns(),
                args={"batch_size": int(batch_size), "size": int(size)},
            )

        chunk = self._storage[indices].to(self.device)
        batch = {
            "obs": chunk[:, self._obs_sl],
            "next_obs": chunk[:, self._nobs_sl],
            "actions": chunk[:, self._act_sl],
            "rewards": chunk[:, self._rew_col],
            "dones": chunk[:, self._done_col],
            "truncated": chunk[:, self._trunc_col],
        }
        if self._critic_dim > 0:
            batch["critic"] = chunk[:, self._critic_sl]
            batch["next_critic"] = chunk[:, self._ncritic_sl]
        if self.trace_recorder is not None:
            self.trace_recorder.add_slice(
                "replay/sample",
                category="replay",
                start_ns=_trace_ns,
                end_ns=time.perf_counter_ns(),
                args={"batch_size": int(batch_size), "device": self.device},
            )
        return batch

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

    def read_recent_fields(
        self,
        field_names: Iterable[str],
        start_ptr: int,
        count: int,
    ) -> dict[str, torch.Tensor]:
        if self._device_authoritative:
            raise RuntimeError("Device-authoritative fields are owned by the replay pipeline")
        index = start_ptr % self.capacity
        fields: dict[str, torch.Tensor] = {}
        for field_name in field_names:
            source = self.field_view(self._storage, field_name)
            if index + count <= self.capacity:
                fields[field_name] = source[index : index + count].clone()
                continue
            split = self.capacity - index
            fields[field_name] = torch.cat(
                [source[index:], source[: count - split]],
                dim=0,
            ).clone()
        return fields
