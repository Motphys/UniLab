"""Synchronous gradient averaging across data-parallel ranks.

Multi-GPU data-parallel off-policy training runs one independent
learner+collector pair per rank (see ``launcher.py``). The runner broadcasts
rank 0's model state before collection starts. During steady-state training,
each learner averages the actor/critic/temperature gradients immediately after
``backward()`` and before clipping or ``optimizer.step()``.

This module owns the process-group lifecycle, the startup parameter broadcast,
flat-gradient collectives, and the small scalar reduction used by the canonical
rank-0 logger. Optimizer state is not communicated: identical initialization
and identical averaged gradients keep each rank's optimizer state aligned.
"""

from __future__ import annotations

import datetime
import json
import time
from collections.abc import Iterable
from typing import cast

import torch
import torch.distributed as dist


class DpParameterSync:
    """Process group for startup state broadcast and steady-state gradients.

    The startup synchronization key order is frozen on the first collective
    (``sorted(keys)``) so every rank walks the identical sequence without
    re-sorting per call; a later call with a different key set is a bug and
    raises ValueError.
    """

    def __init__(
        self,
        *,
        world_size: int,
        rank: int,
        rendezvous_path: str,
        backend: str = "nccl",
        device: str | None = None,
        timeout_s: int = 120,
        gradient_wire_dtype: str = "float32",
    ) -> None:
        if int(world_size) < 2:
            raise ValueError(f"DpParameterSync requires world_size >= 2, got {world_size}")
        if not (0 <= int(rank) < int(world_size)):
            raise ValueError(f"rank {rank} is out of range for world_size={world_size}")
        self.world_size = int(world_size)
        self.rank = int(rank)
        self.rendezvous_path = str(rendezvous_path)
        self.backend = str(backend)
        self.device = None if device is None else torch.device(device)
        self.timeout_s = int(timeout_s)
        if gradient_wire_dtype not in {"float32", "bfloat16"}:
            raise ValueError("gradient_wire_dtype must be 'float32' or 'bfloat16'")
        self.gradient_wire_dtype = gradient_wire_dtype
        self._key_order: tuple[str, ...] | None = None
        self._statistics_schema: dict[str, str] = {}
        self._gradient_sync_time_sec = 0.0
        self._gradient_sync_calls = 0
        self._capturing_gradient_graph = False
        self._gradient_graph_events: list[tuple[torch.cuda.Event, torch.cuda.Event]] = []
        self._gradient_graph_calls = 0
        self._gradient_graph_timing_enabled = True
        self._gradient_graph_pending = False
        self._gradient_buffers: dict[tuple[int, ...], torch.Tensor] = {}
        self._started = False

    def start(self) -> None:
        """Init the process group over a shared-file rendezvous.

        The rendezvous path must be unique per run (the caller derives it
        from the per-run log directory), so no stale FileStore state from a
        previous run can leak in and no pre-clean is needed.

        For NCCL the current CUDA device is pinned to this rank's device
        first: ProcessGroupNCCL binds communicators to the current device,
        and without the pin every rank defaults to cuda:0, which hangs the
        first collective on any rank whose learner lives on another GPU.

        ``NCCL_P2P_DISABLE``/``NCCL_SHM_DISABLE`` default to 1 (env override
        wins): on hosts with broken NCCL peer transport (e.g. RTX 6000D on
        current drivers, where P2P hangs and SHM triggers CUDA illegal
        memory access) the TCP loopback transport is the only reliable
        path.
        """
        if self._started:
            return
        if self.backend == "nccl":
            import os

            os.environ.setdefault("NCCL_P2P_DISABLE", "1")
            os.environ.setdefault("NCCL_SHM_DISABLE", "1")
        if self.backend == "nccl" and self.device is not None and self.device.type == "cuda":
            torch.cuda.set_device(self.device)
        dist.init_process_group(
            backend=self.backend,
            init_method=f"file://{self.rendezvous_path}",
            rank=self.rank,
            world_size=self.world_size,
            timeout=datetime.timedelta(seconds=self.timeout_s),
        )
        self._started = True

    def broadcast_from_rank0(self, tensors: dict[str, torch.Tensor]) -> None:
        """Overwrite every rank's tensors with rank 0's values, in place."""
        with torch.no_grad():
            for key in self._ordered_keys(tensors):
                dist.broadcast(tensors[key], src=0)

    def allreduce_gradients(self, parameters: Iterable[torch.Tensor]) -> None:
        """Average one optimizer's gradients with one flat all-reduce.

        Existing gradients are packed in parameter order. All ranks must
        execute the same optimizer updates and therefore present the same
        gradient layout.
        """
        sync_start = time.perf_counter()
        timing_events = None
        capturing = self._capturing_gradient_graph and torch.cuda.is_current_stream_capturing()
        if capturing:
            self._gradient_graph_calls += 1
        if capturing and self._gradient_graph_timing_enabled:
            timing_events = (
                torch.cuda.Event(enable_timing=True, external=True),
                torch.cuda.Event(enable_timing=True, external=True),
            )
            timing_events[0].record()
        params = [
            parameter
            for parameter in parameters
            if parameter.requires_grad and parameter.grad is not None
        ]
        if not params:
            raise ValueError("gradient synchronization requires at least one existing gradient")

        device = params[0].device
        dtype = params[0].dtype
        for parameter in params:
            if parameter.device != device or parameter.dtype != dtype:
                raise ValueError(
                    "one flat gradient collective requires matching parameter devices and dtypes"
                )

        gradient_numel = sum(parameter.numel() for parameter in params)
        # Compression is opt-in and only affects sufficiently large FP32 groups.
        # Small groups (including temperature) keep their original precision.
        wire_dtype = (
            torch.bfloat16
            if self.gradient_wire_dtype == "bfloat16"
            and dtype == torch.float32
            and gradient_numel >= 1024
            else dtype
        )
        buffer_key = tuple(id(parameter) for parameter in params)
        packed = self._gradient_buffers.get(buffer_key)
        if (
            packed is None
            or packed.numel() != gradient_numel
            or packed.device != device
            or packed.dtype != wire_dtype
        ):
            packed = torch.empty(gradient_numel, device=device, dtype=wire_dtype)
            self._gradient_buffers[buffer_key] = packed
        offset = 0
        packed_views: list[torch.Tensor] = []
        gradients: list[torch.Tensor] = []
        for parameter in params:
            width = parameter.numel()
            gradient = parameter.grad
            assert gradient is not None
            packed_views.append(packed[offset : offset + width].view_as(parameter))
            gradients.append(gradient.detach())
            offset += width
        if wire_dtype != dtype:
            torch._foreach_copy_(packed_views, gradients)
        else:
            for packed_view, gradient in zip(packed_views, gradients):
                packed_view.copy_(gradient)
        dist.all_reduce(packed, op=dist.ReduceOp.SUM)
        if wire_dtype != dtype:
            # Divide after restoring FP32 to avoid another BF16 rounding.
            torch._foreach_copy_(gradients, packed_views)
            torch._foreach_mul_(gradients, 1.0 / self.world_size)
        else:
            packed.div_(self.world_size)
            for gradient, packed_view in zip(gradients, packed_views):
                gradient.copy_(packed_view)

        if timing_events is not None:
            timing_events[1].record()
            self._gradient_graph_events.append(timing_events)
        elif not capturing:
            self._gradient_sync_time_sec += time.perf_counter() - sync_start
            self._gradient_sync_calls += 1

    def begin_gradient_graph_capture(self, *, enable_timing: bool = True) -> None:
        """Start a new CUDA graph's synchronization instrumentation.

        Call outside CUDA capture, around warmup and capture. Warmup metrics
        are discarded; only operations inside CUDA capture create events.
        The owner must release the previous graph before replacing its events.
        """
        self._consume_gradient_graph_timing()
        self._gradient_sync_time_sec = 0.0
        self._gradient_sync_calls = 0
        self._gradient_graph_events.clear()
        self._gradient_graph_calls = 0
        self._gradient_graph_timing_enabled = enable_timing
        self._capturing_gradient_graph = True

    def end_gradient_graph_capture(self) -> None:
        """Finish instrumentation, including when capture raises."""
        self._capturing_gradient_graph = False
        self._gradient_sync_time_sec = 0.0
        self._gradient_sync_calls = 0

    def record_gradient_graph_replay(self) -> None:
        """Account for one replay; call immediately BEFORE graph.replay().

        Consume an uncollected previous replay before its events are overwritten.
        Normally the runner already consumed it at its iteration metrics boundary.
        GPU time includes packing, collective dependencies, averaging and unpacking;
        it is not an isolated wire-transfer measurement.
        """
        self._consume_gradient_graph_timing()
        self._gradient_sync_calls += self._gradient_graph_calls
        self._gradient_graph_pending = bool(self._gradient_graph_events)

    def _consume_gradient_graph_timing(self) -> None:
        if not self._gradient_graph_pending:
            return
        for start, end in self._gradient_graph_events:
            end.synchronize()
            self._gradient_sync_time_sec += start.elapsed_time(end) / 1000.0
        self._gradient_graph_pending = False

    def take_gradient_sync_metrics(self) -> tuple[float, int]:
        """Return and reset this rank's gradient-sync time and call count."""
        self._consume_gradient_graph_timing()
        metrics = (self._gradient_sync_time_sec, self._gradient_sync_calls)
        self._gradient_sync_time_sec = 0.0
        self._gradient_sync_calls = 0
        return metrics

    def allreduce_statistics(
        self,
        *,
        mean: dict[str, float] | None = None,
        total: dict[str, float] | None = None,
    ) -> dict[str, float]:
        """Aggregate sparse scalar statistics on every rank.

        ``mean`` fields are averaged over ranks that supplied the field;
        ``total`` fields are summed. A cached union schema plus a presence mask
        allows optional collector/reward fields to appear at different times on
        different ranks without treating a missing value as zero. Schema
        exchange happens only when a new field appears; the steady-state path
        is two small collectives (schema-change flag + packed scalar reduce).

        This collective is intentionally separate from per-optimizer gradient
        averaging: logging is aggregated once per outer learner iteration.
        """
        mean = dict(mean or {})
        total = dict(total or {})
        overlap = set(mean) & set(total)
        if overlap:
            raise ValueError(
                f"DP statistic fields cannot be both mean and total: {sorted(overlap)}"
            )

        local_schema = {key: "mean" for key in mean}
        local_schema.update({key: "total" for key in total})
        schema_changed = any(
            self._statistics_schema.get(key) != reduction for key, reduction in local_schema.items()
        )
        device = self.device if self.device is not None else torch.device("cpu")
        changed_tensor = torch.tensor(
            [int(schema_changed)],
            dtype=torch.int32,
            device=device,
        )
        dist.all_reduce(changed_tensor, op=dist.ReduceOp.MAX)
        if bool(changed_tensor.item()):
            merged = dict(self._statistics_schema)
            for rank_schema in self._gather_statistics_schemas(local_schema, device=device):
                for key, reduction in rank_schema:
                    existing = merged.get(key)
                    if existing is not None and existing != reduction:
                        raise ValueError(
                            f"DP statistic {key!r} changed reduction from "
                            f"{existing!r} to {reduction!r}"
                        )
                    merged[key] = reduction
            self._statistics_schema = merged

        if not self._statistics_schema:
            return {}

        keys = tuple(sorted(self._statistics_schema))
        values = mean | total
        # Materialize the entire payload on the host before one device transfer.
        # Per-scalar CUDA assignments launch a copy for every value and mask.
        packed = torch.tensor(
            [
                [float(values.get(key, 0.0)) for key in keys],
                [float(key in values) for key in keys],
            ],
            dtype=torch.float64,
            device=device,
        )
        dist.all_reduce(packed, op=dist.ReduceOp.SUM)

        value_sums = packed[0].tolist()
        presence_counts = packed[1].tolist()
        aggregated: dict[str, float] = {}
        for index, key in enumerate(keys):
            count = float(presence_counts[index])
            if count <= 0:
                continue
            value = float(value_sums[index])
            if self._statistics_schema[key] == "mean":
                value /= count
            aggregated[key] = value
        return aggregated

    def _gather_statistics_schemas(
        self,
        local_schema: dict[str, str],
        *,
        device: torch.device,
    ) -> list[tuple[tuple[str, str], ...]]:
        """Exchange sparse logger schemas without NCCL object collectives.

        ``all_gather_object`` stages pickle payloads through an NCCL all-gather.
        That path is both unnecessary for this string-only schema and has
        caused CUDA out-of-range-address faults on supported multi-GPU hosts.
        Rank-ordered byte broadcasts keep the exchange deterministic and use
        the same well-tested NCCL primitive as the startup model sync.
        """
        payload = json.dumps(
            sorted(local_schema.items()),
            ensure_ascii=True,
            separators=(",", ":"),
        ).encode("utf-8")
        local_payload = torch.tensor(list(payload), dtype=torch.uint8, device=device)
        gathered: list[tuple[tuple[str, str], ...]] = []

        for source_rank in range(self.world_size):
            payload_size = torch.tensor(
                [len(payload) if self.rank == source_rank else 0],
                dtype=torch.int64,
                device=device,
            )
            dist.broadcast(payload_size, src=source_rank)
            size = int(payload_size.item())
            rank_payload = torch.empty(size, dtype=torch.uint8, device=device)
            if self.rank == source_rank:
                rank_payload.copy_(local_payload)
            dist.broadcast(rank_payload, src=source_rank)

            decoded = json.loads(bytes(rank_payload.cpu().tolist()).decode("utf-8"))
            gathered.append(
                tuple((cast(str, key), cast(str, reduction)) for key, reduction in decoded)
            )
        return gathered

    def close(self) -> None:
        """Destroy the process group; idempotent and safe before start()."""
        if not self._started:
            return
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()
        self._gradient_buffers.clear()
        self._gradient_graph_events.clear()
        self._gradient_graph_pending = False
        self._gradient_graph_calls = 0
        self._started = False

    def _ordered_keys(self, tensors: dict[str, torch.Tensor]) -> tuple[str, ...]:
        if self._key_order is None:
            self._key_order = tuple(sorted(tensors))
            return self._key_order
        if set(tensors) != set(self._key_order):
            raise ValueError(
                "DpParameterSync tensor keys changed between collectives: "
                f"expected {sorted(self._key_order)}, got {sorted(tensors)}"
            )
        return self._key_order
