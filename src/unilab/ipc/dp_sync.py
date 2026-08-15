"""Synchronous gradient averaging across data-parallel ranks.

Multi-GPU data-parallel off-policy training runs one independent
learner+collector pair per rank (see ``dp_launcher.py``). The runner broadcasts
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
        self._key_order: tuple[str, ...] | None = None
        self._statistics_schema: dict[str, str] = {}
        self._gradient_sync_time_sec = 0.0
        self._gradient_sync_calls = 0
        self._gradient_buffers: dict[tuple[int, ...], torch.Tensor] = {}
        self._cuda_graph_collective_ready = False
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

    def prepare_cuda_graph_collectives(self) -> None:
        """Warm up NCCL before the first graph-captured gradient collective.

        NCCL communicator and collective-specific lazy initialization must run
        eagerly. Capturing the first all-reduce either fails at capture time or
        produces a graph that hangs on replay with the supported TCP-loopback
        transport. This mirrors PyTorch's c10d CUDA Graph tests: one eager
        all-reduce followed by a device synchronize before capture.
        """
        if self._cuda_graph_collective_ready:
            return
        if not self._started:
            raise RuntimeError("start() must run before CUDA Graph collective warmup")
        if self.backend != "nccl" or self.device is None or self.device.type != "cuda":
            raise RuntimeError(
                "optimizer CUDA Graph gradient synchronization requires "
                "an NCCL process group with an explicit CUDA device"
            )

        warmup = torch.ones(1, device=self.device)
        dist.all_reduce(warmup, op=dist.ReduceOp.SUM)
        torch.cuda.synchronize(self.device)
        self._cuda_graph_collective_ready = True

    def allreduce_gradients(self, parameters: Iterable[torch.Tensor]) -> None:
        """Average one optimizer's gradients with one flat all-reduce.

        Existing gradients are packed in parameter order, matching the manual
        collective used by Holosoma FastSAC. All ranks must execute the same
        optimizer graph and therefore present the same gradient layout.
        """
        sync_start = time.perf_counter()
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
        buffer_key = tuple(id(parameter) for parameter in params)
        packed = self._gradient_buffers.get(buffer_key)
        if (
            packed is None
            or packed.numel() != gradient_numel
            or packed.device != device
            or packed.dtype != dtype
        ):
            packed = torch.empty(gradient_numel, device=device, dtype=dtype)
            self._gradient_buffers[buffer_key] = packed
        offset = 0
        for parameter in params:
            width = parameter.numel()
            gradient = parameter.grad
            assert gradient is not None
            packed[offset : offset + width].copy_(gradient.detach().reshape(-1))
            offset += width

        capturing = device.type == "cuda" and torch.cuda.is_current_stream_capturing()
        if capturing and not self._cuda_graph_collective_ready:
            raise RuntimeError(
                "NCCL gradient capture requires prepare_cuda_graph_collectives() first"
            )
        dist.all_reduce(packed, op=dist.ReduceOp.SUM)
        packed.div_(self.world_size)

        offset = 0
        for parameter in params:
            width = parameter.numel()
            gradient = parameter.grad
            assert gradient is not None
            gradient.copy_(packed[offset : offset + width].view_as(parameter))
            offset += width

        # Python executes once while a CUDA Graph is captured, not on replay.
        # Learners report replayed collectives separately so this metric counts
        # actual executions rather than graph construction.
        if not capturing:
            self._gradient_sync_time_sec += time.perf_counter() - sync_start
            self._gradient_sync_calls += 1

    def record_cuda_graph_gradient_replay(self, collective_calls: int) -> None:
        """Account for collectives executed by one optimizer graph replay."""
        if collective_calls < 0:
            raise ValueError(f"collective_calls must be >= 0, got {collective_calls}")
        self._gradient_sync_calls += int(collective_calls)

    def take_gradient_sync_metrics(self) -> tuple[float, int]:
        """Return and reset this rank's gradient-sync time and call count."""
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
        packed = torch.zeros((2, len(keys)), dtype=torch.float64, device=device)
        for index, key in enumerate(keys):
            if key not in values:
                continue
            packed[0, index] = float(values[key])
            packed[1, index] = 1.0
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
        self._cuda_graph_collective_ready = False
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
