"""Periodic parameter averaging across data-parallel ranks (torch.distributed).

Multi-GPU data-parallel off-policy training runs one independent
learner+collector pair per rank (see ``dp_launcher.py``). Ranks start from
different initial parameters (seed = cfg.algo.seed + rank), so the runner
broadcasts rank 0's parameters before the collector starts, then averages
actor/critic parameters every ``dp_sync_interval`` learner iterations at the
update boundary (never overlapping an inference tick).

This module owns the process-group lifecycle, parameter collectives, and the
small scalar reduction used by the canonical rank-0 logger. The runner decides
when to call them. Synchronization is parameter-level by design: gradients are
never exchanged and each rank keeps its own optimizer state (AdamW moments
diverge across ranks — see ``FastSACLearner.dp_sync_tensors``).
"""

from __future__ import annotations

import datetime
from typing import cast

import torch
import torch.distributed as dist


class DpParameterSync:
    """torch.distributed process group for DP parameter broadcast/averaging.

    Both collectives mutate the given tensors **in place**: CUDA-graph capture
    pins parameter addresses and AMP master weights stay fp32, so after an
    in-place all-reduce every rank already holds the mean with no write-back.

    The synchronization key order is frozen on the first collective
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
        path, and it is fast enough for periodic parameter averaging.
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

    def allreduce_mean(self, tensors: dict[str, torch.Tensor]) -> None:
        """Replace every tensor with the cross-rank mean, in place."""
        with torch.no_grad():
            for key in self._ordered_keys(tensors):
                tensor = tensors[key]
                dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
                tensor.div_(self.world_size)

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

        This collective is intentionally separate from parameter averaging:
        logging is aggregated every learner iteration while model parameters
        retain the configured ``dp_sync_interval`` semantics.
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
            gathered: list[object | None] = [None] * self.world_size
            dist.all_gather_object(gathered, tuple(sorted(local_schema.items())))
            merged = dict(self._statistics_schema)
            for rank_schema in gathered:
                assert rank_schema is not None
                for key, reduction in cast(tuple[tuple[str, str], ...], rank_schema):
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

    def close(self) -> None:
        """Destroy the process group; idempotent and safe before start()."""
        if not self._started:
            return
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()
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
