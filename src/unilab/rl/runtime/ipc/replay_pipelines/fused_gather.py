"""Fused fp16 replay gather for CUDA device rings.

Mixed-storage (fp16) rings historically assembled each fp32 learner batch
with three passes: an ``index_select`` into an fp16 scratch buffer, a
full-width up-cast copy, and a scalar-column gather plus overwrite from the
authoritative fp32 block. The kernel below fuses all three into one pass —
it reads one fp16 ring row and the three fp32 scalars, replaces the scalar
columns in registers, and stores one fp32 row — roughly halving gather
traffic versus the fp32 ring's single ``index_select`` and removing the
scratch round-trip (issue #2652).

Triton is imported lazily and only on CUDA devices; when the kernel cannot
be built or verified (no triton, oversized rows, compile or smoke-test
failure), the builder returns ``None`` and the pipeline keeps the eager
three-pass path. MPS learners always use the eager path.
"""

from __future__ import annotations

from collections.abc import Callable

import torch

_MAX_FUSED_ROW_WIDTH = 4096


def _reference_gather(
    ring: torch.Tensor,
    scalars: torch.Tensor,
    indices: torch.Tensor,
    out: torch.Tensor,
    *,
    scalar_start: int,
) -> None:
    """Eager three-pass gather; used only as the smoke-test reference."""
    scratch = torch.empty(
        (int(indices.shape[0]), int(out.shape[1])), dtype=ring.dtype, device=ring.device
    )
    torch.index_select(ring, 0, indices, out=scratch)
    out.copy_(scratch)
    scalar_rows = torch.index_select(scalars, 0, indices)
    out[:, scalar_start : scalar_start + scalar_rows.shape[1]].copy_(scalar_rows)


def build_fused_gather(
    *,
    sample_count: int,
    width: int,
    scalar_start: int,
    device: torch.device,
) -> Callable[[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], None] | None:
    """Build the fused gather closure for one ring geometry.

    Returns ``None`` when the fused path is unavailable and the caller must
    keep its eager fallback. The returned callable writes
    ``out (sample_count, width)`` fp32 rows gathered from the fp16 ``ring``
    with the three scalar columns replaced from the fp32 ``scalars`` block.
    A tiny smoke launch against the eager reference warms the exact compiled
    kernel (same ``BLOCK`` specialization) before training starts.
    """
    if device.type != "cuda":
        return None
    try:
        import triton
        import triton.language as tl
    except Exception:
        return None
    if width > _MAX_FUSED_ROW_WIDTH:
        return None

    @triton.jit
    def fused_gather_kernel(
        ring_ptr,
        sc_ptr,
        out_ptr,
        idx_ptr,
        width,
        scalar_start,
        BLOCK: tl.constexpr,
    ) -> None:
        row = tl.program_id(0)
        src = tl.load(idx_ptr + row).to(tl.int64)
        cols = tl.arange(0, BLOCK)
        mask = cols < width
        values = tl.load(ring_ptr + src * width + cols, mask=mask, other=0).to(tl.float32)
        s0 = tl.load(sc_ptr + src * 3 + 0)
        s1 = tl.load(sc_ptr + src * 3 + 1)
        s2 = tl.load(sc_ptr + src * 3 + 2)
        values = tl.where(cols == scalar_start, s0, values)
        values = tl.where(cols == scalar_start + 1, s1, values)
        values = tl.where(cols == scalar_start + 2, s2, values)
        tl.store(out_ptr + row.to(tl.int64) * width + cols, values, mask=mask)

    block = triton.next_power_of_2(width)
    rows = int(sample_count)

    def launch_n(
        n_rows: int,
        ring: torch.Tensor,
        scalars: torch.Tensor,
        indices: torch.Tensor,
        out: torch.Tensor,
    ) -> None:
        fused_gather_kernel[(n_rows,)](
            ring,
            scalars,
            out,
            indices,
            width,
            scalar_start,
            BLOCK=block,
            num_warps=4,  # pyright: ignore[reportCallIssue]  (triton launch grid kwargs)
        )

    def launch(
        ring: torch.Tensor,
        scalars: torch.Tensor,
        indices: torch.Tensor,
        out: torch.Tensor,
    ) -> None:
        launch_n(rows, ring, scalars, indices, out)

    try:
        smoke_ring = torch.zeros((8, width), dtype=torch.float16, device=device)
        smoke_ring[:, :] = torch.arange(width, dtype=torch.float16, device=device)
        smoke_scalars = torch.arange(24, dtype=torch.float32, device=device).reshape(8, 3)
        smoke_idx = torch.arange(4, dtype=torch.int64, device=device)
        got = torch.full((4, width), float("nan"), dtype=torch.float32, device=device)
        launch_n(4, smoke_ring, smoke_scalars, smoke_idx, got)
        torch.cuda.synchronize(device)
        want = torch.empty((4, width), dtype=torch.float32, device=device)
        _reference_gather(smoke_ring, smoke_scalars, smoke_idx, want, scalar_start=scalar_start)
        if not torch.equal(got, want):
            return None
    except Exception:
        return None

    def gather(
        ring: torch.Tensor,
        scalars: torch.Tensor,
        indices: torch.Tensor,
        out: torch.Tensor,
    ) -> None:
        launch(ring, scalars, indices, out)

    return gather


__all__ = ["build_fused_gather"]
