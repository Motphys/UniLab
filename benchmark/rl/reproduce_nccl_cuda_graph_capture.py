"""Bounded two-rank reproducer for NCCL CUDA Graph capture (issue #978).

Verified on 2 x RTX 6000D, PyTorch 2.7.0+cu128, CUDA 12.8 and NCCL
2.26.2 with P2P/SHM disabled (TCP loopback):

- no warmup: capture fails with ``operation not permitted when stream is capturing``;
- eager all-reduce warmup: synchronous capture and replay pass;
- ``async_op=True`` with captured ``Work.wait()`` also passes;
- both collective modes pass on the default and a side stream.

Every invocation has a parent-enforced timeout, including the intentionally
hanging/error variants. The production FastSAC/FlashSAC path keeps synchronous
all-reduce and performs the required eager warmup before its first capture.

Run:
    uv run benchmark/rl/reproduce_nccl_cuda_graph_capture.py
    uv run benchmark/rl/reproduce_nccl_cuda_graph_capture.py --warmup
    uv run benchmark/rl/reproduce_nccl_cuda_graph_capture.py --warmup --async-op
    uv run benchmark/rl/reproduce_nccl_cuda_graph_capture.py \
        --warmup --async-op --side-stream
"""

from __future__ import annotations

import argparse
import os
import tempfile
import time

import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def _worker(
    rank: int,
    rendezvous_path: str,
    warmup: bool,
    side_stream: bool,
    async_op: bool,
) -> None:
    os.environ.setdefault("NCCL_P2P_DISABLE", "1")
    os.environ.setdefault("NCCL_SHM_DISABLE", "1")
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)
    dist.init_process_group(
        backend="nccl",
        init_method=f"file://{rendezvous_path}",
        rank=rank,
        world_size=2,
    )
    try:
        if warmup:
            warmup_tensor = torch.ones(1, device=device)
            dist.all_reduce(warmup_tensor)
            torch.cuda.synchronize(device)
            print(f"rank={rank} warmup complete", flush=True)

        tensor = torch.full((1024,), rank + 1, dtype=torch.float32, device=device)
        graph = torch.cuda.CUDAGraph()
        if side_stream:
            capture_stream = torch.cuda.Stream(device=device)
            capture_stream.wait_stream(torch.cuda.current_stream(device))
            with torch.cuda.stream(capture_stream), torch.cuda.graph(graph):
                work = dist.all_reduce(tensor, async_op=async_op)
                if work is not None:
                    work.wait()
            torch.cuda.current_stream(device).wait_stream(capture_stream)
        else:
            with torch.cuda.graph(graph):
                work = dist.all_reduce(tensor, async_op=async_op)
                if work is not None:
                    work.wait()
        print(f"rank={rank} capture complete", flush=True)
        graph.replay()
        torch.cuda.synchronize(device)
        torch.testing.assert_close(tensor, torch.full_like(tensor, 3.0))
        print(f"rank={rank} replay complete", flush=True)
        graph.reset()
        torch.cuda.synchronize(device)
    finally:
        dist.destroy_process_group()


def _terminate_processes(processes: list[mp.Process]) -> None:
    for process in processes:
        if process.is_alive():
            process.terminate()
    for process in processes:
        process.join(timeout=5)
        if process.is_alive():
            process.kill()
            process.join()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--warmup", action="store_true")
    parser.add_argument("--side-stream", action="store_true")
    parser.add_argument("--async-op", action="store_true")
    parser.add_argument("--timeout-seconds", type=float, default=20.0)
    args = parser.parse_args()
    if args.timeout_seconds <= 0:
        parser.error("--timeout-seconds must be positive")
    if torch.cuda.device_count() < 2:
        parser.error("the NCCL capture reproducer requires at least two CUDA devices")

    with tempfile.TemporaryDirectory(prefix="issue978-nccl-") as temp_dir:
        rendezvous_path = os.path.join(temp_dir, "store")
        context = mp.spawn(
            _worker,
            args=(rendezvous_path, args.warmup, args.side_stream, args.async_op),
            nprocs=2,
            join=False,
        )
        deadline = time.monotonic() + args.timeout_seconds
        while not context.join(timeout=max(0.0, deadline - time.monotonic())):
            if time.monotonic() >= deadline:
                _terminate_processes(context.processes)
                raise TimeoutError(
                    f"NCCL graph capture did not finish within {args.timeout_seconds:g}s"
                )


if __name__ == "__main__":
    main()
