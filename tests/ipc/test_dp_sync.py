"""Two-process gloo tests for DpParameterSync (CPU-only)."""

from __future__ import annotations

import multiprocessing as mp
from pathlib import Path

import pytest
import torch

from unilab.ipc.dp_launcher import resolve_dp_rendezvous_path
from unilab.ipc.dp_sync import DpParameterSync

_SPAWN_CTX = mp.get_context("spawn")


def _make_tensors(rank: int, *, reversed_order: bool = False) -> dict[str, torch.Tensor]:
    """Deterministic per-rank tensors; distinct values and shapes per key."""
    entries = [
        ("actor.net.weight", torch.full((4, 3), 1.0 + rank, requires_grad=True)),
        ("qnet.head.bias", torch.full((2,), 10.0 + rank, requires_grad=True)),
        ("log_alpha", torch.full((1,), -0.5 + rank, requires_grad=True)),
    ]
    if reversed_order:
        entries.reverse()
    return dict(entries)


def _broadcast_and_gradient_worker(
    rank: int, rendezvous_path: str, reversed_order: bool, result_queue
) -> None:
    tensors = _make_tensors(rank, reversed_order=reversed_order)
    sync = DpParameterSync(
        world_size=2,
        rank=rank,
        rendezvous_path=rendezvous_path,
        backend="gloo",
        timeout_s=60,
    )
    sync.start()

    # Init broadcast: every rank ends up with rank 0's values, bit-exact.
    sync.broadcast_from_rank0(tensors)
    expected = _make_tensors(0)
    for key, tensor in tensors.items():
        assert torch.equal(tensor.detach(), expected[key].detach()), (
            f"rank {rank} broadcast mismatch on {key}"
        )

    grad_ptrs: dict[str, int] = {}
    gradient_base = {"actor.net.weight": 1.0, "qnet.head.bias": 2.0, "log_alpha": 3.0}
    for key, tensor in tensors.items():
        tensor.grad = torch.full_like(tensor, gradient_base[key] + 2.0 * rank)
        grad_ptrs[key] = tensor.grad.data_ptr()
    sync.allreduce_gradients(tensors[key] for key in sorted(tensors))
    for key, tensor in tensors.items():
        assert tensor.grad is not None
        assert torch.allclose(tensor.grad, torch.full_like(tensor, gradient_base[key] + 1.0))
        assert tensor.grad.data_ptr() == grad_ptrs[key]

    sync_time, sync_calls = sync.take_gradient_sync_metrics()
    assert sync_time >= 0.0
    assert sync_calls == 1
    assert sync.take_gradient_sync_metrics() == (0.0, 0)

    sync.close()
    sync.close()  # idempotent
    result_queue.put((rank, sorted(tensors)))


def test_broadcast_then_flat_gradient_mean_two_ranks(tmp_path: Path):
    rendezvous = str(tmp_path / "rendezvous")
    result_queue = _SPAWN_CTX.Queue()
    procs = [
        _SPAWN_CTX.Process(
            target=_broadcast_and_gradient_worker,
            args=(rank, rendezvous, bool(rank), result_queue),
        )
        for rank in range(2)
    ]
    for proc in procs:
        proc.start()
    for proc in procs:
        proc.join(timeout=120)
    results = sorted(result_queue.get(timeout=10) for _ in procs)
    for proc in procs:
        assert proc.exitcode == 0
    # Different insertion order still resolves to the same startup key order.
    assert results[0][1] == results[1][1]


def _statistics_worker(rank: int, rendezvous_path: str, result_queue) -> None:
    sync = DpParameterSync(
        world_size=2,
        rank=rank,
        rendezvous_path=rendezvous_path,
        backend="gloo",
        timeout_s=60,
    )
    sync.start()
    first = sync.allreduce_statistics(
        mean={
            "loss": 1.0 + 2.0 * rank,
            **({"rank0_optional": 10.0} if rank == 0 else {}),
        },
        total={"steps_per_sec": 100.0 * (rank + 1)},
    )
    second = sync.allreduce_statistics(
        mean={
            "loss": 5.0 + 2.0 * rank,
            **({"late_rank1_field": 9.0} if rank == 1 else {}),
        },
        total={"steps_per_sec": 10.0 * (rank + 1)},
    )
    sync.close()
    result_queue.put((rank, first, second))


def test_allreduce_statistics_sums_rates_and_means_sparse_fields(tmp_path: Path):
    rendezvous = str(tmp_path / "rendezvous_statistics")
    result_queue = _SPAWN_CTX.Queue()
    procs = [
        _SPAWN_CTX.Process(
            target=_statistics_worker,
            args=(rank, rendezvous, result_queue),
        )
        for rank in range(2)
    ]
    for proc in procs:
        proc.start()
    for proc in procs:
        proc.join(timeout=120)
    results = sorted(result_queue.get(timeout=10) for _ in procs)
    for proc in procs:
        assert proc.exitcode == 0
    for _, first, second in results:
        assert first == pytest.approx({"loss": 2.0, "rank0_optional": 10.0, "steps_per_sec": 300.0})
        assert second == pytest.approx(
            {"loss": 6.0, "late_rank1_field": 9.0, "steps_per_sec": 30.0}
        )


def test_allreduce_statistics_rejects_overlapping_reduction_modes(tmp_path: Path):
    sync = DpParameterSync(
        world_size=2,
        rank=0,
        rendezvous_path=str(tmp_path / "unused_statistics"),
        backend="gloo",
    )
    with pytest.raises(ValueError, match="both mean and total"):
        sync.allreduce_statistics(mean={"x": 1.0}, total={"x": 2.0})


def test_key_set_change_between_collectives_is_rejected(tmp_path: Path):
    sync = DpParameterSync(
        world_size=2,
        rank=0,
        rendezvous_path=str(tmp_path / "unused"),
        backend="gloo",
    )
    first = {"a": torch.zeros(1), "b": torch.zeros(1)}
    sync._ordered_keys(first)
    with pytest.raises(ValueError, match="tensor keys changed"):
        sync._ordered_keys({"a": torch.zeros(1), "c": torch.zeros(1)})


def test_close_without_start_is_idempotent(tmp_path: Path):
    sync = DpParameterSync(
        world_size=2,
        rank=0,
        rendezvous_path=str(tmp_path / "unused"),
        backend="gloo",
    )
    sync.close()
    sync.close()


def test_invalid_world_size_and_rank_are_rejected(tmp_path: Path):
    path = str(tmp_path / "unused")
    with pytest.raises(ValueError, match="world_size >= 2"):
        DpParameterSync(world_size=1, rank=0, rendezvous_path=path)
    with pytest.raises(ValueError, match="out of range"):
        DpParameterSync(world_size=2, rank=2, rendezvous_path=path)


def test_resolve_dp_rendezvous_path_anchors_on_run_root(tmp_path: Path, monkeypatch):
    # Rank 0 anchors in its own log_dir; spawned ranks use the shared run root.
    monkeypatch.delenv("UNILAB_DP_LOG_DIR", raising=False)
    rank0 = resolve_dp_rendezvous_path(str(tmp_path / "run"), rank=0)
    assert rank0 == str(tmp_path / "run" / ".dp_rendezvous")
    monkeypatch.setenv("UNILAB_DP_LOG_DIR", str(tmp_path / "run"))
    rank1 = resolve_dp_rendezvous_path(str(tmp_path / "run" / "rank1"), rank=1)
    assert rank1 == rank0


@pytest.mark.slow
@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="requires 2 CUDA devices")
def test_nccl_two_gpu_smoke(tmp_path: Path):
    """Real NCCL smoke: startup state, gradients, and logger statistics."""
    rendezvous = str(tmp_path / "rendezvous_nccl")
    result_queue = _SPAWN_CTX.Queue()
    procs = [
        _SPAWN_CTX.Process(
            target=_nccl_smoke_worker,
            args=(rank, rendezvous, result_queue),
        )
        for rank in range(2)
    ]
    for proc in procs:
        proc.start()
    for proc in procs:
        proc.join(timeout=180)
    for proc in procs:
        assert proc.exitcode == 0


def _nccl_smoke_worker(rank: int, rendezvous_path: str, result_queue) -> None:
    device = torch.device(f"cuda:{rank}")
    tensor = torch.nn.Parameter(torch.full((8,), float(rank + 1), device=device))
    sync = DpParameterSync(
        world_size=2,
        rank=rank,
        rendezvous_path=rendezvous_path,
        backend="nccl",
        device=str(device),
        timeout_s=120,
    )
    sync.start()
    tensors = {"w": tensor}
    sync.broadcast_from_rank0(tensors)
    assert torch.equal(tensor, torch.ones(8, device=device))
    tensor.grad = torch.full_like(tensor, float(rank + 1))
    sync.allreduce_gradients((tensor,))
    assert torch.allclose(tensor.grad, torch.full((8,), 1.5, device=device))
    statistics = sync.allreduce_statistics(
        mean={"loss": float(rank + 1)},
        total={"steps_per_sec": float(100 * (rank + 1))},
    )
    assert statistics == pytest.approx({"loss": 1.5, "steps_per_sec": 300.0})
    sync.close()
    result_queue.put(rank)
