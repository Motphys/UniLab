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
        ("actor.net.weight", torch.full((4, 3), 1.0 + rank)),
        ("qnet.head.bias", torch.full((2,), 10.0 + rank)),
        ("log_alpha", torch.full((1,), -0.5 + rank, requires_grad=True)),
    ]
    if reversed_order:
        entries.reverse()
    return dict(entries)


def _broadcast_and_allreduce_worker(
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

    # In-place semantics: allreduce_mean must not rebind the storage.
    ptrs_before = {key: tensor.data_ptr() for key, tensor in tensors.items()}

    # Per-rank perturbation -> allreduce_mean lands on the cross-rank mean.
    for key, tensor in tensors.items():
        with torch.no_grad():
            tensor.add_(float(rank + 1))
    sync.allreduce_mean(tensors)
    for key, tensor in tensors.items():
        base = expected[key].detach()
        assert torch.allclose(tensor.detach(), base + 1.5), (
            f"rank {rank} allreduce mismatch on {key}"
        )
        assert tensor.data_ptr() == ptrs_before[key], f"{key} was not updated in place"

    sync.close()
    sync.close()  # idempotent
    result_queue.put((rank, sorted(tensors)))


def test_broadcast_then_allreduce_mean_two_ranks(tmp_path: Path):
    rendezvous = str(tmp_path / "rendezvous")
    result_queue = _SPAWN_CTX.Queue()
    procs = [
        _SPAWN_CTX.Process(
            target=_broadcast_and_allreduce_worker,
            args=(rank, rendezvous, False, result_queue),
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
    # Both ranks walked the same (sorted) key order.
    assert results[0][1] == results[1][1]


def _key_order_worker(rank: int, rendezvous_path: str, result_queue) -> None:
    # Insertion order differs per rank; the collective order must not.
    tensors = _make_tensors(rank, reversed_order=bool(rank))
    sync = DpParameterSync(
        world_size=2,
        rank=rank,
        rendezvous_path=rendezvous_path,
        backend="gloo",
        timeout_s=60,
    )
    sync.start()
    sync.allreduce_mean(tensors)
    for key, tensor in tensors.items():
        mean = (_make_tensors(0)[key].detach() + _make_tensors(1)[key].detach()) / 2
        assert torch.allclose(tensor.detach(), mean), f"rank {rank} key-order mismatch on {key}"
    sync.close()
    result_queue.put(rank)


def test_collective_key_order_is_insertion_order_independent(tmp_path: Path):
    rendezvous = str(tmp_path / "rendezvous_key_order")
    result_queue = _SPAWN_CTX.Queue()
    procs = [
        _SPAWN_CTX.Process(target=_key_order_worker, args=(rank, rendezvous, result_queue))
        for rank in range(2)
    ]
    for proc in procs:
        proc.start()
    for proc in procs:
        proc.join(timeout=120)
    for proc in procs:
        assert proc.exitcode == 0


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
    """Real NCCL smoke: init + broadcast + allreduce across cuda:0/cuda:1."""
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
    tensor = torch.full((8,), float(rank + 1), device=device)
    sync = DpParameterSync(
        world_size=2,
        rank=rank,
        rendezvous_path=rendezvous_path,
        backend="nccl",
        timeout_s=120,
    )
    sync.start()
    tensors = {"w": tensor}
    sync.broadcast_from_rank0(tensors)
    assert torch.equal(tensor, torch.ones(8, device=device))
    sync.allreduce_mean(tensors)
    assert torch.allclose(tensor, torch.full((8,), 0.5 + 0.5, device=device))
    sync.close()
    result_queue.put(rank)
