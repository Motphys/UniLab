"""Unit tests for the env_ids→mask reset adaptation layer (roadmap #2461)."""

import pytest
import torch

from unilab.base.reset_selectors import (
    ResetMaskBuffer,
    reset_env_ids_mask,
    resolve_reset_env_ids,
    resolve_reset_mask,
)


def test_scatter_matches_fresh_allocation() -> None:
    num_envs = 64
    buffer = ResetMaskBuffer(num_envs, torch.device("cpu"))
    subsets = (
        torch.arange(num_envs, dtype=torch.int64),
        torch.tensor([0, 5, 63], dtype=torch.int64),
        torch.tensor([63, 5, 5, 0], dtype=torch.int64),
        torch.empty(0, dtype=torch.int64),
    )
    for env_ids in subsets:
        expected = reset_env_ids_mask(env_ids, num_envs=num_envs, device=torch.device("cpu"))
        reused = buffer.scatter(env_ids)
        assert torch.equal(reused, expected)


def test_scatter_reuses_storage_without_stale_bits() -> None:
    num_envs = 32
    buffer = ResetMaskBuffer(num_envs, torch.device("cpu"))
    first = buffer.scatter(torch.tensor([1, 2, 3], dtype=torch.int64))
    assert first.sum().item() == 3
    second = buffer.scatter(torch.tensor([10, 20], dtype=torch.int64))
    assert second.data_ptr() == first.data_ptr()
    assert second.sum().item() == 2
    assert not bool(second[1] | second[2] | second[3])
    assert bool(second[10]) and bool(second[20])
    cleared = buffer.scatter(torch.empty(0, dtype=torch.int64))
    assert cleared.data_ptr() == first.data_ptr()
    assert not bool(cleared.any())


def test_scatter_accepts_non_int64_ids() -> None:
    buffer = ResetMaskBuffer(16, torch.device("cpu"))
    mask = buffer.scatter(torch.tensor([0, 7, 15], dtype=torch.int32))
    assert mask.dtype == torch.bool
    assert torch.equal(
        mask,
        reset_env_ids_mask(
            torch.tensor([0, 7, 15], dtype=torch.int64), num_envs=16, device=torch.device("cpu")
        ),
    )


def test_buffers_are_independent() -> None:
    first = ResetMaskBuffer(8, torch.device("cpu"))
    second = ResetMaskBuffer(8, torch.device("cpu"))
    first_mask = first.scatter(torch.tensor([0, 1], dtype=torch.int64))
    second_mask = second.scatter(torch.tensor([7], dtype=torch.int64))
    assert first_mask.data_ptr() != second_mask.data_ptr()
    assert torch.equal(
        first_mask,
        reset_env_ids_mask(torch.tensor([0, 1]), num_envs=8, device=torch.device("cpu")),
    )
    assert torch.equal(
        second_mask,
        reset_env_ids_mask(torch.tensor([7]), num_envs=8, device=torch.device("cpu")),
    )


def test_resolve_ids_matches_allocating_form() -> None:
    num_envs = 16
    buffer = ResetMaskBuffer(num_envs, torch.device("cpu"))
    for selector in (
        None,
        torch.arange(num_envs, dtype=torch.int64)[2:10:3],
        torch.tensor([4, 9], dtype=torch.int64),
    ):
        expected = resolve_reset_mask(
            selector, num_envs=num_envs, device=torch.device("cpu"), owner="test"
        )
        reused = buffer.resolve(selector, owner="test")
        assert torch.equal(reused, expected)


def test_resolve_rejects_slice() -> None:
    num_envs = 16
    selector = slice(2, 10, 3)
    with pytest.raises(TypeError, match="slice"):
        resolve_reset_env_ids(selector, num_envs=num_envs, device=torch.device("cpu"), owner="test")
    buffer = ResetMaskBuffer(num_envs, torch.device("cpu"))
    with pytest.raises(TypeError, match="slice"):
        buffer.resolve(selector, owner="test")
    with pytest.raises(TypeError, match="slice"):
        resolve_reset_mask(selector, num_envs=num_envs, device=torch.device("cpu"), owner="test")


def test_resolve_rejects_bool_mask() -> None:
    num_envs = 8
    buffer = ResetMaskBuffer(num_envs, torch.device("cpu"))
    caller_mask = torch.zeros(num_envs, dtype=torch.bool)
    caller_mask[[3, 4]] = True
    with pytest.raises(TypeError, match="integer row-index tensor"):
        buffer.resolve(caller_mask, owner="test.resolve")
    with pytest.raises(TypeError, match="integer row-index tensor"):
        buffer.resolve(torch.zeros(4, dtype=torch.bool), owner="test.resolve")
    with pytest.raises(TypeError, match="integer row-index tensor"):
        resolve_reset_mask(caller_mask, num_envs=num_envs, device=torch.device("cpu"), owner="t")
