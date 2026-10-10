"""Contract tests for the device-resident motion sampler."""

from __future__ import annotations

import pytest
import torch

from unilab.tasks.motion_tracking.common import tensor_sampler as mt
from unilab.tasks.motion_tracking.common.motion_loader import MotionSampler
from unilab.tasks.motion_tracking.common.tensor_sampler import TensorMotionSampler


def _loader(num_frames: int = 32) -> object:
    loader = object.__new__(MotionSampler.__mro__[0])
    return loader


def _sampler(
    *,
    mode: str = "adaptive",
    num_envs: int = 4,
    num_frames: int = 32,
    bin_count: int = 4,
    start_ratio: float = 0.0,
    initial_frames: torch.Tensor | None = None,
) -> TensorMotionSampler:
    frames = torch.zeros(num_envs, dtype=torch.int32) if initial_frames is None else initial_frames
    return TensorMotionSampler(
        mode=mode,
        num_envs=num_envs,
        num_frames=num_frames,
        clip_offsets=torch.tensor([0], dtype=torch.int64),
        clip_end_frames=torch.tensor([num_frames - 1], dtype=torch.int32),
        bin_count=bin_count,
        adaptive_lambda=0.8,
        adaptive_kernel_size=1,
        adaptive_uniform_ratio=0.1,
        adaptive_alpha=0.5,
        start_ratio=start_ratio,
        initial_frames=frames,
        initial_clip_end_frames=torch.full((num_envs,), num_frames - 1, dtype=torch.int32),
        device=torch.device("cpu"),
    )


def test_sampler_rejects_unsupported_mode_and_invalid_dimensions() -> None:
    with pytest.raises(NotImplementedError, match="adaptive/mixed"):
        _sampler(mode="uniform")
    with pytest.raises(ValueError, match="dimensions must be positive"):
        _sampler(num_envs=0)


def test_adaptive_reset_sampling_is_device_resident_and_deterministic() -> None:
    sampler = _sampler()
    generator = torch.Generator(device="cpu").manual_seed(7)
    rows = torch.tensor([0, 2, 3], dtype=torch.int64)

    first = sampler.sample_frames(rows, generator)
    second = sampler.sample_frames(rows, generator)
    assert first.shape == (3,)
    assert first.dtype == torch.int32
    assert first.device.type == "cpu"
    assert not torch.equal(first, second)
    assert sampler.diagnostics.total == 0

    generator.manual_seed(7)
    repeat = sampler.sample_frames(rows, generator)
    torch.testing.assert_close(repeat, first)


def test_adaptive_failure_stats_match_bin_counts_without_host_sync() -> None:
    sampler = _sampler()
    sampler.current_frames.copy_(torch.tensor([0, 8, 16, 24], dtype=torch.int32))
    terminated = torch.tensor([True, False, True, False])
    before = sampler.bin_failed_count.clone()
    sampler.update_failure_stats(terminated)
    expected = torch.zeros_like(before)
    expected[0] = 1.0
    expected[2] = 1.0
    torch.testing.assert_close(
        sampler.bin_failed_count, 0.5 * expected + 0.5 * before, rtol=0, atol=0
    )
    sampler.update_failure_stats(torch.zeros_like(terminated))
    torch.testing.assert_close(
        sampler.bin_failed_count, 0.5 * expected + 0.5 * before, rtol=0, atol=0
    )


def test_step_updates_frames_and_clip_ends_on_device() -> None:
    sampler = _sampler(initial_frames=torch.tensor([0, 1, 2, 3], dtype=torch.int32))
    active = torch.tensor([True, False, True, True])
    done = sampler.step(active)
    torch.testing.assert_close(
        sampler.current_frames, torch.tensor([1, 1, 3, 4], dtype=torch.int32)
    )
    assert done.numel() == 0
    assert sampler.diagnostics.total == 0


def test_step_selected_rows_does_not_touch_or_read_inactive_rows() -> None:
    sampler = _sampler(initial_frames=torch.tensor([0, 1, 2, 3], dtype=torch.int32))
    sampler.current_clip_end_frames.copy_(torch.tensor([9, 9, 9, 9], dtype=torch.int32))
    active = torch.tensor([True, False, True, False])
    rows = torch.tensor([0, 2], dtype=torch.int64)
    done = sampler.step(active, rows=rows)

    torch.testing.assert_close(
        sampler.current_frames, torch.tensor([1, 1, 3, 3], dtype=torch.int32)
    )
    torch.testing.assert_close(
        sampler.current_clip_end_frames, torch.tensor([31, 9, 31, 9], dtype=torch.int32)
    )
    assert done.numel() == 0
    assert sampler.diagnostics.total == 0


def test_step_full_publishes_mirrors_without_row_gathers() -> None:
    sampler = _sampler(initial_frames=torch.tensor([0, 1, 2, 3], dtype=torch.int32))
    time_steps = torch.tensor([99, 99, 99, 99], dtype=torch.int32)
    active = torch.tensor([True, False, True, False])

    done = sampler.step_full(active, time_steps)

    torch.testing.assert_close(
        sampler.current_frames, torch.tensor([1, 1, 3, 3], dtype=torch.int32)
    )
    torch.testing.assert_close(time_steps, torch.tensor([1, 1, 3, 3], dtype=torch.int32))
    torch.testing.assert_close(
        sampler.current_clip_end_frames,
        sampler._frame_clip_ends.index_select(0, sampler.current_frames),
    )
    assert done.numel() == 0
    assert sampler.diagnostics.total == 0


def test_sampling_metrics_stay_device_scalars() -> None:
    sampler = _sampler()
    sampler.sample_frames(torch.tensor([0], dtype=torch.int64), torch.Generator().manual_seed(1))
    assert sampler.sampling_entropy.device.type == "cpu"
    assert sampler.sampling_top1_prob.device.type == "cpu"
    assert sampler.sampling_top1_bin.device.type == "cpu"
    assert bool(torch.isfinite(sampler.sampling_entropy))


def test_sampling_dispatch_matches_eager_clip_lookup_and_publishes_timing() -> None:
    sampler = _sampler(initial_frames=torch.tensor([0, 1, 2, 3], dtype=torch.int32))
    rows = torch.tensor([1, 3], dtype=torch.int64)
    frames = torch.tensor([31, 7], dtype=torch.int32)

    mt._sampling_dispatch_kernel(
        rows,
        frames,
        sampler._frame_clip_ends,
        sampler.current_frames,
        sampler.current_clip_end_frames,
    )

    torch.testing.assert_close(
        sampler.current_frames, torch.tensor([0, 31, 2, 7], dtype=torch.int32)
    )
    expected_ends = sampler._frame_clip_ends[frames]
    torch.testing.assert_close(sampler.current_clip_end_frames[rows], expected_ends)

    sampler.last_reset_dispatch_ms = 0.0
    generator = torch.Generator(device="cpu").manual_seed(11)
    sampler.sample_frames(rows, generator)
    assert sampler.last_reset_dispatch_ms >= 0.0


def test_frame_indexed_clip_ends_match_searchsorted_ownership() -> None:
    offsets = torch.tensor([0, 7, 19, 30], dtype=torch.int64)
    ends = torch.tensor([6, 18, 29, 31], dtype=torch.int32)
    sampler = TensorMotionSampler(
        mode="adaptive",
        num_envs=7,
        num_frames=32,
        clip_offsets=offsets,
        clip_end_frames=ends,
        bin_count=4,
        adaptive_lambda=0.8,
        adaptive_kernel_size=1,
        adaptive_uniform_ratio=0.1,
        adaptive_alpha=0.5,
        start_ratio=0.0,
        initial_frames=torch.arange(7, dtype=torch.int32),
        initial_clip_end_frames=torch.full((7,), 6, dtype=torch.int32),
        device=torch.device("cpu"),
    )

    frames = torch.arange(32, dtype=torch.int64)
    expected_indices = (torch.searchsorted(sampler._clip_offsets, frames, right=True) - 1).clamp_(
        min=0
    )
    expected = sampler._clip_end_frames.index_select(0, expected_indices)

    torch.testing.assert_close(sampler._frame_clip_ends, expected)
