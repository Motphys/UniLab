"""Device-resident motion sampling for tensor Manager owners.

This owner owns the cold metadata derivation and hot tensor lifecycle: adaptive
failure statistics, frame advance, reset sampling, and sampling diagnostics stay
on one Torch device and consume the Manager-owned Torch generator.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import cast

import torch


def _sampling_dispatch_kernel(
    rows: torch.Tensor,
    frames: torch.Tensor,
    frame_clip_ends: torch.Tensor,
    current_frames: torch.Tensor,
    current_clip_end_frames: torch.Tensor,
) -> None:
    # Keep this tiny selected-row dispatch eager. An Inductor fusion of
    # searchsorted/index-select measured no wall-time benefit at ~39 reset rows,
    # and synthetic wide-clip tensors triggered a Triton index assertion while
    # autotuning this graph. The frame-indexed clip-end table removes the hot
    # searchsorted path while keeping construction cold and device-resident.
    current_frames.index_copy_(0, rows, frames)
    current_clip_end_frames.index_copy_(0, rows, frame_clip_ends.index_select(0, frames))


def _sampling_dispatch_mask_kernel(
    reset_mask: torch.Tensor,
    frames: torch.Tensor,
    frame_clip_ends: torch.Tensor,
    current_frames: torch.Tensor,
    current_clip_end_frames: torch.Tensor,
) -> None:
    selected = torch.where(reset_mask, frames, current_frames)
    clip_ends = frame_clip_ends.index_select(0, frames)
    current_frames.copy_(selected)
    torch.where(reset_mask, clip_ends, current_clip_end_frames, out=current_clip_end_frames)


@dataclass
class TensorMotionSamplerDiagnostics:
    """Count observed compatibility paths that would defeat tensor-only resets.

    The sampler itself no longer transfers rows to the host. We retain a real
    (not synthesized) counter for reset sampling calls so diagnostics can assert
    that reset dispatch happened while host-transfer counts remain structurally
    zero.
    """

    reset_dispatches: int = 0
    reset_host_row_transfers: int = 0
    step_host_mirror_transfers: int = 0

    def record_reset_dispatch(self) -> None:
        self.reset_dispatches += 1

    @property
    def total(self) -> int:
        return self.reset_host_row_transfers + self.step_host_mirror_transfers


class TensorMotionSampler:
    """Torch-native adaptive/mixed motion sampler for ``MotionCommand``."""

    def __init__(
        self,
        *,
        mode: str,
        num_envs: int,
        num_frames: int,
        clip_offsets: torch.Tensor,
        clip_end_frames: torch.Tensor,
        bin_count: int,
        adaptive_lambda: float,
        adaptive_kernel_size: int,
        adaptive_uniform_ratio: float,
        adaptive_alpha: float,
        start_ratio: float,
        initial_frames: torch.Tensor,
        initial_clip_end_frames: torch.Tensor,
        device: torch.device,
    ) -> None:
        if mode not in {"adaptive", "mixed"}:
            raise NotImplementedError(
                f"TensorMotionSampler supports adaptive/mixed modes only, got {mode!r}"
            )
        if num_envs <= 0 or num_frames <= 0 or bin_count <= 0:
            raise ValueError("TensorMotionSampler dimensions must be positive")
        if adaptive_kernel_size <= 0:
            raise ValueError("TensorMotionSampler adaptive_kernel_size must be positive")
        if not 0.0 <= adaptive_uniform_ratio or not 0.0 <= start_ratio <= 1.0:
            raise ValueError("TensorMotionSampler sampling ratios are outside [0, 1]")
        if adaptive_alpha < 0.0:
            raise ValueError("TensorMotionSampler adaptive_alpha must be non-negative")

        self.mode = mode
        self.num_envs = int(num_envs)
        self.num_frames = int(num_frames)
        self.bin_count = int(bin_count)
        self.adaptive_lambda = float(adaptive_lambda)
        self.adaptive_kernel_size = int(adaptive_kernel_size)
        self.adaptive_uniform_ratio = float(adaptive_uniform_ratio)
        self.adaptive_alpha = float(adaptive_alpha)
        self.start_ratio = float(start_ratio)
        self.device = torch.device(device)
        if self.device.type == "cuda" and self.device.index is None:
            self.device = torch.device("cuda", index=torch.cuda.current_device())

        self.current_frames = initial_frames.detach().to(
            device=self.device, dtype=torch.int32, copy=True
        )
        self.current_clip_end_frames = initial_clip_end_frames.detach().to(
            device=self.device, dtype=torch.int32, copy=True
        )
        self._clip_offsets = clip_offsets.detach().to(
            device=self.device, dtype=torch.int64, copy=True
        )
        self._clip_end_frames = clip_end_frames.detach().to(
            device=self.device, dtype=torch.int32, copy=True
        )
        # Expand clip ownership to one int32 entry per frame. This trades a
        # cold O(num_frames) table for removing searchsorted from every selected
        # reset and full-width frame advance.
        expanded_clip_ends = torch.empty((self.num_frames,), dtype=torch.int32)
        for clip_index in range(len(clip_end_frames)):
            start = int(clip_offsets[clip_index])
            end = (
                int(clip_offsets[clip_index + 1])
                if clip_index + 1 < len(clip_offsets)
                else self.num_frames
            )
            expanded_clip_ends[start:end] = clip_end_frames[clip_index]
        self._frame_clip_ends = expanded_clip_ends.to(device=self.device)
        self._bin_failed = torch.zeros(self.bin_count, dtype=torch.float32, device=self.device)
        kernel = torch.tensor(
            [self.adaptive_lambda**index for index in range(self.adaptive_kernel_size)],
            dtype=torch.float32,
        )
        kernel /= kernel.sum()
        self._adaptive_kernel = kernel.to(device=self.device)
        self.sampling_entropy = torch.zeros((), dtype=torch.float32, device=self.device)
        self.sampling_top1_prob = torch.zeros((), dtype=torch.float32, device=self.device)
        self.sampling_top1_bin = torch.zeros((), dtype=torch.float32, device=self.device)
        self._reset_random_offsets = torch.empty(
            self.num_envs, dtype=torch.float32, device=self.device
        )
        self._reset_scaled_frames = torch.empty(
            self.num_envs, dtype=torch.float32, device=self.device
        )
        self._reset_start_gate = torch.empty(self.num_envs, dtype=torch.bool, device=self.device)
        self._reset_start_random = torch.empty(
            self.num_envs, dtype=torch.float32, device=self.device
        )
        self._reset_uniform_frames = torch.empty(
            self.num_envs, dtype=torch.int32, device=self.device
        )
        self._reset_frames = torch.empty(self.num_envs, dtype=torch.int32, device=self.device)
        self._step_next_frames = torch.empty(self.num_envs, dtype=torch.int32, device=self.device)
        self._step_active_rows = torch.empty(self.num_envs, dtype=torch.int64, device=self.device)
        self._step_clamped_rows = torch.empty(self.num_envs, dtype=torch.int64, device=self.device)
        self._step_clip_ends = torch.empty(self.num_envs, dtype=torch.int32, device=self.device)
        self._step_done = torch.empty(self.num_envs, dtype=torch.bool, device=self.device)
        self._failure_frames = torch.empty(self.num_envs, dtype=torch.int64, device=self.device)
        self._failure_bin_indices = torch.empty(
            self.num_envs, dtype=torch.int64, device=self.device
        )
        self._failure_failed_bins = torch.empty(
            self.num_envs, dtype=torch.int64, device=self.device
        )
        self._failure_discard_fill = torch.empty(
            self.num_envs, dtype=torch.int64, device=self.device
        )
        self._failure_counts = torch.zeros(
            self.bin_count + 1,
            dtype=torch.int64,
            device=self.device,
        )
        self._failure_terminated = torch.empty((), dtype=torch.bool, device=self.device)
        self._failure_ones = torch.ones(self.num_envs, dtype=torch.int64, device=self.device)
        self._failure_alpha = torch.empty(1, dtype=torch.float32, device=self.device)
        self._failure_update = torch.empty_like(self._bin_failed)
        self._adaptive_probability_carrier = torch.empty_like(self._bin_failed)
        self._adaptive_probability_sum = torch.empty((), dtype=torch.float32, device=self.device)
        self._adaptive_conv_input = torch.empty(
            (1, 1, self.bin_count + self.adaptive_kernel_size - 1),
            dtype=torch.float32,
            device=self.device,
        )
        self._adaptive_conv_output = torch.empty(
            (1, 1, self.bin_count), dtype=torch.float32, device=self.device
        )
        self._sampling_probability_scratch = torch.empty_like(self._bin_failed)
        self._sampling_log_scratch = torch.empty_like(self._bin_failed)
        self._sampling_entropy_scratch = torch.empty_like(self.sampling_entropy)
        self._sampling_log_bin_count = torch.log(
            torch.tensor(float(self.bin_count), device=self.device)
        )
        self._sampling_top1_index = torch.empty((), dtype=torch.int64, device=self.device)
        self.diagnostics = TensorMotionSamplerDiagnostics()

    @property
    def bin_failed_count(self) -> torch.Tensor:
        return self._bin_failed

    def sample_frames(
        self, reset_mask: torch.Tensor, torch_rng: torch.Generator | None
    ) -> torch.Tensor:
        """Sample full-width frames and publish only selected rows."""
        started = time.perf_counter()
        if (
            reset_mask.ndim != 1
            or reset_mask.shape[0] != self.num_envs
            or reset_mask.dtype != torch.bool
            or reset_mask.device != self.device
        ):
            raise ValueError("TensorMotionSampler reset mask has an invalid shape/device")
        if torch_rng is None:
            raise NotImplementedError("TensorMotionSampler requires the Manager Torch generator")
        self.diagnostics.record_reset_dispatch()
        frames: torch.Tensor
        probabilities: torch.Tensor | None = None
        if self.mode == "adaptive":
            probabilities = self._adaptive_probabilities()
            offsets = self._reset_random_offsets
            torch.rand(
                offsets.shape,
                out=offsets,
                dtype=torch.float32,
                device=self.device,
                generator=torch_rng,
            )
            bins = torch.multinomial(
                probabilities, self.num_envs, replacement=True, generator=torch_rng
            )
            frames = self._reset_frames
            frames.copy_(bins)
            scaled = self._reset_scaled_frames
            torch.add(
                frames,
                offsets,
                out=scaled,
            )
            scaled.div_(self.bin_count)
            scaled.mul_(self.num_frames - 1)
            frames.copy_(scaled)
        else:
            use_start = self._reset_start_gate
            start_random = self._reset_start_random
            torch.rand(
                start_random.shape,
                out=start_random,
                dtype=torch.float32,
                device=self.device,
                generator=torch_rng,
            )
            torch.less(start_random, self.start_ratio, out=use_start)
            uniform_frames = self._reset_uniform_frames
            torch.randint(
                0,
                self.num_frames,
                uniform_frames.shape,
                out=uniform_frames,
                dtype=torch.int32,
                generator=torch_rng,
            )
            frames = self._reset_frames
            torch.where(
                use_start,
                torch.zeros_like(uniform_frames),
                uniform_frames,
                out=frames,
            )
        _sampling_dispatch_mask_kernel(
            reset_mask,
            frames,
            self._frame_clip_ends,
            self.current_frames,
            self.current_clip_end_frames,
        )
        if probabilities is not None:
            self._publish_sampling_metrics(probabilities)
        elif self.mode == "mixed":
            self._publish_mixed_metrics()
        self.last_reset_dispatch_ms = (time.perf_counter() - started) * 1000.0
        return frames

    def update_failure_stats(self, terminated: torch.Tensor) -> None:
        if self.mode != "adaptive":
            return
        frames = self._failure_frames
        frames.copy_(self.current_frames)
        frames.mul_(self.bin_count)
        frames.div_(max(self.num_frames, 1), rounding_mode="floor")
        bin_indices = self._failure_bin_indices
        torch.clamp(frames, max=self.bin_count - 1, out=bin_indices)
        discard_bin = self.bin_count
        discard_fill = self._failure_discard_fill
        discard_fill.fill_(discard_bin)
        failed_bins = self._failure_failed_bins
        torch.where(terminated, bin_indices, discard_fill, out=failed_bins)
        self._failure_counts.zero_()
        self._failure_counts.index_add_(
            0,
            failed_bins,
            self._failure_ones,
        )
        # Keep the all-failed/all-active scalar on-device. Converting ``any()``
        # to Python would synchronize at this point in every update-state pass.
        alpha = self._failure_alpha
        torch.any(terminated, out=self._failure_terminated)
        alpha.copy_(self._failure_terminated)
        alpha.mul_(self.adaptive_alpha)
        update = self._failure_update
        torch.mul(self._bin_failed, 1.0 - alpha, out=update)
        update.addcmul_(self._failure_counts[: self.bin_count], alpha)
        self._bin_failed.copy_(update)

    def step_full(self, active: torch.Tensor, time_steps: torch.Tensor) -> torch.Tensor:
        """Advance and publish full-width frame mirrors without row gathers."""
        if (
            active.shape != (self.num_envs,)
            or active.dtype != torch.bool
            or active.device != self.device
        ):
            raise ValueError("TensorMotionSampler active mask shape/device mismatch")
        if (
            time_steps.shape != (self.num_envs,)
            or time_steps.dtype != torch.int32
            or time_steps.device != self.device
        ):
            raise ValueError("TensorMotionSampler time_steps must be int32 device frames")
        next_frames = self._step_next_frames
        torch.add(self.current_frames, active, out=next_frames)
        next_frames.clamp_(max=self.num_frames - 1)
        frames = self._step_clamped_rows
        frames.copy_(next_frames)
        clip_ends = self._step_clip_ends
        torch.index_select(self._frame_clip_ends, 0, frames, out=clip_ends)
        done = self._step_done
        torch.gt(frames, clip_ends, out=done)
        done.logical_and_(active)
        self.current_frames.copy_(next_frames)
        torch.where(
            active,
            clip_ends,
            self.current_clip_end_frames,
            out=self.current_clip_end_frames,
        )
        time_steps.copy_(self.current_frames)
        return done

    def _adaptive_probabilities(self) -> torch.Tensor:
        probabilities = self._adaptive_probability_carrier
        torch.add(
            self._bin_failed,
            self.adaptive_uniform_ratio / self.bin_count,
            out=probabilities,
        )
        if self.adaptive_kernel_size > 1:
            conv_input = self._adaptive_conv_input
            conv_input[:, :, : self.bin_count].copy_(probabilities)
            conv_input[:, :, self.bin_count :].fill_(probabilities[self.bin_count - 1].detach())
            kernel = self._adaptive_kernel[None, None, :]
            torch.ops.aten.conv1d(
                conv_input,
                cast(torch.Tensor, kernel),
                out=self._adaptive_conv_output,
            )
            probabilities = self._adaptive_conv_output[0, 0]
        total = self._adaptive_probability_sum
        torch.sum(probabilities, dim=0, out=total)
        probabilities.div_(total)
        return probabilities

    def _publish_sampling_metrics(self, probabilities: torch.Tensor) -> None:
        if self.bin_count > 1:
            shifted = self._sampling_probability_scratch
            torch.add(probabilities, 1.0e-12, out=shifted)
            logged = self._sampling_log_scratch
            torch.log(shifted, out=logged)
            logged.mul_(probabilities)
            entropy = self._sampling_entropy_scratch
            torch.sum(logged, dim=0, out=entropy)
            entropy.neg_()
            entropy.div_(self._sampling_log_bin_count)
            self.sampling_entropy.copy_(entropy)
        else:
            self.sampling_entropy.fill_(1.0)
        top1_index = self._sampling_top1_index
        torch.argmax(probabilities, out=top1_index)
        self.sampling_top1_prob.copy_(probabilities[top1_index])
        self.sampling_top1_bin.copy_(top1_index)
        self.sampling_top1_bin.div_(self.bin_count)

    def _publish_mixed_metrics(self) -> None:
        start_mass = self.start_ratio + (1.0 - self.start_ratio) / self.bin_count
        self.sampling_top1_prob.fill_(start_mass)
        self.sampling_top1_bin.fill_(0.0)
        if start_mass >= 1.0 - 1.0e-9:
            self.sampling_entropy.fill_(0.0)
            return
        uniform_mass = (1.0 - self.start_ratio) / self.bin_count
        entropy = -start_mass * torch.log(torch.tensor(start_mass + 1.0e-12, device=self.device))
        entropy -= (
            (self.bin_count - 1)
            * uniform_mass
            * torch.log(torch.tensor(uniform_mass + 1.0e-12, device=self.device))
        )
        denominator = torch.log(torch.tensor(float(self.bin_count), device=self.device))
        self.sampling_entropy.copy_(
            entropy / denominator if self.bin_count > 1 else torch.tensor(1.0)
        )


__all__ = ["TensorMotionSampler", "TensorMotionSamplerDiagnostics"]
