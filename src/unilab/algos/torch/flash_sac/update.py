"""FlashSAC update helpers."""

from __future__ import annotations

import math

import torch


def build_lr_lambda(
    init_lr: float,
    peak_lr: float,
    end_lr: float,
    warmup_steps: int,
    decay_steps: int,
):
    init_ratio = init_lr / peak_lr if peak_lr > 0 else 1.0
    end_ratio = end_lr / peak_lr if peak_lr > 0 else 1.0
    warmup_steps = max(int(warmup_steps), 0)
    decay_steps = max(int(decay_steps), warmup_steps + 1)

    def schedule(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            progress = float(step) / float(max(warmup_steps, 1))
            return init_ratio + (1.0 - init_ratio) * progress
        progress = min(max((step - warmup_steps) / float(decay_steps - warmup_steps), 0.0), 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return end_ratio + (1.0 - end_ratio) * cosine

    return schedule


def select_min_q_log_probs(
    next_q_values: torch.Tensor,
    next_q_log_probs: torch.Tensor,
) -> torch.Tensor:
    num_bins = next_q_log_probs.shape[-1]
    min_indices = next_q_values.argmin(dim=0)
    gather_index = min_indices[None, :, None].expand(1, -1, num_bins)
    return torch.gather(next_q_log_probs, dim=0, index=gather_index)[0]


def compute_categorical_td_target(
    support: torch.Tensor,
    target_log_probs: torch.Tensor,
    reward: torch.Tensor,
    terminated: torch.Tensor,
    truncated: torch.Tensor,
    actor_entropy: torch.Tensor,
    gamma: float,
) -> torch.Tensor:
    batch_size, num_bins = target_log_probs.shape
    support = support.view(1, -1)
    reward = reward.view(-1, 1)
    terminated = terminated.view(-1, 1)
    truncated = truncated.view(-1, 1)
    actor_entropy = actor_entropy.view(-1, 1)

    bootstrap = torch.clamp(1.0 - terminated + truncated, 0.0, 1.0)
    target_bin_values = reward + bootstrap * gamma * (support - actor_entropy)
    target_bin_values = torch.clamp(target_bin_values, float(support.min()), float(support.max()))

    bin_width = float(support[0, 1] - support[0, 0])
    offsets = (target_bin_values - float(support.min())) / max(bin_width, 1e-8)
    lower = torch.floor(offsets).long().clamp(0, num_bins - 1)
    upper = torch.ceil(offsets).long().clamp(0, num_bins - 1)
    frac = offsets - lower.float()

    probs = target_log_probs.exp()
    target_probs = torch.zeros(batch_size, num_bins, dtype=probs.dtype, device=probs.device)
    target_probs.scatter_add_(1, lower, probs * (1.0 - frac))
    target_probs.scatter_add_(1, upper, probs * frac)
    return target_probs


def resolve_target_entropy(
    action_dim: int,
    target_sigma: float,
    target_entropy: float | None,
) -> float:
    if target_entropy is not None:
        return float(target_entropy)
    sigma = max(float(target_sigma), 1e-6)
    per_dim_entropy = 0.5 * math.log(2.0 * math.pi * math.e * sigma * sigma)
    return float(action_dim) * per_dim_entropy
