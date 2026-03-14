"""Numerical stability utilities for RL training."""

import torch


def check_nan_loss(
    loss: torch.Tensor, default_metrics: dict
) -> tuple[torch.Tensor | None, dict | None]:
    """Check if loss contains NaN or Inf values.

    Args:
        loss: Loss tensor to check
        default_metrics: Default metric values to return if NaN detected

    Returns:
        (loss, None) if valid, (None, nan_metrics) if invalid
    """
    if torch.isnan(loss) or torch.isinf(loss):
        nan_metrics = {k: float("nan") for k in default_metrics}
        return None, nan_metrics
    return loss, None


def clip_gradients(parameters, max_norm: float = 10.0):
    """Clip gradients by global norm.

    Args:
        parameters: Model parameters
        max_norm: Maximum gradient norm
    """
    torch.nn.utils.clip_grad_norm_(parameters, max_norm=max_norm)


def safe_tensor(
    tensor: torch.Tensor, nan_value: float = 0.0, clamp_range: tuple = (-10.0, 10.0)
) -> torch.Tensor:
    """Make tensor numerically safe by clamping and replacing NaN values.

    Args:
        tensor: Input tensor
        nan_value: Value to replace NaN with
        clamp_range: (min, max) range to clamp values

    Returns:
        Safe tensor
    """
    tensor = torch.clamp(tensor, clamp_range[0], clamp_range[1])
    return torch.nan_to_num(tensor, nan=nan_value)
