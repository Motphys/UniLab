"""HORA-owned observation helpers for teacher-policy runtime code."""

from __future__ import annotations

from typing import Any

import numpy as np
from tensordict import TensorDict

from unilab.utils.tensor import to_torch


def split_hora_obs_with_priv_info(
    obs: dict[str, np.ndarray],
    info: dict[str, Any] | None = None,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
    """Split HORA env outputs into actor obs, critic obs, and privileged info.

    Args:
        obs: Environment observation dict following the UniLab env contract.
        info: Optional env info dict. When present, ``info["critic_info"]`` is the
            preferred source of HORA privileged info.

    Returns:
        Tuple ``(actor_obs, critic_obs, priv_info)``. ``priv_info`` falls back to the
        extra tail of ``critic_obs`` when no explicit ``critic_info`` is provided.
    """
    actor_obs = obs["obs"]
    critic_obs = obs.get("critic", actor_obs)

    priv_info: np.ndarray | None = None
    if isinstance(info, dict):
        candidate = info.get("critic_info")
        if isinstance(candidate, np.ndarray) and candidate.shape[0] == actor_obs.shape[0]:
            priv_info = candidate

    if (
        priv_info is None
        and critic_obs is not None
        and critic_obs.ndim == 2
        and actor_obs.ndim == 2
        and critic_obs.shape[0] == actor_obs.shape[0]
        and critic_obs.shape[1] > actor_obs.shape[1]
    ):
        priv_info = critic_obs[:, actor_obs.shape[1] :]

    return actor_obs, critic_obs, priv_info


def extract_hora_proprio_hist(info: dict[str, Any] | None) -> np.ndarray | None:
    """Return HORA proprio-history payload from env info when available.

    Args:
        info: Optional env info dict produced by the HORA environment.

    Returns:
        Proprio-history array when present, otherwise ``None``.
    """
    if not isinstance(info, dict):
        return None
    proprio_hist = info.get("proprio_hist")
    return proprio_hist if isinstance(proprio_hist, np.ndarray) else None


def build_hora_obs_tensordict(
    obs: dict[str, np.ndarray],
    *,
    info: dict[str, Any] | None,
    device: str,
    batch_size: int,
    policy_obs: np.ndarray,
) -> TensorDict:
    """Build the HORA PPO/APPO observation TensorDict for teacher-policy runtime.

    Args:
        obs: Environment observation dict following the UniLab env contract.
        info: Optional env info dict containing HORA privileged payloads.
        device: Torch device string used for the returned tensors.
        batch_size: Number of vectorized environments represented by this batch.
        policy_obs: Policy observation array already resolved by the caller.

    Returns:
        TensorDict with generic keys plus HORA-specific ``priv_info`` and optional
        ``proprio_hist`` when the environment provided them.
    """
    actor_obs_np, critic_obs_np, priv_info_np = split_hora_obs_with_priv_info(obs, info)
    td_dict = {
        "actor": to_torch(actor_obs_np, device),
        "policy": to_torch(policy_obs, device),
    }
    if critic_obs_np is not None:
        td_dict["critic"] = to_torch(critic_obs_np, device)
    if priv_info_np is not None:
        td_dict["priv_info"] = to_torch(priv_info_np, device)
    proprio_hist = extract_hora_proprio_hist(info)
    if proprio_hist is not None:
        td_dict["proprio_hist"] = to_torch(proprio_hist, device)
    return TensorDict(td_dict, batch_size=batch_size, device=device)


def build_hora_actor_tensordict(
    actor_obs: np.ndarray,
    *,
    priv_info: np.ndarray,
    device: str,
    batch_size: int,
) -> TensorDict:
    """Build the minimal HORA actor TensorDict for APPO play/inference.

    Args:
        actor_obs: Actor observation array with shape ``(batch, obs_dim)``.
        priv_info: Privileged-info array with shape ``(batch, priv_dim)``.
        device: Torch device string used for the returned tensors.
        batch_size: Number of vectorized environments represented by this batch.

    Returns:
        TensorDict containing grouped HORA actor inputs required by teacher-policy
        inference.
    """
    return TensorDict(
        {
            "actor": to_torch(actor_obs, device),
            "priv_info": to_torch(priv_info, device),
        },
        batch_size=batch_size,
        device=device,
    )
