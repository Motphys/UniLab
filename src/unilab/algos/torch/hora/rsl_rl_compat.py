"""Compatibility helpers for HORA's supported RSL-RL config schemas.

The HORA APPO code uses these helpers to normalize owner configs before
constructing grouped actor/critic modules across supported RSL-RL releases.
"""

from __future__ import annotations

import importlib.metadata
from copy import deepcopy
from functools import lru_cache
from typing import Any

from packaging.version import Version

_MLX_PPO_ONLY_KEYS = {
    "adaptive_kl_beta",
    "adaptive_lr_decay",
    "adaptive_lr_growth",
    "adaptive_lr_update_interval",
    "disable_finite_checks",
    "enable_compile",
    "finite_check_interval",
    "metrics_interval",
    "target_kl_stop",
    "warmup_finite_check_interval",
    "warmup_metrics_interval",
    "warmup_strict_iters",
}


@lru_cache(maxsize=1)
def get_rsl_rl_version() -> str:
    """Resolve the installed RSL-RL package version.

    Args:
        None.

    Returns:
        Installed version string from either ``rsl-rl-lib`` or the legacy
        ``rsl-rl`` package name.
    """
    try:
        return importlib.metadata.version("rsl-rl-lib")
    except importlib.metadata.PackageNotFoundError:
        try:
            return importlib.metadata.version("rsl-rl")
        except importlib.metadata.PackageNotFoundError as exc:
            raise ImportError(
                "rsl_rl is not installed. Install via: pip install rsl-rl-lib"
            ) from exc


def is_rsl_rl_v4() -> bool:
    """Check whether the active RSL-RL runtime is version 4 or newer.

    Args:
        None.

    Returns:
        ``True`` when the installed package version is ``>= 4.0.0``.
    """
    return bool(Version(get_rsl_rl_version()) >= Version("4.0.0"))


def is_rsl_rl_v5() -> bool:
    """Check whether the active RSL-RL runtime is version 5 or newer.

    Args:
        None.

    Returns:
        ``True`` when the installed package version is ``>= 5.0.0``.
    """
    return bool(Version(get_rsl_rl_version()) >= Version("5.0.0"))


def _normalize_obs_groups_for_rsl(cfg: dict[str, Any]) -> None:
    """Translate UniLab owner obs-group aliases into RSL-RL actor/critic groups.

    Args:
        cfg: Mutable RSL-RL config dictionary to normalize in place.

    Returns:
        None. Updates ``cfg["obs_groups"]`` directly.
    """
    obs_groups_raw = cfg.get("obs_groups", {})
    obs_groups = obs_groups_raw if isinstance(obs_groups_raw, dict) else {}

    if "default" in obs_groups:
        if "actor" not in obs_groups:
            obs_groups["actor"] = obs_groups["default"]
        if "critic" not in obs_groups:
            obs_groups["critic"] = obs_groups["default"]
    else:
        # Keep grouped-dict specs intact for owner runtimes like HORA; the legacy
        # v3 -> v4 rename only applies to flat list-based group aliases.
        if isinstance(obs_groups.get("actor"), list):
            obs_groups["actor"] = ["policy"]
        if isinstance(obs_groups.get("critic"), list):
            obs_groups["critic"] = ["policy"]

    cfg["obs_groups"] = obs_groups


def _convert_policy_to_actor_critic(
    cfg: dict[str, Any],
    *,
    distribution_class_name: str,
) -> None:
    """Split a legacy single ``policy`` config into ``actor`` and ``critic`` blocks.

    Args:
        cfg: Mutable RSL-RL config dictionary to normalize in place.
        distribution_class_name: Distribution class name expected by the target
            RSL-RL runtime.

    Returns:
        None. Updates ``cfg`` directly when a legacy ``policy`` block is present.
    """
    empirical_normalization = bool(cfg.pop("empirical_normalization", False))
    cfg.pop("runner_class_name", None)

    if "policy" not in cfg or "actor" in cfg or "critic" in cfg:
        return

    policy = cfg.pop("policy")
    if not isinstance(policy, dict):
        return

    cfg["actor"] = {
        "class_name": "MLPModel",
        "hidden_dims": policy.get("actor_hidden_dims", [256, 256, 256]),
        "activation": policy.get("activation", "elu"),
        "obs_normalization": empirical_normalization,
        "distribution_cfg": {
            "class_name": distribution_class_name,
            "init_std": policy.get("init_noise_std", 1.0),
            "std_type": policy.get("noise_std_type", "scalar"),
        },
    }
    cfg["critic"] = {
        "class_name": "MLPModel",
        "hidden_dims": policy.get("critic_hidden_dims", [256, 256, 256]),
        "activation": policy.get("activation", "elu"),
        "obs_normalization": empirical_normalization,
    }


def _normalize_algorithm_cfg(cfg: dict[str, Any]) -> None:
    """Remove owner-only keys that current RSL-RL releases do not accept.

    Args:
        cfg: Mutable RSL-RL config dictionary to normalize in place.

    Returns:
        None. Updates ``cfg["algorithm"]`` directly when present.
    """
    algorithm_cfg = cfg.get("algorithm")
    if not isinstance(algorithm_cfg, dict):
        return

    algorithm_cfg.setdefault("rnd_cfg", None)
    for key in _MLX_PPO_ONLY_KEYS:
        algorithm_cfg.pop(key, None)


def convert_config_v3_to_v4(cfg: dict[str, Any]) -> dict[str, Any]:
    """Convert a legacy UniLab PPO/APPO config into the RSL-RL v4 schema.

    Args:
        cfg: Resolved owner config dictionary before RSL-RL construction.

    Returns:
        Deep-copied config dictionary aligned with the RSL-RL v4 actor/critic
        schema and obs-group naming.
    """
    converted = deepcopy(cfg)
    _convert_policy_to_actor_critic(
        converted,
        distribution_class_name="rsl_rl.modules.distribution.GaussianDistribution",
    )
    _normalize_algorithm_cfg(converted)
    _normalize_obs_groups_for_rsl(converted)
    if "multi_gpu" not in converted:
        converted["multi_gpu"] = None
    return converted


def convert_config_v5(cfg: dict[str, Any]) -> dict[str, Any]:
    """Convert a legacy UniLab PPO/APPO config into the RSL-RL v5 schema.

    Args:
        cfg: Resolved owner config dictionary before RSL-RL construction.

    Returns:
        Deep-copied config dictionary aligned with the RSL-RL v5 actor/critic
        schema and obs-group naming.
    """
    converted = deepcopy(cfg)
    _convert_policy_to_actor_critic(
        converted,
        distribution_class_name="GaussianDistribution",
    )
    _normalize_algorithm_cfg(converted)
    _normalize_obs_groups_for_rsl(converted)
    return converted
