"""
Compatibility utilities for supporting both rsl_rl 3.x and 4.x.

rsl_rl 4.0 introduced breaking API changes:
  - Config format: single `policy` dict → separate `actor`/`critic` dicts
  - `construct_algorithm` moved from Runner to Algorithm (PPO)
  - `rnd_cfg` must exist in algorithm config (can be None)
  - `empirical_normalization` deprecated

This module provides runtime version detection and config conversion so that
the codebase can work with both versions without code duplication.
"""

import importlib.metadata
from copy import deepcopy

from packaging.version import Version


def get_rsl_rl_version() -> str:
    """Get the installed rsl_rl version string."""
    try:
        return importlib.metadata.version("rsl-rl-lib")
    except importlib.metadata.PackageNotFoundError:
        # Fallback: try the old package name
        try:
            return importlib.metadata.version("rsl-rl")
        except importlib.metadata.PackageNotFoundError:
            raise ImportError("rsl_rl is not installed. Install via: pip install rsl-rl-lib")


def is_rsl_rl_v4() -> bool:
    """Check if the installed rsl_rl version is 4.x or above."""
    version_str = get_rsl_rl_version()
    print(version_str)
    return bool(Version(version_str) >= Version("4.0.0"))


def is_rsl_rl_v5() -> bool:
    """Check if the installed rsl_rl version is 4.x or above."""
    version_str = get_rsl_rl_version()
    print(version_str)
    return bool(Version(version_str) >= Version("5.0.0"))


def convert_config_v3_to_v4(cfg: dict) -> dict:
    """Convert rsl_rl 3.x config format to 4.x format.

    3.x uses a single `policy` dict with class_name="ActorCritic":
        policy:
            class_name: "ActorCritic"
            actor_hidden_dims: [...]
            critic_hidden_dims: [...]
            activation: "elu"
            init_noise_std: 1.0

    4.x uses separate `actor` and `critic` dicts with class_name="MLPModel":
        actor:
            class_name: "MLPModel"
            hidden_dims: [...]
            activation: "elu"
            init_noise_std: 1.0
        critic:
            class_name: "MLPModel"
            hidden_dims: [...]
            activation: "elu"
    """
    cfg = deepcopy(cfg)

    # Remove deprecated fields, but capture value for migration first
    empirical_normalization = cfg.pop("empirical_normalization", False)
    cfg.pop("runner_class_name", None)

    # Convert policy → actor + critic unless explicit custom actor/critic configs are already present.
    if "policy" in cfg and ("actor" not in cfg or "critic" not in cfg):
        policy = cfg.pop("policy")
        cfg["actor"] = {
            "class_name": "MLPModel",
            "hidden_dims": policy.get("actor_hidden_dims", [256, 256, 256]),
            "activation": policy.get("activation", "elu"),
            "obs_normalization": empirical_normalization,
            "distribution_cfg": {
                "class_name": "rsl_rl.modules.distribution.GaussianDistribution",
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

    # 4.x requires rnd_cfg in algorithm config (can be None)
    if "algorithm" in cfg:
        cfg["algorithm"].setdefault("rnd_cfg", None)
        # Remove class_name for algorithm - it's popped by construct_algorithm
        # but needs to stay for the runner to resolve it, so leave it alone

        # Strip mlx_ppo-only parameters that rsl_rl PPO does not accept
        _MLX_PPO_ONLY_KEYS = {
            "adaptive_kl_beta",
            "adaptive_lr_decay",
            "adaptive_lr_growth",
            "adaptive_lr_update_interval",
            "target_kl_stop",
            "metrics_interval",
            "finite_check_interval",
            "enable_compile",
            "warmup_strict_iters",
            "warmup_metrics_interval",
            "warmup_finite_check_interval",
            "disable_finite_checks",
        }
        for key in _MLX_PPO_ONLY_KEYS:
            cfg["algorithm"].pop(key, None)

    # 4.x requires obs_groups
    obs_groups = cfg.get("obs_groups", {})
    if "default" in obs_groups:
        if "actor" not in obs_groups:
            obs_groups["actor"] = obs_groups["default"]
        if "critic" not in obs_groups:
            obs_groups["critic"] = obs_groups["default"]
    else:
        # Fallback if no groups defined at all (from V3 policy)
        if "actor" not in obs_groups:
            obs_groups["actor"] = ["policy"]
        if "critic" not in obs_groups:
            obs_groups["critic"] = ["policy"]

    cfg["obs_groups"] = obs_groups

    # 4.x requires multi_gpu config
    if "multi_gpu" not in cfg:
        cfg["multi_gpu"] = None

    return cfg


def convert_config_v5(cfg: dict) -> dict:
    """Convert config format to 5.x format.


    5.x uses separate `actor` and `critic` dicts with class_name="MLPModel":
        actor:
            class_name: "MLPModel"
            hidden_dims: [...]
            activation: "elu"
            distribution_cfg: { 'class_name':
                                'init_std':
                                'std_type':
                                }
        critic:
            class_name: "MLPModel"
            hidden_dims: [...]
            activation: "elu"
    """
    cfg = deepcopy(cfg)

    # Remove deprecated fields, but capture value for migration first
    empirical_normalization = cfg.pop("empirical_normalization", False)
    cfg.pop("runner_class_name", None)

    # Convert policy → actor + critic unless explicit custom actor/critic configs are already present.
    if "policy" in cfg and ("actor" not in cfg or "critic" not in cfg):
        policy = cfg.pop("policy")
        cfg["actor"] = {
            "class_name": "MLPModel",
            "hidden_dims": policy.get("actor_hidden_dims", [256, 256, 256]),
            "activation": policy.get("activation", "elu"),
            # "init_noise_std": policy.get("init_noise_std", 1.0),
            # "noise_std_type": policy.get("noise_std_type", "scalar"),
            # "stochastic": True,  # Required: MLPModel needs this to create output distribution
            "obs_normalization": empirical_normalization,
            "distribution_cfg": {
                "class_name": "GaussianDistribution",
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

    # 4.x requires rnd_cfg in algorithm config (can be None)
    if "algorithm" in cfg:
        cfg["algorithm"].setdefault("rnd_cfg", None)
        # Remove class_name for algorithm - it's popped by construct_algorithm
        # but needs to stay for the runner to resolve it, so leave it alone

        # Strip mlx_ppo-only parameters that rsl_rl PPO does not accept
        _MLX_PPO_ONLY_KEYS = {
            "adaptive_kl_beta",
            "adaptive_lr_decay",
            "adaptive_lr_growth",
            "adaptive_lr_update_interval",
            "target_kl_stop",
            "metrics_interval",
            "finite_check_interval",
            "enable_compile",
            "warmup_strict_iters",
            "warmup_metrics_interval",
            "warmup_finite_check_interval",
            "disable_finite_checks",
        }
        for key in _MLX_PPO_ONLY_KEYS:
            cfg["algorithm"].pop(key, None)

    # 4.x requires obs_groups
    obs_groups = cfg.get("obs_groups", {})
    if "default" in obs_groups:
        if "actor" not in obs_groups:
            obs_groups["actor"] = obs_groups["default"]
        if "critic" not in obs_groups:
            obs_groups["critic"] = obs_groups["default"]
    else:
        # Fallback if no groups defined at all (from V3 policy)
        if "actor" not in obs_groups:
            obs_groups["actor"] = ["policy"]
        if "critic" not in obs_groups:
            obs_groups["critic"] = ["policy"]

    cfg["obs_groups"] = obs_groups

    return cfg
