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
            raise ImportError(
                "rsl_rl is not installed. Install via: pip install rsl-rl-lib"
            )


def is_rsl_rl_v4() -> bool:
    """Check if the installed rsl_rl version is 4.x or above."""
    version_str = get_rsl_rl_version()
    return Version(version_str) >= Version("4.0.0")


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

    # Convert policy → actor + critic
    if "policy" in cfg:
        policy = cfg.pop("policy")
        cfg["actor"] = {
            "class_name": "MLPModel",
            "hidden_dims": policy.get("actor_hidden_dims", [256, 256, 256]),
            "activation": policy.get("activation", "elu"),
            "init_noise_std": policy.get("init_noise_std", 1.0),
            "stochastic": True,  # Required: MLPModel needs this to create output distribution
        }
        cfg["critic"] = {
            "class_name": "MLPModel",
            "hidden_dims": policy.get("critic_hidden_dims", [256, 256, 256]),
            "activation": policy.get("activation", "elu"),
        }

    # 4.x requires rnd_cfg in algorithm config (can be None)
    if "algorithm" in cfg:
        cfg["algorithm"].setdefault("rnd_cfg", None)
        # Remove class_name for algorithm - it's popped by construct_algorithm
        # but needs to stay for the runner to resolve it, so leave it alone

    # 4.x requires obs_groups
    cfg.setdefault("obs_groups", {"default": ["policy"]})

    # Remove deprecated fields
    cfg.pop("empirical_normalization", None)
    cfg.pop("runner_class_name", None)

    return cfg
