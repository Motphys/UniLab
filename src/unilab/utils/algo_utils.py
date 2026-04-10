"""Common utilities for RL algorithms."""

from __future__ import annotations

import importlib
import logging
from typing import Sequence

logger = logging.getLogger(__name__)

# Attribute name for package-level registry bootstrap contracts.
_REGISTRY_MODULES_ATTR = "__unilab_registry_modules__"

# Default packages to import for env registration bootstrap.
_DEFAULT_REGISTRY_PACKAGES = (
    "unilab.envs.locomotion",
    "unilab.envs.manipulation",
    "unilab.envs.motion_tracking",
)


def ensure_registries(
    packages: Sequence[str] | None = None,
    *,
    optional_packages: Sequence[str] | None = None,
    fail_on_error: bool = True,
) -> None:
    """Import env registry bootstrap modules.

    Args:
        packages: Package or module names to import for env registration.
                  Package-level registry modules should declare
                  ``__unilab_registry_modules__`` as explicit bootstrap targets.
                  Defaults to standard unilab env registry packages.
        optional_packages: List of optional package names that may not be present.
                          Import failures for these are logged as warnings, not raised.
        fail_on_error: If True (default), raise exceptions for non-optional packages.
                      If False, log warnings instead of raising.

    Raises:
        ImportError: If a non-optional package fails to import and fail_on_error is True.
        RuntimeError: If a declared registry module fails to import and fail_on_error is True.
        TypeError: If ``__unilab_registry_modules__`` has an invalid format.
    """
    pkgs = list(packages) if packages is not None else list(_DEFAULT_REGISTRY_PACKAGES)
    optional = set(optional_packages) if optional_packages else set()

    for pkg_name in pkgs:
        is_optional = pkg_name in optional
        try:
            package = importlib.import_module(pkg_name)
        except ImportError as e:
            if is_optional:
                logger.warning("Optional registry package not found: %s (%s)", pkg_name, e)
            elif fail_on_error:
                raise ImportError(
                    f"Failed to import registry package '{pkg_name}'. "
                    f"Add to optional_packages if this is expected to be absent."
                ) from e
            else:
                logger.warning("Registry package not found: %s (%s)", pkg_name, e)
            continue

        modules = getattr(package, _REGISTRY_MODULES_ATTR, ())
        if isinstance(modules, str) or not isinstance(modules, Sequence):
            raise TypeError(
                f"'{pkg_name}.{_REGISTRY_MODULES_ATTR}' must be a sequence of module names."
            )

        for name in modules:
            if not isinstance(name, str) or not name:
                raise TypeError(
                    f"'{pkg_name}.{_REGISTRY_MODULES_ATTR}' entries must be non-empty strings."
                )
            try:
                importlib.import_module(name)
            except Exception as e:
                if fail_on_error and not is_optional:
                    raise RuntimeError(
                        f"Failed to import declared registry module '{name}' from '{pkg_name}'. "
                        f"Fix the import error or add '{pkg_name}' to optional_packages."
                    ) from e
                logger.warning("Failed to import declared registry module '%s': %s", name, e)


def build_actor(
    algo_type, obs_dim, action_dim, actor_hidden_dim, use_layer_norm, device, num_envs=1
):
    """Build the correct actor model based on algorithm type."""
    if algo_type == "sac":
        from unilab.algos.torch.fast_sac.learner import SACActor

        return SACActor(
            obs_dim=obs_dim,
            action_dim=action_dim,
            hidden_dim=actor_hidden_dim,
            use_layer_norm=use_layer_norm,
            device=device,
        )
    elif algo_type == "td3":
        from unilab.algos.torch.fast_td3.learner import TD3Actor

        return TD3Actor(
            obs_dim=obs_dim,
            n_act=action_dim,
            num_envs=num_envs,
            hidden_dim=actor_hidden_dim,
            init_scale=0.01,
            log_std_min=-0.9,
            log_std_max=0.0,
            device=device,
        )
    else:
        raise ValueError(f"Unknown algo_type: {algo_type}")
