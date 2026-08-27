import dataclasses
import importlib
import logging
import os
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import (
    Any,
    Callable,
    Dict,
    Literal,
    Optional,
    Protocol,
    TypeVar,
)

from .base import ABEnv, EnvCfg
from .config_materialization import apply_cfg_overrides
from .config_overrides import (
    CONFIG_MAPPING_POLICY_KEY,
    MANAGER_TERM_MAPPING_POLICY,
)

EnvCfgFactory = Callable[[], EnvCfg]
TEnvCfgFactory = TypeVar("TEnvCfgFactory", bound=EnvCfgFactory)


class EnvFactory(Protocol):
    """Construct an environment for one materialized config and backend."""

    def __call__(
        self,
        cfg: Any,
        *,
        num_envs: int = 1,
        backend_type: str = "mujoco",
    ) -> ABEnv: ...


TEnvFactory = TypeVar("TEnvFactory", bound=EnvFactory)
RewardOverrideField = Literal["reward_config", "rewards"]
_SUPPORTED_SIM_BACKENDS = ("mujoco", "mjwarp", "motrix", "drake", "isaacgym")
_DEFAULT_SIM_BACKEND_ORDER: tuple[str, ...] = ("mujoco", "motrix")
_REGISTRY_MODULES_ATTR = "__unilab_registry_modules__"
_DEFAULT_REGISTRY_PACKAGES = ("unilab.tasks",)
# Environment variable used to extend ensure_registries() with extra packages.
# Mainly intended for test setups that need to ship a fixture-only registry into
# spawn subprocesses (which do not inherit pytest conftest state).
_EXTRA_REGISTRY_PACKAGES_ENV = "UNILAB_EXTRA_REGISTRY_PACKAGES"

logger = logging.getLogger(__name__)


@dataclass
class EnvMeta:
    env_cfg_factory: EnvCfgFactory
    env_factory_dict: Dict[str, EnvFactory] = field(default_factory=dict)

    def available_sim_backend(self) -> Optional[str]:
        """Return the explicit default simulation backend for this environment."""
        for backend in _DEFAULT_SIM_BACKEND_ORDER:
            if backend in self.env_factory_dict:
                return backend
        return next(iter(self.env_factory_dict), None)

    def support_sim_backend(self, sim_backend: str) -> bool:
        """Check if the environment supports a specific simulation backend."""
        return sim_backend in self.env_factory_dict


_envs: Dict[str, EnvMeta] = {}


def contains(name: str) -> bool:
    """Check if an environment configuration is registered."""
    return name in _envs


def _config_factory_name(factory: EnvCfgFactory) -> str:
    return str(getattr(factory, "__qualname__", type(factory).__qualname__))


def _env_factory_name(factory: EnvFactory) -> str:
    return str(getattr(factory, "__qualname__", type(factory).__qualname__))


def register_env_config(name: str, env_cfg_factory: EnvCfgFactory) -> None:
    """Register a zero-argument environment configuration factory."""
    if name in _envs.keys():
        raise ValueError(f"Environment '{name}' is already registered.")
    if not callable(env_cfg_factory):
        raise TypeError(
            f"Environment '{name}' config factory must be callable, got "
            f"{type(env_cfg_factory).__name__}"
        )
    _envs[name] = EnvMeta(env_cfg_factory=env_cfg_factory)


def envcfg(name: str) -> Callable[[TEnvCfgFactory], TEnvCfgFactory]:
    """
    Decorator to register an environment configuration class or factory.

    Usage:
        @envcfg("my-env")
        @dataclass
        class MyEnvCfg(EnvCfg):
            ...

        @envcfg("my-manager-env")
        def make_my_env_cfg() -> EnvCfg:
            ...
    """

    def decorator(factory: TEnvCfgFactory) -> TEnvCfgFactory:
        register_env_config(name, factory)
        return factory

    return decorator


def materialize_env_config(name: str) -> EnvCfg:
    """Construct one config instance from the registered cold-path factory."""
    if name not in _envs:
        raise ValueError(f"Environment '{name}' is not registered.")

    factory = _envs[name].env_cfg_factory
    env_cfg = factory()
    if not isinstance(env_cfg, EnvCfg):
        raise TypeError(
            f"Environment '{name}' config factory '{_config_factory_name(factory)}' returned "
            f"{type(env_cfg).__name__}, expected an EnvCfg instance"
        )
    return env_cfg


def register_env(name: str, env_factory: TEnvFactory, sim_backend: str) -> TEnvFactory:
    """Register and return an environment class or function factory."""
    if sim_backend not in _SUPPORTED_SIM_BACKENDS:
        raise ValueError(
            f"Unsupported simulation backend: {sim_backend}. "
            f"Supported backends: {', '.join(_SUPPORTED_SIM_BACKENDS)}."
        )

    if name not in _envs:
        raise ValueError(
            f"Environment '{name}' is not registered. Please register the config first."
        )

    if not callable(env_factory):
        raise TypeError(
            f"Environment '{name}' backend '{sim_backend}' factory must be callable, got "
            f"{type(env_factory).__name__}"
        )

    if sim_backend in _envs[name].env_factory_dict:
        raise ValueError(
            f"Environment '{name}' with sim backend '{sim_backend}' is already registered."
        )

    _envs[name].env_factory_dict[sim_backend] = env_factory
    return env_factory


def env(name: str, sim_backend: str) -> Callable[[TEnvFactory], TEnvFactory]:
    """
    Decorator to register an environment class or function factory.

    Usage:
        @env("my-env", "mujoco")
        class MyEnv(ABEnv):
            ...

        @env("my-manager-env", "mujoco")
        def make_my_env(cfg, num_envs=1, backend_type="mujoco"):
            ...
    """

    def decorator(factory: TEnvFactory) -> TEnvFactory:
        return register_env(name, factory, sim_backend)

    return decorator


def find_available_sim_backend(env_name: str) -> str:
    """Find the explicit default simulation backend for an environment."""
    if env_name not in _envs:
        raise ValueError(f"Environment '{env_name}' is not registered.")

    meta: EnvMeta = _envs[env_name]
    backend = meta.available_sim_backend()
    if backend is None:
        raise ValueError(f"Environment '{env_name}' does not support any simulation backend.")
    return backend


def resolve_reward_override_field(env_name: str) -> RewardOverrideField:
    """Resolve the Hydra root reward target declared by an env config owner.

    Legacy configs own a ``reward_config`` field. Manager-Based configs opt in
    through the explicit manager-term mapping metadata on ``rewards``. The
    registry resolves this on the config class without constructing an env or
    backend so training adapters do not branch on task names.
    """
    if env_name not in _envs:
        raise ValueError(f"Environment '{env_name}' is not registered.")

    config = materialize_env_config(env_name)
    config_fields = {config_field.name: config_field for config_field in dataclasses.fields(config)}
    rewards_field = config_fields.get("rewards")
    has_manager_rewards = (
        rewards_field is not None
        and rewards_field.metadata.get(CONFIG_MAPPING_POLICY_KEY) == MANAGER_TERM_MAPPING_POLICY
    )
    has_legacy_rewards = "reward_config" in config_fields
    config_owner = type(config).__name__

    if has_manager_rewards and has_legacy_rewards:
        raise ValueError(
            f"Environment '{env_name}' config owner '{config_owner}' declares both "
            "Manager-Based 'rewards' and legacy 'reward_config' targets"
        )
    if has_manager_rewards:
        return "rewards"
    if has_legacy_rewards:
        return "reward_config"
    raise ValueError(
        f"Environment '{env_name}' config owner '{config_owner}' declares no "
        "supported Hydra root reward target; expected legacy 'reward_config' or an "
        "explicitly marked Manager-Based 'rewards' field"
    )


def make(
    name: str,
    sim_backend: Optional[str] = None,
    env_cfg_override: Optional[Dict[str, Any]] = None,
    num_envs: int = 1,
) -> ABEnv:
    """
    Create an environment instance by name.

    Args:
        name: Environment name
        sim_backend: Simulation backend. If None, uses the
            explicit default backend order: "mujoco", then "motrix".
        num_envs: Number of environments to create

    Returns:
        Environment instance
    """
    if name not in _envs:
        raise ValueError(f"Environment '{name}' is not registered.")

    meta: EnvMeta = _envs[name]

    # Create environment config
    env_cfg = materialize_env_config(name)
    if env_cfg_override is not None:
        apply_cfg_overrides(env_cfg, env_cfg_override)

    # Validate config
    env_cfg.validate()

    # Select simulation backend
    if sim_backend is None:
        sim_backend = meta.available_sim_backend()
        if sim_backend is None:
            raise ValueError(f"Environment '{name}' does not support any simulation backend.")

    if not meta.support_sim_backend(sim_backend):
        raise ValueError(
            f"Environment '{name}' does not support simulation backend '{sim_backend}'."
        )

    # Create environment instance
    factory = meta.env_factory_dict[sim_backend]
    env = factory(env_cfg, num_envs=num_envs, backend_type=sim_backend)
    if not isinstance(env, ABEnv):
        raise TypeError(
            f"Environment '{name}' backend '{sim_backend}' factory "
            f"'{_env_factory_name(factory)}' returned {type(env).__name__}, "
            "expected an ABEnv instance"
        )
    return env


def list_registered_envs() -> Dict[str, Dict[str, Any]]:
    """List all registered environments with their available backends."""
    result = {}
    for name, meta in _envs.items():
        result[name] = {
            "config_factory": _config_factory_name(meta.env_cfg_factory),
            "available_backends": list(meta.env_factory_dict.keys()),
        }
    return result


def ensure_registries(
    packages: Sequence[str] | None = None,
    *,
    optional_packages: Sequence[str] | None = None,
    fail_on_error: bool = True,
) -> None:
    """Import env registry bootstrap modules."""
    package_names: list[str] = (
        list(packages) if packages is not None else list(_DEFAULT_REGISTRY_PACKAGES)
    )
    optional = set(optional_packages) if optional_packages else set()

    # Allow extending the default registry packages via env var. This is the
    # only seam that lets a pytest conftest inject test-only envs (e.g.
    # DummyFlatTest) into spawn-based collector subprocesses, which start as
    # fresh interpreters and therefore never execute conftest.py.
    extra_env = os.environ.get(_EXTRA_REGISTRY_PACKAGES_ENV, "").strip()
    if extra_env:
        for extra in extra_env.split(","):
            extra = extra.strip()
            if extra and extra not in package_names:
                package_names.append(extra)
                # Treat env-var-provided packages as optional: a missing import
                # must never break a production training run that happens to
                # have the env var leaked from a parent shell.
                optional.add(extra)

    for package_name in package_names:
        is_optional = package_name in optional
        try:
            package = importlib.import_module(package_name)
        except ImportError as exc:
            if is_optional:
                logging.warning("Optional registry package not found: %s (%s)", package_name, exc)
            elif fail_on_error:
                raise ImportError(
                    f"Failed to import registry package '{package_name}'. "
                    f"Add to optional_packages if this is expected to be absent."
                ) from exc
            else:
                logging.warning("Registry package not found: %s (%s)", package_name, exc)
            continue

        modules = getattr(package, _REGISTRY_MODULES_ATTR, ())
        if isinstance(modules, str) or not isinstance(modules, Sequence):
            raise TypeError(
                f"'{package_name}.{_REGISTRY_MODULES_ATTR}' must be a sequence of module names."
            )

        for module_name in modules:
            if not isinstance(module_name, str) or not module_name:
                raise TypeError(
                    f"'{package_name}.{_REGISTRY_MODULES_ATTR}' entries must be non-empty strings."
                )
            try:
                importlib.import_module(module_name)
            except Exception as exc:
                if fail_on_error and not is_optional:
                    raise RuntimeError(
                        f"Failed to import declared registry module '{module_name}' "
                        f"from '{package_name}'. "
                        f"Fix the import error or add '{package_name}' to optional_packages."
                    ) from exc
                logging.warning(
                    "Failed to import declared registry module '%s': %s", module_name, exc
                )
