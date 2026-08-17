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
    Type,
    TypeVar,
    cast,
    get_args,
    get_origin,
    get_type_hints,
)

from .base import ABEnv, EnvCfg
from .config_overrides import (
    CONFIG_MAPPING_POLICY_KEY,
    MANAGER_PARAMS_MAPPING_POLICY,
    MANAGER_TERM_MAPPING_POLICY,
)

TEnvCfg = TypeVar("TEnvCfg", bound=EnvCfg)
RewardOverrideField = Literal["reward_config", "rewards"]
_SUPPORTED_SIM_BACKENDS = ("mujoco", "mjwarp", "motrix", "drake")
_DEFAULT_SIM_BACKEND_ORDER: tuple[str, ...] = ("mujoco", "motrix")
_REGISTRY_MODULES_ATTR = "__unilab_registry_modules__"
_DEFAULT_REGISTRY_PACKAGES = (
    "unilab.envs.locomotion",
    "unilab.envs.manipulation",
    "unilab.envs.motion_tracking",
)
# Environment variable used to extend ensure_registries() with extra packages.
# Mainly intended for test setups that need to ship a fixture-only registry into
# spawn subprocesses (which do not inherit pytest conftest state).
_EXTRA_REGISTRY_PACKAGES_ENV = "UNILAB_EXTRA_REGISTRY_PACKAGES"

logger = logging.getLogger(__name__)


@dataclass
class EnvMeta:
    env_cfg_cls: Type[EnvCfg]
    env_cls_dict: Dict[str, Type[ABEnv]] = field(default_factory=dict)

    def available_sim_backend(self) -> Optional[str]:
        """Return the explicit default simulation backend for this environment."""
        for backend in _DEFAULT_SIM_BACKEND_ORDER:
            if backend in self.env_cls_dict:
                return backend
        return next(iter(self.env_cls_dict), None)

    def support_sim_backend(self, sim_backend: str) -> bool:
        """Check if the environment supports a specific simulation backend."""
        return sim_backend in self.env_cls_dict


_envs: Dict[str, EnvMeta] = {}


def contains(name: str) -> bool:
    """Check if an environment configuration is registered."""
    return name in _envs


def register_env_config(name: str, env_cfg_cls: Type[EnvCfg]):
    """Register an environment configuration class with a name."""
    if name in _envs.keys():
        raise ValueError(f"Environment '{name}' is already registered.")
    _envs[name] = EnvMeta(env_cfg_cls=env_cfg_cls)


def envcfg(name: str) -> Callable[[Type[TEnvCfg]], Type[TEnvCfg]]:
    """
    Decorator to register an environment configuration class with a name.

    Usage:
        @register_env_config_decorator("my-env")
        @dataclass
        class MyEnvCfg(EnvCfg):
            ...
    """

    def decorator(cls: Type[TEnvCfg]) -> Type[TEnvCfg]:
        register_env_config(name, cls)
        return cls

    return decorator


def register_env(name: str, env_cls: Type[ABEnv], sim_backend: str):
    """Register an environment class with a name and simulation backend."""
    if sim_backend not in _SUPPORTED_SIM_BACKENDS:
        raise ValueError(
            f"Unsupported simulation backend: {sim_backend}. "
            f"Supported backends: {', '.join(_SUPPORTED_SIM_BACKENDS)}."
        )

    if name not in _envs:
        raise ValueError(
            f"Environment '{name}' is not registered. Please register the config first."
        )

    if sim_backend in _envs[name].env_cls_dict:
        raise ValueError(
            f"Environment '{name}' with sim backend '{sim_backend}' is already registered."
        )

    _envs[name].env_cls_dict[sim_backend] = env_cls


def env(name: str, sim_backend: str) -> Callable[[Type[ABEnv]], Type[ABEnv]]:
    """
    Decorator to register an environment class with a name and simulation backend.

    Usage:
        @register_env_decorator("my-env", "np")
        class MyEnv(ABEnv):
            ...
    """

    def decorator(cls: Type[ABEnv]) -> Type[ABEnv]:
        register_env(name, cls, sim_backend)
        return cls

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

    config_cls = _envs[env_name].env_cfg_cls
    config_fields = {
        config_field.name: config_field for config_field in dataclasses.fields(config_cls)
    }
    rewards_field = config_fields.get("rewards")
    has_manager_rewards = (
        rewards_field is not None
        and rewards_field.metadata.get(CONFIG_MAPPING_POLICY_KEY) == MANAGER_TERM_MAPPING_POLICY
    )
    has_legacy_rewards = "reward_config" in config_fields

    if has_manager_rewards and has_legacy_rewards:
        raise ValueError(
            f"Environment '{env_name}' config owner '{config_cls.__name__}' declares both "
            "Manager-Based 'rewards' and legacy 'reward_config' targets"
        )
    if has_manager_rewards:
        return "rewards"
    if has_legacy_rewards:
        return "reward_config"
    raise ValueError(
        f"Environment '{env_name}' config owner '{config_cls.__name__}' declares no "
        "supported Hydra root reward target; expected legacy 'reward_config' or an "
        "explicitly marked Manager-Based 'rewards' field"
    )


def _resolve_dataclass_type(type_hint: Any) -> Optional[Type[Any]]:
    """Strip Optional/Union and return the underlying dataclass type, or None."""
    if type_hint is None:
        return None
    origin = get_origin(type_hint)
    if origin is not None:
        args = get_args(type_hint)
        type_hint = next((arg for arg in args if arg is not type(None)), None)
    if (
        type_hint is not None
        and dataclasses.is_dataclass(type_hint)
        and isinstance(type_hint, type)
    ):
        return cast(Type[Any], type_hint)
    return None


def _construct_dataclass_from_dict(target_type: Type[Any], values: Dict[str, Any]) -> Any:
    try:
        target_obj = target_type()
    except TypeError:
        return target_type(**values)
    apply_cfg_overrides(target_obj, values)
    return target_obj


def _config_mapping_policy(target_obj: Any, field_name: str) -> str | None:
    if not dataclasses.is_dataclass(target_obj) or isinstance(target_obj, type):
        return None
    for config_field in dataclasses.fields(target_obj):
        if config_field.name == field_name:
            policy = config_field.metadata.get(CONFIG_MAPPING_POLICY_KEY)
            return str(policy) if policy is not None else None
    return None


def _is_manager_callable_term_cfg(target_obj: Any) -> bool:
    if not dataclasses.is_dataclass(target_obj) or isinstance(target_obj, type):
        return False
    return any(
        config_field.metadata.get(CONFIG_MAPPING_POLICY_KEY) == MANAGER_PARAMS_MAPPING_POLICY
        for config_field in dataclasses.fields(target_obj)
    )


def _apply_manager_mapping_overrides(
    target_obj: Any,
    field_name: str,
    existing: Any,
    overrides: Any,
    *,
    policy: str,
) -> None:
    owner = f"{type(target_obj).__name__}.{field_name}"
    if not isinstance(existing, dict):
        raise TypeError(
            f"Config field '{owner}' declares manager mapping policy but contains "
            f"{type(existing).__name__}, expected dict"
        )
    if not isinstance(overrides, dict):
        raise TypeError(
            f"Config field '{owner}' must be overridden by a mapping, not "
            f"{type(overrides).__name__}"
        )

    if policy == MANAGER_PARAMS_MAPPING_POLICY:
        for param_name, value in overrides.items():
            current = existing.get(param_name)
            if isinstance(value, dict) and dataclasses.is_dataclass(current):
                apply_cfg_overrides(current, value)
            else:
                existing[param_name] = value
        return

    if policy != MANAGER_TERM_MAPPING_POLICY:
        raise ValueError(f"Config field '{owner}' has unknown mapping policy {policy!r}")

    for term_name, value in overrides.items():
        if term_name not in existing:
            raise ValueError(
                f"Config field '{owner}' has no factory-owned term '{term_name}'; "
                "declare its callable/config in the task Python factory first"
            )
        if value is None:
            existing[term_name] = None
            continue

        current = existing[term_name]
        if current is None:
            raise ValueError(
                f"Config field '{owner}' term '{term_name}' is disabled; set a concrete "
                "config in the task Python factory before overriding its fields"
            )
        if not dataclasses.is_dataclass(current):
            raise TypeError(
                f"Config field '{owner}' term '{term_name}' contains "
                f"{type(current).__name__}, expected a dataclass config"
            )
        if not isinstance(value, dict):
            raise TypeError(
                f"Config field '{owner}' term '{term_name}' must be overridden by a "
                "field mapping or None; replacing the factory-owned term is not allowed"
            )
        apply_cfg_overrides(current, value)


def apply_cfg_overrides(target_obj: Any, overrides: Dict[str, Any]) -> None:
    """Apply a (possibly nested) dict of overrides to ``target_obj`` in place.

    Behavior:
      - For each ``key, value`` in ``overrides``, ``target_obj.key`` must exist
        (otherwise ``ValueError``).
      - If ``value`` is a dict and ``target_obj.key`` is already a dataclass
        instance, recurse into it (deep merge — preserves fields not present
        in ``value``). This is what lets Hydra-style partial overrides like
        ``env.scene.terrain.generator.num_rows=4`` keep ``sub_terrains`` and other
        defaults intact.
      - If ``value`` is a dict and ``target_obj.key`` is currently ``None``,
        instantiate the field's annotated dataclass type from the dict
        (full-construction path).
      - Fields explicitly marked as manager mappings merge only existing
        factory-owned entries. ``None`` disables an entry; unknown entries and
        callable/config replacement fail closed.
      - Otherwise ``setattr`` the value directly (scalar / list / non-dataclass).
    """
    try:
        type_hints = get_type_hints(type(target_obj))
    except Exception:
        type_hints = {}

    for key, value in overrides.items():
        if not hasattr(target_obj, key):
            raise ValueError(f"Config class '{type(target_obj).__name__}' has no attribute '{key}'")
        existing = getattr(target_obj, key)
        mapping_policy = _config_mapping_policy(target_obj, key)
        if mapping_policy is not None:
            _apply_manager_mapping_overrides(
                target_obj,
                key,
                existing,
                value,
                policy=mapping_policy,
            )
            continue
        if key == "func" and _is_manager_callable_term_cfg(target_obj):
            raise ValueError(
                f"Config field '{type(target_obj).__name__}.func' is factory-owned and "
                "cannot be overridden"
            )
        if isinstance(value, dict):
            if dataclasses.is_dataclass(existing) and not isinstance(existing, type):
                apply_cfg_overrides(existing, value)
                continue
            if existing is None:
                target_type = _resolve_dataclass_type(type_hints.get(key))
                if target_type is not None:
                    setattr(target_obj, key, _construct_dataclass_from_dict(target_type, value))
                    continue
        setattr(target_obj, key, value)


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
    env_cfg = meta.env_cfg_cls()
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
    env_cls_any: Any = meta.env_cls_dict[sim_backend]
    env: ABEnv = env_cls_any(env_cfg, num_envs=num_envs, backend_type=sim_backend)
    return env


def list_registered_envs() -> Dict[str, Dict[str, Any]]:
    """List all registered environments with their available backends."""
    result = {}
    for name, meta in _envs.items():
        result[name] = {
            "config_class": meta.env_cfg_cls.__name__,
            "available_backends": list(meta.env_cls_dict.keys()),
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
