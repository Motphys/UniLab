import dataclasses
import importlib
import logging
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import (
    Any,
    Callable,
    Dict,
    Optional,
    Type,
    TypeVar,
    cast,
    get_args,
    get_origin,
    get_type_hints,
)

from .base import ABEnv, EnvCfg

TEnvCfg = TypeVar("TEnvCfg", bound=EnvCfg)
_DEFAULT_SIM_BACKEND_ORDER: tuple[str, ...] = ("mujoco", "motrix")
_REGISTRY_MODULES_ATTR = "__unilab_registry_modules__"
_DEFAULT_REGISTRY_PACKAGES = (
    "unilab.envs.locomotion",
    "unilab.envs.manipulation",
    "unilab.envs.motion_tracking",
)

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
    """Decorate an environment configuration class and register it by name.

    Example::

        @envcfg("my-env")
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
    if sim_backend not in ["mujoco", "motrix"]:
        raise ValueError(
            f"Unsupported simulation backend: {sim_backend}. Only 'mujoco' and 'motrix' are supported."
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
    """Decorate an environment class and register it for a simulation backend.

    Example::

        @env("my-env", "mujoco")
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


def apply_cfg_overrides(target_obj: Any, overrides: Dict[str, Any]) -> None:
    """Apply a (possibly nested) dict of overrides to ``target_obj`` in place.

    Every override key must already exist on ``target_obj``. Nested dict values
    are deep-merged into existing dataclass fields, and ``None`` dataclass fields
    are constructed from their annotations. Scalar, list, and non-dataclass
    values are assigned directly.
    """
    try:
        type_hints = get_type_hints(type(target_obj))
    except Exception:
        type_hints = {}

    for key, value in overrides.items():
        if not hasattr(target_obj, key):
            raise ValueError(f"Config class '{type(target_obj).__name__}' has no attribute '{key}'")
        existing = getattr(target_obj, key)
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
        sim_backend: Simulation backend ("mujoco" or "motrix"). If None, uses the
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
    package_names = list(packages) if packages is not None else list(_DEFAULT_REGISTRY_PACKAGES)
    optional = set(optional_packages) if optional_packages else set()

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
