from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional, Type, TypeVar

from .base import ABEnv, EnvCfg

TEnvCfg = TypeVar("TEnvCfg", bound=EnvCfg)


@dataclass
class EnvMeta:
    env_cfg_cls: Type[EnvCfg]
    env_cls_dict: Dict[str, Type[ABEnv]] = field(default_factory=dict)

    def available_sim_backend(self) -> Optional[str]:
        """Return the first available simulation backend."""
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
    """Find the first available simulation backend for an environment."""
    if env_name not in _envs:
        raise ValueError(f"Environment '{env_name}' is not registered.")

    meta: EnvMeta = _envs[env_name]
    backend = meta.available_sim_backend()
    if backend is None:
        raise ValueError(f"Environment '{env_name}' does not support any simulation backend.")
    return backend


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
        sim_backend: Simulation backend ("mujoco" or "motrix"). If None, uses first available.
        env_cfg_override: Dictionary of config overrides
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
        for key, value in env_cfg_override.items():
            if hasattr(env_cfg, key):
                setattr(env_cfg, key, value)
            else:
                raise ValueError(
                    f"Config class '{env_cfg.__class__.__name__}' has no attribute '{key}'"
                )

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
