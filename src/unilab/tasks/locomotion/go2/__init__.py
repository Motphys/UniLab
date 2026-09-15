"""Hydra-owned Manager-Based Go2 production registrations."""

from unilab.base import registry
from unilab.envs import ManagerBasedRlEnvCfg, make_manager_based_rl_env

registry.register_env_config("Go2JoystickFlat", ManagerBasedRlEnvCfg)
registry.register_env("Go2JoystickFlat", make_manager_based_rl_env, sim_backend="mujoco")
registry.register_env("Go2JoystickFlat", make_manager_based_rl_env, sim_backend="motrix")
registry.register_env("Go2JoystickFlat", make_manager_based_rl_env, sim_backend="drake")
registry.register_env("Go2JoystickFlat", make_manager_based_rl_env, sim_backend="superdex")

__all__: list[str] = []
