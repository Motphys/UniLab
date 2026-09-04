"""Hydra-owned Manager-Based Go1 production registrations."""

from unilab.base import registry
from unilab.envs import ManagerBasedRlEnvCfg, make_manager_based_rl_env

registry.register_env_config("Go1JoystickFlat", ManagerBasedRlEnvCfg)
registry.register_env("Go1JoystickFlat", make_manager_based_rl_env, sim_backend="mujoco")
registry.register_env("Go1JoystickFlat", make_manager_based_rl_env, sim_backend="motrix")
registry.register_env("Go1JoystickFlat", make_manager_based_rl_env, sim_backend="drake")

registry.register_env_config("Go1JoystickRough", ManagerBasedRlEnvCfg)
registry.register_env("Go1JoystickRough", make_manager_based_rl_env, sim_backend="mujoco")
registry.register_env("Go1JoystickRough", make_manager_based_rl_env, sim_backend="motrix")

__all__: list[str] = []
