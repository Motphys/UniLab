"""Hydra-owned Manager-Based Go2W production registrations."""

from unilab.base import registry
from unilab.envs import ManagerBasedRlEnvCfg, make_manager_based_rl_env

registry.register_env_config("Go2WJoystickFlat", ManagerBasedRlEnvCfg)
registry.register_env("Go2WJoystickFlat", make_manager_based_rl_env, sim_backend="mujoco")
registry.register_env("Go2WJoystickFlat", make_manager_based_rl_env, sim_backend="motrix")
registry.register_env("Go2WJoystickFlat", make_manager_based_rl_env, sim_backend="drake")

registry.register_env_config("Go2WJoystickRough", ManagerBasedRlEnvCfg)
registry.register_env("Go2WJoystickRough", make_manager_based_rl_env, sim_backend="mujoco")
registry.register_env("Go2WJoystickRough", make_manager_based_rl_env, sim_backend="motrix")

__all__: list[str] = []
