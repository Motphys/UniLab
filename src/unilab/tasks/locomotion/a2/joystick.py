"""Hydra-owned A2 flat Manager-Based production registration."""

from unilab.base import registry
from unilab.envs import ManagerBasedRlEnvCfg, make_manager_based_rl_env

registry.register_env_config("A2JoystickFlat", ManagerBasedRlEnvCfg)
registry.register_env("A2JoystickFlat", make_manager_based_rl_env, sim_backend="mujoco")
