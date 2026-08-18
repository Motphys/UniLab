"""Registry owner for the Allegro in-hand rotation Manager-Based task."""

from unilab.base import registry
from unilab.envs import ManagerBasedRlEnvCfg, make_manager_based_rl_env

registry.register_env_config("AllegroInhandRotation", ManagerBasedRlEnvCfg)
registry.register_env("AllegroInhandRotation", make_manager_based_rl_env, sim_backend="mujoco")
registry.register_env("AllegroInhandRotation", make_manager_based_rl_env, sim_backend="motrix")
registry.register_env("AllegroInhandRotation", make_manager_based_rl_env, sim_backend="drake")
