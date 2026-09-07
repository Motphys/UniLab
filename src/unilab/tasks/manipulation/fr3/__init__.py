"""Hydra-owned fixed-base FR3 joint-target task registration."""

from unilab.base import registry
from unilab.envs import ManagerBasedRlEnvCfg, make_manager_based_rl_env

registry.register_env_config("FR3JointTarget", ManagerBasedRlEnvCfg)
registry.register_env("FR3JointTarget", make_manager_based_rl_env, sim_backend="superdex")
