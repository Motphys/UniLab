"""Hydra-owned Manager-Based G1 locomotion production registrations."""

from unilab.base import registry
from unilab.envs import ManagerBasedRlEnvCfg

from .manager_terms import G1WalkManagerBasedEnv, make_g1_walk_env

registry.register_env_config("G1WalkFlat", ManagerBasedRlEnvCfg)
registry.register_env("G1WalkFlat", make_g1_walk_env, sim_backend="mujoco")
registry.register_env("G1WalkFlat", make_g1_walk_env, sim_backend="mjwarp")
registry.register_env("G1WalkFlat", make_g1_walk_env, sim_backend="motrix")
registry.register_env("G1WalkFlat", make_g1_walk_env, sim_backend="isaacgym")
registry.register_env("G1WalkFlat", make_g1_walk_env, sim_backend="genesis")
registry.register_env("G1WalkFlat", make_g1_walk_env, sim_backend="isaacsim")
registry.register_env("G1WalkFlat", make_g1_walk_env, sim_backend="newton")

__all__ = [
    "G1WalkManagerBasedEnv",
    "make_g1_walk_env",
]
