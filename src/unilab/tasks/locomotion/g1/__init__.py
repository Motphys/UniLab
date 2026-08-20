"""Hydra-owned Manager-Based G1 locomotion production registrations."""

from unilab.base import registry
from unilab.envs import ManagerBasedRlEnvCfg

from .manager_terms import G1WalkManagerBasedEnv, make_g1_walk_env
from .symmetry import G1SymmetryAugmentation

registry.register_env_config("G1WalkFlat", ManagerBasedRlEnvCfg)
registry.register_env("G1WalkFlat", make_g1_walk_env, sim_backend="mujoco")
registry.register_env("G1WalkFlat", make_g1_walk_env, sim_backend="mjwarp")
registry.register_env("G1WalkFlat", make_g1_walk_env, sim_backend="motrix")

registry.register_env_config("G1WalkRough", ManagerBasedRlEnvCfg)
registry.register_env("G1WalkRough", make_g1_walk_env, sim_backend="mujoco")
registry.register_env("G1WalkRough", make_g1_walk_env, sim_backend="motrix")

registry.register_env_config("G1Walk23DofFlat", ManagerBasedRlEnvCfg)
registry.register_env("G1Walk23DofFlat", make_g1_walk_env, sim_backend="mujoco")
registry.register_env("G1Walk23DofFlat", make_g1_walk_env, sim_backend="motrix")

registry.register_env_config("G1Walk23DofRough", ManagerBasedRlEnvCfg)
registry.register_env("G1Walk23DofRough", make_g1_walk_env, sim_backend="mujoco")
registry.register_env("G1Walk23DofRough", make_g1_walk_env, sim_backend="motrix")

__all__ = [
    "G1SymmetryAugmentation",
    "G1WalkManagerBasedEnv",
    "make_g1_walk_env",
]
