"""EngineAI T800 walk-flat task on the shared Manager-Based runtime."""

from unilab.assets.hub import resolve_robot_asset_dir
from unilab.base import registry
from unilab.envs import ManagerBasedRlEnv, ManagerBasedRlEnvCfg, make_manager_based_rl_env


def make_t800_walk_env(
    cfg: ManagerBasedRlEnvCfg,
    num_envs: int = 1,
    backend_type: str = "mujoco",
) -> ManagerBasedRlEnv:
    """Resolve T800 binary assets before materializing the generic runtime."""
    resolve_robot_asset_dir("robots/t800/assets", marker="LINK_BASE.obj")
    resolve_robot_asset_dir("robots/t800/textures", marker="LINK_BASE.png")
    return make_manager_based_rl_env(cfg, num_envs=num_envs, backend_type=backend_type)


registry.register_env_config("T800WalkFlat", ManagerBasedRlEnvCfg)
registry.register_env("T800WalkFlat", make_t800_walk_env, sim_backend="mujoco")


__all__ = ["make_t800_walk_env"]
