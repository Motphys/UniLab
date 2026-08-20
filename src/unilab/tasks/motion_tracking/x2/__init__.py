"""AgiBot X2 motion profiles on the shared NumPy Manager-Based runtime."""

from unilab.assets.hub import resolve_robot_asset_dir
from unilab.base import registry
from unilab.envs import ManagerBasedRlEnv, ManagerBasedRlEnvCfg, make_manager_based_rl_env


def make_x2_wall_flip_env(
    cfg: ManagerBasedRlEnvCfg,
    num_envs: int = 1,
    backend_type: str = "mujoco",
) -> ManagerBasedRlEnv:
    """Resolve untracked X2 meshes before backend scene materialization."""
    resolve_robot_asset_dir("robots/x2/meshes", marker="pelvis.STL")
    return make_manager_based_rl_env(cfg, num_envs=num_envs, backend_type=backend_type)


registry.register_env_config("X2WallFlipTracking", ManagerBasedRlEnvCfg)
registry.register_env("X2WallFlipTracking", make_x2_wall_flip_env, sim_backend="mujoco")
registry.register_env("X2WallFlipTracking", make_x2_wall_flip_env, sim_backend="motrix")


__all__ = ["make_x2_wall_flip_env"]
