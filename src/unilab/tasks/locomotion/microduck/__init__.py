"""Pollen Robotics MicroDuck velocity task on the Manager-Based runtime."""

from unilab.assets.hub import resolve_robot_asset_dir
from unilab.base import registry
from unilab.envs import ManagerBasedRlEnv, ManagerBasedRlEnvCfg, make_manager_based_rl_env


def make_microduck_velocity_env(
    cfg: ManagerBasedRlEnvCfg,
    num_envs: int = 1,
    backend_type: str = "mujoco",
) -> ManagerBasedRlEnv:
    """Resolve MicroDuck STL assets before materializing the generic runtime."""
    resolve_robot_asset_dir("robots/microduck/assets", marker="trunk_base.stl")
    return make_manager_based_rl_env(cfg, num_envs=num_envs, backend_type=backend_type)


registry.register_env_config("MicroduckVelocityFlat", ManagerBasedRlEnvCfg)
registry.register_env(
    "MicroduckVelocityFlat",
    make_microduck_velocity_env,
    sim_backend="mujoco",
)
registry.register_env(
    "MicroduckVelocityFlat",
    make_microduck_velocity_env,
    sim_backend="mjwarp",
)


__all__ = ["make_microduck_velocity_env"]
