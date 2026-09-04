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

# BAM voltage-actuator variant (issue #1474): mujoco only, because the substep
# state-feedback contract (SimBackend.set_pre_step_control) is unavailable on
# the mjwarp host_numpy profile.
registry.register_env_config("MicroduckVelocityBamFlat", ManagerBasedRlEnvCfg)
registry.register_env(
    "MicroduckVelocityBamFlat",
    make_microduck_velocity_env,
    sim_backend="mujoco",
)

# VelStand (walking + fall recovery) on the ground-contact BAM model; mujoco
# only for the same BAM substep state-feedback reason as the walk variant.
registry.register_env_config("MicroduckVelstandFlat", ManagerBasedRlEnvCfg)
registry.register_env(
    "MicroduckVelstandFlat",
    make_microduck_velocity_env,
    sim_backend="mujoco",
)

# The minimal command/term owners deliberately share the same generic
# ManagerBasedRlEnv factory.  Their behavior is selected entirely by the
# Hydra owner (command, metrics, recorder, and reward terms); the runtime does
# not branch on task names.
for _task_name in ("MicroduckGroundPickFlat", "MicroduckSitStandFlat"):
    registry.register_env_config(_task_name, ManagerBasedRlEnvCfg)
    registry.register_env(_task_name, make_microduck_velocity_env, sim_backend="mjwarp")


__all__ = ["make_microduck_velocity_env"]
