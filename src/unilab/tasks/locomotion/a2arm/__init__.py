"""Manager-Based A2Arm position-force task registration."""

from __future__ import annotations

from unilab.assets.hub import resolve_robot_asset_dir
from unilab.base import registry
from unilab.envs import ManagerBasedRlEnv, ManagerBasedRlEnvCfg, make_manager_based_rl_env


def make_a2arm_pos_force_env(
    cfg: ManagerBasedRlEnvCfg,
    num_envs: int = 1,
    backend_type: str = "mujoco",
) -> ManagerBasedRlEnv:
    """Resolve external meshes before the backend parses the task scene."""
    resolve_robot_asset_dir("robots/a2arm/meshes", marker="adapter_plate.STL")
    return make_manager_based_rl_env(cfg, num_envs=num_envs, backend_type=backend_type)


def make_a2arm_pos_force_cfg() -> ManagerBasedRlEnvCfg:
    return ManagerBasedRlEnvCfg(critic_observation_group="critic")


registry.register_env_config("A2ArmPosForce", make_a2arm_pos_force_cfg)
registry.register_env("A2ArmPosForce", make_a2arm_pos_force_env, sim_backend="mujoco")

__all__ = ["make_a2arm_pos_force_cfg", "make_a2arm_pos_force_env"]
