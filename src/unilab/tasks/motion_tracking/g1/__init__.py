"""G1 motion profiles on the shared NumPy Manager-Based runtime."""

from unilab.base import registry
from unilab.envs import ManagerBasedRlEnvCfg, make_manager_based_rl_env

from .motion_box_loader import BoxMotionData, BoxMotionLoader

G1_MOTION_TASKS = (
    "G1MotionTracking",
    "G1MotionTrackingDeploy",
    "G1MotionTracking23Dof",
    "G1MotionTracking23DofDeploy",
    "G1MotionTrackingSAC",
    "G1MotionTrackingSAC23Dof",
    "G1BoxTracking",
    "G1BoxTracking23Dof",
    "G1ClimbTracking",
    "G1ClimbTracking23Dof",
    "G1FlipTracking",
    "G1FlipTracking23Dof",
    "G1FlipTrackingSAC",
    "G1FlipTrackingSAC23Dof",
    "G1WallFlipTracking",
    "G1WallFlipTracking23Dof",
    "G1WallFlipTrackingSAC",
    "G1WallFlipTrackingSAC23Dof",
    "G1WBTObs",
    "G1WBTObs23Dof",
)

for _task_name in G1_MOTION_TASKS:
    registry.register_env_config(_task_name, ManagerBasedRlEnvCfg)
    registry.register_env(_task_name, make_manager_based_rl_env, sim_backend="mujoco")
    registry.register_env(_task_name, make_manager_based_rl_env, sim_backend="motrix")


__all__ = ["BoxMotionData", "BoxMotionLoader", "G1_MOTION_TASKS"]
