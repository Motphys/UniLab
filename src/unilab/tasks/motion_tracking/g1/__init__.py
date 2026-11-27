"""G1 motion profiles on the shared NumPy Manager-Based runtime."""

from unilab.base import registry
from unilab.envs import ManagerBasedRlEnvCfg, make_manager_based_rl_env

G1_MOTION_TASKS = ("G1MotionTracking",)

for _task_name in G1_MOTION_TASKS:
    registry.register_env_config(_task_name, ManagerBasedRlEnvCfg)
    # One semantic task identity serves PPO/APPO/SAC/FlashSAC. Hydra owner
    # leaves carry the algorithm-specific Manager-Based contracts.
    for sim_backend in ("mujoco", "mjwarp", "genesis", "newton", "motrix", "mtxwarp"):
        registry.register_env(_task_name, make_manager_based_rl_env, sim_backend=sim_backend)
__all__ = ["G1_MOTION_TASKS"]
