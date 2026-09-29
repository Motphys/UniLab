"""G1 motion profiles on the shared NumPy Manager-Based runtime."""

from unilab.base import registry
from unilab.envs import ManagerBasedRlEnvCfg, make_manager_based_rl_env

from .motion_box_loader import BoxMotionData, BoxMotionLoader
from .torch_flashsac_env import make_torch_g1_motion_tracking_flashsac_env

G1_MOTION_TASKS = (
    "G1MotionTracking",
    "G1MotionTrackingSAC",
    "G1BoxTracking",
    "G1FlipTracking",
    "G1FlipTrackingSAC",
    "G1WBTObs",
)

for _task_name in G1_MOTION_TASKS:
    registry.register_env_config(_task_name, ManagerBasedRlEnvCfg)
    if _task_name != "G1MotionTrackingSAC":
        registry.register_env(_task_name, make_manager_based_rl_env, sim_backend="mujoco")
        registry.register_env(_task_name, make_manager_based_rl_env, sim_backend="motrix")

# FlashSAC G1 has a task-owned tensor runtime for the scoped MJWarp/MJBatch
# owners; other G1 tasks retain the general NumPy Manager-Based runtime.
registry.register_env(
    "G1MotionTrackingSAC",
    make_torch_g1_motion_tracking_flashsac_env,
    sim_backend="mujoco",
)
# Motrix is an in-process HOST_BRIDGE backend: the task-owned runtime retains
# its CPU cold-contract proxy while the hot state/sensor/control path uses the
# backend's explicitly negotiated packed host bridge.  Registering the generic
# manager factory here would instead execute NumPy-only observation terms in
# the CUDA runtime selected by this FlashSAC owner.
registry.register_env(
    "G1MotionTrackingSAC",
    make_torch_g1_motion_tracking_flashsac_env,
    sim_backend="motrix",
)

# mjwarp is registered only for G1MotionTrackingSAC (benchmark scope, issue #1292);
# mujoco-warp + warp-lang remain optional deps and other motion tasks keep
# mujoco/motrix until their mjwarp paths are validated.
registry.register_env(
    "G1MotionTrackingSAC",
    make_torch_g1_motion_tracking_flashsac_env,
    sim_backend="mjwarp",
)
# G1 flip tracking is the second scoped tensor task owner. It intentionally
# registers only the validated MJWarp tensor path; MuJoCo/Motrix remain NumPy.
registry.register_env(
    "G1FlipTrackingSAC",
    make_torch_g1_motion_tracking_flashsac_env,
    sim_backend="mjwarp",
)
# genesis/newton implement the motion-body-id capability since unisim-core 1.5.1
# (unilabsim/unisim#137); isaacgym/isaacsim join them since unisim-core 1.7.4
# fixed the subprocess body-state publish/reset paths (unilabsim/unisim#141,
# PR #145). All DEVICE_RESIDENT candidates use the task-owned Torch runtime so
# observations and selected reset never enter the generic NumPy manager path.
for _backend in ("genesis", "newton", "isaacgym", "isaacsim"):
    registry.register_env(
        "G1MotionTrackingSAC", make_torch_g1_motion_tracking_flashsac_env, sim_backend=_backend
    )


__all__ = ["BoxMotionData", "BoxMotionLoader", "G1_MOTION_TASKS"]
