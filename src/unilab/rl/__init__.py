"""Direct RSL-RL integration for UniLab PPO training and playback."""

from unilab.rl.distributed import (
    UNILAB_DP_LOG_DIR,
    apply_rsl_rl_rank_seed,
    current_torch_distributed_local_rank,
    current_torch_distributed_rank,
    current_torch_distributed_world_size,
    finish_rsl_rl_distributed,
    launch_torchrun_workers,
    ppo_samples_per_iteration,
    resolve_collector_cpu_ids,
    resolve_dp_topology,
    resolve_rsl_rl_device,
    rsl_rl_single_process_topology,
    torchrun_ranks_are_colocated,
    validate_dp_launchable,
)
from unilab.rl.vec_env import RslRlVecEnvAdapter, get_policy_obs_dims

__all__ = [
    "RslRlVecEnvAdapter",
    "UNILAB_DP_LOG_DIR",
    "apply_rsl_rl_rank_seed",
    "current_torch_distributed_local_rank",
    "current_torch_distributed_rank",
    "current_torch_distributed_world_size",
    "finish_rsl_rl_distributed",
    "get_policy_obs_dims",
    "launch_torchrun_workers",
    "ppo_samples_per_iteration",
    "resolve_collector_cpu_ids",
    "resolve_dp_topology",
    "resolve_rsl_rl_device",
    "rsl_rl_single_process_topology",
    "torchrun_ranks_are_colocated",
    "validate_dp_launchable",
]
