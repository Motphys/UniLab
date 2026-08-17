"""FastTD3 builder for the device-authoritative replay path."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from omegaconf import DictConfig

from unilab.algos.torch.common.device import get_env_dims
from unilab.algos.torch.fast_td3.learner import FastTD3Learner
from unilab.algos.torch.offpolicy.double_buffer_runner import DoubleBufferOffPolicyRunner
from unilab.utils.nan_guard import NanGuardCfg

if TYPE_CHECKING:
    from unilab.ipc.dp_sync import DpParameterSync


def build_td3_double_buffer_runner(
    cfg: DictConfig,
    *,
    env_cfg_override: dict[str, Any] | None,
    replay_prefetch_mode: str,
    device: str,
    nan_guard_cfg: NanGuardCfg | None = None,
    torch_thread_runtime: dict[str, Any] | None = None,
    collector_cpu_ids: list[int] | None = None,
    dp_sync: DpParameterSync | None = None,
) -> Any:
    """Build TD3 from its Hydra owner config without interpreting it in the entrypoint."""
    obs_dim, action_dim, critic_obs_dim = get_env_dims(
        cfg.training.task_name,
        cfg.training.sim_backend,
        env_cfg_override=env_cfg_override,
    )
    learner = FastTD3Learner(
        obs_dim=obs_dim,
        action_dim=action_dim,
        critic_obs_dim=critic_obs_dim,
        num_envs=cfg.algo.num_envs,
        device=device,
        gamma=cfg.algo.gamma,
        tau=cfg.algo.tau,
        actor_lr=cfg.algo.actor_lr,
        critic_lr=cfg.algo.critic_lr,
        actor_hidden_dim=cfg.algo.actor_hidden_dim,
        critic_hidden_dim=cfg.algo.critic_hidden_dim,
        num_atoms=cfg.algo.num_atoms,
        v_min=cfg.algo.algo_params.v_min,
        v_max=cfg.algo.algo_params.v_max,
        init_scale=cfg.algo.algo_params.init_scale,
        log_std_min=cfg.algo.algo_params.log_std_min,
        log_std_max=cfg.algo.algo_params.log_std_max,
        weight_decay=cfg.algo.algo_params.weight_decay,
        use_cdq=cfg.algo.algo_params.use_cdq,
        policy_noise=cfg.algo.algo_params.policy_noise,
        noise_clip=cfg.algo.algo_params.noise_clip,
        policy_frequency=cfg.algo.policy_frequency,
        obs_normalization=cfg.algo.obs_normalization,
    )

    return DoubleBufferOffPolicyRunner(
        learner=learner,
        env_name=cfg.training.task_name,
        algo_type="td3",
        env_cfg_override=env_cfg_override,
        device=device,
        num_envs=cfg.algo.num_envs,
        replay_buffer_n=cfg.algo.replay_buffer_n,
        batch_size=cfg.algo.batch_size,
        learning_starts=cfg.algo.learning_starts,
        updates_per_step=cfg.algo.updates_per_step,
        policy_frequency=cfg.algo.policy_frequency,
        env_steps_per_sync=cfg.training.env_steps_per_sync,
        obs_normalization=cfg.algo.obs_normalization,
        sim_backend=cfg.training.sim_backend,
        seed=cfg.algo.seed,
        trace_enabled=cfg.training.trace_enabled,
        trace_output_dir=cfg.training.trace_output_dir,
        trace_thread_time=cfg.training.trace_thread_time,
        trace_cuda_events=cfg.training.trace_cuda_events,
        replay_prefetch_mode=replay_prefetch_mode,
        nan_guard_cfg=nan_guard_cfg,
        torch_thread_runtime=torch_thread_runtime,
        collector_cpu_ids=collector_cpu_ids,
        dp_sync=dp_sync,
    )
