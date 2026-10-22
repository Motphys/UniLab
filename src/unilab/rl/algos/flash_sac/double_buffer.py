"""FlashSAC builder for the device-authoritative replay path."""

from __future__ import annotations

from collections.abc import Callable
from functools import partial
from typing import TYPE_CHECKING, Any

from omegaconf import DictConfig

from unilab.rl.algos.flash_sac.learner import FlashSACLearner
from unilab.rl.algos.flash_sac.replay import AgeBiasedReplayPipeline
from unilab.rl.contracts.env import EnvFactory
from unilab.rl.contracts.observations import get_obs_dims
from unilab.rl.runtime.device import get_default_device
from unilab.rl.runtime.ipc.replay_pipelines.device import require_offpolicy_replay_device
from unilab.rl.runtime.offpolicy.double_buffer_runner import (
    DoubleBufferOffPolicyRunner,
    validate_no_inference_request_timeout,
)
from unilab.rl.runtime.placement import InferencePlacement
from unilab.rl.runtime.seed import apply_worker_seed
from unilab.rl.runtime.settings import RuntimeSettings

if TYPE_CHECKING:
    from unilab.rl.runtime.distributed.sync import DpParameterSync


def _algo_param(algo_cfg: DictConfig, name: str, default: Any) -> Any:
    params = algo_cfg.get("algo_params", {})
    if name in params:
        return params[name]
    return algo_cfg.get(name, default)


def _validate_flashsac_double_buffer_runtime(
    cfg: DictConfig,
    *,
    replay_prefetch_mode: str,
) -> None:
    if replay_prefetch_mode != "one_tick":
        raise ValueError("FlashSAC device replay requires replay_prefetch_mode='one_tick'")
    if cfg.algo.algo_params.n_step != 1:
        raise ValueError("FlashSAC-B initially supports n_step=1 only")


def build_double_buffer_runner(
    cfg: DictConfig,
    *,
    env_factory: EnvFactory,
    env_cfg_override: dict[str, Any] | None,
    replay_prefetch_mode: str,
    device: str | None = None,
    nan_guard_cfg: Any | None = None,
    nan_guard_factory: Callable[[Any, int, bool], Any] | None = None,
    torch_thread_runtime: dict[str, Any] | None = None,
    collector_cpu_ids: list[int] | None = None,
    dp_sync: DpParameterSync | None = None,
    backend_device_resolver: Callable[[str, str | None], str | None] | None = None,
    backend_device_configurer: Callable[[str, str | None], str | None] | None = None,
    runtime_settings: RuntimeSettings | None = None,
    inference_placement: InferencePlacement | None = None,
) -> Any:
    """Build FlashSAC with the bounded-ingress device replay pipeline.

    ``runtime_settings`` and ``inference_placement`` are resolved from
    the owner config by the training assembly layer
    (:mod:`unilab.training.tensor_runtime`) and injected here; when omitted the
    runner-side defaults apply. Backend process-device resolution/binding is
    likewise injected by the owner layer (e.g.
    :mod:`unilab.base.process_device`).
    """
    device = require_offpolicy_replay_device(device or get_default_device())
    apply_worker_seed(cfg.algo.seed, torch_runtime=True, cuda=True)
    _validate_flashsac_double_buffer_runtime(
        cfg,
        replay_prefetch_mode=replay_prefetch_mode,
    )
    if runtime_settings is not None:
        batch_size = runtime_settings.batch_size
        updates_per_step = runtime_settings.updates_per_step
    else:
        batch_size = int(cfg.algo.batch_size)
        updates_per_step = int(cfg.algo.updates_per_step)

    validate_no_inference_request_timeout(cfg.training)
    env = env_factory(1, env_cfg_override)
    try:
        obs_dim, critic_obs_dim = get_obs_dims(dict(env.obs_groups_spec))
        action_shape = env.action_space.shape
        assert action_shape is not None
        action_dim = int(action_shape[0])
    finally:
        env.close()

    learner_kwargs = {
        "obs_dim": obs_dim,
        "action_dim": action_dim,
        "critic_obs_dim": critic_obs_dim,
        "gamma": cfg.algo.gamma,
        "tau": cfg.algo.tau,
        "actor_lr": cfg.algo.actor_lr,
        "critic_lr": cfg.algo.critic_lr,
        "actor_hidden_dim": cfg.algo.actor_hidden_dim,
        "critic_hidden_dim": cfg.algo.critic_hidden_dim,
        "actor_num_blocks": cfg.algo.algo_params.actor_num_blocks,
        "critic_num_blocks": cfg.algo.algo_params.critic_num_blocks,
        "num_atoms": cfg.algo.num_atoms,
        "critic_min_v": cfg.algo.algo_params.critic_min_v,
        "critic_max_v": cfg.algo.algo_params.critic_max_v,
        "temp_initial_value": cfg.algo.algo_params.temp_initial_value,
        "temp_target_sigma": cfg.algo.algo_params.temp_target_sigma,
        "temp_target_entropy": cfg.algo.algo_params.temp_target_entropy,
        "actor_bc_alpha": cfg.algo.algo_params.actor_bc_alpha,
        "actor_noise_zeta_mu": cfg.algo.algo_params.actor_noise_zeta_mu,
        "actor_noise_zeta_max": cfg.algo.algo_params.actor_noise_zeta_max,
        "actor_config": _algo_param(cfg.algo, "actor_config", None),
        "learning_rate_init": cfg.algo.algo_params.learning_rate_init,
        "learning_rate_peak": cfg.algo.algo_params.learning_rate_peak,
        "learning_rate_end": cfg.algo.algo_params.learning_rate_end,
        "learning_rate_warmup_steps": cfg.algo.algo_params.learning_rate_warmup_steps,
        "learning_rate_decay_steps": cfg.algo.algo_params.learning_rate_decay_steps,
        "normalize_reward": cfg.algo.algo_params.normalize_reward,
        "normalized_g_max": cfg.algo.algo_params.normalized_g_max,
        "n_step": cfg.algo.algo_params.n_step,
        "obs_normalization": cfg.algo.obs_normalization,
        "use_amp": cfg.training.use_amp,
        "amp_dtype": cfg.algo.algo_params.amp_dtype,
        "use_compile": cfg.algo.algo_params.use_compile,
        "compile_full_objectives": _algo_param(cfg.algo, "compile_full_objectives", None),
        "use_whole_cycle_cuda_graph": _algo_param(cfg.algo, "use_whole_cycle_cuda_graph", None),
        "dp_whole_cycle_cuda_graph": _algo_param(cfg.algo, "dp_whole_cycle_cuda_graph", False),
        "actor_normalize_parameters": bool(
            _algo_param(cfg.algo, "actor_normalize_parameters", True)
        ),
        "critic_normalize_parameters": bool(
            _algo_param(cfg.algo, "critic_normalize_parameters", True)
        ),
    }
    learner = FlashSACLearner(device=device, **learner_kwargs)

    return DoubleBufferOffPolicyRunner(
        learner=learner,
        env_name=cfg.training.task_name,
        algo_type="flashsac",
        env_factory=env_factory,
        num_envs=cfg.algo.num_envs,
        replay_buffer_n=cfg.algo.replay_buffer_n,
        batch_size=batch_size,
        learning_starts=cfg.algo.learning_starts,
        updates_per_step=updates_per_step,
        policy_frequency=cfg.algo.policy_frequency,
        target_frequency=int(_algo_param(cfg.algo, "target_frequency", 1)),
        policy_before_critic=bool(_algo_param(cfg.algo, "policy_before_critic", False)),
        env_steps_per_sync=cfg.training.env_steps_per_sync,
        device=device,
        obs_normalization=cfg.algo.obs_normalization,
        sim_backend=cfg.training.sim_backend,
        env_cfg_override=env_cfg_override,
        seed=cfg.algo.seed,
        trace_enabled=cfg.training.trace_enabled,
        trace_output_dir=cfg.training.trace_output_dir,
        trace_thread_time=cfg.training.trace_thread_time,
        trace_cuda_events=cfg.training.trace_cuda_events,
        replay_prefetch_mode=replay_prefetch_mode,
        nan_guard_cfg=nan_guard_cfg,
        nan_guard_factory=nan_guard_factory,
        torch_thread_runtime=torch_thread_runtime,
        collector_cpu_ids=collector_cpu_ids,
        dp_sync=dp_sync,
        backend_device_resolver=backend_device_resolver,
        backend_device_configurer=backend_device_configurer,
        inference_placement=inference_placement,
        runtime_settings=runtime_settings,
        log_interval=int(cfg.training.log_interval),
        replay_pipeline_factory=partial(
            AgeBiasedReplayPipeline,
            decay_step=int(_algo_param(cfg.algo, "decay_step", 0)),
            min_weight=float(_algo_param(cfg.algo, "replay_min_weight", 0.1)),
            num_buckets=int(_algo_param(cfg.algo, "replay_num_buckets", 2000)),
        ),
    )
