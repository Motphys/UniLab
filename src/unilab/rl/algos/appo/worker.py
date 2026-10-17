"""APPO Rollout Worker — runs in a subprocess.

Collects rollout payloads and writes them to the tensor rollout ring.
"""

from __future__ import annotations

import statistics
import sys
import time
from collections import deque
from queue import Empty, Full
from typing import Any, Dict

import torch
from rsl_rl.utils import resolve_callable

from unilab.rl.runtime.collector_timing import extract_env_step_breakdown_timing_ms
from unilab.rl.runtime.offpolicy.thread_budget import apply_torch_thread_runtime
from unilab.rl.runtime.seed import apply_training_seed


def put_latest_metrics(metrics_queue: Any, msg: dict[str, Any], *, worker_name: str) -> None:
    """Best-effort metrics enqueue that keeps recent data under learner stalls."""
    try:
        metrics_queue.put_nowait(msg)
        return
    except Full:
        pass
    except Exception as e:
        print(f"[{worker_name}] metrics enqueue error: {type(e).__name__}: {e}", file=sys.stderr)
        return

    try:
        metrics_queue.get_nowait()
    except Empty:
        pass
    except Exception as e:
        print(
            f"[{worker_name}] metrics drop stale metrics error: {type(e).__name__}: {e}",
            file=sys.stderr,
        )
        return

    try:
        metrics_queue.put_nowait(msg)
    except Full:
        pass
    except Exception as e:
        print(f"[{worker_name}] metrics enqueue error: {type(e).__name__}: {e}", file=sys.stderr)


def compute_timeout_bootstrap_correction(
    critic: Any,
    collector_device: str,
    gamma: float,
    timeout_mask: torch.Tensor,
    final_obs: torch.Tensor,
    final_critic: torch.Tensor,
) -> torch.Tensor:
    """Compute ``gamma * V(final_observation)`` for current timeout rows."""
    corrections = torch.zeros_like(timeout_mask, dtype=torch.float32)
    from tensordict import TensorDict

    # An env can keep its authoritative state on CPU while policy inference
    # uses a CUDA collector device. Move the boolean selector and terminal
    # rows to one device before indexing, then restore the correction to the
    # reward carrier's device.
    mask = timeout_mask.to(device=final_critic.device)
    critic_input = final_critic[mask].to(device=collector_device)
    critic_td = TensorDict(
        {"policy": critic_input},
        batch_size=critic_input.shape[0],
        device=collector_device,
    )
    with torch.no_grad():
        bootstrap = critic(critic_td).squeeze(-1).to(device=timeout_mask.device)
    corrections[mask.to(device=corrections.device)] = float(gamma) * bootstrap
    return corrections


def write_rollout_bootstrap_observations(
    write_buf: dict[str, torch.Tensor],
    last_obs: torch.Tensor,
    last_critic: torch.Tensor | None,
) -> None:
    write_buf["last_obs"].copy_(last_obs, non_blocking=False)
    if last_critic is not None:
        write_buf["last_critic"].copy_(last_critic, non_blocking=False)


def timeout_bootstrap_mask(terminated: torch.Tensor, truncated: torch.Tensor) -> torch.Tensor:
    if terminated.shape != truncated.shape:
        raise ValueError("APPO terminated/truncated tensors must have the same shape")
    return torch.logical_and(truncated, ~terminated)


def advance_rollout_observations(
    next_obs: torch.Tensor,
    next_critic: torch.Tensor | None,
    collector_device: str | torch.device,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    actor_obs = next_obs.to(device=collector_device, dtype=torch.float32)
    critic_obs = (
        next_critic.to(device=collector_device, dtype=torch.float32)
        if next_critic is not None
        else None
    )
    return actor_obs, critic_obs


def appo_collector_fn(
    stop_event: Any,
    env_factory: Any,
    rl_cfg: dict,
    num_envs: int,
    steps_per_env: int,
    shm_rollout_ring_buffer_name: Dict[str, str],
    sync_primitives: tuple,
    obs_dim: int,
    action_dim: int,
    critic_dim: int,
    actor_weight_sync_name: str,
    actor_weight_param_shapes: dict,
    critic_weight_sync_name: str,
    critic_weight_param_shapes: dict,
    metrics_queue: Any,
    collector_device: str = "cpu",
    env_cfg_override: dict | None = None,
    actor_weight_sync_lock=None,
    critic_weight_sync_lock=None,
    seed: int | None = None,
    nan_guard_cfg=None,
    nan_guard_factory=None,
    torch_thread_runtime=None,
):
    """Entry point for the APPO collector subprocess.

    Creates environment + policy, collects rollouts, writes raw payloads
    to the IPC ring buffer. Error handling is provided by the
    ``_collector_entry_wrapper`` in ``async_runner.py``.
    """
    from copy import deepcopy

    from tensordict import TensorDict

    from unilab.rl.runtime.ipc import SharedWeightSync, TensorRolloutRingBuffer

    apply_torch_thread_runtime(torch_thread_runtime, role="collector", torch_module=torch)
    apply_training_seed(seed, torch_runtime=True, cuda=True)

    # Connect to shared memory
    ring_buffer = TensorRolloutRingBuffer(
        num_envs=num_envs,
        num_steps=steps_per_env,
        obs_dim=obs_dim,
        action_dim=action_dim,
        critic_dim=critic_dim,
        create=False,
        shm_name_prefix=shm_rollout_ring_buffer_name,
    )
    ring_buffer.attach_sync_primitives(*sync_primitives)  # (write_ptr, read_ptr)
    actor_weight_sync = SharedWeightSync(
        actor_weight_param_shapes,
        create=False,
        shm_name=actor_weight_sync_name,
        lock=actor_weight_sync_lock,
    )
    critic_weight_sync = SharedWeightSync(
        critic_weight_param_shapes,
        create=False,
        shm_name=critic_weight_sync_name,
        lock=critic_weight_sync_lock,
    )

    # Create environment through the injected factory (see unilab.rl.contracts.env)
    env = env_factory(num_envs, env_cfg_override)

    if nan_guard_cfg is not None and nan_guard_cfg.enabled:
        from unilab.rl.runtime.nan_guard import NanGuard

        guard_factory = NanGuard if nan_guard_factory is None else nan_guard_factory
        env.set_nan_guard(
            guard_factory(
                nan_guard_cfg,
                num_envs=env.num_envs,
                supports_state_playback=env.play_capabilities.supports_physics_state_playback,
            )
        )

    # Build actor (stochastic MLPModel — mirrors runner._build_learner)
    cfg = dict(rl_cfg)

    obs_example = torch.zeros((num_envs, obs_dim), device=collector_device)
    td_example = TensorDict({"policy": obs_example}, batch_size=num_envs)

    # deepcopy so MLPModel.__init__'s distribution_cfg.pop("class_name") doesn't
    # mutate the shared rl_cfg dict.
    actor_cfg = deepcopy(cfg["actor"])
    actor_cls = resolve_callable(actor_cfg.pop("class_name"))
    actor_cfg.pop("num_actions", None)
    actor = actor_cls(
        td_example,
        cfg.get("obs_groups", {"actor": {"policy": obs_dim}}),
        "actor",
        action_dim,
        **actor_cfg,
    )
    actor = actor.to(collector_device)
    actor.eval()

    critic_obs_dim = critic_dim if critic_dim > 0 else obs_dim
    critic_obs_example = torch.zeros((num_envs, critic_obs_dim), device=collector_device)
    critic_td_example = TensorDict({"policy": critic_obs_example}, batch_size=num_envs)
    critic_cfg = deepcopy(cfg.get("critic") or cfg.get("actor") or {})
    critic_cls = resolve_callable(critic_cfg.pop("class_name", "rsl_rl.models.MLPModel"))
    critic_cfg.pop("num_actions", None)
    critic_cfg.pop("distribution_cfg", None)
    critic = critic_cls(
        critic_td_example,
        cfg.get("obs_groups", {"critic": {"policy": critic_obs_dim}}),
        "critic",
        1,
        **critic_cfg,
    )
    critic = critic.to(collector_device)
    critic.eval()
    # Load initial weights
    actor_sd = dict(actor.state_dict())
    actor_weight_sync.read_weights_into(actor_sd)
    actor.load_state_dict(actor_sd)
    local_actor_weight_version = actor_weight_sync.version

    critic_sd = dict(critic.state_dict())
    critic_weight_sync.read_weights_into(critic_sd)
    critic.load_state_dict(critic_sd)
    local_critic_weight_version = critic_weight_sync.version

    # Reset environment using its authoritative tensor device.
    obs_out, _ = env.reset(None)
    obs_t = obs_out["obs"]
    if not isinstance(obs_t, torch.Tensor):
        raise TypeError(f"APPO actor observation must be torch.Tensor, got {type(obs_t).__name__}")
    obs_t = obs_t.to(device=collector_device, dtype=torch.float32)
    critic_obs_t = obs_out.get("critic")
    if critic_obs_t is not None:
        critic_obs_t = critic_obs_t.to(device=collector_device, dtype=torch.float32)

    # Pre-allocate actor inference TensorDict once. The persistent actor input
    # moves once to the collector device at each observation boundary.
    obs_td = TensorDict({"policy": obs_t}, batch_size=num_envs, device=collector_device)

    total_steps = 0
    # Bounded rolling window of the most recent completed episodes; an
    # unbounded list here grows for the entire run.
    ep_rewards: deque[float] = deque(maxlen=100)
    ep_lengths: deque[int] = deque(maxlen=100)
    current_ep_rewards = torch.zeros(num_envs, dtype=torch.float32, device=obs_t.device)
    current_ep_lengths = torch.zeros(num_envs, dtype=torch.int64, device=obs_t.device)
    reward_component_names = tuple(
        getattr(getattr(env, "algo_capabilities", None), "reward_component_names", ())
    )
    reward_component_sums = (
        torch.zeros(len(reward_component_names), dtype=torch.float32, device=obs_t.device)
        if reward_component_names
        else None
    )

    # Episode completion counters (reset after each metrics report)
    ep_timeouts = 0
    ep_completions = 0

    # Collector timing EMA (milliseconds, α=0.1 → slow-moving average)
    _EMA = 0.1
    ema_mlp_infer_ms: float = 0.0
    ema_env_step_ms: float = 0.0
    latest_rollout_ms: float | None = None
    ema_env_step_breakdown_ms: dict[str, float] = {}

    try:
        while not stop_event.is_set():
            t_rollout_start = time.perf_counter()
            # Pull latest weights from learner
            if actor_weight_sync.version > local_actor_weight_version:
                actor_sd = dict(actor.state_dict())
                local_actor_weight_version = actor_weight_sync.read_weights_into(actor_sd)
                actor.load_state_dict(actor_sd)
            if critic_weight_sync.version > local_critic_weight_version:
                critic_sd = dict(critic.state_dict())
                local_critic_weight_version = critic_weight_sync.read_weights_into(critic_sd)
                critic.load_state_dict(critic_sd)

            # Collect one rollout of length steps_per_env
            write_buf = ring_buffer.write_buffer
            for step in range(steps_per_env):
                t_mlp = time.perf_counter()
                with torch.no_grad():
                    actions_torch = actor(obs_td, stochastic_output=True)
                    log_probs_torch = actor.get_output_log_prob(actions_torch)
                ema_mlp_infer_ms = (1 - _EMA) * ema_mlp_infer_ms + _EMA * (
                    (time.perf_counter() - t_mlp) * 1000
                )

                write_buf["obs"][:, step].copy_(obs_t, non_blocking=False)
                if critic_obs_t is not None:
                    write_buf["critic"][:, step].copy_(critic_obs_t, non_blocking=False)
                write_buf["actions"][:, step].copy_(actions_torch, non_blocking=False)
                write_buf["log_probs"][:, step].copy_(
                    log_probs_torch.reshape(num_envs), non_blocking=False
                )

                # Move only when the actor and environment devices differ. The
                # ring-buffer write above is already the durable action copy.
                action_env = actions_torch.to(device=env.device, dtype=torch.float32)
                if not action_env.is_contiguous():
                    action_env = action_env.contiguous()
                if action_env.shape != (num_envs, action_dim):
                    raise ValueError(f"APPO action shape must be {(num_envs, action_dim)}")
                t_env = time.perf_counter()
                state = env.step(action_env)
                ema_env_step_ms = (1 - _EMA) * ema_env_step_ms + _EMA * (
                    (time.perf_counter() - t_env) * 1000
                )
                for key, value in extract_env_step_breakdown_timing_ms(state.info).items():
                    ema_env_step_breakdown_ms[key] = (1 - _EMA) * ema_env_step_breakdown_ms.get(
                        key, 0.0
                    ) + _EMA * value

                next_obs = state.obs
                next_obs_t = next_obs["obs"]
                next_critic_t = next_obs.get("critic")
                required_state = {
                    "obs": next_obs_t,
                    "reward": state.reward,
                    "terminated": state.terminated,
                    "truncated": state.truncated,
                }
                if next_critic_t is not None:
                    required_state["critic"] = next_critic_t
                for label, value in required_state.items():
                    if not isinstance(value, torch.Tensor):
                        raise TypeError(
                            f"APPO env {label} must be torch.Tensor, got {type(value).__name__}"
                        )
                if state.terminated.dtype != torch.bool or state.truncated.dtype != torch.bool:
                    raise TypeError("APPO terminated/truncated tensors must be torch.bool")
                if not state.reward.is_floating_point():
                    raise TypeError("APPO reward tensor must be floating-point")

                reward_t = state.reward
                if reward_t.device != obs_t.device or reward_t.dtype != torch.float32:
                    reward_t = reward_t.to(device=obs_t.device, dtype=torch.float32)
                reward_t = reward_t.reshape(num_envs)
                reward_components = state.info.get("reward_components")
                if reward_component_sums is not None:
                    if not isinstance(reward_components, torch.Tensor):
                        raise TypeError(
                            "APPO env reward_components must be torch.Tensor when "
                            "reward component names are declared"
                        )
                    if (
                        reward_components.device != reward_component_sums.device
                        or reward_components.dtype != torch.float32
                    ):
                        reward_components = reward_components.to(
                            device=reward_component_sums.device, dtype=torch.float32
                        )
                    reward_component_sums.add_(reward_components.sum(dim=0))
                truncated_t = state.truncated
                if truncated_t.device != obs_t.device:
                    truncated_t = truncated_t.to(device=obs_t.device)
                truncated_t = truncated_t.float()
                combined_done = state.terminated | state.truncated
                if combined_done.device != obs_t.device:
                    combined_done = combined_done.to(device=obs_t.device)
                combined_done_t = combined_done.float()
                final_observation = state.final_observation
                final_obs = (
                    final_observation["obs"]
                    if final_observation is not None and "obs" in final_observation
                    else None
                )
                final_critic = (
                    final_observation["critic"]
                    if final_observation is not None and "critic" in final_observation
                    else None
                )
                timeout_mask = timeout_bootstrap_mask(state.terminated, state.truncated)
                if timeout_mask.device != obs_t.device:
                    timeout_mask = timeout_mask.to(device=obs_t.device)
                environment_rewards = reward_t.clone()
                if bool(torch.any(timeout_mask)):
                    bootstrap_critic = final_critic if final_critic is not None else next_critic_t
                    if bootstrap_critic is None:
                        bootstrap_critic = next_obs_t
                    reward_t += compute_timeout_bootstrap_correction(
                        critic=critic,
                        collector_device=collector_device,
                        gamma=float(cfg["algorithm"].get("gamma", 0.99)),
                        timeout_mask=timeout_mask,
                        final_obs=final_obs if final_obs is not None else next_obs_t,
                        final_critic=bootstrap_critic,
                    )

                write_buf["rewards"][:, step].copy_(reward_t, non_blocking=False)
                write_buf["dones"][:, step].copy_(combined_done_t, non_blocking=False)
                write_buf["truncated"][:, step].copy_(truncated_t, non_blocking=False)

                total_steps += num_envs
                current_ep_rewards += environment_rewards
                current_ep_lengths += 1
                done_rows = torch.nonzero(combined_done_t > 0.5, as_tuple=False).flatten()
                if done_rows.numel() > 0:
                    ep_rewards.extend(current_ep_rewards[done_rows].detach().cpu().tolist())
                    ep_lengths.extend(current_ep_lengths[done_rows].detach().cpu().tolist())
                    current_ep_rewards[done_rows] = 0.0
                    current_ep_lengths[done_rows] = 0
                    ep_completions += int(done_rows.numel())
                    ep_timeouts += int(torch.sum(truncated_t[done_rows] > 0.5).item())

                obs_t, critic_obs_t = advance_rollout_observations(
                    next_obs_t, next_critic_t, collector_device
                )
                obs_td["policy"].copy_(obs_t)

            write_rollout_bootstrap_observations(write_buf, obs_t, critic_obs_t)
            ring_buffer.signal_write_done()  # atomic increment, non-blocking
            latest_rollout_ms = (time.perf_counter() - t_rollout_start) * 1000

            if metrics_queue is not None:
                try:
                    msg: dict[str, Any] = {"total_steps": total_steps}
                    if ep_rewards:
                        msg["return_mean_ep100"] = statistics.mean(ep_rewards)
                        msg["mean_episode_length"] = (
                            statistics.mean(ep_lengths) if ep_lengths else 0.0
                        )
                    if ep_completions > 0:
                        msg["timeout_rate"] = ep_timeouts / ep_completions
                        ep_timeouts = 0
                        ep_completions = 0
                    collector_timing_ms = {
                        "mlp_infer_ms": ema_mlp_infer_ms,
                        "env_step_ms": ema_env_step_ms,
                        **ema_env_step_breakdown_ms,
                    }
                    if latest_rollout_ms is not None:
                        collector_timing_ms["rollout_ms"] = latest_rollout_ms
                    msg["collector_timing_ms"] = collector_timing_ms
                    if reward_component_sums is not None:
                        component_means = (
                            (reward_component_sums.detach() / float(steps_per_env * num_envs))
                            .cpu()
                            .tolist()
                        )
                        msg["reward_components"] = dict(
                            zip(reward_component_names, component_means, strict=True)
                        )
                        reward_component_sums.zero_()
                    put_latest_metrics(metrics_queue, msg, worker_name="APPOWorker")
                except Exception as e:
                    print(
                        f"[APPOWorker] metrics build error: {type(e).__name__}: {e}",
                        file=sys.stderr,
                    )

    except Exception:
        stop_event.set()
        raise

    ring_buffer.close()
    actor_weight_sync.close()
    critic_weight_sync.close()
    env.close()
