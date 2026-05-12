"""Off-policy collector for SAC and TD3.

Collects (obs, action, reward, next_obs, done) transitions using the current
actor policy. Runs in a subprocess; writes to ReplayBuffer.
"""

import queue
import sys
import time
from typing import cast

import numpy as np
import torch

from unilab.algos.torch.common.actor_factory import build_actor
from unilab.base.final_observation import resolve_terminal_observation_contract
from unilab.base.observations import get_obs_dims, split_obs_dict
from unilab.base.registry import ensure_registries
from unilab.training.seed import apply_training_seed


def resolve_collector_actor_dims(
    env,
    obs_dim: int | None = None,
    action_dim: int | None = None,
) -> tuple[int, int]:
    """Resolve actor dims for the collector.

    Prefer explicit dims from the parent process so learner and collector
    build identical actor shapes on override-heavy env paths.
    """
    if obs_dim is None:
        obs_dim, _ = get_obs_dims(env.obs_groups_spec)

    if action_dim is None:
        assert env.action_space.shape is not None
        action_dim = env.action_space.shape[0]

    assert obs_dim is not None
    assert action_dim is not None
    return obs_dim, action_dim


def sample_offpolicy_actions(
    actor,
    algo_type: str,
    obs_torch: torch.Tensor,
    prev_dones_torch: torch.Tensor,
) -> torch.Tensor:
    """Sample collector actions using the algorithm's exploration policy."""
    if algo_type in ("sac", "td3", "flashsac"):
        return cast(
            torch.Tensor,
            actor.explore(obs_torch, dones=prev_dones_torch, deterministic=False),
        )
    raise ValueError(f"Unsupported off-policy algo_type for collector action sampling: {algo_type}")


def _collector_pack_shared_batch(replay_buffer, request: dict, shared_slots) -> dict:
    tick_id = int(request["tick_id"])
    snapshot_ptr = int(replay_buffer.ptr[0])
    snapshot_size = int(replay_buffer.size[0])
    sample_seed = int(request["sample_seed"])
    sample_count = int(request["sample_count"])
    shared_slot = int(request["shared_slot"])
    target_gpu_slot = int(request["target_gpu_slot"])
    learner_hot_gpu_slot = int(request["learner_hot_gpu_slot"])
    if target_gpu_slot == learner_hot_gpu_slot:
        raise RuntimeError(
            "collector_thread pack target_gpu_slot must differ from learner_hot_gpu_slot"
        )
    gen = torch.Generator(device="cpu")
    gen.manual_seed(sample_seed)
    indices = torch.randint(0, snapshot_size, (sample_count,), generator=gen)
    dst = shared_slots[shared_slot]
    pack_begin_ns = time.perf_counter_ns()
    torch.index_select(replay_buffer._storage, 0, indices, out=dst)
    pack_end_ns = time.perf_counter_ns()
    return {
        "tick_id": tick_id,
        "snapshot_ptr": snapshot_ptr,
        "snapshot_size": snapshot_size,
        "sample_seed": sample_seed,
        "sample_count": sample_count,
        "shared_slot": shared_slot,
        "target_gpu_slot": target_gpu_slot,
        "learner_hot_gpu_slot": learner_hot_gpu_slot,
        "pack_layout": "packed",
        "pack_executor": "collector_thread",
        "pack_begin_ns": pack_begin_ns,
        "pack_end_ns": pack_end_ns,
    }


def _service_collector_pack_requests(
    replay_buffer,
    request_queue,
    ready_queue,
    shared_slots,
    trace_recorder=None,
    *,
    block_timeout: float = 0.0,
    pending_request: dict | None = None,
) -> tuple[bool, dict | None]:
    if request_queue is None or ready_queue is None or shared_slots is None:
        return False, pending_request
    request = pending_request
    if request is None:
        try:
            request = (
                request_queue.get(timeout=block_timeout)
                if block_timeout > 0
                else request_queue.get_nowait()
            )
        except queue.Empty:
            return False, None
    if request is None:
        return False, None
    min_snapshot_ptr = int(request.get("min_snapshot_ptr", 0))
    if int(replay_buffer.ptr[0]) < min_snapshot_ptr:
        return False, request
    ready = _collector_pack_shared_batch(replay_buffer, request, shared_slots)
    if trace_recorder:
        trace_recorder.add_slice(
            "collector/cpu_pack_sample_batch",
            category="collector",
            start_ns=int(ready["pack_begin_ns"]),
            end_ns=int(ready["pack_end_ns"]),
            args={
                "tick_id": int(ready["tick_id"]),
                "sample_count": int(ready["sample_count"]),
                "shared_slot": int(ready["shared_slot"]),
                "target_gpu_slot": int(ready["target_gpu_slot"]),
                "learner_hot_gpu_slot": int(ready["learner_hot_gpu_slot"]),
                "pack_layout": "packed",
                "pack_executor": "collector_thread",
                "shared_memory": True,
                "pinned_memory": False,
            },
        )
    ready_queue.put(ready)
    return True, None


def off_policy_collector_fn(
    stop_event,
    env_name: str,
    num_envs: int,
    replay_buffer,
    weight_sync_name: str,
    weight_param_shapes: dict,
    algo_type: str = "sac",
    actor_hidden_dim: int = 512,
    use_layer_norm: bool = True,
    learning_starts: int = 0,
    metrics_queue=None,
    weight_sync_lock=None,
    sync_collection: bool = False,
    collection_ready_queue=None,
    trainer_done_queue=None,
    env_steps_per_sync: int = 1,
    obs_normalization: bool = False,
    shared_obs_normalizer_stats=None,
    sim_backend: str = "mujoco",
    env_cfg_override: dict | None = None,
    obs_dim: int | None = None,
    action_dim: int | None = None,
    actor_kwargs: dict | None = None,
    seed: int | None = None,
    trace_enabled: bool = False,
    trace_thread_time: bool = False,
    collector_pack_request_queue=None,
    collector_pack_ready_queue=None,
    collector_pack_shared_slots=None,
    **kwargs,
):
    """Entry point for the off-policy collector subprocess."""
    import sys
    import traceback

    try:
        print("[Collector] Entry point called", file=sys.stderr, flush=True)
        _run_collector(
            stop_event=stop_event,
            env_name=env_name,
            num_envs=num_envs,
            replay_buffer=replay_buffer,
            weight_sync_name=weight_sync_name,
            weight_param_shapes=weight_param_shapes,
            algo_type=algo_type,
            actor_hidden_dim=actor_hidden_dim,
            use_layer_norm=use_layer_norm,
            learning_starts=learning_starts,
            metrics_queue=metrics_queue,
            weight_sync_lock=weight_sync_lock,
            sync_collection=sync_collection,
            collection_ready_queue=collection_ready_queue,
            trainer_done_queue=trainer_done_queue,
            env_steps_per_sync=env_steps_per_sync,
            obs_normalization=obs_normalization,
            shared_obs_normalizer_stats=shared_obs_normalizer_stats,
            sim_backend=sim_backend,
            env_cfg_override=env_cfg_override,
            obs_dim=obs_dim,
            action_dim=action_dim,
            actor_kwargs=actor_kwargs,
            seed=seed,
            trace_enabled=trace_enabled,
            trace_thread_time=trace_thread_time,
            collector_pack_request_queue=collector_pack_request_queue,
            collector_pack_ready_queue=collector_pack_ready_queue,
            collector_pack_shared_slots=collector_pack_shared_slots,
        )
    except Exception as e:
        print(f"[Collector] Exception: {e}", file=sys.stderr, flush=True)
        traceback.print_exc(file=sys.stderr)
        if metrics_queue is not None:
            try:
                metrics_queue.put_nowait({"error": str(e)})
            except Exception:
                pass


def _run_collector(
    stop_event,
    env_name,
    num_envs,
    replay_buffer,
    weight_sync_name,
    weight_param_shapes,
    algo_type,
    actor_hidden_dim,
    use_layer_norm,
    learning_starts,
    metrics_queue,
    weight_sync_lock,
    sync_collection,
    collection_ready_queue,
    trainer_done_queue,
    env_steps_per_sync,
    obs_normalization,
    shared_obs_normalizer_stats,
    sim_backend,
    env_cfg_override,
    obs_dim,
    action_dim,
    actor_kwargs,
    seed,
    trace_enabled,
    trace_thread_time,
    collector_pack_request_queue,
    collector_pack_ready_queue,
    collector_pack_shared_slots,
):
    del learning_starts
    from unilab.base import registry
    from unilab.ipc import SharedWeightSync

    ensure_registries()
    apply_training_seed(seed, torch_runtime=True, cuda=True)

    trace_recorder = None
    if trace_enabled:
        from unilab.logging.trace_event import TraceRecorder

        trace_recorder = TraceRecorder("offpolicy_collector")

    # Initialize environment
    env = registry.make(
        env_name, num_envs=num_envs, sim_backend=sim_backend, env_cfg_override=env_cfg_override
    )
    if env.state is None:
        env.init_state()

    # Connect to weight sync
    weight_sync = SharedWeightSync(
        weight_param_shapes, create=False, shm_name=weight_sync_name, lock=weight_sync_lock
    )
    weight_sync.trace_recorder = trace_recorder
    weight_sync.trace_thread_time = trace_thread_time

    # Build actor (always on CPU for env interaction)
    obs_dim, action_dim = resolve_collector_actor_dims(
        env,
        obs_dim=obs_dim,
        action_dim=action_dim,
    )
    actor = build_actor(
        algo_type,
        obs_dim,
        action_dim,
        actor_hidden_dim,
        use_layer_norm,
        "cpu",
        num_envs,
        **(actor_kwargs or {}),
    )
    actor.eval()
    replay_buffer.trace_recorder = trace_recorder
    replay_buffer.trace_thread_time = trace_thread_time

    # Load initial weights
    sd = dict(actor.state_dict())
    weight_sync.read_weights_into(sd)
    actor.load_state_dict(sd)
    local_weight_version = weight_sync.version

    total_steps = 0
    ep_rewards = []
    ep_lengths = []
    current_ep_rewards = np.zeros(num_envs, dtype=np.float32)
    current_ep_lengths = np.zeros(num_envs, dtype=np.int32)
    from collections import defaultdict

    ep_reward_components = defaultdict(list)
    timing_accum_ms = defaultdict(float)
    timing_count = 0
    done_count_window = 0
    timeout_count_window = 0
    terminated_count_window = 0

    # Initial step to get first observation
    actions_np = np.zeros((num_envs, action_dim), dtype=np.float32)
    state = env.step(actions_np)
    obs_np, critic_np = split_obs_dict(state.obs)
    obs_np = np.asarray(obs_np, dtype=np.float32)
    critic_np = np.asarray(critic_np, dtype=np.float32)
    prev_dones_np = np.zeros(num_envs, dtype=np.float32)
    import time as _time

    _last_log_time = _time.time()

    # Track env.step calls collected since the last learner phase.
    env_steps_since_sync = 0
    pending_collector_pack_request = None

    # Collection loop
    while not stop_event.is_set():
        # Check for weight updates
        if weight_sync.version > local_weight_version:
            _wt_ns = _time.perf_counter_ns()
            sd = dict(actor.state_dict())
            local_weight_version = weight_sync.read_weights_into(sd)
            actor.load_state_dict(sd)
            if trace_recorder:
                trace_recorder.add_slice(
                    "collector/check_weight_update",
                    category="collector",
                    start_ns=_wt_ns,
                    end_ns=_time.perf_counter_ns(),
                )

            # Update normalizer stats
            if obs_normalization and shared_obs_normalizer_stats is not None:
                stats = shared_obs_normalizer_stats.get()
                if stats is not None:
                    # Apply stats to a local normalizer if needed, or directly to actor
                    pass  # Handled by EmpiricalNormalization in learner if actor possesses it. We need a local normalizer.

        # Normalize obs_np
        obs_np_input = obs_np
        if obs_normalization and shared_obs_normalizer_stats is not None:
            stats = shared_obs_normalizer_stats.get()
            if stats is not None:
                mean, std = stats
                obs_np_input = (obs_np - mean) / (std + 1e-8)

        # Select action
        with torch.no_grad():
            _t_infer = _time.perf_counter()
            _t_infer_ns = _time.perf_counter_ns()
            obs_torch = torch.from_numpy(obs_np_input)
            dones_torch = torch.from_numpy(prev_dones_np)
            actions_torch = sample_offpolicy_actions(
                actor=actor,
                algo_type=algo_type,
                obs_torch=obs_torch,
                prev_dones_torch=dones_torch,
            )
            actions_np = actions_torch.numpy()
            timing_accum_ms["mlp_infer_ms"] += (_time.perf_counter() - _t_infer) * 1000
            if trace_recorder:
                trace_recorder.add_slice(
                    "collector/actor_infer_cpu",
                    category="collector",
                    start_ns=_t_infer_ns,
                    end_ns=_time.perf_counter_ns(),
                )

        # Step environment
        _env_ns = _time.perf_counter_ns()
        state = env.step(actions_np)
        if trace_recorder:
            trace_recorder.add_slice(
                "collector/env_step",
                category="collector",
                start_ns=_env_ns,
                end_ns=_time.perf_counter_ns(),
                args={"num_envs": num_envs},
            )

        timing_info = state.info.get("timing", {})
        if timing_info:
            for key in ("env_step_total_ms", "step_core_ms", "update_state_ms", "reset_done_ms"):
                if key in timing_info:
                    timing_accum_ms[key] += float(timing_info[key])
            timing_count += 1

        # Extract data as numpy
        next_obs_np, next_critic_np = split_obs_dict(state.obs)
        next_obs_np = np.asarray(next_obs_np, dtype=np.float32)
        next_critic_np = np.asarray(next_critic_np, dtype=np.float32)
        rewards_np = np.asarray(state.reward, dtype=np.float32).ravel()

        terminated_np = state.terminated.astype(np.float32, copy=False).ravel()
        truncated_np = state.truncated.astype(np.float32, copy=False).ravel()
        combined_dones = (state.terminated | state.truncated).astype(np.float32, copy=False).ravel()
        prev_dones_np = combined_dones
        done_mask_np = combined_dones > 0.5
        timeout_mask_np = truncated_np > 0.5
        terminated_mask_np = np.logical_and(terminated_np > 0.5, ~timeout_mask_np)

        done_count_window += int(np.count_nonzero(done_mask_np))
        timeout_count_window += int(np.count_nonzero(timeout_mask_np))
        terminated_count_window += int(np.count_nonzero(terminated_mask_np))

        terminal_contract = resolve_terminal_observation_contract(
            next_obs_batch_size=next_obs_np.shape[0],
            final_observation=state.final_observation,
            done=done_mask_np,
            info=state.info,
            truncated=truncated_np,
        )

        # ReplayBuffer `dones` follows the UniLab env lifecycle contract:
        # done = terminated | truncated. Learners use `truncated` to keep
        # bootstrap enabled for timeout/truncation rows.
        _rb_ns = _time.perf_counter_ns()
        replay_buffer.add(
            torch.from_numpy(obs_np),
            torch.from_numpy(actions_np),
            torch.from_numpy(rewards_np),
            torch.from_numpy(next_obs_np),
            torch.from_numpy(combined_dones),
            torch.from_numpy(truncated_np),
            terminal_mask=torch.from_numpy(terminal_contract.terminal_mask),
            terminal_next_obs=(
                torch.from_numpy(terminal_contract.terminal_obs)
                if terminal_contract.terminal_obs is not None
                else None
            ),
            critic=torch.from_numpy(critic_np),
            next_critic=torch.from_numpy(next_critic_np),
            terminal_next_critic=(
                torch.from_numpy(terminal_contract.terminal_critic)
                if terminal_contract.terminal_critic is not None
                else None
            ),
        )
        if trace_recorder:
            trace_recorder.add_slice(
                "collector/replay_add",
                category="collector",
                start_ns=_rb_ns,
                end_ns=_time.perf_counter_ns(),
            )
        _, pending_collector_pack_request = _service_collector_pack_requests(
            replay_buffer,
            collector_pack_request_queue,
            collector_pack_ready_queue,
            collector_pack_shared_slots,
            trace_recorder,
            block_timeout=0.0,
            pending_request=pending_collector_pack_request,
        )

        # Track episode rewards - vectorized
        current_ep_rewards += rewards_np
        current_ep_lengths += 1
        reset_mask = combined_dones > 0.5
        reset_indices = np.where(reset_mask)[0]
        if len(reset_indices) > 0:
            ep_rewards.extend(current_ep_rewards[reset_indices].tolist())
            ep_lengths.extend(current_ep_lengths[reset_indices].tolist())
            current_ep_rewards[reset_indices] = 0.0
            current_ep_lengths[reset_indices] = 0

        obs_np = next_obs_np
        critic_np = next_critic_np
        total_steps += num_envs
        env_steps_since_sync += 1

        # Signal the learner once this collection chunk is ready.
        if (
            sync_collection
            and collection_ready_queue is not None
            and trainer_done_queue is not None
        ):
            if env_steps_since_sync >= env_steps_per_sync:
                _sig_ns = _time.perf_counter_ns()
                collection_ready_queue.put(1)
                if trace_recorder:
                    trace_recorder.add_slice(
                        "collector/signal_ready",
                        category="collector",
                        start_ns=_sig_ns,
                        end_ns=_time.perf_counter_ns(),
                    )
                _wait_ns = _time.perf_counter_ns()
                while not stop_event.is_set():
                    _, pending_collector_pack_request = _service_collector_pack_requests(
                        replay_buffer,
                        collector_pack_request_queue,
                        collector_pack_ready_queue,
                        collector_pack_shared_slots,
                        trace_recorder,
                        block_timeout=0.0,
                        pending_request=pending_collector_pack_request,
                    )
                    try:
                        trainer_done_queue.get(timeout=0.001)
                        _, pending_collector_pack_request = _service_collector_pack_requests(
                            replay_buffer,
                            collector_pack_request_queue,
                            collector_pack_ready_queue,
                            collector_pack_shared_slots,
                            trace_recorder,
                            block_timeout=0.0,
                            pending_request=pending_collector_pack_request,
                        )
                        break
                    except queue.Empty:
                        continue
                if trace_recorder:
                    trace_recorder.add_slice(
                        "collector/wait_trainer_done",
                        category="collector",
                        start_ns=_wait_ns,
                        end_ns=_time.perf_counter_ns(),
                    )
                    if metrics_queue is not None:
                        try:
                            metrics_queue.put_nowait(
                                {"trace_events": trace_recorder.drain_events()}
                            )
                        except Exception:
                            pass
                env_steps_since_sync = 0
        elif env_steps_since_sync >= env_steps_per_sync:
            env_steps_since_sync = 0

        # Progress log every 2 seconds
        now = _time.time()
        if now - _last_log_time > 2.0:
            _last_log_time = now

        # Extract reward components from env info
        log_info = state.info.get("log", {})
        if log_info:
            for k, v in log_info.items():
                if k.startswith("reward/"):
                    ep_reward_components[k].append(v)

        # Send metrics periodically
        if metrics_queue is not None and total_steps % (num_envs * 10) == 0 and ep_rewards:
            import statistics

            try:
                msg = {
                    "total_steps": total_steps,
                    "mean_ep_reward": statistics.mean(ep_rewards[-100:]),
                    "mean_ep_length": statistics.mean(ep_lengths[-100:]) if ep_lengths else 0.0,
                    "buffer_size": int(replay_buffer.size[0]),
                }
                # Add mean reward components
                if ep_reward_components:
                    components_mean = {}
                    for k, vals in ep_reward_components.items():
                        if vals:
                            components_mean[k] = statistics.mean(vals)
                    msg["reward_components"] = components_mean
                    ep_reward_components.clear()  # reset after sending

                if timing_count > 0:
                    msg["collector_timing_ms"] = {
                        k: (v / timing_count) for k, v in timing_accum_ms.items()
                    }
                    timing_accum_ms.clear()
                    timing_count = 0

                if done_count_window > 0:
                    msg["timeout_rate"] = timeout_count_window / done_count_window
                    msg["terminated_rate"] = terminated_count_window / done_count_window
                    done_count_window = 0
                    timeout_count_window = 0
                    terminated_count_window = 0

                if trace_recorder:
                    msg["trace_events"] = trace_recorder.drain_events()

                metrics_queue.put_nowait(msg)
            except Exception as e:
                print(f"[OffPolicyWorker] metrics enqueue error: {e}", file=sys.stderr)

    if metrics_queue is not None and trace_recorder:
        try:
            metrics_queue.put_nowait({"trace_events": trace_recorder.drain_events()})
        except Exception:
            pass
    weight_sync.close()
