"""Environment and replay collector for learner-owned off-policy inference."""

import queue
import sys
import time
from typing import Any, cast

import numpy as np
import torch

from unilab.algos.torch.common.collector_timing import extract_env_step_breakdown_timing_ms
from unilab.algos.torch.offpolicy.thread_budget import apply_torch_thread_runtime
from unilab.base.final_observation import resolve_terminal_observation_contract
from unilab.base.observations import split_obs_dict
from unilab.base.registry import ensure_registries
from unilab.training.seed import apply_training_seed

# Exclusive phases for one collector loop iteration (one vectorized env.step).
# Every key is recorded once per iteration so the reported averages share one
# denominator and can be summed without double counting.
# - replay_write_ms: pack transitions and write them into the bounded ingress
COLLECTOR_TIMING_KEYS = (
    "inference_request_ms",
    "learner_action_wait_ms",
    "env_step_ms",
    "replay_write_ms",
)
COLLECTOR_ACTIVE_TIMING_KEYS = tuple(
    key for key in COLLECTOR_TIMING_KEYS if key != "learner_action_wait_ms"
)


def sample_offpolicy_actions(
    actor,
    algo_type: str,
    obs_torch: torch.Tensor,
    prev_dones_torch: torch.Tensor,
    priv_info_torch: torch.Tensor | None = None,
) -> torch.Tensor:
    """Sample actions using the algorithm's exploration policy."""
    if algo_type in ("sac", "td3", "flashsac"):
        return cast(
            torch.Tensor,
            actor.explore(obs_torch, dones=prev_dones_torch, deterministic=False),
        )
    if algo_type == "hora_sac":
        if priv_info_torch is None:
            raise ValueError("HORA-SAC action sampling requires priv_info_torch.")
        return cast(
            torch.Tensor,
            actor.explore(obs_torch, priv_info_torch, deterministic=False),
        )
    raise ValueError(f"Unsupported off-policy algo_type for learner action sampling: {algo_type}")


def resolve_offpolicy_actor_priv_info(
    *,
    algo_type: str,
    obs_np: np.ndarray,
    critic_np: np.ndarray,
    info: dict | None,
) -> np.ndarray | None:
    """Resolve optional actor context for privileged off-policy actors."""
    if algo_type != "hora_sac":
        return None

    from unilab.algos.torch.hora.observations import split_hora_obs_with_priv_info

    _, _, priv_info_np = split_hora_obs_with_priv_info(
        {"obs": obs_np, "critic": critic_np},
        info,
    )
    if priv_info_np is None:
        raise ValueError(
            "HORA-SAC requires privileged info from info['critic_info'] "
            "or the critic observation tail."
        )
    return np.asarray(priv_info_np, dtype=np.float32)


def _record_timing_ms(timing_accum_ms, timing_counts, key: str, value: float) -> None:
    timing_accum_ms[key] += float(value)
    timing_counts[key] += 1


def _record_phase_ms(cycle_timing_ms: dict[str, float], key: str, start_ns: int) -> int:
    end_ns = time.perf_counter_ns()
    cycle_timing_ms[key] += (end_ns - start_ns) / 1e6
    return end_ns


def compute_collector_active_steps_per_sec(
    collector_timing_ms: dict[str, float],
    *,
    num_envs: int,
) -> float | None:
    """Return collector active throughput excluding explicit wait/coordination time."""
    active_ms = sum(
        float(collector_timing_ms.get(key, 0.0)) for key in COLLECTOR_ACTIVE_TIMING_KEYS
    )
    if active_ms <= 0.0:
        return None
    return int(num_envs) / (active_ms / 1000.0)


def _publish_inference_tick(
    coordination_queue,
    tick_id: int,
    stop_event,
    *,
    timeout: float = 30.0,
) -> bool:
    deadline = time.monotonic() + timeout
    while not stop_event.is_set():
        if time.monotonic() >= deadline:
            raise TimeoutError(f"Timed out publishing off-policy inference tick {tick_id}")
        try:
            coordination_queue.put(int(tick_id), timeout=0.1)
            return True
        except queue.Full:
            continue
    return False


def _wait_for_inference_tick(
    coordination_queue,
    tick_id: int,
    stop_event,
    *,
    timeout: float = 30.0,
) -> bool:
    deadline = time.monotonic() + timeout
    while not stop_event.is_set():
        if time.monotonic() >= deadline:
            raise TimeoutError(f"Timed out waiting for off-policy inference tick {tick_id}")
        try:
            received_tick = int(coordination_queue.get(timeout=0.1))
        except queue.Empty:
            continue
        if received_tick != int(tick_id):
            raise RuntimeError(
                f"Off-policy inference tick mismatch: expected {tick_id}, got {received_tick}"
            )
        return True
    return False


def off_policy_collector_fn(
    stop_event,
    env_name: str,
    num_envs: int,
    replay_buffer,
    inference_slot,
    inference_request_queue,
    inference_response_queue,
    algo_type: str = "sac",
    metrics_queue=None,
    sim_backend: str = "mujoco",
    env_cfg_override: dict | None = None,
    seed: int | None = None,
    trace_enabled: bool = False,
    trace_thread_time: bool = False,
    nan_guard_cfg=None,
    torch_thread_runtime=None,
):
    """Entry point for the off-policy collector subprocess.

    Error handling is provided by ``_collector_entry_wrapper`` in
    ``async_runner.py``.
    """
    _run_collector(
        stop_event=stop_event,
        env_name=env_name,
        num_envs=num_envs,
        replay_buffer=replay_buffer,
        inference_slot=inference_slot,
        inference_request_queue=inference_request_queue,
        inference_response_queue=inference_response_queue,
        algo_type=algo_type,
        metrics_queue=metrics_queue,
        sim_backend=sim_backend,
        env_cfg_override=env_cfg_override,
        seed=seed,
        trace_enabled=trace_enabled,
        trace_thread_time=trace_thread_time,
        nan_guard_cfg=nan_guard_cfg,
        torch_thread_runtime=torch_thread_runtime,
    )


def _run_collector(
    stop_event,
    env_name,
    num_envs,
    replay_buffer,
    inference_slot,
    inference_request_queue,
    inference_response_queue,
    algo_type,
    metrics_queue,
    sim_backend,
    env_cfg_override,
    seed,
    trace_enabled,
    trace_thread_time,
    nan_guard_cfg=None,
    torch_thread_runtime=None,
):
    from unilab.base import registry

    apply_torch_thread_runtime(torch_thread_runtime, role="collector", torch_module=torch)
    ensure_registries()
    apply_training_seed(seed, torch_runtime=False, cuda=False)

    trace_recorder = None
    if trace_enabled:
        from unilab.logging.trace_event import TraceRecorder

        trace_recorder = TraceRecorder("offpolicy_collector")

    # Initialize environment
    env: Any = registry.make(
        env_name, num_envs=num_envs, sim_backend=sim_backend, env_cfg_override=env_cfg_override
    )
    if nan_guard_cfg is not None and nan_guard_cfg.enabled:
        from unilab.utils.nan_guard import NanGuard

        env.set_nan_guard(
            NanGuard(
                nan_guard_cfg,
                num_envs=env.num_envs,
                supports_state_playback=env.play_capabilities.supports_physics_state_playback,
            )
        )
    if env.state is None:
        env.init_state()

    replay_buffer.trace_recorder = trace_recorder
    replay_buffer.trace_thread_time = trace_thread_time
    replay_buffer.attach_stop_event(stop_event)
    total_steps = 0
    ep_rewards = []
    ep_lengths = []
    current_ep_rewards = np.zeros(num_envs, dtype=np.float32)
    current_ep_lengths = np.zeros(num_envs, dtype=np.int32)
    from collections import defaultdict

    ep_reward_components = defaultdict(list)
    timing_accum_ms: defaultdict[str, float] = defaultdict(float)
    timing_counts: defaultdict[str, int] = defaultdict(int)
    done_count_window = 0
    timeout_count_window = 0

    state = env.state
    assert state is not None
    obs_np, critic_np = split_obs_dict(state.obs)
    obs_np = np.asarray(obs_np, dtype=np.float32)
    critic_np = np.asarray(critic_np, dtype=np.float32)
    info_dict = state.info
    prev_dones_np = np.zeros(num_envs, dtype=np.float32)
    import time as _time

    runtime_manifest = {
        "inference_owner": "learner",
        "actor_owned": False,
        "weight_sync_attached": False,
        "torch_inference": False,
        "cuda_context_initialized": bool(torch.cuda.is_initialized()),
    }
    if trace_recorder:
        manifest_ns = _time.perf_counter_ns()
        trace_recorder.add_slice(
            "collector/runtime_manifest",
            category="collector",
            start_ns=manifest_ns,
            end_ns=manifest_ns,
            args=runtime_manifest,
        )
    if metrics_queue is not None:
        manifest_message: dict[str, Any] = {"runtime_manifest": runtime_manifest}
        if trace_recorder:
            manifest_message["trace_events"] = trace_recorder.drain_events()
        try:
            metrics_queue.put_nowait(manifest_message)
        except queue.Full:
            pass

    inference_tick = 0
    # Collection loop
    try:
        while not stop_event.is_set():
            cycle_timing_ms: dict[str, float] = dict.fromkeys(COLLECTOR_TIMING_KEYS, 0.0)
            phase_start_ns = _time.perf_counter_ns()

            actor_context_np = resolve_offpolicy_actor_priv_info(
                algo_type=algo_type,
                obs_np=obs_np,
                critic_np=critic_np,
                info=info_dict,
            )
            actor_input_np = (
                np.concatenate((obs_np, actor_context_np), axis=1)
                if actor_context_np is not None
                else obs_np
            )
            request_ns = _time.perf_counter_ns()
            inference_slot.publish_observation(
                tick_id=inference_tick,
                observations=actor_input_np,
                dones=prev_dones_np,
            )
            if not _publish_inference_tick(
                inference_request_queue,
                inference_tick,
                stop_event,
            ):
                break
            if trace_recorder:
                trace_recorder.add_slice(
                    "collector/inference_request",
                    category="collector",
                    start_ns=request_ns,
                    end_ns=_time.perf_counter_ns(),
                    args={"tick_id": inference_tick},
                )
            phase_start_ns = _record_phase_ms(
                cycle_timing_ms, "inference_request_ms", phase_start_ns
            )
            wait_ns = _time.perf_counter_ns()
            if not _wait_for_inference_tick(
                inference_response_queue,
                inference_tick,
                stop_event,
            ):
                break
            actions_np, policy_version = inference_slot.consume_action(tick_id=inference_tick)
            if trace_recorder:
                trace_recorder.add_slice(
                    "collector/wait_for_learner_action",
                    category="collector",
                    start_ns=wait_ns,
                    end_ns=_time.perf_counter_ns(),
                    args={
                        "tick_id": inference_tick,
                        "policy_version": policy_version,
                    },
                )
            phase_start_ns = _record_phase_ms(
                cycle_timing_ms, "learner_action_wait_ms", phase_start_ns
            )
            inference_tick += 1

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
            phase_start_ns = _record_phase_ms(cycle_timing_ms, "env_step_ms", phase_start_ns)
            cycle_timing_ms.update(extract_env_step_breakdown_timing_ms(state.info))

            # Extract data as numpy
            next_obs_np, next_critic_np = split_obs_dict(state.obs)
            next_obs_np = np.asarray(next_obs_np, dtype=np.float32)
            next_critic_np = np.asarray(next_critic_np, dtype=np.float32)
            rewards_np = np.asarray(state.reward, dtype=np.float32).ravel()

            truncated_np = state.truncated.astype(np.float32, copy=False).ravel()
            combined_dones = (
                (state.terminated | state.truncated).astype(np.float32, copy=False).ravel()
            )
            prev_dones_np = combined_dones
            done_mask_np = combined_dones > 0.5
            timeout_mask_np = truncated_np > 0.5

            done_count_window += int(np.count_nonzero(done_mask_np))
            timeout_count_window += int(np.count_nonzero(timeout_mask_np))

            terminal_contract = resolve_terminal_observation_contract(
                next_obs_batch_size=next_obs_np.shape[0],
                final_observation=state.final_observation,
                done=done_mask_np,
                info=state.info,
                truncated=truncated_np,
            )
            phase_start_ns = _record_phase_ms(cycle_timing_ms, "replay_write_ms", phase_start_ns)

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
            phase_start_ns = _record_phase_ms(cycle_timing_ms, "replay_write_ms", phase_start_ns)

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
            info_dict = state.info
            total_steps += num_envs

            # Extract reward components from env info
            log_info = state.info.get("log", {})
            if log_info:
                for k, v in log_info.items():
                    if k.startswith("reward/"):
                        ep_reward_components[k].append(v)

            # Send metrics periodically
            if metrics_queue is not None and total_steps % (num_envs * 10) == 0:
                import statistics

                try:
                    msg = {
                        "total_steps": total_steps,
                        "buffer_size": int(replay_buffer.size[0]),
                    }
                    if ep_rewards:
                        msg["mean_ep_reward"] = statistics.mean(ep_rewards[-100:])
                        msg["mean_ep_length"] = (
                            statistics.mean(ep_lengths[-100:]) if ep_lengths else 0.0
                        )
                    # Add mean reward components
                    if ep_reward_components:
                        components_mean = {}
                        for k, vals in ep_reward_components.items():
                            if vals:
                                components_mean[k] = statistics.mean(vals)
                        msg["reward_components"] = components_mean
                        ep_reward_components.clear()  # reset after sending

                    if timing_counts:
                        msg["collector_timing_ms"] = {
                            k: (v / timing_counts[k])
                            for k, v in timing_accum_ms.items()
                            if timing_counts[k] > 0
                        }
                        collector_active_steps_per_sec = compute_collector_active_steps_per_sec(
                            msg["collector_timing_ms"],
                            num_envs=num_envs,
                        )
                        if collector_active_steps_per_sec is not None:
                            msg["collector_active_steps_per_sec"] = collector_active_steps_per_sec

                    if done_count_window > 0:
                        msg["timeout_rate"] = timeout_count_window / done_count_window
                        done_count_window = 0
                        timeout_count_window = 0

                    if trace_recorder:
                        msg["trace_events"] = trace_recorder.drain_events()

                    metrics_queue.put_nowait(msg)
                    if "collector_timing_ms" in msg:
                        timing_accum_ms.clear()
                        timing_counts.clear()
                except Exception as e:
                    print(f"[OffPolicyWorker] metrics enqueue error: {e}", file=sys.stderr)
            for key, value in cycle_timing_ms.items():
                _record_timing_ms(timing_accum_ms, timing_counts, key, value)

    finally:
        if metrics_queue is not None and trace_recorder:
            try:
                metrics_queue.put_nowait({"trace_events": trace_recorder.drain_events()})
            except Exception:
                pass
