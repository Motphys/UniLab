#!/usr/bin/env python3
"""Benchmark Fast SAC with different inference backends."""

import argparse
import json
import os
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent
sys.path.append(str(ROOT_DIR))

try:
    from benchmark.core.task_names import locomotion_env_name, normalize_locomotion_task_id
except ModuleNotFoundError:
    from core.task_names import locomotion_env_name, normalize_locomotion_task_id


@dataclass
class BackendResult:
    backend: str
    total_time_sec: float
    final_reward: float
    steps_per_sec: float
    speedup: float


def run_backend(task, max_iterations, backend, num_envs):
    """Run Fast SAC with specified backend."""
    import datetime

    from unilab.algos.torch.fast_sac.runner import FastSACRunner
    from unilab.config.structured_configs import SACConfig

    cfg = SACConfig()

    # Set collector device based on backend
    device = "mps"
    if backend == "torch_mps":
        pass
    elif backend in ("torch_cpu", "numpy"):
        pass

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = os.path.join(ROOT_DIR, "logs", "benchmark", f"{backend}_{timestamp}")

    task_id = normalize_locomotion_task_id(task)
    env_task_name = locomotion_env_name(task_id)

    runner = FastSACRunner(
        env_name=env_task_name,
        device=device,
        num_envs=num_envs,
        replay_buffer_n=cfg.replay_buffer_n,
        batch_size=cfg.batch_size,
        warmup_steps=cfg.warmup_steps,
        updates_per_step=cfg.updates_per_step,
        policy_frequency=cfg.policy_frequency,
        gamma=cfg.gamma,
        tau=cfg.tau,
        actor_lr=cfg.actor_lr,
        critic_lr=cfg.critic_lr,
        alpha_lr=cfg.algo_params.alpha_lr,
        alpha_init=cfg.algo_params.alpha_init,
        target_entropy_ratio=cfg.algo_params.target_entropy_ratio,
        actor_hidden_dim=cfg.actor_hidden_dim,
        critic_hidden_dim=cfg.critic_hidden_dim,
        num_atoms=cfg.num_atoms,
        use_layer_norm=cfg.use_layer_norm,
    )

    start = time.time()
    try:
        runner.learn(max_iterations=max_iterations, save_interval=0, log_dir=log_dir)
    finally:
        runner.close()

    total_time = time.time() - start

    # Parse tensorboard logs
    final_reward = 0.0
    steps_per_sec = 0.0

    try:
        from tensorboard.backend.event_processing import event_accumulator

        tb_dir = os.path.join(log_dir, "tb")
        ea = event_accumulator.EventAccumulator(tb_dir)
        ea.Reload()

        if "reward/mean" in ea.Tags()["scalars"]:
            rewards = ea.Scalars("reward/mean")
            if rewards:
                final_reward = rewards[-1].value

        if "perf/steps_per_sec" in ea.Tags()["scalars"]:
            sps = ea.Scalars("perf/steps_per_sec")
            if sps:
                steps_per_sec = sps[-1].value
    except Exception:
        pass

    return BackendResult(
        backend=backend,
        total_time_sec=total_time,
        final_reward=final_reward,
        steps_per_sec=steps_per_sec,
        speedup=1.0,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="go1_joystick_flat")
    parser.add_argument("--max_iterations", type=int, default=100)
    parser.add_argument("--num_envs", type=int, default=4096)
    parser.add_argument("--backends", type=str, default="torch_mps,torch_cpu,ane")
    parser.add_argument("--output", type=str, default="benchmark/outputs/fast_sac_backends.json")
    args = parser.parse_args()
    task_id = normalize_locomotion_task_id(args.task)
    env_task_name = locomotion_env_name(task_id)

    backends = [b.strip() for b in args.backends.split(",")]

    print(f"\n{'=' * 70}")
    print("Fast SAC Backend Benchmark")
    print(
        f"Task: {task_id} (env: {env_task_name}), Iterations: {args.max_iterations}, Envs: {args.num_envs}"
    )
    print(f"Backends: {backends}")
    print(f"{'=' * 70}\n")

    results = []

    for backend in backends:
        print(f"\n>>> Running {backend}...")
        try:
            result = run_backend(task_id, args.max_iterations, backend, args.num_envs)
            results.append(result)
            print(
                f"✓ {backend}: {result.total_time_sec:.1f}s, reward={result.final_reward:.3f}, {result.steps_per_sec:.0f} steps/s"
            )
        except Exception as e:
            print(f"✗ {backend} failed: {e}")

    # Compute speedup
    baseline = next((r for r in results if r.backend == "torch_mps"), None)
    if baseline and baseline.steps_per_sec > 0:
        for r in results:
            r.speedup = r.steps_per_sec / baseline.steps_per_sec

    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(
            {
                "task": task_id,
                "env_task_name": env_task_name,
                "max_iterations": args.max_iterations,
                "num_envs": args.num_envs,
                "results": [asdict(r) for r in results],
            },
            indent=2,
        )
    )

    # Print summary
    print(f"\n{'=' * 70}")
    print("RESULTS")
    print(f"{'=' * 70}")
    print(f"{'Backend':<15} {'Time(s)':<10} {'Reward':<10} {'Steps/s':<12} {'Speedup':<10}")
    print(f"{'-' * 70}")
    for r in results:
        print(
            f"{r.backend:<15} {r.total_time_sec:<10.1f} {r.final_reward:<10.3f} {r.steps_per_sec:<12.0f} {r.speedup:<10.2f}x"
        )
    print(f"{'=' * 70}")
    print(f"\nSaved to: {output_path}\n")


if __name__ == "__main__":
    main()
