#!/usr/bin/env python3

"""Train PPO with MLX backend."""

from __future__ import annotations

import argparse
from collections import deque
import datetime
import importlib
import json
import os
import pickle
import pkgutil
import math
import statistics
import sys
import time
from pathlib import Path

import mlx.core as mx
import numpy as np
from mlx.utils import tree_map

# Add workspace root to python path dynamically
ROOT_DIR = Path(__file__).parent.parent
sys.path.append(str(ROOT_DIR))


def ensure_registries() -> None:
    """Import env modules so they are registered in `unilab.envs.registry`."""
    try:
        import unilab.envs.locomotion

        package = unilab.envs.locomotion
        if hasattr(package, "__path__"):
            for _, name, _ in pkgutil.walk_packages(package.__path__, package.__name__ + "."):
                try:
                    importlib.import_module(name)
                except Exception:
                    pass
    except ImportError:
        pass


ensure_registries()

from unilab.config import locomotion_params
from unilab.envs import registry
from unilab.utils import render_many
from unilab.algos.mlx.common import EmpiricalDiscountedVariationNormalization, RolloutBuffer
from unilab.algos.mlx.ppo import MLPActorCritic, PPOConfig, PPOTrainer

TASK_STEP_TUNING = {
    # Tuned for faster collection_time on each task.
    "Go1JoystickFlatTerrain": {"threads": "32", "chunk": "4"},
    "Go2JoystickFlatTerrain": {"threads": "56", "chunk": "16"},
    "G1JoystickFlatTerrain": {"threads": "24", "chunk": "4"},
}


class TensorboardScalarWriter:
    """Minimal scalar writer based on tensorboard event files."""

    def __init__(self, log_dir: Path) -> None:
        from tensorboard.compat.proto.event_pb2 import Event
        from tensorboard.compat.proto.summary_pb2 import Summary
        from tensorboard.summary.writer.event_file_writer import EventFileWriter

        self._Event = Event
        self._Summary = Summary
        self._writer = EventFileWriter(str(log_dir))

    def add_scalar(self, tag: str, value: float, step: int) -> None:
        summary = self._Summary(value=[self._Summary.Value(tag=tag, simple_value=float(value))])
        event = self._Event(wall_time=time.time(), step=int(step), summary=summary)
        self._writer.add_event(event)

    def flush(self) -> None:
        self._writer.flush()

    def close(self) -> None:
        self._writer.close()


def get_latest_run(log_dir: Path) -> Path | None:
    """Find latest run directory under a task log root."""
    if not log_dir.exists():
        return None
    runs = sorted([p for p in log_dir.iterdir() if p.is_dir()])
    return runs[-1] if runs else None


def get_latest_checkpoint(run_dir: Path) -> Path | None:
    """Find the latest model_*.safetensors checkpoint in a run dir."""
    if not run_dir.exists():
        return None
    model_files = [p for p in run_dir.glob("model_*.safetensors") if p.is_file()]
    if not model_files:
        return None
    model_files.sort(key=lambda p: int(p.stem.split("_")[1]))
    return model_files[-1]


def save_trainer_state(path: Path, trainer: PPOTrainer, iteration: int) -> None:
    """Save optimizer state and trainer metadata for resume."""
    payload = {
        "iteration": int(iteration),
        "learning_rate": float(trainer.learning_rate),
        "optimizer_state": tree_map(lambda x: x.tolist(), trainer.optimizer.state),
    }
    with path.open("wb") as f:
        pickle.dump(payload, f)


def load_trainer_state(path: Path, trainer: PPOTrainer) -> int:
    """Load optimizer state and trainer metadata."""
    with path.open("rb") as f:
        payload = pickle.load(f)
    trainer.learning_rate = float(payload.get("learning_rate", trainer.learning_rate))
    trainer.optimizer.learning_rate = mx.array(trainer.learning_rate, dtype=mx.float32)
    trainer.optimizer.state = tree_map(lambda x: mx.array(x), payload["optimizer_state"])
    return int(payload.get("iteration", -1))


def build_model(cfg, obs_dim: int, action_dim: int) -> MLPActorCritic:
    """Build actor-critic model from locomotion config."""
    policy_cfg = cfg.policy
    init_noise_std = float(getattr(policy_cfg, "init_noise_std", 1.0))
    init_log_std = float(math.log(max(init_noise_std, 1e-6)))
    obs_norm = bool(getattr(cfg, "empirical_normalization", False))
    noise_std_type = str(getattr(policy_cfg, "noise_std_type", "scalar"))
    state_dependent_std = bool(getattr(policy_cfg, "state_dependent_std", False))
    return MLPActorCritic(
        obs_dim=obs_dim,
        action_dim=action_dim,
        actor_hidden_dims=policy_cfg.actor_hidden_dims,
        critic_hidden_dims=policy_cfg.critic_hidden_dims,
        activation=policy_cfg.activation,
        init_log_std=init_log_std,
        obs_normalization=obs_norm,
        noise_std_type=noise_std_type,
        state_dependent_std=state_dependent_std,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Train or Play PPO with MLX + NumPy only.")
    parser.add_argument("--task", type=str, required=True, help="Task name")
    parser.add_argument("--play_only", action="store_true", help="Play mode only")
    parser.add_argument("--load_run", type=str, default="-1", help="Run ID to load, run path, or model file path")
    parser.add_argument("--env_num", type=int, default=None, help="Number of parallel envs (task default if unset)")
    parser.add_argument("--play_env_num", type=int, default=16, help="Number of play envs")
    parser.add_argument("--play_steps", type=int, default=150, help="Number of steps for play video")
    parser.add_argument("--steps_per_env", type=int, default=None, help="Rollout horizon per iteration")
    parser.add_argument("--max_iterations", type=int, default=None, help="Training iterations")
    parser.add_argument("--learning_rate", type=float, default=None, help="Override learning rate")
    parser.add_argument("--seed", type=int, default=1, help="Random seed")
    parser.add_argument("--log_interval", type=int, default=10, help="Print every N iterations")
    parser.add_argument("--log_root", type=str, default="logs/mlx_rl_train", help="Root directory for training logs")
    parser.add_argument("--save_interval", type=int, default=50, help="Checkpoint save interval")
    args = parser.parse_args()
    if args.env_num is None:
        args.env_num = locomotion_params.get_default_env_num(args.task)

    mx.random.seed(args.seed)

    cfg = locomotion_params.rsl_rl_config(args.task)
    algo_cfg = cfg.algorithm
    profile_collection = os.getenv("UNILAB_PROFILE_COLLECTION", "0") == "1"

    num_steps = int(args.steps_per_env or cfg.num_steps_per_env)
    max_iterations = int(args.max_iterations or cfg.max_iterations)
    learning_rate = float(args.learning_rate or algo_cfg.learning_rate)
    save_interval = int(args.save_interval)
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_root = Path(args.log_root)
    if not log_root.is_absolute():
        log_root = ROOT_DIR / log_root
    task_log_root = log_root / args.task

    # PLAY MODE
    if args.play_only:
        play_env_num = args.play_env_num
        env = registry.make(args.task, num_envs=play_env_num, sim_backend="mujoco")
        obs_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]
        model = build_model(cfg, obs_dim, action_dim)

        load_path: Path | None = None
        if args.load_run == "-1":
            latest_run = get_latest_run(task_log_root)
            if latest_run is not None:
                load_path = get_latest_checkpoint(latest_run)
                run_dir = latest_run
            else:
                run_dir = None
        else:
            candidate = Path(args.load_run)
            if not candidate.exists():
                candidate = task_log_root / args.load_run
            if candidate.is_dir():
                load_path = get_latest_checkpoint(candidate)
                run_dir = candidate
            elif candidate.is_file():
                load_path = candidate
                run_dir = candidate.parent
            else:
                load_path = None
                run_dir = None

        if load_path is None or not load_path.exists():
            print(f"Could not find valid model checkpoint from --load_run={args.load_run}")
            sys.exit(1)

        model.load_weights(str(load_path), strict=True)
        print(f"[MLX PPO] Loaded model: {load_path}")

        if env.state is None:
            env.init_state()
        _, obs, _ = env.reset(mx.arange(env.num_envs, dtype=mx.int32))
        obs = mx.array(obs, dtype=mx.float32)

        state_list = []
        print("[MLX PPO] Collecting physics states for play...")
        for _ in range(args.play_steps):
            obs_mx = mx.array(obs, dtype=mx.float32)
            actions_mx = model.policy(obs_mx)
            actions = mx.where(mx.isfinite(actions_mx), actions_mx, mx.zeros_like(actions_mx))
            state = env.step(actions)
            raw_obs = mx.array(state.obs, dtype=mx.float32)
            bad_mask = mx.logical_not(mx.all(mx.isfinite(raw_obs), axis=1))
            if bool(mx.any(bad_mask).item()):
                bad_indices = [i for i, flag in enumerate(bad_mask.tolist()) if flag]
                _, reset_obs, _ = env.reset(mx.array(bad_indices, dtype=mx.int32))
                raw_obs_rows = raw_obs.tolist()
                reset_rows = mx.array(reset_obs, dtype=mx.float32).tolist()
                for k, idx in enumerate(bad_indices):
                    raw_obs_rows[idx] = reset_rows[k]
                raw_obs = mx.array(raw_obs_rows, dtype=mx.float32)
            obs = mx.nan_to_num(raw_obs, nan=0.0, posinf=0.0, neginf=0.0)
            # Append a copy: physics_state is updated in-place each step, so we must snapshot per frame.
            state_list.append(np.asarray(env.state.physics_state).copy())

        output_dir = run_dir if run_dir is not None else task_log_root
        output_video = output_dir / "play_video.mp4"
        print(f"[MLX PPO] Rendering video to {output_video} ...")
        frames = render_many.render_states_get_frames(
            state_list,
            env.cfg.model_file,
            width=1280,
            height=720,
            camera_id=-1,
        )
        try:
            import mediapy as media  # type: ignore[reportMissingImports]
        except ImportError:
            print("mediapy is required for play video export. Install with `pip install mediapy`.")
            env.close()
            sys.exit(1)
        media.write_video(str(output_video), frames, fps=int(1.0 / env.cfg.ctrl_dt))
        print(f"[MLX PPO] Play video saved: {output_video}")
        env.close()
        return

    # TRAIN MODE
    log_dir = task_log_root / timestamp
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file_path = log_dir / "train.log"
    log_fp = log_file_path.open("a", encoding="utf-8")

    def log(msg: str) -> None:
        print(msg)
        log_fp.write(msg + "\n")
        log_fp.flush()

    run_meta = {
        "task": args.task,
        "env_num": args.env_num,
        "steps_per_env": num_steps,
        "max_iterations": max_iterations,
        "learning_rate": learning_rate,
        "save_interval": save_interval,
        "schedule": str(getattr(algo_cfg, "schedule", "fixed")),
        "desired_kl": float(getattr(algo_cfg, "desired_kl", 0.01)),
        "reward_normalization": bool(getattr(algo_cfg, "reward_normalization", False)),
        "target_kl_stop": (
            float(getattr(algo_cfg, "target_kl_stop"))
            if getattr(algo_cfg, "target_kl_stop", None) is not None
            else None
        ),
        "adaptive_kl_beta": float(getattr(algo_cfg, "adaptive_kl_beta", 0.9)),
        "adaptive_lr_growth": float(getattr(algo_cfg, "adaptive_lr_growth", 1.2)),
        "adaptive_lr_decay": float(getattr(algo_cfg, "adaptive_lr_decay", 1.5)),
        "adaptive_lr_update_interval": int(getattr(algo_cfg, "adaptive_lr_update_interval", 1)),
        "fast_mode": bool(getattr(algo_cfg, "fast_mode", False)),
        "metrics_interval": int(getattr(algo_cfg, "metrics_interval", 1)),
        "finite_check_interval": int(getattr(algo_cfg, "finite_check_interval", 1)),
        "enable_compile": bool(getattr(algo_cfg, "enable_compile", False)),
        "warmup_strict_iters": int(getattr(algo_cfg, "warmup_strict_iters", 0)),
        "warmup_metrics_interval": int(getattr(algo_cfg, "warmup_metrics_interval", 1)),
        "warmup_finite_check_interval": int(getattr(algo_cfg, "warmup_finite_check_interval", 1)),
        "disable_finite_checks": bool(getattr(algo_cfg, "disable_finite_checks", False)),
        "seed": args.seed,
        "timestamp": timestamp,
    }
    (log_dir / "run_config.json").write_text(json.dumps(run_meta, indent=2), encoding="utf-8")

    tb_writer = None
    try:
        tb_writer = TensorboardScalarWriter(log_dir)
    except Exception as e:
        log(f"[Warning] TensorBoard disabled: {e}")

    preset = TASK_STEP_TUNING[args.task]
    os.environ["UNILAB_MLX_STEP_THREADS"] = preset["threads"]
    os.environ["UNILAB_MLX_STEP_CHUNK"] = preset["chunk"]
    env = registry.make(args.task, num_envs=args.env_num, sim_backend="mujoco")
    if bool(getattr(algo_cfg, "fast_mode", False)) and hasattr(env, "_enable_reward_log"):
        env._enable_reward_log = False
    if env.state is None:
        env.init_state()
    _, obs, _ = env.reset(mx.arange(env.num_envs, dtype=mx.int32))
    obs = mx.array(obs, dtype=mx.float32)

    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    model = build_model(cfg, obs_dim, action_dim)
    ppo_cfg = PPOConfig(
        num_learning_epochs=int(algo_cfg.num_learning_epochs),
        num_mini_batches=int(algo_cfg.num_mini_batches),
        clip_param=float(algo_cfg.clip_param),
        gamma=float(algo_cfg.gamma),
        lam=float(algo_cfg.lam),
        value_loss_coef=float(algo_cfg.value_loss_coef),
        entropy_coef=float(algo_cfg.entropy_coef),
        learning_rate=learning_rate,
        use_clipped_value_loss=bool(algo_cfg.use_clipped_value_loss),
        max_grad_norm=float(getattr(algo_cfg, "max_grad_norm", 1.0)),
        schedule=str(getattr(algo_cfg, "schedule", "fixed")),
        desired_kl=float(getattr(algo_cfg, "desired_kl", 0.01)),
        normalize_advantage_per_mini_batch=bool(getattr(algo_cfg, "normalize_advantage_per_mini_batch", False)),
        adaptive_kl_beta=float(getattr(algo_cfg, "adaptive_kl_beta", 0.9)),
        adaptive_lr_growth=float(getattr(algo_cfg, "adaptive_lr_growth", 1.2)),
        adaptive_lr_decay=float(getattr(algo_cfg, "adaptive_lr_decay", 1.5)),
        adaptive_lr_update_interval=int(getattr(algo_cfg, "adaptive_lr_update_interval", 1)),
        fast_mode=bool(getattr(algo_cfg, "fast_mode", False)),
        metrics_interval=int(getattr(algo_cfg, "metrics_interval", 1)),
        finite_check_interval=int(getattr(algo_cfg, "finite_check_interval", 1)),
        enable_compile=bool(getattr(algo_cfg, "enable_compile", False)),
        warmup_strict_iters=int(getattr(algo_cfg, "warmup_strict_iters", 0)),
        warmup_metrics_interval=int(getattr(algo_cfg, "warmup_metrics_interval", 1)),
        warmup_finite_check_interval=int(getattr(algo_cfg, "warmup_finite_check_interval", 1)),
        disable_finite_checks=bool(getattr(algo_cfg, "disable_finite_checks", False)),
        target_kl_stop=(
            float(getattr(algo_cfg, "target_kl_stop"))
            if getattr(algo_cfg, "target_kl_stop", None) is not None
            else None
        ),
    )
    trainer = PPOTrainer(model, ppo_cfg)
    use_reward_norm = bool(getattr(algo_cfg, "reward_normalization", False))
    reward_normalizer = (
        EmpiricalDiscountedVariationNormalization(gamma=ppo_cfg.gamma) if use_reward_norm else None
    )

    if args.load_run != "-1":
        resume_candidate = Path(args.load_run)
        if not resume_candidate.exists():
            resume_candidate = task_log_root / args.load_run
        if resume_candidate.is_dir():
            ckpt = get_latest_checkpoint(resume_candidate)
        elif resume_candidate.is_file():
            ckpt = resume_candidate
        else:
            ckpt = None
        if ckpt is not None and ckpt.exists():
            model.load_weights(str(ckpt), strict=True)
            log(f"[MLX PPO] resumed_from={ckpt}")
            if ckpt.stem.startswith("model_"):
                iter_id = ckpt.stem.split("_")[1]
                trainer_state_path = ckpt.with_name(f"trainer_{iter_id}.pkl")
                if trainer_state_path.exists():
                    resumed_it = load_trainer_state(trainer_state_path, trainer)
                    log(f"[MLX PPO] resumed_trainer_state={trainer_state_path} iter={resumed_it}")

    log(f"[MLX PPO] task={args.task} envs={args.env_num} steps={num_steps} iters={max_iterations}")
    log(f"[MLX PPO] run={timestamp} lr={learning_rate:.6f}")
    log(
        "[MLX PPO] perf_mode fast_mode={} metrics_interval={} finite_check_interval={} compile={}".format(
            ppo_cfg.fast_mode,
            ppo_cfg.metrics_interval,
            ppo_cfg.finite_check_interval,
            ppo_cfg.enable_compile,
        )
    )
    log(f"[MLX PPO] profile profile_collection={profile_collection}")
    log(
        "[MLX PPO] perf_warmup warmup_iters={} warmup_metrics_interval={} warmup_finite_interval={} disable_finite_checks={}".format(
            ppo_cfg.warmup_strict_iters,
            ppo_cfg.warmup_metrics_interval,
            ppo_cfg.warmup_finite_check_interval,
            ppo_cfg.disable_finite_checks,
        )
    )
    log(f"[MLX PPO] log_dir={log_dir}")
    if tb_writer is not None:
        log("[MLX PPO] tensorboard=enabled")

    episode_returns = np.zeros((args.env_num,), dtype=np.float32)
    episode_lengths = np.zeros((args.env_num,), dtype=np.int32)
    reward_window = deque(maxlen=100)
    length_window = deque(maxlen=100)
    collection_size = num_steps * args.env_num
    total_time = 0.0

    for it in range(max_iterations):
        iter_start = time.perf_counter()
        buffer = RolloutBuffer(
            num_steps=num_steps,
            num_envs=args.env_num,
            obs_dim=obs_dim,
            action_dim=action_dim,
            gamma=ppo_cfg.gamma,
            lam=ppo_cfg.lam,
        )

        collect_start = time.perf_counter()
        bad_action_count = 0
        bad_obs_count = 0
        bad_reward_count = 0
        forced_reset_count = 0
        reward_component_sums: dict[str, float] = {}
        reward_component_counts: dict[str, int] = {}
        finite_checks_enabled = not ppo_cfg.disable_finite_checks
        do_finite_checks = finite_checks_enabled and (
            it < ppo_cfg.warmup_strict_iters
            or (it % max(int(ppo_cfg.finite_check_interval), 1) == 0)
        )
        collect_reward_components = not ppo_cfg.fast_mode
        track_episode_stats = True
        model_act_time = 0.0
        env_step_total_time = 0.0
        env_step_core_time = 0.0
        env_step_postprocess_time = 0.0
        env_step_reset_time = 0.0
        env_reset_index_time = 0.0
        env_reset_call_time = 0.0
        env_reset_scatter_time = 0.0
        env_reset_info_merge_time = 0.0
        finite_check_time = 0.0
        buffer_add_time = 0.0
        episode_stats_time = 0.0
        for _ in range(num_steps):
            obs_mx = obs
            t_act0 = time.perf_counter()
            actions_mx, log_probs_mx, values_mx, action_mean_mx, action_std_mx = model.act(obs_mx)
            model_act_time += time.perf_counter() - t_act0
            actions = actions_mx
            if do_finite_checks:
                t_fin0 = time.perf_counter()
                actions = mx.where(mx.isfinite(actions_mx), actions_mx, mx.zeros_like(actions_mx))
                bad_action_count += int(actions.size - int(mx.sum(mx.isfinite(actions)).item()))
                finite_check_time += time.perf_counter() - t_fin0
            # Keep behavior actions consistent with PPO storage (match rsl-rl style).
            executed_actions = actions
            t_env0 = time.perf_counter()
            state = env.step(executed_actions)
            env_step_total_time += time.perf_counter() - t_env0
            if isinstance(state.info, dict):
                timing_info = state.info.get("timing", {})
                if isinstance(timing_info, dict):
                    env_step_core_time += float(timing_info.get("step_core_ms", 0.0)) / 1000.0
                    env_step_postprocess_time += float(timing_info.get("update_state_ms", 0.0)) / 1000.0
                    env_step_reset_time += float(timing_info.get("reset_done_ms", 0.0)) / 1000.0
                    env_reset_index_time += float(timing_info.get("reset_index_extract_ms", 0.0)) / 1000.0
                    env_reset_call_time += float(timing_info.get("reset_call_ms", 0.0)) / 1000.0
                    env_reset_scatter_time += float(timing_info.get("reset_scatter_ms", 0.0)) / 1000.0
                    env_reset_info_merge_time += float(timing_info.get("reset_info_merge_ms", 0.0)) / 1000.0

            raw_rewards = state.reward.astype(mx.float32)
            raw_dones = state.done.astype(mx.float32)
            raw_obs = state.obs.astype(mx.float32)
            if do_finite_checks:
                t_fin0 = time.perf_counter()
                bad_reward_count += int(raw_rewards.size - int(mx.sum(mx.isfinite(raw_rewards)).item()))
                bad_obs_count += int(raw_obs.size - int(mx.sum(mx.isfinite(raw_obs)).item()))

                # If any env has non-finite transition data, force reset only those envs.
                obs_bad_mask = mx.logical_not(mx.all(mx.isfinite(raw_obs), axis=1))
                rew_bad_mask = mx.logical_not(mx.isfinite(raw_rewards))
                done_bad_mask = mx.logical_not(mx.isfinite(raw_dones))
                bad_env_mask = mx.logical_or(obs_bad_mask, mx.logical_or(rew_bad_mask, done_bad_mask))
                if bool(mx.any(bad_env_mask).item()):
                    bad_indices = [i for i, flag in enumerate(bad_env_mask.tolist()) if flag]
                    forced_reset_count += len(bad_indices)
                    _, reset_obs, _ = env.reset(mx.array(bad_indices, dtype=mx.int32))
                    raw_obs_rows = raw_obs.tolist()
                    reset_rows = mx.array(reset_obs, dtype=mx.float32).tolist()
                    for k, idx in enumerate(bad_indices):
                        raw_obs_rows[idx] = reset_rows[k]
                    raw_obs = mx.array(raw_obs_rows, dtype=mx.float32)
                    bad_mask_f32 = bad_env_mask.astype(mx.float32)
                    raw_rewards = raw_rewards * (1.0 - bad_mask_f32)
                    raw_dones = mx.where(bad_env_mask, mx.ones_like(raw_dones), raw_dones)
                rewards = mx.nan_to_num(raw_rewards, nan=0.0, posinf=0.0, neginf=0.0).astype(mx.float32)
                dones = mx.where(mx.isfinite(raw_dones), raw_dones, mx.ones_like(raw_dones)).astype(mx.float32)
                next_obs = mx.nan_to_num(raw_obs, nan=0.0, posinf=0.0, neginf=0.0).astype(mx.float32)
                finite_check_time += time.perf_counter() - t_fin0
            else:
                rewards = raw_rewards.astype(mx.float32)
                dones = raw_dones.astype(mx.float32)
                next_obs = raw_obs.astype(mx.float32)
            if hasattr(state, "truncated"):
                timeouts = mx.array(state.truncated, dtype=mx.float32)
                rewards = rewards + ppo_cfg.gamma * values_mx.astype(mx.float32) * timeouts

            if collect_reward_components and hasattr(state, "info") and isinstance(state.info, dict):
                step_log = state.info.get("log", {})
                if isinstance(step_log, dict):
                    for key, value in step_log.items():
                        try:
                            scalar_value = float(value)
                        except (TypeError, ValueError):
                            continue
                        if not math.isfinite(scalar_value):
                            continue
                        reward_component_sums[key] = reward_component_sums.get(key, 0.0) + scalar_value
                        reward_component_counts[key] = reward_component_counts.get(key, 0) + 1

            rewards_mx = rewards
            if reward_normalizer is not None:
                rewards_mx = mx.squeeze(reward_normalizer(rewards_mx), axis=-1)

            t_buf0 = time.perf_counter()
            buffer.add(
                obs=obs,
                actions=actions_mx,
                log_probs=log_probs_mx,
                action_mean=action_mean_mx,
                action_std=action_std_mx,
                rewards=rewards_mx,
                dones=dones,
                values=values_mx,
            )
            buffer_add_time += time.perf_counter() - t_buf0

            if track_episode_stats:
                t_ep0 = time.perf_counter()
                rewards_np = np.asarray(rewards, dtype=np.float32)
                dones_np = np.asarray(dones, dtype=np.float32)
                episode_returns += rewards_np
                episode_lengths += 1
                done_idx = np.flatnonzero(dones_np > 0.5)
                if done_idx.size > 0:
                    done_returns = episode_returns[done_idx].astype(np.float32, copy=False)
                    done_lengths = episode_lengths[done_idx].astype(np.int32, copy=False)
                    reward_window.extend(done_returns)
                    length_window.extend(done_lengths)
                    episode_returns[done_idx] = 0.0
                    episode_lengths[done_idx] = 0
                episode_stats_time += time.perf_counter() - t_ep0

            obs = next_obs

        collect_time = time.perf_counter() - collect_start
        learn_start = time.perf_counter()
        last_values = model.value(obs)
        buffer.compute_returns_and_advantages(last_values)
        metrics = trainer.update(buffer, iteration=it)
        learn_time = time.perf_counter() - learn_start
        iter_time = time.perf_counter() - iter_start
        total_time += iter_time
        fps = int(collection_size / max(iter_time, 1e-8))
        mean_noise_std = float(mx.mean(mx.exp(model.clipped_log_std())).item())
        current_lr = float(metrics.get("learning_rate", trainer.learning_rate))
        updates_applied = float(metrics.get("updates_applied", 0.0))
        skipped_nonfinite_loss = float(metrics.get("skipped_nonfinite_loss", 0.0))
        skipped_nonfinite_grads = float(metrics.get("skipped_nonfinite_grads", 0.0))
        rolled_back_updates = float(metrics.get("rolled_back_updates", 0.0))
        skipped_nonfinite_metrics = float(metrics.get("skipped_nonfinite_metrics", 0.0))
        early_stopped_kl = float(metrics.get("early_stopped_kl", 0.0))
        clip_fraction = float(metrics.get("clip_fraction", 0.0))
        ratio_mean = float(metrics.get("ratio_mean", 0.0))
        ratio_max = float(metrics.get("ratio_max", 0.0))
        std_mean = float(metrics.get("std_mean", 0.0))
        adv_std = float(metrics.get("adv_std", 0.0))
        value_explained_variance = float(metrics.get("value_explained_variance", 0.0))
        mean_reward = float(statistics.mean(reward_window)) if reward_window else 0.0
        mean_ep_len = float(statistics.mean(length_window)) if length_window else 0.0

        if tb_writer is not None:
            # Align tags with rsl-rl logger conventions as much as possible.
            tb_writer.add_scalar("Loss/surrogate", metrics["surrogate"], it)
            tb_writer.add_scalar("Loss/value_function", metrics["value"], it)
            tb_writer.add_scalar("Loss/entropy", metrics["entropy"], it)
            tb_writer.add_scalar("Loss/approx_kl", metrics["approx_kl"], it)
            tb_writer.add_scalar("Loss/learning_rate", current_lr, it)
            tb_writer.add_scalar("Policy/mean_noise_std", mean_noise_std, it)
            tb_writer.add_scalar("Perf/total_fps", fps, it)
            tb_writer.add_scalar("Perf/collection_time", collect_time, it)
            tb_writer.add_scalar("Perf/learning_time", learn_time, it)
            tb_writer.add_scalar("Perf/iteration_time", iter_time, it)
            if profile_collection:
                tb_writer.add_scalar("Perf/model_act_time", model_act_time, it)
                tb_writer.add_scalar("Perf/env_step_total_time", env_step_total_time, it)
                tb_writer.add_scalar("Perf/env_step_core_time", env_step_core_time, it)
                tb_writer.add_scalar("Perf/env_step_postprocess_time", env_step_postprocess_time, it)
                tb_writer.add_scalar("Perf/env_step_reset_time", env_step_reset_time, it)
                tb_writer.add_scalar("Perf/env_reset_index_time", env_reset_index_time, it)
                tb_writer.add_scalar("Perf/env_reset_call_time", env_reset_call_time, it)
                tb_writer.add_scalar("Perf/env_reset_scatter_time", env_reset_scatter_time, it)
                tb_writer.add_scalar("Perf/env_reset_info_merge_time", env_reset_info_merge_time, it)
                tb_writer.add_scalar("Perf/buffer_add_time", buffer_add_time, it)
                tb_writer.add_scalar("Perf/finite_check_time", finite_check_time, it)
                tb_writer.add_scalar("Perf/episode_stats_time", episode_stats_time, it)
            tb_writer.add_scalar("Perf/non_finite_actions", float(bad_action_count), it)
            tb_writer.add_scalar("Perf/non_finite_obs", float(bad_obs_count), it)
            tb_writer.add_scalar("Perf/non_finite_rewards", float(bad_reward_count), it)
            tb_writer.add_scalar("Perf/forced_resets", float(forced_reset_count), it)
            tb_writer.add_scalar("Perf/updates_applied", updates_applied, it)
            tb_writer.add_scalar("Perf/skipped_nonfinite_loss", skipped_nonfinite_loss, it)
            tb_writer.add_scalar("Perf/skipped_nonfinite_grads", skipped_nonfinite_grads, it)
            tb_writer.add_scalar("Perf/rolled_back_updates", rolled_back_updates, it)
            tb_writer.add_scalar("Perf/skipped_nonfinite_metrics", skipped_nonfinite_metrics, it)
            tb_writer.add_scalar("Perf/early_stopped_kl", early_stopped_kl, it)
            tb_writer.add_scalar("Policy/clip_fraction", clip_fraction, it)
            tb_writer.add_scalar("Policy/ratio_mean", ratio_mean, it)
            tb_writer.add_scalar("Policy/ratio_max", ratio_max, it)
            tb_writer.add_scalar("Policy/std_mean", std_mean, it)
            tb_writer.add_scalar("Policy/adv_std", adv_std, it)
            tb_writer.add_scalar("Value/explained_variance", value_explained_variance, it)
            tb_writer.add_scalar("Train/mean_reward", mean_reward, it)
            tb_writer.add_scalar("Train/mean_episode_length", mean_ep_len, it)
            tb_writer.add_scalar("Train/mean_reward/time", mean_reward, int(total_time))
            tb_writer.add_scalar("Train/mean_episode_length/time", mean_ep_len, int(total_time))
            for key, summed in reward_component_sums.items():
                count = reward_component_counts.get(key, 0)
                if count <= 0:
                    continue
                tb_writer.add_scalar(f"{key}", summed / count, it)
            tb_writer.flush()

        if save_interval > 0 and (it % save_interval == 0 or it == max_iterations - 1):
            ckpt_path = log_dir / f"model_{it}.safetensors"
            model.save_weights(str(ckpt_path))
            trainer_state_path = log_dir / f"trainer_{it}.pkl"
            save_trainer_state(trainer_state_path, trainer, it)
            log(f"[MLX PPO] checkpoint_saved={ckpt_path}")

        if (it + 1) % args.log_interval == 0 or it == 0:
            log(
                "[iter {}/{}] reward={:.3f} ep_len={:.1f} "
                "loss_pi={:.4f} loss_v={:.4f} ent={:.4f} kl={:.5f} lr={:.6f} fps={} "
                "collect={:.3f}s learn={:.3f}s bad(a/o/r)={}/{}/{} forced_reset={} "
                "clip_frac={:.3f} ratio(mean/max)={:.3f}/{:.3f} "
                "std={:.4f} adv_std={:.4f} v_exp={:.3f} "
                "upd={} skip(loss/grad/met)={}/{}/{} rollback={} kl_stop={} "
                "prof(act/step/core/post/reset/buf/fin/ep)={:.3f}/{:.3f}/{:.3f}/{:.3f}/{:.3f}/{:.3f}/{:.3f}/{:.3f} "
                "reset_sub(idx/call/scatter/info)={:.3f}/{:.3f}/{:.3f}/{:.3f}".format(
                    it + 1,
                    max_iterations,
                    mean_reward,
                    mean_ep_len,
                    metrics["surrogate"],
                    metrics["value"],
                    metrics["entropy"],
                    metrics["approx_kl"],
                    current_lr,
                    fps,
                    collect_time,
                    learn_time,
                    bad_action_count,
                    bad_obs_count,
                    bad_reward_count,
                    forced_reset_count,
                    clip_fraction,
                    ratio_mean,
                    ratio_max,
                    std_mean,
                    adv_std,
                    value_explained_variance,
                    int(updates_applied),
                    int(skipped_nonfinite_loss),
                    int(skipped_nonfinite_grads),
                    int(skipped_nonfinite_metrics),
                    int(rolled_back_updates),
                    int(early_stopped_kl),
                    model_act_time,
                    env_step_total_time,
                    env_step_core_time,
                    env_step_postprocess_time,
                    env_step_reset_time,
                    buffer_add_time,
                    finite_check_time,
                    episode_stats_time,
                    env_reset_index_time,
                    env_reset_call_time,
                    env_reset_scatter_time,
                    env_reset_info_merge_time,
                )
            )

    mx.eval(model.parameters())
    env.close()
    log("[MLX PPO] training completed.")
    if tb_writer is not None:
        tb_writer.close()
    log_fp.close()


if __name__ == "__main__":
    main()
