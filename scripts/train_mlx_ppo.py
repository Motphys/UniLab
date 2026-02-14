#!/usr/bin/env python3

"""Train PPO with MLX backend."""

from __future__ import annotations

import argparse
import datetime
import importlib
import json
import pickle
import pkgutil
import sys
import time
from pathlib import Path

import numpy as np
import mlx.core as mx
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
from unilab.envs.utils import render_many
from unilab.algos.mlx_rl import EmpiricalDiscountedVariationNormalization, RolloutBuffer
from unilab.algos.mlx_ppo import MLPActorCritic, PPOConfig, PPOTrainer


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
        "optimizer_state": tree_map(lambda x: np.asarray(x), trainer.optimizer.state),
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
    init_log_std = float(np.log(max(init_noise_std, 1e-6)))
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
    parser.add_argument("--env_num", type=int, default=4096, help="Number of parallel envs")
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

    np.random.seed(args.seed)

    cfg = locomotion_params.rsl_rl_config(args.task)
    algo_cfg = cfg.algorithm

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
        _, obs, _ = env.reset(np.arange(env.num_envs))
        obs = obs.astype(np.float32)

        state_list = []
        print("[MLX PPO] Collecting physics states for play...")
        for _ in range(args.play_steps):
            obs_mx = mx.array(obs, dtype=mx.float32)
            actions_mx = model.policy(obs_mx)
            actions = np.asarray(actions_mx, dtype=np.float32)
            actions = np.where(np.isfinite(actions), actions, 0.0).astype(np.float32)
            state = env.step(actions)
            raw_obs = state.obs
            bad_mask = ~np.isfinite(raw_obs).all(axis=1)
            if np.any(bad_mask):
                bad_indices = np.where(bad_mask)[0]
                _, reset_obs, _ = env.reset(bad_indices)
                raw_obs = raw_obs.copy()
                raw_obs[bad_indices] = reset_obs
            obs = np.nan_to_num(raw_obs, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
            state_list.append(env.state.physics_state.copy())

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
        "forward_velocity_bonus_coef": float(getattr(algo_cfg, "forward_velocity_bonus_coef", 0.0)),
        "seed": args.seed,
        "timestamp": timestamp,
    }
    (log_dir / "run_config.json").write_text(json.dumps(run_meta, indent=2), encoding="utf-8")

    tb_writer = None
    try:
        tb_writer = TensorboardScalarWriter(log_dir)
    except Exception as e:
        log(f"[Warning] TensorBoard disabled: {e}")

    env = registry.make(args.task, num_envs=args.env_num, sim_backend="mujoco")
    if env.state is None:
        env.init_state()
    _, obs, _ = env.reset(np.arange(env.num_envs))
    obs = obs.astype(np.float32)

    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    action_low = env.action_space.low.astype(np.float32)
    action_high = env.action_space.high.astype(np.float32)

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
        target_kl_stop=(
            float(getattr(algo_cfg, "target_kl_stop"))
            if getattr(algo_cfg, "target_kl_stop", None) is not None
            else None
        ),
    )
    trainer = PPOTrainer(model, ppo_cfg)
    use_reward_norm = bool(getattr(algo_cfg, "reward_normalization", False))
    forward_velocity_bonus_coef = float(getattr(algo_cfg, "forward_velocity_bonus_coef", 0.0))
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
    log(f"[MLX PPO] log_dir={log_dir}")
    if tb_writer is not None:
        log("[MLX PPO] tensorboard=enabled")

    episode_returns = np.zeros((args.env_num,), dtype=np.float32)
    episode_lengths = np.zeros((args.env_num,), dtype=np.int32)
    reward_history = []
    length_history = []
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
        clipped_action_count = 0
        bad_obs_count = 0
        bad_reward_count = 0
        forced_reset_count = 0
        reward_component_sums: dict[str, float] = {}
        reward_component_counts: dict[str, int] = {}
        forward_velocity_bonus_total = 0.0
        for _ in range(num_steps):
            model.update_normalization(mx.array(obs, dtype=mx.float32))
            obs_mx = mx.array(obs, dtype=mx.float32)
            actions_mx, log_probs_mx, values_mx, action_mean_mx, action_std_mx = model.act(obs_mx)
            actions = np.asarray(actions_mx, dtype=np.float32)
            bad_action_count += int(np.size(actions) - np.isfinite(actions).sum())
            actions = np.where(np.isfinite(actions), actions, 0.0).astype(np.float32)
            # Keep behavior actions consistent with PPO storage (match rsl-rl style).
            executed_actions = actions
            clipped_action_count += int(np.sum(np.abs(np.clip(actions, action_low, action_high) - actions) > 1e-6))
            state = env.step(executed_actions)

            raw_rewards = state.reward
            raw_dones = state.done
            raw_obs = state.obs
            bad_reward_count += int(np.size(raw_rewards) - np.isfinite(raw_rewards).sum())
            bad_obs_count += int(np.size(raw_obs) - np.isfinite(raw_obs).sum())

            # If any env has non-finite transition data, force reset only those envs.
            obs_bad_mask = ~np.isfinite(raw_obs).all(axis=1)
            rew_bad_mask = ~np.isfinite(raw_rewards)
            done_bad_mask = ~np.isfinite(raw_dones)
            bad_env_mask = np.logical_or(obs_bad_mask, np.logical_or(rew_bad_mask, done_bad_mask))
            if np.any(bad_env_mask):
                bad_indices = np.where(bad_env_mask)[0]
                forced_reset_count += int(bad_indices.size)
                _, reset_obs, _ = env.reset(bad_indices)
                raw_obs = raw_obs.copy()
                raw_rewards = raw_rewards.copy()
                raw_dones = raw_dones.copy()
                raw_obs[bad_indices] = reset_obs
                raw_rewards[bad_indices] = 0.0
                raw_dones[bad_indices] = 1.0

            rewards = np.nan_to_num(raw_rewards, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
            if forward_velocity_bonus_coef > 0.0:
                # Encourage moving along the commanded x velocity to avoid standing local optimum.
                obs_for_bonus = np.nan_to_num(raw_obs, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
                cmd_x = obs_for_bonus[:, -3]
                vel_x = obs_for_bonus[:, 0]
                same_direction_speed = np.maximum(0.0, vel_x * np.sign(cmd_x))
                vel_bonus = forward_velocity_bonus_coef * same_direction_speed * float(env.cfg.ctrl_dt)
                rewards = rewards + vel_bonus.astype(np.float32)
                forward_velocity_bonus_total += float(np.mean(vel_bonus))
            if hasattr(state, "truncated"):
                timeouts = np.asarray(state.truncated, dtype=np.float32)
                rewards = rewards + ppo_cfg.gamma * np.asarray(values_mx, dtype=np.float32) * timeouts

            if hasattr(state, "info") and isinstance(state.info, dict):
                step_log = state.info.get("log", {})
                if isinstance(step_log, dict):
                    for key, value in step_log.items():
                        try:
                            scalar_value = float(value)
                        except (TypeError, ValueError):
                            continue
                        if not np.isfinite(scalar_value):
                            continue
                        reward_component_sums[key] = reward_component_sums.get(key, 0.0) + scalar_value
                        reward_component_counts[key] = reward_component_counts.get(key, 0) + 1

            rewards_mx = mx.array(rewards, dtype=mx.float32)
            if reward_normalizer is not None:
                rewards_mx = mx.squeeze(reward_normalizer(rewards_mx), axis=-1)
            dones = np.where(np.isfinite(raw_dones), raw_dones, 1.0).astype(np.float32)
            next_obs = np.nan_to_num(raw_obs, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
            model.update_normalization(mx.array(next_obs, dtype=mx.float32))

            buffer.add(
                obs=obs_mx,
                actions=actions_mx,
                log_probs=log_probs_mx,
                action_mean=action_mean_mx,
                action_std=action_std_mx,
                rewards=rewards_mx,
                dones=mx.array(dones, dtype=mx.float32),
                values=values_mx,
            )

            episode_returns += rewards
            episode_lengths += 1
            done_idx = np.where(dones > 0.5)[0]
            if done_idx.size > 0:
                reward_history.extend(episode_returns[done_idx].tolist())
                length_history.extend(episode_lengths[done_idx].tolist())
                episode_returns[done_idx] = 0.0
                episode_lengths[done_idx] = 0

            obs = next_obs

        collect_time = time.perf_counter() - collect_start
        learn_start = time.perf_counter()
        last_values = model.value(mx.array(obs, dtype=mx.float32))
        buffer.compute_returns_and_advantages(last_values)
        metrics = trainer.update(buffer)
        learn_time = time.perf_counter() - learn_start
        iter_time = time.perf_counter() - iter_start
        total_time += iter_time
        fps = int(collection_size / max(iter_time, 1e-8))
        mean_noise_std = float(np.mean(np.asarray(mx.exp(model.clipped_log_std()), dtype=np.float32)))
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
        mean_reward = float(np.mean(reward_history[-100:])) if reward_history else 0.0
        mean_ep_len = float(np.mean(length_history[-100:])) if length_history else 0.0

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
            tb_writer.add_scalar("Perf/non_finite_actions", float(bad_action_count), it)
            tb_writer.add_scalar("Perf/clipped_actions", float(clipped_action_count), it)
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
            if forward_velocity_bonus_coef > 0.0:
                tb_writer.add_scalar(
                    "forward_velocity_bonus",
                    forward_velocity_bonus_total / max(num_steps, 1),
                    it,
                )
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
                "clipped_a={} clip_frac={:.3f} ratio(mean/max)={:.3f}/{:.3f} "
                "std={:.4f} adv_std={:.4f} v_exp={:.3f} "
                "upd={} skip(loss/grad/met)={}/{}/{} rollback={} kl_stop={}".format(
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
                    clipped_action_count,
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
