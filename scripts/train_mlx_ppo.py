#!/usr/bin/env python3

"""Train PPO with MLX backend."""

from __future__ import annotations

import datetime
import importlib
import math
import os
import pickle
import pkgutil
import statistics
import sys
import time
from collections import deque
from pathlib import Path

import hydra
import mlx.core as mx
import numpy as np
from mlx.utils import tree_map
from omegaconf import DictConfig, OmegaConf

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

from unilab.algos.mlx.common import EmpiricalDiscountedVariationNormalization, RolloutBuffer
from unilab.algos.mlx.ppo import MLPActorCritic, PPOConfig, PPOTrainer
from unilab.base import registry
from unilab.utils import render_many
from unilab.utils.experiment_tracking import ExperimentTracker
from unilab.utils.obs_utils import flatten_obs_dict
from unilab.utils.onpolicy_logger import OnPolicyLogger

TASK_STEP_TUNING = {
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


def load_trainer_state(path: Path, trainer: PPOTrainer, dtype=mx.float32) -> int:
    """Load optimizer state and trainer metadata."""
    with path.open("rb") as f:
        payload = pickle.load(f)
    trainer.learning_rate = float(payload.get("learning_rate", trainer.learning_rate))
    trainer.optimizer.learning_rate = mx.array(trainer.learning_rate, dtype=dtype)
    return int(payload.get("iteration", -1))


def build_model(cfg, obs_dim: int, action_dim: int, dtype=mx.float32) -> MLPActorCritic:
    """Build actor-critic model from config (expects cfg with .policy and .empirical_normalization)."""
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
        dtype=dtype,
    )


def _get_physics_state_snapshot(env) -> np.ndarray:
    """Get physics state in a backend-compatible way for MuJoCo video rendering."""
    if hasattr(env, "_backend") and hasattr(env._backend, "get_physics_state"):
        return np.asarray(env._backend.get_physics_state(), dtype=np.float32).copy()
    if hasattr(env, "state") and hasattr(env.state, "physics_state"):
        return np.asarray(env.state.physics_state, dtype=np.float32).copy()
    raise AttributeError("Env backend does not expose physics state for video rendering")


def play_mlx_ppo(
    cfg: DictConfig, dtype, use_fp16: bool, resolved_sim_backend: str, task_log_root: Path
) -> str | None:
    """Play mode for MLX PPO."""
    from unilab.utils.reward_utils import extract_reward_config

    env_cfg_override = extract_reward_config(cfg)

    play_model_dtype = mx.float32 if use_fp16 else dtype
    play_env_num = cfg.training.play_env_num
    env = registry.make(
        cfg.training.task_name,
        num_envs=play_env_num,
        sim_backend=resolved_sim_backend,
        env_cfg_override=env_cfg_override,
    )
    obs_dim = sum(env.obs_groups_spec.values())
    action_dim = env.action_space.shape[0]
    model = build_model(cfg.algo, obs_dim, action_dim, dtype=play_model_dtype)

    load_path: Path | None = None
    load_run = cfg.training.load_run
    if load_run == "-1":
        latest_run = get_latest_run(task_log_root)
        if latest_run is not None:
            load_path = get_latest_checkpoint(latest_run)
            run_dir = latest_run
        else:
            run_dir = None
    else:
        candidate = Path(load_run)
        if not candidate.exists():
            candidate = task_log_root / load_run
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
        print(f"Could not find valid model checkpoint from load_run={load_run}")
        env.close()
        return None

    model.load_weights(str(load_path), strict=True)
    print(f"[MLX PPO] Loaded model: {load_path}")

    if env.state is None:
        env.init_state()
    play_reset_indices = np.arange(env.num_envs, dtype=np.int32)
    obs_dict_play, _ = env.reset(play_reset_indices)
    obs = mx.array(flatten_obs_dict(obs_dict_play))

    if resolved_sim_backend == "motrix":
        print("[MLX PPO] Starting interactive visualization (motrix native renderer)...")
        print("[MLX PPO] Close the render window to exit.")
        env._backend.init_renderer()

        last_render_time = time.perf_counter()
        render_dt = 1.0 / 60.0

        try:
            while True:
                obs_for_model = (
                    obs.astype(play_model_dtype)
                    if getattr(obs, "dtype", None) != play_model_dtype
                    else obs
                )
                actions_mx = model.policy(obs_for_model)
                actions = mx.where(mx.isfinite(actions_mx), actions_mx, mx.zeros_like(actions_mx))
                actions = (
                    actions.astype(dtype) if getattr(actions, "dtype", None) != dtype else actions
                )
                env_actions = np.asarray(actions)
                state = env.step(env_actions)
                raw_obs = flatten_obs_dict(state.obs)
                obs = mx.nan_to_num(raw_obs, nan=0.0, posinf=0.0, neginf=0.0)

                current_time = time.perf_counter()
                elapsed = current_time - last_render_time
                if elapsed < render_dt:
                    time.sleep(render_dt - elapsed)
                last_render_time = time.perf_counter()

                env._backend.render()
        except Exception as e:
            if "RenderClosedError" in str(type(e).__name__):
                print("[MLX PPO] Render window closed.")
            else:
                raise
        env.close()
        return None

    output_dir = run_dir if run_dir is not None else task_log_root
    output_video = output_dir / "play_video.mp4"

    state_list = []
    print("[MLX PPO] Collecting physics states for play...")
    for _ in range(cfg.training.play_steps):
        obs_for_model = (
            obs.astype(play_model_dtype) if getattr(obs, "dtype", None) != play_model_dtype else obs
        )
        actions_mx = model.policy(obs_for_model)
        actions = mx.where(mx.isfinite(actions_mx), actions_mx, mx.zeros_like(actions_mx))
        actions = actions.astype(dtype) if getattr(actions, "dtype", None) != dtype else actions
        env_actions = np.asarray(actions)
        state = env.step(env_actions)
        raw_obs = flatten_obs_dict(state.obs)
        obs = mx.nan_to_num(raw_obs, nan=0.0, posinf=0.0, neginf=0.0)
        state_list.append(_get_physics_state_snapshot(env))

    print(f"[MLX PPO] Rendering video to {output_video} ...")
    frames = render_many.render_states_get_frames(
        state_list,
        env.cfg.model_file,
        width=1280,
        height=720,
        camera_id=-1,
    )
    try:
        import mediapy as media
    except ImportError:
        print("mediapy is required for play video export. Install with `pip install mediapy`.")
        env.close()
        return None
    media.write_video(str(output_video), frames, fps=int(1.0 / env.cfg.ctrl_dt))
    print(f"[MLX PPO] Play video saved: {output_video}")
    env.close()
    return str(output_video)


@hydra.main(version_base="1.3", config_path="../conf/ppo", config_name="config_mlx")
def main(cfg: DictConfig) -> None:
    task_name = cfg.training.task_name
    resolved_sim_backend = cfg.training.sim_backend

    use_fp16 = cfg.training.fp16
    if use_fp16:
        os.environ["UNILAB_MLX_DTYPE"] = "float16"
    dtype = mx.float16 if use_fp16 else mx.float32
    model_dtype = mx.float32 if use_fp16 else dtype

    mx.random.seed(cfg.training.seed)

    algo_cfg = cfg.algo.algorithm
    profile_collection = os.getenv("UNILAB_PROFILE_COLLECTION", "0") == "1"

    num_steps = cfg.algo.num_steps_per_env
    max_iterations = cfg.algo.max_iterations
    learning_rate = float(algo_cfg.learning_rate)
    save_interval = cfg.training.save_interval

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_root = Path(cfg.training.log_root)
    if not log_root.is_absolute():
        log_root = ROOT_DIR / log_root
    task_log_root = log_root / task_name

    if cfg.training.play_only:
        play_mlx_ppo(cfg, dtype, use_fp16, resolved_sim_backend, task_log_root)
        return

    # TRAIN MODE
    log_dir = task_log_root / f"{timestamp}_{resolved_sim_backend}"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file_path = log_dir / "train.log"
    log_fp = log_file_path.open("a", encoding="utf-8")

    def log(msg: str) -> None:
        if cfg.training.logger != "no_print":
            print(msg)
        log_fp.write(msg + "\n")
        log_fp.flush()

    tracker = ExperimentTracker(
        root_dir=ROOT_DIR,
        log_dir=log_dir,
        algo_name="mlx_ppo",
        task_name=task_name,
        sim_backend=resolved_sim_backend,
        training_cfg=cfg.training,
        full_cfg=cfg,
        device="mps",
    )
    tracker.start()

    from unilab.utils.reward_utils import extract_reward_config

    env_cfg_override = extract_reward_config(cfg)

    preset = TASK_STEP_TUNING.get(task_name, {"threads": "32", "chunk": "16"})
    os.environ["UNILAB_MUJOCO_STEP_THREADS"] = preset["threads"]
    os.environ["UNILAB_MUJOCO_STEP_CHUNK"] = preset["chunk"]
    env = registry.make(
        task_name,
        num_envs=cfg.algo.num_envs,
        sim_backend=resolved_sim_backend,
        env_cfg_override=env_cfg_override,
    )
    if env.state is None:
        env.init_state()
    reset_indices = np.arange(env.num_envs, dtype=np.int32)
    obs_dict, _ = env.reset(reset_indices)
    obs = mx.array(flatten_obs_dict(obs_dict))

    obs_dim = sum(env.obs_groups_spec.values())
    action_dim = env.action_space.shape[0]

    model = build_model(cfg.algo, obs_dim, action_dim, dtype=model_dtype)
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
        normalize_advantage_per_mini_batch=bool(
            getattr(algo_cfg, "normalize_advantage_per_mini_batch", False)
        ),
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
        EmpiricalDiscountedVariationNormalization(gamma=ppo_cfg.gamma, dtype=model_dtype)
        if use_reward_norm
        else None
    )

    load_run = cfg.training.load_run
    if load_run != "-1":
        resume_candidate = Path(load_run)
        if not resume_candidate.exists():
            resume_candidate = task_log_root / load_run
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
                    resumed_it = load_trainer_state(trainer_state_path, trainer, dtype=model_dtype)
                    log(f"[MLX PPO] resumed_trainer_state={trainer_state_path} iter={resumed_it}")

    log(
        f"[MLX PPO] task={task_name} backend={resolved_sim_backend} "
        f"envs={cfg.algo.num_envs} steps={num_steps} iters={max_iterations}"
    )
    log(f"[MLX PPO] run={timestamp} lr={learning_rate:.6f} fp16={use_fp16}")
    log(
        "[MLX PPO] perf_mode fast_mode={} metrics_interval={} compile={}".format(
            ppo_cfg.fast_mode,
            ppo_cfg.metrics_interval,
            ppo_cfg.enable_compile,
        )
    )
    log(f"[MLX PPO] profile profile_collection={profile_collection}")
    log(
        "[MLX PPO] perf_warmup warmup_iters={} warmup_metrics_interval={}".format(
            ppo_cfg.warmup_strict_iters,
            ppo_cfg.warmup_metrics_interval,
        )
    )
    log(f"[MLX PPO] log_dir={log_dir}")

    rich_logger = OnPolicyLogger(
        algo_name="MLX_PPO",
        max_iterations=max_iterations,
        num_envs=cfg.algo.num_envs,
        num_steps=num_steps,
        env_name=task_name,
        log_dir=log_dir,
        log_backend=cfg.training.logger,
    )
    rich_logger.start()

    episode_returns = np.zeros((cfg.algo.num_envs,), dtype=(np.float16 if use_fp16 else np.float32))
    episode_lengths = np.zeros((cfg.algo.num_envs,), dtype=np.int32)
    reward_window = deque(maxlen=100)
    length_window = deque(maxlen=100)
    collection_size = num_steps * cfg.algo.num_envs
    total_time = 0.0
    best_mean_reward = float("-inf")
    last_mean_reward = 0.0
    last_ckpt_path: Path | None = None

    for it in range(max_iterations):
        iter_start = time.perf_counter()
        buffer = RolloutBuffer(
            num_steps=num_steps,
            num_envs=cfg.algo.num_envs,
            obs_dim=obs_dim,
            action_dim=action_dim,
            gamma=ppo_cfg.gamma,
            lam=ppo_cfg.lam,
            dtype=dtype,
        )

        collect_start = time.perf_counter()
        reward_component_sums: dict[str, float] = {}
        reward_component_counts: dict[str, int] = {}
        collect_reward_components = True
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
        buffer_add_time = 0.0
        episode_stats_time = 0.0
        for _ in range(num_steps):
            obs_for_model = obs.astype(model_dtype) if obs.dtype != model_dtype else obs
            t_act0 = time.perf_counter()
            actions_mx, log_probs_mx, values_mx, action_mean_mx, action_std_mx = model.act(
                obs_for_model
            )
            model_act_time += time.perf_counter() - t_act0
            actions = mx.where(mx.isfinite(actions_mx), actions_mx, mx.zeros_like(actions_mx))
            executed_actions = actions.astype(dtype) if actions.dtype != dtype else actions
            t_env0 = time.perf_counter()
            env_actions = np.asarray(executed_actions)
            state = env.step(env_actions)
            env_step_total_time += time.perf_counter() - t_env0
            if isinstance(state.info, dict):
                timing_info = state.info.get("timing", {})
                if isinstance(timing_info, dict):
                    env_step_core_time += float(timing_info.get("step_core_ms", 0.0)) / 1000.0
                    env_step_postprocess_time += (
                        float(timing_info.get("update_state_ms", 0.0)) / 1000.0
                    )
                    env_step_reset_time += float(timing_info.get("reset_done_ms", 0.0)) / 1000.0
                    env_reset_index_time += (
                        float(timing_info.get("reset_index_extract_ms", 0.0)) / 1000.0
                    )
                    env_reset_call_time += float(timing_info.get("reset_call_ms", 0.0)) / 1000.0
                    env_reset_scatter_time += (
                        float(timing_info.get("reset_scatter_ms", 0.0)) / 1000.0
                    )
                    env_reset_info_merge_time += (
                        float(timing_info.get("reset_info_merge_ms", 0.0)) / 1000.0
                    )

            raw_rewards = mx.array(state.reward)
            raw_dones = mx.array(state.done)
            raw_obs = mx.array(flatten_obs_dict(state.obs))
            rewards = mx.nan_to_num(raw_rewards, nan=0.0, posinf=0.0, neginf=0.0)
            dones = mx.where(mx.isfinite(raw_dones), raw_dones, mx.ones_like(raw_dones)).astype(
                dtype
            )
            next_obs = mx.nan_to_num(raw_obs, nan=0.0, posinf=0.0, neginf=0.0)
            if hasattr(state, "truncated"):
                timeouts = mx.array(state.truncated, dtype=dtype)
                rewards = rewards + ppo_cfg.gamma * values_mx.astype(rewards.dtype) * timeouts
            if rewards.dtype != dtype:
                rewards = rewards.astype(dtype)

            if (
                collect_reward_components
                and hasattr(state, "info")
                and isinstance(state.info, dict)
            ):
                step_log = state.info.get("log", {})
                if isinstance(step_log, dict):
                    for key, value in step_log.items():
                        try:
                            scalar_value = float(value)
                        except (TypeError, ValueError):
                            continue
                        if not math.isfinite(scalar_value):
                            continue
                        reward_component_sums[key] = (
                            reward_component_sums.get(key, 0.0) + scalar_value
                        )
                        reward_component_counts[key] = reward_component_counts.get(key, 0) + 1

            rewards_mx = (
                rewards.astype(model_dtype)
                if reward_normalizer is not None and rewards.dtype != model_dtype
                else rewards
            )
            if reward_normalizer is not None:
                rewards_mx = mx.squeeze(reward_normalizer(rewards_mx), axis=-1)
            if reward_normalizer is not None and dtype != model_dtype:
                rewards_mx = rewards_mx.astype(dtype)

            t_buf0 = time.perf_counter()
            buffer.add(
                obs=obs,
                actions=actions_mx.astype(dtype) if actions_mx.dtype != dtype else actions_mx,
                log_probs=log_probs_mx.astype(dtype)
                if log_probs_mx.dtype != dtype
                else log_probs_mx,
                action_mean=action_mean_mx.astype(dtype)
                if action_mean_mx.dtype != dtype
                else action_mean_mx,
                action_std=action_std_mx.astype(dtype)
                if action_std_mx.dtype != dtype
                else action_std_mx,
                rewards=rewards_mx,
                dones=dones,
                values=values_mx.astype(dtype) if values_mx.dtype != dtype else values_mx,
            )
            buffer_add_time += time.perf_counter() - t_buf0

            if track_episode_stats:
                t_ep0 = time.perf_counter()
                rewards_np = np.nan_to_num(np.asarray(rewards), nan=0.0, posinf=0.0, neginf=0.0)
                dones_np = np.asarray(dones)
                episode_returns += rewards_np
                episode_lengths += 1
                done_idx = np.flatnonzero(dones_np > 0.5)
                if done_idx.size > 0:
                    done_returns = episode_returns[done_idx]
                    done_lengths = episode_lengths[done_idx].astype(np.int32, copy=False)
                    reward_window.extend(done_returns)
                    length_window.extend(done_lengths)
                    episode_returns[done_idx] = 0.0
                    episode_lengths[done_idx] = 0
                episode_stats_time += time.perf_counter() - t_ep0

            obs = next_obs

        collect_time = time.perf_counter() - collect_start
        learn_start = time.perf_counter()
        last_values = model.value(obs.astype(model_dtype) if obs.dtype != model_dtype else obs)
        last_values_buf = last_values.astype(dtype) if last_values.dtype != dtype else last_values
        buffer.compute_returns_and_advantages(last_values_buf)
        metrics = trainer.update(buffer, iteration=it)
        learn_time = time.perf_counter() - learn_start
        iter_time = time.perf_counter() - iter_start
        total_time += iter_time
        mean_reward = float(statistics.mean(reward_window)) if reward_window else 0.0
        mean_ep_len = float(statistics.mean(length_window)) if length_window else 0.0
        last_mean_reward = mean_reward
        best_mean_reward = max(best_mean_reward, mean_reward)

        reward_components_avg = {}
        for key, summed in reward_component_sums.items():
            count = reward_component_counts.get(key, 0)
            if count > 0:
                reward_components_avg[key] = summed / count

        rich_logger.log_step(
            iteration=it,
            metrics={
                "surrogate": metrics["surrogate"],
                "value": metrics["value"],
                "entropy": metrics["entropy"],
                "approx_kl": metrics["approx_kl"],
            },
            reward=mean_reward,
            reward_components=reward_components_avg,
            collect_time=collect_time,
            train_time=learn_time,
        )
        rich_logger.update_ep_length(mean_ep_len)

        if save_interval > 0 and (it % save_interval == 0 or it == max_iterations - 1):
            ckpt_path = log_dir / f"model_{it}.safetensors"
            model.save_weights(str(ckpt_path))
            trainer_state_path = log_dir / f"trainer_{it}.pkl"
            save_trainer_state(trainer_state_path, trainer, it)
            rich_logger.log_save(str(ckpt_path))
            last_ckpt_path = ckpt_path

    mx.eval(model.parameters())
    env.close()
    log("[MLX PPO] training completed.")
    rich_logger.finish()
    train_summary = {
        "status": "completed",
        "completed_iterations": max_iterations,
        "total_env_steps": collection_size * max_iterations,
        "final_mean_reward": last_mean_reward if reward_window else None,
        "best_mean_reward": best_mean_reward if reward_window else None,
        "mean_episode_length": float(statistics.mean(length_window)) if length_window else None,
        "last_checkpoint": str(last_ckpt_path) if last_ckpt_path is not None else None,
        "training_wall_time_sec": total_time,
    }
    tracker.update_summary(train_summary)

    play_video_path = None
    if not cfg.training.no_play:
        play_video_path = play_mlx_ppo(cfg, dtype, use_fp16, resolved_sim_backend, task_log_root)
        tracker.log_video(play_video_path)

    tracker.finish()
    log_fp.close()


if __name__ == "__main__":
    main()
