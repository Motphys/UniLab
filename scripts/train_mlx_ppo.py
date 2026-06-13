#!/usr/bin/env python3

"""Train PPO with MLX backend."""

from __future__ import annotations

import datetime
import importlib
import math
import os
import pickle
import statistics
import sys
import time
from collections import deque
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, cast

import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf

if TYPE_CHECKING:
    from unilab.algos.mlx.ppo import MLPActorCritic, PPOTrainer

ROOT_DIR = Path(__file__).parent.parent
sys.path.append(str(ROOT_DIR))

from unilab.base.observations import flatten_obs_dict
from unilab.logging import OnPolicyLogger
from unilab.training import (
    BackendAdapter,
    apply_configured_training_seed,
    create_env,
    ensure_registries,
    get_log_root,
    log_playback_plan,
    parse_checkpoint_path,
    setup_logger,
    should_run_playback,
)
from unilab.training import (
    get_latest_checkpoint as get_latest_checkpoint_common,
)
from unilab.training import (
    get_latest_run as get_latest_run_common,
)
from unilab.training.experiment import ExperimentTracker
from unilab.training.sim2sim import policy_load_dim_guard, resolve_sim2sim_config

ensure_registries()

_MLX_RUNTIME: SimpleNamespace | None = None


def _require_mlx_runtime() -> SimpleNamespace:
    global _MLX_RUNTIME
    if _MLX_RUNTIME is None:
        mx = importlib.import_module("mlx.core")
        tree_map = importlib.import_module("mlx.utils").tree_map
        mlx_common = importlib.import_module("unilab.algos.mlx.common")
        mlx_ppo = importlib.import_module("unilab.algos.mlx.ppo")
        _MLX_RUNTIME = SimpleNamespace(
            mx=mx,
            tree_map=tree_map,
            EmpiricalDiscountedVariationNormalization=(
                mlx_common.EmpiricalDiscountedVariationNormalization
            ),
            RolloutBuffer=mlx_common.RolloutBuffer,
            MLPActorCritic=mlx_ppo.MLPActorCritic,
            PPOConfig=mlx_ppo.PPOConfig,
            PPOTrainer=mlx_ppo.PPOTrainer,
        )
    return _MLX_RUNTIME


mx = SimpleNamespace(
    array=np.array,
    float16=np.float16,
    float32=np.float32,
    sum=np.sum,
)


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
        summary = cast(Any, self._Summary())
        summary_value = summary.value.add()
        summary_value.tag = tag
        summary_value.simple_value = float(value)
        event = cast(Any, self._Event())
        event.wall_time = time.time()
        event.step = int(step)
        event.summary.CopyFrom(summary)
        self._writer.add_event(event)

    def flush(self) -> None:
        self._writer.flush()

    def close(self) -> None:
        self._writer.close()


def get_latest_run(log_dir: Path) -> Path | None:
    return cast(Path | None, get_latest_run_common(log_dir))


def get_latest_checkpoint(run_dir: Path) -> Path | None:
    return cast(Path | None, get_latest_checkpoint_common(run_dir, suffix=".safetensors"))


def save_trainer_state(path: Path, trainer: Any, iteration: int) -> None:
    """Save optimizer state and trainer metadata for resume."""
    tree_map = _require_mlx_runtime().tree_map
    payload = {
        "iteration": int(iteration),
        "learning_rate": float(trainer.learning_rate),
        "optimizer_state": tree_map(lambda x: x.tolist(), trainer.optimizer.state),
    }
    with path.open("wb") as f:
        pickle.dump(payload, f)


def load_trainer_state(path: Path, trainer: Any, dtype: Any = None) -> int:
    """Load optimizer state and trainer metadata."""
    mx = _require_mlx_runtime().mx
    if dtype is None:
        dtype = mx.float32
    with path.open("rb") as f:
        payload = pickle.load(f)
    trainer.learning_rate = float(payload.get("learning_rate", trainer.learning_rate))
    trainer.optimizer.learning_rate = mx.array(trainer.learning_rate, dtype=dtype)
    return int(payload.get("iteration", -1))


def build_model(cfg, obs_dim: int, action_dim: int, dtype: Any = None) -> Any:
    """Build actor-critic model from config (expects cfg with .policy and .empirical_normalization)."""
    runtime = _require_mlx_runtime()
    if dtype is None:
        dtype = runtime.mx.float32
    policy_cfg = cfg.policy
    init_noise_std = float(getattr(policy_cfg, "init_noise_std", 1.0))
    init_log_std = float(math.log(max(init_noise_std, 1e-6)))
    obs_norm = bool(getattr(cfg, "empirical_normalization", False))
    noise_std_type = str(getattr(policy_cfg, "noise_std_type", "scalar"))
    state_dependent_std = bool(getattr(policy_cfg, "state_dependent_std", False))
    return runtime.MLPActorCritic(
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


def get_time_limit_bootstrap_values(state: Any, model: Any, model_dtype: Any = None) -> Any | None:
    """Return V(final_observation) for current timeout envs when available."""
    use_mlx_runtime = type(model).__module__.startswith("unilab.algos.mlx")
    mx_mod = _require_mlx_runtime().mx if use_mlx_runtime else mx
    if model_dtype is None:
        model_dtype = mx_mod.float32
    timeout_mask = np.asarray(state.truncated, dtype=bool)
    if not np.any(timeout_mask):
        return None
    final_observation = state.final_observation
    if final_observation is None and isinstance(state.info, dict):
        final_observation = state.info.get("final_observation")
    if not isinstance(final_observation, dict):
        return None
    final_obs = mx_mod.array(flatten_obs_dict(final_observation))
    if getattr(final_obs, "dtype", None) != model_dtype:
        final_obs = final_obs.astype(model_dtype)
    return model.value(final_obs)


def _get_log_root(cfg: DictConfig) -> Path:
    return cast(Path, get_log_root(ROOT_DIR, cfg))


def play_mlx_ppo(cfg: DictConfig, dtype, use_fp16: bool, resolved_sim_backend: str) -> str | None:
    """Play mode for MLX PPO."""
    mx = _require_mlx_runtime().mx

    task_log_root = _get_log_root(cfg) / cfg.training.task_name
    load_path, run_dir = parse_checkpoint_path(cfg, root_dir=ROOT_DIR, suffix=".safetensors")
    if load_path is None or run_dir is None or not load_path.exists():
        print(f"Could not find valid model checkpoint from load_run={cfg.algo.load_run}")
        return None

    cfg = (
        resolve_sim2sim_config(
            run_dir,
            cfg,
            algo_name="ppo",
            strict=bool(getattr(cfg.training, "sim2sim_strict", True)),
        )
        or cfg
    )

    env_cfg_override = BackendAdapter(
        cfg, root_dir=ROOT_DIR, algo_name="ppo"
    ).build_task_env_cfg_override()

    play_model_dtype = mx.float32 if use_fp16 else dtype
    play_env_num = cfg.training.play_env_num
    env = cast(
        Any,
        create_env(
            cfg,
            num_envs=play_env_num,
            env_cfg_override=env_cfg_override,
        ),
    )
    obs_dim = sum(env.obs_groups_spec.values())
    action_shape = env.action_space.shape
    if action_shape is None:
        raise ValueError("env.action_space.shape must be defined")
    action_dim = int(action_shape[0])
    model = build_model(cfg.algo, obs_dim, action_dim, dtype=play_model_dtype)

    with policy_load_dim_guard(env_obs_dim=obs_dim, env_action_dim=action_dim, algo_name="ppo"):
        model.load_weights(str(load_path), strict=True)
    print(f"[MLX PPO] Loaded model: {load_path}")

    # Export actor to ONNX (convert MLX weights → PyTorch)
    if run_dir is not None:
        import onnxruntime as ort
        import torch
        import torch.nn as nn
        import torch.nn.functional as F

        activation_fn = nn.ELU if cfg.algo.policy.activation == "elu" else nn.Tanh

        class _TorchActor(nn.Module):
            def __init__(self, obs_dim: int, action_dim: int, hidden_dims: list[int], act_cls):
                super().__init__()
                layers: list[nn.Module] = []
                in_dim = obs_dim
                for h in hidden_dims:
                    layers.append(nn.Linear(in_dim, h))
                    layers.append(act_cls())
                    in_dim = h
                layers.append(nn.Linear(in_dim, action_dim))
                self.net = nn.Sequential(*layers)

            def forward(self, obs: torch.Tensor) -> torch.Tensor:
                return self.net(obs)

        hidden_dims = list(cfg.algo.policy.actor_hidden_dims)
        torch_actor = _TorchActor(obs_dim, action_dim, hidden_dims, activation_fn)

        # Copy MLX weights → PyTorch
        mlx_layers = model.actor.layers
        torch_idx = 0
        for mlx_layer in mlx_layers:
            w = np.array(mlx_layer.weight, copy=True)
            b = np.array(mlx_layer.bias, copy=True)
            torch_actor.net[torch_idx].weight.data = torch.from_numpy(w.copy())
            torch_actor.net[torch_idx].bias.data = torch.from_numpy(b.copy())
            torch_idx += 2  # skip activation

        torch_actor.eval()
        onnx_path = str(run_dir / "policy.onnx")
        dummy_input = torch.randn(1, obs_dim)
        with torch.inference_mode():
            torch.onnx.export(
                torch_actor,
                (dummy_input,),
                onnx_path,
                input_names=["obs"],
                output_names=["action"],
                opset_version=17,
            )
        print(f"Exported actor ONNX to {onnx_path}")

        # Verify
        verify_input = torch.randn(1, obs_dim)
        with torch.inference_mode():
            pt_out = torch_actor(verify_input).numpy()
        ort_out = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"]).run(
            None, {"obs": verify_input.numpy().astype(np.float32)}
        )[0]
        max_diff = np.max(np.abs(pt_out - ort_out))
        mean_diff = np.mean(np.abs(pt_out - ort_out))
        print(f"ONNX vs PyTorch — max_diff: {max_diff:.2e}, mean_diff: {mean_diff:.2e}")
        if max_diff > 1e-4:
            print("WARNING: ONNX output diverges from PyTorch!")
        else:
            print("ONNX export verified OK.")

    if env.state is None:
        env.init_state()
    play_reset_indices = np.arange(env.num_envs, dtype=np.int32)
    obs_dict_play, _ = env.reset(play_reset_indices)
    obs = mx.array(flatten_obs_dict(obs_dict_play))

    def _play_step(current_obs):
        obs_for_model = (
            current_obs.astype(play_model_dtype)
            if getattr(current_obs, "dtype", None) != play_model_dtype
            else current_obs
        )
        actions_mx = model.policy(obs_for_model)
        actions = mx.where(mx.isfinite(actions_mx), actions_mx, mx.zeros_like(actions_mx))
        actions = actions.astype(dtype) if getattr(actions, "dtype", None) != dtype else actions
        state = env.step(np.asarray(actions))
        raw_obs = mx.array(flatten_obs_dict(state.obs))
        return mx.nan_to_num(raw_obs, nan=0.0, posinf=0.0, neginf=0.0)

    output_dir = run_dir if run_dir is not None else task_log_root
    try:
        play_video_path = env.run_playback_mode(
            play_render_mode=getattr(cfg.training, "play_render_mode", "auto"),
            play_steps=getattr(cfg.training, "play_steps", None),
            output_video=output_dir / "play_video.mp4",
            initialize=lambda: obs,
            step=_play_step,
            camera_kwargs={
                "cam_distance": getattr(cfg.training, "cam_distance", 2.0),
                "cam_elevation": getattr(cfg.training, "cam_elevation", -20.0),
                "cam_azimuth": getattr(cfg.training, "cam_azimuth", 90.0),
                "cam_lookat": getattr(cfg.training, "cam_lookat", None),
                "cam_tracking": getattr(cfg.training, "cam_tracking", False),
                "cam_tracking_env_idx": getattr(cfg.training, "cam_tracking_env_idx", 0),
                "cam_tracking_extra_envs": getattr(cfg.training, "cam_tracking_extra_envs", 2),
            },
            on_plan=lambda plan: log_playback_plan(plan, prefix="[MLX PPO] "),
        )
    except ImportError:
        print("mediapy is required for play video export. Install with `pip install mediapy`.")
        env.close()
        return None
    if play_video_path is not None:
        print(f"[MLX PPO] Play video saved: {play_video_path}")
    else:
        print("[MLX PPO] Playback done.")
    env.close()
    return play_video_path


@hydra.main(version_base="1.3", config_path="../conf/ppo", config_name="config_mlx")
def main(cfg: DictConfig) -> None:
    runtime = _require_mlx_runtime()
    mx = runtime.mx
    PPOConfig = runtime.PPOConfig
    PPOTrainer = runtime.PPOTrainer
    RolloutBuffer = runtime.RolloutBuffer
    EmpiricalDiscountedVariationNormalization = runtime.EmpiricalDiscountedVariationNormalization
    task_name = cfg.training.task_name
    resolved_sim_backend = cfg.training.sim_backend

    use_fp16 = cfg.training.fp16
    if use_fp16:
        os.environ["UNILAB_MLX_DTYPE"] = "float16"
    dtype = mx.float16 if use_fp16 else mx.float32
    model_dtype = mx.float32 if use_fp16 else dtype

    seed_info = apply_configured_training_seed(
        cfg,
        torch_runtime=False,
        cuda=False,
        mlx_runtime=True,
    )

    algo_cfg = cfg.algo.algorithm
    profile_collection = os.getenv("UNILAB_PROFILE_COLLECTION", "0") == "1"

    num_steps = cfg.algo.num_steps_per_env
    max_iterations = cfg.algo.max_iterations
    learning_rate = float(algo_cfg.learning_rate)
    save_interval = cfg.algo.save_interval

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_root = _get_log_root(cfg)
    task_log_root = log_root / task_name

    if cfg.training.play_only:
        play_mlx_ppo(cfg, dtype, use_fp16, resolved_sim_backend)
        return

    # TRAIN MODE
    log_dir = task_log_root / f"{timestamp}_{resolved_sim_backend}"
    script_logger = setup_logger(log_dir, "mlx_ppo", echo=cfg.training.logger != "no_print")

    def log(msg: str) -> None:
        script_logger.info(msg)

    tracker = ExperimentTracker(
        root_dir=ROOT_DIR,
        log_dir=log_dir,
        algo_name="mlx_ppo",
        task_name=task_name,
        sim_backend=resolved_sim_backend,
        training_cfg=cfg.training,
        full_cfg=cfg,
        device="mps",
        seed_info=seed_info,
    )
    tracker.start()

    env_cfg_override = BackendAdapter(
        cfg, root_dir=ROOT_DIR, algo_name="ppo"
    ).build_task_env_cfg_override()

    env = cast(
        Any,
        create_env(
            cfg,
            num_envs=cfg.algo.num_envs,
            env_cfg_override=env_cfg_override,
        ),
    )
    if env.state is None:
        env.init_state()
    reset_indices = np.arange(env.num_envs, dtype=np.int32)
    obs_dict, _ = env.reset(reset_indices)
    obs = mx.array(flatten_obs_dict(obs_dict))

    obs_dim = sum(env.obs_groups_spec.values())
    action_shape = env.action_space.shape
    if action_shape is None:
        raise ValueError("env.action_space.shape must be defined")
    action_dim = int(action_shape[0])

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
        metrics_interval=int(getattr(algo_cfg, "metrics_interval", 8)),
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

    load_run = cfg.algo.load_run
    if load_run != "-1":
        ckpt, _ = parse_checkpoint_path(cfg, root_dir=ROOT_DIR, suffix=".safetensors")
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
        "[MLX PPO] perf_mode metrics_interval={} compile={}".format(
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
    reward_window: deque[float] = deque(maxlen=100)
    length_window: deque[int] = deque(maxlen=100)
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
            raw_dones = mx.array(state.terminated | state.truncated)
            raw_obs = mx.array(flatten_obs_dict(state.obs))
            rewards = mx.nan_to_num(raw_rewards, nan=0.0, posinf=0.0, neginf=0.0)
            dones = mx.where(mx.isfinite(raw_dones), raw_dones, mx.ones_like(raw_dones)).astype(
                dtype
            )
            next_obs = mx.nan_to_num(raw_obs, nan=0.0, posinf=0.0, neginf=0.0)
            timeouts = mx.array(state.truncated, dtype=dtype)
            timeout_bootstrap_values = get_time_limit_bootstrap_values(state, model, model_dtype)
            if timeout_bootstrap_values is None:
                timeout_bootstrap_values = values_mx
            rewards = (
                rewards + ppo_cfg.gamma * timeout_bootstrap_values.astype(rewards.dtype) * timeouts
            )
            if rewards.dtype != dtype:
                rewards = rewards.astype(dtype)

            if collect_reward_components and isinstance(state.info, dict):
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
    if should_run_playback(
        play_only=False,
        no_play=cfg.training.no_play,
        play_render_mode=getattr(cfg.training, "play_render_mode", "auto"),
    ):
        play_video_path = play_mlx_ppo(cfg, dtype, use_fp16, resolved_sim_backend)
        tracker.log_video(play_video_path)

    tracker.finish()


if __name__ == "__main__":
    main()
