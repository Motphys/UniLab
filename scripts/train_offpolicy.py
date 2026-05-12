"""Unified off-policy training entry for SAC and TD3."""

from __future__ import annotations

import datetime
import os
import sys
from pathlib import Path
from typing import Any, cast

import hydra
from omegaconf import DictConfig

ROOT_DIR = Path(__file__).parent.parent
sys.path.append(str(ROOT_DIR))

from unilab.training import (
    BackendAdapter,
    apply_configured_training_seed,
    assert_offpolicy_task_choice_matches_algo,
    create_env,
    ensure_registries,
    get_log_root,
)
from unilab.training import (
    resolve_checkpoint_path as resolve_checkpoint_path_common,
)
from unilab.training.experiment import ExperimentTracker
from unilab.visualization import render_play_mode


def default_device(torch_module, preferred: str | None = None) -> str:
    """Resolve runtime device with optional user override."""
    if preferred:
        return preferred
    if torch_module.cuda.is_available():
        return "cuda"
    if torch_module.backends.mps.is_available():
        return "mps"
    return "cpu"


def resolve_checkpoint_path(
    root_dir: Path, algo_log_name: str, task: str, load_run: str
) -> tuple[str | None, str | None]:
    checkpoint_path, checkpoint_dir = resolve_checkpoint_path_common(
        Path(root_dir) / "logs" / algo_log_name / task,
        load_run,
        suffix=".pt",
    )
    return (
        str(checkpoint_path) if checkpoint_path is not None else None,
        str(checkpoint_dir) if checkpoint_dir is not None else None,
    )


def extract_reset_obs(reset_result):
    """Extract obs_dict from env.reset(...) using the current (obs_dict, info_dict) contract."""
    if isinstance(reset_result, tuple):
        if len(reset_result) == 2:
            obs_out, _ = reset_result
            return obs_out
    raise ValueError(f"Unexpected env.reset return format: {type(reset_result)!r}")


def resolve_play_obs_dim(obs_groups_spec: dict[str, int]) -> int:
    from unilab.base.observations import get_obs_dims

    obs_dim, _ = get_obs_dims(obs_groups_spec)
    return int(obs_dim)


def extract_play_obs(obs_dict):
    from unilab.base.observations import split_obs_dict

    obs_out, _ = split_obs_dict(obs_dict)
    return obs_out


def build_offpolicy_env_cfg_override(algo_name: str, cfg: DictConfig) -> dict[str, Any] | None:
    assert_offpolicy_task_choice_matches_algo(cfg, algo_name=algo_name)
    return cast(
        dict[str, Any] | None,
        BackendAdapter(cfg, root_dir=ROOT_DIR, algo_name=algo_name).build_task_env_cfg_override(),
    )


def build_runner(algo_name: str, cfg: DictConfig):
    """Build algorithm runner from unified Hydra config."""
    env_cfg_override = build_offpolicy_env_cfg_override(algo_name, cfg)

    replay_pipeline = getattr(cfg.training, "replay_pipeline", "auto")
    if replay_pipeline == "auto":
        replay_pipeline = "cpu_pinned_double_buffer" if algo_name == "sac" else "gpu_cache"
    if replay_pipeline not in {"auto", "gpu_cache", "cpu_pinned_double_buffer"}:
        raise ValueError(
            f"Unsupported training.replay_pipeline={replay_pipeline!r}; "
            "expected 'auto', 'gpu_cache' or 'cpu_pinned_double_buffer'"
        )
    replay_prefetch_mode = getattr(cfg.training, "replay_prefetch_mode", "one_tick")
    if replay_prefetch_mode != "one_tick":
        raise ValueError(
            f"Unsupported training.replay_prefetch_mode={replay_prefetch_mode!r}; "
            "expected 'one_tick'"
        )
    verbose_metrics = bool(getattr(cfg.training, "verbose_metrics", False))
    if replay_pipeline == "cpu_pinned_double_buffer" and algo_name not in {"sac", "flashsac"}:
        raise ValueError(f"cpu_pinned_double_buffer is not enabled for algo={algo_name}")

    if algo_name == "flashsac" and cfg.training.num_gpus > 1:
        raise ValueError("FlashSAC does not support training.num_gpus > 1")

    if algo_name == "sac":
        if replay_pipeline == "cpu_pinned_double_buffer" and cfg.training.num_gpus > 1:
            raise ValueError("cpu_pinned_double_buffer is initially single-GPU only")
        if replay_pipeline == "cpu_pinned_double_buffer" and cfg.training.no_sync_collection:
            raise ValueError(
                "cpu_pinned_double_buffer requires synchronized collection in this MVP"
            )

        from unilab.algos.torch.common.device import get_env_dims
        from unilab.algos.torch.fast_sac.learner import FastSACLearner
        from unilab.algos.torch.fast_sac.runner import FastSACRunner
        from unilab.utils.device import get_default_device

        # Multi-GPU path
        if cfg.training.num_gpus > 1:
            from unilab.algos.torch.offpolicy.multi_gpu_runner import MultiGPUOffPolicyRunner

            ensure_registries()
            device = cfg.training.device or get_default_device()
            env = create_env(
                cfg,
                num_envs=1,
                env_cfg_override=env_cfg_override,
            )
            assert env.action_space.shape
            from unilab.base.observations import get_obs_dims

            obs_dim, critic_dim = get_obs_dims(env.obs_groups_spec)
            action_dim = env.action_space.shape[0]
            env.close()

            learner_kwargs = {
                "obs_dim": obs_dim,
                "action_dim": action_dim,
                "gamma": cfg.algo.gamma,
                "tau": cfg.algo.tau,
                "actor_lr": cfg.algo.actor_lr,
                "critic_lr": cfg.algo.critic_lr,
                "alpha_lr": cfg.algo.algo_params.alpha_lr,
                "alpha_init": cfg.algo.algo_params.alpha_init,
                "target_entropy_ratio": cfg.algo.algo_params.target_entropy_ratio,
                "actor_hidden_dim": cfg.algo.actor_hidden_dim,
                "critic_hidden_dim": cfg.algo.critic_hidden_dim,
                "num_atoms": cfg.algo.num_atoms,
                "use_layer_norm": cfg.algo.use_layer_norm,
                "max_grad_norm": cfg.algo.algo_params.max_grad_norm,
                "use_amp": cfg.training.use_amp,
                "critic_obs_dim": critic_dim,
                "use_symmetry": cfg.algo.use_symmetry,
            }
            MultiGPUOffPolicyRunner.validate_capabilities(
                algo_type="sac",
                learner_kwargs=learner_kwargs,
                num_gpus=cfg.training.num_gpus,
            )
            main_learner = FastSACLearner(device=device, **learner_kwargs)

            return MultiGPUOffPolicyRunner(
                learner=main_learner,
                env_name=cfg.training.task_name,
                algo_type="sac",
                learner_kwargs=learner_kwargs,
                num_gpus=cfg.training.num_gpus,
                num_envs=cfg.algo.num_envs,
                replay_buffer_n=cfg.algo.replay_buffer_n,
                batch_size=cfg.algo.batch_size,
                learning_starts=cfg.algo.learning_starts,
                updates_per_step=cfg.algo.updates_per_step,
                policy_frequency=cfg.algo.policy_frequency,
                sync_collection=not cfg.training.no_sync_collection,
                env_steps_per_sync=cfg.training.env_steps_per_sync,
                device=device,
                actor_hidden_dim=cfg.algo.actor_hidden_dim,
                use_layer_norm=cfg.algo.use_layer_norm,
                obs_normalization=False,
                sim_backend=cfg.training.sim_backend,
                env_cfg_override=env_cfg_override,
                seed=cfg.algo.seed,
                trace_enabled=cfg.training.trace_enabled,
                trace_output_dir=cfg.training.trace_output_dir,
                trace_thread_time=cfg.training.trace_thread_time,
                trace_cuda_events=cfg.training.trace_cuda_events,
            )

        if replay_pipeline == "cpu_pinned_double_buffer":
            from unilab.algos.torch.offpolicy.double_buffer_runner import (
                DoubleBufferOffPolicyRunner,
            )
            from unilab.base.registry import ensure_registries as _ensure

            _ensure()
            _device = cfg.training.device or get_default_device()
            if not str(_device).startswith("cuda"):
                raise ValueError("cpu_pinned_double_buffer requires a CUDA device")
            _env = create_env(cfg, num_envs=1, env_cfg_override=env_cfg_override)
            assert _env.action_space.shape
            from unilab.base.observations import get_obs_dims as _get_obs_dims

            _obs_dim, _critic_dim = _get_obs_dims(_env.obs_groups_spec)
            _action_dim = _env.action_space.shape[0]
            _symmetry_aug = None
            if cfg.algo.use_symmetry:
                _symmetry_aug = _env.build_symmetry_augmentation(device=_device)
                if _symmetry_aug is None:
                    _env.close()
                    raise ValueError(
                        f"{cfg.training.task_name} does not provide symmetry augmentation"
                    )
            _env.close()

            _batch_size = cfg.algo.batch_size
            if _symmetry_aug is not None:
                if _batch_size % _symmetry_aug.batch_multiplier != 0:
                    raise ValueError(
                        "Symmetry augmentation requires batch_size divisible by "
                        f"{_symmetry_aug.batch_multiplier}, got {_batch_size}"
                    )
                _batch_size = _batch_size // _symmetry_aug.batch_multiplier

            _learner = FastSACLearner(
                obs_dim=_obs_dim,
                action_dim=_action_dim,
                device=_device,
                gamma=cfg.algo.gamma,
                tau=cfg.algo.tau,
                actor_lr=cfg.algo.actor_lr,
                critic_lr=cfg.algo.critic_lr,
                alpha_lr=cfg.algo.algo_params.alpha_lr,
                alpha_init=cfg.algo.algo_params.alpha_init,
                target_entropy_ratio=cfg.algo.algo_params.target_entropy_ratio,
                actor_hidden_dim=cfg.algo.actor_hidden_dim,
                critic_hidden_dim=cfg.algo.critic_hidden_dim,
                num_atoms=cfg.algo.num_atoms,
                use_layer_norm=cfg.algo.use_layer_norm,
                max_grad_norm=cfg.algo.algo_params.max_grad_norm,
                use_amp=cfg.training.use_amp,
                use_symmetry=cfg.algo.use_symmetry,
                symmetry_augmentation=_symmetry_aug,
                critic_obs_dim=_critic_dim,
            )

            return DoubleBufferOffPolicyRunner(
                learner=_learner,
                env_name=cfg.training.task_name,
                algo_type="sac",
                num_envs=cfg.algo.num_envs,
                replay_buffer_n=cfg.algo.replay_buffer_n,
                batch_size=_batch_size,
                learning_starts=cfg.algo.learning_starts,
                updates_per_step=cfg.algo.updates_per_step,
                policy_frequency=cfg.algo.policy_frequency,
                sync_collection=not cfg.training.no_sync_collection,
                env_steps_per_sync=cfg.training.env_steps_per_sync,
                device=_device,
                actor_hidden_dim=cfg.algo.actor_hidden_dim,
                use_layer_norm=cfg.algo.use_layer_norm,
                obs_normalization=cfg.algo.obs_normalization,
                sim_backend=cfg.training.sim_backend,
                env_cfg_override=env_cfg_override,
                trace_enabled=cfg.training.trace_enabled,
                trace_output_dir=cfg.training.trace_output_dir,
                trace_thread_time=cfg.training.trace_thread_time,
                trace_cuda_events=cfg.training.trace_cuda_events,
                replay_prefetch_mode=replay_prefetch_mode,
                verbose_metrics=verbose_metrics,
                seed=cfg.algo.seed,
            )

        return FastSACRunner(
            env_name=cfg.training.task_name,
            env_cfg_override=env_cfg_override,
            device=cfg.training.device,
            num_envs=cfg.algo.num_envs,
            replay_buffer_n=cfg.algo.replay_buffer_n,
            batch_size=cfg.algo.batch_size,
            learning_starts=cfg.algo.learning_starts,
            updates_per_step=cfg.algo.updates_per_step,
            policy_frequency=cfg.algo.policy_frequency,
            gamma=cfg.algo.gamma,
            tau=cfg.algo.tau,
            actor_lr=cfg.algo.actor_lr,
            critic_lr=cfg.algo.critic_lr,
            alpha_lr=cfg.algo.algo_params.alpha_lr,
            alpha_init=cfg.algo.algo_params.alpha_init,
            target_entropy_ratio=cfg.algo.algo_params.target_entropy_ratio,
            obs_normalization=cfg.algo.obs_normalization,
            actor_hidden_dim=cfg.algo.actor_hidden_dim,
            critic_hidden_dim=cfg.algo.critic_hidden_dim,
            num_atoms=cfg.algo.num_atoms,
            use_layer_norm=cfg.algo.use_layer_norm,
            max_grad_norm=cfg.algo.algo_params.max_grad_norm,
            use_amp=cfg.training.use_amp,
            sync_collection=not cfg.training.no_sync_collection,
            env_steps_per_sync=cfg.training.env_steps_per_sync,
            sim_backend=cfg.training.sim_backend,
            use_symmetry=cfg.algo.use_symmetry,
            seed=cfg.algo.seed,
            trace_enabled=cfg.training.trace_enabled,
            trace_output_dir=cfg.training.trace_output_dir,
            trace_thread_time=cfg.training.trace_thread_time,
            trace_cuda_events=cfg.training.trace_cuda_events,
        )

    if algo_name == "td3":
        from unilab.algos.torch.fast_td3.runner import FastTD3Runner

        return FastTD3Runner(
            env_name=cfg.training.task_name,
            env_cfg_override=env_cfg_override,
            device=cfg.training.device,
            num_envs=cfg.algo.num_envs,
            replay_buffer_n=cfg.algo.replay_buffer_n,
            batch_size=cfg.algo.batch_size,
            learning_starts=cfg.algo.learning_starts,
            num_updates=cfg.algo.updates_per_step,
            policy_frequency=cfg.algo.policy_frequency,
            sync_collection=not cfg.training.no_sync_collection,
            env_steps_per_sync=cfg.training.env_steps_per_sync,
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
            policy_noise=cfg.algo.algo_params.policy_noise,
            noise_clip=cfg.algo.algo_params.noise_clip,
            weight_decay=cfg.algo.algo_params.weight_decay,
            use_cdq=cfg.algo.algo_params.use_cdq,
            obs_normalization=cfg.algo.obs_normalization,
            sim_backend=cfg.training.sim_backend,
            seed=cfg.algo.seed,
            trace_enabled=cfg.training.trace_enabled,
            trace_output_dir=cfg.training.trace_output_dir,
            trace_thread_time=cfg.training.trace_thread_time,
            trace_cuda_events=cfg.training.trace_cuda_events,
        )

    if algo_name == "flashsac":
        if replay_pipeline == "cpu_pinned_double_buffer":
            from unilab.algos.torch.flash_sac.double_buffer import (
                build_flashsac_double_buffer_runner,
            )

            return build_flashsac_double_buffer_runner(
                cfg,
                env_cfg_override=env_cfg_override,
                replay_prefetch_mode=replay_prefetch_mode,
                verbose_metrics=verbose_metrics,
            )

        from unilab.algos.torch.flash_sac.runner import FlashSACRunner

        return FlashSACRunner(
            env_name=cfg.training.task_name,
            env_cfg_override=env_cfg_override,
            device=cfg.training.device,
            num_envs=cfg.algo.num_envs,
            replay_buffer_n=cfg.algo.replay_buffer_n,
            batch_size=cfg.algo.batch_size,
            learning_starts=cfg.algo.learning_starts,
            updates_per_step=cfg.algo.updates_per_step,
            policy_frequency=cfg.algo.policy_frequency,
            gamma=cfg.algo.gamma,
            tau=cfg.algo.tau,
            actor_lr=cfg.algo.actor_lr,
            critic_lr=cfg.algo.critic_lr,
            obs_normalization=cfg.algo.obs_normalization,
            actor_hidden_dim=cfg.algo.actor_hidden_dim,
            critic_hidden_dim=cfg.algo.critic_hidden_dim,
            num_atoms=cfg.algo.num_atoms,
            use_amp=cfg.training.use_amp,
            sync_collection=not cfg.training.no_sync_collection,
            env_steps_per_sync=cfg.training.env_steps_per_sync,
            sim_backend=cfg.training.sim_backend,
            actor_num_blocks=cfg.algo.algo_params.actor_num_blocks,
            critic_num_blocks=cfg.algo.algo_params.critic_num_blocks,
            actor_bc_alpha=cfg.algo.algo_params.actor_bc_alpha,
            actor_noise_zeta_mu=cfg.algo.algo_params.actor_noise_zeta_mu,
            actor_noise_zeta_max=cfg.algo.algo_params.actor_noise_zeta_max,
            critic_min_v=cfg.algo.algo_params.critic_min_v,
            critic_max_v=cfg.algo.algo_params.critic_max_v,
            target_sigma=cfg.algo.algo_params.temp_target_sigma,
            target_entropy=cfg.algo.algo_params.temp_target_entropy,
            temp_initial_value=cfg.algo.algo_params.temp_initial_value,
            learning_rate_init=cfg.algo.algo_params.learning_rate_init,
            learning_rate_peak=cfg.algo.algo_params.learning_rate_peak,
            learning_rate_end=cfg.algo.algo_params.learning_rate_end,
            learning_rate_warmup_steps=cfg.algo.algo_params.learning_rate_warmup_steps,
            learning_rate_decay_steps=cfg.algo.algo_params.learning_rate_decay_steps,
            normalize_reward=cfg.algo.algo_params.normalize_reward,
            normalized_g_max=cfg.algo.algo_params.normalized_g_max,
            n_step=cfg.algo.algo_params.n_step,
            use_compile=cfg.algo.algo_params.use_compile,
            seed=cfg.algo.seed,
            trace_enabled=cfg.training.trace_enabled,
            trace_output_dir=cfg.training.trace_output_dir,
            trace_thread_time=cfg.training.trace_thread_time,
            trace_cuda_events=cfg.training.trace_cuda_events,
        )

    raise ValueError(f"Unsupported algo: {algo_name}")


def play_offpolicy(algo_name: str, cfg: DictConfig) -> str | None:
    """Play pipeline for off-policy algorithms."""
    import numpy as np
    import torch

    from unilab.algos.torch.common.actor_factory import build_actor

    env_cfg_override = build_offpolicy_env_cfg_override(algo_name, cfg)

    device = default_device(torch, cfg.training.device)
    print(f"Using device for play: {device}")

    env = cast(
        Any,
        create_env(
            cfg,
            num_envs=cfg.training.play_env_num,
            env_cfg_override=env_cfg_override,
        ),
    )
    obs_dim = resolve_play_obs_dim(env.obs_groups_spec)
    action_shape = env.action_space.shape
    if action_shape is None:
        raise ValueError("env.action_space.shape must be defined")
    action_dim = int(action_shape[0])

    normalizer = None
    if algo_name == "sac":
        actor = build_actor(
            "sac",
            obs_dim,
            action_dim,
            cfg.algo.actor_hidden_dim,
            cfg.algo.use_layer_norm,
            device,
        )
    elif algo_name == "td3":
        import torch

        from unilab.algos.torch.fast_td3.learner import EmpiricalNormalization, TD3Actor

        actor = TD3Actor(
            obs_dim,
            action_dim,
            cfg.training.play_env_num,
            cfg.algo.algo_params.init_scale,
            cfg.algo.actor_hidden_dim,
            cfg.algo.algo_params.log_std_min,
            cfg.algo.algo_params.log_std_max,
            torch.device(device),
        )
        if cfg.algo.obs_normalization:
            normalizer = EmpiricalNormalization(shape=obs_dim, device=device)
    elif algo_name == "flashsac":
        actor = build_actor(
            "flashsac",
            obs_dim,
            action_dim,
            cfg.algo.actor_hidden_dim,
            cfg.algo.use_layer_norm,
            device,
            actor_num_blocks=cfg.algo.algo_params.actor_num_blocks,
            actor_noise_zeta_mu=cfg.algo.algo_params.actor_noise_zeta_mu,
            actor_noise_zeta_max=cfg.algo.algo_params.actor_noise_zeta_max,
        )
        if cfg.algo.obs_normalization:
            from unilab.algos.torch.common.normalization import EmpiricalNormalization

            normalizer = EmpiricalNormalization(shape=obs_dim, device=device)
    else:
        raise ValueError(f"Unsupported algo: {algo_name}")

    actor.eval()

    load_path, load_path_dir = resolve_checkpoint_path(
        ROOT_DIR,
        cfg.algo.algo_log_name,
        cfg.training.task_name,
        cfg.algo.load_run,
    )
    if not load_path or not os.path.exists(load_path):
        print(f"Could not find checkpoint. load_path={load_path}")
        return None

    print(f"Loading model: {load_path}")
    checkpoint = torch.load(load_path, map_location=device, weights_only=True)
    if algo_name in ("sac", "flashsac"):
        actor.load_state_dict(checkpoint["actor"])
        if normalizer and checkpoint.get("obs_normalizer"):
            normalizer.load_state_dict(checkpoint["obs_normalizer"])
            normalizer.eval()
    else:
        actor_state = {k: v for k, v in checkpoint["actor"].items() if k not in ("noise_scales",)}
        actor.load_state_dict(actor_state, strict=False)
        if normalizer and checkpoint.get("obs_normalizer"):
            normalizer.load_state_dict(checkpoint["obs_normalizer"])
            normalizer.eval()

    # Export actor to ONNX
    if load_path_dir is not None:
        onnx_path = os.path.join(load_path_dir, "policy.onnx")
        dummy_input = torch.randn(1, obs_dim, device=device)
        with torch.inference_mode():
            if normalizer:
                dummy_input = normalizer(dummy_input, update=False)
            if algo_name in ("sac", "flashsac"):
                export_module = actor.as_export_module()
                output_names = ["action"]
            else:
                export_module = actor
                output_names = ["action"]
            torch.onnx.export(
                export_module,
                (dummy_input,),
                onnx_path,
                input_names=["obs"],
                output_names=output_names,
                opset_version=17,
            )
        print(f"Exported actor ONNX to {onnx_path}")

        # Verify ONNX output matches PyTorch
        import onnxruntime as ort

        sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
        verify_input = torch.randn(1, obs_dim, device=device)
        with torch.inference_mode():
            onnx_feed = normalizer(verify_input, update=False) if normalizer else verify_input
            pt_output = export_module(onnx_feed)
            if isinstance(pt_output, tuple):
                pt_output = pt_output[0]
            pt_np = pt_output.cpu().numpy()
        onnx_output = sess.run(None, {"obs": onnx_feed.cpu().numpy().astype(np.float32)})[0]
        max_diff = np.max(np.abs(pt_np - onnx_output))
        mean_diff = np.mean(np.abs(pt_np - onnx_output))
        print(f"ONNX vs PyTorch — max_diff: {max_diff:.2e}, mean_diff: {mean_diff:.2e}")
        if max_diff > 1e-4:
            print("WARNING: ONNX output diverges from PyTorch!")
        else:
            print("ONNX export verified OK.")

    if env.state is None:
        env.init_state()

    def _policy_step(obs_np: np.ndarray) -> np.ndarray:
        obs_torch = torch.from_numpy(obs_np).to(device)
        if normalizer:
            obs_torch = normalizer(obs_torch, update=False)
        actions_np = (
            actor.explore(obs_torch, deterministic=True).cpu().numpy()
            if algo_name in ("sac", "flashsac")
            else actor(obs_torch).cpu().numpy()
        )
        state = env.step(actions_np)
        return np.asarray(extract_play_obs(state.obs), dtype=np.float32)

    # Use Motrix native rendering only when no video output directory is available.
    if cfg.training.sim_backend == "motrix" and load_path_dir is None:
        print("Starting interactive visualization (motrix native renderer)...")
        print("Close the render window to exit.")

        with torch.inference_mode():
            try:
                render_play_mode(
                    env,
                    sim_backend="motrix",
                    num_steps=None,
                    initialize=lambda: np.asarray(
                        extract_play_obs(
                            extract_reset_obs(
                                env.reset(np.arange(cfg.training.play_env_num, dtype=np.int32))
                            )
                        ),
                        dtype=np.float32,
                    ),
                    step=_policy_step,
                )
            except Exception as e:
                if "RenderClosedError" in str(type(e).__name__):
                    print("Render window closed.")
                else:
                    raise
        return None

    if load_path_dir is None:
        print(f"Could not resolve checkpoint directory. load_path_dir={load_path_dir}")
        return None

    record_video = bool(getattr(cfg.training, "play_record_video", True))
    output_video = os.path.join(load_path_dir, "play_video.mp4") if record_video else None
    if record_video:
        print("Rendering video ...")
    else:
        print("Running playback without video recording...")
    print("Rendering playback frames...")
    with torch.inference_mode():
        render_play_mode(
            env,
            sim_backend=cfg.training.sim_backend,
            headless=bool(getattr(cfg.training, "play_headless", True)),
            record_video=record_video,
            num_steps=cfg.training.play_steps,
            output_video=output_video,
            initialize=lambda: np.asarray(
                extract_play_obs(
                    extract_reset_obs(
                        env.reset(np.arange(cfg.training.play_env_num, dtype=np.int32))
                    )
                ),
                dtype=np.float32,
            ),
            step=_policy_step,
            camera_kwargs={
                "cam_distance": cfg.training.cam_distance,
                "cam_elevation": cfg.training.cam_elevation,
                "cam_azimuth": cfg.training.cam_azimuth,
            },
        )
    if record_video:
        print(f"Saving video to {output_video} ...")
    print("Done.")
    return output_video


@hydra.main(version_base="1.3", config_path="../conf/offpolicy", config_name="config")
def main(cfg: DictConfig) -> None:
    ensure_registries()

    seed_info = apply_configured_training_seed(cfg, torch_runtime=True, cuda=True)
    algo_name = cfg.algo.algo
    task_name = cfg.training.task_name
    assert_offpolicy_task_choice_matches_algo(cfg, algo_name=algo_name)

    if cfg.training.log_dir is None:
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        log_dir = str(
            get_log_root(ROOT_DIR, cfg) / task_name / f"{timestamp}_{cfg.training.sim_backend}"
        )
    else:
        log_dir = cfg.training.log_dir

    import torch

    tracker = None
    if not cfg.training.play_only:
        tracker = ExperimentTracker(
            root_dir=ROOT_DIR,
            log_dir=log_dir,
            algo_name=algo_name,
            task_name=task_name,
            sim_backend=cfg.training.sim_backend,
            training_cfg=cfg.training,
            full_cfg=cfg,
            device=default_device(torch, cfg.training.device),
            seed_info=seed_info,
        )
        tracker.start()

    try:
        if not cfg.training.play_only:
            runner = build_runner(algo_name, cfg)
            try:
                runner.learn(
                    max_iterations=cfg.algo.max_iterations,
                    save_interval=cfg.algo.save_interval,
                    log_dir=log_dir,
                    logger_type=cfg.training.logger,
                )
                if tracker is not None:
                    tracker.update_summary(getattr(runner, "last_run_summary", None))
            finally:
                runner.close()

        if cfg.training.play_only or not cfg.training.no_play:
            print("@" * 50)
            play_video_path = play_offpolicy(algo_name, cfg)
            if tracker is not None:
                tracker.log_video(play_video_path)
    finally:
        if tracker is not None:
            tracker.finish()


if __name__ == "__main__":
    main()
