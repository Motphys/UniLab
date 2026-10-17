"""Train APPO agent — native multiprocessing."""

from __future__ import annotations

import datetime
import os
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

import hydra
import torch
from omegaconf import DictConfig, OmegaConf
from unisim.backend.base import log_playback_plan

from unilab.base.config_adapter import (
    BackendAdapter,
    create_env,
)
from unilab.base.env_factory import registry_env_factory
from unilab.base.process_device import (
    apply_backend_env_device_override,
    apply_sim_carrier_device_override,
    canonicalize_sim_carrier_device,
    configure_backend_process_device,
    pin_genesis_device_before_cuda_init,
)
from unilab.rl import RslRlVecEnvAdapter
from unilab.rl.algos.appo.runtime import resolve_appo_runtime
from unilab.rl.runtime.offpolicy.thread_budget import (
    apply_torch_thread_runtime,
    resolve_torch_thread_runtime,
)
from unilab.training import (
    algo_config_dict,
    build_run_dir_name,
    ensure_registries,
    get_log_root,
    is_viser_play_render_mode,
    nonfatal_play_step,
    resolve_nan_guard_cfg,
    should_run_playback,
)
from unilab.training.experiment import ExperimentTracker
from unilab.training.onnx_export import export_policy_onnx, verify_policy_onnx
from unilab.training.seed import apply_configured_training_seed
from unilab.training.tensor_runtime import select_sim_carrier_device
from unilab.utils.checkpoint import resolve_appo_checkpoint_path
from unilab.visualization.interactive_playback import (
    RslRlPlaybackConfig,
    create_appo_playback_session,
    normalize_checkpoint_value,
)
from unilab.visualization.playback import camera_cfg_from_training


def _training_resume_requested(load_run: Any) -> bool:
    if load_run is None:
        return False
    return str(load_run) not in {"", "-1"}


def _is_cuda_device(device: str | None) -> bool:
    """Return whether a device alias names CUDA (indexed or unindexed)."""

    return device is not None and str(device).strip().lower().split(":", 1)[0] == "cuda"


def _env_carrier_device(
    cfg: DictConfig,
    collector_device: str | None,
    *,
    runner_device: str | None = None,
) -> str | None:
    """Resolve the process device used by Manager Torch carriers.

    The APPO collector owns the environment even when its actor runs on another
    accelerator family. An explicit CUDA collector request therefore wins;
    otherwise fall back to the runner request. Device auto-detection is handled
    separately and must not opt CPU-authoritative HOST_BRIDGE managers into CUDA.
    """
    if collector_device is not None and _is_cuda_device(collector_device):
        return collector_device
    return (
        runner_device
        if runner_device is not None
        else OmegaConf.select(
            cfg,
            "training.device",
            default=None,
        )
    )


def build_appo_runner_kwargs(
    cfg: DictConfig,
    env_cfg_override: dict | None,
    collector_device: str | None,
    rl_cfg: dict[str, Any] | None = None,
    torch_thread_runtime: dict[str, Any] | None = None,
) -> dict:
    if rl_cfg is None:
        rl_cfg = algo_config_dict(cfg)
    if torch_thread_runtime is None:
        torch_thread_runtime = resolve_torch_thread_runtime(
            getattr(cfg.training, "torch_threads", None)
        )

    routed_env_cfg_override = apply_backend_env_device_override(
        env_cfg_override,
        str(cfg.training.sim_backend),
        learner_device=_env_carrier_device(cfg, collector_device),
    )
    routed_env_cfg_override = apply_sim_carrier_device_override(
        routed_env_cfg_override,
        str(cfg.training.sim_backend),
        sim_carrier_device=select_sim_carrier_device(cfg),
        learner_device=_env_carrier_device(cfg, collector_device),
    )

    runner_kwargs = {
        "env_name": cfg.training.task_name,
        "env_factory": registry_env_factory(
            str(cfg.training.task_name), str(cfg.training.sim_backend)
        ),
        "env_cfg_overrides": routed_env_cfg_override,
        "rl_cfg": rl_cfg,
        "device": cfg.training.device,
        "collector_device": collector_device,
        "num_envs": cfg.algo.num_envs,
        "steps_per_env": cfg.algo.steps_per_env,
        "sim_backend": cfg.training.sim_backend,
        "seed": rl_cfg.get("seed"),
        "torch_thread_runtime": torch_thread_runtime,
        "log_interval": int(cfg.training.log_interval),
    }
    if cfg.training.replay_queue_size is not None:
        runner_kwargs["replay_queue_size"] = cfg.training.replay_queue_size
    load_run = OmegaConf.select(cfg, "algo.load_run", default="-1")
    if _training_resume_requested(load_run):
        resume_path, _ = resolve_appo_checkpoint_path(
            os.path.join(_get_log_root(cfg), cfg.training.task_name),
            str(load_run),
        )
        if resume_path is None:
            raise FileNotFoundError(f"Could not resolve APPO resume checkpoint: {load_run}")
        runner_kwargs["resume_path"] = resume_path

    nan_guard_cfg = resolve_nan_guard_cfg(cfg.training)
    if nan_guard_cfg is not None:
        runner_kwargs["nan_guard_cfg"] = nan_guard_cfg
        from unilab.training.tensor_runtime_hooks import build_tensor_nan_guard_factory

        runner_kwargs["nan_guard_factory"] = build_tensor_nan_guard_factory()
    return runner_kwargs


def apply_appo_runtime_flags(
    rl_cfg: dict[str, Any],
    cfg: DictConfig,
    *,
    training_enabled: bool,
) -> None:
    algorithm_cfg = rl_cfg.setdefault("algorithm", {})
    if not isinstance(algorithm_cfg, dict):
        return
    if not training_enabled:
        algorithm_cfg["enable_compile"] = False


def run_motrix_play_loop(
    env,
    actor,
    device: str,
    play_env_num: int,
    num_steps: int | None = None,
) -> None:
    if env.state is None:
        env.init_state()

    with torch.inference_mode():
        env.run_playback(
            num_steps=num_steps,
            initialize=lambda: _motrix_playback_obs(
                env.reset(torch.ones(play_env_num, dtype=torch.bool, device=env.device))
            ),
            step=lambda obs: _motrix_playback_step(
                env=env,
                actor=actor,
                device=device,
                play_env_num=play_env_num,
                obs=obs,
            ),
        )


def _motrix_playback_obs(state: Any) -> torch.Tensor:
    """Extract the actor observation tensor from a TorchEnv reset result."""
    obs = state[0].obs if hasattr(state[0], "obs") else state[0]["obs"]
    if not isinstance(obs, torch.Tensor):
        raise TypeError(
            f"APPO playback reset observation must be a tensor, got {type(obs).__name__}"
        )
    return obs


def _motrix_playback_step(
    *,
    env: Any,
    actor: Callable[[Any], Any],
    device: str,
    play_env_num: int,
    obs: Any,
) -> torch.Tensor:
    """Run one APPO policy step and return the next actor observation tensor."""
    from tensordict import TensorDict

    if not isinstance(obs, torch.Tensor):
        raise TypeError(
            f"APPO playback step observation must be a tensor, got {type(obs).__name__}"
        )
    action = actor(
        TensorDict(
            {"policy": obs.to(device=device, dtype=torch.float32)},
            batch_size=play_env_num,
        )
    ).to(device=env.device, dtype=torch.float32)
    return _motrix_playback_obs((env.step(action.contiguous()).obs, None))


def _get_log_root(cfg: DictConfig) -> str:
    return str(get_log_root(Path.cwd(), cfg))


def play_appo(
    cfg: DictConfig,
    rl_cfg: dict[str, Any],
    *,
    root_dir: Path | None = None,
    resolve_checkpoint_path: Callable[[DictConfig], tuple[str | None, str | None]] | None = None,
) -> str | None:
    """Play mode for the default APPO runtime.

    Args:
        cfg: Resolved Hydra config for the current run.
        rl_cfg: Resolved algorithm config dictionary from Hydra composition.
        root_dir: Optional project root forwarded by generic runtime callers.
            The default APPO runtime does not need it and ignores the value.
        resolve_checkpoint_path: Optional checkpoint resolver injected by the
            generic script. When omitted, this function falls back to the
            default log-root based APPO checkpoint resolution.

    Returns:
        Output video path for offscreen rendering, or ``None`` when running the
        native Motrix viewer or when no checkpoint could be resolved.
    """
    del root_dir

    if resolve_checkpoint_path is not None:
        load_path, load_path_dir = resolve_checkpoint_path(cfg)
    else:
        log_root = _get_log_root(cfg)
        base_log_dir = os.path.join(log_root, cfg.training.task_name)
        load_path, load_path_dir = resolve_appo_checkpoint_path(base_log_dir, cfg.algo.load_run)

    if not load_path or not os.path.exists(load_path):
        print(f"Could not find run to load. load_path={load_path}")
        return None

    device = cfg.training.device or (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Using device for play: {device}")

    playback_cfg = RslRlPlaybackConfig(
        task=str(cfg.training.task_name),
        load_run=str(cfg.algo.load_run),
        checkpoint=normalize_checkpoint_value(
            OmegaConf.select(cfg, "algo.checkpoint", default=None)
        ),
        action_mode="policy",
        policy_obs_mode="flat",
        algo_log_name=str(cfg.algo.algo_log_name),
        log_root=None,
        num_envs=cfg.training.play_env_num,
    )
    # Genesis pins CUDA_VISIBLE_DEVICES for a non-zero request; adopt the bound
    # in-process device for the policy and the env override.  CUDA-only backends
    # reject a CPU/MPS learner before playback constructs.
    bound_device = configure_backend_process_device(str(cfg.training.sim_backend), device)
    if bound_device is not None:
        device = bound_device
    play_env_cfg_override = apply_backend_env_device_override(
        BackendAdapter(cfg, root_dir=Path.cwd(), algo_name="appo").build_play_env_cfg_override(),
        str(cfg.training.sim_backend),
        learner_device=device,
    )
    play_env_cfg_override = apply_sim_carrier_device_override(
        play_env_cfg_override,
        str(cfg.training.sim_backend),
        sim_carrier_device=select_sim_carrier_device(cfg),
        learner_device=device,
    )
    session, _policy_obs_mode, _checkpoint_path = create_appo_playback_session(
        playback_cfg=playback_cfg,
        cfg=cfg,
        rl_cfg=rl_cfg,
        env_factory=lambda n: create_env(
            cfg,
            num_envs=n,
            env_cfg_override=play_env_cfg_override,
        ),
        root_dir=Path.cwd(),
        device=device,
        wrapper_cls=RslRlVecEnvAdapter,
    )
    env = session.env
    actor = session.actor
    # The checkpoint early-return above guarantees a loaded actor here.
    assert actor is not None

    # Export actor to ONNX
    if load_path_dir is not None:
        with nonfatal_play_step("ONNX export"):
            import torch.nn as nn

            class _DeterministicAPPOActor(nn.Module):
                def __init__(self, mlp: nn.Module):
                    super().__init__()
                    self.mlp = mlp

                def forward(self, obs: torch.Tensor) -> torch.Tensor:
                    return self.mlp(obs)

            export_module = _DeterministicAPPOActor(actor.mlp)
            onnx_path = os.path.join(load_path_dir, "policy.onnx")
            obs_dim = int(session.wrapped_env.num_obs)
            dummy_input = torch.randn(1, obs_dim, device=device)
            export_policy_onnx(export_module, onnx_path, (dummy_input,), input_names=["obs"])

            # Verify ONNX output matches PyTorch
            verify_input = torch.randn(1, obs_dim, device=device)
            verify_policy_onnx(export_module, onnx_path, (verify_input,), input_names=["obs"])

    if is_viser_play_render_mode(getattr(cfg.training, "play_render_mode", "auto")):
        # Browser-based viser playback renders through the shared MuJoCo
        # playback shell; it replaces backend-native playback and records no
        # video. Blocks until Ctrl+C. Imported lazily: the viser scene module
        # binds the optional `mujoco` package, which non-MuJoCo-shell training
        # environments do not install.
        from unilab.visualization.viser_playback import run_viser_playback_from_cfg

        run_viser_playback_from_cfg(session, cfg, entrypoint="train_appo play")
        return None

    play_video_path: str | None = None
    with nonfatal_play_step("video rendering"):
        with torch.inference_mode():
            play_video_path = env.run_playback_mode(
                play_render_mode=getattr(cfg.training, "play_render_mode", "auto"),
                play_steps=getattr(cfg.training, "play_steps", None),
                output_video=os.path.join(load_path_dir, "play_video.mp4")
                if load_path_dir
                else None,
                render_spacing=float(
                    getattr(cfg.training, "render_spacing", getattr(env.cfg, "render_spacing", 1.0))
                ),
                initialize=session.reset,
                step=lambda _obs: session.step_once(),
                camera_kwargs=camera_cfg_from_training(cfg.training),
                on_plan=log_playback_plan,
            )
    if play_video_path is not None:
        print(f"Saving video to {play_video_path} ...")
    print("Done.")
    return cast("str | None", play_video_path)


@hydra.main(version_base="1.3", config_path="../conf/appo", config_name="config")
def main(cfg: DictConfig) -> None:
    # Genesis/Quadrants binds the first CUDA_VISIBLE_DEVICES entry, and even
    # torch.cuda.is_available() latches the variable in the CUDA runtime, so
    # the pin must precede registry bootstrap and device auto-detection
    # (issue #1508).  APPO has no device topology; only an explicit learner
    # device can require the pin.
    pinned_device = pin_genesis_device_before_cuda_init(
        str(cfg.training.sim_backend),
        learner_device=cfg.training.device,
    )

    ensure_registries()

    # Convert algo config to plain dict for APPORunner / RSL-RL internals
    rl_cfg = algo_config_dict(cfg)
    apply_appo_runtime_flags(rl_cfg, cfg, training_enabled=not cfg.training.play_only)
    appo_runtime = resolve_appo_runtime(rl_cfg, default_play_fn=play_appo)

    if cfg.training.log_dir is None:
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        log_root = _get_log_root(cfg)
        log_dir = os.path.join(
            log_root,
            cfg.training.task_name,
            build_run_dir_name(timestamp, str(cfg.training.sim_backend)),
        )
    else:
        log_dir = cfg.training.log_dir

    collector_device = cfg.training.collector_device
    if collector_device == "gpu":
        collector_device = "mps" if torch.backends.mps.is_available() else "cuda"

    learner_device = cfg.training.device or (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    if pinned_device is not None:
        # The process was pinned to its requested GPU; use the in-process
        # index for both learner and collector.
        learner_device = pinned_device
        if collector_device is not None and _is_cuda_device(collector_device):
            collector_device = pinned_device

    # A non-zero Genesis request pins CUDA_VISIBLE_DEVICES process-wide
    # (Quadrants only honors the first visible device); the bound in-process
    # device replaces both the learner and collector device.  CUDA-only backends
    # reject a CPU/MPS learner before training constructs.
    bound_device = configure_backend_process_device(str(cfg.training.sim_backend), learner_device)
    if bound_device is not None:
        learner_device = bound_device
        if collector_device is not None and _is_cuda_device(collector_device):
            collector_device = bound_device

    env_cfg_override = apply_backend_env_device_override(
        BackendAdapter(cfg, root_dir=Path.cwd(), algo_name="appo").build_task_env_cfg_override(),
        str(cfg.training.sim_backend),
        learner_device=_env_carrier_device(cfg, collector_device, runner_device=learner_device),
    )
    env_cfg_override = apply_sim_carrier_device_override(
        env_cfg_override,
        str(cfg.training.sim_backend),
        sim_carrier_device=select_sim_carrier_device(cfg),
        learner_device=_env_carrier_device(cfg, collector_device, runner_device=learner_device),
    )
    seed_info = apply_configured_training_seed(cfg, torch_runtime=True, cuda=True)
    resolved_sim_carrier = (
        canonicalize_sim_carrier_device(env_cfg_override.get("sim_carrier_device"))
        if str(cfg.training.sim_backend) == "mujoco"
        else None
    )
    torch_thread_runtime = resolve_torch_thread_runtime(
        getattr(cfg.training, "torch_threads", None)
    )
    apply_torch_thread_runtime(torch_thread_runtime, role="learner")

    tracker = None
    if not cfg.training.play_only:
        tracker = ExperimentTracker(
            root_dir=Path.cwd(),
            log_dir=log_dir,
            algo_name="appo",
            task_name=cfg.training.task_name,
            sim_backend=cfg.training.sim_backend,
            training_cfg=cfg.training,
            full_cfg=cfg,
            device=learner_device,
            configured_device=str(cfg.training.device or "auto"),
            collector_device=collector_device,
            device_resolution={
                "learner": {
                    "configured": str(cfg.training.device or "auto"),
                    "resolved": str(learner_device),
                },
                **(
                    {
                        "collector": {
                            "configured": str(cfg.training.collector_device or "auto"),
                            "resolved": str(collector_device),
                        }
                    }
                    if collector_device is not None
                    else {}
                ),
                **(
                    {
                        "sim_carrier": {
                            "configured": select_sim_carrier_device(cfg),
                            "resolved": resolved_sim_carrier,
                        }
                    }
                    if resolved_sim_carrier is not None
                    else {}
                ),
            },
            seed_info=seed_info,
        )
        tracker.start()

    try:
        if not cfg.training.play_only:
            runner = appo_runtime.runner_cls(
                **build_appo_runner_kwargs(
                    cfg,
                    env_cfg_override=env_cfg_override,
                    collector_device=collector_device,
                    rl_cfg=rl_cfg,
                    torch_thread_runtime=torch_thread_runtime,
                )
            )

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

        if should_run_playback(
            play_only=cfg.training.play_only,
            no_play=cfg.training.no_play,
            play_render_mode=getattr(cfg.training, "play_render_mode", "auto"),
        ):
            play_video_path = appo_runtime.play_fn(
                cfg,
                rl_cfg,
                root_dir=Path.cwd(),
                resolve_checkpoint_path=lambda current_cfg: resolve_appo_checkpoint_path(
                    os.path.join(_get_log_root(current_cfg), current_cfg.training.task_name),
                    current_cfg.algo.load_run,
                ),
            )
            if tracker is not None:
                tracker.log_video(play_video_path)
    finally:
        if tracker is not None:
            tracker.finish()


if __name__ == "__main__":
    main()
