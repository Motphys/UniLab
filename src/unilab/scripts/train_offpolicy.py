"""Shared off-policy (SAC/TD3/FlashSAC) train/play implementation.

This module is no longer runnable directly; use the per-algorithm entry
scripts instead: ``unilab/scripts/train_sac.py``, ``unilab/scripts/train_td3.py``, and
``unilab/scripts/train_flashsac.py``.
"""

from __future__ import annotations

import datetime
import os
import sys
from contextlib import nullcontext
from pathlib import Path
from typing import Any, cast

from omegaconf import DictConfig, OmegaConf
from uni_rl.ipc.dp_launcher import (
    UNILAB_DP_LOG_DIR,
    DpRankSupervisor,
    apply_dp_rank_config,
    current_dp_rank,
    resolve_collector_cpu_ids,
    resolve_dp_rank_device,
    resolve_dp_rendezvous_path,
    resolve_dp_topology,
    validate_dp_launchable,
)
from unisim.backend.base import log_playback_plan

from unilab.base.config_adapter import create_env
from unilab.base.env_factory import registry_env_factory
from unilab.base.process_device import bind_backend_process_device, configure_backend_process_device
from unilab.training import (
    assert_offpolicy_task_choice_matches_algo,
    build_run_dir_name,
    ensure_registries,
    get_log_root,
    resolve_nan_guard_cfg,
    should_run_playback,
)
from unilab.training.experiment import ExperimentTracker
from unilab.training.onnx_export import export_policy_onnx, verify_policy_onnx
from unilab.utils.checkpoint import (
    resolve_offpolicy_checkpoint_path as resolve_checkpoint_path,
)
from unilab.utils.seed import apply_configured_training_seed
from unilab.visualization.interactive_playback import (
    RslRlPlaybackConfig,
    create_sac_playback_session,
    default_device,
    resolve_play_actor_spec,
    resolve_play_obs_dims,
)
from unilab.visualization.interactive_playback import (
    build_offpolicy_env_cfg_override as _build_offpolicy_env_cfg_override,
)
from unilab.visualization.interactive_playback import (
    build_offpolicy_play_env_cfg_override as _build_offpolicy_play_env_cfg_override,
)


def enable_faulthandler() -> None:
    """Enable fatal-signal Python stack dumps unless explicitly disabled."""
    if os.environ.get("UNILAB_FAULTHANDLER", "1").lower() in {"0", "false", "no", "off"}:
        return
    try:
        import faulthandler

        if not faulthandler.is_enabled():
            faulthandler.enable(all_threads=True)
    except Exception as exc:
        print(f"[train_offpolicy] faulthandler unavailable: {exc}", file=sys.stderr)


def build_failure_summary(exc: BaseException, run_summary: Any | None = None) -> dict[str, Any]:
    summary = dict(run_summary) if isinstance(run_summary, dict) else {}
    if summary.get("status") == "completed":
        summary["status"] = "failed"
    else:
        summary.setdefault("status", "failed")
    summary["error_type"] = type(exc).__name__
    summary["error"] = str(exc)
    return summary


def build_offpolicy_env_cfg_override(algo_name: str, cfg: DictConfig) -> dict[str, Any] | None:
    return _build_offpolicy_env_cfg_override(algo_name, cfg, root_dir=Path.cwd())


def build_offpolicy_play_env_cfg_override(algo_name: str, cfg: DictConfig) -> dict[str, Any] | None:
    return _build_offpolicy_play_env_cfg_override(algo_name, cfg, root_dir=Path.cwd())


def build_runner(algo_name: str, cfg: DictConfig, log_dir: str | None = None):
    """Build algorithm runner from unified Hydra config."""
    env_cfg_override = build_offpolicy_env_cfg_override(algo_name, cfg)
    env_factory = registry_env_factory(str(cfg.training.task_name), str(cfg.training.sim_backend))
    from uni_rl.offpolicy.thread_budget import (
        apply_torch_thread_runtime,
        resolve_torch_thread_runtime,
    )

    # Cold-path DP CPU partition: each rank's collector owns one contiguous
    # CPU block (single rank keeps the legacy unset behavior). The ids only
    # reach the collector env override — never the num_envs=1 probe envs,
    # whose MuJoCo pool would size itself from len(cpu_ids).
    # world_size comes from training.devices (rank 0 has no UNILAB_DP_* env;
    # only spawned ranks carry it), rank from the env (0 for rank 0).
    dp_devices = resolve_dp_topology(cfg.training.devices)
    dp_world_size = len(dp_devices) if dp_devices is not None else 1
    dp_rank = current_dp_rank()
    from unilab.utils.device import get_default_device

    rank_device = resolve_dp_rank_device(dp_devices, dp_rank) or get_default_device()
    # Bind backend-global device state before algorithm builders materialize
    # their probe envs. The spawned collector repeats this binding in its own
    # process using the same rank-local device.
    configure_backend_process_device(str(cfg.training.sim_backend), rank_device)
    host_cpu_count = os.cpu_count() or 1
    explicit_cpu_ids = getattr(cfg.training, "dp_collector_cpu_ids", None)
    if explicit_cpu_ids is not None:
        explicit_cpu_ids = cast(list, OmegaConf.to_container(explicit_cpu_ids, resolve=True))
    collector_cpu_ids = resolve_collector_cpu_ids(
        dp_world_size,
        dp_rank,
        host_cpu_count,
        explicit=explicit_cpu_ids,
    )

    # Cold-path DP process-group assembly. world_size == 1 keeps dp_sync=None
    # (bit-identical single-rank path); multi-rank learners attach the group's
    # flat-gradient collective at their optimizer boundaries.
    dp_sync = None
    if dp_world_size > 1:
        if dp_rank == 0 and log_dir is None:
            raise ValueError(
                "build_runner requires log_dir for multi-GPU data-parallel rank 0 "
                "(it anchors the DP rendezvous FileStore)"
            )
        from uni_rl.ipc.dp_sync import DpParameterSync

        dp_sync = DpParameterSync(
            world_size=dp_world_size,
            rank=dp_rank,
            rendezvous_path=resolve_dp_rendezvous_path(cast(str, log_dir), rank=dp_rank),
            device=rank_device,
        )

    torch_thread_runtime = resolve_torch_thread_runtime(
        getattr(cfg.training, "torch_threads", None),
        cpu_count=host_cpu_count // dp_world_size if dp_world_size > 1 else None,
    )
    apply_torch_thread_runtime(torch_thread_runtime, role="learner")

    _nan_guard_cfg = resolve_nan_guard_cfg(cfg.training)

    replay_prefetch_mode = getattr(cfg.training, "replay_prefetch_mode", "one_tick")
    if replay_prefetch_mode != "one_tick":
        raise ValueError(
            f"Unsupported training.replay_prefetch_mode={replay_prefetch_mode!r}; "
            "expected 'one_tick'"
        )
    from uni_rl.ipc.replay_pipelines.gpu_resident import require_offpolicy_replay_device

    replay_device = require_offpolicy_replay_device(rank_device)
    builder_kwargs: dict[str, Any] = {
        "env_factory": env_factory,
        "env_cfg_override": env_cfg_override,
        "replay_prefetch_mode": replay_prefetch_mode,
        "device": replay_device,
        "nan_guard_cfg": _nan_guard_cfg,
        "torch_thread_runtime": torch_thread_runtime,
        "collector_cpu_ids": collector_cpu_ids,
        "dp_sync": dp_sync,
        "backend_device_binder": bind_backend_process_device,
    }
    if algo_name == "sac":
        from uni_rl.fast_sac.double_buffer import (
            build_sac_double_buffer_runner,
        )

        runner = build_sac_double_buffer_runner(cfg, **builder_kwargs)
    elif algo_name == "td3":
        from uni_rl.fast_td3.double_buffer import (
            build_td3_double_buffer_runner,
        )

        runner = build_td3_double_buffer_runner(cfg, **builder_kwargs)
    elif algo_name == "flashsac":
        from uni_rl.flash_sac.double_buffer import (
            build_flashsac_double_buffer_runner,
        )

        runner = build_flashsac_double_buffer_runner(cfg, **builder_kwargs)
    else:
        raise ValueError(f"Unsupported algo: {algo_name}")

    return runner


def play_offpolicy(algo_name: str, cfg: DictConfig) -> str | None:
    """Play pipeline for off-policy algorithms."""
    import torch

    load_path, load_path_dir = resolve_checkpoint_path(
        Path.cwd(),
        cfg.algo.algo_log_name,
        cfg.training.task_name,
        cfg.algo.load_run,
    )
    if not load_path or not os.path.exists(load_path):
        print(f"Could not find checkpoint. load_path={load_path}")
        return None

    devices = resolve_dp_topology(cfg.training.devices)
    device = default_device(torch, resolve_dp_rank_device(devices, current_dp_rank()))
    print(f"Using device for play: {device}")

    playback_cfg = RslRlPlaybackConfig(
        task=str(cfg.training.task_name),
        load_run=str(cfg.algo.load_run),
        checkpoint=None,
        action_mode="policy",
        policy_obs_mode="actor",
        algo_log_name=str(cfg.algo.algo_log_name),
        log_root=None,
        num_envs=int(cfg.training.play_env_num),
    )
    session, _policy_obs_mode, _checkpoint_path = create_sac_playback_session(
        playback_cfg=playback_cfg,
        cfg=cfg,
        env_factory=lambda n: create_env(
            cfg,
            num_envs=n,
            env_cfg_override=build_offpolicy_play_env_cfg_override(algo_name, cfg),
        ),
        root_dir=Path.cwd(),
        device=device,
        algo_name=algo_name,
    )
    env = cast(Any, session.env)
    actor = session.actor
    normalizer = session.normalizer
    actor_algo_type = session.actor_algo_type

    # Export actor to ONNX
    if load_path_dir is not None and bool(getattr(cfg.training, "export_onnx", True)):
        obs_dim, critic_obs_dim = resolve_play_obs_dims(env.obs_groups_spec)
        _, actor_kwargs = resolve_play_actor_spec(
            algo_name,
            cfg,
            obs_dim=obs_dim,
            critic_obs_dim=critic_obs_dim,
        )
        onnx_path = os.path.join(load_path_dir, "policy.onnx")
        dummy_input = torch.randn(1, obs_dim, device=device)
        dummy_priv_info = (
            torch.zeros(
                (1, int(actor_kwargs["priv_info_dim"])),
                device=device,
                dtype=dummy_input.dtype,
            )
            if actor_algo_type == "hora_sac"
            else None
        )
        with torch.inference_mode():
            if normalizer:
                dummy_input = normalizer(dummy_input, update=False)
            assert actor is not None
            if algo_name in ("sac", "flashsac"):
                export_module = actor.as_export_module()
            else:
                export_module = actor
            export_inputs = (
                (dummy_input, dummy_priv_info) if dummy_priv_info is not None else (dummy_input,)
            )
        input_names = ["obs", "priv_info"] if dummy_priv_info is not None else ["obs"]
        export_policy_onnx(export_module, onnx_path, export_inputs, input_names=input_names)

        # Verify ONNX output matches PyTorch
        verify_input = torch.randn(1, obs_dim, device=device)
        with torch.inference_mode():
            onnx_feed = normalizer(verify_input, update=False) if normalizer else verify_input
            verify_priv_info = (
                torch.zeros((1, int(actor_kwargs["priv_info_dim"])), device=device)
                if actor_algo_type == "hora_sac"
                else None
            )
        verify_inputs = (
            (onnx_feed, verify_priv_info) if verify_priv_info is not None else (onnx_feed,)
        )
        verify_policy_onnx(export_module, onnx_path, verify_inputs, input_names=input_names)
    elif load_path_dir is not None:
        print("Skipping ONNX export because training.export_onnx=false.")

    with torch.inference_mode():
        play_video_path = env.run_playback_mode(
            play_render_mode=getattr(cfg.training, "play_render_mode", "auto"),
            play_steps=getattr(cfg.training, "play_steps", None),
            output_video=os.path.join(load_path_dir, "play_video.mp4") if load_path_dir else None,
            initialize=session.reset,
            step=lambda _obs: session.step_once(),
            camera_kwargs={
                "cam_distance": cfg.training.cam_distance,
                "cam_elevation": cfg.training.cam_elevation,
                "cam_azimuth": cfg.training.cam_azimuth,
            },
            on_plan=log_playback_plan,
        )
    if play_video_path is not None:
        print(f"Saving video to {play_video_path} ...")
    print("Done.")
    return cast("str | None", play_video_path)


def main(cfg: DictConfig) -> None:
    enable_faulthandler()
    ensure_registries()

    devices = resolve_dp_topology(cfg.training.devices)
    rank = current_dp_rank()
    rank_device = apply_dp_rank_config(cfg, devices, rank)

    seed_info = apply_configured_training_seed(cfg, torch_runtime=True, cuda=True)
    algo_name = cfg.algo.algo
    task_name = cfg.training.task_name
    assert_offpolicy_task_choice_matches_algo(cfg, algo_name=algo_name)

    if cfg.training.log_dir is None:
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        run_dir_name = build_run_dir_name(
            timestamp,
            str(cfg.training.sim_backend),
            world_size=len(devices) if devices is not None else 1,
        )
        log_dir = str(get_log_root(Path.cwd(), cfg) / task_name / run_dir_name)
    else:
        log_dir = cfg.training.log_dir
    if rank > 0:
        # Spawned ranks reuse the canonical run directory but never create
        # logging backends, checkpoints, summaries, or traces there.
        log_dir = os.environ[UNILAB_DP_LOG_DIR]

    supervisor: DpRankSupervisor | None = None
    if devices is not None and rank == 0 and len(devices) > 1:
        validate_dp_launchable(devices)
        supervisor = DpRankSupervisor(devices, log_dir)

    import torch

    tracker = None
    if not cfg.training.play_only and rank == 0:
        tracker = ExperimentTracker(
            root_dir=Path.cwd(),
            log_dir=log_dir,
            algo_name=algo_name,
            task_name=task_name,
            sim_backend=cfg.training.sim_backend,
            training_cfg=cfg.training,
            full_cfg=cfg,
            device=default_device(torch, rank_device),
            seed_info=seed_info,
        )
        tracker.start()

    try:
        with supervisor if supervisor is not None else nullcontext():
            if not cfg.training.play_only:
                runner = None
                try:
                    runner = build_runner(algo_name, cfg, log_dir=log_dir)
                    runner.learn(
                        max_iterations=cfg.algo.max_iterations,
                        save_interval=cfg.algo.save_interval,
                        log_dir=log_dir,
                        logger_type=cfg.training.logger,
                    )
                    run_summary = getattr(runner, "last_run_summary", None)
                    if isinstance(run_summary, dict) and run_summary.get("status") not in (
                        None,
                        "completed",
                    ):
                        raise RuntimeError(
                            f"Off-policy training ended with status={run_summary.get('status')!r}"
                        )
                    if tracker is not None:
                        tracker.update_summary(run_summary)
                except BaseException as exc:
                    if tracker is not None:
                        tracker.update_summary(
                            build_failure_summary(exc, getattr(runner, "last_run_summary", None))
                        )
                    raise
                finally:
                    if runner is not None:
                        runner.close()

            if rank == 0 and should_run_playback(
                play_only=cfg.training.play_only,
                no_play=cfg.training.no_play,
                play_render_mode=getattr(cfg.training, "play_render_mode", "auto"),
            ):
                print("@" * 50)
                play_video_path = play_offpolicy(algo_name, cfg)
                if tracker is not None:
                    tracker.log_video(play_video_path)
    finally:
        if tracker is not None:
            tracker.finish()


if __name__ == "__main__":
    raise SystemExit(
        "unilab/scripts/train_offpolicy.py is a shared implementation module and is no "
        "longer runnable directly. Use unilab/scripts/train_sac.py, "
        "unilab/scripts/train_td3.py, or unilab/scripts/train_flashsac.py instead."
    )
