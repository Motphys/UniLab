"""Shared off-policy (SAC/FlashSAC) train/play implementation.

This module is no longer runnable directly; use the per-algorithm entry
scripts instead: ``unilab/scripts/train_sac.py``,
``unilab/scripts/train_flashsac.py``.
"""

from __future__ import annotations

import datetime
import os
import sys
from contextlib import contextmanager, nullcontext
from functools import partial
from pathlib import Path
from typing import Any, cast

from omegaconf import DictConfig, OmegaConf
from uni_rl.ipc.dp_launcher import (
    UNILAB_DP_LOG_DIR,
    UNILAB_DP_WORLD_SIZE,
    DpRankSupervisor,
    apply_dp_rank_config,
    current_dp_rank,
    current_dp_world_size,
    reject_removed_device_config,
    resolve_collector_cpu_ids,
    resolve_dp_rank_device,
    resolve_dp_rendezvous_path,
    selected_visible_entries,
)
from unisim.backend.base import log_playback_plan

from unilab.base.config_adapter import create_env
from unilab.base.env_factory import registry_env_factory
from unilab.base.process_device import (
    apply_backend_env_device_override,
    apply_manager_torch_device_override,
    bind_backend_process_device_for_backend,
    configure_backend_process_device,
    pin_genesis_device_before_cuda_init,
    resolve_backend_env_device_id,
    resolve_backend_process_device,
)
from unilab.training import (
    assert_offpolicy_task_choice_matches_algo,
    build_run_dir_name,
    ensure_registries,
    get_log_root,
    is_viser_play_render_mode,
    nonfatal_play_step,
    parse_checkpoint_path,
    resolve_nan_guard_cfg,
    should_run_playback,
)
from unilab.training.cuda_mps_cli import (
    CudaMpsCliError,
    select_environment_for_gpu,
)
from unilab.training.cuda_process_sharing import (
    CudaProcessSharingEvidence,
    probe_cuda_process_sharing,
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
    resolve_play_obs_dims,
)


def _rank_local_cuda_device() -> str | None:
    """Resolve local ``cuda:0`` without requiring a new sibling API export."""
    raw = os.environ.get("CUDA_VISIBLE_DEVICES")
    if raw is None:
        return None
    entries = tuple(entry.strip() for entry in raw.split(",") if entry.strip())
    return "cuda:0" if len(entries) == 1 else None


def _prepare_rank_zero_dp_visibility(world_size: int) -> None:
    """Give rank 0 its one-entry slice of a multi-entry parent CUDA mask.

    ``DpRankSupervisor`` spawns ranks 1..N-1 with one opaque parent entry each.
    Rank 0 previously retained the whole mask, so ``resolve_dp_rank_device``
    rejected the multi-GPU launch before the supervisor could start. This is a
    pre-CUDA, rank-0-only remap: child environments are derived from the parent
    entries before this function changes the current process namespace.
    """

    if world_size <= 1 or current_dp_rank() != 0:
        return
    raw = os.environ.get("CUDA_VISIBLE_DEVICES")
    entries = tuple(entry.strip() for entry in (raw or "").split(",") if entry.strip())
    if len(entries) != world_size:
        return
    if len(entries) <= 1:
        return
    global _DP_PARENT_VISIBLE_ENTRIES
    _DP_PARENT_VISIBLE_ENTRIES = entries
    os.environ["CUDA_VISIBLE_DEVICES"] = entries[0]


_DP_PARENT_VISIBLE_ENTRIES: tuple[str, ...] | None = None


def _build_dp_rank_supervisor(world_size: int, log_dir: str) -> DpRankSupervisor:
    """Build the rank supervisor while preserving the original parent mask."""

    previous = os.environ.get("CUDA_VISIBLE_DEVICES")
    if _DP_PARENT_VISIBLE_ENTRIES is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(_DP_PARENT_VISIBLE_ENTRIES)
    try:
        return DpRankSupervisor(world_size=world_size, log_dir=log_dir)
    finally:
        if previous is None:
            os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = previous


def _export_inferred_dp_world_size(world_size: int) -> None:
    """Make an inferred parent launch visible to runner assembly and children."""

    if world_size > 1 and current_dp_rank() == 0 and not os.environ.get(UNILAB_DP_WORLD_SIZE):
        os.environ[UNILAB_DP_WORLD_SIZE] = str(world_size)


def _requested_dp_world_size() -> int:
    """Infer a parent off-policy DP request from opaque CUDA visibility.

    A parent with one entry is an ordinary single-rank launch. A parent with
    multiple opaque entries requests one rank per entry, matching the
    documented off-policy topology. Spawned ranks carry the authoritative
    ``UNILAB_DP_*`` environment and have one-entry visibility, so they never
    enter this inference path.
    """

    raw = os.environ.get("CUDA_VISIBLE_DEVICES")
    if raw is None or not raw.strip() or raw.strip() == "-1":
        return 1
    entries = tuple(entry.strip() for entry in raw.split(",") if entry.strip())
    return len(entries) or 1


from unilab.visualization.interactive_playback import (
    build_offpolicy_env_cfg_override as _build_offpolicy_env_cfg_override,
)
from unilab.visualization.interactive_playback import (
    build_offpolicy_play_env_cfg_override as _build_offpolicy_play_env_cfg_override,
)
from unilab.visualization.playback import camera_cfg_from_training


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


@contextmanager
def _genesis_probe_para_level(sim_backend: str):
    """Keep the one-env Genesis probe on the collector's kernel cache lane."""
    if sim_backend != "genesis":
        yield
        return

    previous_value = os.environ.get("GS_PARA_LEVEL")
    os.environ["GS_PARA_LEVEL"] = "2"
    try:
        yield
    finally:
        if previous_value is None:
            os.environ.pop("GS_PARA_LEVEL", None)
        else:
            os.environ["GS_PARA_LEVEL"] = previous_value


def build_offpolicy_env_cfg_override(algo_name: str, cfg: DictConfig) -> dict[str, Any] | None:
    base = _build_offpolicy_env_cfg_override(algo_name, cfg, root_dir=Path.cwd())
    rank = current_dp_rank()
    from unilab.utils.device import get_default_device

    rank_device = resolve_dp_rank_device(rank) or get_default_device()
    routed = apply_backend_env_device_override(
        base,
        str(cfg.training.sim_backend),
        learner_device=rank_device,
    )
    return apply_manager_torch_device_override(
        routed,
        str(cfg.training.sim_backend),
        collector_tensor_device=str(
            OmegaConf.select(cfg, "training.collector_tensor_device", default="cpu")
        ),
        learner_device=rank_device,
    )


def build_offpolicy_play_env_cfg_override(algo_name: str, cfg: DictConfig) -> dict[str, Any] | None:
    base = _build_offpolicy_play_env_cfg_override(algo_name, cfg, root_dir=Path.cwd())
    rank = current_dp_rank()
    from unilab.utils.device import get_default_device

    rank_device = resolve_dp_rank_device(rank) or get_default_device()
    routed = apply_backend_env_device_override(
        base,
        str(cfg.training.sim_backend),
        learner_device=rank_device,
    )
    return apply_manager_torch_device_override(
        routed,
        str(cfg.training.sim_backend),
        collector_tensor_device=str(
            OmegaConf.select(cfg, "training.collector_tensor_device", default="cpu")
        ),
        learner_device=rank_device,
    )


def build_runner(algo_name: str, cfg: DictConfig, log_dir: str | None = None):
    """Build algorithm runner from unified Hydra config."""
    # Cold-path DP CPU partition: each rank's collector owns one contiguous
    # CPU block (single rank keeps the legacy unset behavior). The ids only
    # reach the collector env override — never the num_envs=1 probe envs,
    # whose MuJoCo pool would size itself from len(cpu_ids).
    # World size comes from the launcher environment (rank 0 has no UNILAB_DP_*
    # env; spawned ranks carry it).
    dp_world_size = current_dp_world_size()
    dp_rank = current_dp_rank()
    from unilab.utils.device import get_default_device

    rank_device = resolve_dp_rank_device(dp_rank) or get_default_device()
    # Bind backend-global device state before algorithm builders materialize
    # their probe envs. The spawned collector repeats this binding in its own
    # process using the same rank-local device.  A non-zero Genesis request
    # pins CUDA_VISIBLE_DEVICES for the whole rank process (Quadrants only
    # honors the first visible device), so the bound in-process device
    # replaces rank_device for the learner, the probe, and the override below.
    bound_device = configure_backend_process_device(
        str(cfg.training.sim_backend),
        rank_device,
    )
    if bound_device is not None:
        rank_device = bound_device
    collector_device = resolve_backend_process_device(
        str(cfg.training.sim_backend),
        rank_device,
    )
    _select_cuda_mps_daemon_environment(
        getattr(cfg.training, "cuda_process_sharing", None),
        str(cfg.training.sim_backend),
        rank_device,
    )
    cuda_process_sharing = probe_cuda_process_sharing(
        getattr(cfg.training, "cuda_process_sharing", None),
        rank_device,
        collector_device,
        backend=str(cfg.training.sim_backend),
        rank=dp_rank,
        world_size=dp_world_size,
    )

    env_factory = registry_env_factory(str(cfg.training.task_name), str(cfg.training.sim_backend))
    from uni_rl.offpolicy.thread_budget import (
        apply_torch_thread_runtime,
        resolve_torch_thread_runtime,
    )

    # Every rank is single-GPU; rank-local visibility routes the backend payload.
    env_cfg_override = apply_backend_env_device_override(
        build_offpolicy_env_cfg_override(algo_name, cfg),
        str(cfg.training.sim_backend),
        learner_device=rank_device,
    )
    # Device-resident backend capabilities and rank process-device binding own
    # Manager public tensor placement (issue #1743); no environment switch or
    # manual placement injection remains.
    host_cpu_count = os.cpu_count() or 1
    explicit_cpu_ids = getattr(cfg.training, "dp_collector_cpu_ids", None)
    if explicit_cpu_ids is not None:
        explicit_cpu_ids = cast(list, OmegaConf.to_container(explicit_cpu_ids, resolve=True))
    collector_cpu_ids = resolve_collector_cpu_ids(
        dp_world_size,
        dp_rank,
        None,
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
    if _nan_guard_cfg is None:
        nan_guard_factory = None
    else:
        from unilab.training.tensor_runtime_hooks import build_tensor_nan_guard_factory

        nan_guard_factory = build_tensor_nan_guard_factory()

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
        "nan_guard_factory": nan_guard_factory,
        "torch_thread_runtime": torch_thread_runtime,
        "collector_cpu_ids": collector_cpu_ids,
        "dp_sync": dp_sync,
        # Keep the injected collector hook top-level/pickleable while selecting
        # the backend lazily inside the spawned process.  The historical
        # one-argument binder remains available for mjwarp compatibility; new
        # backends must not accidentally bind through the wrong runtime.
        "backend_device_binder": partial(
            bind_backend_process_device_for_backend,
            str(cfg.training.sim_backend),
        ),
    }
    with _genesis_probe_para_level(str(cfg.training.sim_backend)):
        if algo_name == "sac":
            from uni_rl.algos.sac.double_buffer import (
                build_sac_double_buffer_runner,
            )

            runner = build_sac_double_buffer_runner(cfg, **builder_kwargs)
        elif algo_name == "flashsac":
            from uni_rl.algos.flash_sac.double_buffer import (
                build_flashsac_double_buffer_runner,
            )

            runner = build_flashsac_double_buffer_runner(cfg, **builder_kwargs)
        else:
            raise ValueError(f"Unsupported algo: {algo_name}")

    _attach_cuda_process_sharing_manifest(runner, cuda_process_sharing)
    return runner


def _select_cuda_mps_daemon_environment(
    requested: Any,
    backend: str,
    learner_device: str,
) -> None:
    """Bind the sole live UniLab-recorded daemon in this trainer process.

    Explicit deployment environment wins. Otherwise an ``mps`` request resolves
    the user-owned daemon record for the rank-local GPU before the fail-closed
    probe. This never starts or stops a daemon and never mutates a parent shell.
    """

    del backend
    if requested != "mps" or os.environ.get("CUDA_MPS_PIPE_DIRECTORY"):
        return
    try:
        import torch
    except ImportError:
        return
    if not torch.cuda.is_available():
        return
    normalized_device = learner_device.strip().lower()
    if normalized_device == "cuda":
        index = 0
    else:
        base, separator, index_text = normalized_device.partition(":")
        if base != "cuda" or not separator or not index_text.isdigit():
            return
        index = int(index_text)
    if index >= int(torch.cuda.device_count()):
        return
    properties = torch.cuda.get_device_properties(index)
    uuid = str(getattr(properties, "uuid", "")).strip()
    if not uuid:
        return
    try:
        environment = select_environment_for_gpu(uuid)
    except CudaMpsCliError as exc:
        raise ValueError(str(exc)) from exc
    os.environ.setdefault("CUDA_MPS_PIPE_DIRECTORY", environment["CUDA_MPS_PIPE_DIRECTORY"])
    os.environ.setdefault("CUDA_MPS_LOG_DIRECTORY", environment["CUDA_MPS_LOG_DIRECTORY"])


def _attach_cuda_process_sharing_manifest(
    runner: Any,
    evidence: CudaProcessSharingEvidence,
) -> None:
    """Merge validated mode evidence into the producer-owned manifest."""

    runtime_manifest = getattr(runner, "runtime_manifest", None)
    if isinstance(runtime_manifest, dict):
        runtime_manifest["cuda_process_sharing"] = evidence.manifest()
        return
    runner.runtime_manifest = {"cuda_process_sharing": evidence.manifest()}


def resolve_train_resume_checkpoint(cfg: DictConfig) -> str | None:
    """Resolve the opt-in training-resume checkpoint selected by ``algo.resume``.

    Resume is explicit: the ``algo.load_run`` default of ``-1`` (latest run)
    must not silently turn a fresh training launch into a resume. When
    ``algo.resume`` is true, ``algo.load_run``/``algo.checkpoint`` select the
    checkpoint exactly like play mode does.
    """
    if not bool(OmegaConf.select(cfg, "algo.resume", default=False)):
        return None
    load_path, _ = parse_checkpoint_path(cfg, root_dir=Path.cwd())
    if load_path is None or not load_path.is_file():
        raise RuntimeError(
            "algo.resume=true but no checkpoint could be resolved: "
            f"algo.load_run={cfg.algo.load_run!r} "
            f"task={cfg.training.task_name}. Use algo.load_run=<run-dir> "
            "and optionally algo.checkpoint=<iteration-or-filename>."
        )
    return str(load_path)


def play_offpolicy(
    algo_name: str,
    cfg: DictConfig,
    *,
    load_run: str | None = None,
) -> str | None:
    """Play pipeline for off-policy algorithms."""
    import torch

    selected_load_run = str(cfg.algo.load_run if load_run is None else load_run)
    load_path, load_path_dir = resolve_checkpoint_path(
        Path.cwd(),
        cfg.algo.algo_log_name,
        cfg.training.task_name,
        selected_load_run,
    )
    if not load_path or not os.path.exists(load_path):
        print(f"Could not find checkpoint. load_path={load_path}")
        return None

    dp_rank = current_dp_rank()
    device = default_device(torch, resolve_dp_rank_device(dp_rank))
    # Genesis pins CUDA_VISIBLE_DEVICES for a non-zero request; adopt the bound
    # in-process device for the policy and the env override.  CUDA-only
    # backends also reject a CPU/MPS learner here, before playback constructs.
    bound_device = configure_backend_process_device(str(cfg.training.sim_backend), device)
    if bound_device is not None:
        device = bound_device
    play_env_cfg_override = apply_backend_env_device_override(
        build_offpolicy_play_env_cfg_override(algo_name, cfg),
        str(cfg.training.sim_backend),
        learner_device=device,
    )
    print(f"Using device for play: {device}")

    playback_cfg = RslRlPlaybackConfig(
        task=str(cfg.training.task_name),
        load_run=selected_load_run,
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
            env_cfg_override=play_env_cfg_override,
        ),
        root_dir=Path.cwd(),
        device=device,
        algo_name=algo_name,
    )
    env = cast(Any, session.env)
    actor = session.actor
    normalizer = session.normalizer

    # Export actor to ONNX
    if load_path_dir is not None and bool(getattr(cfg.training, "export_onnx", True)):
        with nonfatal_play_step("ONNX export"):
            obs_dim, _ = resolve_play_obs_dims(env.obs_groups_spec)
            onnx_path = os.path.join(load_path_dir, "policy.onnx")
            dummy_input = torch.randn(1, obs_dim, device=device)
            with torch.inference_mode():
                if normalizer:
                    dummy_input = normalizer(dummy_input, update=False)
                assert actor is not None
                if algo_name in ("sac", "flashsac"):
                    export_module = actor.as_export_module()
                else:
                    export_module = actor
                export_inputs = (dummy_input,)
            input_names = ["obs"]
            export_policy_onnx(export_module, onnx_path, export_inputs, input_names=input_names)

            # Verify ONNX output matches PyTorch
            verify_input = torch.randn(1, obs_dim, device=device)
            with torch.inference_mode():
                onnx_feed = normalizer(verify_input, update=False) if normalizer else verify_input
            verify_inputs = (onnx_feed,)
            verify_policy_onnx(export_module, onnx_path, verify_inputs, input_names=input_names)
    elif load_path_dir is not None:
        print("Skipping ONNX export because training.export_onnx=false.")

    if is_viser_play_render_mode(getattr(cfg.training, "play_render_mode", "auto")):
        # Browser-based viser playback renders through the shared MuJoCo
        # playback shell; it replaces backend-native playback and records no
        # video. Blocks until Ctrl+C. Imported lazily: the viser scene module
        # binds the optional `mujoco` package, which non-MuJoCo-shell training
        # environments do not install.
        from unilab.visualization.viser_playback import run_viser_playback_from_cfg

        run_viser_playback_from_cfg(session, cfg, entrypoint=f"train_{algo_name} play")
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
                initialize=session.reset,
                step=lambda _obs: session.step_once(),
                camera_kwargs=camera_cfg_from_training(cfg.training),
                on_plan=log_playback_plan,
            )
    if play_video_path is not None:
        print(f"Saving video to {play_video_path} ...")
    print("Done.")
    return cast("str | None", play_video_path)


def main(cfg: DictConfig) -> None:
    enable_faulthandler()
    import torch

    reject_removed_device_config(OmegaConf.select(cfg, "training.devices", default=None))
    rank = current_dp_rank()
    world_size = current_dp_world_size()
    if world_size == 1 and rank == 0:
        world_size = _requested_dp_world_size()
    if world_size > 1:
        if rank == 0:
            selected_visible_entries(world_size=world_size)
        _export_inferred_dp_world_size(world_size)
    _prepare_rank_zero_dp_visibility(world_size)
    # Genesis/Quadrants binds the first CUDA_VISIBLE_DEVICES entry, and even
    # torch.cuda.is_available() latches the variable in the CUDA runtime, so
    # the pin must precede registry bootstrap and device auto-detection
    # (issue #1508).  Pure config topology resolves without touching torch.
    pinned_device = pin_genesis_device_before_cuda_init(
        str(cfg.training.sim_backend),
        learner_device=OmegaConf.select(cfg, "training.device", default=None),
    )

    ensure_registries()

    rank_device = apply_dp_rank_config(cfg, rank)
    if pinned_device is not None:
        # The process was pinned to its rank GPU; use the in-process index.
        rank_device = pinned_device

    # Bind before seed initialization and before any rank-local env/probe is
    # materialized.  ``build_runner`` repeats the binding defensively because
    # it is also a public assembly seam used by tests and custom callers.  A
    # non-zero Genesis request pins CUDA_VISIBLE_DEVICES here; the bound
    # in-process device replaces rank_device for the tracker and runner.
    bound_device = configure_backend_process_device(str(cfg.training.sim_backend), rank_device)
    if bound_device is not None:
        rank_device = bound_device

    seed_info = apply_configured_training_seed(cfg, torch_runtime=True, cuda=True)
    algo_name = cfg.algo.algo
    task_name = cfg.training.task_name
    assert_offpolicy_task_choice_matches_algo(cfg, algo_name=algo_name)

    if cfg.training.log_dir is None:
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        run_dir_name = build_run_dir_name(
            timestamp,
            str(cfg.training.sim_backend),
            world_size=world_size,
        )
        log_dir = str(get_log_root(Path.cwd(), cfg) / task_name / run_dir_name)
    else:
        log_dir = cfg.training.log_dir
    if rank > 0:
        # Spawned ranks reuse the canonical run directory but never create
        # logging backends, checkpoints, summaries, or traces there.
        log_dir = os.environ[UNILAB_DP_LOG_DIR]

    supervisor: DpRankSupervisor | None = None
    if rank == 0 and world_size > 1:
        supervisor = _build_dp_rank_supervisor(world_size=world_size, log_dir=log_dir)

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
                        resume_checkpoint=resolve_train_resume_checkpoint(cfg),
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
                play_video_path = play_offpolicy(
                    algo_name,
                    cfg,
                    load_run=None if cfg.training.play_only else str(Path(log_dir).resolve()),
                )
                if tracker is not None:
                    tracker.log_video(play_video_path)
    finally:
        if tracker is not None:
            tracker.finish()


if __name__ == "__main__":
    raise SystemExit(
        "unilab/scripts/train_offpolicy.py is a shared implementation module and is no "
        "longer runnable directly. Use unilab/scripts/train_sac.py or "
        "unilab/scripts/train_flashsac.py instead."
    )
