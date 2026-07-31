import datetime
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any, cast

import hydra
import torch
from omegaconf import DictConfig, OmegaConf

ROOT_DIR = Path(__file__).parent.parent
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from unilab.algos.torch.rsl_rl_runtime import (
    RslRlPPORuntime,
    resolve_rsl_rl_ppo_runtime,
    validate_rsl_rl_ppo_runtime_owner,
)
from unilab.base.backend.mujoco.xml import materialize_scene_visual_override
from unilab.training import (
    BackendAdapter,
    EntrypointContractError,
    EntrypointRoute,
    apply_configured_training_seed,
    build_entrypoint_receipt,
    create_env,
    ensure_registries,
    get_latest_checkpoint,
    get_latest_run,
    get_log_root,
    guarded_policy_load,
    log_playback_plan,
    parse_checkpoint_path,
    policy_load_target,
    preflight_policy_source,
    require_entrypoint_route,
    require_policy_load_contracts,
    resolve_entrypoint_contract,
    resolve_ppo_operation,
    should_run_playback,
)
from unilab.training.experiment import (
    ExperimentTracker,
    patch_rsl_rl_resume_state,
    patch_rsl_rl_wandb_writer,
)
from unilab.training.rsl_rl import (
    RslRlVecEnvWrapper,
    normalize_ppo_train_cfg,
)
from unilab.training.rsl_rl import (
    validate_rsl_rl_checkpoint as _validate_rsl_rl_checkpoint,
)
from unilab.training.rsl_rl_device import (
    DeviceRslRlVecEnvWrapper,
    build_device_rsl_rl_run_summary_diagnostics,
)
from unilab.utils.device import get_default_device

try:
    from rsl_rl.runners import OnPolicyRunner
except ImportError:
    print("Could not import rsl_rl. Please ensure it is installed.")
    sys.exit(1)


EXPORT_POLICY = False


def _patch_runner_action_std_logging(runner: Any) -> None:
    original_log = runner.logger.log

    def _safe_log(self, *args, **kwargs):
        policy = runner.alg.get_policy()
        dist = policy.distribution
        if dist.std_type == "scalar":
            std = dist.std_param
        else:
            std = torch.exp(dist.log_std_param)
        kwargs["action_std"] = std.detach().clone()
        return original_log(*args, **kwargs)

    runner.logger.log = _safe_log.__get__(runner.logger, type(runner.logger))


def _backend_adapter(cfg: DictConfig) -> BackendAdapter:
    return BackendAdapter(
        cfg,
        root_dir=ROOT_DIR,
        algo_name="ppo",
        scene_materializer=materialize_scene_visual_override,
    )


def build_ppo_env_cfg_override(cfg: DictConfig) -> dict[str, Any]:
    return cast(dict[str, Any], _backend_adapter(cfg).build_task_env_cfg_override())


def build_ppo_play_env_cfg_override(cfg: DictConfig) -> dict[str, Any]:
    return cast(dict[str, Any], _backend_adapter(cfg).build_play_env_cfg_override())


def run_motrix_rsl_play_loop(
    wrapped_env,
    policy,
    *,
    render_spacing: float,
    render_offset_mode: str,
    num_steps: int | None = None,
) -> None:
    env = wrapped_env.env

    with torch.inference_mode():
        env.run_playback(
            render_spacing=render_spacing,
            render_offset_mode=render_offset_mode,
            num_steps=num_steps,
            initialize=lambda: wrapped_env.reset()[0],
            step=lambda obs: wrapped_env.step(policy(obs))[0],
        )


def _get_log_root(cfg: DictConfig) -> str:
    return str(get_log_root(ROOT_DIR, cfg))


def _algo_config_dict(cfg: DictConfig) -> dict[str, Any]:
    train_cfg_raw = OmegaConf.to_container(cfg.algo, resolve=True)
    if not isinstance(train_cfg_raw, dict):
        raise TypeError("cfg.algo must resolve to a dict")
    return cast(dict[str, Any], train_cfg_raw)


def _resolve_ppo_runtime(rl_cfg: dict[str, Any]) -> RslRlPPORuntime:
    """Resolve wrapper, runner, and cold-path arguments from owner config."""

    return resolve_rsl_rl_ppo_runtime(
        rl_cfg,
        default_wrapper_cls=RslRlVecEnvWrapper,
        default_runner_cls=OnPolicyRunner,
    )


def _validate_ppo_runtime_owner(cfg: DictConfig, runtime: RslRlPPORuntime) -> None:
    """Validate resolver-declared backend/profile constraints before env creation."""

    execution_profile = OmegaConf.select(cfg, "training.execution_profile", default=None)
    validate_rsl_rl_ppo_runtime_owner(
        runtime,
        sim_backend=str(cfg.training.sim_backend),
        execution_profile=None if execution_profile is None else str(execution_profile),
    )


def _peak_process_rss_bytes() -> int | None:
    try:
        import resource

        value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    except (ImportError, OSError, ValueError):
        return None
    return value if sys.platform == "darwin" else value * 1024


def _peak_cuda_memory_bytes(device: str) -> tuple[int | None, int | None]:
    resolved = torch.device(device)
    if resolved.type != "cuda" or not torch.cuda.is_available():
        return None, None
    return (
        int(torch.cuda.max_memory_allocated(resolved)),
        int(torch.cuda.max_memory_reserved(resolved)),
    )


def apply_ppo_runtime_flags(
    train_cfg: dict[str, Any],
    cfg: DictConfig,
    *,
    training_enabled: bool,
) -> None:
    algorithm_cfg = train_cfg.setdefault("algorithm", {})
    if not isinstance(algorithm_cfg, dict):
        return
    if not training_enabled:
        algorithm_cfg["enable_compile"] = False


def _format_play_checkpoint_error(
    cfg: DictConfig,
    *,
    task_log_root: Path,
    load_path: Path | None,
    load_path_dir: Path | None,
) -> str:
    selected_checkpoint = OmegaConf.select(cfg, "algo.checkpoint", default=-1)
    checkpoint_hint = (
        f" algo.checkpoint={selected_checkpoint!r}"
        if selected_checkpoint not in (None, "", -1, "-1")
        else ""
    )

    if load_path_dir is not None and load_path is None and checkpoint_hint:
        reason = f"Requested checkpoint was not found under resolved_run={load_path_dir}."
    elif not task_log_root.exists():
        reason = "Task log root does not exist."
    else:
        latest_run = get_latest_run(task_log_root)
        if latest_run is None:
            reason = "No run directories were found under the task log root."
        elif get_latest_checkpoint(latest_run) is None:
            reason = f"Resolved latest run has no model_*.pt checkpoint files: {latest_run}."
        else:
            reason = "Requested run or checkpoint could not be resolved."

    return (
        "Could not resolve a checkpoint for play mode. "
        f"{reason} task={cfg.training.task_name} task_log_root={task_log_root} "
        f"algo.load_run={cfg.algo.load_run!r}{checkpoint_hint}."
        " Use algo.load_run=<run-dir-or-checkpoint-path> "
        "and optionally algo.checkpoint=<iteration-or-filename>."
    )


def _resolve_play_num_steps(cfg: DictConfig) -> int | None:
    play_steps = OmegaConf.select(cfg, "training.play_steps", default=None)
    if play_steps is None:
        return None
    return int(play_steps)


def _sim2sim_strict(cfg: DictConfig) -> bool:
    return bool(OmegaConf.select(cfg, "training.sim2sim_strict", default=True))


def _resolve_required_checkpoint(
    cfg: DictConfig,
    *,
    operation: str,
) -> tuple[Path, Path]:
    load_path, load_path_dir = parse_checkpoint_path(cfg, root_dir=ROOT_DIR)
    if load_path is None or load_path_dir is None or not load_path.exists():
        task_log_root = get_log_root(ROOT_DIR, cfg) / str(cfg.training.task_name)
        detail = _format_play_checkpoint_error(
            cfg,
            task_log_root=task_log_root,
            load_path=load_path,
            load_path_dir=load_path_dir,
        ).replace("play mode", f"{operation} mode")
        raise EntrypointContractError(detail)
    return load_path, load_path_dir


def _policy_load_target_from_wrapper(wrapped_env: Any):
    managed_policy_abi = getattr(wrapped_env, "managed_policy_abi_snapshot", None)
    observation_dim = getattr(wrapped_env, "num_obs", None)
    action_dim = getattr(wrapped_env, "num_actions", None)
    return policy_load_target(
        managed_policy_abi=managed_policy_abi,
        observation_dim=None if observation_dim is None else int(observation_dim),
        action_dim=None if action_dim is None else int(action_dim),
    )


def _load_runner_checkpoint(
    *,
    runner: Any,
    load_path: Path,
    load_path_dir: Path,
    cfg: DictConfig,
    contract: Any,
    wrapped_env: Any,
    device: str,
    load_cfg: dict[str, bool] | None = None,
) -> Any:
    target = _policy_load_target_from_wrapper(wrapped_env)
    with guarded_policy_load(
        contract=contract,
        source_run_dir=load_path_dir,
        target_cfg=cfg,
        target=target,
        algo_name="ppo",
        strict=_sim2sim_strict(cfg),
    ):
        kwargs: dict[str, Any] = {"map_location": device}
        if load_cfg is not None:
            kwargs["load_cfg"] = load_cfg
        runner.load(str(load_path), **kwargs)
    return target


def _export_runner_policy(
    *,
    runner: Any,
    output_dir: Path,
    formats: tuple[str, ...],
) -> tuple[Path, ...]:
    outputs: list[Path] = []
    if "onnx" in formats:
        runner.export_policy_to_onnx(path=str(output_dir))
        outputs.append(output_dir / "policy.onnx")
    if "jit" in formats:
        runner.export_policy_to_jit(path=str(output_dir))
        outputs.append(output_dir / "policy.pt")
    missing = [path for path in outputs if not path.is_file()]
    if missing:
        raise EntrypointContractError(
            "Policy export returned without producing required artifacts: "
            + ", ".join(str(path) for path in missing)
        )
    return tuple(outputs)


def _write_entrypoint_receipt(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(path)


def _require_checkpoint_artifact(path: Path) -> Path:
    if not path.is_file() or path.stat().st_size <= 0:
        raise EntrypointContractError(
            f"Checkpoint-save route returned without producing a non-empty artifact: {path}"
        )
    return path


def play_rsl_rl(cfg: DictConfig, device: str) -> str | None:
    """Play mode for RSL-RL."""
    contract, checkpoint_contract = require_policy_load_contracts(cfg, EntrypointRoute.PLAY)
    load_path, load_path_dir = _resolve_required_checkpoint(cfg, operation="play")
    cfg = preflight_policy_source(
        source_run_dir=load_path_dir,
        target_cfg=cfg,
        algo_name="ppo",
        strict=_sim2sim_strict(cfg),
    )
    _validate_rsl_rl_checkpoint(load_path)
    print(f"Loading latest model: {load_path}")
    contract, checkpoint_contract = require_policy_load_contracts(cfg, EntrypointRoute.PLAY)
    rl_cfg = _algo_config_dict(cfg)
    ppo_runtime = _resolve_ppo_runtime(rl_cfg)
    _validate_ppo_runtime_owner(cfg, ppo_runtime)
    env_cfg_override = build_ppo_play_env_cfg_override(cfg)
    env = create_env(
        cfg,
        num_envs=cfg.training.play_env_num,
        env_cfg_override=env_cfg_override,
    )
    try:
        wrapped_env = ppo_runtime.wrapper_cls(
            env,
            device=device,
            **ppo_runtime.wrapper_kwargs,
        )
        train_cfg = normalize_ppo_train_cfg(rl_cfg)
        apply_ppo_runtime_flags(train_cfg, cfg, training_enabled=False)
        if "runner" not in train_cfg:
            train_cfg["runner"] = {}
        train_cfg["runner"]["logger"] = "none"

        runner = cast(
            Any,
            ppo_runtime.runner_cls(
                cast(Any, wrapped_env),
                train_cfg,
                log_dir=None,
                device=device,
            ),
        )
        target = _load_runner_checkpoint(
            runner=runner,
            load_path=load_path,
            load_path_dir=load_path_dir,
            cfg=cfg,
            contract=checkpoint_contract,
            wrapped_env=wrapped_env,
            device=device,
        )
        policy = runner.get_inference_policy(device=device)
        exported: tuple[Path, ...] = ()
        if EXPORT_POLICY:
            export_contract = require_entrypoint_route(
                resolve_entrypoint_contract(cfg, EntrypointRoute.EXPORT)
            )
            exported = _export_runner_policy(
                runner=runner,
                output_dir=load_path_dir,
                formats=export_contract.export_formats,
            )
        num_steps = _resolve_play_num_steps(cfg)
        output_video = load_path_dir / "play_video.mp4"
        playback_mode: str | None = None

        def _log_plan(plan) -> None:
            nonlocal playback_mode
            playback_mode = plan.mode
            log_playback_plan(plan)

        try:
            with torch.inference_mode():
                play_video_path = env.run_playback_mode(
                    play_render_mode=getattr(cfg.training, "play_render_mode", "auto"),
                    play_steps=num_steps,
                    output_video=output_video,
                    render_spacing=float(
                        getattr(
                            cfg.training,
                            "render_spacing",
                            getattr(env.cfg, "render_spacing", 1.0),
                        )
                    ),
                    render_offset_mode=str(getattr(env.cfg, "render_offset_mode", "grid")),
                    initialize=lambda: wrapped_env.reset()[0],
                    step=lambda obs: wrapped_env.step(policy(obs))[0],
                    camera_kwargs={
                        "cam_distance": cfg.training.cam_distance,
                        "cam_elevation": cfg.training.cam_elevation,
                        "cam_azimuth": cfg.training.cam_azimuth,
                        "cam_lookat": getattr(cfg.training, "cam_lookat", None),
                        "cam_tracking": getattr(cfg.training, "cam_tracking", False),
                        "cam_tracking_env_idx": getattr(cfg.training, "cam_tracking_env_idx", 0),
                        "cam_tracking_extra_envs": getattr(
                            cfg.training, "cam_tracking_extra_envs", 2
                        ),
                    },
                    on_plan=_log_plan,
                    extra_data_getter=(
                        (lambda: getattr(env, "curr_ee_goal_world", None))
                        if hasattr(env, "curr_ee_goal_world")
                        else None
                    ),
                )
        except Exception as exc:
            if cfg.training.sim_backend == "motrix" and "RenderClosedError" in str(
                type(exc).__name__
            ):
                print("Render window closed.")
                play_video_path = None
            else:
                raise
        receipt_outputs = (*exported, *((output_video,) if play_video_path else ()))
        _write_entrypoint_receipt(
            load_path_dir / "entrypoint_play_receipt.json",
            {
                **build_entrypoint_receipt(
                    contract,
                    checkpoint=load_path,
                    target=target,
                    outputs=receipt_outputs,
                ),
                "checkpoint_load_contract": checkpoint_contract.to_snapshot(),
            },
        )
        if playback_mode != "none" and num_steps is not None:
            print("Done.")
        return play_video_path
    finally:
        env.close()


def export_rsl_rl(cfg: DictConfig, device: str) -> tuple[Path, ...]:
    """Load and export an RSL-RL policy without entering a renderer."""

    contract, checkpoint_contract = require_policy_load_contracts(cfg, EntrypointRoute.EXPORT)
    load_path, load_path_dir = _resolve_required_checkpoint(cfg, operation="export")
    cfg = preflight_policy_source(
        source_run_dir=load_path_dir,
        target_cfg=cfg,
        algo_name="ppo",
        strict=_sim2sim_strict(cfg),
    )
    _validate_rsl_rl_checkpoint(load_path)
    contract, checkpoint_contract = require_policy_load_contracts(cfg, EntrypointRoute.EXPORT)
    rl_cfg = _algo_config_dict(cfg)
    ppo_runtime = _resolve_ppo_runtime(rl_cfg)
    _validate_ppo_runtime_owner(cfg, ppo_runtime)
    env = create_env(
        cfg,
        num_envs=int(cfg.training.play_env_num),
        env_cfg_override=build_ppo_play_env_cfg_override(cfg),
    )
    try:
        wrapped_env = ppo_runtime.wrapper_cls(
            env,
            device=device,
            **ppo_runtime.wrapper_kwargs,
        )
        train_cfg = normalize_ppo_train_cfg(rl_cfg)
        apply_ppo_runtime_flags(train_cfg, cfg, training_enabled=False)
        train_cfg.setdefault("runner", {})["logger"] = "none"
        runner = cast(
            Any,
            ppo_runtime.runner_cls(
                cast(Any, wrapped_env),
                train_cfg,
                log_dir=None,
                device=device,
            ),
        )
        target = _load_runner_checkpoint(
            runner=runner,
            load_path=load_path,
            load_path_dir=load_path_dir,
            cfg=cfg,
            contract=checkpoint_contract,
            wrapped_env=wrapped_env,
            device=device,
        )
        outputs = _export_runner_policy(
            runner=runner,
            output_dir=load_path_dir,
            formats=contract.export_formats,
        )
        _write_entrypoint_receipt(
            load_path_dir / "entrypoint_export_receipt.json",
            {
                **build_entrypoint_receipt(
                    contract,
                    checkpoint=load_path,
                    target=target,
                    outputs=outputs,
                ),
                "checkpoint_load_contract": checkpoint_contract.to_snapshot(),
            },
        )
        print("Exported policy artifacts: " + ", ".join(str(path) for path in outputs))
        return outputs
    finally:
        env.close()


@hydra.main(version_base="1.3", config_path="../conf/ppo", config_name="config")
def main(cfg: DictConfig) -> None:
    ensure_registries()
    operation = resolve_ppo_operation(cfg)
    route = (
        EntrypointRoute.RESUME
        if operation is EntrypointRoute.TRAIN and str(cfg.algo.load_run) != "-1"
        else operation
    )
    route_contract = require_entrypoint_route(resolve_entrypoint_contract(cfg, route))
    checkpoint_load_contract = None
    if route is EntrypointRoute.RESUME:
        route_contract, checkpoint_load_contract = require_policy_load_contracts(cfg, route)
    seed_info = apply_configured_training_seed(cfg, torch_runtime=True, cuda=True)
    device = get_default_device()
    print(f"Using device: {device}")

    if operation is EntrypointRoute.PLAY:
        play_rsl_rl(cfg, device)
        return
    if operation is EntrypointRoute.EXPORT:
        export_rsl_rl(cfg, device)
        return

    resume_path: Path | None = None
    resume_dir: Path | None = None
    if route is EntrypointRoute.RESUME:
        resume_path, resume_dir = _resolve_required_checkpoint(cfg, operation="resume")
        cfg = preflight_policy_source(
            source_run_dir=resume_dir,
            target_cfg=cfg,
            algo_name="ppo",
            strict=_sim2sim_strict(cfg),
        )
        _validate_rsl_rl_checkpoint(resume_path)
        route_contract, checkpoint_load_contract = require_policy_load_contracts(cfg, route)
    checkpoint_save_contract = require_entrypoint_route(
        resolve_entrypoint_contract(cfg, EntrypointRoute.CHECKPOINT_SAVE)
    )
    rl_cfg = _algo_config_dict(cfg)
    ppo_runtime = _resolve_ppo_runtime(rl_cfg)
    _validate_ppo_runtime_owner(cfg, ppo_runtime)
    env_cfg_override = build_ppo_env_cfg_override(cfg)

    # Compute effective max_iterations (supports num_timesteps override)
    max_iterations = cfg.algo.max_iterations
    if cfg.training.num_timesteps:
        n_steps_per_iter = cfg.algo.num_steps_per_env * cfg.algo.num_envs
        max_iterations = max(1, int(cfg.training.num_timesteps / n_steps_per_iter))
        print(
            f"Overriding max_iterations to {max_iterations} based on "
            f"num_timesteps {cfg.training.num_timesteps}"
        )

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_root = _get_log_root(cfg)
    log_dir = str(
        Path(log_root) / cfg.training.task_name / f"{timestamp}_{cfg.training.sim_backend}"
    )

    tracker = ExperimentTracker(
        root_dir=ROOT_DIR,
        log_dir=log_dir,
        algo_name="ppo",
        task_name=cfg.training.task_name,
        sim_backend=cfg.training.sim_backend,
        training_cfg=cfg.training,
        full_cfg=cfg,
        device=device,
        seed_info=seed_info,
    )

    env: Any | None = None
    try:
        if operation is EntrypointRoute.TRAIN:
            env = create_env(
                cfg,
                num_envs=cfg.algo.num_envs,
                env_cfg_override=env_cfg_override,
            )

            nan_guard_cfg = getattr(cfg.training, "nan_guard", None)
            if nan_guard_cfg is not None and getattr(nan_guard_cfg, "enabled", False):
                from unilab.utils.nan_guard import NanGuard, NanGuardCfg

                guard = NanGuard(
                    NanGuardCfg(
                        enabled=True,
                        buffer_size=int(getattr(nan_guard_cfg, "buffer_size", 100)),
                        max_envs_to_dump=int(getattr(nan_guard_cfg, "max_envs_to_dump", 5)),
                        output_dir=getattr(nan_guard_cfg, "output_dir", None),
                    ),
                    num_envs=env.num_envs,
                    supports_state_playback=env.play_capabilities.supports_physics_state_playback,
                )
                env.set_nan_guard(guard)

            wrapped_env = ppo_runtime.wrapper_cls(
                env,
                device=device,
                **ppo_runtime.wrapper_kwargs,
            )
            policy_target = _policy_load_target_from_wrapper(wrapped_env)

            tracker.set_managed_policy_abi(policy_target.managed_policy_abi)
            tracker.start()

            train_cfg = normalize_ppo_train_cfg(rl_cfg)
            apply_ppo_runtime_flags(train_cfg, cfg, training_enabled=True)
            if "runner" not in train_cfg:
                train_cfg["runner"] = {}

            logger_type = (
                cfg.training.logger if cfg.training.logger in ["tensorboard", "wandb"] else "none"
            )
            train_cfg["runner"]["logger"] = logger_type
            train_cfg["logger"] = logger_type

            patch_rsl_rl_resume_state()

            if logger_type == "wandb":
                patch_rsl_rl_wandb_writer()
                wandb_settings = tracker.wandb_settings
                train_cfg["wandb_project"] = wandb_settings["project"]
                train_cfg["wandb_entity"] = wandb_settings["entity"]
                train_cfg["wandb_group"] = wandb_settings["group"]
                train_cfg["wandb_job_type"] = wandb_settings["job_type"]
                train_cfg["wandb_tags"] = wandb_settings["tags"]
                train_cfg["wandb_notes"] = wandb_settings["notes"]
                train_cfg["wandb_mode"] = wandb_settings["mode"]

            runner = cast(
                Any,
                ppo_runtime.runner_cls(
                    cast(Any, wrapped_env),
                    train_cfg,
                    log_dir=log_dir,
                    device=device,
                ),
            )
            _patch_runner_action_std_logging(runner)

            if resume_path is not None and resume_dir is not None:
                assert checkpoint_load_contract is not None
                print(f"Resuming from {resume_path}")
                policy_target = _load_runner_checkpoint(
                    runner=runner,
                    load_path=resume_path,
                    load_path_dir=resume_dir,
                    cfg=cfg,
                    contract=checkpoint_load_contract,
                    wrapped_env=wrapped_env,
                    device=device,
                )

            train_start_wall = time.time()
            runner.learn(num_learning_iterations=max_iterations, init_at_random_ep_len=True)
            assert log_dir is not None
            peak_gpu_allocated, peak_gpu_reserved = _peak_cuda_memory_bytes(device)
            last_checkpoint = _require_checkpoint_artifact(
                Path(log_dir) / f"model_{int(runner.current_learning_iteration)}.pt"
            )
            train_summary = {
                "status": "completed",
                "completed_iterations": int(runner.current_learning_iteration) + 1,
                "total_env_steps": int(getattr(runner.logger, "tot_timesteps", 0)),
                "final_mean_reward": (
                    float(statistics.mean(runner.logger.rewbuffer))
                    if len(getattr(runner.logger, "rewbuffer", [])) > 0
                    else None
                ),
                "best_mean_reward": (
                    float(max(runner.logger.rewbuffer))
                    if len(getattr(runner.logger, "rewbuffer", [])) > 0
                    else None
                ),
                "mean_episode_length": (
                    float(statistics.mean(runner.logger.lenbuffer))
                    if len(getattr(runner.logger, "lenbuffer", [])) > 0
                    else None
                ),
                "last_checkpoint": str(last_checkpoint),
                "entrypoint_receipt": {
                    **build_entrypoint_receipt(
                        route_contract,
                        checkpoint=resume_path,
                        target=policy_target,
                        outputs=(last_checkpoint,),
                    ),
                    "checkpoint_save_contract": checkpoint_save_contract.to_snapshot(),
                    "checkpoint_load_contract": (
                        None
                        if checkpoint_load_contract is None
                        else checkpoint_load_contract.to_snapshot()
                    ),
                },
                "training_wall_time_sec": time.time() - train_start_wall,
                "peak_process_rss_bytes": _peak_process_rss_bytes(),
                "peak_gpu_memory_allocated_bytes": peak_gpu_allocated,
                "peak_gpu_memory_reserved_bytes": peak_gpu_reserved,
            }
            if isinstance(wrapped_env, DeviceRslRlVecEnvWrapper):
                train_summary.update(
                    build_device_rsl_rl_run_summary_diagnostics(wrapped_env, runner)
                )
            tracker.update_summary(train_summary)
            env.close()
            env = None

        if should_run_playback(
            play_only=cfg.training.play_only,
            no_play=cfg.training.no_play,
            play_render_mode=getattr(cfg.training, "play_render_mode", "auto"),
        ):
            play_video_path = play_rsl_rl(cfg, device)
            tracker.log_video(play_video_path)
    finally:
        if env is not None:
            env.close()
        tracker.finish()


if __name__ == "__main__":
    EXPORT_POLICY = True
    main()
