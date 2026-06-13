"""Train APPO agent — native multiprocessing."""

from __future__ import annotations

import datetime
import os
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

import hydra
import torch
from omegaconf import DictConfig, OmegaConf

ROOT_DIR = Path(__file__).parent.parent
sys.path.append(str(ROOT_DIR))

from unilab.algos.torch.appo.runtime import resolve_appo_runtime
from unilab.training import (
    BackendAdapter,
    apply_configured_training_seed,
    create_env,
    ensure_registries,
    get_log_root,
    log_playback_plan,
    should_run_playback,
)
from unilab.training.experiment import ExperimentTracker
from unilab.training.sim2sim import policy_load_dim_guard, resolve_sim2sim_config


def _training_resume_requested(load_run: Any) -> bool:
    if load_run is None:
        return False
    return str(load_run) not in {"", "-1"}


def build_appo_runner_kwargs(
    cfg: DictConfig,
    env_cfg_override: dict | None,
    collector_device: str | None,
    rl_cfg: dict[str, Any] | None = None,
) -> dict:
    if rl_cfg is None:
        rl_cfg_raw = OmegaConf.to_container(cfg.algo, resolve=True)
        if not isinstance(rl_cfg_raw, dict):
            raise TypeError("cfg.algo must resolve to a dict")
        rl_cfg = cast(dict[str, Any], rl_cfg_raw)

    runner_kwargs = {
        "env_name": cfg.training.task_name,
        "env_cfg_overrides": env_cfg_override,
        "rl_cfg": rl_cfg,
        "device": cfg.training.device,
        "collector_device": collector_device,
        "num_envs": cfg.algo.num_envs,
        "steps_per_env": cfg.algo.steps_per_env,
        "sim_backend": cfg.training.sim_backend,
        "seed": rl_cfg.get("seed"),
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

    nan_guard_cfg = getattr(cfg.training, "nan_guard", None)
    if nan_guard_cfg is not None and getattr(nan_guard_cfg, "enabled", False):
        from unilab.utils.nan_guard import NanGuardCfg

        runner_kwargs["nan_guard_cfg"] = NanGuardCfg(
            enabled=True,
            buffer_size=int(getattr(nan_guard_cfg, "buffer_size", 100)),
            max_envs_to_dump=int(getattr(nan_guard_cfg, "max_envs_to_dump", 5)),
            output_dir=getattr(nan_guard_cfg, "output_dir", None),
        )
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
    import numpy as np
    from tensordict import TensorDict

    if env.state is None:
        env.init_state()

    with torch.inference_mode():
        env.run_playback(
            num_steps=num_steps,
            initialize=lambda: np.asarray(
                env.reset(np.arange(play_env_num, dtype=np.int32))[0]["obs"],
                dtype=np.float32,
            ),
            step=lambda obs_np: np.asarray(
                env.step(
                    actor(
                        TensorDict(
                            {"policy": torch.from_numpy(obs_np).to(device)}, batch_size=play_env_num
                        )
                    )
                    .cpu()
                    .numpy()
                    .astype(np.float32)
                ).obs["obs"],
                dtype=np.float32,
            ),
        )


def resolve_appo_checkpoint_path(
    base_log_dir: str | Path,
    load_run: str | int,
) -> tuple[str | None, str | None]:
    from unilab.training import resolve_checkpoint_path

    checkpoint_path, checkpoint_dir = resolve_checkpoint_path(
        base_log_dir,
        str(load_run),
        suffix=".pt",
    )
    return (
        str(checkpoint_path) if checkpoint_path is not None else None,
        str(checkpoint_dir) if checkpoint_dir is not None else None,
    )


def _get_log_root(cfg: DictConfig) -> str:
    return str(get_log_root(ROOT_DIR, cfg))


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
    import numpy as np
    from rsl_rl.utils import resolve_callable
    from tensordict import TensorDict

    if resolve_checkpoint_path is not None:
        load_path, load_path_dir = resolve_checkpoint_path(cfg)
    else:
        log_root = _get_log_root(cfg)
        base_log_dir = os.path.join(log_root, cfg.training.task_name)
        load_path, load_path_dir = resolve_appo_checkpoint_path(base_log_dir, cfg.algo.load_run)

    if not load_path or not os.path.exists(load_path):
        print(f"Could not find run to load. load_path={load_path}")
        return None

    cfg = (
        resolve_sim2sim_config(
            load_path_dir,
            cfg,
            algo_name="appo",
            strict=bool(getattr(cfg.training, "sim2sim_strict", True)),
        )
        or cfg
    )

    env_cfg_override = BackendAdapter(
        cfg, root_dir=ROOT_DIR, algo_name="appo"
    ).build_task_env_cfg_override()

    device = cfg.training.device or (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Using device for play: {device}")

    env = cast(
        Any,
        create_env(
            cfg,
            num_envs=cfg.training.play_env_num,
            env_cfg_override=env_cfg_override,
        ),
    )
    from unilab.base.observations import get_obs_dims

    obs_dim, critic_dim = get_obs_dims(env.obs_groups_spec)
    action_shape = env.action_space.shape
    if action_shape is None:
        raise ValueError("env.action_space.shape must be defined")
    action_dim = int(action_shape[0])

    rl_cfg_dict = dict(rl_cfg)
    if "obs_groups" not in rl_cfg_dict:
        rl_cfg_dict["obs_groups"] = {
            "actor": {"policy": obs_dim},
            "critic": {"policy": critic_dim if critic_dim > 0 else obs_dim},
        }
    else:
        actor_group = rl_cfg_dict["obs_groups"].get(
            "actor", rl_cfg_dict["obs_groups"].get("policy", {})
        )
        if isinstance(actor_group, dict) and "policy" in actor_group:
            actor_group["policy"] = obs_dim
        critic_group = rl_cfg_dict["obs_groups"].get("critic")
        if critic_group is None:
            rl_cfg_dict["obs_groups"]["critic"] = {
                "policy": critic_dim if critic_dim > 0 else obs_dim
            }
        elif isinstance(critic_group, dict) and "policy" in critic_group:
            critic_group["policy"] = critic_dim if critic_dim > 0 else obs_dim

    from copy import deepcopy

    obs_example = torch.zeros((cfg.training.play_env_num, obs_dim), device=device)
    td_example = TensorDict({"policy": obs_example}, batch_size=cfg.training.play_env_num)

    actor_cfg = deepcopy(rl_cfg_dict["actor"])
    actor_cls = resolve_callable(actor_cfg.pop("class_name"))
    actor_cfg.pop("num_actions", None)
    actor = actor_cls(td_example, rl_cfg_dict["obs_groups"], "actor", action_dim, **actor_cfg)
    actor = actor.to(device)
    actor.eval()

    print(f"Loading model: {load_path}")
    checkpoint = torch.load(load_path, map_location=device, weights_only=True)
    with policy_load_dim_guard(env_obs_dim=obs_dim, env_action_dim=action_dim, algo_name="appo"):
        actor.load_state_dict(checkpoint["actor"])

    # Export actor to ONNX
    if load_path_dir is not None:
        import numpy as np
        import torch.nn as nn

        class _DeterministicAPPOActor(nn.Module):
            def __init__(self, mlp: nn.Module):
                super().__init__()
                self.mlp = mlp

            def forward(self, obs: torch.Tensor) -> torch.Tensor:
                return self.mlp(obs)

        export_module = _DeterministicAPPOActor(actor.mlp)
        onnx_path = os.path.join(load_path_dir, "policy.onnx")
        dummy_input = torch.randn(1, obs_dim, device=device)
        with torch.inference_mode():
            torch.onnx.export(
                export_module,
                (dummy_input,),
                onnx_path,
                input_names=["obs"],
                output_names=["action"],
                opset_version=17,
            )
        print(f"Exported actor ONNX to {onnx_path}")

        import onnxruntime as ort

        sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
        verify_input = torch.randn(1, obs_dim, device=device)
        with torch.inference_mode():
            pt_output = export_module(verify_input).cpu().numpy()
        onnx_output = sess.run(None, {"obs": verify_input.cpu().numpy().astype(np.float32)})[0]
        max_diff = np.max(np.abs(pt_output - onnx_output))
        mean_diff = np.mean(np.abs(pt_output - onnx_output))
        print(f"ONNX vs PyTorch — max_diff: {max_diff:.2e}, mean_diff: {mean_diff:.2e}")
        if max_diff > 1e-4:
            print("WARNING: ONNX output diverges from PyTorch!")
        else:
            print("ONNX export verified OK.")

    if env.state is None:
        env.init_state()

    with torch.inference_mode():
        play_video_path = env.run_playback_mode(
            play_render_mode=getattr(cfg.training, "play_render_mode", "auto"),
            play_steps=getattr(cfg.training, "play_steps", None),
            output_video=os.path.join(load_path_dir, "play_video.mp4") if load_path_dir else None,
            render_spacing=float(
                getattr(cfg.training, "render_spacing", getattr(env.cfg, "render_spacing", 1.0))
            ),
            initialize=lambda: np.asarray(
                env.reset(np.arange(cfg.training.play_env_num, dtype=np.int32))[0]["obs"],
                dtype=np.float32,
            ),
            step=lambda obs_np: np.asarray(
                env.step(
                    actor(
                        TensorDict(
                            {"policy": torch.from_numpy(obs_np).to(device)},
                            batch_size=cfg.training.play_env_num,
                        )
                    )
                    .cpu()
                    .numpy()
                    .astype(np.float32)
                ).obs["obs"],
                dtype=np.float32,
            ),
            camera_kwargs={
                "cam_distance": cfg.training.cam_distance,
                "cam_elevation": cfg.training.cam_elevation,
                "cam_azimuth": cfg.training.cam_azimuth,
                "cam_lookat": getattr(cfg.training, "cam_lookat", None),
                "cam_tracking": getattr(cfg.training, "cam_tracking", False),
                "cam_tracking_env_idx": getattr(cfg.training, "cam_tracking_env_idx", 0),
                "cam_tracking_extra_envs": getattr(cfg.training, "cam_tracking_extra_envs", 2),
            },
            on_plan=log_playback_plan,
        )
    if play_video_path is not None:
        print(f"Saving video to {play_video_path} with mediapy...")
    print("Done.")
    return play_video_path


@hydra.main(version_base="1.3", config_path="../conf/appo", config_name="config")
def main(cfg: DictConfig) -> None:
    ensure_registries()

    seed_info = apply_configured_training_seed(cfg, torch_runtime=True, cuda=True)
    env_cfg_override = BackendAdapter(
        cfg, root_dir=ROOT_DIR, algo_name="appo"
    ).build_task_env_cfg_override()

    # Convert algo config to plain dict for APPORunner / RSL-RL internals
    rl_cfg_raw = OmegaConf.to_container(cfg.algo, resolve=True)
    if not isinstance(rl_cfg_raw, dict):
        raise TypeError("cfg.algo must resolve to a dict")
    rl_cfg = cast(dict[str, Any], rl_cfg_raw)
    apply_appo_runtime_flags(rl_cfg, cfg, training_enabled=not cfg.training.play_only)
    appo_runtime = resolve_appo_runtime(rl_cfg, default_play_fn=play_appo)

    if cfg.training.log_dir is None:
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        log_root = _get_log_root(cfg)
        log_dir = os.path.join(
            log_root,
            cfg.training.task_name,
            f"{timestamp}_{cfg.training.sim_backend}",
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

    tracker = None
    if not cfg.training.play_only:
        tracker = ExperimentTracker(
            root_dir=ROOT_DIR,
            log_dir=log_dir,
            algo_name="appo",
            task_name=cfg.training.task_name,
            sim_backend=cfg.training.sim_backend,
            training_cfg=cfg.training,
            full_cfg=cfg,
            device=learner_device,
            collector_device=collector_device,
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
                root_dir=ROOT_DIR,
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
