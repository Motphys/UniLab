"""Train APPO agent — native multiprocessing."""

from __future__ import annotations

import datetime
import os
import sys
from pathlib import Path
from typing import Any, cast

import hydra
import torch
from omegaconf import DictConfig, OmegaConf

ROOT_DIR = Path(__file__).parent.parent
sys.path.append(str(ROOT_DIR))

from unilab.training import (
    BackendAdapter,
    create_env,
    ensure_registries,
    get_log_root,
    render_play_mode,
)
from unilab.utils.experiment_tracking import ExperimentTracker


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
    }
    if cfg.training.replay_queue_size is not None:
        runner_kwargs["replay_queue_size"] = cfg.training.replay_queue_size
    return runner_kwargs


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
        render_play_mode(
            env,
            sim_backend="motrix",
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


def _is_hora_appo_config(rl_cfg: dict[str, Any]) -> bool:
    actor_cfg = rl_cfg.get("actor", {})
    if not isinstance(actor_cfg, dict):
        return False
    return "HoraActorModel" in str(actor_cfg.get("class_name", ""))


def play_appo(cfg: DictConfig, rl_cfg: dict[str, Any]) -> str | None:
    """Play mode for APPO."""
    import numpy as np
    from rsl_rl.utils import resolve_callable
    from tensordict import TensorDict

    from unilab.utils.rsl_rl_compat import convert_config_v3_to_v4, is_rsl_rl_v4, is_rsl_rl_v5

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
    from unilab.utils.obs_utils import get_obs_dims

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

    if is_rsl_rl_v5():
        pass
    elif is_rsl_rl_v4():
        rl_cfg_dict = convert_config_v3_to_v4(rl_cfg_dict)

    from copy import deepcopy

    obs_example = torch.zeros((cfg.training.play_env_num, obs_dim), device=device)
    td_example = TensorDict({"policy": obs_example}, batch_size=cfg.training.play_env_num)

    actor_cfg = deepcopy(rl_cfg_dict["actor"])
    actor_cls = resolve_callable(actor_cfg.pop("class_name"))
    actor_cfg.pop("num_actions", None)
    actor = actor_cls(td_example, rl_cfg_dict["obs_groups"], "actor", action_dim, **actor_cfg)
    actor = actor.to(device)
    actor.eval()

    log_root = _get_log_root(cfg)
    base_log_dir = os.path.join(log_root, cfg.training.task_name)
    load_path, load_path_dir = resolve_appo_checkpoint_path(base_log_dir, cfg.algo.load_run)

    if not load_path or not os.path.exists(load_path):
        print(f"Could not find run to load. load_path={load_path}")
        return None

    print(f"Loading model: {load_path}")
    checkpoint = torch.load(load_path, map_location=device, weights_only=True)
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

    if cfg.training.sim_backend == "motrix":
        print("Starting interactive visualization (motrix native renderer)...")
        print("Close the render window to exit.")
        try:
            run_motrix_play_loop(
                env=env,
                actor=actor,
                device=device,
                play_env_num=cfg.training.play_env_num,
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

    output_video = os.path.join(load_path_dir, "play_video.mp4")
    print(f"Rendering video to {output_video}...")

    if env.state is None:
        env.init_state()
    num_steps = int(getattr(cfg.training, "play_steps", 1000))

    print("Collecting physics states...")
    with torch.inference_mode():
        render_play_mode(
            env,
            sim_backend=cfg.training.sim_backend,
            render_spacing=float(
                getattr(cfg.training, "render_spacing", getattr(env.cfg, "render_spacing", 1.0))
            ),
            num_steps=num_steps,
            output_video=output_video,
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
        )
    print(f"Saving video to {output_video} with mediapy...")
    print("Done.")
    return output_video


@hydra.main(version_base="1.3", config_path="../conf/appo", config_name="config")
def main(cfg: DictConfig) -> None:
    ensure_registries()

    env_cfg_override = BackendAdapter(
        cfg, root_dir=ROOT_DIR, algo_name="appo"
    ).build_task_env_cfg_override()

    # Convert algo config to plain dict for APPORunner / RSL-RL internals
    rl_cfg_raw = OmegaConf.to_container(cfg.algo, resolve=True)
    if not isinstance(rl_cfg_raw, dict):
        raise TypeError("cfg.algo must resolve to a dict")
    rl_cfg = cast(dict[str, Any], rl_cfg_raw)

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
        )
        tracker.start()

    try:
        if not cfg.training.play_only:
            if _is_hora_appo_config(rl_cfg):
                from unilab.algos.torch.hora import HoraAPPORunner

                runner_cls = HoraAPPORunner
            else:
                from unilab.algos.torch.appo.runner import APPORunner

                runner_cls = APPORunner

            runner = runner_cls(
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

        if cfg.training.play_only or not cfg.training.no_play:
            if _is_hora_appo_config(rl_cfg):
                from unilab.algos.torch.hora import play_hora_appo

                play_video_path = play_hora_appo(
                    cfg,
                    rl_cfg,
                    root_dir=ROOT_DIR,
                    resolve_checkpoint_path=lambda current_cfg: resolve_appo_checkpoint_path(
                        os.path.join(_get_log_root(current_cfg), current_cfg.training.task_name),
                        current_cfg.algo.load_run,
                    ),
                )
            else:
                play_video_path = play_appo(cfg, rl_cfg)
            if tracker is not None:
                tracker.log_video(play_video_path)
    finally:
        if tracker is not None:
            tracker.finish()


if __name__ == "__main__":
    main()
