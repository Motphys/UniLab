"""HORA-owned APPO entry helpers."""

from __future__ import annotations

import os
from collections.abc import Callable
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, cast

import torch
from omegaconf import DictConfig

from unilab.algos.torch.hora.appo_runner import HoraAPPORunner
from unilab.algos.torch.hora.rsl_rl_compat import (
    convert_config_v3_to_v4,
    is_rsl_rl_v4,
    is_rsl_rl_v5,
)
from unilab.base.observations import get_obs_dims
from unilab.training import BackendAdapter, create_env, log_playback_plan

from .models import build_hora_shared_actor_critic
from .observations import build_hora_actor_tensordict, split_hora_obs_with_priv_info
from .runtime import is_hora_appo_runtime


@dataclass(frozen=True)
class HoraAPPORuntime:
    """Resolved HORA APPO entrypoints used by the generic APPO script.

    Args:
        runner_cls: Runner class used for HORA APPO training mode.
        play_fn: Play-mode callable used for HORA APPO checkpoint playback.

    Returns:
        Immutable entrypoint bundle consumed by generic APPO script assembly.
    """

    runner_cls: type[HoraAPPORunner]
    play_fn: Callable[..., str | None]


def resolve_hora_appo_runtime(rl_cfg: dict[str, Any]) -> HoraAPPORuntime | None:
    """Resolve HORA APPO entrypoints from an explicit runtime marker.

    Args:
        rl_cfg: Resolved algorithm config dictionary from Hydra composition.

    Returns:
        ``HoraAPPORuntime`` when the owner config selects HORA APPO, otherwise
        ``None``.
    """
    if not is_hora_appo_runtime(rl_cfg):
        return None
    return HoraAPPORuntime(runner_cls=HoraAPPORunner, play_fn=play_hora_appo)


def _update_hora_obs_groups(
    rl_cfg: dict[str, Any],
    *,
    obs_dim: int,
    priv_info_dim: int,
) -> None:
    """Update grouped actor/critic dims for the HORA APPO runtime.

    Args:
        rl_cfg: Mutable algorithm config dictionary to update in place.
        obs_dim: Actor observation dimension reported by the env contract.
        priv_info_dim: Privileged-info dimension reported by the env contract.

    Returns:
        None. Mutates ``rl_cfg["obs_groups"]`` directly.
    """
    obs_groups = rl_cfg.setdefault("obs_groups", {})
    actor_group = obs_groups.setdefault("actor", {})
    critic_group = obs_groups.setdefault("critic", {})
    if isinstance(actor_group, dict):
        actor_group["actor"] = obs_dim
        actor_group["priv_info"] = priv_info_dim
    if isinstance(critic_group, dict):
        critic_group["actor"] = obs_dim
        critic_group["priv_info"] = priv_info_dim


def play_hora_appo(
    cfg: DictConfig,
    rl_cfg: dict[str, Any],
    *,
    root_dir,
    resolve_checkpoint_path,
) -> str | None:
    """Play HORA APPO checkpoints with grouped actor and privileged inputs."""
    import numpy as np
    from rsl_rl.utils import resolve_callable
    from tensordict import TensorDict

    env_cfg_override = BackendAdapter(
        cfg,
        root_dir=root_dir,
        algo_name="appo",
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
    obs_dim, _ = get_obs_dims(env.obs_groups_spec)
    if env.state is None:
        env.init_state()
    _, _, state_priv_info = split_hora_obs_with_priv_info(
        env.state.obs,
        env.state.info if env.state is not None else None,
    )
    priv_info_dim = int(state_priv_info.shape[1]) if state_priv_info is not None else 0
    if priv_info_dim <= 0:
        raise ValueError("HORA APPO play requires privileged info from the environment.")

    action_shape = env.action_space.shape
    if action_shape is None:
        raise ValueError("env.action_space.shape must be defined")
    action_dim = int(action_shape[0])

    rl_cfg_dict = dict(rl_cfg)
    _update_hora_obs_groups(rl_cfg_dict, obs_dim=obs_dim, priv_info_dim=priv_info_dim)

    if is_rsl_rl_v5():
        pass
    elif is_rsl_rl_v4():
        rl_cfg_dict = convert_config_v3_to_v4(rl_cfg_dict)

    obs_example = torch.zeros((cfg.training.play_env_num, obs_dim), device=device)
    td_example = TensorDict(
        {
            "actor": obs_example,
            "priv_info": torch.zeros((cfg.training.play_env_num, priv_info_dim), device=device),
        },
        batch_size=cfg.training.play_env_num,
    )

    actor_cfg = deepcopy(rl_cfg_dict["actor"])
    actor_cls = resolve_callable(actor_cfg.pop("class_name"))
    actor_cfg.pop("num_actions", None)

    critic_cfg = deepcopy(rl_cfg_dict.get("critic") or rl_cfg_dict.get("actor") or {})
    critic_cfg.pop("class_name", None)
    critic_cfg.pop("num_actions", None)
    critic_cfg.pop("distribution_cfg", None)

    shared_model = build_hora_shared_actor_critic(
        obs_dim=obs_dim,
        action_dim=action_dim,
        priv_info_dim=priv_info_dim,
        actor_cfg=actor_cfg,
        critic_cfg=critic_cfg,
    ).to(device)
    actor = actor_cls(
        td_example,
        rl_cfg_dict["obs_groups"],
        "actor",
        action_dim,
        shared_model=shared_model,
        **actor_cfg,
    )
    actor = actor.to(device)
    actor.eval()

    load_path, load_path_dir = resolve_checkpoint_path(cfg)
    if not load_path or not os.path.exists(load_path):
        print(f"Could not find run to load. load_path={load_path}")
        return None

    print(f"Loading model: {load_path}")
    checkpoint = torch.load(load_path, map_location=device, weights_only=True)
    actor.load_state_dict(checkpoint["actor"])

    current_priv_info: np.ndarray | None = None

    def initialize_play_obs() -> np.ndarray:
        nonlocal current_priv_info
        obs_out, info_out = env.reset(np.arange(cfg.training.play_env_num, dtype=np.int32))
        actor_obs, _, priv_info = split_hora_obs_with_priv_info(obs_out, info_out)
        current_priv_info = priv_info.astype(np.float32) if priv_info is not None else None
        return np.asarray(actor_obs, dtype=np.float32)

    def step_play_obs(obs_np: np.ndarray) -> np.ndarray:
        nonlocal current_priv_info
        if current_priv_info is None:
            raise ValueError("HORA APPO play step is missing privileged info.")
        td = build_hora_actor_tensordict(
            obs_np,
            priv_info=current_priv_info,
            device=device,
            batch_size=cfg.training.play_env_num,
        )
        actions = actor(td).cpu().numpy().astype(np.float32)
        state = env.step(actions)
        actor_obs, _, priv_info = split_hora_obs_with_priv_info(state.obs, state.info)
        current_priv_info = priv_info.astype(np.float32) if priv_info is not None else None
        return np.asarray(actor_obs, dtype=np.float32)

    print("Collecting physics states...")
    with torch.inference_mode():
        play_video_path = cast(
            str | None,
            env.run_playback_mode(
                play_render_mode=getattr(cfg.training, "play_render_mode", "auto"),
                play_steps=getattr(cfg.training, "play_steps", None),
                output_video=os.path.join(load_path_dir, "play_video.mp4")
                if load_path_dir
                else None,
                render_spacing=float(
                    getattr(cfg.training, "render_spacing", getattr(env.cfg, "render_spacing", 1.0))
                ),
                initialize=initialize_play_obs,
                step=step_play_obs,
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
            ),
        )
    if play_video_path is not None:
        print(f"Saving video to {play_video_path} with mediapy...")
    print("Done.")
    return play_video_path


__all__ = ["HoraAPPORunner", "HoraAPPORuntime", "play_hora_appo", "resolve_hora_appo_runtime"]
