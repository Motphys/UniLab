"""Off-policy play helpers shared by training scripts and playback entrypoints."""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

from omegaconf import DictConfig, OmegaConf

from unilab.training.common import assert_offpolicy_task_choice_matches_algo


def default_device(torch_module, preferred: str | None = None) -> str:
    """Resolve runtime device with optional user override."""
    if preferred:
        return preferred
    if torch_module.cuda.is_available():
        return "cuda"
    xpu = getattr(torch_module, "xpu", None)
    xpu_is_available = getattr(xpu, "is_available", None)
    if callable(xpu_is_available) and xpu_is_available():
        return "xpu"
    if torch_module.backends.mps.is_available():
        return "mps"
    return "cpu"


def extract_reset_obs(reset_result):
    """Extract obs_dict from env.reset(...) using the current (obs_dict, info_dict) contract."""
    if isinstance(reset_result, tuple):
        if len(reset_result) == 2:
            obs_out, _ = reset_result
            return obs_out
    raise ValueError(f"Unexpected env.reset return format: {type(reset_result)!r}")


def resolve_play_obs_dim(obs_groups_spec: dict[str, int]) -> int:
    obs_dim, _ = resolve_play_obs_dims(obs_groups_spec)
    return obs_dim


def resolve_play_obs_dims(obs_groups_spec: dict[str, int]) -> tuple[int, int]:
    from unilab.base.observations import get_obs_dims

    obs_dim, critic_obs_dim = get_obs_dims(obs_groups_spec)
    return int(obs_dim), int(critic_obs_dim)


def extract_play_obs(obs_dict):
    from unilab.base.observations import split_obs_dict

    obs_out, _ = split_obs_dict(obs_dict)
    return obs_out


def resolve_play_actor_spec(
    algo_name: str,
    cfg: DictConfig,
    *,
    obs_dim: int,
    critic_obs_dim: int,
) -> tuple[str, dict[str, Any]]:
    """Resolve the actor implementation and model kwargs used by off-policy play."""
    if algo_name != "sac":
        return algo_name, {}

    from unilab.algos.torch.offpolicy.runtime import resolve_custom_offpolicy_runtime

    rl_cfg = cast(dict[str, Any], OmegaConf.to_container(cfg.algo, resolve=True))
    custom_runtime = resolve_custom_offpolicy_runtime(rl_cfg)
    if custom_runtime is None:
        return "sac", {}

    actor_algo_type = str(custom_runtime.algo_type or algo_name)
    actor_kwargs = custom_runtime.build_model_kwargs(
        obs_dim=int(obs_dim),
        critic_obs_dim=int(critic_obs_dim),
    )
    return actor_algo_type, actor_kwargs


def build_offpolicy_env_cfg_override(
    algo_name: str,
    cfg: DictConfig,
    *,
    root_dir: str | Path,
) -> dict[str, Any] | None:
    """Build the task env override for off-policy play through the backend adapter."""
    from unilab.training.backend_adapter import BackendAdapter

    assert_offpolicy_task_choice_matches_algo(cfg, algo_name=algo_name)
    return cast(
        dict[str, Any] | None,
        BackendAdapter(cfg, root_dir=root_dir, algo_name=algo_name).build_task_env_cfg_override(),
    )
