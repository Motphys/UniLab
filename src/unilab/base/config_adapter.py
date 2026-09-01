"""Resolved config adaptation for training entrypoints."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any, Callable

from omegaconf import DictConfig, OmegaConf

from unilab.base import registry
from unilab.base.backend import materialize_scene_visual_override
from unilab.base.scene import SceneCfg
from unilab.utils.reward import extract_reward_config


class BackendAdapter:
    """Build env/play overrides from the final composed config."""

    def __init__(
        self,
        cfg: DictConfig,
        *,
        root_dir: str | Path,
        algo_name: str | None = None,
        scene_materializer: Callable[..., str] = materialize_scene_visual_override,
    ) -> None:
        self.cfg = cfg
        self.root_dir = Path(root_dir)
        self.algo_name = algo_name
        self.scene_materializer = scene_materializer

    def build_task_env_cfg_override(self) -> dict[str, Any]:
        """Build env_cfg_override from the resolved reward + env sections."""
        registry.ensure_registries()
        task_name = str(self.cfg.training.task_name)
        reward_target = registry.resolve_reward_override_field(task_name)
        env_overrides = self._to_plain_dict(getattr(self.cfg, "env", None))
        if reward_target in env_overrides:
            raise ValueError(
                f"Task '{task_name}' declares both Hydra root 'reward' and "
                f"'env.{reward_target}'; use the root reward owner only"
            )

        env_cfg_override = extract_reward_config(self.cfg, target_field=reward_target)
        env_cfg_override.update(env_overrides)

        return env_cfg_override

    def build_play_env_cfg_override(self) -> dict[str, Any]:
        """Build play-mode overrides from an optional backend-agnostic play profile."""
        env_cfg_override = self.build_task_env_cfg_override()
        # IsaacSim must know the render intent before its Python 3.11 worker
        # launches Kit.  Keep this routing in the config adapter (the owner
        # layer) so training env construction remains headless and the
        # renderer never leaks into runners/learners.  Other backends do not
        # consume this field and retain their existing play contracts.
        if str(OmegaConf.select(self.cfg, "training.sim_backend", default="")) == "isaacsim":
            play_render_mode = OmegaConf.select(
                self.cfg, "training.play_render_mode", default="auto"
            )
            env_cfg_override["isaacsim_render_mode"] = (
                "auto" if play_render_mode is None else str(play_render_mode)
            )
        play_profile = getattr(self.cfg, "play_profile", None)
        if (
            play_profile is None
            or not getattr(play_profile, "enabled", False)
            or not self.cfg.training.play_only
        ):
            return env_cfg_override

        env_profile = getattr(play_profile, "env", None)
        if env_profile is not None:
            self._apply_env_profile(env_cfg_override, env_profile)

        scene_override = getattr(play_profile, "scene", None)
        if scene_override is None or not getattr(scene_override, "enabled", False):
            return env_cfg_override

        source_model_file = getattr(scene_override, "source_model_file", None)
        if not source_model_file:
            raise ValueError("play_profile.scene.source_model_file must be configured")

        materialized_model_file = self.scene_materializer(
            self._resolve_root_relative_path(str(source_model_file)),
            ground_texture_file=(
                self._resolve_root_relative_path(str(scene_override.ground_texture_file))
                if getattr(scene_override, "ground_texture_file", None)
                else None
            ),
            ground_texrepeat=getattr(scene_override, "ground_texrepeat", None),
            skybox_rgb1=getattr(scene_override, "skybox_rgb1", None),
            skybox_rgb2=getattr(scene_override, "skybox_rgb2", None),
        )
        scene = env_cfg_override.get("scene")
        if scene is None:
            env_cfg_override["scene"] = SceneCfg(model_file=materialized_model_file)
        elif isinstance(scene, SceneCfg):
            env_cfg_override["scene"] = replace(scene, model_file=materialized_model_file)
        elif isinstance(scene, dict):
            env_cfg_override["scene"] = {**scene, "model_file": materialized_model_file}
        else:
            raise TypeError(
                "play_profile.scene can only override a missing, SceneCfg, or mapping scene; "
                f"got {type(scene).__name__}"
            )
        return env_cfg_override

    def _apply_env_profile(self, env_cfg_override: dict[str, Any], env_profile: Any) -> None:
        env_cfg_override.update(self._to_plain_dict(env_profile))

    def _resolve_root_relative_path(self, path_value: str) -> str:
        candidate = Path(path_value)
        if candidate.is_absolute():
            return str(candidate)
        return str((self.root_dir / candidate).resolve())

    def _to_plain_dict(self, value: Any) -> dict[str, Any]:
        if OmegaConf.is_config(value):
            resolved = OmegaConf.to_container(value, resolve=True)
        elif isinstance(value, dict):
            resolved = value
        else:
            return {}
        if not isinstance(resolved, dict):
            return {}
        return {str(key): item for key, item in resolved.items()}


def create_env(
    cfg: DictConfig,
    *,
    num_envs: int,
    env_cfg_override: dict[str, Any] | None = None,
    sim_backend: str | None = None,
    task_name: str | None = None,
):
    """Construct an environment via the registry using the current Hydra config."""
    from unilab.base import registry

    return registry.make(
        task_name or str(OmegaConf.select(cfg, "training.task_name")),
        num_envs=num_envs,
        sim_backend=sim_backend or str(OmegaConf.select(cfg, "training.sim_backend")),
        env_cfg_override=env_cfg_override,
    )
