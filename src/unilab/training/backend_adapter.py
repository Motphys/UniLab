"""Resolved config adaptation for training entrypoints."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from omegaconf import DictConfig, OmegaConf

from unilab.base.backend.mujoco.xml import materialize_scene_visual_override
from unilab.base.scene import SceneCfg
from unilab.training.reward import extract_reward_config


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
        env_cfg_override = extract_reward_config(self.cfg)
        env_cfg_override.update(self._to_plain_dict(getattr(self.cfg, "env", None)))

        return env_cfg_override

    def build_play_env_cfg_override(self) -> dict[str, Any]:
        """Build play-mode overrides from an optional backend-agnostic play profile."""
        env_cfg_override = self.build_task_env_cfg_override()
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

        env_cfg_override["scene"] = SceneCfg(
            model_file=self.scene_materializer(
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
