"""Backend-specific config adaptation for training entrypoints."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Callable

from omegaconf import DictConfig, OmegaConf

from unilab.utils.reward_utils import extract_reward_config
from unilab.utils.xml_utils import materialize_scene_visual_override


class BackendAdapter:
    """Apply backend-specific config and env overrides while respecting CLI overrides."""

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
        self.explicit_keys = {
            arg.split("=", 1)[0].lstrip("+") for arg in sys.argv[1:] if "=" in arg
        }

    def build_task_env_cfg_override(self) -> dict[str, Any]:
        """Build env_cfg_override for training/play entrypoints."""
        env_cfg_override = extract_reward_config(self.cfg)
        motrix_legacy = getattr(self.cfg, "motrix_legacy", None)
        if (
            motrix_legacy is None
            or not getattr(motrix_legacy, "enabled", False)
            or self.cfg.training.sim_backend != "motrix"
        ):
            return env_cfg_override

        applies_to = getattr(motrix_legacy, "applies_to", None)
        target_algo = getattr(applies_to, "algo", None) if applies_to is not None else None
        if self.algo_name is not None and target_algo is not None and target_algo != self.algo_name:
            return env_cfg_override

        algo_overrides = getattr(motrix_legacy, "algo_overrides", None)
        if algo_overrides is not None:
            self._apply_nested_overrides(
                self.cfg.algo,
                algo_overrides,
                base_path="algo",
            )

        env_profile = getattr(motrix_legacy, "env_cfg_override", None)
        if env_profile is not None:
            self._apply_env_profile(env_cfg_override, env_profile)

        return env_cfg_override

    def build_play_env_cfg_override(self) -> dict[str, Any]:
        """Build play-mode overrides, including Motrix scene customization when configured."""
        env_cfg_override = self.build_task_env_cfg_override()
        play_profile = getattr(self.cfg, "motrix_play_only", None)
        if (
            play_profile is None
            or not getattr(play_profile, "enabled", False)
            or self.cfg.training.sim_backend != "motrix"
            or not self.cfg.training.play_only
        ):
            return env_cfg_override

        training_overrides = getattr(play_profile, "training_overrides", None)
        if training_overrides is not None:
            self._apply_nested_overrides(
                self.cfg.training,
                training_overrides,
                base_path="training",
            )

        env_profile = getattr(play_profile, "env_cfg_override", None)
        if env_profile is not None:
            self._apply_env_profile(env_cfg_override, env_profile)

        scene_override = getattr(play_profile, "scene_override", None)
        if scene_override is None or not getattr(scene_override, "enabled", False):
            return env_cfg_override

        source_model_file = getattr(scene_override, "source_model_file", None)
        if not source_model_file:
            raise ValueError("motrix_play_only.scene_override.source_model_file must be configured")

        env_cfg_override["model_file"] = self.scene_materializer(
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
        return env_cfg_override

    def _apply_nested_overrides(self, target: Any, overrides: Any, *, base_path: str) -> None:
        if OmegaConf.is_config(overrides):
            override_dict = OmegaConf.to_container(overrides, resolve=True)
        elif isinstance(overrides, dict):
            override_dict = overrides
        else:
            return

        if not isinstance(override_dict, dict):
            return

        for key, value in override_dict.items():
            key_str = str(key)
            path = f"{base_path}.{key_str}"
            if isinstance(value, dict):
                if path in self.explicit_keys:
                    continue
                if key_str not in target or target[key_str] is None:
                    target[key_str] = {}
                self._apply_nested_overrides(target[key_str], value, base_path=path)
                continue
            if path not in self.explicit_keys:
                target[key_str] = value

    def _apply_env_profile(self, env_cfg_override: dict[str, Any], env_profile: Any) -> None:
        if OmegaConf.is_config(env_profile):
            env_profile_dict = OmegaConf.to_container(env_profile, resolve=True)
        elif isinstance(env_profile, dict):
            env_profile_dict = env_profile
        else:
            return

        if not isinstance(env_profile_dict, dict):
            return

        reward_explicit = any(
            key == "reward"
            or key.startswith("reward.")
            or key.startswith("reward@")
            or key.startswith("reward_")
            for key in self.explicit_keys
        )
        for key, value in env_profile_dict.items():
            key_str = str(key)
            if key_str == "reward_config" and reward_explicit:
                continue
            env_cfg_override[key_str] = value

    def _resolve_root_relative_path(self, path_value: str) -> str:
        candidate = Path(path_value)
        if candidate.is_absolute():
            return str(candidate)
        return str((self.root_dir / candidate).resolve())
