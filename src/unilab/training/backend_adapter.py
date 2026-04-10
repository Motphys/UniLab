"""Backend-specific config adaptation for training entrypoints."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Callable

from omegaconf import DictConfig, OmegaConf

from unilab.utils.reward_utils import extract_reward_config
from unilab.utils.xml_utils import materialize_scene_visual_override


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
        self.explicit_keys = {
            arg.split("=", 1)[0].lstrip("+") for arg in sys.argv[1:] if "=" in arg
        }

    def _is_motrix(self) -> bool:
        return bool(self.cfg.training.sim_backend == "motrix")

    def _assert_task_backend_identity(self) -> None:
        selected_backend = getattr(self.cfg, "_selected_sim_backend", None)
        if selected_backend is None:
            return
        resolved_backend = str(self.cfg.training.sim_backend)
        if str(selected_backend) != resolved_backend:
            raise ValueError(
                "Task owner config is inconsistent with training.sim_backend. "
                "`task=<task>/<backend>` is the backend selection contract; "
                "`training.sim_backend` is the selected task owner's identity field, "
                "not a standalone backend switch."
            )

    def build_task_env_cfg_override(self) -> dict[str, Any]:
        """Build env_cfg_override from the resolved reward + env sections."""
        self._assert_task_backend_identity()
        env_cfg_override = extract_reward_config(self.cfg)
        env_cfg_override.update(self._to_plain_dict(getattr(self.cfg, "env", None)))

        return env_cfg_override

    def build_play_env_cfg_override(self) -> dict[str, Any]:
        """Build play-mode overrides, including Motrix scene customization when configured."""
        env_cfg_override = self.build_task_env_cfg_override()
        play_profile = getattr(self.cfg, "motrix_play_only", None)
        if (
            play_profile is None
            or not getattr(play_profile, "enabled", False)
            or not self._is_motrix()
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
        override_dict = self._to_plain_dict(overrides)
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
