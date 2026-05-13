"""Shared playback helper utilities."""

from __future__ import annotations

from typing import Any


def env_cfg_value(env: Any, name: str, default: Any) -> Any:
    cfg = getattr(env, "cfg", None)
    if cfg is None:
        return default
    return getattr(cfg, name, default)
