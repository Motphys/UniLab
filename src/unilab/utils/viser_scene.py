from __future__ import annotations

import warnings

from unilab.visualization.viser_scene import (
    VISER_AVAILABLE,
    MujocoViserScene,
    build_visible_env_indices,
)

warnings.warn(
    "`unilab.utils.viser_scene` is deprecated and will be removed in 0.2.0; "
    "use `unilab.visualization.viser_scene` instead.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = ["VISER_AVAILABLE", "MujocoViserScene", "build_visible_env_indices"]
