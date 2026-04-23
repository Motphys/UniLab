from __future__ import annotations

import warnings

from unilab.visualization.render_many import *  # noqa: F403

warnings.warn(
    "`unilab.utils.render_many` is deprecated and will be removed in 0.2.0; "
    "use `unilab.visualization.render_many` instead.",
    DeprecationWarning,
    stacklevel=2,
)
