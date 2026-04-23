from __future__ import annotations

import warnings

from unilab.base.backend.xml import *  # noqa: F403

warnings.warn(
    "`unilab.utils.xml_utils` is deprecated and will be removed in 0.2.0; "
    "use `unilab.base.backend.xml` instead.",
    DeprecationWarning,
    stacklevel=2,
)
