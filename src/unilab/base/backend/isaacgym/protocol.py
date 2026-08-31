"""Compatibility import for the canonical subprocess IPC protocol.

New workers must load ``subprocess_ipc/protocol.py`` by path.  This module is
kept so existing IsaacGym callers and tests retain their import path.
"""

from unilab.base.backend.subprocess_ipc import protocol as _protocol
from unilab.base.backend.subprocess_ipc.protocol import *  # noqa: F401,F403

# Keep the historical module path a faithful compatibility surface.  In
# particular, callers that introspect ``__all__`` (and the Python 3.8 worker
# loader used by older integrations) should see the same public names as the
# canonical shared module rather than an empty star-import namespace.
__all__ = _protocol.__all__
