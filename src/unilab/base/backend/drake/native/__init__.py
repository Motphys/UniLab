from __future__ import annotations

import sys


def _pydrake_loaded() -> bool:
    return any(name == "pydrake" or name.startswith("pydrake.") for name in sys.modules)


if _pydrake_loaded():
    _NATIVE_IMPORT_ERROR: ImportError | None = ImportError(
        "Native DrakeEnvPool cannot be imported after pydrake has already been "
        "imported in this process. Start a fresh process for native DrakeUni."
    )
    NativeDrakeEnvPool = None  # type: ignore[assignment]

    def native_available() -> bool:
        return False

else:
    try:
        from ._drake_env_pool import NativeDrakeEnvPool, native_available
    except ImportError as exc:  # pragma: no cover - optional local extension.
        _NATIVE_IMPORT_ERROR = exc
        NativeDrakeEnvPool = None  # type: ignore[assignment]

        def native_available() -> bool:
            return False

    else:
        _NATIVE_IMPORT_ERROR = None


def native_import_error() -> ImportError | None:
    return _NATIVE_IMPORT_ERROR


__all__ = ["NativeDrakeEnvPool", "native_available", "native_import_error"]
