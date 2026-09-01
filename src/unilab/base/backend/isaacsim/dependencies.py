"""Runtime discovery for the external IsaacSim 5.1 / IsaacLab worker."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from unilab.base.backend.subprocess_ipc.runtime import WorkerRuntime, build_worker_environment

ENV_HOME = "UNILAB_ISAACSIM_HOME"
ENV_PYTHON = "UNILAB_ISAACSIM_PYTHON"

_SETUP_HINT = (
    "Install the dedicated IsaacSim/IsaacLab worker environment with "
    "scripts/tools/setup_isaacsim_env.sh (see the IsaacSim backend page in "
    "docs/sphinx/source/*/2-user_guide/3-backends/5-isaacsim.md)."
)


class IsaacSimDependencyError(ImportError):
    """Raised with an actionable setup hint when the worker runtime is absent."""


@dataclass(frozen=True)
class IsaacSimRuntime(WorkerRuntime):
    """Resolved on-disk locations for the Python 3.11 IsaacSim worker."""

    isaaclab_source: Path | None = None

    @property
    def isaacsim_python(self) -> Path:
        """Compatibility alias used by worker payloads and diagnostics."""
        return self.python


def default_isaacsim_home() -> Path:
    return Path(os.environ.get(ENV_HOME, "~/.unilab/isaacsim")).expanduser()


def _candidate_paths(home: Path) -> IsaacSimRuntime:
    venv = home / "venv"
    return IsaacSimRuntime(
        python=venv / "bin" / "python",
        package_path=venv / "lib" / "python3.11" / "site-packages",
        lib_path=venv / "lib",
        bin_path=venv / "bin",
        isaaclab_source=home / "IsaacLab" / "source",
    )


def _runtime_from_override(python: Path, home: Path) -> IsaacSimRuntime:
    """Build a runtime from an explicitly supplied interpreter.

    ``UNILAB_ISAACSIM_PYTHON`` is intentionally a complete escape hatch for
    installations that do not use the helper script's directory layout.  If
    the interpreter looks like ``<venv>/bin/python``, we opportunistically
    discover its sibling library/site-packages directories; otherwise those
    paths remain unset and the interpreter's own environment is trusted.
    """
    bin_path = python.parent if python.parent.is_dir() else None
    venv_root = python.parent.parent if python.parent.name in {"bin", "Scripts"} else None
    package_path: Path | None = None
    lib_path: Path | None = None
    if venv_root is not None:
        candidate_lib = venv_root / "lib"
        if candidate_lib.is_dir():
            lib_path = candidate_lib
            site_candidates = sorted(candidate_lib.glob("python*/site-packages"))
            if site_candidates:
                package_path = site_candidates[0]

    source_candidates = [home / "IsaacLab" / "source"]
    if venv_root is not None:
        source_candidates.append(venv_root.parent / "IsaacLab" / "source")
    isaaclab_source = next((path for path in source_candidates if path.is_dir()), None)
    return IsaacSimRuntime(
        python=python,
        package_path=package_path,
        lib_path=lib_path,
        bin_path=bin_path,
        isaaclab_source=isaaclab_source,
    )


def resolve_isaacsim_runtime(python_override: str | None = None) -> IsaacSimRuntime:
    """Resolve the interpreter and package/source layout without importing Kit."""
    home = default_isaacsim_home()
    candidate = _candidate_paths(home)
    override = python_override or os.environ.get(ENV_PYTHON)
    if override:
        python = Path(override).expanduser()
        if not python.is_file():
            raise IsaacSimDependencyError(
                f"isaacsim backend could not find the Python 3.11 worker interpreter at "
                f"{python}. {_SETUP_HINT}"
            )
        # An explicit interpreter may belong to a completely custom layout;
        # do not reject it merely because the default helper-tree paths are
        # absent.  The worker will report import errors with its full
        # traceback if that environment is not actually IsaacSim-capable.
        return _runtime_from_override(python, home)

    python = candidate.python
    if not python.is_file():
        raise IsaacSimDependencyError(
            f"isaacsim backend could not find the Python 3.11 worker interpreter at "
            f"{python}. {_SETUP_HINT}"
        )
    # The venv may use an editable IsaacLab install or a regular wheel.  A
    # package directory is still required as a cheap, import-free sanity check.
    if candidate.package_path is not None and not candidate.package_path.is_dir():
        raise IsaacSimDependencyError(
            f"isaacsim backend could not find the worker site-packages directory at "
            f"{candidate.package_path}. {_SETUP_HINT}"
        )
    if candidate.isaaclab_source is not None and not candidate.isaaclab_source.is_dir():
        raise IsaacSimDependencyError(
            f"isaacsim backend could not find the IsaacLab source directory at "
            f"{candidate.isaaclab_source}. {_SETUP_HINT}"
        )
    if candidate.lib_path is not None and not candidate.lib_path.is_dir():
        raise IsaacSimDependencyError(
            f"isaacsim backend could not find the worker library directory at "
            f"{candidate.lib_path}. {_SETUP_HINT}"
        )
    return IsaacSimRuntime(
        python=python,
        package_path=candidate.package_path,
        lib_path=candidate.lib_path,
        bin_path=candidate.bin_path,
        isaaclab_source=candidate.isaaclab_source,
    )


def isaacsim_runtime_available() -> bool:
    try:
        resolve_isaacsim_runtime()
    except IsaacSimDependencyError:
        return False
    return True


def build_worker_env(runtime: IsaacSimRuntime) -> dict[str, str]:
    """Build an environment that prefers the pinned Kit/venv libraries."""
    env = build_worker_environment(runtime)
    env.setdefault("OMNI_KIT_ACCEPT_EULA", "1")
    if runtime.isaaclab_source is not None:
        old = env.get("PYTHONPATH", "")
        source = str(runtime.isaaclab_source)
        env["PYTHONPATH"] = f"{source}:{old}" if old else source
    return env


__all__ = [
    "ENV_HOME",
    "ENV_PYTHON",
    "IsaacSimDependencyError",
    "IsaacSimRuntime",
    "build_worker_env",
    "default_isaacsim_home",
    "isaacsim_runtime_available",
    "resolve_isaacsim_runtime",
]
