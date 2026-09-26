from __future__ import annotations

from pathlib import Path

from packaging.requirements import Requirement

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python < 3.11
    import tomli as tomllib


ROOT = Path(__file__).resolve().parents[2]


def _dependencies(path: Path) -> dict[str, Requirement]:
    project = tomllib.loads(path.read_text(encoding="utf-8"))
    requirements = (Requirement(value) for value in project["project"]["dependencies"])
    return {requirement.name: requirement for requirement in requirements}


def test_rocm_profile_preserves_non_substituted_runtime_dependencies() -> None:
    default_dependencies = _dependencies(ROOT / "pyproject.toml")
    rocm_dependencies = _dependencies(ROOT / "pyproject.rocm.toml")

    # The ROCm profile may replace constraints and package sources, but every
    # default runtime dependency must remain present by name.
    missing = set(default_dependencies) - set(rocm_dependencies)

    assert not missing
    assert "numba" in rocm_dependencies
    assert "prettytable" in rocm_dependencies


def test_rocm_lock_contains_profile_runtime_dependencies() -> None:
    lock = tomllib.loads((ROOT / "uv.rocm.lock").read_text(encoding="utf-8"))
    packages = {package["name"] for package in lock["package"]}
    root = next(package for package in lock["package"] if package["name"] == "unilab")
    root_dependencies = {dependency["name"] for dependency in root["dependencies"]}

    assert {"numba", "prettytable"} <= packages
    assert {"numba", "prettytable"} <= root_dependencies
