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


def _project_scripts(path: Path) -> dict[str, str]:
    project = tomllib.loads(path.read_text(encoding="utf-8"))
    return dict(project["project"]["scripts"])


def _project(path: Path) -> dict:
    return tomllib.loads(path.read_text(encoding="utf-8"))


def test_rocm_profile_preserves_non_substituted_runtime_dependencies() -> None:
    default_dependencies = _dependencies(ROOT / "pyproject.toml")
    rocm_dependencies = _dependencies(ROOT / "pyproject.rocm.toml")

    # The ROCm profile may replace constraints and package sources, but every
    # default runtime dependency must remain present by name.
    missing = set(default_dependencies) - set(rocm_dependencies)

    assert not missing
    assert "prettytable" in rocm_dependencies
    assert "numba" not in default_dependencies
    assert "numba" not in rocm_dependencies


def test_rocm_profile_preserves_project_scripts() -> None:
    default_scripts = _project_scripts(ROOT / "pyproject.toml")
    rocm_scripts = _project_scripts(ROOT / "pyproject.rocm.toml")

    assert rocm_scripts == default_scripts
    assert rocm_scripts["uni-cumps"] == "unilab.training.cuda_mps_cli:main"


def test_rocm_lock_contains_profile_runtime_dependencies() -> None:
    lock = tomllib.loads((ROOT / "uv.rocm.lock").read_text(encoding="utf-8"))
    packages = {package["name"] for package in lock["package"]}
    root = next(package for package in lock["package"] if package["name"] == "unilab")
    root_dependencies = {dependency["name"] for dependency in root["dependencies"]}

    assert "prettytable" in packages
    assert "prettytable" in root_dependencies
    assert "numba" not in root_dependencies


def test_ray_local_optional_dependencies_stay_branch_local() -> None:
    for path in (ROOT / "pyproject.toml", ROOT / "pyproject.rocm.toml"):
        project = _project(path)
        extras = project["project"]["optional-dependencies"]
        assert extras["ray-local"] == ["uni-ray[warp]"]
        assert "uni-ray" not in _dependencies(path)

        sources = project["tool"]["uv"]["sources"]
        assert sources["unisim-core"] == [
            {"path": "../unisim", "editable": True, "marker": "extra == 'ray-local'"}
        ]
        assert sources["uni-ray"] == [
            {"path": "../uni_ray", "editable": True, "marker": "extra == 'ray-local'"}
        ]
