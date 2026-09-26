from __future__ import annotations

from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python < 3.11
    import tomli as tomllib


def test_torch_cuda_source_covers_windows_and_linux() -> None:
    pyproject = Path(__file__).resolve().parents[2] / "pyproject.toml"
    data = tomllib.loads(pyproject.read_text(encoding="utf-8"))

    torch_sources = data["tool"]["uv"]["sources"]["torch"]
    if any(source.get("index") == "pytorch-rocm72" for source in torch_sources):
        rocm_sources = [
            source for source in torch_sources if source.get("index") == "pytorch-rocm72"
        ]
        assert {source["marker"] for source in rocm_sources} == {
            "sys_platform == 'linux' and platform_machine == 'x86_64'"
        }
        return

    cu130_sources = [source for source in torch_sources if source.get("index") == "r2-cu130"]

    # cu128 tops out at torch 2.11; torch>=2.12 CUDA wheels for linux/win all
    # ship from cu130, so a single source entry covers both.
    assert [source["marker"] for source in cu130_sources] == [
        "sys_platform=='linux' or sys_platform=='win32'"
    ]


def test_windows_lock_uses_cuda_torch() -> None:
    lockfile = Path(__file__).resolve().parents[2] / "uv.lock"
    lock = tomllib.loads(lockfile.read_text(encoding="utf-8"))

    root = next(package for package in lock["package"] if package["name"] == "unilab")
    torch_dependencies = [dep for dep in root["dependencies"] if dep["name"] == "torch"]

    rocm_dependency = next(
        (dep for dep in torch_dependencies if "rocm" in dep.get("source", {}).get("registry", "")),
        None,
    )
    if rocm_dependency is not None:
        assert {
            "name": "torch",
            "version": "2.14.0+rocm7.2",
            "source": {"registry": "https://download.pytorch.org/whl/rocm7.2"},
            "marker": "platform_machine == 'x86_64' and sys_platform == 'linux'",
        } in torch_dependencies
        return

    cu130_dependency = next(
        dep
        for dep in torch_dependencies
        if dep["version"] == "2.14.0+cu130"
        and dep["source"] == {"registry": "https://download-r2.pytorch.org/whl/cu130"}
    )
    # uv adds impossible-extra guards for the Newton/mjwarp and Newton/mujoco
    # conflict matrix.  Assert the platform clauses remain present while
    # allowing those generated guards to evolve.
    assert "sys_platform == 'linux'" in cu130_dependency["marker"]
    assert "sys_platform == 'win32'" in cu130_dependency["marker"]

    torch_packages = [package for package in lock["package"] if package["name"] == "torch"]
    cu130_package = next(
        package
        for package in torch_packages
        if package["source"] == {"registry": "https://download-r2.pytorch.org/whl/cu130"}
    )

    assert cu130_package["version"] == "2.14.0+cu130"
    assert any(
        "sys_platform == 'win32'" in marker for marker in cu130_package["resolution-markers"]
    )

    wheel_urls = [wheel["url"] for wheel in cu130_package["wheels"]]
    assert any("torch-2.14.0%2Bcu130" in url and "win_amd64.whl" in url for url in wheel_urls)
