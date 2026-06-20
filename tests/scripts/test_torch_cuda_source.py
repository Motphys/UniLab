from __future__ import annotations

from pathlib import Path

import tomllib


def test_torch_cuda_source_covers_windows_and_linux() -> None:
    pyproject = Path(__file__).resolve().parents[2] / "pyproject.toml"
    data = tomllib.loads(pyproject.read_text(encoding="utf-8"))

    torch_sources = data["tool"]["uv"]["sources"]["torch"]
    cu128_sources = [source for source in torch_sources if source.get("index") == "pytorch-cu128"]

    assert len(cu128_sources) == 1
    marker = cu128_sources[0]["marker"]
    assert "sys_platform=='linux'" in marker
    assert "sys_platform=='win32'" in marker


def test_windows_lock_uses_cuda_torch() -> None:
    lockfile = Path(__file__).resolve().parents[2] / "uv.lock"
    lock = tomllib.loads(lockfile.read_text(encoding="utf-8"))

    root = next(package for package in lock["package"] if package["name"] == "unilab")
    torch_dependencies = [dep for dep in root["dependencies"] if dep["name"] == "torch"]

    assert {
        "name": "torch",
        "version": "2.7.0+cu128",
        "source": {"registry": "https://download.pytorch.org/whl/cu128"},
        "marker": "sys_platform == 'linux' or sys_platform == 'win32'",
    } in torch_dependencies

    torch_packages = [package for package in lock["package"] if package["name"] == "torch"]
    cu128_package = next(
        package
        for package in torch_packages
        if package["source"] == {"registry": "https://download.pytorch.org/whl/cu128"}
    )

    assert cu128_package["version"] == "2.7.0+cu128"
    assert any(
        "sys_platform == 'win32'" in marker for marker in cu128_package["resolution-markers"]
    )

    wheel_urls = [wheel["url"] for wheel in cu128_package["wheels"]]
    assert any("torch-2.7.0%2Bcu128" in url and "win_amd64.whl" in url for url in wheel_urls)
