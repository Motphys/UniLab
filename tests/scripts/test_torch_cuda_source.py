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
    cu128_sources = [source for source in torch_sources if source.get("index") == "pytorch-cu128"]
    cu130_sources = [source for source in torch_sources if source.get("index") == "r2-cu130"]

    assert {source["marker"] for source in cu128_sources} == {
        "sys_platform=='linux' and platform_machine=='x86_64'",
        "sys_platform=='win32'",
    }
    assert [source["marker"] for source in cu130_sources] == [
        "sys_platform=='linux' and platform_machine=='aarch64'"
    ]


def test_windows_lock_uses_cuda_torch() -> None:
    lockfile = Path(__file__).resolve().parents[2] / "uv.lock"
    lock = tomllib.loads(lockfile.read_text(encoding="utf-8"))

    root = next(package for package in lock["package"] if package["name"] == "unilab")
    torch_dependencies = [dep for dep in root["dependencies"] if dep["name"] == "torch"]

    assert {
        "name": "torch",
        "version": "2.7.0+cu128",
        "source": {"registry": "https://download.pytorch.org/whl/cu128"},
        "marker": "(platform_machine == 'x86_64' and sys_platform == 'linux') or sys_platform == 'win32'",
    } in torch_dependencies
    assert {
        "name": "torch",
        "version": "2.9.0+cu130",
        "source": {"registry": "https://download-r2.pytorch.org/whl/cu130"},
        "marker": "platform_machine == 'aarch64' and sys_platform == 'linux'",
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
