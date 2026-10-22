from __future__ import annotations

from pathlib import Path

from packaging.requirements import Requirement

from unilab import diagnostics

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python < 3.11
    import tomli as tomllib


ROOT = Path(__file__).resolve().parents[2]


def _project(path: Path) -> dict[str, object]:
    return tomllib.loads(path.read_text(encoding="utf-8"))


def _dependencies(project: dict[str, object]) -> dict[str, Requirement]:
    values = (Requirement(value) for value in project["project"]["dependencies"])  # type: ignore[index]
    return {requirement.name: requirement for requirement in values}


def test_single_project_uses_wheel_variant_dependencies() -> None:
    project = _project(ROOT / "pyproject.toml")
    dependencies = _dependencies(project)

    assert dependencies["unisim-core"].specifier.contains("2.1.1")
    assert dependencies["torch"].specifier.contains("2.10.0")
    assert not list(ROOT.glob("pyproject.*rocm*.toml"))
    assert not list(ROOT.glob("uv.*rocm*.lock"))
    assert "sources" not in project["tool"]["uv"]  # type: ignore[index]
    assert "index" not in project["tool"]["uv"]  # type: ignore[index]


def test_lock_records_accelerator_torch_variants() -> None:
    lock = tomllib.loads((ROOT / "uv.lock").read_text(encoding="utf-8"))
    torch = next(package for package in lock["package"] if package["name"] == "torch")
    wheels = [wheel["url"] for wheel in torch["wheels"]]

    assert torch["version"] == "2.10.0"
    assert torch["source"]["registry"] == "https://wheelnext.github.io/variants-index/v0.0.3"
    assert any("cp312-cp312-manylinux_2_28_x86_64-rocm7.0.whl" in url for url in wheels)
    assert any("cp312-cp312-manylinux_2_28_x86_64-cuda13.0.whl" in url for url in wheels)
    assert any("cp312-cp312-manylinux_2_28_x86_64-null.whl" in url for url in wheels)
    assert any("cp312-cp312-linux_x86_64-xpu.whl" in url for url in wheels)


def test_lock_uses_unisim_wheel() -> None:
    lock = tomllib.loads((ROOT / "uv.lock").read_text(encoding="utf-8"))
    unisim = next(package for package in lock["package"] if package["name"] == "unisim-core")

    assert unisim["version"] == "2.1.1"
    unisim_wheels = [wheel["url"] for wheel in unisim["wheels"]]
    assert any(url.endswith("unisim_core-2.1.1-py3-none-any.whl") for url in unisim_wheels)


def test_doctor_expects_wheelnext_uv() -> None:
    assert diagnostics.WHEELNEXT_UV_VERSION == "0.9.14"


def test_uv_installer_pins_wheelnext_build() -> None:
    installer = (ROOT / "scripts/tools/install_uv_wheelnext.sh").read_text(encoding="utf-8")

    assert 'VERSION="0.9.14"' in installer
    assert "https://wheelnext.astral.sh/v0.0.3" in installer
    assert "scripts/tools/install_uv_wheelnext.sh" in (ROOT / "Dockerfile").read_text(
        encoding="utf-8"
    )
    assert (ROOT / "scripts/tools/install_uv_wheelnext.sh").stat().st_mode & 0o111
    action = (ROOT / ".github/actions/wheelnext-uv/action.yml").read_text(encoding="utf-8")
    assert 'echo "PATH=$install_dir:$PATH" >> "$GITHUB_ENV"' in action
    dockerfile_lines = (ROOT / "Dockerfile").read_text(encoding="utf-8").splitlines()
    copy_index = next(
        index for index, line in enumerate(dockerfile_lines) if line == "COPY . /workspace/UniLab"
    )
    installer_index = next(
        index
        for index, line in enumerate(dockerfile_lines)
        if "install_uv_wheelnext.sh" in line and line.startswith("RUN ")
    )
    sync_index = next(
        index
        for index, line in enumerate(dockerfile_lines)
        if line.startswith("RUN uv sync --dev --extra mujoco")
    )
    assert copy_index < installer_index < sync_index
    assert "NV_VARIANT_PROVIDER_FORCE_CUDA_DRIVER_VERSION=12.8" in "\n".join(dockerfile_lines)


def test_project_workflows_override_stable_uv_with_wheelnext() -> None:
    workflow_names = ("ci.yml", "docs.yml", "tensor-runtime-ci.yml")
    workflows = [ROOT / ".github/workflows" / name for name in workflow_names]
    assert all(workflow.is_file() for workflow in workflows)
    for workflow in workflows:
        lines = workflow.read_text(encoding="utf-8").splitlines()
        setup_indexes = [index for index, line in enumerate(lines) if "astral-sh/setup-uv@" in line]
        wheel_indexes = [
            index for index, line in enumerate(lines) if "./.github/actions/wheelnext-uv" in line
        ]
        assert len(setup_indexes) == len(wheel_indexes)
        for position, setup_index in enumerate(setup_indexes):
            next_setup = (
                setup_indexes[position + 1] if position + 1 < len(setup_indexes) else len(lines)
            )
            assert setup_index < wheel_indexes[position] < next_setup


def test_wheelnext_workflows_do_not_mix_stable_uv_cache_hooks() -> None:
    import yaml

    workflow_names = ("ci.yml", "docs.yml", "tensor-runtime-ci.yml")
    for workflow_name in workflow_names:
        workflow = yaml.safe_load(
            (ROOT / ".github/workflows" / workflow_name).read_text(encoding="utf-8")
        )
        for job in workflow["jobs"].values():
            steps = job.get("steps", [])
            for position, step in enumerate(steps):
                if step.get("uses", "").startswith("astral-sh/setup-uv@"):
                    next_step = steps[position + 1]
                    if next_step.get("uses") == "./.github/actions/wheelnext-uv":
                        assert step.get("with", {}).get("enable-cache") is False, (
                            workflow_name,
                            job,
                            step,
                        )
