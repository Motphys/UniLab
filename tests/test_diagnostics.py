from __future__ import annotations

import subprocess
from collections.abc import Sequence
from pathlib import Path
from types import SimpleNamespace

from unilab import diagnostics


def _torch_module(
    *,
    hip: str | None,
    cuda_available: bool = True,
    device_count: int | None = 8,
) -> SimpleNamespace:
    def visible_device_count() -> int:
        if device_count is None:
            raise RuntimeError("device probe unavailable")
        return device_count

    return SimpleNamespace(
        __version__="2.10.0-test",
        version=SimpleNamespace(cuda="13.0", hip=hip),
        cuda=SimpleNamespace(
            is_available=lambda: cuda_available,
            device_count=visible_device_count,
        ),
    )


def _completed_process(returncode: int = 0) -> subprocess.CompletedProcess[bytes]:
    return subprocess.CompletedProcess(args=[], returncode=returncode)


def test_amd_gpu_detection_combines_kfd_and_management_tools(monkeypatch) -> None:
    monkeypatch.setattr(diagnostics, "KFD_PATH", Path("/does/not/exist"))
    commands: list[Sequence[str]] = []

    def run_command(command: Sequence[str], **kwargs: object) -> subprocess.CompletedProcess[bytes]:
        commands.append(command)
        return _completed_process(0 if command[0] == "rocm-smi" else 1)

    result = diagnostics.detect_amd_gpu(run_command)

    assert result.detected
    assert result.evidence == ("rocm-smi --showproductname",)
    assert commands == [command for command in diagnostics.AMD_GPU_COMMANDS]


def test_reports_wrong_torch_build_on_amd_host_with_repair_commands() -> None:
    result = diagnostics.check_accelerator(
        _torch_module(hip=None),
        diagnostics.AmdGpu(detected=True, evidence=("/dev/kfd",)),
        diagnostics.Installer(version="uv-wheelnext 0.9.14 (test)"),
    )

    assert not result.ok
    assert "ERROR: an AMD GPU is present, but Torch is not a ROCm/HIP build." in result.lines
    assert "Required uv: uv-wheelnext 0.9.14" in result.lines
    assert "  uv sync --extra mujoco --reinstall-package torch" in result.lines
    assert "  uv pip install --reinstall torch" in result.lines


def test_accepts_matching_rocm_torch_and_amd_host() -> None:
    result = diagnostics.check_accelerator(
        _torch_module(hip="7.2.53211"),
        diagnostics.AmdGpu(detected=True, evidence=("/dev/kfd", "rocm-smi --showproductname")),
        diagnostics.Installer(version="uv-wheelnext 0.9.14 (test)"),
    )

    assert result.ok
    assert "ROCm/HIP Torch and AMD GPU detection agree." in result.lines


def test_reports_rocm_torch_without_detectable_amd_interface() -> None:
    result = diagnostics.check_accelerator(
        _torch_module(hip="7.2.53211"),
        diagnostics.AmdGpu(detected=False, evidence=()),
        diagnostics.Installer(version="uv-wheelnext 0.9.14 (test)"),
    )

    assert not result.ok
    assert "no AMD GPU interface was detected" in "\n".join(result.lines)


def test_reports_rocm_torch_when_driver_or_visibility_is_unavailable() -> None:
    result = diagnostics.check_accelerator(
        _torch_module(hip="7.2.53211", cuda_available=False, device_count=0),
        diagnostics.AmdGpu(detected=True, evidence=("/dev/kfd",)),
        diagnostics.Installer(version="uv-wheelnext 0.9.14 (test)"),
    )

    assert not result.ok
    assert "reports no available CUDA device" in "\n".join(result.lines)


def test_reports_rocm_torch_when_device_count_probe_fails() -> None:
    result = diagnostics.check_accelerator(
        _torch_module(hip="7.2.53211", device_count=None),
        diagnostics.AmdGpu(detected=True, evidence=("/dev/kfd",)),
        diagnostics.Installer(version="uv-wheelnext 0.9.14 (test)"),
    )

    assert not result.ok
    assert "reports no available CUDA device" in "\n".join(result.lines)


def test_non_amd_installations_do_not_fail_without_rocm_torch() -> None:
    result = diagnostics.check_accelerator(
        _torch_module(hip=None, cuda_available=False, device_count=0),
        diagnostics.AmdGpu(detected=False, evidence=()),
        diagnostics.Installer(version="uv-wheelnext 0.9.14 (test)"),
    )

    assert result.ok
    assert "No AMD ROCm/Torch mismatch was detected." in result.lines


def test_unilab_doctor_command_prints_result_and_exit_status(monkeypatch, capsys) -> None:
    diagnostic = diagnostics.AcceleratorDiagnostic(ok=False, lines=("diagnostic output",))
    monkeypatch.setattr(
        diagnostics,
        "check_accelerator",
        lambda torch_module, amd_gpu=None: diagnostic,
    )

    exit_code = diagnostics.main(["doctor"])

    assert exit_code == 1
    assert capsys.readouterr().out == "diagnostic output\n"
