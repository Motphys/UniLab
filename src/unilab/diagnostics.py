#!/usr/bin/env python3
"""Diagnose AMD/ROCm Torch build and host interface agreement."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

CommandRunner = Callable[..., subprocess.CompletedProcess[bytes]]
KFD_PATH = Path("/dev/kfd")
AMD_GPU_COMMANDS = (
    ("rocm-smi", "--showproductname"),
    ("amd-smi", "static"),
)
WHEELNEXT_UV_VERSION = "0.9.14"


@dataclass(frozen=True)
class AmdGpu:
    detected: bool
    evidence: tuple[str, ...]


@dataclass(frozen=True)
class Installer:
    version: str


@dataclass(frozen=True)
class AcceleratorDiagnostic:
    ok: bool
    lines: tuple[str, ...]


def detect_amd_gpu(run_command: CommandRunner = subprocess.run) -> AmdGpu:
    """Detect an AMD GPU using stable kernel and ROCm management interfaces."""
    evidence: list[str] = []
    if KFD_PATH.exists():
        evidence.append("/dev/kfd")

    for command in AMD_GPU_COMMANDS:
        try:
            result = run_command(
                command,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
                timeout=5,
            )
        except (OSError, subprocess.TimeoutExpired):
            continue
        if result.returncode == 0:
            evidence.append(" ".join(command))

    return AmdGpu(detected=bool(evidence), evidence=tuple(evidence))


def detect_uv(run_command: CommandRunner = subprocess.run) -> Installer:
    """Report the uv version that will perform any suggested repair."""
    executable = os.environ.get("UV") or shutil.which("uv")
    if executable is None:
        return Installer(version="unavailable")
    try:
        result = run_command(
            (executable, "--version"),
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            check=False,
            timeout=5,
        )
    except (OSError, subprocess.TimeoutExpired):
        return Installer(version="unavailable")
    return Installer(version=result.stdout.decode(errors="replace").strip())


def _torch_property(torch_module: Any, path: str) -> str:
    value: Any = torch_module
    for component in path.split("."):
        value = getattr(value, component, None)
    if callable(value):
        try:
            value = value()
        except Exception:
            return "unavailable"
    if value is None:
        return "None"
    return str(value)


def check_accelerator(
    torch_module: Any,
    amd_gpu: AmdGpu | None = None,
    installer: Installer | None = None,
) -> AcceleratorDiagnostic:
    """Check that a Torch binary build matches detectable AMD hardware."""
    selected_amd_gpu = detect_amd_gpu() if amd_gpu is None else amd_gpu
    selected_installer = detect_uv() if installer is None else installer
    hip = _torch_property(torch_module, "version.hip")
    cuda_available = _torch_property(torch_module, "cuda.is_available")
    device_count = _torch_property(torch_module, "cuda.device_count")
    lines = [
        f"Torch: {_torch_property(torch_module, '__version__')}",
        f"Torch CUDA version: {_torch_property(torch_module, 'version.cuda')}",
        f"Torch HIP version: {hip}",
        f"AMD GPU detected: {'yes' if selected_amd_gpu.detected else 'no'}",
        f"uv: {selected_installer.version}",
    ]
    if selected_amd_gpu.evidence:
        lines.append(f"AMD GPU evidence: {', '.join(selected_amd_gpu.evidence)}")

    if selected_amd_gpu.detected and hip == "None":
        lines.extend(
            [
                "ERROR: an AMD GPU is present, but Torch is not a ROCm/HIP build.",
                f"Required uv: uv-wheelnext {WHEELNEXT_UV_VERSION}",
                "Repair a source checkout with:",
                "  uv sync --extra mujoco --reinstall-package torch",
                "Repair an installed distribution with:",
                "  uv pip install --reinstall torch",
            ]
        )
        return AcceleratorDiagnostic(ok=False, lines=tuple(lines))

    if hip != "None":
        if not selected_amd_gpu.detected:
            lines.extend(
                [
                    "ERROR: Torch is a ROCm/HIP build, but no AMD GPU interface was detected.",
                    "Check that the process can access /dev/kfd and that rocm-smi or amd-smi works.",
                ]
            )
            return AcceleratorDiagnostic(ok=False, lines=tuple(lines))
        try:
            available_device_count = int(device_count)
        except ValueError:
            available_device_count = -1
        if cuda_available != "True" or available_device_count <= 0:
            lines.extend(
                [
                    "ERROR: Torch is a ROCm/HIP build but reports no available CUDA device.",
                    "Check the kernel driver, ROCm userspace, /dev/kfd permissions, and GPU visibility.",
                ]
            )
            return AcceleratorDiagnostic(ok=False, lines=tuple(lines))
        lines.append("ROCm/HIP Torch and AMD GPU detection agree.")
        return AcceleratorDiagnostic(ok=True, lines=tuple(lines))

    lines.append("No AMD ROCm/Torch mismatch was detected.")
    return AcceleratorDiagnostic(ok=True, lines=tuple(lines))


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="unilab", description=__doc__)
    parser.add_argument("command", choices=("doctor",), help="diagnostic command to run")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    _parse_args(argv)
    diagnostic = check_accelerator(torch)
    print("\n".join(diagnostic.lines))
    return 0 if diagnostic.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
