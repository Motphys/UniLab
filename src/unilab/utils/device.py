from __future__ import annotations

import platform
import re
import subprocess
from functools import lru_cache
from typing import Callable, cast

import torch


def _xpu_available() -> bool:
    xpu = getattr(torch, "xpu", None)
    is_available = getattr(xpu, "is_available", None)
    return bool(callable(is_available) and is_available())


def get_default_device() -> str:
    """Detect the best available device."""
    if torch.cuda.is_available():
        return "cuda"
    if _xpu_available():
        return "xpu"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _device_count(device_type: str) -> int | None:
    if device_type == "cuda":
        return int(torch.cuda.device_count())
    if device_type == "xpu":
        xpu = getattr(torch, "xpu", None)
        device_count = getattr(xpu, "device_count", None)
        if callable(device_count):
            return int(cast(Callable[[], int], device_count)())
    return None


def _mps_available() -> bool:
    mps = getattr(torch.backends, "mps", None)
    is_available = getattr(mps, "is_available", None)
    return bool(callable(is_available) and is_available())


def _parse_device_alias(value: str) -> tuple[str, int | None]:
    raw = value.strip().lower()
    if not raw:
        raise ValueError("Device alias must not be empty")
    if ":" not in raw:
        return raw, None
    base, index_text = raw.split(":", 1)
    if not index_text:
        raise ValueError(f"Device alias {value!r} has an empty index")
    try:
        index = int(index_text)
    except ValueError as exc:
        raise ValueError(f"Device alias {value!r} has a non-integer index") from exc
    if index < 0:
        raise ValueError(f"Device alias {value!r} has a negative index")
    return base, index


def _resolve_indexed_device(device_type: str, index: int | None, original: str) -> str:
    count = _device_count(device_type)
    if count is not None and index is not None and index >= count:
        raise ValueError(
            f"Requested device {original!r} resolves to {device_type}:{index}, "
            f"but only {count} {device_type} device(s) are available"
        )
    return device_type if index is None else f"{device_type}:{index}"


def _resolve_mps_alias(index: int | None, original: str) -> str:
    if not _mps_available():
        raise ValueError(f"Requested device {original!r} requires MPS, but MPS is unavailable")
    if index not in (None, 0):
        raise ValueError(
            f"Requested device {original!r} cannot be mapped to MPS; only index 0 is valid"
        )
    return "mps"


def resolve_torch_device_alias(device: str | None, *, default: str = "cpu") -> str:
    """Resolve a cross-platform torch device alias to a concrete device string.

    ``gpu`` is an abstract accelerator alias. ``cuda`` is also accepted on
    macOS/MPS for config portability and maps to ``mps`` when CUDA is absent.
    The function validates the resolved device and never silently falls back to
    CPU for unavailable accelerators.
    """
    original = default if device is None else str(device)
    base, index = _parse_device_alias(original)

    if base == "cpu":
        if index is not None:
            raise ValueError(f"CPU device {original!r} must not include an index")
        return "cpu"

    if base == "mps":
        return _resolve_mps_alias(index, original)

    if base == "xpu":
        if not _xpu_available():
            raise ValueError(f"Requested device {original!r} requires XPU, but XPU is unavailable")
        return _resolve_indexed_device("xpu", index, original)

    if base == "cuda":
        if torch.cuda.is_available():
            return _resolve_indexed_device("cuda", index, original)
        if _mps_available():
            return _resolve_mps_alias(index, original)
        raise ValueError(f"Requested device {original!r} requires CUDA, but CUDA is unavailable")

    if base == "gpu":
        if torch.cuda.is_available():
            return _resolve_indexed_device("cuda", index, original)
        if _xpu_available():
            return _resolve_indexed_device("xpu", index, original)
        if _mps_available():
            return _resolve_mps_alias(index, original)
        raise ValueError(
            f"Requested device {original!r} requires an accelerator, but none is available"
        )

    raise ValueError(f"Unsupported device alias {original!r}; expected cpu, gpu, cuda, mps, or xpu")


def _is_macos() -> bool:
    return platform.system() == "Darwin"


def _is_linux() -> bool:
    return platform.system() == "Linux"


def _is_windows() -> bool:
    return platform.system() == "Windows"


def _get_device_info_macos() -> dict[str, str]:
    """Collect hardware info on macOS via system_profiler."""
    info: dict[str, str] = {
        "chip": "unknown",
        "cpu_total_cores": "unknown",
        "cpu_performance_cores": "unknown",
        "cpu_efficiency_cores": "unknown",
        "gpu_cores": "unknown",
        "memory": "unknown",
    }
    try:
        hw_text = subprocess.check_output(
            ["system_profiler", "SPHardwareDataType"], text=True, stderr=subprocess.DEVNULL
        )
        disp_text = subprocess.check_output(
            ["system_profiler", "SPDisplaysDataType"], text=True, stderr=subprocess.DEVNULL
        )
    except Exception:
        return info

    chip_match = re.search(r"Chip:\s*(.+)", hw_text)
    if chip_match:
        info["chip"] = chip_match.group(1).strip()

    mem_match = re.search(r"Memory:\s*(.+)", hw_text)
    if mem_match:
        info["memory"] = mem_match.group(1).strip()

    # Apple Silicon core descriptions vary by generation:
    #   M3/M4: "10 performance and 4 efficiency"
    #   M5 Pro/Max: "6 super and 12 performance"
    cpu_match = re.search(
        r"Total Number of Cores:\s*(\d+)\s*\(\s*(\d+)\s*(\w+)\s+and\s+(\d+)\s*(\w+)\s*\)",
        hw_text,
    )
    if cpu_match:
        total, count1, type1, count2, type2 = cpu_match.groups()
        info["cpu_total_cores"] = total
        info["cpu_core_type_1"] = type1
        info["cpu_core_count_1"] = count1
        info["cpu_core_type_2"] = type2
        info["cpu_core_count_2"] = count2
        # Backward-compat keys for legacy P+E format
        if type1 == "performance" and type2 == "efficiency":
            info["cpu_performance_cores"] = count1
            info["cpu_efficiency_cores"] = count2
        elif type1 == "super" and type2 == "performance":
            info["cpu_super_cores"] = count1
            info["cpu_performance_cores"] = count2
    else:
        cpu_total_match = re.search(r"Total Number of Cores:\s*(\d+)", hw_text)
        if cpu_total_match:
            info["cpu_total_cores"] = cpu_total_match.group(1)

    gpu_match = re.search(r"Type:\s*GPU[\s\S]*?Total Number of Cores:\s*(\d+)", disp_text)
    if gpu_match:
        info["gpu_cores"] = gpu_match.group(1)

    return info


def _get_device_info_linux() -> dict[str, str]:
    """Collect hardware info on Linux via /proc and common CLI tools."""
    info: dict[str, str] = {
        "chip": "unknown",
        "cpu_total_cores": "unknown",
        "cpu_performance_cores": "unknown",
        "cpu_efficiency_cores": "unknown",
        "gpu_name": "unknown",
        "memory": "unknown",
    }
    # CPU model
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as f:
            cpuinfo = f.read()
        model_match = re.search(r"^model name\s*:\s*(.+)$", cpuinfo, re.MULTILINE)
        if model_match:
            info["chip"] = model_match.group(1).strip()
        # Count physical cores (unique core id per physical id)
        pairs = re.findall(r"physical id\s*:\s*(\d+).*?core id\s*:\s*(\d+)", cpuinfo, re.DOTALL)
        if pairs:
            info["cpu_total_cores"] = str(len(set(pairs)))
        else:
            processor_count = len(re.findall(r"^processor\s*:", cpuinfo, re.MULTILINE))
            if processor_count:
                info["cpu_total_cores"] = str(processor_count)
    except Exception:
        pass
    # Total memory
    try:
        with open("/proc/meminfo", encoding="utf-8") as f:
            meminfo = f.read()
        mem_match = re.search(r"MemTotal:\s*(\d+)\s*kB", meminfo)
        if mem_match:
            mem_gb = int(mem_match.group(1)) / 1024 / 1024
            info["memory"] = f"{mem_gb:.1f} GB"
    except Exception:
        pass
    # GPU via nvidia-smi
    try:
        gpu_out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        if gpu_out:
            # Take the first GPU line
            first_line = gpu_out.splitlines()[0]
            parts = [p.strip() for p in first_line.split(",")]
            info["gpu_name"] = parts[0]
            if len(parts) > 1:
                info["gpu_memory"] = parts[1]
    except Exception:
        pass
    # GPU via rocm-smi (AMD)
    if info["gpu_name"] == "unknown":
        try:
            rocm_out = subprocess.check_output(
                ["rocm-smi", "--showproductname"],
                text=True,
                stderr=subprocess.DEVNULL,
            )
            gpu_match = re.search(r"Card series\s*:\s*(.+)", rocm_out, re.IGNORECASE)
            if gpu_match:
                info["gpu_name"] = gpu_match.group(1).strip()
        except Exception:
            pass
    # GPU memory via amd-smi (AMD ROCm). On unified-memory APUs this reports
    # the BIOS-allocated visible VRAM slice, e.g. 96 GB out of 128 GB.
    try:
        amd_smi_out = subprocess.check_output(
            ["amd-smi", "metric"],
            text=True,
            stderr=subprocess.DEVNULL,
        )
        vram_match = re.search(r"TOTAL_VISIBLE_VRAM:\s*(\d+)\s*MB", amd_smi_out)
        if vram_match:
            info["gpu_memory"] = f"{int(vram_match.group(1))} MB"
        gtt_match = re.search(r"TOTAL_GTT:\s*(\d+)\s*MB", amd_smi_out)
        if gtt_match:
            info["gpu_gtt_memory"] = f"{int(gtt_match.group(1))} MB"
    except Exception:
        pass
    # Fallback GPU via lspci (AMD/ATI, Intel iGPU/Arc, and others)
    if info["gpu_name"] == "unknown":
        try:
            lspci_out = subprocess.check_output(["lspci"], text=True, stderr=subprocess.DEVNULL)
            for line in lspci_out.splitlines():
                if "VGA" in line or "Display" in line or "3D" in line:
                    if "AMD" in line or "ATI" in line:
                        match = re.search(r"\[AMD/ATI\]\s*(.+)", line)
                        if match:
                            name = match.group(1).strip()
                            name = re.sub(r"\s*\(rev.*\)", "", name)
                            info["gpu_name"] = name
                            break
                    elif "Intel" in line:
                        # e.g. "Intel Corporation Meteor Lake-P [Intel Arc Graphics] (rev 08)"
                        match = re.search(r"\[([^\]]+)\]", line)
                        if match:
                            info["gpu_name"] = match.group(1).strip()
                            break
        except Exception:
            pass
    # If GPU name is still generic/unknown, try to infer from CPU model (APUs)
    if info["gpu_name"] in ("unknown", "AMD Radeon Graphics"):
        chip = info.get("chip", "")
        match = re.search(r"w(?:ith)?/\s*(Radeon\s+[\w\s\+]+)", chip, re.IGNORECASE)
        if match:
            info["gpu_name"] = match.group(1).strip()
    return info


def _get_device_info_windows() -> dict[str, str]:
    """Collect hardware info on Windows via wmic."""
    info: dict[str, str] = {
        "chip": "unknown",
        "cpu_total_cores": "unknown",
        "cpu_performance_cores": "unknown",
        "cpu_efficiency_cores": "unknown",
        "gpu_name": "unknown",
        "memory": "unknown",
    }
    try:
        cpu_out = subprocess.check_output(
            ["wmic", "cpu", "get", "Name,NumberOfCores", "/format:csv"],
            text=True,
            stderr=subprocess.DEVNULL,
        )
        lines = [l for l in cpu_out.splitlines() if l.strip() and not l.strip().startswith("Node")]
        if lines:
            parts = lines[0].split(",")
            if len(parts) >= 3:
                info["cpu_total_cores"] = parts[1].strip()
                info["chip"] = parts[2].strip()
    except Exception:
        pass
    # Memory
    try:
        mem_out = subprocess.check_output(
            ["wmic", "ComputerSystem", "get", "TotalPhysicalMemory", "/format:csv"],
            text=True,
            stderr=subprocess.DEVNULL,
        )
        lines = [l for l in mem_out.splitlines() if l.strip() and not l.strip().startswith("Node")]
        if lines:
            parts = lines[0].split(",")
            if len(parts) >= 2:
                mem_gb = int(parts[1].strip()) / 1024**3
                info["memory"] = f"{mem_gb:.1f} GB"
    except Exception:
        pass
    # GPU via nvidia-smi (also available on Windows)
    try:
        gpu_out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        if gpu_out:
            first_line = gpu_out.splitlines()[0]
            parts = [p.strip() for p in first_line.split(",")]
            info["gpu_name"] = parts[0]
            if len(parts) > 1:
                info["gpu_memory"] = parts[1]
    except Exception:
        pass
    return info


@lru_cache(maxsize=1)
def get_device_info_dict() -> dict[str, str]:
    """Collect static hardware metadata (chip, cores, GPU, memory) for run records."""
    base: dict[str, str] = {"platform": platform.platform()}
    if _is_macos():
        base.update(_get_device_info_macos())
    elif _is_linux():
        base.update(_get_device_info_linux())
    elif _is_windows():
        base.update(_get_device_info_windows())
    return base


def get_device_info_line() -> str:
    d = get_device_info_dict()
    if _is_macos():
        # Build core-type summary dynamically so M5 (super+performance) is shown correctly
        if d.get("cpu_core_type_1") and d.get("cpu_core_type_2"):
            t1 = d["cpu_core_type_1"][0].upper()
            t2 = d["cpu_core_type_2"][0].upper()
            core_summary = f"{d['cpu_core_count_1']}{t1}+{d['cpu_core_count_2']}{t2}"
        elif (
            d.get("cpu_performance_cores") != "unknown"
            and d.get("cpu_efficiency_cores") != "unknown"
        ):
            core_summary = f"{d['cpu_performance_cores']}P+{d['cpu_efficiency_cores']}E"
        else:
            core_summary = "unknown"
        return (
            f"Device: {d.get('chip', 'unknown')} | "
            f"CPU: {d.get('cpu_total_cores', 'unknown')} cores "
            f"({core_summary}) | "
            f"GPU: {d.get('gpu_cores', 'unknown')} cores | "
            f"Memory: {d.get('memory', 'unknown')}"
        )
    else:
        gpu_part = d.get("gpu_name", "unknown")
        if "gpu_memory" in d:
            gpu_part += f" ({d['gpu_memory']})"
        return (
            f"CPU: {d.get('chip', 'unknown')} ({d.get('cpu_total_cores', 'unknown')} cores) | "
            f"GPU: {gpu_part} | "
            f"Memory: {d.get('memory', 'unknown')}"
        )
