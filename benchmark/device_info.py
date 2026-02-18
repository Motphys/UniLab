from __future__ import annotations

import platform
import re
import subprocess
from functools import lru_cache
from typing import Dict


@lru_cache(maxsize=1)
def get_device_info_dict() -> Dict[str, str]:
    info = {
        "platform": platform.platform(),
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

    cpu_match = re.search(
        r"Total Number of Cores:\s*(\d+)\s*\((\d+)\s*performance and (\d+)\s*efficiency\)",
        hw_text,
    )
    if cpu_match:
        info["cpu_total_cores"], info["cpu_performance_cores"], info["cpu_efficiency_cores"] = (
            cpu_match.groups()
        )
    else:
        cpu_total_match = re.search(r"Total Number of Cores:\s*(\d+)", hw_text)
        if cpu_total_match:
            info["cpu_total_cores"] = cpu_total_match.group(1)

    gpu_match = re.search(r"Type:\s*GPU[\s\S]*?Total Number of Cores:\s*(\d+)", disp_text)
    if gpu_match:
        info["gpu_cores"] = gpu_match.group(1)

    return info


def get_device_info_line() -> str:
    d = get_device_info_dict()
    return (
        f"Device: {d['chip']} | CPU: {d['cpu_total_cores']} cores "
        f"({d['cpu_performance_cores']}P+{d['cpu_efficiency_cores']}E) | "
        f"GPU: {d['gpu_cores']} cores | Memory: {d['memory']}"
    )
