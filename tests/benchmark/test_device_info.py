from __future__ import annotations

import benchmark.core.device_info as device_info


def test_linux_device_info_reads_amd_visible_vram(monkeypatch):
    cpuinfo = "model name\t: AMD RYZEN AI MAX+ 395 w/ Radeon 8060S\nprocessor\t: 0\n"
    meminfo = "MemTotal:       32486180 kB\n"
    amd_smi_metric = """
GPU: 0
    MEM_USAGE:
        TOTAL_VRAM: 98304 MB
        USED_VRAM: 3603 MB
        FREE_VRAM: 94701 MB
        TOTAL_VISIBLE_VRAM: 98304 MB
        USED_VISIBLE_VRAM: 3603 MB
        FREE_VISIBLE_VRAM: 94701 MB
        TOTAL_GTT: 15862 MB
        USED_GTT: 158 MB
        FREE_GTT: 15704 MB
"""

    def fake_open(path, *args, **kwargs):
        del args, kwargs
        if path == "/proc/cpuinfo":
            from io import StringIO

            return StringIO(cpuinfo)
        if path == "/proc/meminfo":
            from io import StringIO

            return StringIO(meminfo)
        raise FileNotFoundError(path)

    def fake_check_output(cmd, *args, **kwargs):
        del args, kwargs
        if cmd[0] == "nvidia-smi":
            raise FileNotFoundError(cmd[0])
        if cmd[:2] == ["rocm-smi", "--showproductname"]:
            return "Card series: AMD Radeon Graphics\n"
        if cmd[:2] == ["amd-smi", "metric"]:
            return amd_smi_metric
        raise FileNotFoundError(cmd[0])

    monkeypatch.setattr(device_info, "open", fake_open, raising=False)
    monkeypatch.setattr(device_info.subprocess, "check_output", fake_check_output)

    info = device_info._get_device_info_linux()

    assert info["gpu_name"] == "Radeon 8060S"
    assert info["gpu_memory"] == "98304 MB"
    assert info["gpu_gtt_memory"] == "15862 MB"
    assert info["memory"] == "31.0 GB"
