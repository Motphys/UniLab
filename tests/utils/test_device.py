from __future__ import annotations

from types import SimpleNamespace

import pytest

import unilab.utils.device as device_mod


def _patch_devices(
    monkeypatch: pytest.MonkeyPatch,
    *,
    cuda: bool = False,
    cuda_count: int = 0,
    xpu: bool = False,
    xpu_count: int = 0,
    mps: bool = False,
) -> None:
    monkeypatch.setattr(device_mod.torch.cuda, "is_available", lambda: cuda)
    monkeypatch.setattr(device_mod.torch.cuda, "device_count", lambda: cuda_count)
    monkeypatch.setattr(
        device_mod.torch,
        "xpu",
        SimpleNamespace(
            is_available=lambda: xpu,
            device_count=lambda: xpu_count,
        ),
        raising=False,
    )
    monkeypatch.setattr(device_mod.torch.backends.mps, "is_available", lambda: mps)


def test_resolve_torch_device_alias_defaults_to_cpu(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_devices(monkeypatch)

    assert device_mod.resolve_torch_device_alias(None) == "cpu"
    assert device_mod.resolve_torch_device_alias("cpu") == "cpu"


def test_resolve_torch_device_alias_gpu_prefers_cuda(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_devices(monkeypatch, cuda=True, cuda_count=2, mps=True)

    assert device_mod.resolve_torch_device_alias("gpu") == "cuda"
    assert device_mod.resolve_torch_device_alias("gpu:1") == "cuda:1"


def test_resolve_torch_device_alias_gpu_uses_xpu_before_mps(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_devices(monkeypatch, xpu=True, xpu_count=2, mps=True)

    assert device_mod.resolve_torch_device_alias("gpu") == "xpu"
    assert device_mod.resolve_torch_device_alias("gpu:1") == "xpu:1"


@pytest.mark.parametrize("alias", ["gpu", "gpu:0", "cuda", "cuda:0"])
def test_resolve_torch_device_alias_macos_compat_maps_gpu_and_cuda_to_mps(
    monkeypatch: pytest.MonkeyPatch,
    alias: str,
) -> None:
    _patch_devices(monkeypatch, mps=True)

    assert device_mod.resolve_torch_device_alias(alias) == "mps"


@pytest.mark.parametrize("alias", ["gpu:1", "cuda:1", "mps:1"])
def test_resolve_torch_device_alias_mps_rejects_nonzero_index(
    monkeypatch: pytest.MonkeyPatch,
    alias: str,
) -> None:
    _patch_devices(monkeypatch, mps=True)

    with pytest.raises(ValueError, match="MPS"):
        device_mod.resolve_torch_device_alias(alias)


def test_resolve_torch_device_alias_rejects_missing_cuda_index(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_devices(monkeypatch, cuda=True, cuda_count=1)

    with pytest.raises(ValueError, match="only 1 cuda device"):
        device_mod.resolve_torch_device_alias("gpu:1")


def test_resolve_torch_device_alias_rejects_unavailable_accelerator(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_devices(monkeypatch)

    with pytest.raises(ValueError, match="none is available"):
        device_mod.resolve_torch_device_alias("gpu")


def test_linux_device_info_reads_amd_visible_vram(monkeypatch: pytest.MonkeyPatch) -> None:
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

    monkeypatch.setattr(device_mod, "open", fake_open, raising=False)
    monkeypatch.setattr(device_mod.subprocess, "check_output", fake_check_output)

    info = device_mod._get_device_info_linux()

    assert info["gpu_name"] == "Radeon 8060S"
    assert info["gpu_memory"] == "98304 MB"
    assert info["gpu_gtt_memory"] == "15862 MB"
    assert info["memory"] == "31.0 GB"


def test_linux_device_info_reads_intel_igpu_from_lspci(monkeypatch: pytest.MonkeyPatch) -> None:
    cpuinfo = "model name\t: Intel(R) Core(TM) Ultra 9 185H\nprocessor\t: 0\n"
    meminfo = "MemTotal:       31692928 kB\n"
    lspci_out = (
        "00:02.0 VGA compatible controller: "
        "Intel Corporation Meteor Lake-P [Intel Arc Graphics] (rev 08)\n"
    )

    def fake_open(path, *args, **kwargs):
        del args, kwargs
        from io import StringIO

        if path == "/proc/cpuinfo":
            return StringIO(cpuinfo)
        if path == "/proc/meminfo":
            return StringIO(meminfo)
        raise FileNotFoundError(path)

    def fake_check_output(cmd, *args, **kwargs):
        del args, kwargs
        if cmd[0] == "lspci":
            return lspci_out
        # No nvidia-smi, rocm-smi, or amd-smi on Intel iGPU systems
        raise FileNotFoundError(cmd[0])

    monkeypatch.setattr(device_mod, "open", fake_open, raising=False)
    monkeypatch.setattr(device_mod.subprocess, "check_output", fake_check_output)

    info = device_mod._get_device_info_linux()

    assert info["gpu_name"] == "Intel Arc Graphics"
    assert info["chip"] == "Intel(R) Core(TM) Ultra 9 185H"
    assert info["memory"] == "30.2 GB"


def test_get_device_info_dict_reports_core_fields() -> None:
    info = device_mod.get_device_info_dict()
    assert info["platform"]
    assert "chip" in info
    assert "cpu_total_cores" in info
    assert "memory" in info
    assert "gpu_name" in info or "gpu_cores" in info
