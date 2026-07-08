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
