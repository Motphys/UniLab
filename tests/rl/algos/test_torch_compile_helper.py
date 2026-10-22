from __future__ import annotations

import importlib.util
from collections.abc import Callable
from typing import Any

import torch

from unilab.rl.algos.common import compile as compile_helper


def test_hip_runtime_detection_preserves_rocm_compatibility(monkeypatch) -> None:
    monkeypatch.setattr(torch.version, "hip", "6.3", raising=False)
    assert compile_helper.is_hip_runtime() is True

    monkeypatch.setattr(torch.version, "hip", None, raising=False)
    assert compile_helper.is_hip_runtime() is False


def _without_triton(name: str, *args: Any, **kwargs: Any) -> Any:
    if name == "triton":
        return None
    return _ORIGINAL_FIND_SPEC(name, *args, **kwargs)


_ORIGINAL_FIND_SPEC = importlib.util.find_spec


def test_torch_compile_cuda_fallback_warns_when_triton_missing(monkeypatch, capsys) -> None:
    def fake_compile(fn: Callable[..., Any], **kwargs: Any) -> Callable[..., Any]:
        return fn

    fake_compile.__module__ = "torch"
    monkeypatch.setattr(torch, "compile", fake_compile)
    monkeypatch.setattr(importlib.util, "find_spec", _without_triton)
    compile_helper._WARNED_REASONS.clear()

    assert compile_helper.get_torch_compile_for_device("cuda", warn=True) is None

    stderr = capsys.readouterr().err
    assert "WARNING: torch.compile is unavailable for CUDA" in stderr
    assert "Triton is not" in stderr
    assert "installed" in stderr


def test_torch_compile_cuda_fallback_warning_is_once(monkeypatch, capsys) -> None:
    def fake_compile(fn: Callable[..., Any], **kwargs: Any) -> Callable[..., Any]:
        return fn

    fake_compile.__module__ = "torch"
    monkeypatch.setattr(torch, "compile", fake_compile)
    monkeypatch.setattr(importlib.util, "find_spec", _without_triton)
    compile_helper._WARNED_REASONS.clear()

    compile_helper.get_torch_compile_for_device("cuda", warn=True)
    compile_helper.get_torch_compile_for_device("cuda", warn=True)

    assert capsys.readouterr().err.count("torch.compile is unavailable") == 1


def test_torch_compile_cuda_returns_available_compile_without_warning(monkeypatch, capsys) -> None:
    def fake_compile(fn: Callable[..., Any], **kwargs: Any) -> Callable[..., Any]:
        return fn

    fake_compile.__module__ = "test"
    monkeypatch.setattr(torch, "compile", fake_compile)
    compile_helper._WARNED_REASONS.clear()

    assert compile_helper.get_torch_compile_for_device("cuda", warn=True) is fake_compile
    assert capsys.readouterr().err == ""


def test_torch_210_cuda_compile_disables_shape_padding(monkeypatch) -> None:
    from torch._inductor import config as inductor_config

    monkeypatch.setattr(torch, "__version__", "2.10.0+test")
    monkeypatch.setattr(inductor_config, "shape_padding", inductor_config.shape_padding)

    compile_helper.configure_torch_compile_compat()

    assert inductor_config.shape_padding is False


def test_torch_compile_device_helper_supports_mps_without_triton(monkeypatch) -> None:
    def fake_compile(fn: Callable[..., Any], **kwargs: Any) -> Callable[..., Any]:
        return fn

    fake_compile.__module__ = "torch"
    monkeypatch.setattr(torch, "compile", fake_compile)
    monkeypatch.setattr(importlib.util, "find_spec", _without_triton)

    assert compile_helper.get_torch_compile_for_device("mps", warn=True) is fake_compile


def test_torch_compile_device_helper_rejects_unsupported_devices(monkeypatch) -> None:
    monkeypatch.setattr(torch, "compile", lambda fn: fn)

    assert compile_helper.get_torch_compile_for_device("cpu", warn=True) is None
    assert compile_helper.get_torch_compile_for_device("xpu", warn=True) is None
