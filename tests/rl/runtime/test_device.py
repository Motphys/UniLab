"""Torch device runtime primitives for the RL runtime (owner: rl.runtime.device).

Merged from unilab_rl @ 3654e4b (unilab-rl 1.4.15), previously
``uni_rl.utils.device``.  Host hardware-info collection is owned by
:mod:`unilab.utils.device` and covered by ``tests/utils/test_device.py``;
backend process-device routing is owned by :mod:`unilab.base.process_device`
and covered by ``tests/base/backend/test_process_device.py``.
"""

from __future__ import annotations

import pytest
import torch

import unilab.rl.runtime.device as device_mod


def test_sync_device_uses_one_cuda_stream_api(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[str, torch.device]] = []

    class FakeStream:
        def synchronize(self) -> None:
            calls.append(("stream", torch.device("cuda:0")))

    monkeypatch.setattr(
        device_mod.torch.cuda,
        "current_stream",
        lambda device: FakeStream(),
    )
    device_mod.sync_device("cuda:0", stream=True)
    assert calls == [("stream", torch.device("cuda:0"))]


def test_device_runtime_capabilities_report_sync_and_timing_policy() -> None:
    assert device_mod.device_runtime_capabilities("cpu").manifest() == {
        "device_type": "cpu",
        "sync_api": "none",
        "stream_sync_api": None,
        "has_timing_events": False,
        "sync_policy": "capability",
        "timing_policy": "capability",
    }
    cuda = device_mod.device_runtime_capabilities("cuda")
    assert cuda.sync_api == "torch.cuda.synchronize"
    assert cuda.stream_sync_api == "torch.cuda.current_stream.synchronize"
    assert cuda.has_timing_events is True


def test_device_runtime_capabilities_fail_closed_for_unknown_devices() -> None:
    with pytest.raises((ValueError, RuntimeError), match="tpu"):
        device_mod.device_runtime_capabilities("tpu")


def test_synchronized_device_timer_uses_wall_clock_boundary() -> None:
    timer = device_mod.SynchronizedDeviceTimer("cpu")
    timer.start()
    timer.stop()
    assert timer.elapsed_s() >= 0.0
