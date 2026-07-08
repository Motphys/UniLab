from __future__ import annotations

import os

import pytest

from unilab.algos.torch.offpolicy.thread_budget import (
    apply_torch_thread_runtime,
    format_torch_thread_runtime,
    resolve_torch_thread_runtime,
    torch_thread_env,
)


class _FakeTorch:
    def __init__(self) -> None:
        self.num_threads = 0
        self.num_interop_threads = 96
        self.inductor_compile_threads = 0
        self._inductor = type(
            "_FakeInductor",
            (),
            {"config": type("_FakeInductorConfig", (), {"compile_threads": 0})()},
        )()

    def set_num_threads(self, value: int) -> None:
        self.num_threads = int(value)

    def get_num_interop_threads(self) -> int:
        return self.num_interop_threads

    def set_num_interop_threads(self, value: int) -> None:
        self.num_interop_threads = int(value)


def test_resolve_torch_thread_runtime_auto_caps_many_core_host() -> None:
    runtime = resolve_torch_thread_runtime(
        {
            "enabled": True,
            "learner_num_threads": "auto",
            "collector_num_threads": "auto",
            "learner_num_interop_threads": 1,
            "collector_num_interop_threads": 1,
            "compile_threads": "auto",
        },
        cpu_count=160,
    )

    assert runtime["learner"]["num_threads"] == 8
    assert runtime["collector"]["num_threads"] == 4
    assert runtime["compile_threads"] == 2
    assert runtime["learner"]["num_interop_threads"] == 1
    assert runtime["collector"]["num_interop_threads"] == 1


def test_apply_torch_thread_runtime_sets_role_env_and_torch(monkeypatch: pytest.MonkeyPatch):
    for key in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "TORCH_NUM_THREADS",
        "TORCHINDUCTOR_COMPILE_THREADS",
    ):
        monkeypatch.delenv(key, raising=False)

    fake_torch = _FakeTorch()
    runtime = resolve_torch_thread_runtime(
        {
            "learner_num_threads": 6,
            "collector_num_threads": 3,
            "learner_num_interop_threads": 2,
            "collector_num_interop_threads": 1,
            "compile_threads": 2,
            "set_env_vars": True,
        },
        cpu_count=64,
    )

    applied = apply_torch_thread_runtime(runtime, role="collector", torch_module=fake_torch)

    assert applied == {
        "enabled": True,
        "role": "collector",
        "num_threads": 3,
        "num_interop_threads": 1,
        "compile_threads": 2,
    }
    assert fake_torch.num_threads == 3
    assert fake_torch.num_interop_threads == 1
    assert fake_torch._inductor.config.compile_threads == 2
    assert os.environ["TORCH_NUM_THREADS"] == "3"
    assert os.environ["TORCHINDUCTOR_COMPILE_THREADS"] == "2"


def test_torch_thread_env_temporarily_sets_child_startup_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TORCH_NUM_THREADS", "6")
    monkeypatch.setenv("OMP_NUM_THREADS", "6")
    monkeypatch.delenv("TORCHINDUCTOR_COMPILE_THREADS", raising=False)
    runtime = resolve_torch_thread_runtime(
        {
            "learner_num_threads": 6,
            "collector_num_threads": 3,
            "compile_threads": 2,
            "set_env_vars": True,
        },
        cpu_count=64,
    )

    with torch_thread_env(runtime, role="collector"):
        assert os.environ["TORCH_NUM_THREADS"] == "3"
        assert os.environ["OMP_NUM_THREADS"] == "3"
        assert os.environ["TORCHINDUCTOR_COMPILE_THREADS"] == "2"

    assert os.environ["TORCH_NUM_THREADS"] == "6"
    assert os.environ["OMP_NUM_THREADS"] == "6"
    assert "TORCHINDUCTOR_COMPILE_THREADS" not in os.environ


def test_disabled_torch_thread_runtime_does_not_validate_thread_fields() -> None:
    runtime = resolve_torch_thread_runtime(
        {
            "enabled": False,
            "learner_num_threads": 0,
            "collector_num_threads": 0,
        },
        cpu_count=64,
    )

    assert runtime["enabled"] is False


def test_format_torch_thread_runtime_reports_roles() -> None:
    runtime = resolve_torch_thread_runtime(
        {
            "learner_num_threads": 6,
            "collector_num_threads": 3,
            "learner_num_interop_threads": 2,
            "collector_num_interop_threads": 1,
            "compile_threads": 2,
        },
        cpu_count=64,
    )

    assert format_torch_thread_runtime(runtime) == (
        "Torch thread budget: learner=6 intra/2 inter, collector=3 intra/1 inter, compile_workers=2"
    )
