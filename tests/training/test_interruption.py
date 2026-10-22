from __future__ import annotations

import signal

import pytest

import unilab.interruption as interruption
from unilab.interruption import run_process_with_graceful_interrupt
from unilab.training import graceful_keyboard_interrupt


def test_graceful_keyboard_interrupt_runs_cleanup_and_exits_130(
    capsys: pytest.CaptureFixture[str],
) -> None:
    cleanup: list[str] = []

    @graceful_keyboard_interrupt
    def entrypoint() -> None:
        try:
            raise KeyboardInterrupt
        finally:
            cleanup.append("closed")

    with pytest.raises(SystemExit) as exit_info:
        entrypoint()

    assert exit_info.value.code == 130
    assert cleanup == ["closed"]
    assert "Interrupted by Ctrl+C; shutdown completed." in capsys.readouterr().err


class _FakeChildProcess:
    def __init__(self, *, interrupt_twice: bool = False) -> None:
        self.interrupt_twice = interrupt_twice
        self.returncode: int | None = None
        self.terminated = False

    def poll(self) -> int | None:
        return self.returncode

    def wait(self, timeout: float | None = None) -> int | None:
        del timeout
        handler = signal.getsignal(signal.SIGINT)
        handler(signal.SIGINT, None)
        if self.interrupt_twice:
            handler(signal.SIGINT, None)
            return None
        self.returncode = 130
        return self.returncode

    def terminate(self) -> None:
        self.terminated = True
        self.returncode = -signal.SIGTERM

    def kill(self) -> None:
        self.returncode = -signal.SIGKILL


def test_wait_for_process_gracefully_returns_child_interrupt_status(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    child = _FakeChildProcess()
    monkeypatch.setattr(interruption.subprocess, "Popen", lambda *_args, **_kwargs: child)

    assert run_process_with_graceful_interrupt(["child"]) == 130
    assert not child.terminated


def test_second_ctrl_c_forces_child_exit(monkeypatch: pytest.MonkeyPatch) -> None:
    child = _FakeChildProcess(interrupt_twice=True)
    monkeypatch.setattr(interruption.subprocess, "Popen", lambda *_args, **_kwargs: child)

    assert run_process_with_graceful_interrupt(["child"]) == 130
    assert child.terminated
