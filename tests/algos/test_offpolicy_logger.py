from __future__ import annotations

from typing import Any

from unilab.logging.offpolicy import OffPolicyLogger


def test_offpolicy_logger_defers_warmup_refresh_until_training_step(monkeypatch) -> None:
    logger = OffPolicyLogger(
        algo_name="SAC",
        max_iterations=2,
        num_envs=8,
        env_name="Dummy",
        log_backend="none",
    )
    refresh_calls: list[bool] = []

    def fake_refresh(*, force: bool = False) -> None:
        refresh_calls.append(force)

    monkeypatch.setattr(logger, "_refresh", fake_refresh)

    logger.log_buffer_fill(32, 64)
    logger.log_status("Replay pipeline: cpu_pinned_double_buffer")

    assert refresh_calls == []
    assert logger._buffer_size == 32
    assert logger._buffer_target == 64

    logger.log_status("[red]ERROR: Collector died[/]")
    assert refresh_calls == [True]

    refresh_calls.clear()
    logger.log_step(
        iteration=1,
        metrics={"loss/q": 0.5},
        reward=1.0,
        extra_info={"throughput_steps": 8},
    )
    logger.log_status("Training")
    logger.log_buffer_fill(64, 64)

    assert refresh_calls == [False, False, False]


def test_offpolicy_logger_stop_live_lets_rich_do_final_refresh() -> None:
    logger = OffPolicyLogger(
        algo_name="SAC",
        max_iterations=2,
        num_envs=8,
        env_name="Dummy",
        log_backend="none",
    )

    class _FakeLive:
        def __init__(self) -> None:
            self.update_calls: list[bool] = []
            self.stop_calls = 0

        def update(self, renderable: Any, *, refresh: bool) -> None:
            del renderable
            self.update_calls.append(refresh)

        def stop(self) -> None:
            self.stop_calls += 1

    live = _FakeLive()
    logger._live = live  # type: ignore[assignment]
    logger._last_live_refresh_time = 123.0

    logger._stop_live()

    assert live.update_calls == [False]
    assert live.stop_calls == 1
    assert logger._live is None
    assert logger._last_live_refresh_time is None
