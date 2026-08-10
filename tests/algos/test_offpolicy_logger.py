from __future__ import annotations

from typing import Any

from rich.console import Console

import unilab.logging.common as common_logger_module
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
    logger.log_status("Replay storage: device-authoritative bounded ingress")

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


def test_offpolicy_logger_displays_env_step_breakdown_as_indented_children() -> None:
    logger = OffPolicyLogger(
        algo_name="SAC",
        max_iterations=2,
        num_envs=8,
        env_name="Dummy",
        log_backend="none",
    )
    logger.update_collector_timing(
        {
            "weight_sync_ms": 0.1,
            "action_select_ms": 0.2,
            "env_step_ms": 14.0,
            "env_step_backend_ms": 12.5,
            "env_step_update_state_ms": 1.0,
            "env_step_reset_done_ms": 0.5,
            "replay_ms": 0.3,
        }
    )

    table = logger._build_timing_table()
    collector_cells = list(table.columns[2].cells)
    collector_value_cells = list(table.columns[3].cells)

    assert collector_cells == [
        "Weight Sync",
        "Action Select",
        "Env Step",
        "[dim]  Backend Step[/]",
        "[dim]  Update State[/]",
        "[dim]  Reset Done[/]",
        "Replay",
    ]
    assert collector_value_cells == [
        "0.1ms",
        "0.2ms",
        "14.0ms",
        "[dim cyan]12.5ms ─┤[/]",
        "[dim cyan]1.0ms ─┤[/]",
        "[dim cyan]0.5ms ─┘[/]",
        "0.3ms",
    ]

    console = Console(width=100, record=True, force_terminal=False)
    with console.capture() as capture:
        console.print(table)
    connector_columns = [
        line.index(connector)
        for line in capture.get().splitlines()
        for connector in ("┤", "┘")
        if connector in line
    ]
    assert len(connector_columns) == 3
    assert len(set(connector_columns)) == 1


def test_offpolicy_reward_component_names_stay_on_one_line_at_narrow_width() -> None:
    logger = OffPolicyLogger(
        algo_name="SAC",
        max_iterations=2,
        num_envs=8,
        env_name="Dummy",
        log_backend="no_print",
    )
    logger._reward_history.extend([1.0, 2.0])
    logger._latest_reward_components = {
        "reward/penalty_action_rate": -0.1,
        "reward/penalty_ang_vel_xy": -0.2,
        "reward/penalty_orientation": -0.3,
    }
    console = Console(width=47, record=True, force_terminal=False)

    console.print(logger._build_reward_table())
    output_lines = console.export_text().splitlines()

    for component in ("penalty action rate", "penalty ang vel xy", "penalty orientation"):
        matching_lines = [line for line in output_lines if component in line]
        assert len(matching_lines) == 1


def test_offpolicy_logger_training_timer_excludes_warmup_from_elapsed_and_eta(
    monkeypatch,
) -> None:
    now = 100.0
    monkeypatch.setattr(common_logger_module.time, "time", lambda: now)

    logger = OffPolicyLogger(
        algo_name="FastSAC",
        max_iterations=2,
        num_envs=8,
        env_name="G1WalkFlat",
        log_backend="no_print",
    )
    logger._unicode_console = False
    logger.start()

    now = 130.0
    assert logger.start_training_timer() == 130.0

    now = 132.0
    logger.log_step(iteration=1, extra_info={"throughput_steps": 8})

    header = logger._build_compact_header(include_status=False, include_identity=False)
    assert "time 2s" in header.plain
    assert "ETA 2s" in header.plain
    assert "30s" not in header.plain


def test_offpolicy_logger_moves_identity_and_iteration_to_panel_title() -> None:
    logger = OffPolicyLogger(
        algo_name="FastSAC",
        max_iterations=5000,
        num_envs=4096,
        env_name="G1WalkFlat",
        log_backend="no_print",
    )
    logger._unicode_console = True
    logger.start_training_timer()
    logger.log_step(iteration=5000, extra_info={"throughput_steps": 4096})

    display = logger._build_display()
    header = logger._build_compact_header(
        include_status=False,
        include_identity=False,
        include_iteration=False,
    )

    assert isinstance(display.title, type(header))
    assert display.title.plain == (
        " 🚀 UniLab Off-Policy Training | FastSAC | G1WalkFlat | iter 5000/5000 "
    )
    assert "FastSAC" not in header.plain
    assert "G1WalkFlat" not in header.plain
    assert "iter 5000/5000" not in header.plain
    assert "Training" not in header.plain
