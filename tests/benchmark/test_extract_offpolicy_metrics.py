from __future__ import annotations

import json
import math
from pathlib import Path

import pytest
from scripts.benchmark.rl import extract_offpolicy_metrics as extract

torch = pytest.importorskip("torch")
from torch.utils.tensorboard import SummaryWriter  # noqa: E402


def _write_tfevents(log_dir: Path, series: dict[str, list[float]]) -> None:
    log_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(log_dir))
    try:
        for tag, values in series.items():
            for step, value in enumerate(values):
                writer.add_scalar(tag, value, step)
    finally:
        writer.close()


def _run_json(log_dir: Path, capsys: pytest.CaptureFixture) -> dict[str, float]:
    assert extract.main([str(log_dir), "--json", "--last", "20"]) == 0
    return json.loads(capsys.readouterr().out)


def test_extract_new_schema_tags_with_unit_conversion(tmp_path: Path, capsys) -> None:
    _write_tfevents(
        tmp_path,
        {
            "Perf/iteration_time": [1.0, 1.5],  # seconds
            "Perf/learning_time": [0.4, 0.6],  # seconds
            "Perf/total_fps": [100.0, 300.0],
            "Perf/learner_collector_wait_ms": [5.0, 7.0],
        },
    )
    rows = _run_json(tmp_path, capsys)
    assert rows["iter_ms"] == pytest.approx(1_250.0)
    assert rows["learner_train_ms"] == pytest.approx(500.0)
    assert rows["steps_per_sec"] == pytest.approx(200.0)
    assert rows["learner_collector_wait_ms"] == pytest.approx(6.0)
    # Retired charts without a canonical replacement stay NaN on new runs.
    assert math.isnan(rows["collector_active_steps_per_sec"])
    assert math.isnan(rows["collector_cycle_ms"])


def test_extract_legacy_tags_fall_back_unscaled(tmp_path: Path, capsys) -> None:
    _write_tfevents(
        tmp_path,
        {
            "perf/iter_ms": [1_000.0, 1_500.0],
            "timing/learner_train_ms": [400.0, 600.0],
            "perf/steps_per_sec": [100.0, 300.0],
            "perf/collector_active_steps_per_sec": [50.0, 70.0],
        },
    )
    rows = _run_json(tmp_path, capsys)
    assert rows["iter_ms"] == pytest.approx(1_250.0)
    assert rows["learner_train_ms"] == pytest.approx(500.0)
    assert rows["steps_per_sec"] == pytest.approx(200.0)
    assert rows["collector_active_steps_per_sec"] == pytest.approx(60.0)


def test_extract_prefers_new_tag_when_both_exist(tmp_path: Path, capsys) -> None:
    _write_tfevents(
        tmp_path,
        {
            "Perf/total_fps": [999.0],
            "perf/steps_per_sec": [111.0],
        },
    )
    rows = _run_json(tmp_path, capsys)
    assert rows["steps_per_sec"] == pytest.approx(999.0)


def test_extract_missing_event_file_is_an_error(tmp_path: Path, capsys) -> None:
    assert extract.main([str(tmp_path)]) == 1
    assert "No event file found" in capsys.readouterr().err
