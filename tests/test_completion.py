from __future__ import annotations

from pathlib import Path

from unilab.tools.completion import build_metadata, complete_words


def _write_completion_fixture(root: Path) -> None:
    (root / "pyproject.toml").write_text(
        "\n".join(
            [
                "[project.scripts]",
                'train = "unilab.cli:train_main"',
                'eval = "unilab.cli:eval_main"',
                'demo = "unilab.cli:demo_main"',
            ]
        ),
        encoding="utf-8",
    )
    (root / "benchmark" / "core").mkdir(parents=True)
    (root / "benchmark" / "benchmark_sim.py").write_text("", encoding="utf-8")
    (root / "benchmark" / "core" / "runner.py").write_text("", encoding="utf-8")
    (root / "scripts").mkdir()
    (root / "scripts" / "play_viser.py").write_text("", encoding="utf-8")


def test_uv_run_command_position_includes_project_scripts_and_run_paths(tmp_path: Path) -> None:
    _write_completion_fixture(tmp_path)
    metadata = build_metadata(tmp_path)

    assert "demo" in complete_words(["uv", "run", "d"], 2, metadata)
    assert complete_words(["uv", "run", "b"], 2, metadata) == ["benchmark/"]


def test_uv_run_path_completion_is_hierarchical(tmp_path: Path) -> None:
    _write_completion_fixture(tmp_path)
    metadata = build_metadata(tmp_path)

    benchmark_choices = complete_words(["uv", "run", "benchmark/"], 2, metadata)
    assert "benchmark/benchmark_sim.py" in benchmark_choices
    assert "benchmark/core/" in benchmark_choices
    assert "benchmark/core/runner.py" not in benchmark_choices

    assert complete_words(["uv", "run", "benchmark/core/"], 2, metadata) == [
        "benchmark/core/runner.py"
    ]


def test_uv_run_unknown_command_arguments_defer_to_shell_completion(tmp_path: Path) -> None:
    _write_completion_fixture(tmp_path)
    metadata = build_metadata(tmp_path)

    assert complete_words(["uv", "run", "benchmark/benchmark_sim.py", ""], 3, metadata) == []
