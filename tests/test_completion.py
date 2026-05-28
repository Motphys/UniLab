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
    owner_files = {
        root / "conf" / "ppo" / "task" / "go1" / "mujoco.yaml": """
training:
  task_name: Go1
  sim_backend: mujoco
""",
        root / "conf" / "ppo" / "task" / "go1" / "mujoco_hora.yaml": """
defaults:
  - /task/go1/mujoco
  - _self_
algo:
  algo_log_name: hora_ppo
""",
        root / "conf" / "ppo" / "task" / "go1" / "motrix_lab.yaml": """
training:
  task_name: Go1
  sim_backend: motrix
""",
        root / "conf" / "ppo" / "task" / "go2" / "mujoco_lab.yaml": """
training:
  task_name: Go2
  sim_backend: mujoco
""",
        root / "conf" / "ppo" / "task" / "go3" / "mujoco_custom.yaml": """
training:
  task_name: Go3
  sim_backend: mujoco
  log_root: custom_logs
""",
    }
    for path, content in owner_files.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content.strip() + "\n", encoding="utf-8")

    for path in [
        root / "logs" / "rsl_rl_ppo" / "Go1" / "2026-01-01_00-00-00_mujoco",
        root / "logs" / "rsl_rl_ppo" / "Go1" / "2026-01-02_00-00-00_mujoco",
        root / "logs" / "rsl_rl_ppo" / "Go2" / "2026-02-01_00-00-00_mujoco",
        root / "logs" / "hora_ppo" / "Go1" / "2026-03-01_00-00-00_mujoco",
        root / "custom_logs" / "Go3" / "2026-04-01_00-00-00_mujoco",
    ]:
        path.mkdir(parents=True)


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


def test_eval_load_run_value_position_completes_latest_run_alias(tmp_path: Path) -> None:
    _write_completion_fixture(tmp_path)
    metadata = build_metadata(tmp_path)

    choices = complete_words(["uv", "run", "eval", "--load-run", ""], 4, metadata)
    assert choices == ["-1"]
    assert "--algo" not in choices
    assert complete_words(["uv", "run", "eval", "--load-run", "-"], 4, metadata) == ["-1"]
    assert complete_words(["uv", "run", "eval", "--load-run", "run"], 4, metadata) == []


def test_eval_load_run_value_position_completes_task_run_dirs(tmp_path: Path) -> None:
    _write_completion_fixture(tmp_path)
    metadata = build_metadata(tmp_path)

    assert complete_words(
        [
            "uv",
            "run",
            "eval",
            "--algo",
            "ppo",
            "--task",
            "go1",
            "--sim",
            "mujoco",
            "--load-run",
            "",
        ],
        10,
        metadata,
    ) == ["-1", "2026-01-01_00-00-00_mujoco", "2026-01-02_00-00-00_mujoco"]
    assert complete_words(
        [
            "uv",
            "run",
            "eval",
            "--algo",
            "ppo",
            "--task",
            "go1",
            "--sim",
            "mujoco",
            "--load-run",
            "2026-01-02",
        ],
        10,
        metadata,
    ) == ["2026-01-02_00-00-00_mujoco"]


def test_eval_load_run_completion_respects_profile_log_name(tmp_path: Path) -> None:
    _write_completion_fixture(tmp_path)
    metadata = build_metadata(tmp_path)

    assert complete_words(
        [
            "uv",
            "run",
            "eval",
            "--algo",
            "ppo",
            "--task",
            "go1",
            "--sim",
            "mujoco",
            "--profile",
            "hora",
            "--load-run",
            "",
        ],
        12,
        metadata,
    ) == ["-1", "2026-03-01_00-00-00_mujoco"]


def test_eval_load_run_completion_respects_training_log_root(tmp_path: Path) -> None:
    _write_completion_fixture(tmp_path)
    metadata = build_metadata(tmp_path)

    assert complete_words(
        [
            "uv",
            "run",
            "eval",
            "--algo",
            "ppo",
            "--task",
            "go3",
            "--sim",
            "mujoco",
            "--profile",
            "custom",
            "--load-run",
            "",
        ],
        12,
        metadata,
    ) == ["-1", "2026-04-01_00-00-00_mujoco"]


def test_train_profile_value_position_completes_profile_names(tmp_path: Path) -> None:
    _write_completion_fixture(tmp_path)
    metadata = build_metadata(tmp_path)

    assert complete_words(
        ["uv", "run", "train", "--algo", "ppo", "--sim", "mujoco", "--profile", ""],
        8,
        metadata,
    ) == ["custom", "hora", "lab"]
    choices = complete_words(
        ["uv", "run", "train", "--algo", "ppo", "--sim", "mujoco", "--profile", "h"],
        8,
        metadata,
    )
    assert choices == ["hora"]
    assert "--algo" not in choices
    assert complete_words(
        ["uv", "run", "train", "--algo", "ppo", "--task", "go1", "--sim", "mujoco", "--profile", ""],
        10,
        metadata,
    ) == ["hora"]
    assert complete_words(
        ["uv", "run", "train", "--algo", "ppo", "--task", "go1", "--sim", "motrix", "--profile", ""],
        10,
        metadata,
    ) == ["lab"]
    assert complete_words(
        ["uv", "run", "eval", "--algo", "ppo", "--sim", "mujoco", "--profile", "h"],
        8,
        metadata,
    ) == ["hora"]


def test_task_completion_respects_selected_profile(tmp_path: Path) -> None:
    _write_completion_fixture(tmp_path)
    metadata = build_metadata(tmp_path)

    assert complete_words(
        ["uv", "run", "train", "--algo", "ppo", "--sim", "mujoco", "--profile", "hora", "--task", ""],
        10,
        metadata,
    ) == ["go1"]
