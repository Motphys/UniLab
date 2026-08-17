from __future__ import annotations

import ast
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from unilab.tools import bones_seed_csv
from unilab.tools.bones_seed_csv import (
    ROOT_COLUMNS,
    euler_deg_to_quat_wxyz,
    load_header,
    parse_joint_names,
    resolve_input_files,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATHS = (
    REPO_ROOT / "scripts" / "motion" / "bones_seed_csv_to_npz.py",
    REPO_ROOT / "scripts" / "motion" / "replay_bones_seed_csv.py",
)
SHARED_NAMES = {
    "ROOT_COLUMNS",
    "EXPECTED_JOINT_COUNT",
    "natural_sort_key",
    "resolve_input_files",
    "load_header",
    "parse_joint_names",
    "_swap_case_for_mujoco_euler",
    "euler_deg_to_quat_wxyz",
}


def _joint_columns() -> list[str]:
    return [f"joint_{index:02d}_dof" for index in range(29)]


def test_resolve_input_files_uses_natural_order_and_accepts_single_csv(tmp_path: Path) -> None:
    csv_paths = [tmp_path / name for name in ("flip_10.csv", "flip_2.csv", "flip_1.csv")]
    for path in csv_paths:
        path.write_text("", encoding="utf-8")
    (tmp_path / "ignored.txt").write_text("", encoding="utf-8")

    assert resolve_input_files(str(tmp_path)) == [csv_paths[2], csv_paths[1], csv_paths[0]]
    assert resolve_input_files(str(csv_paths[1])) == [csv_paths[1]]


def test_resolve_input_files_preserves_path_errors(tmp_path: Path) -> None:
    missing = tmp_path / "missing"
    with pytest.raises(FileNotFoundError, match="Input path not found"):
        resolve_input_files(str(missing))

    non_csv = tmp_path / "motion.txt"
    non_csv.write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="Expected a CSV file"):
        resolve_input_files(str(non_csv))

    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()
    with pytest.raises(FileNotFoundError, match="No CSV files found in directory"):
        resolve_input_files(str(empty_dir))


def test_load_header_and_parse_joint_names_accept_bones_seed_schema(tmp_path: Path) -> None:
    csv_file = tmp_path / "motion.csv"
    header = [*ROOT_COLUMNS, *_joint_columns()]
    csv_file.write_text(", ".join(header) + "\n", encoding="utf-8")

    loaded_header = load_header(csv_file)

    assert loaded_header == header
    assert parse_joint_names(loaded_header, csv_file) == [
        name.removesuffix("_dof") for name in _joint_columns()
    ]


@pytest.mark.parametrize(
    ("header", "message"),
    [
        (["frame", *ROOT_COLUMNS[1:], *_joint_columns()], "Unexpected root columns"),
        ([*ROOT_COLUMNS, *_joint_columns()[:-1]], "has 28 joint columns"),
        (
            [*ROOT_COLUMNS, *_joint_columns()[:-1], _joint_columns()[0]],
            "duplicate joint columns",
        ),
        ([*ROOT_COLUMNS, *_joint_columns()[:-1], "bad_joint"], "non-joint columns"),
    ],
)
def test_parse_joint_names_preserves_schema_errors(header: list[str], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        parse_joint_names(header, Path("motion.csv"))


@pytest.mark.parametrize(
    ("order", "expected"),
    [
        (
            "xyz",
            np.asarray([[1.0, 0.0, 0.0, 0.0], [0.95154852, 0.23929834, 0.18930786, 0.03813458]]),
        ),
        (
            "XYZ",
            np.asarray([[1.0, 0.0, 0.0, 0.0], [0.94371436, 0.26853582, 0.14487813, 0.12767944]]),
        ),
    ],
)
def test_euler_deg_to_quat_wxyz_matches_existing_golden(order: str, expected: np.ndarray) -> None:
    pytest.importorskip("mujoco")
    euler_deg = np.asarray([[0.0, 0.0, 0.0], [30.0, 20.0, 10.0]], dtype=np.float64)

    actual = euler_deg_to_quat_wxyz(euler_deg, order)

    assert actual.dtype == np.float64
    np.testing.assert_allclose(actual, expected, atol=1.0e-8, rtol=0.0)


def test_bones_seed_scripts_use_shared_owner_exports() -> None:
    pytest.importorskip("mujoco")
    from scripts.motion import bones_seed_csv_to_npz, replay_bones_seed_csv

    for script in (bones_seed_csv_to_npz, replay_bones_seed_csv):
        assert script.ROOT_COLUMNS == bones_seed_csv.ROOT_COLUMNS
        for name in (
            "resolve_input_files",
            "load_header",
            "parse_joint_names",
            "euler_deg_to_quat_wxyz",
        ):
            assert getattr(script, name).__module__ == bones_seed_csv.__name__


def test_bones_seed_scripts_do_not_redefine_shared_contract() -> None:
    for script_path in SCRIPT_PATHS:
        module = ast.parse(script_path.read_text(encoding="utf-8"))
        defined_names = {
            node.name
            for node in module.body
            if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
        }
        assigned_names = {
            target.id
            for node in module.body
            if isinstance(node, (ast.Assign, ast.AnnAssign))
            for target in (node.targets if isinstance(node, ast.Assign) else [node.target])
            if isinstance(target, ast.Name)
        }
        assert SHARED_NAMES.isdisjoint(defined_names | assigned_names)


@pytest.mark.parametrize("script_path", SCRIPT_PATHS)
def test_bones_seed_cli_help_smoke(script_path: Path) -> None:
    pytest.importorskip("mujoco")
    env = os.environ.copy()
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["MUJOCO_GL"] = "disable"

    result = subprocess.run(
        [sys.executable, str(script_path), "--help"],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert result.returncode == 0, result.stderr or result.stdout
