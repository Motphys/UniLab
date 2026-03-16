"""Tests for script entry-point utilities (pure functions and argument parsing).

Coverage targets:
  - train_offpolicy.py: build_parser(), default_device(), resolve_checkpoint_path()
  - train_mlx_ppo.py:   get_latest_run(), get_latest_checkpoint()  (skipped if mlx absent)
  - play_interactive.py: resolve_checkpoint()                       (skipped if mujoco absent)
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

_SCRIPTS_DIR = Path(__file__).parent.parent.parent / "scripts"


def _load_script(name: str):
    """Load a scripts/<name>.py as a fresh module (no __init__ required)."""
    path = _SCRIPTS_DIR / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

try:
    import sys as _sys

    import mlx.core  # noqa: F401

    _HAS_MLX = _sys.platform == "darwin"
except ImportError:
    _HAS_MLX = False

try:
    import mujoco  # noqa: F401

    _HAS_MUJOCO = True
except ImportError:
    _HAS_MUJOCO = False


# ---------------------------------------------------------------------------
# train_offpolicy.py — build_parser()
# ---------------------------------------------------------------------------


def _offpolicy():
    return _load_script("train_offpolicy")


def test_offpolicy_parser_default_algo():
    args = _offpolicy().build_parser().parse_args([])
    assert args.algo == "sac"


def test_offpolicy_parser_default_task():
    args = _offpolicy().build_parser().parse_args([])
    assert args.task == "Go1JoystickFlatTerrain"


def test_offpolicy_parser_default_logger():
    args = _offpolicy().build_parser().parse_args([])
    assert args.logger == "tensorboard"


def test_offpolicy_parser_default_sim_backend():
    args = _offpolicy().build_parser().parse_args([])
    assert args.sim_backend == "mujoco"


def test_offpolicy_parser_default_play_flags():
    args = _offpolicy().build_parser().parse_args([])
    assert args.play_only is False
    assert args.no_play is False
    assert args.load_run == "-1"


def test_offpolicy_parser_invalid_algo_exits():
    """Unknown algo must cause argparse to exit."""
    with pytest.raises(SystemExit):
        _offpolicy().build_parser().parse_args(["--algo", "dqn"])


def test_offpolicy_parser_invalid_logger_exits():
    with pytest.raises(SystemExit):
        _offpolicy().build_parser().parse_args(["--logger", "mlflow"])


def test_offpolicy_parser_algo_td3():
    args = _offpolicy().build_parser().parse_args(["--algo", "td3"])
    assert args.algo == "td3"


# ---------------------------------------------------------------------------
# train_offpolicy.py — default_device()
# ---------------------------------------------------------------------------


def test_offpolicy_default_device_preferred_cpu():
    mock_torch = MagicMock()
    assert _offpolicy().default_device(mock_torch, preferred="cpu") == "cpu"


def test_offpolicy_default_device_preferred_cuda():
    mock_torch = MagicMock()
    assert _offpolicy().default_device(mock_torch, preferred="cuda") == "cuda"


def test_offpolicy_default_device_cuda_available():
    mock_torch = MagicMock()
    mock_torch.cuda.is_available.return_value = True
    assert _offpolicy().default_device(mock_torch) == "cuda"


def test_offpolicy_default_device_mps_fallback():
    mock_torch = MagicMock()
    mock_torch.cuda.is_available.return_value = False
    mock_torch.backends.mps.is_available.return_value = True
    assert _offpolicy().default_device(mock_torch) == "mps"


def test_offpolicy_default_device_cpu_fallback():
    mock_torch = MagicMock()
    mock_torch.cuda.is_available.return_value = False
    mock_torch.backends.mps.is_available.return_value = False
    assert _offpolicy().default_device(mock_torch) == "cpu"


# ---------------------------------------------------------------------------
# train_offpolicy.py — resolve_checkpoint_path()
# ---------------------------------------------------------------------------


def test_resolve_checkpoint_no_base_dir(tmp_path):
    """load_run='-1' with no log directory → (None, None)."""
    path, path_dir = _offpolicy().resolve_checkpoint_path(tmp_path, "sac", "MyTask", "-1")
    assert path is None
    assert path_dir is None


def test_resolve_checkpoint_explicit_existing_file(tmp_path):
    """load_run = absolute path to existing .pt → returns that path."""
    model_file = tmp_path / "model_100.pt"
    model_file.write_bytes(b"")
    path, path_dir = _offpolicy().resolve_checkpoint_path(
        tmp_path, "sac", "MyTask", str(model_file)
    )
    assert path == str(model_file)
    assert path_dir == str(tmp_path)


def test_resolve_checkpoint_latest_picks_highest_iter(tmp_path):
    """load_run='-1' picks model with numerically highest iteration."""
    task_dir = tmp_path / "logs" / "sac" / "MyTask" / "run1"
    task_dir.mkdir(parents=True)
    (task_dir / "model_10.pt").write_bytes(b"")
    (task_dir / "model_50.pt").write_bytes(b"")
    (task_dir / "model_100.pt").write_bytes(b"")

    path, path_dir = _offpolicy().resolve_checkpoint_path(tmp_path, "sac", "MyTask", "-1")
    assert path is not None
    assert "model_100.pt" in path


def test_resolve_checkpoint_explicit_run_name(tmp_path):
    """load_run = run-directory name under the log root."""
    task_dir = tmp_path / "logs" / "sac" / "MyTask" / "myrun"
    task_dir.mkdir(parents=True)
    (task_dir / "model_5.pt").write_bytes(b"")

    path, path_dir = _offpolicy().resolve_checkpoint_path(tmp_path, "sac", "MyTask", "myrun")
    assert path is not None
    assert "model_5.pt" in path
    assert path_dir == str(task_dir)


def test_resolve_checkpoint_nonexistent_explicit_path(tmp_path):
    """load_run points to a path that doesn't exist → (None, None)."""
    path, path_dir = _offpolicy().resolve_checkpoint_path(
        tmp_path, "sac", "MyTask", "/nonexistent/model.pt"
    )
    assert path is None
    assert path_dir is None


def test_resolve_checkpoint_empty_run_dir(tmp_path):
    """Run directory exists but has no model_*.pt → (None, None)."""
    task_dir = tmp_path / "logs" / "sac" / "MyTask" / "run1"
    task_dir.mkdir(parents=True)

    path, path_dir = _offpolicy().resolve_checkpoint_path(tmp_path, "sac", "MyTask", "-1")
    assert path is None


# ---------------------------------------------------------------------------
# train_mlx_ppo.py — get_latest_run() / get_latest_checkpoint()
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _HAS_MLX, reason="mlx not installed")
def test_mlx_get_latest_run_nonexistent_dir(tmp_path):
    mod = _load_script("train_mlx_ppo")
    assert mod.get_latest_run(tmp_path / "nonexistent") is None


@pytest.mark.skipif(not _HAS_MLX, reason="mlx not installed")
def test_mlx_get_latest_run_empty_dir(tmp_path):
    mod = _load_script("train_mlx_ppo")
    assert mod.get_latest_run(tmp_path) is None


@pytest.mark.skipif(not _HAS_MLX, reason="mlx not installed")
def test_mlx_get_latest_run_returns_last_sorted(tmp_path):
    mod = _load_script("train_mlx_ppo")
    (tmp_path / "2024-01-01_mujoco").mkdir()
    (tmp_path / "2024-03-15_mujoco").mkdir()
    (tmp_path / "2024-02-10_mujoco").mkdir()
    result = mod.get_latest_run(tmp_path)
    assert result is not None
    assert result.name == "2024-03-15_mujoco"


@pytest.mark.skipif(not _HAS_MLX, reason="mlx not installed")
def test_mlx_get_latest_checkpoint_nonexistent_dir(tmp_path):
    mod = _load_script("train_mlx_ppo")
    assert mod.get_latest_checkpoint(tmp_path / "no_such_dir") is None


@pytest.mark.skipif(not _HAS_MLX, reason="mlx not installed")
def test_mlx_get_latest_checkpoint_empty_dir(tmp_path):
    mod = _load_script("train_mlx_ppo")
    assert mod.get_latest_checkpoint(tmp_path) is None


@pytest.mark.skipif(not _HAS_MLX, reason="mlx not installed")
def test_mlx_get_latest_checkpoint_picks_highest_iter(tmp_path):
    mod = _load_script("train_mlx_ppo")
    (tmp_path / "model_0.safetensors").write_bytes(b"")
    (tmp_path / "model_50.safetensors").write_bytes(b"")
    (tmp_path / "model_200.safetensors").write_bytes(b"")
    result = mod.get_latest_checkpoint(tmp_path)
    assert result is not None
    assert result.name == "model_200.safetensors"


@pytest.mark.skipif(not _HAS_MLX, reason="mlx not installed")
def test_mlx_get_latest_checkpoint_ignores_non_safetensors(tmp_path):
    """Only .safetensors files count; .pt files must be ignored."""
    mod = _load_script("train_mlx_ppo")
    (tmp_path / "model_999.pt").write_bytes(b"")  # should be ignored
    assert mod.get_latest_checkpoint(tmp_path) is None


# ---------------------------------------------------------------------------
# play_interactive.py — resolve_checkpoint()
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _HAS_MUJOCO, reason="mujoco not installed")
def test_play_resolve_checkpoint_nonexistent_run(tmp_path):
    """Passing a non-existent explicit path returns None."""
    mod = _load_script("play_interactive")
    result = mod.resolve_checkpoint("MyTask", str(tmp_path / "no_run"))
    assert result is None


@pytest.mark.skipif(not _HAS_MUJOCO, reason="mujoco not installed")
def test_play_resolve_checkpoint_dir_with_model(tmp_path):
    """Directory path containing model_*.pt files resolves to the latest."""
    mod = _load_script("play_interactive")
    run_dir = tmp_path / "2024-01-01_mujoco"
    run_dir.mkdir()
    (run_dir / "model_10.pt").write_bytes(b"")
    (run_dir / "model_50.pt").write_bytes(b"")

    result = mod.resolve_checkpoint("MyTask", str(run_dir))
    assert result is not None
    assert "model_50.pt" in result


@pytest.mark.skipif(not _HAS_MUJOCO, reason="mujoco not installed")
def test_play_resolve_checkpoint_explicit_file(tmp_path):
    """Absolute path to existing .pt file returns that path unchanged."""
    mod = _load_script("play_interactive")
    model_file = tmp_path / "model_99.pt"
    model_file.write_bytes(b"")
    result = mod.resolve_checkpoint("MyTask", str(model_file))
    assert result == str(model_file)


@pytest.mark.skipif(not _HAS_MUJOCO, reason="mujoco not installed")
def test_play_resolve_checkpoint_empty_dir(tmp_path):
    """Directory with no model_*.pt files returns None."""
    mod = _load_script("play_interactive")
    run_dir = tmp_path / "run1"
    run_dir.mkdir()
    result = mod.resolve_checkpoint("MyTask", str(run_dir))
    assert result is None
