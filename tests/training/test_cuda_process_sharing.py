"""Fail-closed CUDA process-sharing probe tests."""

from __future__ import annotations

import platform
import socket
import stat
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from unilab.training.cuda_process_sharing import (
    probe_cuda_process_sharing,
    resolve_mps_visibility_entries,
)


class _Completed:
    def __init__(self, stdout: str = "", returncode: int = 0) -> None:
        self.stdout = stdout
        self.returncode = returncode


def _torch(uuid: str) -> Any:
    properties = SimpleNamespace(uuid=uuid)
    return SimpleNamespace(
        version=SimpleNamespace(hip=None),
        cuda=SimpleNamespace(
            is_available=lambda: True,
            device_count=lambda: 1,
            get_device_properties=lambda _index: properties,
        ),
    )


def _fake_commands(gpu_uuid: str = "GPU-a", *, server_pid: int = 2768293) -> Any:
    def run_command(command: list[str], **kwargs: Any) -> _Completed:
        if command[0] == "nvidia-smi":
            return _Completed(f"0, {gpu_uuid}\n")
        if command == ["nvidia-cuda-mps-control"]:
            assert kwargs["input"] == "get_server_list\n"
            return _Completed(f"{server_pid}\n")
        raise AssertionError(f"unexpected command: {command}")

    return run_command


def _fake_uuid_table_commands(stdout: str) -> Any:
    def run_command(command: list[str], **kwargs: Any) -> _Completed:
        assert command[0] == "nvidia-smi"
        return _Completed(stdout)

    return run_command


@pytest.fixture
def linux_mps(monkeypatch: pytest.MonkeyPatch, short_unix_socket_root: Path):
    monkeypatch.setattr(platform, "system", lambda: "Linux")
    monkeypatch.setenv("CUDA_MPS_PIPE_DIRECTORY", str(short_unix_socket_root / "pipe"))
    monkeypatch.setenv("CUDA_MPS_LOG_DIRECTORY", str(short_unix_socket_root / "log"))
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-a")
    control = short_unix_socket_root / "pipe" / "control"
    control.parent.mkdir(parents=True)
    control.touch()
    control.chmod(0o666)
    return control


def _socket_control(path: Any) -> None:
    path.unlink()
    listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    listener.bind(str(path))
    path.chmod(0o666)


def test_disabled_default_does_not_probe_host(monkeypatch: pytest.MonkeyPatch) -> None:
    def reject(*args: Any, **kwargs: Any) -> None:
        del args, kwargs
        raise AssertionError("disabled mode must not probe CUDA or MPS")

    monkeypatch.setattr(platform, "system", lambda: "not-linux")
    monkeypatch.delenv("CUDA_MPS_PIPE_DIRECTORY", raising=False)
    monkeypatch.delenv("CUDA_MPS_LOG_DIRECTORY", raising=False)
    evidence = probe_cuda_process_sharing(
        None,
        "cpu",
        None,
        backend="mujoco",
        torch_module=None,
        run_command=reject,
    )

    assert evidence.to_dict() == {
        "configured": None,
        "effective": None,
        "validated": False,
        "learner_device": "cpu",
        "collector_device": None,
        "learner_gpu_uuid": None,
        "collector_gpu_uuid": None,
        "pipe_directory_env": None,
        "pipe_directory": None,
        "control_pipe": None,
        "control_pipe_kind": None,
        "server_pid": None,
        "ambient_server": False,
        "rank": 0,
        "world_size": 1,
    }


def test_enabled_probe_remains_nvidia_gated(linux_mps: Any) -> None:
    """Transport's CUDA-family name does not admit AMD or a non-NVIDIA daemon."""
    rocm_torch = _torch("GPU-a")
    rocm_torch.version = SimpleNamespace(hip="6.3")

    with pytest.raises(ValueError, match="supports NVIDIA CUDA only"):
        probe_cuda_process_sharing(
            "mps",
            "cuda:0",
            "cuda:0",
            backend="mjwarp",
            torch_module=rocm_torch,
            run_command=_fake_commands(),
        )


def test_valid_single_rank_request_records_server_evidence(linux_mps: Any) -> None:
    _socket_control(linux_mps)

    evidence = probe_cuda_process_sharing(
        "mps",
        "cuda:0",
        "cuda:0",
        backend="mjwarp",
        torch_module=_torch("GPU-a"),
        run_command=_fake_commands(),
    )

    assert evidence.to_dict() == {
        "configured": "mps",
        "effective": "mps",
        "validated": True,
        "learner_device": "cuda:0",
        "collector_device": "cuda:0",
        "learner_gpu_uuid": "A",
        "collector_gpu_uuid": "A",
        "pipe_directory_env": str(linux_mps.parent),
        "pipe_directory": str(linux_mps.parent),
        "control_pipe": str(linux_mps),
        "control_pipe_kind": "unix_socket",
        "server_pid": 2768293,
        "ambient_server": False,
        "rank": 0,
        "world_size": 1,
    }


def test_valid_identity_compares_physical_uuid_not_ordinal(linux_mps: Any) -> None:
    _socket_control(linux_mps)

    evidence = probe_cuda_process_sharing(
        "mps",
        "cuda:0",
        "cuda:0",
        backend="mjwarp",
        torch_module=_torch("GPU-a"),
        run_command=_fake_commands(),
    )

    assert evidence.validated
    assert evidence.learner_gpu_uuid == evidence.collector_gpu_uuid


def test_single_host_dp_rank_records_rank_scoped_evidence(linux_mps: Any) -> None:
    _socket_control(linux_mps)

    evidence = probe_cuda_process_sharing(
        "mps",
        "cuda:0",
        "cuda:0",
        backend="mjwarp",
        rank=1,
        world_size=2,
        torch_module=_torch("GPU-a"),
        run_command=_fake_commands(),
    )

    assert evidence.rank == 1
    assert evidence.world_size == 2
    assert evidence.validated


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"requested": "invalid"}, "expected null or 'mps'"),
        ({"backend": "mujoco"}, "only the mjwarp backend"),
        ({"rank": 2, "world_size": 2}, "valid rank in"),
        ({"collector_device": None}, "requires a CUDA collector"),
        ({"learner_device": "cpu"}, "requires CUDA learner and collector"),
    ],
)
def test_first_prerequisites_fail_closed(
    linux_mps: Any,
    monkeypatch: pytest.MonkeyPatch,
    overrides: dict[str, Any],
    message: str,
) -> None:
    del monkeypatch
    kwargs: dict[str, Any] = {
        "requested": "mps",
        "learner_device": "cuda:0",
        "collector_device": "cuda:0",
        "backend": "mjwarp",
        "world_size": 1,
        "torch_module": _torch("GPU-a"),
        "run_command": _fake_commands(),
    }
    kwargs.update(overrides)

    with pytest.raises(ValueError, match=message):
        probe_cuda_process_sharing(**kwargs)


def test_non_linux_rejects(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(platform, "system", lambda: "Darwin")

    with pytest.raises(ValueError, match="requires Linux"):
        probe_cuda_process_sharing(
            "mps",
            "cuda:0",
            "cuda:0",
            backend="mjwarp",
            torch_module=_torch("GPU-a"),
            run_command=_fake_commands(),
        )


def test_unavailable_cuda_rejects(linux_mps: Any) -> None:
    unavailable = _torch("GPU-a")
    unavailable.cuda.is_available = lambda: False

    with pytest.raises(ValueError, match="CUDA is unavailable"):
        probe_cuda_process_sharing(
            "mps",
            "cuda:0",
            "cuda:0",
            backend="mjwarp",
            torch_module=unavailable,
            run_command=_fake_commands(),
        )


def test_gpu_identity_mismatch_rejects(linux_mps: Any) -> None:
    _socket_control(linux_mps)

    with pytest.raises(ValueError, match=r"learner='A'.*collector='B'"):
        probe_cuda_process_sharing(
            "mps",
            "cuda:0",
            "cuda:0",
            backend="mjwarp",
            torch_module=_torch("GPU-a"),
            run_command=_fake_commands(gpu_uuid="GPU-b"),
        )


def test_missing_control_pipe_rejects_with_start_command(
    linux_mps: Any,
) -> None:
    linux_mps.unlink()

    with pytest.raises(ValueError, match="nvidia-cuda-mps-control -d"):
        probe_cuda_process_sharing(
            "mps",
            "cuda:0",
            "cuda:0",
            backend="mjwarp",
            torch_module=_torch("GPU-a"),
            run_command=_fake_commands(),
        )


def test_regular_file_is_not_a_control_pipe(linux_mps: Any) -> None:
    assert not stat.S_ISSOCK(linux_mps.stat().st_mode)

    with pytest.raises(ValueError, match="not a daemon control pipe"):
        probe_cuda_process_sharing(
            "mps",
            "cuda:0",
            "cuda:0",
            backend="mjwarp",
            torch_module=_torch("GPU-a"),
            run_command=_fake_commands(),
        )


def test_unreachable_daemon_rejects_without_fallback(linux_mps: Any) -> None:
    _socket_control(linux_mps)

    def failing_daemon(command: list[str], **kwargs: Any) -> _Completed:
        del kwargs
        if command[0] == "nvidia-smi":
            return _Completed("0, GPU-a\n")
        return _Completed("Cannot find MPS control daemon process", returncode=1)

    with pytest.raises(ValueError, match="could not reach the control daemon"):
        probe_cuda_process_sharing(
            "mps",
            "cuda:0",
            "cuda:0",
            backend="mjwarp",
            torch_module=_torch("GPU-a"),
            run_command=failing_daemon,
        )


def test_unconfigured_cuda_rank_records_ambient_server(linux_mps: Any) -> None:
    """A transparent environment-level daemon must be self-reported by the run."""

    _socket_control(linux_mps)

    evidence = probe_cuda_process_sharing(
        None,
        "cuda:0",
        "cuda:0",
        backend="mujoco",
        torch_module=_torch("GPU-a"),
        run_command=_fake_commands(),
    )

    assert evidence.configured is None
    assert evidence.effective is None
    assert not evidence.validated
    assert evidence.ambient_server
    assert evidence.pipe_directory_env == str(linux_mps.parent)
    assert evidence.pipe_directory == str(linux_mps.parent)
    assert evidence.control_pipe == str(linux_mps)
    assert evidence.control_pipe_kind == "unix_socket"
    assert evidence.server_pid == 2768293
    assert evidence.learner_gpu_uuid == "A"
    assert evidence.collector_gpu_uuid == "A"


def test_unconfigured_cuda_rank_without_daemon_records_absence(linux_mps: Any) -> None:
    linux_mps.unlink()

    evidence = probe_cuda_process_sharing(
        None,
        "cuda:0",
        "cuda:0",
        backend="mujoco",
        torch_module=_torch("GPU-a"),
        run_command=_fake_commands(),
    )

    assert not evidence.validated
    assert not evidence.ambient_server
    assert evidence.pipe_directory == str(linux_mps.parent)
    assert evidence.control_pipe == str(linux_mps)
    assert evidence.control_pipe_kind is None
    assert evidence.server_pid is None
    assert evidence.learner_gpu_uuid == "A"
    assert evidence.collector_gpu_uuid == "A"


def test_unconfigured_cpu_rank_skips_host_probe(monkeypatch: pytest.MonkeyPatch) -> None:
    def reject(*args: Any, **kwargs: Any) -> None:
        del args, kwargs
        raise AssertionError("a cpu rank must not probe CUDA or MPS")

    monkeypatch.setenv("CUDA_MPS_PIPE_DIRECTORY", "/tmp/nvidia-mps-test")
    evidence = probe_cuda_process_sharing(
        None,
        "cpu",
        None,
        backend="mujoco",
        torch_module=None,
        run_command=reject,
    )

    assert evidence.pipe_directory_env == "/tmp/nvidia-mps-test"
    assert evidence.pipe_directory is None
    assert evidence.control_pipe is None
    assert evidence.control_pipe_kind is None
    assert evidence.server_pid is None
    assert evidence.learner_gpu_uuid is None
    assert evidence.collector_gpu_uuid is None
    assert not evidence.ambient_server


def test_ambient_probe_records_unresolvable_facts_as_none(linux_mps: Any) -> None:
    """Ambient observation never fails closed, even with broken host tools."""

    _socket_control(linux_mps)
    failing_torch = SimpleNamespace(
        version=SimpleNamespace(hip=None),
        cuda=SimpleNamespace(is_available=lambda: (_ for _ in ()).throw(RuntimeError("no cuda"))),
    )

    def failing_commands(command: list[str], **kwargs: Any) -> Any:
        del command, kwargs
        raise RuntimeError("no host tools")

    evidence = probe_cuda_process_sharing(
        None,
        "cuda:0",
        "cuda:0",
        backend="mujoco",
        torch_module=failing_torch,
        run_command=failing_commands,
    )

    assert evidence.control_pipe_kind == "unix_socket"
    assert evidence.server_pid is None
    assert evidence.learner_gpu_uuid is None
    assert evidence.collector_gpu_uuid is None
    assert not evidence.ambient_server
    assert not evidence.validated


def test_mps_visibility_entries_resolve_numeric_indices() -> None:
    """Numeric entries become host UUIDs in nvidia-smi display form."""

    resolved = resolve_mps_visibility_entries(
        ("0", "1"),
        run_command=_fake_uuid_table_commands("0, GPU-aaaa\n1, GPU-bbbb\n"),
    )

    assert resolved == ("GPU-aaaa", "GPU-bbbb")


def test_mps_visibility_entries_pass_through_non_numeric() -> None:
    """UUID, MIG, and CUDA-disabled entries never touch host tooling."""

    def forbidden_command(command: list[str], **kwargs: Any) -> Any:
        raise AssertionError(f"unexpected command: {command}")

    entries = ("GPU-aaaa", "MIG-aaaa/0/0", "-1")

    assert resolve_mps_visibility_entries(entries, run_command=forbidden_command) == entries


def test_mps_visibility_entries_normalize_mixed_masks() -> None:
    resolved = resolve_mps_visibility_entries(
        ("GPU-bbbb", "0"),
        run_command=_fake_uuid_table_commands("0, GPU-aaaa\n"),
    )

    assert resolved == ("GPU-bbbb", "GPU-aaaa")


def test_mps_visibility_entries_unknown_index_fails_closed() -> None:
    with pytest.raises(ValueError, match="could not resolve"):
        resolve_mps_visibility_entries(
            ("3",),
            run_command=_fake_uuid_table_commands("0, GPU-aaaa\n"),
        )


def test_mps_visibility_entries_missing_host_table_fails_closed() -> None:
    def failing_command(command: list[str], **kwargs: Any) -> Any:
        raise RuntimeError("no nvidia-smi")

    with pytest.raises(ValueError, match="could not resolve"):
        resolve_mps_visibility_entries(("0",), run_command=failing_command)
