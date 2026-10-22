"""Tests for the explicit, user-owned CUDA MPS management CLI."""

from __future__ import annotations

import json
import os
import platform
import socket
import stat
import tempfile
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from unilab.training import cuda_mps_cli as mps

GPU_A = "GPU-aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"
GPU_B = "GPU-bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb"


class Completed:
    def __init__(self, stdout: str = "", stderr: str = "", returncode: int = 0) -> None:
        self.stdout = stdout
        self.stderr = stderr
        self.returncode = returncode


def fake_run() -> Any:
    def run(command: list[str], **_kwargs: Any) -> Completed:
        if command[0] == "nvidia-smi":
            return Completed(f"0, {GPU_A}, Default\n1, {GPU_B}, Default\n")
        raise AssertionError(f"unexpected command {command}")

    return run


def fake_platform(monkeypatch: pytest.MonkeyPatch, name: str = "Linux") -> None:
    monkeypatch.setattr(platform, "system", lambda: name)


def live_proc_stat(pid: int = 4242, start: int = 777) -> Any:
    return lambda checked: (pid, start) if checked == pid else None


def torch_with_uuid(uuid: str = GPU_A) -> Any:
    return SimpleNamespace(
        version=SimpleNamespace(hip=None),
        cuda=SimpleNamespace(
            is_available=lambda: True,
            device_count=lambda: 1,
            get_device_properties=lambda _index: SimpleNamespace(uuid=uuid),
        ),
    )


@pytest.fixture
def linux_host(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    fake_platform(monkeypatch)
    root = tmp_path / "registry"
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    (tmp_path / "home").mkdir()
    return root


def make_control(pipe_dir: Path) -> Path:
    pipe_dir.mkdir(parents=True, exist_ok=True)
    control = pipe_dir / "control"
    listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    listener.bind(str(control))
    control.chmod(0o666)
    return control


def daemon_record(root: Path, **overrides: Any) -> mps.DaemonRecord:
    values: dict[str, Any] = {
        "name": "test-daemon",
        "uid": os.getuid(),
        "host": platform.node() or "unknown-host",
        "topology_mode": mps.SINGLE_GPU_MODE,
        "gpu_uuids": (GPU_A,),
        "pipe_directory": str(root / "pipe"),
        "log_directory": str(root / "log"),
        "pid": 4242,
        "process_start_ticks": 777,
        "created_at": 1.0,
    }
    values.update(overrides)
    return mps.DaemonRecord(**values)


def write_record(root: Path, record: mps.DaemonRecord) -> Path:
    path = root / record.name / "daemon.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record.to_dict()), encoding="utf-8")
    return path


def test_visible_gpus_canonicalizes_uuids_and_visibility() -> None:
    gpus = mps.visible_gpus(GPU_A, fake_run())

    assert [(gpu.index, gpu.uuid) for gpu in gpus] == [(0, GPU_A)]


def test_bare_torch_gpu_uuid_is_canonicalized() -> None:
    assert mps._canonical_uuid("aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa") == GPU_A


def test_mig_selectors_are_explicitly_rejected() -> None:
    with pytest.raises(mps.CudaMpsError, match="MIG device"):
        mps.resolve_topology("MIG-abc", run_command=fake_run())


def test_multi_gpu_selector_parses_future_dp_shape_but_fails_closed() -> None:
    with pytest.raises(mps.CudaMpsError, match="single-task multi-GPU"):
        mps.resolve_topology(f"{GPU_A},{GPU_B}", run_command=fake_run())


def test_all_selector_reports_task_per_gpu_shape_but_fails_for_start() -> None:
    plan = mps.resolve_topology("all", run_command=fake_run())

    assert plan.topology_mode == mps.TASK_PER_GPU_MODE
    assert len(plan.gpus) == 2


def test_single_gpu_defaults_need_no_selector_or_name() -> None:
    plan = mps.resolve_topology(None, cuda_visible_devices=GPU_A, run_command=fake_run())

    assert plan.selector == "auto"
    assert plan.topology_mode == mps.SINGLE_GPU_MODE
    assert [gpu.uuid for gpu in plan.gpus] == [GPU_A]


def test_missing_selector_rejects_ambiguous_multi_gpu_host() -> None:
    with pytest.raises(mps.CudaMpsError, match="Specify --gpus"):
        mps.resolve_topology(None, run_command=fake_run())


def test_start_rejects_task_per_gpu_until_lease_contract_lands() -> None:
    with pytest.raises(mps.CudaMpsError, match="current support is single_gpu"):
        mps.start_daemon("all", root=Path("/unused"), run_command=fake_run())


def test_status_is_read_only_and_reports_compatible_topology_modes(
    linux_host: Path,
) -> None:
    status = mps.host_status(root=linux_host, run_command=fake_run())

    assert status.nvidia_smi_available
    assert [gpu.uuid for gpu in status.gpus] == [GPU_A, GPU_B]
    assert status.compatible_topology_modes == (
        mps.SINGLE_GPU_MODE,
        mps.SINGLE_TASK_MULTI_GPU_MODE,
        mps.TASK_PER_GPU_MODE,
    )


def test_status_reports_managed_control_pipe_and_diagnostic(
    linux_host: Path,
) -> None:
    record = daemon_record(linux_host, name="pipe-owner")
    write_record(linux_host, record)
    make_control(Path(record.pipe_directory))

    status = mps.host_status(root=linux_host, run_command=fake_run())

    assert status.control_pipes == (
        mps.ControlPipeStatus(
            path=str(Path(record.pipe_directory) / "control"),
            kind="unix_socket",
            managed_names=("pipe-owner",),
        ),
    )


def test_status_remains_useful_without_nvidia_smi(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_platform(monkeypatch, "Darwin")

    def unavailable(command: list[str], **_kwargs: Any) -> Completed:
        raise FileNotFoundError(command[0])

    status = mps.host_status(run_command=unavailable)

    assert status.platform == "Darwin"
    assert not status.nvidia_smi_available
    assert status.gpus == ()
    assert status.diagnostic is not None
    assert status.diagnostic.startswith("Could not execute nvidia-smi:")
    assert status.diagnostic.endswith("Verify the NVIDIA deployment and PATH.")


def test_record_identity_rejects_other_user_or_host() -> None:
    record = daemon_record(Path("/unused"), uid=os.getuid() + 1)

    with pytest.raises(mps.CudaMpsError, match="belongs to UID"):
        mps._validate_record_identity(record, uid=os.getuid(), host="here")


def test_proc_stat_parses_parenthesized_command_names(tmp_path: Path) -> None:
    stat = tmp_path / "stat"
    stat.write_text(
        "2551360 (nvidia-cuda-mps) S 977138 2551360 2551360 0 -1 4194368 103 0 0 0 0 0 0 0 0 20 0 2 131077450 0 0 0\n"
    )

    contents = stat.read_text()
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(mps.Path, "read_text", lambda _self, **_kwargs: contents)
        assert mps._proc_stat(2551360) == (2551360, 131077450)


def test_proc_stat_rejects_pid_mismatch(tmp_path: Path) -> None:
    stat = tmp_path / "stat"
    stat.write_text("999 (weird name) S 0 0 0 0 -1 0 0 0 0 0 0 0 0 20 0 2 0 1\n")

    contents = stat.read_text()
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(mps.Path, "read_text", lambda _self, **_kwargs: contents)
        assert mps._proc_stat(2551360) is None


def test_daemon_liveness_uses_pid_and_start_time() -> None:
    record = daemon_record(Path("/unused"))

    assert mps.daemon_process_is_live(record, proc_stat=live_proc_stat())
    assert not mps.daemon_process_is_live(
        daemon_record(Path("/unused"), pid=9999), proc_stat=live_proc_stat()
    )
    assert not mps.daemon_process_is_live(
        daemon_record(Path("/unused"), process_start_ticks=9999), proc_stat=live_proc_stat()
    )


def test_start_daemon_uses_explicit_directories_and_records_identity(
    linux_host: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def reject_default(_name: str) -> Path:
        raise AssertionError("explicit paths must not touch the default runtime root")

    monkeypatch.setattr(mps, "_default_runtime_root", reject_default)
    monkeypatch.setattr(mps.shutil, "which", lambda _name: "/fake/bin/nvidia-cuda-mps-control")
    pipe_dir = linux_host / "explicit-pipe"
    log_dir = linux_host / "explicit-log"
    started: list[tuple[list[str], dict[str, Any]]] = []

    def run(command: list[str], **kwargs: Any) -> Completed:
        started.append((command, kwargs))
        if command[0] == "nvidia-smi":
            return Completed(f"0, {GPU_A}, Default\n")
        if command == list(mps._DEFAULT_CONTROL_COMMAND):
            make_control(pipe_dir)
            (pipe_dir / "nvidia-cuda-mps-control.pid").write_text("4242\n")
            monkeypatch.setattr(mps, "_proc_stat", live_proc_stat())
            return Completed("")
        if command == ["nvidia-cuda-mps-control"]:
            assert kwargs["input"] == "get_server_list\n"
            return Completed("")
        raise AssertionError(f"unexpected command {command}")

    record = mps.start_daemon(
        GPU_A,
        name="prod",
        pipe_dir=str(pipe_dir),
        log_dir=str(log_dir),
        root=linux_host,
        run_command=run,
    )

    assert record.pid == 4242
    assert record.process_start_ticks == 777
    assert record.gpu_uuids == (GPU_A,)
    assert record.topology_mode == mps.SINGLE_GPU_MODE
    assert started[1][1]["env"]["CUDA_VISIBLE_DEVICES"] == GPU_A
    assert started[1][1]["env"]["CUDA_MPS_PIPE_DIRECTORY"] == str(pipe_dir)
    assert (linux_host / "prod" / "daemon.json").is_file()


def test_start_daemon_waits_until_control_answers_query(
    linux_host: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(mps.shutil, "which", lambda _name: "/fake/bin/nvidia-cuda-mps-control")
    pipe_dir = linux_host / "ready-pipe"
    log_dir = linux_host / "ready-log"
    queries: list[list[str]] = []

    def run(command: list[str], **kwargs: Any) -> Completed:
        if command[0] == "nvidia-smi":
            return Completed(f"0, {GPU_A}, Default\n")
        if command == list(mps._DEFAULT_CONTROL_COMMAND):
            make_control(pipe_dir)
            (pipe_dir / "nvidia-cuda-mps-control.pid").write_text("4242\n")
            monkeypatch.setattr(mps, "_proc_stat", live_proc_stat())
            return Completed("")
        if command == ["nvidia-cuda-mps-control"]:
            assert kwargs["input"] == "get_server_list\n"
            queries.append(command)
            assert kwargs["env"]["CUDA_MPS_PIPE_DIRECTORY"] == str(pipe_dir)
            assert kwargs["timeout"] > 0.25
            if len(queries) == 1:
                return Completed("control is starting", returncode=1)
            return Completed("")
        raise AssertionError(f"unexpected command {command}")

    record = mps.start_daemon(
        GPU_A,
        name="ready",
        pipe_dir=str(pipe_dir),
        log_dir=str(log_dir),
        root=linux_host,
        run_command=run,
    )

    assert len(queries) == 2
    assert record.pid == 4242


def test_start_daemon_cleans_up_unrecorded_control_after_readiness_failure(
    linux_host: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(mps.shutil, "which", lambda _name: "/fake/bin/nvidia-cuda-mps-control")
    pipe_dir = linux_host / "failed-pipe"
    log_dir = linux_host / "failed-log"
    quit_commands = 0

    def run(command: list[str], **kwargs: Any) -> Completed:
        nonlocal quit_commands
        if command[0] == "nvidia-smi":
            return Completed(f"0, {GPU_A}, Default\n")
        if command == list(mps._DEFAULT_CONTROL_COMMAND):
            make_control(pipe_dir)
            (pipe_dir / "nvidia-cuda-mps-control.pid").write_text("4242\n")
            monkeypatch.setattr(mps, "_proc_stat", lambda _pid: None)
            return Completed("")
        if command == list(mps._QUIT_COMMAND):
            assert kwargs["input"] == "quit\n"
            quit_commands += 1
            assert kwargs["env"]["CUDA_MPS_PIPE_DIRECTORY"] == str(pipe_dir)
            return Completed("")
        raise AssertionError(f"unexpected command {command}")

    monkeypatch.setattr(mps, "_wait_for_control_query", lambda *_args, **_kwargs: False)
    monkeypatch.setattr(
        mps,
        "_remove_orphan_default_control_sockets",
        lambda pipe_directory: pipe_directory.mkdir(parents=True, exist_ok=True),
    )

    with pytest.raises(mps.CudaMpsError, match="did not answer a readiness query"):
        mps.start_daemon(
            GPU_A,
            name="failed",
            pipe_dir=str(pipe_dir),
            log_dir=str(log_dir),
            root=linux_host,
            run_command=run,
        )

    assert quit_commands == 1
    assert not (linux_host / "failed" / "daemon.json").exists()


def test_default_daemon_name_uses_complete_uuid() -> None:
    name = mps._default_name((mps.GpuIdentity(0, GPU_A),))

    assert name == "gpu-aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"


def test_start_daemon_rejects_long_unix_socket_path(linux_host: Path) -> None:
    long_dir = linux_host / ("x" * 96)

    with pytest.raises(mps.CudaMpsError, match="MPS control socket path is too long"):
        mps.start_daemon(
            GPU_A,
            name="prod",
            pipe_dir=str(long_dir / "pipe"),
            log_dir=str(linux_host / "log"),
            root=linux_host,
            run_command=fake_run(),
        )


def test_start_daemon_recovers_only_inert_failed_control_artifacts(
    linux_host: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    short_root = tmp_path.parent / "u"
    monkeypatch.setattr(mps.tempfile, "gettempdir", lambda: str(short_root))
    monkeypatch.setattr(mps.shutil, "which", lambda _name: "/fake/bin/nvidia-cuda-mps-control")
    pipe_dir = short_root / "p" / "prod" / "pipe"
    pipe_dir.mkdir(parents=True)
    (pipe_dir / "control_lock").touch()
    log_fifo = pipe_dir / "log"
    os.mkfifo(log_fifo)
    preserved = pipe_dir / "unrelated"
    preserved.touch()
    (pipe_dir / "control").touch()

    with pytest.raises(mps.CudaMpsError, match="Refusing to attach"):
        mps.start_daemon(
            GPU_A,
            name="prod",
            pipe_dir=str(pipe_dir),
            log_dir=str(short_root / "p" / "prod" / "log"),
            root=linux_host,
            run_command=fake_run(),
        )

    assert not (pipe_dir / "control_lock").exists()
    assert not log_fifo.exists()
    assert preserved.exists()


def test_start_daemon_failure_reports_control_log_and_cleans_artifacts(
    linux_host: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(mps.shutil, "which", lambda _name: "/fake/bin/nvidia-cuda-mps-control")
    short_root = tmp_path.parent / "u2"
    monkeypatch.setattr(mps.tempfile, "gettempdir", lambda: str(short_root))
    pipe_dir = short_root / "p" / "prod" / "pipe"
    log_dir = short_root / "p" / "prod" / "log"
    log_dir.mkdir(parents=True)
    (log_dir / "control.log").write_text("control failed\n", encoding="utf-8")

    def run(command: list[str], **kwargs: Any) -> Completed:
        if command[0] == "nvidia-smi":
            return Completed(f"0, {GPU_A}, Default\n")
        if command == list(mps._DEFAULT_CONTROL_COMMAND):
            pipe_dir.mkdir(parents=True, exist_ok=True)
            (pipe_dir / "control_lock").touch()
            return Completed(returncode=1)
        raise AssertionError(f"unexpected command {command}")

    with pytest.raises(mps.CudaMpsError, match="control failed"):
        mps.start_daemon(
            GPU_A,
            name="prod",
            pipe_dir=str(pipe_dir),
            log_dir=str(log_dir),
            root=linux_host,
            run_command=run,
        )

    assert not (pipe_dir / "control_lock").exists()


def test_default_runtime_recovers_orphan_control_sockets() -> None:
    with tempfile.TemporaryDirectory() as raw:
        root = Path(raw)
        monkeypatch_root = root / "tmp"
        runtime = monkeypatch_root / "uni-cumps" / "gpu-test" / "pipe"
        runtime.mkdir(parents=True)
        control = runtime / "control"
        # Create actual socket nodes like NVIDIA leaves behind.
        socket.socket(socket.AF_UNIX, socket.SOCK_STREAM).bind(str(control))
        socket.socket(socket.AF_UNIX, socket.SOCK_STREAM).bind(str(runtime / "control_privileged"))

        with pytest.MonkeyPatch.context() as monkeypatch:
            monkeypatch.setattr(mps.tempfile, "gettempdir", lambda: str(monkeypatch_root))
            root_path = mps._default_runtime_root("gpu-test")
            mps._remove_orphan_default_control_sockets(root_path / "pipe")

        assert not control.exists()
        assert not (runtime / "control_privileged").exists()


def test_default_runtime_preserves_live_control_with_stale_pid_file(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with tempfile.TemporaryDirectory() as raw:
        runtime = Path(raw) / "uni-cumps" / "gpu-test" / "pipe"
        runtime.mkdir(parents=True)
        (runtime / "nvidia-cuda-mps-control.pid").write_text("4242\n", encoding="utf-8")
        control = runtime / "control"
        socket.socket(socket.AF_UNIX, socket.SOCK_STREAM).bind(str(control))

        monkeypatch.setattr(mps, "_proc_stat", live_proc_stat())
        monkeypatch.setattr(
            mps,
            "_process_name_is_alive",
            lambda pid, name: pid == 4242 and name.startswith("nvidia"),
        )
        mps._remove_orphan_default_control_sockets(runtime)

        assert control.exists()


def test_default_runtime_removes_orphan_socket_with_stale_pid_file() -> None:
    with tempfile.TemporaryDirectory() as raw:
        runtime = Path(raw) / "uni-cumps" / "gpu-test" / "pipe"
        runtime.mkdir(parents=True)
        (runtime / "nvidia-cuda-mps-control.pid").write_text("99999999\n", encoding="utf-8")
        control = runtime / "control"
        socket.socket(socket.AF_UNIX, socket.SOCK_STREAM).bind(str(control))

        mps._remove_orphan_default_control_sockets(runtime)

        assert not control.exists()


def test_start_daemon_refuses_existing_unmanaged_control_path(
    linux_host: Path,
) -> None:
    pipe_dir = linux_host / "pipe"
    pipe_dir.mkdir(parents=True)
    (pipe_dir / "control").touch()

    with pytest.raises(mps.CudaMpsError, match="Refusing to attach"):
        mps.start_daemon(
            GPU_A,
            name="prod",
            pipe_dir=str(pipe_dir),
            log_dir=str(linux_host / "log"),
            root=linux_host,
            run_command=fake_run(),
        )


def test_start_daemon_rejects_live_duplicate_record(
    linux_host: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    record = daemon_record(linux_host, name="prod")
    write_record(linux_host, record)
    monkeypatch.setattr(mps, "daemon_process_is_live", lambda _daemon, **_kwargs: True)

    def reject_start(command: list[str], **_kwargs: Any) -> Completed:
        if command[0] == "nvidia-smi":
            return Completed(f"0, {GPU_A}, Default\n")
        raise AssertionError("start must fail before invoking the control daemon")

    with pytest.raises(mps.CudaMpsError, match="already live"):
        mps.start_daemon(
            GPU_A,
            name="prod",
            pipe_dir=str(linux_host / "new-pipe"),
            log_dir=str(linux_host / "new-log"),
            root=linux_host,
            run_command=reject_start,
        )


def test_start_daemon_quarantines_stale_record(
    linux_host: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(mps.shutil, "which", lambda _name: "/fake/bin/nvidia-cuda-mps-control")
    write_record(linux_host, daemon_record(linux_host, name="prod"))
    monkeypatch.setattr(mps, "_proc_stat", lambda _pid: None)
    pipe_dir = linux_host / "new-pipe"

    def run(command: list[str], **kwargs: Any) -> Completed:
        if command[0] == "nvidia-smi":
            if "--query-gpu=index,uuid" in command:
                return Completed(f"0, {GPU_A}\n")
            return Completed(f"0, {GPU_A}, Default\n")
        if command == list(mps._DEFAULT_CONTROL_COMMAND):
            make_control(pipe_dir)
            (pipe_dir / "nvidia-cuda-mps-control.pid").write_text("4242\n")
            monkeypatch.setattr(mps, "_proc_stat", live_proc_stat())
            return Completed("")
        if command == ["nvidia-cuda-mps-control"]:
            assert kwargs["input"] == "get_server_list\n"
            return Completed("")
        raise AssertionError(f"unexpected command {command}")

    record = mps.start_daemon(
        GPU_A,
        name="prod",
        pipe_dir=str(pipe_dir),
        log_dir=str(linux_host / "new-log"),
        root=linux_host,
        run_command=run,
    )

    assert record.pid == 4242
    assert list((linux_host / ".stale").iterdir())


def test_select_environment_for_gpu_requires_sole_live_matching_record(
    linux_host: Path,
) -> None:
    record = daemon_record(linux_host, name="prod")
    write_record(linux_host, record)

    with pytest.raises(mps.CudaMpsError, match="No live UniLab-recorded CUDA MPS daemon"):
        mps.select_environment_for_gpu(GPU_A, root=linux_host, proc_stat=lambda _pid: None)

    values = mps.select_environment_for_gpu(
        GPU_A, root=linux_host, proc_stat=live_proc_stat(record.pid)
    )
    assert values["CUDA_MPS_PIPE_DIRECTORY"] == record.pipe_directory
    assert values["CUDA_MPS_LOG_DIRECTORY"] == record.log_directory

    other = daemon_record(
        linux_host,
        name="other",
        pid=4243,
        process_start_ticks=778,
        gpu_uuids=(GPU_A,),
    )
    write_record(linux_host, other)
    with pytest.raises(mps.CudaMpsError, match="Multiple live UniLab-recorded daemons"):
        mps.select_environment_for_gpu(
            GPU_A,
            root=linux_host,
            proc_stat=lambda pid: (pid, 777 if pid == record.pid else 778),
        )


def test_environment_for_daemon_does_not_mutate_process_environment(
    linux_host: Path,
) -> None:
    before = os.environ.get("CUDA_VISIBLE_DEVICES")
    values = mps.environment_for_daemon(daemon_record(linux_host))

    assert values == {
        "CUDA_VISIBLE_DEVICES": GPU_A,
        "CUDA_MPS_PIPE_DIRECTORY": str(linux_host / "pipe"),
        "CUDA_MPS_LOG_DIRECTORY": str(linux_host / "log"),
    }
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == before


def test_stop_daemon_sends_quit_only_through_owned_record(
    linux_host: Path,
) -> None:
    write_record(linux_host, daemon_record(linux_host))
    commands: list[tuple[list[str], dict[str, Any]]] = []

    def run(command: list[str], **kwargs: Any) -> Completed:
        commands.append((command, kwargs))
        return Completed("")

    stopped = mps.stop_daemon(
        "test-daemon", root=linux_host, run_command=run, proc_stat=live_proc_stat()
    )

    assert stopped.pid == 4242
    assert commands == [(list(mps._QUIT_COMMAND), commands[0][1])]
    assert commands[0][1]["input"] == "quit\n"
    assert commands[0][1]["env"]["CUDA_MPS_PIPE_DIRECTORY"] == str(linux_host / "pipe")
    assert not (linux_host / "test-daemon" / "daemon.json").exists()


def test_stop_without_record_cannot_quit_unmanaged_daemon(linux_host: Path) -> None:
    with pytest.raises(mps.CudaMpsError, match="UniLab can stop only recorded daemons"):
        mps.stop_daemon("unknown", root=linux_host, run_command=fake_run())


def test_stop_stale_record_removes_record_without_contacting_daemon(
    linux_host: Path,
) -> None:
    path = write_record(linux_host, daemon_record(linux_host))

    with pytest.raises(mps.CudaMpsError, match="stale"):
        mps.stop_daemon("test-daemon", root=linux_host, run_command=fake_run())

    assert not path.exists()


def test_stop_all_scopes_to_current_user_host_and_managed_records(
    linux_host: Path,
) -> None:
    write_record(linux_host, daemon_record(linux_host, name="a"))
    write_record(linux_host, daemon_record(linux_host, name="b", uid=os.getuid() + 1))
    commands: list[list[str]] = []

    def run(command: list[str], **_kwargs: Any) -> Completed:
        commands.append(command)
        return Completed("")

    stopped = mps.stop_all_daemons(root=linux_host, run_command=run, proc_stat=live_proc_stat())

    assert [daemon.name for daemon in stopped] == ["a"]
    assert commands == [list(mps._QUIT_COMMAND)]
    assert (linux_host / "a" / "daemon.json").exists() is False
    assert (linux_host / "b" / "daemon.json").exists()


def test_doctor_uses_sole_live_managed_daemon_pipe(
    linux_host: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake_platform(monkeypatch)
    record = daemon_record(linux_host, name="prod")
    write_record(linux_host, record)
    make_control(Path(record.pipe_directory))
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", GPU_A)
    monkeypatch.setattr(mps, "daemon_process_is_live", lambda _daemon, **_kwargs: True)
    commands: list[tuple[list[str], dict[str, Any]]] = []

    def run(command: list[str], **kwargs: Any) -> Completed:
        commands.append((command, kwargs))
        if command[0] == "nvidia-smi":
            if "--query-gpu=index,uuid" in command:
                return Completed(f"0, {GPU_A}\n")
            return Completed(f"0, {GPU_A}, Default\n")
        if command == ["nvidia-cuda-mps-control"]:
            assert kwargs["input"] == "get_server_list\n"
            return Completed("4242\n")
        raise AssertionError(f"unexpected command {command}")

    monkeypatch.setenv("CUDA_MPS_PIPE_DIRECTORY", record.pipe_directory)
    monkeypatch.setenv("CUDA_MPS_LOG_DIRECTORY", record.log_directory)
    result = mps.doctor(
        None,
        root=linux_host,
        cuda_visible_devices=GPU_A,
        run_command=run,
        torch_module=torch_with_uuid(),
    )

    assert result["valid"] is True
    assert result["cuda_process_sharing"]["control_pipe"] == str(
        Path(record.pipe_directory) / "control"
    )
    daemon_query = commands[-1]
    assert daemon_query[0] == ["nvidia-cuda-mps-control"]
    assert daemon_query[1]["input"] == "get_server_list\n"
    assert daemon_query[1]["env"]["CUDA_MPS_PIPE_DIRECTORY"] == record.pipe_directory


def test_doctor_reports_missing_control_daemon_and_exits_invalid(
    linux_host: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake_platform(monkeypatch)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", GPU_A)
    pipe_dir = linux_host / "pipe"
    make_control(pipe_dir)
    monkeypatch.setenv("CUDA_MPS_PIPE_DIRECTORY", str(pipe_dir))
    monkeypatch.setenv("CUDA_MPS_LOG_DIRECTORY", str(linux_host / "log"))
    monkeypatch.setattr(mps, "_proc_stat", live_proc_stat())

    def run(command: list[str], **kwargs: Any) -> Completed:
        if command[0] == "nvidia-smi":
            if "--query-gpu=index,uuid" in command:
                return Completed(f"0, {GPU_A}\n")
            return Completed(f"0, {GPU_A}, Default\n")
        if command == ["nvidia-cuda-mps-control"]:
            assert kwargs["input"] == "get_server_list\n"
            return Completed("Cannot find MPS control daemon process", returncode=1)
        raise AssertionError(f"unexpected command {command}")

    result = mps.doctor("0", root=linux_host, run_command=run, torch_module=torch_with_uuid())

    assert result["plan"]["gpus"] == [{"index": 0, "uuid": GPU_A}]
    assert result["valid"] is False
    assert "could not reach the control daemon" in result["cuda_process_sharing"]["error"]


def test_default_daemon_record_selects_sole_live_record(
    linux_host: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    write_record(linux_host, daemon_record(linux_host, name="prod"))
    monkeypatch.setattr(mps, "daemon_process_is_live", lambda _daemon, **_kwargs: True)

    assert mps.read_daemon_record(linux_host, None).name == "prod"


def test_default_daemon_record_rejects_zero_or_multiple_live_records(
    linux_host: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with pytest.raises(mps.CudaMpsError, match="no live UniLab-recorded daemon"):
        mps.read_daemon_record(linux_host, None)

    write_record(linux_host, daemon_record(linux_host, name="a"))
    write_record(linux_host, daemon_record(linux_host, name="b"))
    monkeypatch.setattr(mps, "daemon_process_is_live", lambda _daemon, **_kwargs: True)

    with pytest.raises(mps.CudaMpsError, match="multiple live daemons"):
        mps.read_daemon_record(linux_host, None)


def test_cli_parser_accepts_flat_command_namespace() -> None:
    args = mps._parser().parse_args(["status", "--json"])

    assert args.command == "status"
    assert args.json


def test_cli_status_json_is_machine_readable(
    capsys: pytest.CaptureFixture[str], linux_host: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake_platform(monkeypatch, "Darwin")

    def unavailable(command: list[str], **_kwargs: Any) -> Completed:
        raise FileNotFoundError(command[0])

    def unavailable_visibility(
        _cuda_visible_devices: str | None, _run_command: Any
    ) -> tuple[mps.GpuIdentity, ...]:
        raise mps.CudaMpsError("nvidia-smi unavailable")

    monkeypatch.setattr(mps, "visible_gpus", unavailable_visibility)

    assert mps.main(["status", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["platform"] == "Darwin"
    assert payload["compatible_topology_modes"] == [
        "single_gpu",
        "single_task_multi_gpu",
        "task_per_gpu",
    ]


def test_cli_env_prints_posix_exports_without_mutating_environment(
    capsys: pytest.CaptureFixture[str], linux_host: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    write_record(linux_host, daemon_record(linux_host, name="prod"))
    monkeypatch.setattr(mps, "_proc_stat", live_proc_stat())
    monkeypatch.setattr(
        mps, "read_daemon_record", lambda _root, name: daemon_record(linux_host, name=name)
    )
    monkeypatch.setattr(mps, "daemon_process_is_live", lambda _daemon, **_kwargs: True)

    assert mps.main(["env", "--name", "prod"]) == 0
    output = capsys.readouterr().out
    assert f"export CUDA_VISIBLE_DEVICES={GPU_A}" in output
    assert "export CUDA_MPS_PIPE_DIRECTORY=" in output
    assert "CUDA_VISIBLE_DEVICES" not in os.environ or os.environ["CUDA_VISIBLE_DEVICES"] != GPU_A


def test_completion_offers_cumps_commands() -> None:
    from unilab import cli_completion

    assert "uni-cumps" in cli_completion.complete_words(["uv", "run", "uni-"], 2)
    assert cli_completion.complete_words(["uv", "run", "uni-cumps", ""], 3) == [
        "status",
        "doctor",
        "start",
        "stop",
        "env",
    ]
    assert cli_completion.complete_words(["uv", "run", "uni-cumps", "env", "-"], 4) == [
        "--name",
        "--shell",
        "--json",
    ]


def test_regular_file_control_pipe_is_rejected_by_probe(linux_host: Path) -> None:
    control = make_control(linux_host / "pipe")
    control.unlink()
    control.touch()
    assert not stat.S_ISSOCK(control.stat().st_mode)
    del linux_host
