"""User-owned CUDA MPS daemon management for the UniLab CLI.

This module owns command-level lifecycle operations for explicitly requested
MPS control daemons.  It intentionally treats daemon ownership as an external,
security-sensitive host service: every stop is gated by a UniLab daemon record
containing the recorded process identity, and records are scoped by user and
host.  It never mutates the calling shell's CUDA environment.

The topology model distinguishes single-GPU, single-task multi-GPU, and
task-per-GPU placements.  Only single-GPU execution is currently accepted; the
other shapes fail closed rather than silently choosing a future topology.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import re
import shlex
import shutil
import stat
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
from unittest.mock import patch

from unilab.training.cuda_process_sharing import canonicalize_gpu_uuid, probe_cuda_process_sharing

ALL_SELECTOR = "all"
SINGLE_GPU_MODE = "single_gpu"
SINGLE_TASK_MULTI_GPU_MODE = "single_task_multi_gpu"
TASK_PER_GPU_MODE = "task_per_gpu"
COMPATIBLE_TOPOLOGY_MODES = (SINGLE_GPU_MODE, SINGLE_TASK_MULTI_GPU_MODE, TASK_PER_GPU_MODE)
_SUPPORTED_TOPOLOGY_MODES = (SINGLE_GPU_MODE,)
_NAME_PATTERN = re.compile(r"^[a-z0-9][a-z0-9-]{0,63}$")
_DEFAULT_CONTROL_COMMAND = ("nvidia-cuda-mps-control", "-d")
_QUIT_COMMAND = ("nvidia-cuda-mps-control",)
_SERVER_LIST_QUERY_COMMAND = ("nvidia-cuda-mps-control",)
_MAX_CONTROL_SOCKET_BYTES = 95
_INERT_CONTROL_ARTIFACTS = ("control_lock", "log")
_GPU_QUERY_COMMAND = (
    "nvidia-smi",
    "--query-gpu=index,uuid,compute_mode",
    "--format=csv,noheader",
)


class CudaMpsError(RuntimeError):
    """A user-actionable, fail-closed CUDA MPS CLI error."""


class StaleCudaMpsDaemonError(CudaMpsError):
    """A recorded daemon identity is no longer live; its record was cleaned."""


@dataclass(frozen=True)
class GpuIdentity:
    """Canonical GPU identity suitable for persisted topology records."""

    index: int
    uuid: str

    def to_dict(self) -> dict[str, str | int]:
        return {"index": self.index, "uuid": self.uuid}


@dataclass(frozen=True)
class DaemonRecord:
    """A UniLab-owned daemon identity; absence means no stop authority."""

    name: str
    uid: int
    host: str
    topology_mode: str
    gpu_uuids: tuple[str, ...]
    pipe_directory: str
    log_directory: str
    pid: int
    process_start_ticks: int
    created_at: float

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["gpu_uuids"] = list(self.gpu_uuids)
        return value


@dataclass(frozen=True)
class ControlPipeStatus:
    """A discovered control pipe and whether a UniLab record owns it."""

    path: str
    kind: str
    managed_names: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "kind": self.kind,
            "managed_names": list(self.managed_names),
        }


@dataclass(frozen=True)
class HostStatus:
    """Read-only host, control-pipe, and managed-daemon status."""

    platform: str
    nvidia_smi_available: bool
    cuda_visible_devices: str | None
    gpus: tuple[GpuIdentity, ...]
    control_pipes: tuple[ControlPipeStatus, ...]
    managed_daemons: tuple[DaemonRecord, ...]
    diagnostic: str | None = None
    compatible_topology_modes: tuple[str, ...] = COMPATIBLE_TOPOLOGY_MODES

    def to_dict(self) -> dict[str, Any]:
        return {
            "platform": self.platform,
            "nvidia_smi_available": self.nvidia_smi_available,
            "diagnostic": self.diagnostic,
            "cuda_visible_devices": self.cuda_visible_devices,
            "gpus": [gpu.to_dict() for gpu in self.gpus],
            "control_pipes": [pipe.to_dict() for pipe in self.control_pipes],
            "managed_daemons": [daemon.to_dict() for daemon in self.managed_daemons],
            "compatible_topology_modes": list(self.compatible_topology_modes),
        }


@dataclass(frozen=True)
class TopologyPlan:
    """A resolved selector before any host service is created."""

    selector: str
    topology_mode: str
    gpus: tuple[GpuIdentity, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "selector": self.selector,
            "topology_mode": self.topology_mode,
            "gpus": [gpu.to_dict() for gpu in self.gpus],
        }


RunCommand = Callable[..., subprocess.CompletedProcess[str]]


def _run_command(
    run_command: RunCommand,
    command: Sequence[str],
    *,
    timeout: float = 5.0,
    env: Mapping[str, str] | None = None,
    input: str | None = None,
) -> str:
    try:
        kwargs: dict[str, Any] = {
            "text": True,
            "capture_output": True,
            "timeout": timeout,
            "check": False,
        }
        if env is not None:
            kwargs["env"] = dict(env)
        if input is not None:
            kwargs["input"] = input
        result = run_command(list(command), **kwargs)
    except (OSError, subprocess.SubprocessError) as exc:
        raise CudaMpsError(
            f"Could not execute {command[0]}: {exc!r}. Verify the NVIDIA deployment and PATH."
        ) from exc
    if result.returncode != 0:
        detail = (result.stderr or result.stdout or "").strip()
        suffix = f": {detail}" if detail else ""
        raise CudaMpsError(f"{command[0]} failed with exit code {result.returncode}{suffix}.")
    return result.stdout or ""


def _canonical_uuid(value: str, *, field: str = "uuid") -> str:
    uuid = value.strip()
    if uuid.upper().startswith("MIG-"):
        raise CudaMpsError(
            f"MIG device {value!r} is not supported by uni-cumps yet; select a physical GPU UUID."
        )
    # Torch properties commonly expose the bare UUID without NVIDIA's GPU- tag.
    if not uuid.upper().startswith("GPU-") and len(uuid) == 36 and uuid.count("-") == 4:
        uuid = f"GPU-{uuid}"
    if not uuid.upper().startswith("GPU-"):
        raise CudaMpsError(f"Invalid NVIDIA GPU {field}: {value!r}; expected a GPU-<id> UUID.")
    token = canonicalize_gpu_uuid(uuid)
    if not token:
        raise CudaMpsError(f"Invalid NVIDIA GPU {field}: {value!r}; expected a GPU-<id> UUID.")
    # Persisted/display form: the shared comparison token re-rendered in the
    # canonical dashed ``GPU-<id>`` shape (nvidia-smi emits lowercase hex).
    if len(token) == 32:
        token = f"{token[:8]}-{token[8:12]}-{token[12:16]}-{token[16:20]}-{token[20:]}"
    return f"GPU-{token.lower()}"


def _parse_gpu_rows(output: str) -> tuple[GpuIdentity, ...]:
    gpus: list[GpuIdentity] = []
    for raw_line in output.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        fields = [field.strip() for field in line.split(",")]
        if len(fields) != 3 or not fields[0].isdigit():
            raise CudaMpsError(f"Could not parse nvidia-smi GPU row: {raw_line!r}.")
        index, uuid, _compute_mode = fields
        gpus.append(GpuIdentity(int(index), _canonical_uuid(uuid)))
    return tuple(gpus)


def discover_gpus(
    run_command: RunCommand = subprocess.run,
) -> tuple[GpuIdentity, ...]:
    """Discover physical host GPUs through the public ``nvidia-smi`` CLI."""

    output = _run_command(run_command, _GPU_QUERY_COMMAND)
    return _parse_gpu_rows(output)


def _visible_entries(value: str | None) -> tuple[str, ...]:
    if value is None or not value.strip() or value.strip() == "all":
        return ()
    return tuple(entry.strip() for entry in value.split(",") if entry.strip())


def _apply_visibility(
    gpus: Sequence[GpuIdentity], cuda_visible_devices: str | None
) -> tuple[GpuIdentity, ...]:
    entries = _visible_entries(cuda_visible_devices)
    if not entries:
        return tuple(gpus)
    selected: list[GpuIdentity] = []
    for entry in entries:
        upper = entry.upper()
        if upper.isdigit():
            matches = [gpu for gpu in gpus if gpu.index == int(upper)]
            if not matches:
                raise CudaMpsError(f"CUDA_VISIBLE_DEVICES entry {entry!r} has no visible host GPU.")
            selected.extend(matches)
            continue
        canonical = _canonical_uuid(entry)
        matches = [gpu for gpu in gpus if gpu.uuid == canonical]
        if not matches:
            raise CudaMpsError(
                f"CUDA_VISIBLE_DEVICES entry {entry!r} has no visible host GPU UUID."
            )
        selected.extend(matches)
    return tuple(selected)


def visible_gpus(
    cuda_visible_devices: str | None = None,
    run_command: RunCommand = subprocess.run,
) -> tuple[GpuIdentity, ...]:
    """Return canonical GPU identities visible under the current host mask."""

    return _apply_visibility(discover_gpus(run_command), cuda_visible_devices)


def _name_is_safe(name: str) -> bool:
    return _NAME_PATTERN.fullmatch(name) is not None


def _record_root(root: Path | None = None) -> Path:
    base = root if root is not None else Path.home() / ".cache" / "unilab" / "cuda-mps"
    try:
        base.expanduser().resolve()
    except OSError as exc:
        raise CudaMpsError(f"Could not resolve CUDA MPS record directory {base}: {exc}.") from exc
    return base


def _validate_record_identity(
    daemon: DaemonRecord, *, uid: int | None = None, host: str | None = None
) -> None:
    current_uid = os.getuid() if uid is None else uid
    current_host = _hostname() if host is None else host
    if daemon.uid != current_uid or daemon.host != current_host:
        raise CudaMpsError(
            f"Daemon record {daemon.name!r} belongs to UID {daemon.uid} on host "
            f"{daemon.host!r}; this process is UID {current_uid} on {current_host!r}."
        )


def _daemon_record_path(root: Path, name: str) -> Path:
    if not _name_is_safe(name):
        raise CudaMpsError(
            "Daemon name must match [a-z0-9][a-z0-9-]{0,63}; use a label, not a path."
        )
    return root / name / "daemon.json"


def _record_from_mapping(mapping: Mapping[str, Any], name: str) -> DaemonRecord:
    required = (
        "uid",
        "host",
        "topology_mode",
        "gpu_uuids",
        "pipe_directory",
        "log_directory",
        "pid",
        "process_start_ticks",
        "created_at",
    )
    missing = [key for key in required if key not in mapping]
    if missing:
        raise CudaMpsError(
            f"Daemon record {name!r} is missing required fields: {', '.join(missing)}."
        )
    try:
        return DaemonRecord(
            name=name,
            uid=int(mapping["uid"]),
            host=str(mapping["host"]),
            topology_mode=str(mapping["topology_mode"]),
            gpu_uuids=tuple(str(value) for value in mapping["gpu_uuids"]),
            pipe_directory=str(mapping["pipe_directory"]),
            log_directory=str(mapping["log_directory"]),
            pid=int(mapping["pid"]),
            process_start_ticks=int(mapping["process_start_ticks"]),
            created_at=float(mapping["created_at"]),
        )
    except (TypeError, ValueError) as exc:
        raise CudaMpsError(f"Daemon record {name!r} contains invalid values: {exc}.") from exc


def read_daemon_record(
    root: Path | None,
    name: str | None,
    *,
    uid: int | None = None,
    host: str | None = None,
) -> DaemonRecord:
    """Load and scope-check a UniLab daemon record.

    ``name=None`` selects the sole current-user/current-host live daemon. The
    common single-GPU workflow therefore needs no daemon label.
    """

    if name is None:
        live = [
            daemon
            for daemon in list_daemon_records(root, uid=uid, host=host)
            if daemon_process_is_live(daemon)
        ]
        if len(live) == 1:
            return live[0]
        detail = (
            "no live UniLab-recorded daemon exists"
            if not live
            else f"multiple live daemons exist: {', '.join(daemon.name for daemon in live)}"
        )
        raise CudaMpsError(
            f"Cannot select a default daemon because {detail}; pass --name explicitly."
        )
    path = _daemon_record_path(_record_root(root), name)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise CudaMpsError(
            f"No UniLab CUDA MPS daemon record named {name!r}. UniLab can stop only recorded daemons."
        ) from exc
    except (OSError, json.JSONDecodeError) as exc:
        raise CudaMpsError(f"Could not read daemon record {path}: {exc}.") from exc
    if not isinstance(payload, dict):
        raise CudaMpsError(f"Daemon record {path} must contain a JSON object.")
    daemon = _record_from_mapping(payload, name)
    _validate_record_identity(daemon, uid=uid, host=host)
    return daemon


def list_daemon_records(
    root: Path | None = None, *, uid: int | None = None, host: str | None = None
) -> tuple[DaemonRecord, ...]:
    """List current-user/current-host daemon records without mutating them."""

    base = _record_root(root)
    records: list[DaemonRecord] = []
    if not base.exists():
        return tuple(records)
    try:
        children = sorted(base.iterdir(), key=lambda path: path.name)
    except OSError as exc:
        raise CudaMpsError(f"Could not inspect CUDA MPS record directory {base}: {exc}.") from exc
    current_uid = os.getuid() if uid is None else uid
    current_host = _hostname() if host is None else host
    for child in children:
        if not child.is_dir() or not _name_is_safe(child.name):
            continue
        path = child / "daemon.json"
        if not path.is_file():
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(payload, dict):
            continue
        try:
            daemon = _record_from_mapping(payload, child.name)
        except CudaMpsError:
            continue
        if daemon.uid == current_uid and daemon.host == current_host:
            records.append(daemon)
    return tuple(records)


def _hostname() -> str:
    return platform.node() or "unknown-host"


def _proc_stat(pid: int) -> tuple[int, int] | None:
    """Parse procfs PID/start-time without treating ``(comm)`` as an integer.

    Linux writes the process name as a parenthesized field and it may contain
    spaces.  Split only after the final closing parenthesis, then account for
    the two fields already consumed by ``pid`` and ``comm``.
    """

    try:
        text = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8")
        pid_text = text.split(" ", 1)[0]
        after_comm = text[text.rindex(")") + 2 :]
        tail = after_comm.split()
    except (FileNotFoundError, ProcessLookupError, PermissionError, OSError, ValueError):
        return None
    if len(tail) <= 19 or pid_text != str(pid):
        return None
    try:
        # Fields after ``comm`` start at procfs field 3. starttime is field 22,
        # hence index 19 in this tail.
        return int(pid_text), int(tail[19])
    except ValueError:
        return None


def _process_name_is_alive(pid: int, expected_name: str) -> bool:
    """Match a procfs comm name, allowing its conventional 15-character limit."""

    try:
        text = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8")
        process_name = text[text.index("(") + 1 : text.rindex(")")].strip()
    except (OSError, ValueError):
        return False
    return process_name == expected_name or process_name.startswith(expected_name[:15])


def daemon_process_is_live(
    daemon: DaemonRecord, *, proc_stat: Callable[[int], tuple[int, int] | None] = _proc_stat
) -> bool:
    """Check both PID and Linux process start-time to avoid PID reuse."""

    identity = proc_stat(daemon.pid)
    return identity is not None and identity == (daemon.pid, daemon.process_start_ticks)


def _control_pipe_kind(path: Path) -> str | None:
    try:
        mode = path.stat().st_mode
    except OSError:
        return None
    if stat.S_ISSOCK(mode):
        return "unix_socket"
    if stat.S_ISFIFO(mode):
        return "fifo"
    return None


def _wait_for_control(pipe_directory: Path, timeout: float = 5.0) -> bool:
    control = pipe_directory / "control"
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            mode = control.stat().st_mode
        except FileNotFoundError:
            time.sleep(0.05)
            continue
        except OSError:
            return False
        if stat.S_ISSOCK(mode) or stat.S_ISFIFO(mode):
            return True
        time.sleep(0.05)
    return False


def _wait_for_control_query(
    environment: Mapping[str, str],
    run_command: RunCommand,
    *,
    timeout: float = 5.0,
) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        remaining = max(0.0, deadline - time.monotonic())
        try:
            _run_command(
                run_command,
                _SERVER_LIST_QUERY_COMMAND,
                input="get_server_list\n",
                timeout=remaining,
                env=environment,
            )
        except CudaMpsError:
            time.sleep(min(0.05, max(0.0, deadline - time.monotonic())))
            continue
        return True
    return False


def _stop_unrecorded_control(
    environment: Mapping[str, str],
    run_command: RunCommand,
) -> bool:
    try:
        _run_command(run_command, _QUIT_COMMAND, input="quit\n", env=environment)
    except CudaMpsError:
        return False
    return True


def _remove_inert_control_artifacts(pipe_directory: Path) -> None:
    """Remove only artifacts left by a failed control launch; preserve sockets."""

    if not pipe_directory.exists():
        return
    for name in _INERT_CONTROL_ARTIFACTS:
        path = pipe_directory / name
        try:
            if path.is_fifo() or (name == "control_lock" and path.is_file()):
                path.unlink()
        except OSError as exc:
            raise CudaMpsError(f"Could not clean failed-control artifact {path}: {exc}.") from exc


def _remove_orphan_default_control_sockets(pipe_directory: Path) -> None:
    """Remove UniLab-default sockets orphaned after a control process exit.

    Explicit user-supplied directories are never modified. In the default
    runtime root, NVIDIA leaves ``control`` and ``control_privileged`` sockets
    after the control process exits, which otherwise blocks every restart. A
    stale PID file alone is not ownership evidence: remove the sockets only
    when its PID is absent or no longer has the expected control process name.
    """

    identity = _control_daemon_identity(pipe_directory)
    if identity is not None and _process_name_is_alive(identity[0], "nvidia-cuda-mps-control"):
        return
    for name in ("control", "control_privileged"):
        path = pipe_directory / name
        try:
            if path.is_socket():
                path.unlink()
        except OSError as exc:
            raise CudaMpsError(f"Could not clean orphan MPS socket {path}: {exc}.") from exc


def _control_log_tail(log_directory: Path, limit: int = 12) -> str | None:
    path = log_directory / "control.log"
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return None
    tail = "\n".join(lines[-limit:]).strip()
    return tail or None


def _control_daemon_identity(pipe_directory: Path) -> tuple[int, int] | None:
    pid: int | None = None
    try:
        pid = int(
            (pipe_directory / "nvidia-cuda-mps-control.pid").read_text(encoding="utf-8").strip()
        )
    except (OSError, ValueError):
        return None
    return _proc_stat(pid)


def _default_name(gpus: Sequence[GpuIdentity]) -> str:
    """Use the complete UUID so shortened prefixes cannot collide."""

    uuid = gpus[0].uuid.removeprefix("GPU-").lower()
    return f"gpu-{uuid}"


def _default_runtime_root(daemon_name: str) -> Path:
    """Return a short, user-private runtime root safe for Unix socket paths."""

    root = Path(tempfile.gettempdir()) / "uni-cumps" / daemon_name
    base = root.parent
    try:
        base.mkdir(mode=0o700, parents=True, exist_ok=True)
        root.mkdir(mode=0o700, parents=True, exist_ok=True)
        root.chmod(0o700)
    except OSError as exc:
        raise CudaMpsError(
            f"Could not create user-owned MPS runtime directory {root}: {exc}."
        ) from exc
    return root


def _write_daemon_record(root: Path, daemon: DaemonRecord) -> Path:
    directory = root / daemon.name
    path = directory / "daemon.json"
    try:
        directory.mkdir(parents=True, exist_ok=True)
        if path.exists():
            raise CudaMpsError(f"Refusing to overwrite daemon record {path}.")
        path.write_text(
            json.dumps(daemon.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    except OSError as exc:
        raise CudaMpsError(f"Could not write daemon record {path}: {exc}.") from exc
    return path


def _unlink_record(root: Path, daemon: DaemonRecord) -> None:
    path = _daemon_record_path(root, daemon.name)
    try:
        path.unlink()
        path.parent.rmdir()
    except OSError as exc:
        raise CudaMpsError(f"Could not remove daemon record {path}: {exc}.") from exc


def _resolve_topology(selector: str | None, gpus: Sequence[GpuIdentity]) -> TopologyPlan:
    if selector is None:
        if len(gpus) == 1:
            return TopologyPlan("auto", SINGLE_GPU_MODE, tuple(gpus))
        raise CudaMpsError(
            "Multiple GPUs are visible. Specify --gpus <index-or-uuid> to select one GPU."
        )
    values = [entry.strip() for entry in selector.split(",") if entry.strip()]
    if selector.strip() == ALL_SELECTOR:
        return TopologyPlan(ALL_SELECTOR, TASK_PER_GPU_MODE, tuple(gpus))
    if not values:
        raise CudaMpsError("GPU selector must not be empty.")
    if len(values) > 1:
        raise CudaMpsError(
            "Multi-GPU MPS selectors parse but are not supported yet. A multi-GPU selector "
            "resembles single-task multi-GPU training (discussion #2063 / roadmap #1672); "
            "run one explicit daemon per future topology until that gate lands."
        )
    value = values[0]
    if value.upper().isdigit():
        matches = [gpu for gpu in gpus if gpu.index == int(value)]
        selected = tuple(matches)
    else:
        canonical = _canonical_uuid(value)
        matches = [gpu for gpu in gpus if gpu.uuid == canonical]
        selected = tuple(matches)
    if not selected:
        raise CudaMpsError(
            f"GPU selector {value!r} does not match a visible physical GPU UUID or index."
        )
    return TopologyPlan(value, SINGLE_GPU_MODE, selected)


def resolve_topology(
    selector: str | None,
    *,
    cuda_visible_devices: str | None = None,
    run_command: RunCommand = subprocess.run,
) -> TopologyPlan:
    """Resolve selector syntax to canonical GPU UUIDs and a topology mode.

    ``None`` selects the single visible GPU. On a single-GPU host this is the
    default CLI path: users need no selector, daemon name, or custom paths.
    """

    visible = visible_gpus(cuda_visible_devices, run_command)
    return _resolve_topology(selector, visible)


def daemon_status(
    daemon: DaemonRecord,
    *,
    proc_stat: Callable[[int], tuple[int, int] | None] = _proc_stat,
) -> dict[str, Any]:
    """Return record identity plus PID/start-time liveness evidence."""

    return {
        **daemon.to_dict(),
        "process_live": daemon_process_is_live(daemon, proc_stat=proc_stat),
        "record_state": "live" if daemon_process_is_live(daemon, proc_stat=proc_stat) else "stale",
    }


def host_status(
    root: Path | None = None,
    *,
    cuda_visible_devices: str | None = None,
    run_command: RunCommand = subprocess.run,
    uid: int | None = None,
    host: str | None = None,
) -> HostStatus:
    """Build read-only host capability and managed-daemon status."""

    visible: tuple[GpuIdentity, ...] = ()
    diagnostic: str | None = None
    try:
        visible = visible_gpus(cuda_visible_devices, run_command)
        available = True
    except CudaMpsError as exc:
        available = False
        diagnostic = str(exc)
    daemons = list_daemon_records(root, uid=uid, host=host)
    by_pipe: dict[str, list[str]] = {}
    for daemon in daemons:
        by_pipe.setdefault(daemon.pipe_directory, []).append(daemon.name)
    pipes: list[ControlPipeStatus] = []
    for pipe_directory, names in sorted(by_pipe.items()):
        path = Path(pipe_directory) / "control"
        try:
            path.stat()
        except OSError:
            continue
        kind = _control_pipe_kind(path)
        if kind is None:
            continue
        pipes.append(
            ControlPipeStatus(
                path=str(path),
                kind=kind,
                managed_names=tuple(sorted(names)),
            )
        )
    return HostStatus(
        platform=platform.system(),
        nvidia_smi_available=available,
        cuda_visible_devices=cuda_visible_devices,
        gpus=visible,
        control_pipes=tuple(pipes),
        managed_daemons=daemons,
        diagnostic=diagnostic,
    )


def _require_linux_nvidia(gpus: Sequence[GpuIdentity]) -> None:
    if platform.system() != "Linux":
        raise CudaMpsError(
            f"CUDA MPS management requires Linux and NVIDIA CUDA; current platform is {platform.system()!r}."
        )
    if not gpus:
        raise CudaMpsError("No visible physical NVIDIA CUDA GPU was found.")


def _require_executable(name: str) -> None:
    if shutil.which(name) is None:
        raise CudaMpsError(
            f"Required executable {name!r} was not found on PATH; install the NVIDIA deployment."
        )


def start_daemon(
    selector: str | None,
    *,
    name: str | None = None,
    pipe_dir: str | None = None,
    log_dir: str | None = None,
    root: Path | None = None,
    cuda_visible_devices: str | None = None,
    run_command: RunCommand = subprocess.run,
    daemon: bool = True,
    foreground_runner: Callable[[Mapping[str, str]], int] | None = None,
) -> DaemonRecord:
    """Start an explicitly requested, user-owned daemon and record its identity.

    The caller remains responsible for exporting the resulting environment into
    a training launcher; this command never changes its parent shell.
    """

    gpus = visible_gpus(cuda_visible_devices, run_command)
    _require_linux_nvidia(gpus)
    plan = _resolve_topology(selector, gpus)
    if plan.topology_mode not in _SUPPORTED_TOPOLOGY_MODES:
        raise CudaMpsError(
            f"Topology mode {plan.topology_mode!r} is not supported yet; current support is single_gpu."
        )
    daemon_name = name or _default_name(plan.gpus)
    if not _name_is_safe(daemon_name):
        raise CudaMpsError(
            "Daemon name must match [a-z0-9][a-z0-9-]{0,63}; use a label, not a path."
        )
    base = _record_root(root)
    try:
        existing_path = _daemon_record_path(base, daemon_name)
        if existing_path.exists():
            existing = read_daemon_record(base, daemon_name)
            if daemon_process_is_live(existing):
                raise CudaMpsError(
                    f"Recorded daemon {daemon_name!r} is already live with PID {existing.pid}. "
                    "Reuse `uni-cumps env` or stop it explicitly first."
                )
            quarantine = base / ".stale" / f"{daemon_name}-{int(time.time() * 1000)}"
            quarantine.parent.mkdir(parents=True, exist_ok=True)
            existing_path.parent.rename(quarantine)
    except CudaMpsError:
        raise
    except OSError as exc:
        raise CudaMpsError(f"Could not prepare daemon record for {daemon_name!r}: {exc}.") from exc
    # Fully explicit directories must work even if another user's daemon owns
    # the default temporary root.
    runtime_base = (
        _default_runtime_root(daemon_name) if pipe_dir is None or log_dir is None else Path(".")
    )
    using_default_runtime = pipe_dir is None
    pipe_value = pipe_dir or str(runtime_base / "pipe")
    log_value = log_dir or str(runtime_base / "log")
    try:
        pipe_directory = Path(pipe_value).expanduser().resolve(strict=False)
        log_directory = Path(log_value).expanduser().resolve(strict=False)
    except OSError as exc:
        raise CudaMpsError(f"Could not resolve MPS pipe/log directories: {exc}.") from exc
    if pipe_directory == log_directory:
        raise CudaMpsError("MPS pipe and log directories must be different absolute paths.")
    control = pipe_directory / "control"
    control_bytes = len(str(control).encode())
    if control_bytes > _MAX_CONTROL_SOCKET_BYTES:
        raise CudaMpsError(
            f"MPS control socket path is too long ({control_bytes} bytes; maximum "
            f"{_MAX_CONTROL_SOCKET_BYTES}): {control}. Choose a shorter --pipe-dir, such as "
            "/tmp/<short-name>/pipe."
        )
    _remove_inert_control_artifacts(pipe_directory)
    if using_default_runtime:
        _remove_orphan_default_control_sockets(pipe_directory)
    if control.exists():
        raise CudaMpsError(
            f"Refusing to attach to an existing control path {control}; remove/rename it or choose a new daemon."
        )
    _require_executable("nvidia-cuda-mps-control")
    env = {
        **os.environ,
        "CUDA_VISIBLE_DEVICES": ",".join(gpu.uuid for gpu in plan.gpus),
        "CUDA_MPS_PIPE_DIRECTORY": str(pipe_directory),
        "CUDA_MPS_LOG_DIRECTORY": str(log_directory),
    }
    try:
        pipe_directory.mkdir(mode=0o700, parents=True, exist_ok=True)
        log_directory.mkdir(mode=0o700, parents=True, exist_ok=True)
        pipe_directory.chmod(0o700)
        log_directory.chmod(0o700)
    except OSError as exc:
        raise CudaMpsError(f"Could not create user-owned MPS directories: {exc}.") from exc
    command = list(_DEFAULT_CONTROL_COMMAND)
    if daemon:
        try:
            _run_command(run_command, command, env=env)
        except CudaMpsError as exc:
            _remove_inert_control_artifacts(pipe_directory)
            tail = _control_log_tail(log_directory)
            detail = f" Control log tail:\n{tail}" if tail is not None else ""
            raise CudaMpsError(f"{exc}{detail}") from exc
    else:
        runner = foreground_runner if foreground_runner is not None else _foreground_control
        return_code = runner(env)
        if return_code != 0:
            raise CudaMpsError(
                f"Foreground MPS control exited with code {return_code}; no daemon record was written."
            )
    if not _wait_for_control(pipe_directory):
        raise CudaMpsError(
            f"MPS control did not create a usable control pipe at {control}. "
            "Inspect CUDA_MPS_LOG_DIRECTORY and the host deployment."
        )
    if not _wait_for_control_query(env, run_command):
        stopped = _stop_unrecorded_control(env, run_command)
        if using_default_runtime:
            _remove_orphan_default_control_sockets(pipe_directory)
        tail = _control_log_tail(log_directory)
        detail = f" Control log tail:\n{tail}" if tail is not None else ""
        if not stopped:
            detail += " The launched control daemon may still be running."
        raise CudaMpsError(
            "MPS control created a pipe but did not answer a readiness query. "
            f"Inspect CUDA_MPS_LOG_DIRECTORY and the host deployment.{detail}"
        )
    process_identity = _control_daemon_identity(pipe_directory)
    if process_identity is None:
        raise CudaMpsError(
            "MPS control did not expose a verifiable nvidia-cuda-mps-control PID; "
            "refusing to record a daemon UniLab cannot safely stop."
        )
    record = DaemonRecord(
        name=daemon_name,
        uid=os.getuid(),
        host=_hostname(),
        topology_mode=plan.topology_mode,
        gpu_uuids=tuple(gpu.uuid for gpu in plan.gpus),
        pipe_directory=str(pipe_directory),
        log_directory=str(log_directory),
        pid=process_identity[0],
        process_start_ticks=process_identity[1],
        created_at=time.time(),
    )
    _write_daemon_record(base, record)
    return record


def _foreground_control(env: Mapping[str, str]) -> int:
    try:
        return subprocess.run(
            ["nvidia-cuda-mps-control"],
            env=dict(env),
            check=False,
        ).returncode
    except KeyboardInterrupt:
        return 130


def environment_for_daemon(daemon: DaemonRecord) -> dict[str, str]:
    """Return launcher-facing CUDA environment without mutating this process."""

    return {
        "CUDA_VISIBLE_DEVICES": ",".join(daemon.gpu_uuids),
        "CUDA_MPS_PIPE_DIRECTORY": daemon.pipe_directory,
        "CUDA_MPS_LOG_DIRECTORY": daemon.log_directory,
    }


def select_environment_for_gpu(
    gpu_uuid: str,
    *,
    root: Path | None = None,
    proc_stat: Callable[[int], tuple[int, int] | None] = _proc_stat,
) -> dict[str, str]:
    """Select the sole live user-owned daemon environment for one GPU.

    This is launcher-facing data only. It does not mutate the current process,
    start a daemon, stop a daemon, or provide stop authority.
    """

    canonical = _canonical_uuid(gpu_uuid)
    matching = [
        daemon
        for daemon in list_daemon_records(root)
        if canonical in daemon.gpu_uuids and daemon_process_is_live(daemon, proc_stat=proc_stat)
    ]
    if not matching:
        raise CudaMpsError(
            f"No live UniLab-recorded CUDA MPS daemon covers GPU {canonical}. "
            "Run `uni-cumps start` first or set CUDA_MPS_PIPE_DIRECTORY explicitly."
        )
    if len(matching) > 1:
        names = ", ".join(daemon.name for daemon in matching)
        raise CudaMpsError(
            f"Multiple live UniLab-recorded daemons cover GPU {canonical}: {names}. "
            "Set CUDA_MPS_PIPE_DIRECTORY to the intended daemon explicitly."
        )
    return environment_for_daemon(matching[0])


def stop_daemon(
    name: str,
    *,
    root: Path | None = None,
    run_command: RunCommand = subprocess.run,
    proc_stat: Callable[[int], tuple[int, int] | None] = _proc_stat,
) -> DaemonRecord:
    """Stop only a daemon whose UniLab record and process identity still match."""

    base = _record_root(root)
    daemon = read_daemon_record(base, name)
    if not daemon_process_is_live(daemon, proc_stat=proc_stat):
        _unlink_record(base, daemon)
        raise StaleCudaMpsDaemonError(
            f"Recorded daemon {name!r} is stale; removed its record without contacting an MPS control."
        )
    env = {
        **os.environ,
        "CUDA_MPS_PIPE_DIRECTORY": daemon.pipe_directory,
        "CUDA_MPS_LOG_DIRECTORY": daemon.log_directory,
    }
    _run_command(run_command, _QUIT_COMMAND, input="quit\n", env=env)  # type: ignore[call-overload]
    _unlink_record(base, daemon)
    return daemon


def stop_all_daemons(
    *,
    root: Path | None = None,
    run_command: RunCommand = subprocess.run,
    proc_stat: Callable[[int], tuple[int, int] | None] = _proc_stat,
) -> tuple[DaemonRecord, ...]:
    """Stop every current-user/current-host daemon recorded by UniLab."""

    stopped: list[DaemonRecord] = []
    for daemon in list_daemon_records(root):
        try:
            stop_daemon(daemon.name, root=root, run_command=run_command, proc_stat=proc_stat)
        except StaleCudaMpsDaemonError:
            # A stale record has already been cleaned; continue stopping live ones.
            continue
        stopped.append(daemon)
    return tuple(stopped)


def doctor(
    selector: str | None,
    *,
    root: Path | None = None,
    cuda_visible_devices: str | None = None,
    run_command: RunCommand = subprocess.run,
    torch_module: Any = None,
    uid: int | None = None,
    host: str | None = None,
) -> dict[str, Any]:
    """Validate topology and daemon state without creating or changing anything."""

    visible = visible_gpus(cuda_visible_devices, run_command)
    _require_linux_nvidia(visible)
    plan = _resolve_topology(selector=selector, gpus=visible)
    if plan.topology_mode not in _SUPPORTED_TOPOLOGY_MODES:
        raise CudaMpsError(
            f"Topology mode {plan.topology_mode!r} is not supported yet; current support is single_gpu."
        )
    # Doctor diagnoses host state; it does not rely on an already-created training env.
    if cuda_visible_devices is None:
        cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
        visible = _apply_visibility(visible, cuda_visible_devices)
        plan = _resolve_topology(selector=selector, gpus=visible)
    matching = [
        daemon
        for daemon in list_daemon_records(root, uid=uid, host=host)
        if tuple(daemon.gpu_uuids) == tuple(gpu.uuid for gpu in plan.gpus)
        and daemon.topology_mode == plan.topology_mode
    ]
    managed = [daemon_status(daemon) for daemon in matching]
    live = [daemon for daemon in matching if daemon_process_is_live(daemon)]
    selected_daemon = live[0] if len(live) == 1 else None
    try:
        probe_environment = {
            "CUDA_VISIBLE_DEVICES": ",".join(gpu.uuid for gpu in plan.gpus),
        }
        if selected_daemon is not None:
            probe_environment["CUDA_MPS_PIPE_DIRECTORY"] = selected_daemon.pipe_directory
            probe_environment["CUDA_MPS_LOG_DIRECTORY"] = selected_daemon.log_directory
        environment_patch = patch.dict(os.environ, probe_environment)
        with environment_patch:
            evidence = probe_cuda_process_sharing(
                "mps",
                "cuda:0",
                "cuda:0",
                backend="mjwarp",
                torch_module=torch_module,
                run_command=lambda command, **kwargs: run_command(
                    command,
                    **{
                        **kwargs,
                        "env": {
                            **kwargs.get("env", {}),
                            **probe_environment,
                        },
                    },
                ),
            )
        training_validation = evidence.to_dict()
        valid = evidence.validated
    except Exception as exc:  # probe errors are diagnostics, not doctor termination
        training_validation = {"error": str(exc)}
        valid = False
    return {
        "compatible_topology_modes": list(COMPATIBLE_TOPOLOGY_MODES),
        "plan": plan.to_dict(),
        "managed_daemons": managed,
        "cuda_process_sharing": training_validation,
        "valid": valid,
    }


def _print_json(value: Mapping[str, Any]) -> None:
    print(json.dumps(value, indent=2, sort_keys=True))


def _parser() -> argparse.ArgumentParser:
    return _mps_parser()


def _mps_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="uni-cumps")
    subparsers = parser.add_subparsers(dest="command", required=True)
    status_parser = subparsers.add_parser("status", help="show read-only host and daemon status")
    status_parser.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    doctor_parser = subparsers.add_parser("doctor", help="validate a topology without mutating it")
    doctor_parser.add_argument(
        "--gpus",
        default=None,
        help="one GPU index/UUID (default: the sole visible GPU)",
    )
    doctor_parser.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    start_parser = subparsers.add_parser("start", help="start a user-owned MPS control daemon")
    start_parser.add_argument(
        "--gpus",
        default=None,
        help="one GPU index/UUID (default: the sole visible GPU)",
    )
    start_parser.add_argument(
        "--name", default=None, help="daemon label (default: gpu-<uuid-prefix>)"
    )
    start_parser.add_argument(
        "--pipe-dir", default=None, help="pipe directory (default: user-owned UniLab cache)"
    )
    start_parser.add_argument(
        "--log-dir", default=None, help="log directory (default: user-owned UniLab cache)"
    )
    start_mode = start_parser.add_mutually_exclusive_group()
    start_mode.add_argument("--daemon", action="store_true", help="run the control daemon detached")
    start_mode.add_argument("--foreground", action="store_true", help="run control in this process")
    stop_parser = subparsers.add_parser("stop", help="stop a UniLab-recorded daemon")
    stop_parser.add_argument(
        "--name", default=None, help="daemon label (default: the sole live daemon)"
    )
    stop_parser.add_argument("--all", action="store_true", help="stop all current-user records")
    env_parser = subparsers.add_parser(
        "env", help="print launcher environment for a recorded daemon"
    )
    env_parser.add_argument(
        "--name", default=None, help="daemon label (default: the sole live daemon)"
    )
    env_parser.add_argument("--shell", choices=("posix",), default="posix")
    env_parser.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the CUDA MPS management CLI and return a process exit code."""

    args = _parser().parse_args(argv)
    try:
        if args.command == "status":
            status = host_status()
            if args.json:
                _print_json(status.to_dict())
            else:
                print(
                    f"platform={status.platform} nvidia_smi_available={status.nvidia_smi_available}"
                )
                for gpu in status.gpus:
                    print(f"gpu index={gpu.index} uuid={gpu.uuid}")
                if not status.managed_daemons:
                    print("managed_daemons=none")
                for daemon in status.managed_daemons:
                    state = "live" if daemon_process_is_live(daemon) else "stale"
                    print(
                        f"daemon name={daemon.name} state={state} pid={daemon.pid} "
                        f"mode={daemon.topology_mode} gpus={','.join(daemon.gpu_uuids)}"
                    )
                print("compatible_topology_modes=" + ",".join(COMPATIBLE_TOPOLOGY_MODES))
            return 0

        if args.command == "doctor":
            result = doctor(args.gpus)
            if args.json:
                _print_json(result)
            else:
                print(f"topology_mode={result['plan']['topology_mode']}")
                print(f"gpus={','.join(gpu['uuid'] for gpu in result['plan']['gpus'])}")
                print(f"valid={result['valid']}")
                for daemon in result["managed_daemons"]:
                    print(
                        f"daemon name={daemon['name']} state={daemon['record_state']} pid={daemon['pid']}"
                    )
                error = result["cuda_process_sharing"].get("error")
                if error:
                    print(f"error={error}")
            return 0 if result["valid"] else 1

        if args.command == "start":
            record = start_daemon(
                args.gpus,
                name=args.name,
                pipe_dir=args.pipe_dir,
                log_dir=args.log_dir,
                daemon=not args.foreground,
            )
            _print_json(record.to_dict())
            return 0

        if args.command == "stop":
            if args.all:
                stopped = stop_all_daemons()
            else:
                stopped = (stop_daemon(args.name),)
            _print_json({"stopped": [daemon.to_dict() for daemon in stopped]})
            return 0

        if args.command == "env":
            daemon = read_daemon_record(None, args.name)
            if not daemon_process_is_live(daemon):
                raise CudaMpsError(
                    f"Recorded daemon {args.name!r} is stale; start it again before requesting env."
                )
            values = environment_for_daemon(daemon)
            if args.json:
                _print_json({"name": daemon.name, "env": values})
            else:
                for key, value in values.items():
                    print(f"export {key}={shlex.quote(value)}")
            return 0

        raise AssertionError(f"Unhandled command: {args.command}")
    except CudaMpsError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
