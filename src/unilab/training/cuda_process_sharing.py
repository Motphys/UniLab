"""Fail-closed validation for CUDA process-sharing execution modes.

The off-policy learner and collector are separate processes on one rank-local
GPU.  CUDA MPS is an execution-sharing mode for that required topology: it lets
their independent CUDA contexts overlap kernels instead of being arbitrated by
coarse driver time-slicing.

This module never starts or stops an MPS daemon.  An explicit request is either
validated against the host state or rejected before UniLab constructs a learner,
collector, or environment.

An unconfigured (``null``) request still records ambient host MPS state: an
environment-level daemon reachable through ``CUDA_MPS_PIPE_DIRECTORY`` (or the
default pipe directory) multiplexes this rank's CUDA contexts transparently,
so every run must be able to self-report that state.  Ambient observation is
best-effort and never fails closed; unresolvable facts record ``None``.
"""

from __future__ import annotations

import os
import platform
import stat
import subprocess
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import MutableMapping

SUPPORTED_CUDA_PROCESS_SHARING_BACKENDS = frozenset({"mjwarp"})
_DEFAULT_PIPE_DIRECTORY = "/tmp/nvidia-mps"
_MPS_CONTROL_QUERY_TIMEOUT_SEC = 5.0


@dataclass(frozen=True)
class CudaProcessSharingEvidence:
    """Structured per-rank evidence for a requested mode plus ambient state.

    ``pipe_directory_env`` is the raw ``CUDA_MPS_PIPE_DIRECTORY`` value (``None``
    when unset); ``pipe_directory``/``control_pipe``/``control_pipe_kind``/
    ``server_pid`` describe the effective pipe this rank's CUDA contexts would
    use.  ``ambient_server`` is true only for an unconfigured run whose
    effective control pipe answered a daemon query — the transparent-MPS case.
    """

    configured: str | None
    effective: str | None
    validated: bool
    rank: int = 0
    world_size: int = 1
    learner_device: str | None = None
    collector_device: str | None = None
    learner_gpu_uuid: str | None = None
    collector_gpu_uuid: str | None = None
    pipe_directory_env: str | None = None
    pipe_directory: str | None = None
    control_pipe: str | None = None
    control_pipe_kind: str | None = None
    server_pid: int | None = None
    ambient_server: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Return the JSON-shaped diagnostics stored in a runtime manifest."""
        return {
            "configured": self.configured,
            "effective": self.effective,
            "validated": self.validated,
            "rank": self.rank,
            "world_size": self.world_size,
            "learner_device": self.learner_device,
            "collector_device": self.collector_device,
            "learner_gpu_uuid": self.learner_gpu_uuid,
            "collector_gpu_uuid": self.collector_gpu_uuid,
            "pipe_directory_env": self.pipe_directory_env,
            "pipe_directory": self.pipe_directory,
            "control_pipe": self.control_pipe,
            "control_pipe_kind": self.control_pipe_kind,
            "server_pid": self.server_pid,
            "ambient_server": self.ambient_server,
        }


def _configured_mode(requested: Any) -> str | None:
    if requested is None:
        return None
    if isinstance(requested, str) and requested.strip().lower() == "mps":
        return "mps"
    raise ValueError(
        "Unsupported training.cuda_process_sharing="
        f"{requested!r}; expected null or 'mps'. An explicit 'mps' request "
        "never falls back to multi-context CUDA execution."
    )


def _cuda_index(device: str, requested: str) -> int:
    value = device.strip()
    if value.lower() == "cuda":
        return 0
    base, separator, index_text = value.partition(":")
    if base.lower() != "cuda" or not separator or not index_text:
        raise ValueError(
            "training.cuda_process_sharing='mps' requires CUDA learner and "
            f"collector devices; got learner_device={device!r}."
        )
    try:
        index = int(index_text)
    except ValueError as exc:
        raise ValueError(
            f"training.cuda_process_sharing='mps' received invalid CUDA device {device!r}."
        ) from exc
    if index < 0:
        raise ValueError(
            f"training.cuda_process_sharing='mps' received invalid CUDA device {device!r}."
        )
    return index


def _torch_device_uuid(
    torch_module: Any,
    index: int,
    *,
    device_kind: str,
) -> str:
    if not torch_module.cuda.is_available():
        raise ValueError(
            "training.cuda_process_sharing requires CUDA, but CUDA is "
            f"unavailable for the {device_kind} device cuda:{index}."
        )
    if torch_module.version.hip is not None:
        raise ValueError(
            "training.cuda_process_sharing supports NVIDIA CUDA only; "
            "this process reports a ROCm/HIP Torch build."
        )
    device_count = int(torch_module.cuda.device_count())
    if index >= device_count:
        raise ValueError(
            f"training.cuda_process_sharing {device_kind} device "
            f"cuda:{index} is out of range; torch.cuda.device_count()={device_count}."
        )
    properties = torch_module.cuda.get_device_properties(index)
    uuid = str(getattr(properties, "uuid", "") or "").strip()
    if not uuid:
        raise ValueError(
            f"training.cuda_process_sharing could not resolve the {device_kind} CUDA device UUID."
        )
    return canonicalize_gpu_uuid(uuid)


def _parse_csv_row(row: str) -> list[str]:
    return [field.strip() for field in row.split(",")]


def canonicalize_gpu_uuid(value: str) -> str:
    """Return the case- and dash-insensitive NVIDIA GPU UUID comparison token.

    Torch, ``nvidia-smi``, and ``CUDA_VISIBLE_DEVICES`` expose the same UUID
    with different cases, dashes, and an optional ``GPU-`` prefix; equality
    checks must compare this token rather than the raw strings.  The
    persisted/display form stays ``GPU-<id>`` (see ``cuda_mps_cli``).
    """
    return value.strip().upper().removeprefix("GPU-").replace("-", "")


def _nvidia_uuid_table(run_command: Callable[..., Any]) -> dict[str, str] | None:
    """Return the host ``nvidia-smi`` index→UUID table, or None when unavailable."""
    try:
        result = run_command(
            ["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader"],
            text=True,
            capture_output=True,
            timeout=_MPS_CONTROL_QUERY_TIMEOUT_SEC,
            check=True,
        )
    except Exception:
        return None
    stdout = getattr(result, "stdout", None)
    if stdout is None:
        return None
    rows = [_parse_csv_row(row) for row in str(stdout).splitlines() if row.strip()]
    if not rows or any(len(row) != 2 for row in rows):
        return None
    return {row[0]: row[1] for row in rows if row[0].isdigit()}


def _nvidia_uuid(
    run_command: Callable[..., Any],
    visible_entries: Sequence[str],
    index: int,
) -> str | None:
    """Map a visibility-relative index to its host UUID, if evidence exists."""
    table = _nvidia_uuid_table(run_command)
    if table is None:
        return None
    by_host_index = {key: canonicalize_gpu_uuid(value) for key, value in table.items()}
    if index < len(visible_entries):
        token = canonicalize_gpu_uuid(visible_entries[index])
        if token.isdigit():
            return by_host_index.get(token)
        if token in by_host_index.values():
            return token
    return by_host_index.get(str(index))


def resolve_mps_visibility_entries(
    entries: Sequence[str],
    *,
    run_command: Callable[..., Any] = subprocess.run,
) -> tuple[str, ...]:
    """Return ``CUDA_VISIBLE_DEVICES`` entries with numeric indices as UUIDs.

    A per-GPU MPS server enumerates only its own device, so a client whose
    one-entry visibility mask holds a nonzero numeric index overruns that
    namespace and fails CUDA initialization with "No CUDA GPUs are available".
    UUID entries survive that server-side remapping, so an explicit
    ``training.cuda_process_sharing=mps`` launch normalizes visibility before
    CUDA latches the mask.  Non-numeric entries (UUID, MIG, ``-1``) pass
    through unchanged; the returned UUIDs keep the ``nvidia-smi`` display form.

    Raises:
        ValueError: A numeric entry could not be mapped to a host GPU UUID.
    """
    if not any(entry.isdigit() for entry in entries):
        return tuple(entries)
    table = _nvidia_uuid_table(run_command)
    normalized: list[str] = []
    for entry in entries:
        if not entry.isdigit():
            normalized.append(entry)
            continue
        uuid = None if table is None else table.get(entry)
        if uuid is None:
            raise ValueError(
                "training.cuda_process_sharing='mps' could not resolve "
                f"CUDA_VISIBLE_DEVICES entry {entry!r} to a host GPU UUID; MPS "
                "clients require UUID visibility entries. Check "
                "nvidia-smi --query-gpu=index,uuid --format=csv,noheader."
            )
        normalized.append(uuid)
    return tuple(normalized)


def _server_evidence(
    run_command: Callable[..., Any],
    control_pipe: Path,
) -> int | None:
    """Query the existing control daemon; never launch or terminate one."""
    environment = os.environ.copy()
    environment["CUDA_MPS_PIPE_DIRECTORY"] = str(control_pipe.parent)
    try:
        result = run_command(
            ["nvidia-cuda-mps-control"],
            input="get_server_list\n",
            text=True,
            capture_output=True,
            timeout=_MPS_CONTROL_QUERY_TIMEOUT_SEC,
            check=True,
            env=environment,
        )
    except Exception:
        return None
    queried_pid: int | None = None
    for line in str(result.stdout).splitlines():
        line = line.strip()
        if line.isdigit():
            queried_pid = int(line)
            break
    pid_file = control_pipe.parent / "nvidia-cuda-mps-control.pid"
    try:
        file_pid = int(pid_file.read_text(encoding="utf-8").strip())
    except (OSError, ValueError):
        file_pid = None
    return queried_pid if queried_pid is not None else file_pid


def _start_command(pipe_directory: Path) -> str:
    log_directory = os.environ.get("CUDA_MPS_LOG_DIRECTORY")
    log_suffix = (
        log_directory if log_directory is not None else f"{pipe_directory.parent.as_posix()}/log"
    )
    return (
        "mkdir -p "
        f"{pipe_directory.as_posix()} {log_suffix} && "
        f"export CUDA_MPS_PIPE_DIRECTORY={pipe_directory.as_posix()} && "
        f"export CUDA_MPS_LOG_DIRECTORY={log_suffix} && "
        "nvidia-cuda-mps-control -d"
    )


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


def _visible_entries(raw: str | None) -> tuple[str, ...]:
    if raw is None or not raw.strip() or raw.strip() == "all":
        return ()
    return tuple(entry.strip() for entry in raw.split(",") if entry.strip())


def _ambient_cuda_index(device: str | None) -> int | None:
    """Parse a CUDA device string without rejecting other device families."""

    if device is None:
        return None
    value = str(device).strip().lower()
    if value == "cuda":
        return 0
    base, separator, index_text = value.partition(":")
    if base != "cuda" or not separator:
        return None
    try:
        index = int(index_text)
    except ValueError:
        return None
    return index if index >= 0 else None


def _probe_ambient(
    evidence: CudaProcessSharingEvidence,
    *,
    torch_module: Any | None,
    run_command: Callable[..., Any],
) -> CudaProcessSharingEvidence:
    """Record ambient host MPS state for an unconfigured rank.

    Environment-level MPS multiplexes this rank's CUDA contexts transparently
    when the effective pipe directory holds a live control pipe, so an
    unconfigured run still self-reports pipe, daemon, and GPU identity facts.
    Nothing here validates, starts, or stops anything: every host query is
    best-effort and unresolved facts record ``None``.
    """

    pipe_env = os.environ.get("CUDA_MPS_PIPE_DIRECTORY")
    evidence = replace(evidence, pipe_directory_env=pipe_env)
    learner_index = _ambient_cuda_index(evidence.learner_device)
    collector_index = _ambient_cuda_index(evidence.collector_device)
    if learner_index is None and collector_index is None:
        return evidence

    pipe_directory = Path(pipe_env or _DEFAULT_PIPE_DIRECTORY)
    control_pipe = pipe_directory / "control"
    pipe_kind = _control_pipe_kind(control_pipe)
    server_pid = _server_evidence(run_command, control_pipe) if pipe_kind is not None else None

    learner_uuid: str | None = None
    if learner_index is not None:
        try:
            if torch_module is None:
                import torch

                torch_module = torch
            learner_uuid = _torch_device_uuid(torch_module, learner_index, device_kind="learner")
        except Exception:
            learner_uuid = None
    collector_uuid: str | None = None
    if collector_index is not None:
        visible_entries = _visible_entries(os.environ.get("CUDA_VISIBLE_DEVICES"))
        collector_uuid = _nvidia_uuid(run_command, visible_entries, collector_index)

    return replace(
        evidence,
        pipe_directory=str(pipe_directory),
        control_pipe=str(control_pipe),
        control_pipe_kind=pipe_kind,
        server_pid=server_pid,
        learner_gpu_uuid=learner_uuid,
        collector_gpu_uuid=collector_uuid,
        ambient_server=server_pid is not None,
    )


def probe_cuda_process_sharing(
    requested: Any,
    learner_device: str,
    collector_device: str | None,
    *,
    backend: str,
    rank: int = 0,
    world_size: int = 1,
    torch_module: Any | None = None,
    run_command: Callable[..., Any] = subprocess.run,
) -> CudaProcessSharingEvidence:
    """Validate one rank's requested CUDA process-sharing mode.

    Args:
        requested: Owner value ``null`` or ``mps``.
        learner_device: Rank-local learner CUDA device.
        collector_device: Backend-bound collector CUDA device, when applicable.
        backend: Configured owner backend identity.
        rank: Current off-policy DP rank.
        world_size: Current single-host off-policy DP world size.
        torch_module: Injectable Torch module for deterministic tests.
        run_command: Injectable subprocess runner for deterministic tests.

    Returns:
        Evidence suitable for a per-rank runtime-manifest diagnostic section.

    Raises:
        ValueError: The first unmet prerequisite for an explicit request.
    """

    configured = _configured_mode(requested)
    if configured is None:
        return _probe_ambient(
            CudaProcessSharingEvidence(
                configured=None,
                effective=None,
                validated=False,
                rank=rank,
                world_size=world_size,
                learner_device=str(learner_device),
                collector_device=str(collector_device) if collector_device is not None else None,
            ),
            torch_module=torch_module,
            run_command=run_command,
        )

    normalized_backend = str(backend).strip().lower()
    if normalized_backend not in SUPPORTED_CUDA_PROCESS_SHARING_BACKENDS:
        raise ValueError(
            "training.cuda_process_sharing='mps' supports only the mjwarp "
            f"backend in this release; got training.sim_backend={backend!r}."
        )
    if platform.system() != "Linux":
        raise ValueError(
            "training.cuda_process_sharing='mps' requires Linux; "
            f"got platform.system()={platform.system()!r}."
        )
    if (
        isinstance(rank, bool)
        or not isinstance(rank, int)
        or rank < 0
        or isinstance(world_size, bool)
        or not isinstance(world_size, int)
        or world_size < 1
        or rank >= world_size
    ):
        raise ValueError(
            "training.cuda_process_sharing='mps' requires a valid rank in "
            f"[0, world_size); got rank={rank!r}, world_size={world_size!r}."
        )
    if collector_device is None:
        raise ValueError(
            "training.cuda_process_sharing='mps' requires a CUDA collector "
            "process device; the mjwarp learner/collector rank-local binding "
            "did not produce one."
        )

    learner_index = _cuda_index(str(learner_device), configured)
    collector_index = _cuda_index(str(collector_device), configured)
    pipe_directory = Path(os.environ.get("CUDA_MPS_PIPE_DIRECTORY") or _DEFAULT_PIPE_DIRECTORY)
    control_pipe = pipe_directory / "control"
    visible_entries = _visible_entries(os.environ.get("CUDA_VISIBLE_DEVICES"))
    evidence = CudaProcessSharingEvidence(
        configured=configured,
        effective=configured,
        validated=True,
        rank=rank,
        world_size=world_size,
        learner_device=f"cuda:{learner_index}",
        collector_device=f"cuda:{collector_index}",
        pipe_directory_env=os.environ.get("CUDA_MPS_PIPE_DIRECTORY"),
        pipe_directory=str(pipe_directory),
        control_pipe=str(control_pipe),
    )

    if torch_module is None:
        import torch

        torch_module = torch
    learner_uuid = _torch_device_uuid(
        torch_module,
        learner_index,
        device_kind="learner",
    )
    collector_uuid = _nvidia_uuid(run_command, visible_entries, collector_index)
    evidence = replace(
        evidence,
        learner_gpu_uuid=learner_uuid,
        collector_gpu_uuid=collector_uuid,
    )

    if learner_uuid is not None and collector_uuid is not None and learner_uuid != collector_uuid:
        raise ValueError(
            "training.cuda_process_sharing='mps' requires the learner and "
            "collector to share one physical GPU. Resolved UUIDs: "
            f"learner={learner_uuid!r}, collector={collector_uuid!r}."
        )
    if collector_uuid is None:
        raise ValueError(
            "training.cuda_process_sharing='mps' could not resolve the "
            f"collector CUDA UUID for {collector_device!r}; run nvidia-smi "
            "--query-gpu=index,uuid --format=csv and verify the rank-local mask."
        )

    if not control_pipe.exists():
        raise ValueError(
            "training.cuda_process_sharing='mps' found no control pipe at "
            f"{control_pipe}. Start an existing host daemon with: "
            f"{_start_command(pipe_directory)}"
        )
    pipe_kind = _control_pipe_kind(control_pipe)
    if pipe_kind is None:
        raise ValueError(
            "training.cuda_process_sharing='mps' requires a live control socket "
            f"or FIFO at {control_pipe}; that path is not a daemon control pipe."
        )
    if not os.access(control_pipe, os.R_OK | os.W_OK):
        raise ValueError(
            "training.cuda_process_sharing='mps' cannot access control pipe "
            f"{control_pipe}; the launching user must be able to read and write "
            "the daemon's explicitly owned pipe directory."
        )
    server_pid = _server_evidence(run_command, control_pipe)
    if server_pid is None:
        raise ValueError(
            "training.cuda_process_sharing='mps' could not reach the control "
            f"daemon through {control_pipe}. Verify the daemon and retry: "
            f"{_start_command(pipe_directory)}"
        )

    return replace(
        evidence,
        learner_gpu_uuid=learner_uuid,
        collector_gpu_uuid=collector_uuid,
        control_pipe_kind=pipe_kind,
        server_pid=server_pid,
    )


def apply_cuda_process_sharing_manifest(
    runtime_manifest: MutableMapping[str, Any],
    evidence: CudaProcessSharingEvidence,
) -> None:
    """Record process-sharing evidence in a producer-owned manifest mapping."""

    runtime_manifest["cuda_process_sharing"] = evidence.to_dict()


__all__ = [
    "CudaProcessSharingEvidence",
    "SUPPORTED_CUDA_PROCESS_SHARING_BACKENDS",
    "apply_cuda_process_sharing_manifest",
    "canonicalize_gpu_uuid",
    "probe_cuda_process_sharing",
    "resolve_mps_visibility_entries",
]
