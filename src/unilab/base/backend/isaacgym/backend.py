"""Host-side ``SimBackend`` implementation driving an IsaacGym subprocess.

IsaacGym (Preview 4, EOL) only supports Python 3.6-3.8 and cannot be installed
into the main UniLab environment, so physics runs in a dedicated worker
interpreter (see ``dependencies.py``).  This class owns the worker process,
the shared-memory state slots, and the numpy-facing contract; the worker
(``worker.py``) owns the actual IsaacGym tensors.

Design invariants:

- The constructor is light: it only validates arguments.  ``materialize()``
  spawns the worker, runs the INIT/META handshake, creates the shared-memory
  slots, and attaches them (ATTACH_SLOTS) on the cold path.
- Bulk state crosses the process boundary through ``multiprocessing.shared_memory``
  only.  The stdin/stdout pipe carries length-prefixed pickle commands
  (``protocol.py``).  Hot-path getters read the shm caches written at the last
  STEP/SET_STATE barrier; they never round-trip the worker.
- ``step(ctrl)`` semantics match the MuJoCo motor profile: ``ctrl`` is the
  per-DoF actuation force (torque), applied through IsaacGym's
  ``set_dof_actuation_force_tensor`` with ``DOF_MODE_EFFORT``.
- Quaternions on the public surface are ``wxyz`` (IsaacGym tensors are
  ``xyzw``; the worker converts at the shm boundary).  ``set_state`` qvel root
  columns carry body-frame angular velocity per the ``SimBackend`` contract;
  the worker converts to IsaacGym's world-frame angular velocity.
- Domain randomization, native rendering/playback, and pre-step control
  callbacks are fail-closed (see ``get_dr_capabilities`` /
  ``apply_interval_randomization`` / ``set_pre_step_control``).

Sensor contract: IsaacGym has no MuJoCo sensor concept.  The scene MJCF is
scanned once on the cold path (``sensors.py``) and supported sensors
(``gyro``, ``velocimeter``, ``framequat``, ``framepos``, ``contact`` with
``data="found"``) are computed from the shm tensor caches; everything else
fails closed with an explanatory ``NotImplementedError``.

Known real-runtime gaps to validate on hardware (the development machine has
no IsaacGym install, so all runtime behavior is covered by a mock worker):

- MJCF importer fidelity (fixed-joint handling, armature, actuator mapping);
- whether ``set_dof_state_tensor_indexed`` / ``set_actor_root_state_tensor_indexed``
  semantics match the wrapped-full-buffer pattern used here;
- body-state staleness right after SET_STATE (IsaacGym has no kinematics-only
  forward call; rigid-body/contact slots refresh at the next physics step).
"""

from __future__ import annotations

import atexit
import os
import select
import subprocess
import tempfile
import time
from collections.abc import Sequence
from dataclasses import dataclass
from multiprocessing import shared_memory
from pathlib import Path
from typing import Any, BinaryIO, cast

import numpy as np

from unilab.base.backend.base import (
    BackendRootStateLayout,
    SimBackend,
)
from unilab.base.scene import SceneCfg
from unilab.dr.types import (
    DomainRandomizationCapabilities,
    IntervalRandomizationPlan,
    ResetRandomizationPayload,
)
from unilab.utils.rotation import (
    np_quat_apply_batched,
    np_quat_apply_inverse_batched,
    np_quat_mul_batched,
)

from . import protocol
from .dependencies import (
    IsaacGymRuntime,
    build_worker_env,
    resolve_isaacgym_runtime,
)
from .sensors import (
    KIND_CONTACT_FOUND,
    KIND_FRAMEPOS,
    KIND_FRAMEQUAT,
    KIND_FRAMEZAXIS,
    KIND_GYRO,
    KIND_LOCAL_LINVEL,
    SceneMetadata,
    SceneSensorSpec,
    UnsupportedSensorSpec,
    scan_scene_metadata,
)

_MODULE_DIR = Path(__file__).resolve().parent
_WORKER_PATH = _MODULE_DIR / "worker.py"
_PROTOCOL_PATH = _MODULE_DIR / "protocol.py"

_DEFAULT_WORKER_TIMEOUT_S = 120.0
_SHUTDOWN_TIMEOUT_S = 5.0
_STDERR_TAIL_BYTES = 4096

_ROOT_QPOS_DIM = 7
_ROOT_QVEL_DIM = 6


class IsaacGymWorkerError(RuntimeError):
    """Raised when the worker reports an error, dies, or stops responding.

    Carries the worker-side traceback and/or the captured stderr tail so
    crashes inside the Python 3.8 process remain diagnosable from the host.
    """

    def __init__(
        self,
        message: str,
        *,
        worker_traceback: str | None = None,
        stderr_tail: str | None = None,
    ) -> None:
        super().__init__(message)
        self.worker_traceback = worker_traceback
        self.stderr_tail = stderr_tail


@dataclass(frozen=True)
class IsaacGymModelInfo:
    """Opaque backend-owned model metadata returned by ``IsaacGymBackend.model``."""

    num_dof: int
    num_bodies: int
    dof_names: tuple[str, ...]
    body_names: tuple[str, ...]
    gravity: tuple[float, float, float]
    use_gpu_pipeline: bool


def _release_worker(
    proc: "subprocess.Popen[bytes] | None",
    shm_handles: list[shared_memory.SharedMemory],
    stderr_file: Any,
) -> None:
    """Best-effort worker shutdown: SHUTDOWN handshake, then terminate, then kill.

    Called from ``close()``/``__del__``/``atexit`` so the worker never survives
    the host process or leaks shared-memory segments.
    """
    if proc is not None and proc.poll() is None:
        try:
            if proc.stdin is not None:
                protocol.send_message(cast(BinaryIO, proc.stdin), protocol.CMD_SHUTDOWN)
            proc.wait(timeout=_SHUTDOWN_TIMEOUT_S)
        except Exception:
            try:
                proc.terminate()
                proc.wait(timeout=_SHUTDOWN_TIMEOUT_S)
            except Exception:
                try:
                    proc.kill()
                    proc.wait(timeout=_SHUTDOWN_TIMEOUT_S)
                except Exception:
                    pass
    for handle in shm_handles:
        try:
            handle.close()
            handle.unlink()
        except Exception:
            pass
    if stderr_file is not None:
        try:
            stderr_file.close()
        except Exception:
            pass


def _read_exactly_with_deadline(stream: Any, size: int, deadline: float) -> bytes:
    """Read exactly ``size`` bytes from an unbuffered stream before ``deadline``.

    ``stream`` must be unbuffered (``Popen(..., bufsize=0)``) so ``select`` on
    the underlying fd cannot be desynchronized by read-ahead buffering.
    """
    chunks: list[bytes] = []
    remaining = size
    while remaining > 0:
        wait = deadline - time.monotonic()
        if wait <= 0:
            raise TimeoutError(f"timed out waiting for {size} bytes from worker")
        readable, _, _ = select.select([stream], [], [], wait)
        if not readable:
            raise TimeoutError(f"timed out waiting for {size} bytes from worker")
        chunk = stream.read(remaining)
        if not chunk:
            raise protocol.WorkerDisconnectedError(
                f"worker pipe closed while reading {size} bytes (got {size - remaining})"
            )
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


class IsaacGymBackend(SimBackend):
    """``SimBackend`` implementation delegating physics to an IsaacGym worker."""

    # Same column-stability contract as the other backends: non-applicable
    # sub-steps report 0.0.
    _SET_STATE_TIMING_ZERO_KEYS = (
        "set_state_mask_ms",
        "set_state_data_slice_ms",
        "set_state_data_reset_ms",
        "set_state_clear_forces_ms",
        "set_state_geom_overrides_ms",
        "set_state_reset_rand_ms",
        "set_state_set_dof_vel_ms",
        "set_state_set_dof_pos_ms",
        "set_state_actuator_ctrl_ms",
        "set_state_forward_kinematic_ms",
        "set_state_refresh_pose_cache_ms",
        "set_state_invalidate_velocity_ms",
        "set_state_qpos_convert_ms",
        "set_state_pool_reset_ms",
        "set_state_state_scatter_ms",
    )

    def __init__(
        self,
        scene: SceneCfg,
        num_envs: int,
        sim_dt: float,
        *,
        base_name: str | None = None,
        device_id: int | None = None,
        worker_timeout_s: float | None = None,
        worker_command: list[str] | None = None,
        headless: bool = True,
        **unexpected_kwargs: Any,
    ) -> None:
        if unexpected_kwargs:
            names = ", ".join(sorted(unexpected_kwargs))
            raise TypeError(f"IsaacGymBackend does not accept backend options: {names}")
        if isinstance(num_envs, bool) or int(num_envs) <= 0:
            raise ValueError(f"num_envs must be a positive integer, got {num_envs!r}")
        if float(sim_dt) <= 0.0:
            raise ValueError(f"sim_dt must be positive, got {sim_dt!r}")
        if worker_command is not None and (
            not isinstance(worker_command, list)
            or not worker_command
            or not all(isinstance(part, str) for part in worker_command)
        ):
            raise TypeError(
                "worker_command must be a non-empty list of strings or None, "
                f"got {worker_command!r}"
            )
        if scene.fragment_files:
            raise NotImplementedError(
                "isaacgym backend does not compose MuJoCo scene fragments; provide a "
                "self-contained MJCF scene through scene.model_file"
            )
        if scene.terrain is not None:
            raise NotImplementedError(
                "isaacgym backend does not support generated terrain scenes yet"
            )

        self._scene = scene
        self._num_envs = int(num_envs)
        self._sim_dt = float(sim_dt)
        self._base_name = base_name
        self._device_id = 0 if device_id is None else int(device_id)
        self._worker_timeout_s = (
            _DEFAULT_WORKER_TIMEOUT_S if worker_timeout_s is None else float(worker_timeout_s)
        )
        if self._worker_timeout_s <= 0.0:
            raise ValueError(f"worker_timeout_s must be positive, got {self._worker_timeout_s!r}")
        self._worker_command = list(worker_command) if worker_command is not None else None
        self._headless = bool(headless)
        self.backend_type = "isaacgym"
        self._pre_step_control_fn = None
        self._scene_cleanup_handle = None

        # Everything below is materialized lazily in materialize().
        self._proc: subprocess.Popen[bytes] | None = None
        self._shm_handles: dict[str, shared_memory.SharedMemory] = {}
        self._slots: dict[str, np.ndarray] = {}
        self._stderr_file: Any = None
        self._worker_dead_error: IsaacGymWorkerError | None = None
        self._model_info: IsaacGymModelInfo | None = None
        self._scene_metadata: SceneMetadata | None = None
        self._sensor_map: dict[str, tuple[SceneSensorSpec, int]] = {}
        self._body_id_by_name: dict[str, int] = {}
        self._dof_id_by_name: dict[str, int] = {}
        self._joint_range: np.ndarray | None = None
        self._ctrl_range: np.ndarray | None = None
        self._base_body_id = 0

    # ------------------------------------------------------------------ #
    # Worker lifecycle (cold path)
    # ------------------------------------------------------------------ #

    def materialize(self) -> None:
        """Spawn the worker, run the handshake, and bind shared-memory slots."""
        if self._proc is not None:
            return
        runtime: IsaacGymRuntime | None = None
        if self._worker_command is None:
            runtime = resolve_isaacgym_runtime()
            command = [str(runtime.python), str(_WORKER_PATH)]
            env = build_worker_env(runtime)
        else:
            command = list(self._worker_command)
            env = None
        isaacgym_python = str(runtime.isaacgym_python) if runtime is not None else ""

        self._stderr_file = tempfile.TemporaryFile(mode="w+b", prefix="isaacgym_worker_stderr_")
        try:
            self._proc = subprocess.Popen(
                [*command, "--protocol", str(_PROTOCOL_PATH)],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=self._stderr_file,
                bufsize=0,
                env=env,
            )
        except OSError as exc:
            raise IsaacGymWorkerError(f"failed to spawn isaacgym worker {command}: {exc}") from exc

        try:
            meta = self._request(
                protocol.CMD_INIT,
                {
                    "model_file": str(Path(self._scene.model_file).expanduser()),
                    "num_envs": self._num_envs,
                    "sim_dt": self._sim_dt,
                    "device_id": self._device_id,
                    "headless": self._headless,
                    "isaacgym_python": isaacgym_python,
                },
                expect=protocol.CMD_META,
            )
            self._bind_model_metadata(meta)
            self._allocate_slots()
            self._request(
                protocol.CMD_ATTACH, {"slots": self._slot_specs()}, expect=protocol.CMD_READY
            )
            self._scene_metadata = scan_scene_metadata(
                str(Path(self._scene.model_file).expanduser())
            )
            self._sensor_map = self._resolve_sensor_map()
            if self._base_name is not None:
                try:
                    self._base_body_id = self._body_id_by_name[self._base_name]
                except KeyError as exc:
                    raise ValueError(
                        f"Base body {self._base_name!r} not found in isaacgym model"
                    ) from exc
        except Exception:
            self.close()
            raise
        # Subprocess liveness cannot rely on __del__ alone during interpreter
        # teardown; the atexit hook is unregistered by close().
        atexit.register(self.close)

    def _bind_model_metadata(self, meta: dict[str, Any]) -> None:
        num_dof = int(meta["num_dof"])
        num_bodies = int(meta["num_bodies"])
        dof_names = tuple(str(name) for name in meta["dof_names"])
        body_names = tuple(str(name) for name in meta["body_names"])
        if len(dof_names) != num_dof or len(body_names) != num_bodies:
            raise IsaacGymWorkerError(
                "isaacgym worker metadata is inconsistent: "
                f"num_dof={num_dof} vs {len(dof_names)} names, "
                f"num_bodies={num_bodies} vs {len(body_names)} names"
            )
        gravity = tuple(float(value) for value in meta["gravity"])
        self._model_info = IsaacGymModelInfo(
            num_dof=num_dof,
            num_bodies=num_bodies,
            dof_names=dof_names,
            body_names=body_names,
            gravity=(gravity[0], gravity[1], gravity[2]),
            use_gpu_pipeline=bool(meta.get("use_gpu_pipeline", False)),
        )
        self._body_id_by_name = {name: index for index, name in enumerate(body_names)}
        self._dof_id_by_name = {name: index for index, name in enumerate(dof_names)}
        lower = np.asarray(meta["dof_lower"], dtype=np.float32)
        upper = np.asarray(meta["dof_upper"], dtype=np.float32)
        self._joint_range = (
            np.stack([lower, upper], axis=1) if num_dof > 0 else np.zeros((0, 2), np.float32)
        )
        effort = np.asarray(meta["effort"], dtype=np.float32)
        self._ctrl_range = (
            np.stack([-effort, effort], axis=1) if num_dof > 0 else np.zeros((0, 2), np.float32)
        )

    def _allocate_slots(self) -> None:
        assert self._model_info is not None
        shapes = protocol.slot_shapes(
            self._num_envs, self._model_info.num_dof, self._model_info.num_bodies
        )
        for name in protocol.SLOT_NAMES:
            shape = shapes[name]
            handle = shared_memory.SharedMemory(create=True, size=protocol.slot_nbytes(name, shape))
            self._shm_handles[name] = handle
            self._slots[name] = np.ndarray(
                shape, dtype=protocol.slot_dtype(name), buffer=handle.buf
            )

    def _slot_specs(self) -> dict[str, dict[str, Any]]:
        return {
            name: {
                "shm": handle.name,
                "shape": list(self._slots[name].shape),
                "dtype": str(self._slots[name].dtype),
            }
            for name, handle in self._shm_handles.items()
        }

    def _resolve_sensor_map(self) -> dict[str, tuple[SceneSensorSpec, int]]:
        """Resolve supported scene sensors to (spec, body_id) on the cold path."""
        assert self._scene_metadata is not None
        resolved: dict[str, tuple[SceneSensorSpec, int]] = {}
        for name, spec in self._scene_metadata.sensors.items():
            body_id = self._body_id_by_name.get(spec.body_name)
            if body_id is None:
                # The MJCF importer may drop or rename bodies; record as
                # unsupported so access fails closed with context.
                self._scene_metadata.unsupported_sensors[name] = _unsupported_spec(
                    spec,
                    f"sensor body {spec.body_name!r} is not present in the IsaacGym "
                    "asset rigid-body list",
                )
                continue
            resolved[name] = (spec, body_id)
        return resolved

    # ------------------------------------------------------------------ #
    # Request/response plumbing
    # ------------------------------------------------------------------ #

    def _stderr_tail(self) -> str:
        if self._stderr_file is None:
            return ""
        try:
            self._stderr_file.flush()
            self._stderr_file.seek(0, os.SEEK_END)
            size = self._stderr_file.tell()
            self._stderr_file.seek(max(0, size - _STDERR_TAIL_BYTES))
            return str(self._stderr_file.read().decode("utf-8", errors="replace"))
        except Exception:
            return ""

    def _request(self, cmd: str, payload: Any, *, expect: str) -> Any:
        if self._worker_dead_error is not None:
            raise IsaacGymWorkerError(
                f"isaacgym worker is unavailable from an earlier failure; refusing {cmd}"
            ) from self._worker_dead_error
        proc = self._proc
        if proc is None or proc.stdin is None or proc.stdout is None:
            raise IsaacGymWorkerError(
                "isaacgym backend is not materialized; call materialize() first"
            )
        if proc.poll() is not None:
            error = IsaacGymWorkerError(
                f"isaacgym worker exited with code {proc.returncode} before {cmd}; "
                f"stderr tail:\n{self._stderr_tail()}",
                stderr_tail=self._stderr_tail(),
            )
            self._worker_dead_error = error
            raise error
        try:
            protocol.send_message(cast(BinaryIO, proc.stdin), cmd, payload)
            message = self._recv_with_timeout(proc.stdout, self._worker_timeout_s, cmd)
        except IsaacGymWorkerError as exc:
            self._worker_dead_error = exc
            self._kill_worker()
            raise
        if message["cmd"] == protocol.CMD_ERROR:
            error = IsaacGymWorkerError(
                protocol.format_worker_error(message["payload"]),
                worker_traceback=message["payload"].get("traceback"),
            )
            raise error
        if message["cmd"] != expect:
            raise IsaacGymWorkerError(
                f"isaacgym worker replied {message['cmd']!r} to {cmd}, expected {expect!r}"
            )
        return message.get("payload")

    def _recv_with_timeout(self, stream: Any, timeout_s: float, cmd: str) -> dict[str, Any]:
        deadline = time.monotonic() + timeout_s
        try:
            header = _read_exactly_with_deadline(stream, protocol.HEADER_SIZE, deadline)
            body = _read_exactly_with_deadline(stream, protocol.unpack_header(header), deadline)
        except TimeoutError as exc:
            raise IsaacGymWorkerError(
                f"isaacgym worker did not answer {cmd} within {timeout_s}s; "
                f"stderr tail:\n{self._stderr_tail()}",
                stderr_tail=self._stderr_tail(),
            ) from exc
        except protocol.WorkerDisconnectedError as exc:
            raise IsaacGymWorkerError(
                f"isaacgym worker closed its pipe during {cmd} "
                f"(exit code {self._proc.poll() if self._proc else '?'}); "
                f"stderr tail:\n{self._stderr_tail()}",
                stderr_tail=self._stderr_tail(),
            ) from exc
        return protocol.decode_message(body)

    def _kill_worker(self) -> None:
        proc = self._proc
        if proc is not None and proc.poll() is None:
            proc.kill()
            try:
                proc.wait(timeout=_SHUTDOWN_TIMEOUT_S)
            except Exception:
                pass

    def close(self) -> None:
        """Shut down the worker and release shared memory. Idempotent."""
        proc, self._proc = self._proc, None
        shm_handles, self._shm_handles = list(self._shm_handles.values()), {}
        self._slots = {}
        stderr_file, self._stderr_file = self._stderr_file, None
        try:
            atexit.unregister(self.close)
        except Exception:
            pass
        _release_worker(proc, shm_handles, stderr_file)

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass
        try:
            super().__del__()
        except Exception:
            pass

    # ------------------------------------------------------------------ #
    # Materialized-state guards and views
    # ------------------------------------------------------------------ #

    def _require_materialized(self) -> IsaacGymModelInfo:
        if self._model_info is None:
            raise RuntimeError(
                "IsaacGymBackend metadata is unavailable before materialize(); "
                "the worker handshake runs on the materialize cold path"
            )
        return self._model_info

    def _require_state(self, operation: str) -> None:
        self._require_materialized()
        if self._worker_dead_error is not None:
            raise IsaacGymWorkerError(
                f"isaacgym worker is unavailable from an earlier failure; refusing {operation}"
            ) from self._worker_dead_error
        if not self._slots:
            raise IsaacGymWorkerError(
                f"isaacgym backend is closed or not materialized; refusing {operation}"
            )

    # ------------------------------------------------------------------ #
    # SimBackend properties and cold metadata
    # ------------------------------------------------------------------ #

    @property
    def num_envs(self) -> int:
        return self._num_envs

    @property
    def model(self) -> IsaacGymModelInfo:
        """Return backend-owned model metadata; never a live physics object."""
        return self._require_materialized()

    @property
    def num_actuators(self) -> int:
        return self._require_materialized().num_dof

    @property
    def num_dof_vel(self) -> int:
        return self._require_materialized().num_dof

    def get_actuator_ctrl_range(self) -> np.ndarray:
        self._require_materialized()
        assert self._ctrl_range is not None
        return self._ctrl_range.copy()

    def get_actuator_names(self) -> tuple[str, ...]:
        return self._require_materialized().dof_names

    def get_actuator_joint_names(self) -> tuple[str, ...]:
        """IsaacGym drives one DoF per joint, so actuator order equals DoF order."""
        return self._require_materialized().dof_names

    def get_actuator_gains(self) -> tuple[np.ndarray, np.ndarray]:
        """Effort-mode dofs carry no PD gains; report zeros explicitly."""
        info = self._require_materialized()
        zeros = np.zeros((info.num_dof,), dtype=np.float32)
        return zeros, zeros.copy()

    def get_scene_model_file(self) -> str | None:
        return str(self._scene.model_file)

    def get_keyframe_qpos(self, name: str) -> np.ndarray:
        metadata = self._scene_metadata
        if metadata is None:
            self._require_materialized()
            metadata = self._scene_metadata
        assert metadata is not None
        try:
            qpos = metadata.keyframes[name]
        except KeyError as exc:
            available = ", ".join(sorted(metadata.keyframes))
            raise ValueError(f"Keyframe {name!r} not found; available: {available}") from exc
        nq = _ROOT_QPOS_DIM + self._require_materialized().num_dof
        if qpos.size != nq:
            raise ValueError(
                f"Keyframe {name!r} qpos has {qpos.size} entries; expected {nq} "
                f"(7 root + {nq - 7} dofs) for the isaacgym layout"
            )
        return qpos.copy()

    def get_default_qpos(self) -> np.ndarray:
        info = self._require_materialized()
        qpos = np.zeros((_ROOT_QPOS_DIM + info.num_dof,), dtype=np.float32)
        qpos[3] = 1.0
        return qpos

    def get_default_dof_pos(self) -> np.ndarray:
        info = self._require_materialized()
        return np.zeros((info.num_dof,), dtype=np.float32)

    def get_init_qvel(self) -> np.ndarray:
        info = self._require_materialized()
        return np.zeros((_ROOT_QVEL_DIM + info.num_dof,), dtype=np.float32)

    def get_root_state_layout(self, root_body_name: str) -> BackendRootStateLayout:
        info = self._require_materialized()
        if root_body_name != info.body_names[0]:
            raise NotImplementedError(
                "backend 'isaacgym' capability 'root-state layout' requires "
                f"{root_body_name!r} to be the actor root body {info.body_names[0]!r}"
            )
        return BackendRootStateLayout(
            qpos_indices=tuple(range(_ROOT_QPOS_DIM)),
            qvel_indices=tuple(range(_ROOT_QVEL_DIM)),
        )

    def get_body_ids(self, names: Sequence[str]) -> np.ndarray:
        self._require_materialized()
        resolved: list[int] = []
        for name in names:
            try:
                resolved.append(self._body_id_by_name[str(name)])
            except KeyError as exc:
                raise ValueError(f"Body {name!r} not found in isaacgym model") from exc
        return np.asarray(resolved, dtype=np.int32)

    def get_motion_body_ids(self, names: Sequence[str]) -> np.ndarray:
        return self.get_body_ids(names)

    def get_joint_range(self) -> np.ndarray | None:
        self._require_materialized()
        if self._joint_range is None or self._joint_range.size == 0:
            return None
        return self._joint_range.copy()

    def get_gravity(self) -> np.ndarray:
        info = self._require_materialized()
        return np.asarray(info.gravity, dtype=np.float32).copy()

    def get_joint_dof_pos_indices(self, names: Sequence[str]) -> np.ndarray:
        return self._resolve_dof_ids(names)

    def get_joint_dof_vel_indices(self, names: Sequence[str]) -> np.ndarray:
        return self._resolve_dof_ids(names)

    def get_joint_state_qpos_indices(self, names: Sequence[str]) -> np.ndarray:
        return self._resolve_dof_ids(names) + _ROOT_QPOS_DIM

    def get_joint_state_qvel_indices(self, names: Sequence[str]) -> np.ndarray:
        return self._resolve_dof_ids(names) + _ROOT_QVEL_DIM

    def _resolve_dof_ids(self, names: Sequence[str]) -> np.ndarray:
        self._require_materialized()
        resolved: list[int] = []
        for name in names:
            try:
                resolved.append(self._dof_id_by_name[str(name)])
            except KeyError as exc:
                raise ValueError(f"Joint {name!r} not found in isaacgym model") from exc
        return np.asarray(resolved, dtype=np.int32)

    # ------------------------------------------------------------------ #
    # Simulation control
    # ------------------------------------------------------------------ #

    def set_pre_step_control(self, fn: Any | None) -> None:
        if fn is not None:
            raise NotImplementedError(
                "isaacgym rejects host pre-step callbacks; a per-substep callback "
                "cannot cross the worker process boundary inside one physics substep."
            )
        self._pre_step_control_fn = None

    def step(self, ctrl: np.ndarray, nsteps: int = 1) -> dict[str, dict[str, float]]:
        self._require_state("step")
        if isinstance(nsteps, bool) or int(nsteps) <= 0:
            raise ValueError(f"nsteps must be a positive integer, got {nsteps!r}")
        info = self._require_materialized()
        ctrl_array = np.asarray(ctrl, dtype=np.float32)
        expected = (self._num_envs, info.num_dof)
        if ctrl_array.shape != expected:
            raise ValueError(f"ctrl must have shape {expected}, got {ctrl_array.shape}")

        t0 = time.perf_counter()
        np.copyto(self._slots["ctrl"], ctrl_array)
        payload = self._request(
            protocol.CMD_STEP, {"nsteps": int(nsteps)}, expect=protocol.CMD_READY
        )
        ipc_ms = (time.perf_counter() - t0) * 1000.0
        timing = dict(payload.get("timing", {})) if isinstance(payload, dict) else {}
        timing["worker_ipc_total_ms"] = ipc_ms
        return {"timing": timing}

    def set_state(
        self,
        env_indices: np.ndarray,
        qpos: np.ndarray,
        qvel: np.ndarray,
        randomization: ResetRandomizationPayload | None = None,
    ) -> dict[str, dict[str, float]]:
        self._require_state("set_state")
        if randomization is not None and not randomization.is_empty():
            requested = ", ".join(sorted(randomization.requested_terms()))
            raise NotImplementedError(
                f"isaacgym does not support reset domain randomization terms: {requested}."
            )
        info = self._require_materialized()
        rows = np.asarray(env_indices, dtype=np.intp)
        if rows.ndim != 1:
            raise ValueError(f"env_indices must be one-dimensional, got shape {rows.shape}")
        if np.any(rows < 0) or np.any(rows >= self._num_envs):
            raise ValueError(f"env_indices must be in [0, {self._num_envs}), got {rows}")
        if np.unique(rows).size != rows.size:
            raise ValueError("env_indices must not contain duplicate rows")
        nq = _ROOT_QPOS_DIM + info.num_dof
        nv = _ROOT_QVEL_DIM + info.num_dof
        qpos_array = np.asarray(qpos, dtype=np.float32)
        qvel_array = np.asarray(qvel, dtype=np.float32)
        if qpos_array.shape != (rows.size, nq):
            raise ValueError(f"qpos must have shape ({rows.size}, {nq}), got {qpos_array.shape}")
        if qvel_array.shape != (rows.size, nv):
            raise ValueError(f"qvel must have shape ({rows.size}, {nv}), got {qvel_array.shape}")

        timing: dict[str, float] = {key: 0.0 for key in self._SET_STATE_TIMING_ZERO_KEYS}
        timing.update(
            {
                "set_state_reset_upload_ms": 0.0,
                "set_state_reset_forward_ms": 0.0,
                "set_state_host_cache_refresh_ms": 0.0,
                "set_state_internal_gap_ms": 0.0,
            }
        )
        if rows.size == 0:
            return {"timing": timing}

        t0 = time.perf_counter()
        count = int(rows.size)
        np.copyto(self._slots["reset_env_ids"][:count], rows.astype(np.int32))
        np.copyto(self._slots["reset_qpos"][:count], qpos_array)
        np.copyto(self._slots["reset_qvel"][:count], qvel_array)
        payload = self._request(protocol.CMD_SET_STATE, {"count": count}, expect=protocol.CMD_READY)
        ipc_ms = (time.perf_counter() - t0) * 1000.0
        if isinstance(payload, dict):
            worker_timing = payload.get("timing", {})
            timing["set_state_reset_upload_ms"] = float(
                worker_timing.get("set_state_reset_upload_ms", 0.0)
            )
            timing["set_state_host_cache_refresh_ms"] = float(
                worker_timing.get("set_state_host_cache_refresh_ms", 0.0)
            )
        timing["set_state_internal_gap_ms"] = (
            ipc_ms - timing["set_state_reset_upload_ms"] - timing["set_state_host_cache_refresh_ms"]
        )
        return {"timing": timing}

    def get_dr_capabilities(self) -> DomainRandomizationCapabilities:
        """Advertise no DR until per-env model mutation is effect-tested."""
        return DomainRandomizationCapabilities()

    def apply_interval_randomization(self, plan: IntervalRandomizationPlan) -> None:
        if plan.is_empty():
            return
        raise NotImplementedError(
            "isaacgym does not support interval randomization; disable "
            "push/body-force/body-velocity terms in the owner YAML."
        )

    # ------------------------------------------------------------------ #
    # Cached state getters (shm views; no worker round trip)
    # ------------------------------------------------------------------ #

    def _root_slot(self) -> np.ndarray:
        self._require_state("state read")
        return self._slots["root_state"]

    def _body_slot(self) -> np.ndarray:
        self._require_state("state read")
        return self._slots["body_state"]

    def get_base_pos(self) -> np.ndarray:
        return self._root_slot()[:, 0:3]

    def get_base_quat(self) -> np.ndarray:
        return self._root_slot()[:, 3:7]

    def get_base_lin_vel(self) -> np.ndarray:
        return self._root_slot()[:, 7:10]

    def get_base_ang_vel(self) -> np.ndarray:
        return self._root_slot()[:, 10:13]

    def get_dof_pos(self) -> np.ndarray:
        self._require_state("get_dof_pos")
        return self._slots["dof_state"][:, :, 0]

    def get_dof_vel(self) -> np.ndarray:
        self._require_state("get_dof_vel")
        return self._slots["dof_state"][:, :, 1]

    def _selected_body_state(self, body_ids: np.ndarray) -> np.ndarray:
        ids = np.asarray(body_ids, dtype=np.intp)
        info = self._require_materialized()
        if ids.ndim != 1 or np.any(ids < 0) or np.any(ids >= info.num_bodies):
            raise ValueError(
                f"body_ids must be a 1-D array in [0, {info.num_bodies}), got {body_ids!r}"
            )
        return self._body_slot()[:, ids, :]

    def get_body_pos_w(self, body_ids: np.ndarray) -> np.ndarray:
        return self._selected_body_state(body_ids)[:, :, 0:3]

    def get_body_quat_w(self, body_ids: np.ndarray) -> np.ndarray:
        return self._selected_body_state(body_ids)[:, :, 3:7]

    def get_body_lin_vel_w(self, body_ids: np.ndarray) -> np.ndarray:
        return self._selected_body_state(body_ids)[:, :, 7:10]

    def get_body_ang_vel_w(self, body_ids: np.ndarray) -> np.ndarray:
        return self._selected_body_state(body_ids)[:, :, 10:13]

    def get_body_pos_b(self, body_ids: np.ndarray) -> np.ndarray:
        base_state = self._body_slot()[:, self._base_body_id, :]
        rel = self.get_body_pos_w(body_ids) - base_state[:, 0:3][:, None, :]
        return np_quat_apply_inverse_batched(base_state[:, 3:7][:, None, :], rel)

    def get_body_quat_b(self, body_ids: np.ndarray) -> np.ndarray:
        inverse = self._body_slot()[:, self._base_body_id, 3:7].copy()
        inverse[:, 1:4] = -inverse[:, 1:4]
        quat_w = self.get_body_quat_w(body_ids)
        return np_quat_mul_batched(inverse[:, None, :], quat_w)

    def get_body_lin_vel_b(self, body_ids: np.ndarray) -> np.ndarray:
        # Contract definition (#1254): world velocity rotated into each body's
        # own frame, identical to the MuJoCo/mjwarp backends.
        return np_quat_apply_inverse_batched(
            self.get_body_quat_w(body_ids), self.get_body_lin_vel_w(body_ids)
        )

    def get_body_ang_vel_b(self, body_ids: np.ndarray) -> np.ndarray:
        return np_quat_apply_inverse_batched(
            self.get_body_quat_w(body_ids), self.get_body_ang_vel_w(body_ids)
        )

    # ------------------------------------------------------------------ #
    # Sensors
    # ------------------------------------------------------------------ #

    def get_sensor_data(self, name: str) -> np.ndarray:
        """Compute one mapped scene sensor from the shm state caches.

        See ``sensors.py`` for the MJCF-sensor → tensor-quantity mapping table.
        Names that are not declared in the scene raise ``ValueError``; declared
        but unmappable sensors fail closed with ``NotImplementedError``.
        """
        self._require_state("get_sensor_data")
        metadata = self._scene_metadata
        assert metadata is not None
        mapped = self._sensor_map.get(name)
        if mapped is None:
            unsupported = metadata.unsupported_sensors.get(name)
            if unsupported is not None:
                raise NotImplementedError(
                    f"isaacgym cannot serve sensor {name!r}: {unsupported.reason}"
                )
            available = ", ".join(sorted(self._sensor_map))
            raise ValueError(f"Sensor {name!r} not found; available: {available}")
        spec, body_id = mapped
        state = self._body_slot()[:, body_id, :]
        kind = spec.kind
        local_quat = np.asarray(spec.local_quat, dtype=np.float32)[None, :]
        local_pos = np.asarray(spec.local_pos, dtype=np.float32)[None, :]
        if kind == KIND_GYRO:
            body_frame = np_quat_apply_inverse_batched(state[:, 3:7], state[:, 10:13])
            return np_quat_apply_inverse_batched(local_quat, body_frame).astype(np.float32)
        if kind == KIND_LOCAL_LINVEL:
            body_frame = np_quat_apply_inverse_batched(state[:, 3:7], state[:, 7:10])
            return np_quat_apply_inverse_batched(local_quat, body_frame).astype(np.float32)
        if kind == KIND_FRAMEQUAT:
            return np_quat_mul_batched(state[:, 3:7], local_quat).astype(np.float32)
        if kind == KIND_FRAMEPOS:
            offset = np_quat_apply_batched(state[:, 3:7], local_pos)
            return (state[:, 0:3] + offset).astype(np.float32)
        if kind == KIND_FRAMEZAXIS:
            frame_quat = np_quat_mul_batched(state[:, 3:7], local_quat)
            z_axis = np.array([[0.0, 0.0, 1.0]], dtype=np.float32)
            return np_quat_apply_batched(frame_quat, z_axis).astype(np.float32)
        if kind == KIND_CONTACT_FOUND:
            force = self._slots["contact_force"][:, body_id, :]
            return (np.linalg.norm(force, axis=-1, keepdims=True) > 0.0).astype(np.float32)
        raise NotImplementedError(f"isaacgym sensor kind {kind!r} is not implemented")


def _unsupported_spec(spec: SceneSensorSpec, reason: str) -> UnsupportedSensorSpec:
    return UnsupportedSensorSpec(name=spec.name, reason=reason)


__all__ = ["IsaacGymBackend", "IsaacGymModelInfo", "IsaacGymWorkerError"]
