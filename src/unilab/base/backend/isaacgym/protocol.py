"""Shared IPC protocol between the host ``IsaacGymBackend`` and its worker.

Both ends load this module by file path: the host through the regular package
import, the Python 3.8 worker through ``importlib`` on an explicit path.  It
must therefore stay compatible with Python 3.8 and import only the standard
library and NumPy — never ``unilab`` itself.

Wire protocol: every control message is a dict ``{"cmd": str, "payload": ...}``
pickled with protocol 4 (the highest version Python 3.8 understands) and
framed with an 8-byte little-endian length prefix.  Bulk state never travels
through the pipe; it lives in ``multiprocessing.shared_memory`` slots whose
layout is declared by :func:`default_slot_specs`.

Conventions shared by both ends (aligned with the ``SimBackend`` contract):

- ``root_state``/``body_state`` quaternions are ``wxyz``; the worker converts
  from IsaacGym's native ``xyzw`` at the shm boundary.
- Angular velocities in shm slots are world-frame; the ``set_state`` qvel
  root columns carry body-frame angular velocity (converted by the worker).
- ``ctrl`` is per-DoF actuation force (torque), matching the MuJoCo backend's
  ``step(ctrl)`` semantics for motor actuators.
"""

from __future__ import annotations

import pickle
import struct
import traceback
from typing import Any, BinaryIO, Dict, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Commands and replies
# ---------------------------------------------------------------------------

CMD_INIT = "INIT"
CMD_ATTACH = "ATTACH_SLOTS"
CMD_STEP = "STEP"
CMD_SET_STATE = "SET_STATE"
CMD_REFRESH = "REFRESH"
CMD_GET_META = "GET_META"
CMD_INIT_RENDERER = "INIT_RENDERER"
CMD_RENDER_FRAME = "RENDER_FRAME"
CMD_CAPTURE_FRAME = "CAPTURE_FRAME"
CMD_SHUTDOWN = "SHUTDOWN"

CMD_READY = "READY"
CMD_META = "META"
CMD_ERROR = "ERROR"

_PICKLE_PROTOCOL = 4
_HEADER = struct.Struct("<Q")
HEADER_SIZE = _HEADER.size


def pack_message(cmd: str, payload: Any = None) -> bytes:
    """Serialize one message body (without the length header)."""
    return pickle.dumps({"cmd": cmd, "payload": payload}, protocol=_PICKLE_PROTOCOL)


def unpack_header(data: bytes) -> int:
    """Decode an 8-byte length header into the message body size."""
    (size,) = _HEADER.unpack(data)
    return int(size)


def decode_message(body: bytes) -> Dict[str, Any]:
    """Decode one pickled message body and validate its envelope."""
    message = pickle.loads(body)
    if not isinstance(message, dict) or "cmd" not in message:
        raise ValueError(f"malformed worker message: {message!r}")
    return message


# ---------------------------------------------------------------------------
# Message framing
# ---------------------------------------------------------------------------


class WorkerDisconnectedError(EOFError):
    """Raised when the pipe to the worker closes mid-message or at EOF."""


def send_message(stream: BinaryIO, cmd: str, payload: Any = None) -> None:
    """Write one framed pickle message and flush."""
    body = pack_message(cmd, payload)
    stream.write(_HEADER.pack(len(body)))
    stream.write(body)
    stream.flush()


def _read_exactly(stream: BinaryIO, size: int) -> bytes:
    chunks = []
    remaining = size
    while remaining > 0:
        chunk = stream.read(remaining)
        if not chunk:
            raise WorkerDisconnectedError(
                f"pipe closed while reading {size} bytes (got {size - remaining})"
            )
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def recv_message(stream: BinaryIO) -> Dict[str, Any]:
    """Read one framed pickle message."""
    size = unpack_header(_read_exactly(stream, _HEADER.size))
    return decode_message(_read_exactly(stream, size))


# ---------------------------------------------------------------------------
# Shared-memory slot layout
# ---------------------------------------------------------------------------

# name -> (dtype, per-env trailing shape or None for reset staging slots)
_SLOT_DTYPES: Dict[str, str] = {
    "ctrl": "float32",
    "root_state": "float32",
    "dof_state": "float32",
    "body_state": "float32",
    "contact_force": "float32",
    "reset_env_ids": "int32",
    "reset_qpos": "float32",
    "reset_qvel": "float32",
}

SLOT_NAMES = tuple(_SLOT_DTYPES)


def slot_shapes(num_envs: int, num_dof: int, num_bodies: int) -> Dict[str, Tuple[int, ...]]:
    """Return the fixed shm slot shapes for one backend instance.

    ``num_dof`` excludes the floating root; the full qpos/qvel are therefore
    ``7 + num_dof`` / ``6 + num_dof``.  Reset staging slots are sized at the
    full batch so any subset of env rows can be staged in one transaction.
    """
    if num_envs <= 0 or num_dof < 0 or num_bodies <= 0:
        raise ValueError(
            f"slot shapes require num_envs>0, num_dof>=0, num_bodies>0; "
            f"got {num_envs}, {num_dof}, {num_bodies}"
        )
    nq = 7 + num_dof
    nv = 6 + num_dof
    return {
        "ctrl": (num_envs, num_dof),
        "root_state": (num_envs, 13),
        "dof_state": (num_envs, num_dof, 2),
        "body_state": (num_envs, num_bodies, 13),
        "contact_force": (num_envs, num_bodies, 3),
        "reset_env_ids": (num_envs,),
        "reset_qpos": (num_envs, nq),
        "reset_qvel": (num_envs, nv),
    }


def slot_dtype(name: str) -> np.dtype:
    try:
        return np.dtype(_SLOT_DTYPES[name])
    except KeyError as exc:
        raise ValueError(f"unknown shm slot {name!r}; known: {sorted(_SLOT_DTYPES)}") from exc


def slot_nbytes(name: str, shape: Tuple[int, ...]) -> int:
    return int(np.prod(shape, dtype=np.int64)) * int(slot_dtype(name).itemsize)


# ---------------------------------------------------------------------------
# Error payloads
# ---------------------------------------------------------------------------


def serialize_exception(exc: BaseException) -> Dict[str, str]:
    """Pack a worker-side exception into a picklable ERROR payload."""
    return {
        "type": type(exc).__name__,
        "message": str(exc),
        "traceback": traceback.format_exc(),
    }


def format_worker_error(payload: Dict[str, str]) -> str:
    """Render an ERROR payload for the host-side exception message."""
    return (
        f"isaacgym worker raised {payload.get('type', 'Error')}: "
        f"{payload.get('message', '')}\n"
        f"worker traceback:\n{payload.get('traceback', '<unavailable>')}"
    )


# ---------------------------------------------------------------------------
# Quaternion helpers (wxyz, shared so both ends agree on conversions)
# ---------------------------------------------------------------------------


def xyzw_to_wxyz(quat: np.ndarray) -> np.ndarray:
    """Reorder (..., 4) quaternions from xyzw to wxyz."""
    quat = np.asarray(quat)
    return quat[..., [3, 0, 1, 2]]


def wxyz_to_xyzw(quat: np.ndarray) -> np.ndarray:
    """Reorder (..., 4) quaternions from wxyz to xyzw."""
    quat = np.asarray(quat)
    return quat[..., [1, 2, 3, 0]]


def quat_rotate(quat_wxyz: np.ndarray, vec: np.ndarray) -> np.ndarray:
    """Rotate (..., 3) vectors by (..., 4) wxyz quaternions (batched)."""
    q = np.asarray(quat_wxyz, dtype=np.float64)
    v = np.asarray(vec, dtype=np.float64)
    w = q[..., 0:1]
    u = q[..., 1:4]
    uv = np.cross(u, v)
    uuv = np.cross(u, uv)
    return v + 2.0 * (w * uv + uuv)


def quat_rotate_inverse(quat_wxyz: np.ndarray, vec: np.ndarray) -> np.ndarray:
    """Rotate (..., 3) vectors by the inverse of (..., 4) wxyz quaternions."""
    q = np.asarray(quat_wxyz, dtype=np.float64).copy()
    q[..., 1:4] = -q[..., 1:4]
    return quat_rotate(q, vec)


__all__ = [
    "CMD_ATTACH",
    "CMD_CAPTURE_FRAME",
    "CMD_ERROR",
    "CMD_GET_META",
    "CMD_INIT",
    "CMD_INIT_RENDERER",
    "CMD_META",
    "CMD_READY",
    "CMD_REFRESH",
    "CMD_RENDER_FRAME",
    "CMD_SET_STATE",
    "CMD_SHUTDOWN",
    "CMD_STEP",
    "HEADER_SIZE",
    "SLOT_NAMES",
    "WorkerDisconnectedError",
    "decode_message",
    "format_worker_error",
    "pack_message",
    "quat_rotate",
    "quat_rotate_inverse",
    "recv_message",
    "send_message",
    "serialize_exception",
    "slot_dtype",
    "slot_nbytes",
    "slot_shapes",
    "unpack_header",
    "wxyz_to_xyzw",
    "xyzw_to_wxyz",
]
