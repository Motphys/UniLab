"""Canonical pipe and shared-memory protocol for subprocess backends.

The module is loaded both by the host interpreter and by external workers via
an explicit file path.  Keep it compatible with Python 3.8 and import only the
standard library plus NumPy.
"""

from __future__ import annotations

import pickle
import struct
import traceback
from typing import Any, BinaryIO, Dict, Tuple

import numpy as np

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
    return pickle.dumps({"cmd": cmd, "payload": payload}, protocol=_PICKLE_PROTOCOL)


def unpack_header(data: bytes) -> int:
    (size,) = _HEADER.unpack(data)
    return int(size)


def decode_message(body: bytes) -> Dict[str, Any]:
    message = pickle.loads(body)
    if not isinstance(message, dict) or "cmd" not in message:
        raise ValueError(f"malformed worker message: {message!r}")
    return message


class WorkerDisconnectedError(EOFError):
    """Raised when a worker pipe closes before a complete message arrives."""


def send_message(stream: BinaryIO, cmd: str, payload: Any = None) -> None:
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
    size = unpack_header(_read_exactly(stream, _HEADER.size))
    return decode_message(_read_exactly(stream, size))


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
    if num_envs <= 0 or num_dof < 0 or num_bodies <= 0:
        raise ValueError(
            "slot shapes require num_envs>0, num_dof>=0, num_bodies>0; "
            f"got {num_envs}, {num_dof}, {num_bodies}"
        )
    return {
        "ctrl": (num_envs, num_dof),
        "root_state": (num_envs, 13),
        "dof_state": (num_envs, num_dof, 2),
        "body_state": (num_envs, num_bodies, 13),
        "contact_force": (num_envs, num_bodies, 3),
        "reset_env_ids": (num_envs,),
        "reset_qpos": (num_envs, 7 + num_dof),
        "reset_qvel": (num_envs, 6 + num_dof),
    }


def slot_dtype(name: str) -> np.dtype:
    try:
        return np.dtype(_SLOT_DTYPES[name])
    except KeyError as exc:
        raise ValueError(f"unknown shm slot {name!r}; known: {sorted(_SLOT_DTYPES)}") from exc


def slot_nbytes(name: str, shape: Tuple[int, ...]) -> int:
    return int(np.prod(shape, dtype=np.int64)) * int(slot_dtype(name).itemsize)


def serialize_exception(exc: BaseException) -> Dict[str, str]:
    return {
        "type": type(exc).__name__,
        "message": str(exc),
        "traceback": traceback.format_exc(),
    }


def format_worker_error(payload: Dict[str, str], backend: str = "subprocess") -> str:
    return (
        f"{backend} worker raised {payload.get('type', 'Error')}: "
        f"{payload.get('message', '')}\n"
        f"worker traceback:\n{payload.get('traceback', '<unavailable>')}"
    )


def xyzw_to_wxyz(quat: np.ndarray) -> np.ndarray:
    return np.asarray(quat)[..., [3, 0, 1, 2]]


def wxyz_to_xyzw(quat: np.ndarray) -> np.ndarray:
    return np.asarray(quat)[..., [1, 2, 3, 0]]


def quat_rotate(quat_wxyz: np.ndarray, vec: np.ndarray) -> np.ndarray:
    q = np.asarray(quat_wxyz, dtype=np.float64)
    v = np.asarray(vec, dtype=np.float64)
    w = q[..., 0:1]
    u = q[..., 1:4]
    uv = np.cross(u, v)
    uuv = np.cross(u, uv)
    return v + 2.0 * (w * uv + uuv)


def quat_rotate_inverse(quat_wxyz: np.ndarray, vec: np.ndarray) -> np.ndarray:
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
