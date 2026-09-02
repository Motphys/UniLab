"""Deterministic mock IsaacGym worker for protocol-level backend tests.

Runs on the host Python (no IsaacGym install required) and speaks the exact
``unilab.base.backend.isaacgym.protocol`` wire protocol against shared-memory
slots.  The physics model is a deterministic kinematic fake, documented here
so tests can assert exact values:

- STEP integrates ``dof_vel += ctrl * dt`` then ``dof_pos += dof_vel * dt``,
  and applies free fall to the root: ``lin_vel.z += -9.81 * dt`` then
  ``pos += lin_vel * dt`` (per substep).
- ``body_state`` row 0 mirrors the root; body ``i > 0`` sits at a static
  ``[0.1 * i, 0, 0]` offset from the root with the root's orientation.
- ``contact_force[:, 1, 0]`` is ``sum(|ctrl|)`` so contact "found" flips with
  nonzero control.
- SET_STATE applies the reset slots verbatim (root columns pass through
  without frame conversion — the mock does not model quat frame math).

Failure modes are selected with ``UNILAB_ISAACGYM_MOCK_BEHAVIOR``:
``ok`` (default), ``fail_init`` (raise during INIT), ``die_on_step``
(``os._exit(3)`` on the first STEP), ``hang_on_step`` (sleep on the first
STEP), ``no_graphics`` (INIT reports no graphics context and INIT_RENDERER
fails), ``viewer_fails`` (INIT_RENDERER cannot create the viewer),
``close_on_render`` (the first RENDER_FRAME reports the window closed),
``render_meta_missing``, ``render_meta_mode_mismatch``,
``render_meta_size_mismatch``, and ``render_meta_graphics_mismatch`` corrupt
the IsaacSim render handshake; ``capture_uniform``, ``capture_float``, and
``capture_wrong_shape`` return malformed camera frames.
Model dims come from ``UNILAB_ISAACGYM_MOCK_DOF_NAMES`` /
``UNILAB_ISAACGYM_MOCK_BODY_NAMES`` (comma-separated).
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys
import time
from typing import Any, Dict, List, Tuple

import numpy as np

_GRAVITY_Z = -9.81


def _load_protocol(path: str) -> Any:
    spec = importlib.util.spec_from_file_location("unilab_isaacgym_mock_protocol", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load protocol module from {path!r}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _MockSim:
    def __init__(
        self,
        num_envs: int,
        sim_dt: float,
        dof_names: List[str],
        body_names: List[str],
        graphics_enabled: bool = True,
        startup_render_mode: str | None = None,
        startup_width: int = 1280,
        startup_height: int = 720,
    ):
        self.graphics_enabled = graphics_enabled
        self.startup_render_mode = startup_render_mode
        self.startup_width = int(startup_width)
        self.startup_height = int(startup_height)
        self.num_envs = num_envs
        self.sim_dt = sim_dt
        self.dof_names = dof_names
        self.body_names = body_names
        self.num_dof = len(dof_names)
        self.num_bodies = len(body_names)
        self.root = np.zeros((num_envs, 13), dtype=np.float32)
        self.root[:, 2] = 1.0
        self.root[:, 3] = 1.0  # identity quat, wxyz
        self.dof = np.zeros((num_envs, self.num_dof, 2), dtype=np.float32)
        self.slots: Dict[str, np.ndarray] = {}
        self._shm_handles: List[Any] = []
        self.viewer_open = False
        self.capture_ready = False
        self.capture_width = 0
        self.capture_height = 0

    def attach(self, slots: Dict[str, Dict[str, Any]]) -> None:
        from multiprocessing import resource_tracker, shared_memory

        for name, spec in slots.items():
            handle = shared_memory.SharedMemory(name=spec["shm"], create=False)
            resource_tracker.unregister(handle._name, "shared_memory")
            self.slots[name] = np.ndarray(
                tuple(spec["shape"]), dtype=np.dtype(spec["dtype"]), buffer=handle.buf
            )
            self._shm_handles.append(handle)
        self.write_state_slots()

    def write_state_slots(self) -> None:
        np.copyto(self.slots["root_state"], self.root)
        np.copyto(self.slots["dof_state"], self.dof)
        body = self.slots["body_state"]
        for body_index in range(self.num_bodies):
            body[:, body_index, :] = self.root
            if body_index > 0:
                body[:, body_index, 0:3] = self.root[:, 0:3] + np.float32(0.1 * body_index) * (
                    np.array([1.0, 0.0, 0.0], dtype=np.float32)
                )
        contact = self.slots["contact_force"]
        contact.fill(0.0)
        if self.num_bodies > 1:
            # Tests bind contact sensors to the deepest link; report the mock
            # force there.
            contact[:, -1, 0] = np.abs(self.slots["ctrl"]).sum(axis=1)

    def step(self, nsteps: int) -> None:
        ctrl = self.slots["ctrl"]
        dt = np.float32(self.sim_dt)
        for _ in range(nsteps):
            self.dof[:, :, 1] += ctrl * dt
            self.dof[:, :, 0] += self.dof[:, :, 1] * dt
            self.root[:, 9] += np.float32(_GRAVITY_Z) * dt  # lin_vel z
            self.root[:, 0:3] += self.root[:, 7:10] * dt
        self.write_state_slots()

    def apply_keyframe(self, qpos_values: Any, joint_names: List[str]) -> None:
        """Apply the INIT keyframe pose with the same name-mapping rules as the
        real worker (root columns pass through in the mock's wxyz convention)."""
        qpos = np.asarray(qpos_values, dtype=np.float32).reshape(-1)
        expected = 7 + self.num_dof
        if qpos.size != expected:
            raise RuntimeError("keyframe qpos has %d entries; expected %d" % (qpos.size, expected))
        index_by_name = {name: index for index, name in enumerate(joint_names)}
        if len(joint_names) != self.num_dof:
            raise RuntimeError(
                "mjcf_joint_names has %d entries but the asset exposes %d dofs"
                % (len(joint_names), self.num_dof)
            )
        for dof_index, dof_name in enumerate(self.dof_names):
            if dof_name not in index_by_name:
                raise RuntimeError(
                    "isaacgym asset dof %r is missing from mjcf_joint_names" % dof_name
                )
            self.dof[:, dof_index, 0] = qpos[7 + index_by_name[dof_name]]
        self.root[:, 0:3] = qpos[0:3]
        self.root[:, 3:7] = qpos[3:7]

    def set_state(self, count: int) -> None:
        env_ids = self.slots["reset_env_ids"][:count].astype(np.int64)
        qpos = self.slots["reset_qpos"][:count]
        qvel = self.slots["reset_qvel"][:count]
        self.root[env_ids, 0:3] = qpos[:, 0:3]
        self.root[env_ids, 3:7] = qpos[:, 3:7]
        self.root[env_ids, 7:10] = qvel[:, 0:3]
        self.root[env_ids, 10:13] = qvel[:, 3:6]
        self.dof[env_ids, :, 0] = qpos[:, 7 : 7 + self.num_dof]
        self.dof[env_ids, :, 1] = qvel[:, 6 : 6 + self.num_dof]
        self.write_state_slots()

    def meta(self, behavior: str = "ok") -> Dict[str, Any]:
        result = {
            "num_dof": self.num_dof,
            "num_bodies": self.num_bodies,
            "dof_names": list(self.dof_names),
            "body_names": list(self.body_names),
            "dof_lower": [-1.5] * self.num_dof,
            "dof_upper": [1.5] * self.num_dof,
            "effort": [100.0] * self.num_dof,
            "gravity": [0.0, 0.0, _GRAVITY_Z],
            "use_gpu_pipeline": False,
            "graphics_enabled": self.graphics_enabled,
            # The IsaacSim host validates this optional subprocess metadata;
            # keeping it in the deterministic mock exercises that handshake
            # without importing Kit. IsaacGym simply ignores the fields.
            "env_origins": [[float(i) * 2.0, 0.0, 0.0] for i in range(self.num_envs)],
            "collision_filtering_applied": True,
        }
        if self.startup_render_mode is not None:
            result.update(
                {
                    "render_mode": self.startup_render_mode,
                    "render_width": self.startup_width,
                    "render_height": self.startup_height,
                }
            )
            if behavior == "render_meta_missing":
                result.pop("render_mode")
            elif behavior == "render_meta_mode_mismatch":
                result["render_mode"] = (
                    "record" if self.startup_render_mode != "record" else "interactive"
                )
            elif behavior == "render_meta_size_mismatch":
                result["render_width"] = self.startup_width + 1
            elif behavior == "render_meta_graphics_mismatch":
                result["graphics_enabled"] = not self.graphics_enabled
        return result

    def close(self) -> None:
        for handle in self._shm_handles:
            try:
                handle.close()
            except Exception:
                pass
        self._shm_handles = []

    def init_renderer(self, payload: Dict[str, Any], behavior: str) -> Dict[str, Any]:
        if not self.graphics_enabled:
            raise RuntimeError(
                "isaacgym rendering requires a GPU sim (device_id >= 0); this sim was "
                "created without a graphics context"
            )
        headless = bool(payload.get("headless", False))
        capture = bool(payload.get("capture", False))
        if self.startup_render_mode is not None:
            expected = "record" if (headless or capture) else "interactive"
            if expected != self.startup_render_mode:
                raise RuntimeError(
                    f"mock renderer mode mismatch: startup={self.startup_render_mode}, requested={expected}"
                )
            width = int(payload.get("width", self.startup_width))
            height = int(payload.get("height", self.startup_height))
            if (width, height) != (self.startup_width, self.startup_height):
                raise RuntimeError("mock renderer dimensions differ from INIT")
        if not headless:
            if behavior == "viewer_fails":
                raise RuntimeError(
                    "isaacgym create_viewer failed (no display reachable); use "
                    "play_render_mode=record for headless video capture"
                )
            self.viewer_open = True
        if capture:
            self.capture_ready = True
            self.capture_width = int(payload.get("width", 1280))
            self.capture_height = int(payload.get("height", 720))
        return {"viewer": self.viewer_open, "capture": self.capture_ready}

    def render_frame(self, behavior: str) -> Dict[str, Any]:
        if not self.viewer_open:
            raise RuntimeError("isaacgym viewer is not initialized; call INIT_RENDERER first")
        if behavior == "close_on_render":
            self.viewer_open = False
            return {"closed": True}
        return {"closed": False}

    def capture_frame(self, behavior: str = "ok") -> Dict[str, Any]:
        if not self.capture_ready:
            raise RuntimeError(
                "isaacgym capture camera is not initialized; call INIT_RENDERER first"
            )
        # Deterministic gradient so tests can verify the frame is not blank.
        rows = np.arange(self.capture_height, dtype=np.uint8)[:, None]
        cols = np.arange(self.capture_width, dtype=np.uint8)[None, :]
        frame = np.zeros((self.capture_height, self.capture_width, 3), dtype=np.uint8)
        frame[:, :, 0] = rows
        frame[:, :, 1] = cols
        frame[:, :, 2] = 128
        if behavior == "capture_uniform":
            frame.fill(7)
        elif behavior == "capture_float":
            frame = frame.astype(np.float32)
        elif behavior == "capture_wrong_shape":
            frame = frame[:, :, :2]
        return {"frame": frame, "width": self.capture_width, "height": self.capture_height}


def _csv_env(name: str, default: List[str]) -> List[str]:
    value = os.environ.get(name, "").strip()
    if not value:
        return list(default)
    return [part.strip() for part in value.split(",") if part.strip()]


def main(argv: List[str]) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", required=True)
    args = parser.parse_args(argv)
    protocol = _load_protocol(args.protocol)
    behavior = os.environ.get("UNILAB_ISAACGYM_MOCK_BEHAVIOR", "ok")

    sim: _MockSim | None = None
    stdin = sys.stdin.buffer
    stdout = sys.stdout.buffer

    def dispatch(cmd: str, payload: Any) -> Tuple[str, Any]:
        nonlocal sim
        if cmd == protocol.CMD_INIT:
            if behavior == "fail_init":
                raise RuntimeError("mock init failure")
            sim = _MockSim(
                num_envs=int(payload["num_envs"]),
                sim_dt=float(payload["sim_dt"]),
                dof_names=_csv_env("UNILAB_ISAACGYM_MOCK_DOF_NAMES", ["j0", "j1", "j2"]),
                body_names=_csv_env(
                    "UNILAB_ISAACGYM_MOCK_BODY_NAMES", ["base", "link0", "link1", "link2"]
                ),
                graphics_enabled=(
                    behavior != "no_graphics"
                    and str(payload.get("render_mode", "interactive")) != "none"
                ),
                startup_render_mode=(
                    str(payload["render_mode"]) if "render_mode" in payload else None
                ),
                startup_width=int(payload.get("render_width", 1280)),
                startup_height=int(payload.get("render_height", 720)),
            )
            keyframe_qpos = payload.get("keyframe_qpos")
            if keyframe_qpos is not None:
                sim.apply_keyframe(keyframe_qpos, payload.get("mjcf_joint_names") or [])
            return protocol.CMD_META, sim.meta(behavior)
        assert sim is not None
        if cmd == protocol.CMD_ATTACH:
            sim.attach(payload["slots"])
            return protocol.CMD_READY, None
        if cmd == protocol.CMD_STEP:
            if behavior == "die_on_step":
                sys.stderr.write("mock dying on step\n")
                sys.stderr.flush()
                os._exit(3)
            if behavior == "hang_on_step":
                time.sleep(3600.0)
            t0 = time.perf_counter()
            sim.step(int(payload["nsteps"]))
            physics_ms = (time.perf_counter() - t0) * 1000.0
            return protocol.CMD_READY, {
                "timing": {
                    "control_upload_ms": 0.0,
                    "physics_ms": physics_ms,
                    "state_refresh_ms": 0.0,
                }
            }
        if cmd == protocol.CMD_SET_STATE:
            sim.set_state(int(payload["count"]))
            return protocol.CMD_READY, {
                "timing": {
                    "set_state_reset_upload_ms": 0.0,
                    "set_state_host_cache_refresh_ms": 0.0,
                }
            }
        if cmd == protocol.CMD_REFRESH:
            sim.write_state_slots()
            return protocol.CMD_READY, None
        if cmd == protocol.CMD_GET_META:
            return protocol.CMD_META, sim.meta(behavior)
        if cmd == protocol.CMD_INIT_RENDERER:
            return protocol.CMD_META, sim.init_renderer(payload, behavior)
        if cmd == protocol.CMD_RENDER_FRAME:
            return protocol.CMD_META, sim.render_frame(behavior)
        if cmd == protocol.CMD_CAPTURE_FRAME:
            return protocol.CMD_META, sim.capture_frame(behavior)
        raise ValueError(f"unknown command {cmd!r}")

    while True:
        try:
            message = protocol.recv_message(stdin)
        except (EOFError, protocol.WorkerDisconnectedError):
            return 0
        cmd = message["cmd"]
        if cmd == protocol.CMD_SHUTDOWN:
            if sim is not None:
                sim.close()
            protocol.send_message(stdout, protocol.CMD_READY)
            return 0
        try:
            reply_cmd, reply_payload = dispatch(cmd, message.get("payload"))
        except Exception as exc:  # noqa: BLE001 - every worker error crosses the wire
            protocol.send_message(stdout, protocol.CMD_ERROR, protocol.serialize_exception(exc))
            continue
        protocol.send_message(stdout, reply_cmd, reply_payload)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
