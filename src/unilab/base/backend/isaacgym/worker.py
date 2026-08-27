"""Python 3.8 worker process for the out-of-process IsaacGym backend.

PYTHON 3.8 COMPATIBILITY: IsaacGym (Preview 4, EOL) only supports Python
3.6-3.8, so this file runs on the dedicated ``hsgym`` conda interpreter.  Keep
it stdlib + numpy + torch + isaacgym only, and never import ``unilab`` — the
shared protocol module is loaded by file path (``--protocol``) because the
worker interpreter has no access to the main environment's site-packages.

Message loop: read one framed command from stdin, dispatch, write one framed
reply to stdout.  Bulk state crosses the process boundary through shared
memory slots declared by the host (see ``protocol.slot_shapes``); the pipe
only carries commands, metadata, and error payloads.
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys
import time
from typing import Any, Dict, List, Tuple

import numpy as np


def _load_protocol(path: str) -> Any:
    """Load the shared protocol module by file path (no package import)."""
    spec = importlib.util.spec_from_file_location("unilab_isaacgym_protocol", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load protocol module from {path!r}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _WorkerContext:
    """Owns the IsaacGym sim, tensor views, and attached shared-memory slots."""

    def __init__(self, protocol: Any) -> None:
        self.protocol = protocol
        # IsaacGym/torch modules are imported inside init_sim; they do not
        # exist on the host interpreter, so these stay ``Any``.
        self.gymapi: Any = None
        self.gymtorch: Any = None
        self.torch: Any = None
        self.gym: Any = None
        self.sim: Any = None
        self.num_envs = 0
        self.num_dof = 0
        self.num_bodies = 0
        self.sim_dt = 0.0
        self.device = "cpu"
        self.use_gpu_pipeline = False
        self.env_handles: List[Any] = []
        self.actor_handles: List[Any] = []
        self.slots: Dict[str, np.ndarray] = {}
        self._shm_handles: List[Any] = []
        self._root_state: Any = None
        self._dof_state: Any = None
        self._body_state: Any = None
        self._contact_force: Any = None

    # ------------------------------------------------------------------ #
    # INIT
    # ------------------------------------------------------------------ #

    def init_sim(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        isaacgym_python = payload["isaacgym_python"]
        if isaacgym_python not in sys.path:
            sys.path.insert(0, isaacgym_python)
        import torch  # noqa: PLC0415
        from isaacgym import gymapi, gymtorch  # noqa: PLC0415

        self.gymapi = gymapi
        self.gymtorch = gymtorch
        self.torch = torch

        gymapi = self.gymapi
        self.num_envs = int(payload["num_envs"])
        self.sim_dt = float(payload["sim_dt"])
        device_id = int(payload.get("device_id", 0))
        self.use_gpu_pipeline = device_id >= 0
        self.device = "cuda:%d" % device_id if self.use_gpu_pipeline else "cpu"

        self.gym = gymapi.acquire_gym()
        sim_params = gymapi.SimParams()
        sim_params.dt = self.sim_dt
        sim_params.substeps = 1
        sim_params.up_axis = gymapi.UpAxis.UP_AXIS_Z
        sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.81)
        sim_params.physx.solver_type = 1
        sim_params.physx.num_position_iterations = 4
        sim_params.physx.num_velocity_iterations = 1
        sim_params.physx.num_threads = 0
        sim_params.physx.use_gpu = self.use_gpu_pipeline
        sim_params.use_gpu_pipeline = self.use_gpu_pipeline
        graphics_device_id = -1 if payload.get("headless", True) else device_id
        self.sim = self.gym.create_sim(device_id, graphics_device_id, gymapi.SIM_PHYSX, sim_params)
        if self.sim is None:
            raise RuntimeError(
                "isaacgym create_sim failed (device_id=%d, gpu_pipeline=%s)"
                % (device_id, self.use_gpu_pipeline)
            )

        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        self.gym.add_ground(self.sim, plane_params)

        model_file = os.fspath(payload["model_file"])
        asset_root, asset_file = os.path.split(model_file)
        if not asset_file.lower().endswith((".xml", ".mjcf")):
            raise RuntimeError(
                "isaacgym backend currently loads MJCF scenes only; got asset file "
                "%r. Convert the task scene or extend the worker asset loader." % asset_file
            )
        asset_options = gymapi.AssetOptions()
        asset_options.flip_visual_attachments = True
        asset_options.default_dof_drive_mode = int(gymapi.DOF_MODE_EFFORT)
        asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)
        if asset is None:
            raise RuntimeError(
                "isaacgym load_asset failed for %r. MJCF import requires the file to be "
                "self-contained for IsaacGym's importer (some MuJoCo elements are "
                "unsupported); run the worker command manually for the importer log." % model_file
            )

        dof_props = self.gym.get_asset_dof_properties(asset)
        # Torque-controlled dofs: ctrl maps 1:1 to actuation force, matching the
        # MuJoCo motor semantics of SimBackend.step(ctrl).
        dof_props["driveMode"][:].fill(gymapi.DOF_MODE_EFFORT)
        dof_props["stiffness"][:].fill(0.0)
        dof_props["damping"][:].fill(0.0)

        spacing = 2.0
        num_per_row = max(1, int(np.ceil(np.sqrt(self.num_envs))))
        env_lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        env_upper = gymapi.Vec3(spacing, spacing, 0.0)
        pose = gymapi.Transform()
        pose.p = gymapi.Vec3(0.0, 0.0, 0.0)
        pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)
        for env_index in range(self.num_envs):
            env_handle = self.gym.create_env(self.sim, env_lower, env_upper, num_per_row)
            actor_handle = self.gym.create_actor(env_handle, asset, pose, "robot", env_index, 0)
            self.gym.set_actor_dof_properties(env_handle, actor_handle, dof_props)
            self.env_handles.append(env_handle)
            self.actor_handles.append(actor_handle)

        self.gym.prepare_sim(self.sim)
        self._acquire_tensors()

        self.num_dof = int(self.gym.get_asset_dof_count(asset))
        self.num_bodies = int(self.gym.get_asset_rigid_body_count(asset))
        lower = np.asarray(dof_props["lower"], dtype=np.float64)
        upper = np.asarray(dof_props["upper"], dtype=np.float64)
        effort = np.asarray(dof_props["effort"], dtype=np.float64)
        return {
            "num_dof": self.num_dof,
            "num_bodies": self.num_bodies,
            "dof_names": list(self.gym.get_asset_dof_names(asset)),
            "body_names": list(self.gym.get_asset_rigid_body_names(asset)),
            "dof_lower": lower.tolist(),
            "dof_upper": upper.tolist(),
            "effort": effort.tolist(),
            "gravity": [0.0, 0.0, -9.81],
            "use_gpu_pipeline": self.use_gpu_pipeline,
        }

    def _acquire_tensors(self) -> None:
        gym = self.gym
        gymtorch = self.gymtorch
        self._root_state = gymtorch.wrap_tensor(gym.acquire_actor_root_state_tensor(self.sim))
        self._dof_state = gymtorch.wrap_tensor(gym.acquire_dof_state_tensor(self.sim))
        self._body_state = gymtorch.wrap_tensor(gym.acquire_rigid_body_state_tensor(self.sim))
        self._contact_force = gymtorch.wrap_tensor(gym.acquire_net_contact_force_tensor(self.sim))

    # ------------------------------------------------------------------ #
    # Shared-memory slots
    # ------------------------------------------------------------------ #

    def attach_slots(self, payload: Dict[str, Any]) -> None:
        """Attach host-created shm slots and detach them from resource tracking.

        Python's shared_memory resource tracker would otherwise unlink the
        host-owned segments when this worker exits (CPython issue 39959), so
        every attached name is unregistered here; the host owns unlinking.
        """
        from multiprocessing import resource_tracker, shared_memory  # noqa: PLC0415

        for name, spec in payload["slots"].items():
            handle = shared_memory.SharedMemory(name=spec["shm"], create=False)
            resource_tracker.unregister(handle._name, "shared_memory")  # type: ignore[attr-defined]
            array = np.ndarray(
                tuple(spec["shape"]), dtype=np.dtype(spec["dtype"]), buffer=handle.buf
            )
            self.slots[name] = array
            self._shm_handles.append(handle)
        self.refresh_state_slots()

    # ------------------------------------------------------------------ #
    # State exchange
    # ------------------------------------------------------------------ #

    def _refresh_tensors(self) -> None:
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)

    def refresh_state_slots(self) -> None:
        """Copy the latest tensor state into every host-visible shm slot."""
        protocol = self.protocol
        self._refresh_tensors()
        root = self._root_state.view(self.num_envs, -1, 13)[:, 0, :].cpu().numpy()
        root_slot = self.slots["root_state"]
        root_slot[:, 0:3] = root[:, 0:3]
        root_slot[:, 3:7] = protocol.xyzw_to_wxyz(root[:, 3:7])
        root_slot[:, 7:13] = root[:, 7:13]
        np.copyto(
            self.slots["dof_state"],
            self._dof_state.view(self.num_envs, self.num_dof, 2).cpu().numpy(),
        )
        bodies = self._body_state.view(self.num_envs, self.num_bodies, 13).cpu().numpy()
        body_slot = self.slots["body_state"]
        body_slot[:, :, 0:3] = bodies[:, :, 0:3]
        body_slot[:, :, 3:7] = protocol.xyzw_to_wxyz(bodies[:, :, 3:7])
        body_slot[:, :, 7:13] = bodies[:, :, 7:13]
        np.copyto(
            self.slots["contact_force"],
            self._contact_force.view(self.num_envs, self.num_bodies, 3).cpu().numpy(),
        )

    def step(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        nsteps = int(payload["nsteps"])
        timings: Dict[str, float] = {}
        t0 = time.perf_counter()
        torch_ctrl = self.torch.from_numpy(np.ascontiguousarray(self.slots["ctrl"])).to(self.device)
        self.gym.set_dof_actuation_force_tensor(
            self.sim, self.gymtorch.unwrap_tensor(torch_ctrl.reshape(-1).contiguous())
        )
        timings["control_upload_ms"] = (time.perf_counter() - t0) * 1000.0

        t0 = time.perf_counter()
        for _ in range(nsteps):
            self.gym.simulate(self.sim)
            self.gym.fetch_results(self.sim, True)
        timings["physics_ms"] = (time.perf_counter() - t0) * 1000.0

        t0 = time.perf_counter()
        self.refresh_state_slots()
        timings["state_refresh_ms"] = (time.perf_counter() - t0) * 1000.0
        return {"timing": timings}

    def set_state(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        protocol = self.protocol
        torch = self.torch
        timings: Dict[str, float] = {}
        t0 = time.perf_counter()
        count = int(payload["count"])
        env_ids = np.ascontiguousarray(self.slots["reset_env_ids"][:count])
        qpos = np.ascontiguousarray(self.slots["reset_qpos"][:count])
        qvel = np.ascontiguousarray(self.slots["reset_qvel"][:count])

        root = np.zeros((count, 13), dtype=np.float32)
        root[:, 0:3] = qpos[:, 0:3]
        root[:, 3:7] = protocol.wxyz_to_xyzw(qpos[:, 3:7])
        root[:, 7:10] = qvel[:, 0:3]
        # Contract qvel carries body-frame angular velocity; IsaacGym root
        # states take world-frame angular velocity.
        root[:, 10:13] = protocol.quat_rotate(qpos[:, 3:7], qvel[:, 3:6]).astype(np.float32)
        # Indexed writes mutate the shared wrapped buffers in place and then
        # commit through the full tensors (the IsaacGym indexed API pattern:
        # one actor per env, so the global actor index equals the env index).
        env_id_tensor = torch.from_numpy(env_ids.astype(np.int32)).to(self.device)
        root_view = self._root_state.view(self.num_envs, -1, 13)
        root_view[env_id_tensor.long(), 0, :] = torch.from_numpy(root).to(self.device)
        self.gym.set_actor_root_state_tensor_indexed(
            self.sim,
            self.gymtorch.unwrap_tensor(self._root_state),
            self.gymtorch.unwrap_tensor(env_id_tensor),
            count,
        )

        dof = np.zeros((count, self.num_dof, 2), dtype=np.float32)
        dof[:, :, 0] = qpos[:, 7 : 7 + self.num_dof]
        dof[:, :, 1] = qvel[:, 6 : 6 + self.num_dof]
        dof_view = self._dof_state.view(self.num_envs, self.num_dof, 2)
        dof_view[env_id_tensor.long(), :, :] = torch.from_numpy(dof).to(self.device)
        self.gym.set_dof_state_tensor_indexed(
            self.sim,
            self.gymtorch.unwrap_tensor(self._dof_state),
            self.gymtorch.unwrap_tensor(env_id_tensor),
            count,
        )
        timings["set_state_reset_upload_ms"] = (time.perf_counter() - t0) * 1000.0

        # IsaacGym has no kinematics-only forward call; root/dof slots reflect
        # the applied state immediately, while body/contact slots stay as of
        # the last physics step until the next STEP.
        t0 = time.perf_counter()
        self.refresh_state_slots()
        timings["set_state_host_cache_refresh_ms"] = (time.perf_counter() - t0) * 1000.0
        return {"timing": timings}

    def get_meta(self) -> Dict[str, Any]:
        return {
            "num_dof": self.num_dof,
            "num_bodies": self.num_bodies,
            "use_gpu_pipeline": self.use_gpu_pipeline,
        }

    def shutdown(self) -> None:
        if self.gym is not None and self.sim is not None:
            self.gym.destroy_sim(self.sim)
            self.sim = None
        for handle in self._shm_handles:
            try:
                handle.close()
            except Exception:
                pass
        self._shm_handles = []


def _dispatch(ctx: _WorkerContext, protocol: Any, cmd: str, payload: Any) -> Tuple[str, Any]:
    if cmd == protocol.CMD_INIT:
        return protocol.CMD_META, ctx.init_sim(payload)
    if cmd == protocol.CMD_ATTACH:
        ctx.attach_slots(payload)
        return protocol.CMD_READY, None
    if cmd == protocol.CMD_STEP:
        return protocol.CMD_READY, ctx.step(payload)
    if cmd == protocol.CMD_SET_STATE:
        return protocol.CMD_READY, ctx.set_state(payload)
    if cmd == protocol.CMD_REFRESH:
        ctx.refresh_state_slots()
        return protocol.CMD_READY, None
    if cmd == protocol.CMD_GET_META:
        return protocol.CMD_META, ctx.get_meta()
    raise ValueError(f"unknown command {cmd!r}")


def main(argv: List[str]) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", required=True, help="path to protocol.py")
    args = parser.parse_args(argv)
    protocol = _load_protocol(args.protocol)
    ctx = _WorkerContext(protocol)

    stdin = sys.stdin.buffer
    stdout = sys.stdout.buffer
    while True:
        try:
            message = protocol.recv_message(stdin)
        except (EOFError, protocol.WorkerDisconnectedError):
            return 0
        cmd = message["cmd"]
        payload = message.get("payload")
        if cmd == protocol.CMD_SHUTDOWN:
            try:
                ctx.shutdown()
            finally:
                protocol.send_message(stdout, protocol.CMD_READY)
            return 0
        try:
            reply_cmd, reply_payload = _dispatch(ctx, protocol, cmd, payload)
        except Exception as exc:  # noqa: BLE001 - every worker error crosses the wire
            protocol.send_message(stdout, protocol.CMD_ERROR, protocol.serialize_exception(exc))
            continue
        protocol.send_message(stdout, reply_cmd, reply_payload)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
