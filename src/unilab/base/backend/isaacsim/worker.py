"""IsaacSim/IsaacLab Python 3.11 worker for the UniLab subprocess backend.

The worker is intentionally self-contained.  It imports Kit and IsaacLab only
after receiving ``INIT`` and communicates with the host through the canonical
``subprocess_ipc.protocol`` module loaded by path.  Control messages stay on
the pipe; all batched numeric state is copied into shared-memory slots.

This worker implements the currently supported IsaacSim slice: headless
MJCF-backed articulation physics, masked root/joint reset, and implicit
position-target control.  Rendering and domain-randomization commands fail
closed at the host contract boundary.
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys
import time
from typing import Any

import numpy as np


def _load_protocol(path: str) -> Any:
    spec = importlib.util.spec_from_file_location("unilab_subprocess_protocol", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load protocol module from {path!r}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _tensor_numpy(value: Any) -> np.ndarray:
    """Detach one IsaacLab tensor at the worker/shm boundary."""
    # IsaacLab's ``Articulation.data`` quantities are torch tensors.  Keep the
    # conversion explicit at this one cold/IO boundary: probing arbitrary
    # backend objects with ``hasattr``/``getattr`` in the physics path can hide
    # API drift and violates the backend-isolation contract.  A non-tensor is
    # an implementation error and should fail loudly instead of being
    # silently coerced through NumPy.
    return value.detach().cpu().numpy().astype(np.float32, copy=False)


def _to_tensor(torch: Any, value: np.ndarray, device: str) -> Any:
    return torch.as_tensor(np.ascontiguousarray(value), dtype=torch.float32, device=device)


def _quat_rotate_wxyz(quat: np.ndarray, vec: np.ndarray) -> np.ndarray:
    """Rotate vectors by wxyz quaternions (used for reset body->world angvel)."""
    q = np.asarray(quat, dtype=np.float64)
    v = np.asarray(vec, dtype=np.float64)
    w = q[..., 0:1]
    u = q[..., 1:4]
    uv = np.cross(u, v)
    uuv = np.cross(u, uv)
    return (v + 2.0 * (w * uv + uuv)).astype(np.float32)


def _resolve_articulation_root_prim_path(usd_path: str, root_name: str) -> str:
    """Resolve the imported articulation root relative to the asset prim.

    MJCF conversion does not promise a fixed nesting depth.  G1 currently
    yields ``/<asset>/<pelvis>/<pelvis>`` while other assets may expose the
    articulation root directly under the asset prim.  Discover the prim once
    during materialization and fail closed when the requested root is
    ambiguous or absent; never guess this path on a physics hot path.
    """
    if not root_name:
        raise ValueError("root_name must be a non-empty body name")
    try:
        from pxr import Usd, UsdPhysics  # type: ignore[import-not-found]
    except ImportError as exc:  # pragma: no cover - only runs in external worker
        raise RuntimeError("IsaacSim USD bindings are unavailable") from exc

    stage = Usd.Stage.Open(str(usd_path))
    if stage is None:
        raise RuntimeError(f"IsaacSim could not open converted USD asset {usd_path!r}")
    default_prim = stage.GetDefaultPrim()
    if not default_prim or not default_prim.IsValid():
        raise RuntimeError(f"IsaacSim converted USD asset {usd_path!r} has no valid default prim")
    asset_path = str(default_prim.GetPath()).rstrip("/")
    candidates = []
    for prim in stage.Traverse():
        path = str(prim.GetPath())
        if not path.startswith(asset_path + "/"):
            continue
        if path.rsplit("/", 1)[-1] != root_name:
            continue
        if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
            candidates.append(path)
    if len(candidates) != 1:
        raise RuntimeError(
            "IsaacSim converted USD articulation root lookup for body "
            f"{root_name!r} expected one ArticulationRootAPI prim below "
            f"{asset_path!r}, found {candidates or '<none>'}"
        )
    relative = candidates[0][len(asset_path) :]
    if not relative.startswith("/"):
        raise RuntimeError(
            f"IsaacSim articulation root {candidates[0]!r} is not below {asset_path!r}"
        )
    return relative


class _WorkerContext:
    def __init__(self, protocol: Any) -> None:
        self.protocol = protocol
        self.num_envs = 0
        self.num_dof = 0
        self.num_bodies = 0
        self.sim_dt = 0.0
        self.device = "cuda:0"
        self.sim: Any = None
        self.robot: Any = None
        self.simulation_app: Any = None
        self.torch: Any = None
        self.native_joint_names: list[str] = []
        self.native_body_names: list[str] = []
        self.contract_joint_names: list[str] = []
        self.contract_body_names: list[str] = []
        self.native_joint_for_contract: np.ndarray = np.empty(0, dtype=np.int64)
        self.native_body_for_contract: np.ndarray = np.empty(0, dtype=np.int64)
        # Physical clones are translated apart in the worker so that their
        # collision geometry does not overlap.  UniLab's flat-scene contract
        # exposes per-environment local coordinates (``env_origins`` is zero),
        # therefore these offsets stay private to the worker and are removed
        # at the shared-memory boundary.
        self.env_origins = np.empty((0, 3), dtype=np.float32)
        self.env_prim_paths: list[str] = []
        self.collision_filtering_applied = False
        self.slots: dict[str, np.ndarray] = {}
        self._shm_handles: list[Any] = []

    # ------------------------------------------------------------------
    # Cold-path materialization
    # ------------------------------------------------------------------

    def init_sim(self, payload: dict[str, Any]) -> dict[str, Any]:
        os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "1")
        self.num_envs = int(payload["num_envs"])
        self.sim_dt = float(payload["sim_dt"])
        device_id = int(payload.get("device_id", 0))
        if device_id < 0:
            raise NotImplementedError(
                "isaacsim requires a CUDA device; CPU IsaacLab physics is outside the "
                "supported subprocess profile"
            )
        self.device = f"cuda:{device_id}"

        # Kit must be launched before importing IsaacSim/IsaacLab modules.
        from isaaclab.app import AppLauncher  # type: ignore[import-not-found]

        self.simulation_app = AppLauncher({"headless": True, "device": self.device}).app

        import isaaclab.sim as sim_utils  # type: ignore[import-not-found]
        import isaacsim.core.utils.prims as prim_utils  # type: ignore[import-not-found]
        import torch  # type: ignore[import-not-found]
        from isaaclab.actuators import ImplicitActuatorCfg  # type: ignore[import-not-found]
        from isaaclab.assets import Articulation, ArticulationCfg  # type: ignore[import-not-found]
        from isaaclab.sim.converters import (  # type: ignore[import-not-found]
            MjcfConverter,
            MjcfConverterCfg,
        )
        from isaacsim.core.cloner import GridCloner  # type: ignore[import-not-found]
        from isaacsim.core.utils.extensions import (
            enable_extension,  # type: ignore[import-not-found]
        )

        self.torch = torch
        # The extension is enabled explicitly because IsaacSim 5.1 does not
        # guarantee the MJCF importer is active in a bare headless AppLauncher.
        enable_extension("isaacsim.asset.importer.mjcf")

        model_file = os.fspath(payload["model_file"])
        converter = MjcfConverter(
            MjcfConverterCfg(
                asset_path=model_file,
                fix_base=False,
                import_sites=True,
                make_instanceable=True,
                self_collision=False,
            )
        )

        # Build a deterministic environment grid.  The USD importer owns the
        # robot hierarchy; only these Xforms and the articulation wrapper are
        # created here, so no asset/XML parsing occurs on a hot path.  The
        # translations are private worker offsets; state is normalized back to
        # local coordinates before it is published to the host.
        cloner = GridCloner(spacing=2.0)
        cloner.define_base_env("/World/envs")
        self.env_prim_paths = cloner.generate_paths("/World/envs/env", self.num_envs)
        # The source Xform must exist before GridCloner.clone.  The returned
        # transforms are the authoritative world origins (a centered grid for
        # two or more environments), so no duplicate hand-written grid math is
        # needed here.
        prim_utils.create_prim(self.env_prim_paths[0], "Xform")
        self.env_origins = np.asarray(
            cloner.clone(
                source_prim_path=self.env_prim_paths[0],
                prim_paths=self.env_prim_paths,
                replicate_physics=False,
                copy_from_source=True,
            ),
            dtype=np.float32,
        )
        expected_origins = (self.num_envs, 3)
        if self.env_origins.shape != expected_origins or not np.isfinite(self.env_origins).all():
            raise RuntimeError(
                "IsaacSim GridCloner returned invalid environment origins: "
                f"shape={self.env_origins.shape}, expected={expected_origins}"
            )

        root_name = str(payload.get("root_body_name") or "")
        if not root_name:
            raise ValueError(
                "isaacsim INIT requires root_body_name so articulation_root_prim_path "
                "is explicit and importer discovery cannot choose a wrong root"
            )
        # IsaacLab resolves this path relative to each /Robot instance.  The
        # converter's nesting is asset-dependent, so discover it from the
        # converted USD stage rather than baking in the G1 layout.
        articulation_root = _resolve_articulation_root_prim_path(converter.usd_path, root_name)
        joint_names = [str(name) for name in (payload.get("mjcf_joint_names") or [])]
        if not joint_names:
            raise ValueError("isaacsim INIT requires the MJCF joint-name contract")
        gains = self._actuator_dicts(payload, joint_names)
        robot_cfg = ArticulationCfg(
            prim_path="/World/envs/env_.*/Robot",
            articulation_root_prim_path=articulation_root,
            spawn=sim_utils.UsdFileCfg(usd_path=converter.usd_path),
            actuators={
                "all": ImplicitActuatorCfg(
                    joint_names_expr=[".*"],
                    stiffness=gains["stiffness"],
                    damping=gains["damping"],
                    effort_limit_sim=gains["effort"],
                    armature=gains["armature"],
                    friction=gains["friction"],
                )
            },
        )
        self.robot = Articulation(robot_cfg)
        sim_cfg = sim_utils.SimulationCfg(dt=self.sim_dt, device=self.device)
        self.sim = sim_utils.SimulationContext(sim_cfg)
        # Apply IsaacLab's PhysX collision-group filtering before the first
        # reset/step.  Without this stage operation, the translated clones
        # can still collide when a reset puts two local roots at the same pose.
        # Failing closed is important: an unfiltered batch is not equivalent
        # to the SimBackend's independent-environment contract.
        if self.num_envs > 1:
            cloner.filter_collisions(
                self._physics_scene_path(),
                "/World/collisions",
                self.env_prim_paths,
            )
            self.collision_filtering_applied = True
        self.sim.reset()
        self.robot.update(self.sim_dt)

        self.native_joint_names = [str(name) for name in self.robot.joint_names]
        self.native_body_names = [str(name) for name in self.robot.body_names]
        self.num_dof = int(self.robot.num_joints)
        self.num_bodies = int(self.robot.num_bodies)
        self.contract_joint_names = joint_names
        self.contract_body_names = [
            str(name) for name in (payload.get("mjcf_body_names") or self.native_body_names)
        ]
        self.native_joint_for_contract = self._build_permutation(
            self.native_joint_names, self.contract_joint_names, "joint"
        )
        self.native_body_for_contract = self._build_permutation(
            self.native_body_names, self.contract_body_names, "body"
        )

        keyframe_qpos = payload.get("keyframe_qpos")
        if keyframe_qpos is not None:
            self._apply_keyframe(keyframe_qpos)
        return {
            "num_dof": self.num_dof,
            "num_bodies": self.num_bodies,
            # Expose UniLab contract order, not the importer/native order.
            "dof_names": list(self.contract_joint_names),
            "body_names": list(self.contract_body_names),
            "dof_lower": self._joint_limits()[0],
            "dof_upper": self._joint_limits()[1],
            "effort": self._joint_limits()[2],
            "gravity": [0.0, 0.0, -9.81],
            "use_gpu_pipeline": True,
            "graphics_enabled": False,
            "native_dof_names": list(self.native_joint_names),
            "native_body_names": list(self.native_body_names),
            "usd_path": str(converter.usd_path),
            "env_origins": self.env_origins.tolist(),
            "collision_filtering_applied": self.collision_filtering_applied,
        }

    def _physics_scene_path(self) -> str:
        """Find the stage's PhysX scene prim on the materialization path."""
        try:
            from pxr import PhysxSchema  # type: ignore[import-not-found]
        except ImportError as exc:  # pragma: no cover - external worker only
            raise RuntimeError("IsaacSim PhysX USD bindings are unavailable") from exc
        for prim in self.robot.stage.Traverse():
            if prim.HasAPI(PhysxSchema.PhysxSceneAPI):
                return str(prim.GetPath())
        raise RuntimeError(
            "IsaacSim stage has no PhysxSceneAPI; cannot filter environment collisions"
        )

    @staticmethod
    def _build_permutation(native: list[str], contract: list[str], kind: str) -> np.ndarray:
        if len(native) != len(contract) or len(set(native)) != len(native):
            raise RuntimeError(
                f"isaacsim importer returned invalid {kind} names: native={native}, contract={contract}"
            )
        native_ids = {name: index for index, name in enumerate(native)}
        missing = [name for name in contract if name not in native_ids]
        extra = [name for name in native if name not in set(contract)]
        if missing or extra or len(set(contract)) != len(contract):
            raise RuntimeError(
                f"isaacsim importer {kind} mapping mismatch: missing={missing}, extra={extra}, "
                f"native={native}, contract={contract}"
            )
        return np.asarray([native_ids[name] for name in contract], dtype=np.int64)

    @staticmethod
    def _actuator_dicts(payload: dict[str, Any], names: list[str]) -> dict[str, dict[str, float]]:
        def values(key: str, default: float = 0.0) -> dict[str, float]:
            raw = list(payload.get(key) or [])
            if len(raw) != len(names):
                raise RuntimeError(
                    f"{key} has {len(raw)} values but the MJCF contract has {len(names)} joints"
                )
            return {name: float(raw[index]) for index, name in enumerate(names)}

        effort = values("dof_effort")
        # PhysX/IsaacLab reject an infinite or excessively large effort in
        # some releases.  The host uses 1e20 as the unlimited sentinel; use
        # the documented finite implicit-actuator ceiling in the worker.
        effort = {
            name: (1.0e9 if value <= 0.0 or value >= 1.0e19 else value)
            for name, value in effort.items()
        }
        return {
            "stiffness": values("dof_stiffness"),
            "damping": values("dof_damping"),
            "effort": effort,
            "armature": values("dof_armature"),
            "friction": values("dof_friction"),
        }

    def _joint_limits(self) -> tuple[list[float], list[float], list[float]]:
        limits = _tensor_numpy(self.robot.data.joint_pos_limits)[0]
        efforts = _tensor_numpy(self.robot.data.joint_effort_limits)[0]
        # Reorder native metadata to the public contract order.
        limits = limits[self.native_joint_for_contract]
        efforts = efforts[self.native_joint_for_contract]
        return limits[:, 0].tolist(), limits[:, 1].tolist(), efforts.tolist()

    def _apply_keyframe(self, qpos_values: Any) -> None:
        qpos = np.asarray(qpos_values, dtype=np.float32).reshape(-1)
        if qpos.size != 7 + self.num_dof:
            raise RuntimeError(
                f"keyframe qpos has {qpos.size} entries; expected {7 + self.num_dof}"
            )
        env_ids = self.torch.arange(self.num_envs, dtype=self.torch.long, device=self.device)
        root_pose_np = np.broadcast_to(qpos[:7], (self.num_envs, 7)).copy()
        root_pose_np[:, :3] += self.env_origins
        root_pose = _to_tensor(self.torch, root_pose_np, self.device)
        root_vel = self.torch.zeros(
            (self.num_envs, 6), dtype=self.torch.float32, device=self.device
        )
        native_pos = np.zeros((self.num_envs, self.num_dof), dtype=np.float32)
        native_pos[:, self.native_joint_for_contract] = qpos[7:][None, :]
        joint_pos = _to_tensor(self.torch, native_pos, self.device)
        joint_vel = self.torch.zeros_like(joint_pos)
        self.robot.write_root_pose_to_sim(root_pose, env_ids=env_ids)
        # UniLab's root state is the link-frame state.  IsaacLab's similarly
        # named ``write_root_velocity_to_sim`` targets the COM frame, so use
        # the explicit link writer here.
        self.robot.write_root_link_velocity_to_sim(root_vel, env_ids=env_ids)
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
        self.robot.reset(env_ids)
        self.robot.update(self.sim_dt)

    # ------------------------------------------------------------------
    # Shared-memory attachment and state exchange
    # ------------------------------------------------------------------

    def attach_slots(self, payload: dict[str, Any]) -> None:
        from multiprocessing import resource_tracker, shared_memory

        for name, spec in payload["slots"].items():
            handle = shared_memory.SharedMemory(name=spec["shm"], create=False)
            # The host owns unlinking; prevent the worker's resource tracker
            # from unlinking the segment when Kit exits.
            resource_tracker.unregister(handle._name, "shared_memory")  # type: ignore[attr-defined]
            self.slots[name] = np.ndarray(
                tuple(spec["shape"]), dtype=np.dtype(spec["dtype"]), buffer=handle.buf
            )
            self._shm_handles.append(handle)
        self.refresh_state_slots()

    def _state_tensors(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        data = self.robot.data
        root = _tensor_numpy(data.root_link_state_w)
        dof_pos = _tensor_numpy(data.joint_pos)
        dof_vel = _tensor_numpy(data.joint_vel)
        body = _tensor_numpy(data.body_link_state_w)
        if root.shape != (self.num_envs, 13):
            raise RuntimeError(
                f"IsaacLab root state shape is {root.shape}, expected ({self.num_envs}, 13)"
            )
        # Return root, dof(pos/vel), body separately; body is reordered below.
        dof = np.stack((dof_pos, dof_vel), axis=-1)
        return root, dof, body

    def refresh_state_slots(self) -> None:
        root, dof, body = self._state_tensors()
        # IsaacLab reports world-frame positions.  Remove the private clone
        # translation before publishing UniLab's local-frame state.
        root = root.copy()
        body = body.copy()
        root[:, :3] -= self.env_origins
        body[:, :, :3] -= self.env_origins[:, None, :]
        np.copyto(self.slots["root_state"], root)
        np.copyto(self.slots["dof_state"], dof[:, self.native_joint_for_contract, :])
        np.copyto(self.slots["body_state"], body[:, self.native_body_for_contract, :])
        # IsaacLab's Articulation tensor does not expose a generic net-contact
        # force slot.  Keep the slot deterministic and let the host sensor map
        # fail closed for contact declarations.
        self.slots["contact_force"].fill(0.0)

    def step(self, payload: dict[str, Any]) -> dict[str, Any]:
        ctrl = np.asarray(self.slots["ctrl"], dtype=np.float32)
        if ctrl.shape != (self.num_envs, self.num_dof):
            raise ValueError(
                f"ctrl slot has shape {ctrl.shape}; expected {(self.num_envs, self.num_dof)}"
            )
        native_target = np.zeros_like(ctrl)
        native_target[:, self.native_joint_for_contract] = ctrl
        target = _to_tensor(self.torch, native_target, self.device)
        self.robot.set_joint_position_target(target)
        nsteps = int(payload["nsteps"])
        if nsteps <= 0:
            raise ValueError(f"nsteps must be positive, got {nsteps}")
        t0 = time.perf_counter()
        for _ in range(nsteps):
            self.robot.write_data_to_sim()
            self.sim.step(render=False)
            self.robot.update(self.sim_dt)
        physics_ms = (time.perf_counter() - t0) * 1000.0
        t0 = time.perf_counter()
        self.refresh_state_slots()
        refresh_ms = (time.perf_counter() - t0) * 1000.0
        return {
            "timing": {
                "control_upload_ms": 0.0,
                "physics_ms": physics_ms,
                "state_refresh_ms": refresh_ms,
            }
        }

    def set_state(self, payload: dict[str, Any]) -> dict[str, Any]:
        count = int(payload["count"])
        if count < 0 or count > self.num_envs:
            raise ValueError(f"reset count must be in [0, {self.num_envs}], got {count}")
        env_ids_np = np.asarray(self.slots["reset_env_ids"][:count], dtype=np.int64)
        qpos = np.asarray(self.slots["reset_qpos"][:count], dtype=np.float32)
        qvel = np.asarray(self.slots["reset_qvel"][:count], dtype=np.float32)
        if np.unique(env_ids_np).size != env_ids_np.size:
            raise ValueError("reset environment ids must not contain duplicates")
        if np.any(env_ids_np < 0) or np.any(env_ids_np >= self.num_envs):
            raise ValueError("reset environment ids are out of range")
        expected_qpos = (count, 7 + self.num_dof)
        expected_qvel = (count, 6 + self.num_dof)
        if qpos.shape != expected_qpos:
            raise ValueError(f"reset qpos has shape {qpos.shape}; expected {expected_qpos}")
        if qvel.shape != expected_qvel:
            raise ValueError(f"reset qvel has shape {qvel.shape}; expected {expected_qvel}")
        env_ids = self.torch.as_tensor(env_ids_np, dtype=self.torch.long, device=self.device)
        root_pose_np = qpos[:, :7].copy()
        root_pose_np[:, :3] += self.env_origins[env_ids_np]
        root_pose = _to_tensor(self.torch, root_pose_np, self.device)
        root_velocity_np = np.empty((count, 6), dtype=np.float32)
        root_velocity_np[:, :3] = qvel[:, :3]
        root_velocity_np[:, 3:] = _quat_rotate_wxyz(qpos[:, 3:7], qvel[:, 3:6])
        native_pos = np.zeros((count, self.num_dof), dtype=np.float32)
        native_vel = np.zeros_like(native_pos)
        native_pos[:, self.native_joint_for_contract] = qpos[:, 7 : 7 + self.num_dof]
        native_vel[:, self.native_joint_for_contract] = qvel[:, 6 : 6 + self.num_dof]
        self.robot.write_root_pose_to_sim(root_pose, env_ids=env_ids)
        self.robot.write_root_link_velocity_to_sim(
            _to_tensor(self.torch, root_velocity_np, self.device), env_ids=env_ids
        )
        self.robot.write_joint_state_to_sim(
            _to_tensor(self.torch, native_pos, self.device),
            _to_tensor(self.torch, native_vel, self.device),
            env_ids=env_ids,
        )
        self.robot.reset(env_ids)
        self.robot.update(self.sim_dt)
        t0 = time.perf_counter()
        self.refresh_state_slots()
        return {
            "timing": {
                "set_state_reset_upload_ms": 0.0,
                "set_state_host_cache_refresh_ms": (time.perf_counter() - t0) * 1000.0,
            }
        }

    def get_meta(self) -> dict[str, Any]:
        return {
            "num_dof": self.num_dof,
            "num_bodies": self.num_bodies,
            "dof_names": list(self.contract_joint_names),
            "body_names": list(self.contract_body_names),
            "gravity": [0.0, 0.0, -9.81],
            "use_gpu_pipeline": True,
            "graphics_enabled": False,
            "env_origins": self.env_origins.tolist(),
            "collision_filtering_applied": self.collision_filtering_applied,
        }

    def shutdown(self) -> None:
        for handle in self._shm_handles:
            try:
                handle.close()
            except Exception:
                pass
        self._shm_handles = []
        if self.simulation_app is not None:
            try:
                self.simulation_app.close()
            except Exception:
                pass
            self.simulation_app = None


def _dispatch(ctx: _WorkerContext, protocol: Any, cmd: str, payload: Any) -> tuple[str, Any]:
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
    raise NotImplementedError(
        f"isaacsim worker command {cmd!r} is unsupported; this backend is headless physics only"
    )


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", required=True)
    args = parser.parse_args(argv)
    protocol = _load_protocol(args.protocol)
    ctx = _WorkerContext(protocol)

    # Kit and extension startup can write banners to fd 1.  Preserve a private
    # protocol fd and route all incidental output to stderr before INIT.
    protocol_out = os.fdopen(os.dup(1), "wb")
    os.dup2(2, 1)
    stdin = sys.stdin.buffer
    stdout = protocol_out
    while True:
        try:
            message = protocol.recv_message(stdin)
        except (EOFError, protocol.WorkerDisconnectedError):
            ctx.shutdown()
            return 0
        cmd = message["cmd"]
        if cmd == protocol.CMD_SHUTDOWN:
            try:
                ctx.shutdown()
            finally:
                protocol.send_message(stdout, protocol.CMD_READY)
            return 0
        try:
            reply_cmd, reply_payload = _dispatch(ctx, protocol, cmd, message.get("payload"))
        except Exception as exc:  # noqa: BLE001 - every worker error crosses the wire
            protocol.send_message(stdout, protocol.CMD_ERROR, protocol.serialize_exception(exc))
            continue
        protocol.send_message(stdout, reply_cmd, reply_payload)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
