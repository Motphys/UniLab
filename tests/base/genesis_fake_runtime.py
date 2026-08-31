"""Fake Genesis runtime for host-free tests of the ``genesis`` backend.

The fake mirrors the exact API surface the adapter consumes (verified against
genesis-world 1.3.3 on a real RTX 4090) and implements just enough physics to
exercise contract semantics: PD position control held across substeps,
env-subset state writes, and link-addressed state reads.  It deliberately
uses real torch tensors at the boundary so the adapter's host-cache path
(pinned ``copy_`` D2H) runs unmodified.

Model spec (must stay in sync with ``TINY_MODEL_XML`` in the test file):
bodies world(0)/pelvis(1, free)/thigh(2, joint_a)/shank(3, joint_b); nq=9,
nv=8, nu=2.  links_pos: world at origin, pelvis at qpos root xyz, children at
fixed pelvis-frame offsets; links_quat: children inherit the pelvis quat
(joint rotation is intentionally not modeled — tests never rely on it).
"""

from __future__ import annotations

import types

import numpy as np
import torch

from unilab.utils.rotation import np_quat_apply_batched

# Mirrors TINY_MODEL_XML.  INIT_QPOS equals mj.qpos0 of that document
# (pelvis body pos z=0.8) so materialize-time state matches the MJCF scan.
INIT_QPOS = np.array([0.0, 0.0, 0.8, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
NQ = 9
NV = 8
N_LINKS = 4
BODY_NAMES = ("world", "pelvis", "thigh", "shank")
LINK_MASS = (0.0, 5.0, 2.0, 1.0)
# Fixed child-link offsets in the pelvis frame (fake kinematics simplification).
LINK_OFFSET = ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, -0.3), (0.0, 0.0, -0.6))
ACTUATED_DOFS = (6, 7)
INIT_KP = (10.0, 20.0)
INIT_KV = (2.0, 3.0)
GEOM_TABLE = (
    # (name, link_idx, friction, contype, conaffinity)
    ("floor", 0, 1.0, 1, 1),
    ("pelvis_geom", 1, 0.9, 1, 1),
    ("thigh_geom", 2, 0.8, 1, 1),
    ("foot_geom", 3, 0.6, 1, 1),
)
SIM_DT = 0.005
PD_INERTIA = 0.1


class _FakeJoint:
    def __init__(self, name, dofs_idx_local, qs_idx_local, link):
        self.name = name
        self.dofs_idx_local = list(dofs_idx_local)
        self.qs_idx_local = list(qs_idx_local)
        self.n_dofs = len(self.dofs_idx_local)
        self.n_qs = len(self.qs_idx_local)
        self.link = link


class _FakeLink:
    def __init__(self, name, idx):
        self.name = name
        self.idx = idx
        self.idx_local = idx
        self.joints: list[_FakeJoint] = []


class _FakeGeom:
    def __init__(self, idx, link, friction, contype, conaffinity):
        self.idx = idx
        self.link = link
        self.friction = friction
        self.contype = contype
        self.conaffinity = conaffinity


class _FakeIMU:
    def __init__(self, entity_idx, link_idx_local, pos_offset):
        self.entity_idx = entity_idx
        self.link_idx_local = link_idx_local
        self.pos_offset = pos_offset
        self.lin_acc: torch.Tensor | None = None
        self.ang_vel: torch.Tensor | None = None

    def read(self):
        return types.SimpleNamespace(lin_acc=self.lin_acc, ang_vel=self.ang_vel)


class _FakeEntity:
    def __init__(self, morph):
        self.morph = morph
        self.idx = 0
        self.link_start = 0
        self.n_dofs = NV
        self.n_qs = NQ
        self.n_links = N_LINKS
        self.links = [_FakeLink(name, idx) for idx, name in enumerate(BODY_NAMES)]
        root = _FakeJoint("floating_base_joint", range(6), range(7), self.links[1])
        joint_a = _FakeJoint("joint_a", (6,), (7,), self.links[2])
        joint_b = _FakeJoint("joint_b", (7,), (8,), self.links[3])
        self.links[1].joints.append(root)
        self.links[2].joints.append(joint_a)
        self.links[3].joints.append(joint_b)
        self.joints = [root, joint_a, joint_b]
        self.geoms = [
            _FakeGeom(idx, self.links[link_idx], friction, contype, conaffinity)
            for idx, (_, link_idx, friction, contype, conaffinity) in enumerate(GEOM_TABLE)
        ]
        self.built_envs = 0
        self.control_calls: list[np.ndarray] = []
        self.step_count = 0

    def get_link(self, name):
        for link in self.links:
            if link.name == name:
                return link
        raise KeyError(f"Link not found for name: {name}")

    def _build(self, n_envs: int) -> None:
        self.built_envs = n_envs
        self.qpos = torch.from_numpy(np.broadcast_to(INIT_QPOS, (n_envs, NQ)).copy())
        self.qvel = torch.zeros(n_envs, NV)
        self.ctrl_target = torch.zeros(n_envs, len(ACTUATED_DOFS))
        kp = torch.zeros(n_envs, NV)
        kv = torch.zeros(n_envs, NV)
        for dof, gain, kd in zip(ACTUATED_DOFS, INIT_KP, INIT_KV, strict=True):
            kp[:, dof] = gain
            kv[:, dof] = kd
        self.dof_kp = kp
        self.dof_kv = kv
        self.link_mass = torch.from_numpy(
            np.broadcast_to(np.asarray(LINK_MASS, dtype=np.float32), (n_envs, N_LINKS)).copy()
        )
        self.net_contact_force = torch.zeros(n_envs, N_LINKS, 3)

    # -- getters (torch tensors, like genesis) -------------------------- #
    def get_qpos(self, qs_idx_local=None, envs_idx=None):
        return self.qpos

    def get_dofs_velocity(self, dofs_idx_local=None, envs_idx=None):
        return self.qvel

    def get_dofs_kp(self, dofs_idx_local=None, envs_idx=None):
        return self.dof_kp

    def get_dofs_kv(self, dofs_idx_local=None, envs_idx=None):
        return self.dof_kv

    def _link_pose(self):
        quat = self.qpos[:, 3:7].cpu().numpy()
        pos = np.zeros((self.built_envs, N_LINKS, 3), dtype=np.float32)
        quat_out = np.zeros((self.built_envs, N_LINKS, 4), dtype=np.float32)
        quat_out[:, :, 0] = 1.0
        for link_idx in range(1, N_LINKS):
            offset = np.broadcast_to(
                np.asarray(LINK_OFFSET[link_idx], dtype=np.float32), (self.built_envs, 3)
            )
            pos[:, link_idx, :] = self.qpos[:, 0:3].cpu().numpy() + np_quat_apply_batched(
                quat, offset
            )
            quat_out[:, link_idx, :] = quat
        return pos, quat_out

    def get_links_pos(self, links_idx_local=None, envs_idx=None, **kwargs):
        pos, _ = self._link_pose()
        return torch.from_numpy(pos)

    def get_links_quat(self, links_idx_local=None, envs_idx=None, **kwargs):
        _, quat = self._link_pose()
        return torch.from_numpy(quat)

    def get_links_vel(self, links_idx_local=None, envs_idx=None, **kwargs):
        vel = np.zeros((self.built_envs, N_LINKS, 3), dtype=np.float32)
        vel[:, 1:, :] = self.qvel[:, 0:3].cpu().numpy()[:, None, :]
        return torch.from_numpy(vel)

    def get_links_ang(self, links_idx_local=None, envs_idx=None):
        quat = self.qpos[:, 3:7].cpu().numpy()
        ang = np.zeros((self.built_envs, N_LINKS, 3), dtype=np.float32)
        ang[:, 1:, :] = np_quat_apply_batched(quat, self.qvel[:, 3:6].cpu().numpy())[:, None, :]
        return torch.from_numpy(ang)

    def get_links_net_contact_force(self, envs_idx=None):
        return self.net_contact_force

    # -- setters --------------------------------------------------------- #
    @staticmethod
    def _rows(envs_idx):
        return None if envs_idx is None else list(envs_idx)

    def set_qpos(
        self, qpos, qs_idx_local=None, envs_idx=None, *, zero_velocity=True, skip_forward=False
    ):
        rows = self._rows(envs_idx)
        value = torch.as_tensor(qpos, dtype=torch.float32).cpu()
        if rows is None:
            self.qpos = value.clone()
        else:
            self.qpos[rows] = value
        if zero_velocity:
            if rows is None:
                self.qvel.zero_()
            else:
                self.qvel[rows] = 0.0

    def set_dofs_velocity(
        self, velocity=None, dofs_idx_local=None, envs_idx=None, *, skip_forward=False
    ):
        rows = self._rows(envs_idx)
        value = torch.as_tensor(velocity, dtype=torch.float32).cpu()
        if rows is None:
            self.qvel = value.clone()
        else:
            self.qvel[rows] = value

    def control_dofs_position(self, position, dofs_idx_local=None, envs_idx=None):
        value = torch.as_tensor(position, dtype=torch.float32).cpu()
        self.control_calls.append(value.numpy().copy())
        self.ctrl_target = value.clone()

    def _set_dof_rows(self, table, value, dofs_idx_local, envs_idx):
        rows = self._rows(envs_idx)
        dofs = list(dofs_idx_local) if dofs_idx_local is not None else list(range(NV))
        tensor = torch.as_tensor(value, dtype=torch.float32).cpu()
        if rows is None:
            table[:, dofs] = tensor
        else:
            table[np.ix_(rows, dofs)] = tensor

    def set_dofs_kp(self, kp, dofs_idx_local=None, envs_idx=None):
        self._set_dof_rows(self.dof_kp, kp, dofs_idx_local, envs_idx)

    def set_dofs_kv(self, kv, dofs_idx_local=None, envs_idx=None):
        self._set_dof_rows(self.dof_kv, kv, dofs_idx_local, envs_idx)

    def set_links_inertial_mass(self, inertial_mass, links_idx_local=None, envs_idx=None):
        rows = self._rows(envs_idx)
        links = list(links_idx_local) if links_idx_local is not None else list(range(N_LINKS))
        tensor = torch.as_tensor(inertial_mass, dtype=torch.float32).cpu()
        if rows is None:
            self.link_mass[:, links] = tensor
        else:
            self.link_mass[np.ix_(rows, links)] = tensor

    # -- physics --------------------------------------------------------- #
    def step(self):
        q = self.qpos.numpy()
        dq = self.qvel.numpy()
        tau = self.dof_kp.numpy()[:, ACTUATED_DOFS] * (self.ctrl_target.numpy() - q[:, 7:9])
        tau -= self.dof_kv.numpy()[:, ACTUATED_DOFS] * dq[:, 6:8]
        dq[:, 6:8] += tau * SIM_DT / PD_INERTIA
        dq[:, 2] -= 9.81 * SIM_DT
        q[:, 7:9] += dq[:, 6:8] * SIM_DT
        q[:, 0:3] += dq[:, 0:3] * SIM_DT
        self.qpos = torch.from_numpy(q.astype(np.float32))
        self.qvel = torch.from_numpy(dq.astype(np.float32))
        self.step_count += 1


class _FakeRigidSolver:
    def __init__(self):
        self.external_forces: list[tuple[np.ndarray, list[int]]] = []

    def apply_links_external_force(self, force, links_idx=None, envs_idx=None, **kwargs):
        value = torch.as_tensor(force, dtype=torch.float32).cpu().numpy().copy()
        self.external_forces.append((value, list(links_idx)))


class _FakeScene:
    def __init__(self, sim_options=None, rigid_options=None, show_viewer=None, **kwargs):
        self.sim_options = sim_options
        self.rigid_options = rigid_options
        self.show_viewer = show_viewer
        self.entities: list[_FakeEntity] = []
        self.sensors: list[_FakeIMU] = []
        self.sim = types.SimpleNamespace(rigid_solver=_FakeRigidSolver())

    def add_entity(self, morph):
        entity = _FakeEntity(morph)
        entity.idx = len(self.entities)
        self.entities.append(entity)
        return entity

    def add_sensor(self, sensor):
        self.sensors.append(sensor)
        return sensor

    def build(self, n_envs=0, **kwargs):
        for entity in self.entities:
            entity._build(n_envs)
        for sensor in self.sensors:
            sensor.lin_acc = torch.zeros(n_envs, 3)
            sensor.ang_vel = torch.zeros(n_envs, 3)

    def step(self):
        for entity in self.entities:
            entity.step()


def make_fake_genesis() -> types.SimpleNamespace:
    """Create a fresh fake ``genesis`` module with lifecycle counters."""
    fake = types.SimpleNamespace()
    fake.gpu = "fake-gpu"
    fake.cpu = "fake-cpu"
    fake.init_count = 0
    fake.destroy_count = 0

    def init(backend=None, logging_level=None, **kwargs):
        fake.init_count += 1

    def destroy():
        fake.destroy_count += 1

    options = types.SimpleNamespace(
        SimOptions=lambda **kw: types.SimpleNamespace(**kw),
        RigidOptions=lambda **kw: types.SimpleNamespace(kwargs=kw),
    )
    fake.init = init
    fake.destroy = destroy
    fake.options = options
    fake.morphs = types.SimpleNamespace(MJCF=lambda **kw: types.SimpleNamespace(**kw))
    fake.sensors = types.SimpleNamespace(IMU=_FakeIMU)
    fake.Scene = _FakeScene
    fake.integrator = types.SimpleNamespace(
        Euler="Euler",
        implicitfast="implicitfast",
        approximate_implicitfast="approximate_implicitfast",
    )
    fake.constraint_solver = types.SimpleNamespace(Newton="Newton", CG="CG")
    fake.friction_cone = types.SimpleNamespace(pyramidal="pyramidal", elliptic="elliptic")
    return fake
