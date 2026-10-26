"""CUDA device-placement contracts for the Manager Torch runtime (issue #2279)."""

from __future__ import annotations

from collections import Counter
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest
import torch
from unisim.backend.base import (
    RootStateLayout,
    SimBackend,
    TensorDataPlane,
    TensorExecution,
    TensorLifecycleCapabilities,
    TensorProcessTopology,
    TrackedBodyStateViews,
)

from unilab.base.entity import EntityCfg, EntityScene, SceneTensorReadSpec
from unilab.managers import (
    ObservationGroupCfg,
    ObservationManager,
    ObservationTermCfg,
    RewardManager,
    RewardTermCfg,
    TerminationManager,
    TerminationTermCfg,
)
from unilab.managers._noise import GaussianNoiseCfg, NoiseModelWithAdditiveBiasCfg
from unilab.tasks.locomotion.common import manager_terms

from .conftest import FakeEnv

requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")


def _cuda() -> torch.device:
    return torch.device("cuda", index=torch.cuda.current_device())


@requires_cuda
def test_alive_reward_term_allocates_on_env_device() -> None:
    env = cast(Any, SimpleNamespace(num_envs=4, device=_cuda()))
    result = manager_terms.alive(env)

    assert result.dtype == torch.float32
    assert result.shape == (env.num_envs,)
    assert result.device == env.device
    assert bool(result.all())


@requires_cuda
def test_observation_scale_table_lives_on_env_device() -> None:
    env = FakeEnv(device="cuda")
    manager = ObservationManager(
        {
            "policy": ObservationGroupCfg(
                terms={
                    "obs": ObservationTermCfg(func=lambda env: env.obs, scale=(2.0, 0.5)),
                }
            )
        },
        env,
    )

    assert manager._scale_tensors
    for scale_tensor in manager._scale_tensors.values():
        assert scale_tensor.device == env.device
        assert scale_tensor.dtype == torch.float32

    expected = env.obs * torch.tensor((2.0, 0.5), dtype=torch.float32, device=env.device)
    computed = manager.compute()["policy"]
    assert isinstance(computed, torch.Tensor)
    assert computed.device == env.device
    torch.testing.assert_close(computed, expected)
    group = manager.compute_group("policy")
    assert isinstance(group, torch.Tensor)
    assert group.device == env.device
    torch.testing.assert_close(group, expected)


@requires_cuda
def test_observation_row_scoped_indexing_and_reset_on_cuda() -> None:
    env = FakeEnv(device="cuda")
    manager = ObservationManager(
        {
            "policy": ObservationGroupCfg(
                terms={
                    "obs": ObservationTermCfg(func=lambda env: env.obs, scale=(2.0, 0.5)),
                }
            )
        },
        env,
    )
    manager.compute()

    terms = manager.get_active_iterable_terms(1)
    assert terms[0][0] == "policy-obs"
    assert terms[0][1] == pytest.approx([4.0, 1.5])

    assert manager.reset(torch.tensor([0, 2], dtype=torch.int64, device=env.device)) == {}
    recomputed = manager.compute()["policy"]
    assert isinstance(recomputed, torch.Tensor)
    assert recomputed.device == env.device


@requires_cuda
def test_reward_manager_terms_and_buffers_on_cuda() -> None:
    env = FakeEnv(device="cuda")

    def ones(env: FakeEnv) -> torch.Tensor:
        return torch.ones(env.num_envs, dtype=torch.float32, device=env.device)

    manager = RewardManager(
        {
            "pos": RewardTermCfg(func=ones, weight=2.0),
            "neg": RewardTermCfg(func=ones, weight=-0.5),
        },
        env,
    )
    reward = manager.compute(dt=0.1)

    assert reward.dtype == torch.float32
    assert reward.shape == (env.num_envs,)
    assert reward.device == env.device
    torch.testing.assert_close(reward, torch.full((env.num_envs,), 0.15, device=env.device))
    assert manager._reward_buf.dtype == torch.float32
    assert manager._reward_buf.shape == (env.num_envs,)
    assert manager._reward_buf.device == env.device
    means = manager.step_reward_means
    assert means.dtype == torch.float32
    assert means.device == env.device
    assert means.shape == (2,)
    torch.testing.assert_close(means, torch.tensor([2.0, -0.5], device=env.device))


@requires_cuda
def test_termination_manager_terms_on_cuda() -> None:
    env = FakeEnv(device="cuda")
    flags = torch.tensor([True, False, False, True], dtype=torch.bool, device=env.device)
    manager = TerminationManager(
        {"failure": TerminationTermCfg(func=lambda env: flags.clone())},
        env,
    )

    dones = manager.compute()

    assert dones.dtype == torch.bool
    assert dones.shape == (env.num_envs,)
    assert dones.device == env.device
    assert bool(dones.equal(flags))


@requires_cuda
@pytest.mark.parametrize(("env_device", "term_device"), (("cuda", "cpu"), ("cpu", "cuda")))
def test_observation_manager_rejects_wrong_device_term(env_device: str, term_device: str) -> None:
    env = FakeEnv(device=env_device)
    cfg = {
        "policy": ObservationGroupCfg(
            terms={
                "bad": ObservationTermCfg(
                    func=lambda env: torch.ones(
                        (env.num_envs, 2), dtype=torch.float32, device=term_device
                    )
                ),
            }
        )
    }
    with pytest.raises(TypeError, match="must return float32 observations on"):
        ObservationManager(cfg, env)


@requires_cuda
@pytest.mark.parametrize(("env_device", "term_device"), (("cuda", "cpu"), ("cpu", "cuda")))
def test_reward_manager_rejects_wrong_device_term(env_device: str, term_device: str) -> None:
    env = FakeEnv(device=env_device)
    manager = RewardManager(
        {
            "bad": RewardTermCfg(
                func=lambda env: torch.ones(env.num_envs, dtype=torch.float32, device=term_device),
                weight=1.0,
            )
        },
        env,
    )
    with pytest.raises(ValueError, match="returned device .* expected"):
        manager.compute(dt=0.1)


@requires_cuda
@pytest.mark.parametrize(("env_device", "term_device"), (("cuda", "cpu"), ("cpu", "cuda")))
def test_termination_manager_rejects_wrong_device_term(env_device: str, term_device: str) -> None:
    env = FakeEnv(device=env_device)
    manager = TerminationManager(
        {
            "bad": TerminationTermCfg(
                func=lambda env: torch.zeros(env.num_envs, dtype=torch.bool, device=term_device)
            )
        },
        env,
    )
    with pytest.raises(ValueError, match="returned device .* expected"):
        manager.compute()


@requires_cuda
def test_observation_noise_model_bias_buffer_on_cuda() -> None:
    env = FakeEnv(device="cuda")
    manager = ObservationManager(
        {
            "policy": ObservationGroupCfg(
                terms={
                    "state": ObservationTermCfg(
                        func=lambda env: env.obs,
                        noise=NoiseModelWithAdditiveBiasCfg(
                            noise_cfg=GaussianNoiseCfg(mean=0.0, std=0.1),
                            bias_noise_cfg=GaussianNoiseCfg(mean=0.0, std=0.1),
                        ),
                    ),
                },
                enable_corruption=True,
            )
        },
        env,
    )

    corrupted = manager.compute()["policy"]
    model = manager._group_obs_class_instances["policy"]["state"]
    assert model._bias.device == env.device
    assert model._bias.dtype == torch.float32
    assert isinstance(corrupted, torch.Tensor)
    assert corrupted.device == env.device
    assert corrupted.dtype == torch.float32


class _CudaBackend:
    backend_type = "fake"
    num_envs = 2
    num_actuators = 0

    def __init__(self) -> None:
        self.sensor_calls: Counter[str] = Counter()
        self.sensor_values = {
            "gyro": np.asarray([[0.1, 0.2, 0.3], [-0.1, -0.2, -0.3]], dtype=np.float32),
            "upvector": np.asarray([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0]], dtype=np.float32),
        }
        self.sensor_values.update(
            {
                "track_pos_w_base": torch.from_numpy(np.zeros((2, 3), dtype=np.float32)),
                "track_quat_w_base": torch.from_numpy(
                    np.asarray(
                        [
                            [1.0, 0.0, 0.0, 0.0],
                            [np.sqrt(0.5), np.sqrt(0.5), 0.0, 0.0],
                        ],
                        dtype=np.float32,
                    )
                ),
                "track_linvel_w_base": torch.from_numpy(
                    np.asarray([[0.5, 0.25, 0.0], [-0.5, -0.25, 0.1]], dtype=np.float32)
                ),
                "track_angvel_w_base": torch.from_numpy(
                    np.asarray([[0.1, 0.2, 0.3], [-0.1, -0.2, -0.3]], dtype=np.float32)
                ),
            }
        )
        self.joint_names = ("hip", "knee", "ankle")
        self.dof_pos = np.asarray(
            [[0.4, 0.1, -0.2], [0.0, 0.3, 0.7]],
            dtype=np.float32,
        )
        self.dof_vel = np.asarray(
            [[1.0, 2.0, 3.0], [-1.0, -2.0, -3.0]],
            dtype=np.float32,
        )
        self.body_pos = np.zeros((2, 1, 3), dtype=np.float32)
        self.body_quat = np.zeros((2, 1, 4), dtype=np.float32)
        self.body_quat[0, 0, 0] = 1.0
        half = np.sqrt(0.5)
        self.body_quat[1, 0] = [half, half, 0.0, 0.0]
        self.body_lin_vel_w = np.asarray(
            [[[0.5, 0.25, 0.0]], [[-0.5, -0.25, 0.1]]], dtype=np.float32
        )
        self.body_ang_vel_w = np.asarray(
            [[[0.1, 0.2, 0.3]], [[-0.1, -0.2, -0.3]]], dtype=np.float32
        )

    def get_tensor_capabilities(self):
        return TensorLifecycleCapabilities(
            execution=TensorExecution.DEVICE_RESIDENT,
            state_views=True,
            state_fields=frozenset(("qpos", "qvel")),
            sensor_views=True,
            process_topology=TensorProcessTopology.IN_PROCESS,
            data_plane=TensorDataPlane.DIRECT,
            stream_event_ownership="caller owns the Torch stream",
            torch_devices=("cuda",),
        )

    def get_state_views(self, names, device=None):
        return {
            "qpos": torch.as_tensor(self.dof_pos, dtype=torch.float32, device=device),
            "qvel": torch.as_tensor(self.dof_vel, dtype=torch.float32, device=device),
        }

    def get_sensor_view(self, name, device=None):
        return torch.as_tensor(self.sensor_values[name], dtype=torch.float32, device=device)

    def get_tracked_body_views(self, body_names, device=None):
        names = tuple(name.rsplit("/", 1)[-1] for name in body_names)
        if names != ("base",):
            raise KeyError(names)
        return TrackedBodyStateViews(
            body_names=names,
            pos_w=torch.as_tensor(self.body_pos, dtype=torch.float32, device=device),
            quat_w=torch.as_tensor(self.body_quat, dtype=torch.float32, device=device),
            lin_vel_w=torch.as_tensor(self.body_lin_vel_w, dtype=torch.float32, device=device),
            ang_vel_w=torch.as_tensor(self.body_ang_vel_w, dtype=torch.float32, device=device),
        )

    def get_joint_dof_pos_indices(self, names) -> np.ndarray:
        return np.asarray([self.joint_names.index(name) for name in names], dtype=np.int32)

    def get_joint_dof_vel_indices(self, names) -> np.ndarray:
        return self.get_joint_dof_pos_indices(names)

    def get_joint_state_qpos_indices(self, names) -> np.ndarray:
        return self.get_joint_dof_pos_indices(names)

    def get_joint_state_qvel_indices(self, names) -> np.ndarray:
        return self.get_joint_dof_pos_indices(names)

    def get_body_ids(self, names) -> np.ndarray:
        if tuple(names) != ("base",):
            raise KeyError(names)
        return np.asarray([0], dtype=np.int32)

    def get_dof_pos(self) -> np.ndarray:
        return self.dof_pos

    def get_dof_vel(self) -> np.ndarray:
        return self.dof_vel

    def get_default_dof_pos(self) -> np.ndarray:
        return np.asarray([0.1, 0.2, 0.3], dtype=np.float32)

    def get_joint_range(self) -> np.ndarray:
        return np.tile(np.asarray([[-1.0, 1.0]], dtype=np.float32), (3, 1))

    def get_root_state_layout(self, root_body_name: str) -> RootStateLayout:
        if root_body_name != "base":
            raise KeyError(root_body_name)
        return RootStateLayout(tuple(range(7)), tuple(range(6)))

    def get_default_qpos(self) -> np.ndarray:
        return np.asarray([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0], dtype=np.float32)

    def get_init_qvel(self) -> np.ndarray:
        return np.zeros(6, dtype=np.float32)

    def get_body_pos_w(self, ids: np.ndarray) -> np.ndarray:
        return self.body_pos[:, ids]

    def get_body_quat_w(self, ids: np.ndarray) -> np.ndarray:
        return self.body_quat[:, ids]

    def get_body_lin_vel_w(self, ids: np.ndarray) -> np.ndarray:
        return self.body_lin_vel_w[:, ids]

    def get_body_ang_vel_w(self, ids: np.ndarray) -> np.ndarray:
        return self.body_ang_vel_w[:, ids]


@requires_cuda
def test_cuda_packed_read_plan_and_default_torch_tables() -> None:
    device = _cuda()
    backend = _CudaBackend()
    scene = EntityScene(
        {
            "robot": EntityCfg(
                root_body_name="base",
                joint_names=backend.joint_names,
                body_names=("base",),
            )
        },
        cast(SimBackend, backend),
    )
    plan = scene.compile_tensor_reads(
        device,
        (
            SceneTensorReadSpec(entity="robot", body_names=("base",)),
            SceneTensorReadSpec(entity="robot", sensor_names=("gyro", "upvector")),
        ),
    )
    plan.refresh()

    assert plan.device == device

    joints = plan.joint_tensor_view("robot")
    for view in (joints.joint_pos, joints.joint_vel):
        assert view.dtype == torch.float32
        assert view.device == device

    sensors = plan.sensor_tensor_views("robot", ("gyro", "upvector"))
    for view in sensors.values.values():
        assert view.dtype == torch.float32
        assert view.device == device

    body = plan.body_tensor_view("robot")
    for view in (body.pos_w, body.quat_w, body.lin_vel_w, body.ang_vel_w):
        assert view.dtype == torch.float32
        assert view.device == device

    data = scene["robot"].data
    for table in (
        data.default_joint_pos(device),
        data.default_joint_vel(device),
        data.default_root_state(device),
        data.gravity_vec_w(device),
        data.soft_joint_pos_limits(device),
    ):
        assert table.dtype == torch.float32
        assert table.device == device
