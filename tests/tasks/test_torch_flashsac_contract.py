from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from unisim.backend.base import (
    TensorDataPlane,
    TensorExecution,
    TensorLifecycleCapabilities,
    TensorProcessTopology,
)

from unilab.base import registry
from unilab.base.config_adapter import BackendAdapter
from unilab.base.config_materialization import apply_cfg_overrides
from unilab.base.torch_env import TorchEnvState
from unilab.base.variants import FixedModelVariantCatalogCfg, FixedModelVariantCfg
from unilab.envs import ManagerBasedRlEnvCfg
from unilab.tasks.motion_tracking.g1 import torch_flashsac_env as module

ROOT_DIR = Path(__file__).parents[2]
CONF_DIR = ROOT_DIR / "src" / "unilab" / "conf" / "flashsac"


def _materialize_task(task: str, *, algo: str = "flashsac") -> ManagerBasedRlEnvCfg:
    GlobalHydra.instance().clear()
    config_dir = CONF_DIR if algo == "flashsac" else ROOT_DIR / "src/unilab/conf/sac"
    with initialize_config_dir(config_dir=str(config_dir), version_base="1.3"):
        composed = compose("config", overrides=[f"task={task}"])
    override = BackendAdapter(
        composed, root_dir=ROOT_DIR, algo_name=algo
    ).build_task_env_cfg_override()
    cfg = ManagerBasedRlEnvCfg()
    apply_cfg_overrides(cfg, override)
    return cfg


def test_torch_owner_fingerprint_accepts_both_canonical_backends() -> None:
    mujoco = _materialize_task("g1_motion_tracking/mujoco")
    mjwarp = _materialize_task("g1_motion_tracking/mjwarp")
    expected = module._TORCH_G1_FLASHSAC_OWNER_IDENTITY_V2
    assert module._torch_g1_flashsac_owner_identity(mujoco) == expected
    assert module._torch_g1_flashsac_owner_identity(mjwarp) == expected


def test_flashsac_motrix_owner_uses_task_host_bridge_runtime() -> None:
    """The FlashSAC Motrix owner must bypass CUDA-generic NumPy observations."""
    registry.ensure_registries()
    factory = registry._envs["G1MotionTrackingSAC"].env_factory_dict["motrix"]

    assert factory is module.make_torch_g1_motion_tracking_flashsac_env


def test_reusable_tensor_runtime_accepts_second_g1_manager_owner() -> None:
    cfg = _materialize_task("g1_motion_tracking/mjwarp_tensor", algo="sac")
    assert cfg.tensor_runtime is True
    assert module._torch_g1_flashsac_owner_identity(cfg) == module._TORCH_G1_SAC_OWNER_IDENTITY_V1


def test_reusable_tensor_runtime_accepts_second_manager_based_task() -> None:
    cfg = _materialize_task("g1_flip_tracking/mjwarp_tensor", algo="sac")
    assert cfg.tensor_runtime is True
    assert cfg.commands["motion"].sampling_mode == "mixed"
    assert module._torch_g1_flashsac_owner_identity(cfg) == (
        module._TORCH_G1_FLIP_SAC_OWNER_IDENTITY_V1
    )


@pytest.mark.parametrize(
    "mutate",
    [
        lambda cfg: setattr(cfg.observations["actor"].terms["base_lin_vel"].noise, "n_min", -0.2),
        lambda cfg: cfg.rewards["motion_global_root_pos"].params.__setitem__("std", 0.4),
        lambda cfg: setattr(cfg.actions["joint_pos"], "clip", {"joint": (0.0, 1.0)}),
        lambda cfg: setattr(cfg.commands["motion"].params, "adaptive_alpha", 0.1),
    ],
)
def test_torch_owner_fingerprint_fails_closed(mutate, monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = _materialize_task("g1_motion_tracking/mujoco")
    mutate(cfg)

    def fail(_name, *_args, **_kwargs):
        raise AssertionError("backend must not be created for a mutated owner")

    monkeypatch.setattr(module, "create_backend", fail)
    with pytest.raises(ValueError, match="canonical owner contract"):
        module.make_torch_g1_motion_tracking_flashsac_env(cfg, num_envs=2, backend_type="mujoco")


def test_torch_terminations_do_not_inject_reward_contact_policy() -> None:
    """Only declared termination terms may terminate a G1 episode."""
    env = module.TorchG1MotionTrackingFlashSACEnv.__new__(module.TorchG1MotionTrackingFlashSACEnv)
    env._anchor_idx = 0
    env._ee_ids = torch.tensor([1], dtype=torch.int64)
    env._terminations = {
        "anchor_pos": {"threshold": 0.5},
        "anchor_ori": {"threshold": 0.8},
        "ee_body_pos": {"threshold": 0.5},
        # This reward term has no termination counterpart and must not be
        # consumed as an undeclared termination policy.
        "undesired_contacts": {"threshold": 0.05},
    }
    env._motion_body_pos = torch.tensor([[[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]]])
    env._robot_body_pos = torch.tensor([[[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]]])
    identity_quat = torch.tensor([[[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]]])
    env._motion_body_quat = identity_quat
    env._robot_body_quat = identity_quat
    env._body_pos_relative = env._motion_body_pos.clone()

    assert not env._compute_terminations().any()


def test_torch_owner_rejects_fixed_model_variants_before_backend_creation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = _materialize_task("g1_motion_tracking/mujoco")
    cfg.fixed_model_variants = FixedModelVariantCatalogCfg(
        variants=(FixedModelVariantCfg(name="variant", source_model_file="variant.xml"),)
    )

    def fail(_name, *_args, **_kwargs):
        raise AssertionError("fixed variants must fail before backend creation")

    monkeypatch.setattr(module, "create_backend", fail)
    with pytest.raises(ValueError, match="does not support fixed model variants"):
        module.make_torch_g1_motion_tracking_flashsac_env(cfg, num_envs=2, backend_type="mujoco")


def test_torch_observation_noise_uses_configured_bounds_and_keeps_critic_clean() -> None:
    env = module.TorchG1MotionTrackingFlashSACEnv.__new__(module.TorchG1MotionTrackingFlashSACEnv)
    num_envs, num_joints, num_bodies = 2, 29, 2
    device = torch.device("cpu")
    env._torch = torch  # pyright: ignore[reportAttributeAccessIssue]
    env._device = device
    env._motion_joint_pos = torch.arange(num_envs * num_joints, dtype=torch.float32).reshape(
        num_envs, num_joints
    )
    env._motion_joint_vel = torch.full((num_envs, num_joints), 0.25)
    env._motion_anchor_pos_b = torch.full((num_envs, 3), 0.5)
    env._motion_anchor_ori_b = torch.full((num_envs, 6), -0.5)
    env._linvel = torch.full((num_envs, 3), 1.0)
    env._gyro = torch.full((num_envs, 3), -1.0)
    env._default_joint_pos = torch.full((num_envs, num_joints), 0.125)
    env._joint_default_bias = torch.full((num_envs, num_joints), -0.25)
    env._default_joint_vel = torch.full((num_envs, num_joints), 0.5)
    env._joint_pos = torch.full((num_envs, num_joints), 0.75)
    env._joint_vel = torch.full((num_envs, num_joints), -0.75)
    env._raw_actions = torch.linspace(-1, 1, num_envs * num_joints).reshape(num_envs, num_joints)
    env._robot_body_pos_b = torch.full((num_envs, num_bodies, 3), 2.0)
    env._robot_body_ori_b = torch.full((num_envs, num_bodies, 6), -2.0)
    env._actor_corruption = True
    env._actor_joint_pos_biased = False
    env._critic_prefix_names = (
        "command",
        "motion_anchor_pos_b",
        "motion_anchor_ori_b",
        "base_lin_vel",
        "base_ang_vel",
        "joint_pos",
        "joint_vel",
        "actions",
    )
    env._observation_noise = module.TensorObservationNoise(
        torch.tensor([-0.1, -0.2, -0.01, -1.5] * 16, dtype=torch.float32),
        torch.tensor([0.1, 0.2, 0.01, 1.5] * 16, dtype=torch.float32),
    )
    env._rng = torch.Generator(device=device).manual_seed(7)

    result = env._compute_observations()

    actor = result["obs"]
    critic = result["critic"]
    cursor = num_joints * 2 + 3 + 6
    width = env._observation_noise.lower.numel()
    noise_generator = torch.Generator(device=device).manual_seed(7)
    noise = torch.rand((num_envs, width), generator=noise_generator, device=device)
    noise *= env._observation_noise.upper - env._observation_noise.lower
    noise += env._observation_noise.lower
    torch.testing.assert_close(
        actor[:, cursor : cursor + width], critic[:, cursor : cursor + width] + noise
    )
    torch.testing.assert_close(actor[:, :cursor], critic[:, :cursor])


def test_second_g1_owner_keeps_actor_encoder_bias_out_of_critic() -> None:
    env = module.TorchG1MotionTrackingFlashSACEnv.__new__(module.TorchG1MotionTrackingFlashSACEnv)
    env._torch = torch
    env._device = torch.device("cpu")
    env._actor_joint_pos_biased = True
    env._actor_corruption = False
    env._critic_prefix_names = (
        "command",
        "motion_anchor_pos_b",
        "motion_anchor_ori_b",
        "base_lin_vel",
        "base_ang_vel",
        "joint_vel",
        "actions",
        "joint_pos",
    )
    env._motion_joint_pos = torch.zeros((2, 29))
    env._motion_joint_vel = torch.zeros((2, 29))
    env._motion_anchor_pos_b = torch.zeros((2, 3))
    env._motion_anchor_ori_b = torch.zeros((2, 6))
    env._linvel = torch.zeros((2, 3))
    env._gyro = torch.zeros((2, 3))
    env._joint_pos = torch.linspace(0.25, 0.5, 2 * 29).reshape(2, 29)
    env._joint_vel = torch.linspace(-0.5, -0.25, 2 * 29).reshape(2, 29)
    env._default_joint_pos = torch.zeros((2, 29))
    env._default_joint_vel = torch.zeros((2, 29))
    env._joint_default_bias = torch.zeros((2, 29))
    env._encoder_bias = torch.full((2, 29), 0.125)
    env._raw_actions = torch.linspace(-1.0, 1.0, 2 * 29).reshape(2, 29)
    env._robot_body_pos_b = torch.zeros((2, 2, 3))
    env._robot_body_ori_b = torch.zeros((2, 2, 6))

    result = env._compute_observations(corrupt=False)
    joint_start = 58 + 3 + 6 + 3 + 3
    critic_joint_start = joint_start + 2 * 29
    torch.testing.assert_close(
        result["obs"][:, joint_start : joint_start + 29],
        result["critic"][:, critic_joint_start : critic_joint_start + 29] + 0.125,
    )


def test_manual_torch_reset_clears_only_selected_done_flags(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    env = module.TorchG1MotionTrackingFlashSACEnv.__new__(module.TorchG1MotionTrackingFlashSACEnv)
    env._num_envs = 3
    env._device = torch.device("cpu")
    env._state = TorchEnvState(
        obs={"obs": torch.zeros((3, 2))},
        reward=torch.zeros(3),
        terminated=torch.tensor([True, False, False]),
        truncated=torch.tensor([False, True, False]),
        info={},
    )
    monkeypatch.setattr(env, "_reset_rows", lambda rows, obs: obs)
    obs, info = env.reset(env_indices=torch.tensor([0, 2]))
    assert obs["obs"].shape == (2, 2)
    assert info == {"log": {}}
    assert not bool(env._state.terminated.any())
    assert torch.equal(env._state.truncated, torch.tensor([False, True, False]))


def test_torch_device_state_store_renegotiates_and_preserves_policy_sensor_boundary() -> None:
    class Backend:
        backend_type = "fake-device"

        def get_tensor_capabilities(self):
            return TensorLifecycleCapabilities(
                execution=TensorExecution.DEVICE_RESIDENT,
                state_views=True,
                state_fields=frozenset({"qpos", "qvel"}),
                sensor_views=True,
                stepping=True,
                selected_reset=True,
                process_topology=TensorProcessTopology.IN_PROCESS,
                data_plane=TensorDataPlane.DIRECT,
                stream_event_ownership="backend completes fake refresh; caller owns stream",
                torch_devices=("cpu",),
            )

        def get_state_views(self, names, device=None):
            assert names == ("qpos", "qvel")
            return {
                "qpos": torch.zeros((2, 9), device=device),
                "qvel": torch.zeros((2, 8), device=device),
            }

        def get_sensor_view(self, name, device=None):
            if name == "pelvis_local_linvel":
                return torch.ones((2, 3), device=device)
            if name == "torso_gyro":
                return torch.full((2, 3), -1.0, device=device)
            prefix = name.rsplit("_", maxsplit=1)[0]
            shape = (2, 4) if "quat" in prefix else (2, 3)
            return torch.full(shape, 2.0, device=device)

    env = module.TorchG1MotionTrackingFlashSACEnv.__new__(module.TorchG1MotionTrackingFlashSACEnv)
    env._device = torch.device("cpu")
    env._state_store = module.TensorDeviceStateStore(
        backend=Backend(),  # pyright: ignore[reportArgumentType]
        device=env.device,
        num_envs=2,
        joint_qpos_ids=np.array([7, 8], dtype=np.int64),
        joint_qvel_ids=np.array([6, 7], dtype=np.int64),
        body_names=("pelvis", "torso"),
        body_ids=np.array([0, 1], dtype=np.intp),
    )

    env._read_robot_state()

    torch.testing.assert_close(env._linvel, torch.ones((2, 3)))
    torch.testing.assert_close(env._gyro, torch.full((2, 3), -1.0))
    torch.testing.assert_close(env._robot_body_pos, torch.full((2, 2, 3), 2.0))


def test_torch_owner_close_drops_backend_view_aliases_before_backend_cleanup() -> None:
    """Owner teardown must relinquish CUDA IPC views before backend close."""
    backend = SimpleNamespace(cleanup_scene_assets=lambda: None)
    env = module.TorchG1MotionTrackingFlashSACEnv.__new__(module.TorchG1MotionTrackingFlashSACEnv)
    env._backend = backend
    env._state_store = SimpleNamespace()
    env._qpos = torch.zeros(1)
    env._qvel = torch.zeros(1)
    env._joint_pos = torch.zeros(1)
    env._joint_vel = torch.zeros(1)
    env._state = TorchEnvState(
        obs={},
        reward=torch.zeros(1),
        terminated=torch.zeros(1, dtype=torch.bool),
        truncated=torch.zeros(1, dtype=torch.bool),
        info={},
    )

    def fail_after_owner_aliases_are_dropped() -> None:
        assert env._state_store is None
        assert env._qpos is None
        assert env._qvel is None
        assert env._joint_pos is None
        assert env._joint_vel is None
        assert env._state is None

    backend.cleanup_scene_assets = fail_after_owner_aliases_are_dropped

    env.close()
    env.close()


@pytest.mark.parametrize(
    ("backend_name", "execution", "packed"),
    [
        ("fake-device", TensorExecution.DEVICE_RESIDENT, False),
        ("fake-host", TensorExecution.HOST_BRIDGE, True),
    ],
)
def test_torch_backend_validation_is_capability_driven(
    backend_name: str, execution: TensorExecution, packed: bool
) -> None:
    class Backend:
        backend_type = backend_name

        def tensor_execution(self):
            return execution

        def get_tensor_capabilities(self):
            return TensorLifecycleCapabilities(
                execution=execution,
                state_views=True,
                state_fields=frozenset({"qpos", "qvel"}),
                sensor_views=True,
                stepping=True,
                selected_reset=True,
                packed_host_bridge=packed,
                process_topology=TensorProcessTopology.IN_PROCESS,
                data_plane=(
                    TensorDataPlane.DIRECT
                    if execution is TensorExecution.DEVICE_RESIDENT
                    else TensorDataPlane.HOST_BRIDGE
                ),
                stream_event_ownership="backend completes fake lifecycle; caller owns stream",
                torch_devices=("cpu",),
            )

        def get_state_views(self, names, device=None):
            assert names == ("qpos", "qvel")
            return {
                "qpos": torch.zeros((2, 9), device=device),
                "qvel": torch.zeros((2, 8), device=device),
            }

        def get_sensor_view(self, name, device=None):
            if name == "pelvis_local_linvel":
                return torch.ones((2, 3), device=device)
            if name == "torso_gyro":
                return torch.full((2, 3), -1.0, device=device)
            prefix = name.rsplit("_", maxsplit=1)[0]
            shape = (2, 4) if "quat" in prefix else (2, 3)
            return torch.full(shape, 2.0, device=device)

        def get_body_ids(self, names):
            return np.arange(len(names), dtype=np.intp)

    env = module.TorchG1MotionTrackingFlashSACEnv.__new__(module.TorchG1MotionTrackingFlashSACEnv)
    env._cfg = _materialize_task("g1_motion_tracking/mujoco")
    env._device = torch.device("cpu")
    env._backend = Backend()

    env._validate_backend()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
@pytest.mark.parametrize("exact_device", [False, True])
def test_g1_backend_validation_accepts_cuda_family_and_current_exact_device(exact_device) -> None:
    class Backend:
        backend_type = "fake-device"

        def tensor_execution(self):
            return TensorExecution.DEVICE_RESIDENT

        def get_tensor_capabilities(self):
            torch_devices = ("cuda",)
            if exact_device:
                torch_devices = (f"cuda:{torch.cuda.current_device()}",)
            return TensorLifecycleCapabilities(
                execution=TensorExecution.DEVICE_RESIDENT,
                state_views=True,
                state_fields=frozenset({"qpos", "qvel"}),
                sensor_views=True,
                stepping=True,
                selected_reset=True,
                process_topology=TensorProcessTopology.IN_PROCESS,
                data_plane=TensorDataPlane.DIRECT,
                stream_event_ownership="backend completes fake lifecycle",
                torch_devices=torch_devices,
            )

        def get_state_views(self, names, device=None):
            assert names == ("qpos", "qvel")
            return {
                "qpos": torch.zeros((2, 9), device=device),
                "qvel": torch.zeros((2, 8), device=device),
            }

        def get_sensor_view(self, name, device=None):
            shape = (2, 4) if "quat" in name else (2, 3)
            return torch.ones(shape, device=device)

        def get_body_ids(self, names):
            return np.arange(len(names), dtype=np.intp)

    env = module.TorchG1MotionTrackingFlashSACEnv.__new__(module.TorchG1MotionTrackingFlashSACEnv)
    env._cfg = _materialize_task("g1_motion_tracking/mujoco")
    env._device = torch.device("cuda", index=torch.cuda.current_device())
    env._backend = Backend()

    env._validate_backend()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_g1_backend_validation_rejects_wrong_exact_cuda_device() -> None:
    class Backend:
        backend_type = "fake-device"

        def tensor_execution(self):
            return TensorExecution.DEVICE_RESIDENT

        def get_tensor_capabilities(self):
            wrong_index = (torch.cuda.current_device() + 1) % max(2, torch.cuda.device_count() + 1)
            return TensorLifecycleCapabilities(
                execution=TensorExecution.DEVICE_RESIDENT,
                process_topology=TensorProcessTopology.IN_PROCESS,
                data_plane=TensorDataPlane.DIRECT,
                stream_event_ownership="backend completes fake lifecycle",
                torch_devices=(f"cuda:{wrong_index}",),
            )

    env = module.TorchG1MotionTrackingFlashSACEnv.__new__(module.TorchG1MotionTrackingFlashSACEnv)
    env._device = torch.device("cuda", index=torch.cuda.current_device())
    env._backend = Backend()

    with pytest.raises(RuntimeError, match="did not accept Torch device"):
        env._validate_backend()


def test_torch_factory_delegates_noncanonical_backends_to_capability_validation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = _materialize_task("g1_motion_tracking/mujoco")
    created: dict[str, object] = {}

    class Backend:
        backend_type = "fake-device"

    def fake_create_backend(*args, **kwargs):
        created["args"] = args
        created["kwargs"] = kwargs
        backend = Backend()
        created["backend"] = backend
        return backend

    sentinel = object()
    monkeypatch.setattr(module, "create_backend", fake_create_backend)
    monkeypatch.setattr(
        module,
        "TorchG1MotionTrackingFlashSACEnv",
        lambda cfg, backend, num_envs, device: (sentinel, backend),
    )

    result = module.make_torch_g1_motion_tracking_flashsac_env(
        cfg,
        num_envs=2,
        backend_type="fake-device",
    )

    assert result == (sentinel, created["backend"])
    assert created["args"][0] == "fake-device"


def test_device_resident_cold_contract_proxy_skips_generic_tensor_reads() -> None:
    """The proxy must not ask CUDA-only backends for a synthetic CPU plane."""
    proxy = module._DeviceResidentColdContractProxy.__new__(module._DeviceResidentColdContractProxy)
    proxy.scene = SimpleNamespace(_tensor_read_plan=object())

    proxy._compile_tensor_read_plan()

    assert proxy.scene._tensor_read_plan is None
