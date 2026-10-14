"""Focused contract tests for the NumPy Manager-Based environment lifecycle."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest
import torch
from unisim.backend.base import (
    DebugPrimitive,
    HostBridgeTransferPlan,
    SelectedResetPublication,
    SimBackend,
    TensorDataPlane,
    TensorExecution,
    TensorIOSpec,
    TensorLifecycleCapabilities,
    TensorProcessTopology,
    TensorRuntimeDiagnostic,
    TrackedBodyStateViews,
)
from unisim.dr.types import DomainRandomizationCapabilities

import unilab.envs.manager_based_rl_env as manager_env_module
from unilab.assets import ASSETS_ROOT_PATH
from unilab.base.backend_factory import create_backend, env_backend_kwargs
from unilab.base.backend_timing import UPDATE_STATE_DETAIL_TIMING_KEYS
from unilab.base.entity import EntityCfg
from unilab.base.env_update_graph import EnvUpdateGraphMode, validate_env_update_graph_manifest
from unilab.base.scene import SceneCfg
from unilab.base.torch_env import TorchEnv, TorchEnvState
from unilab.envs import (
    ManagerBasedRLEnv,
    ManagerBasedRlEnv,
    ManagerBasedRLEnvCfg,
    ManagerBasedRlEnvCfg,
    make_manager_based_rl_env,
    mdp,
)
from unilab.managers import (
    ActionTerm,
    ActionTermCfg,
    CommandTerm,
    CommandTermCfg,
    CurriculumTermCfg,
    EventTermCfg,
    MetricsTermCfg,
    NullCommandManager,
    ObservationGroupCfg,
    ObservationTermCfg,
    RecorderTerm,
    RecorderTermCfg,
    ResetOwner,
    ResetOwnerCfg,
    RewardTermCfg,
    TerminationTermCfg,
)
from unilab.managers.scene_entity_config import SceneEntityCfg
from unilab.managers.torch_rng import TorchManagerRng


class _FakeBackend:
    backend_type = "fake"

    def __init__(
        self,
        num_envs: int,
        *,
        reject_pre_step: bool = False,
        reject_materialize: bool = False,
        packed_reset: bool = True,
        torch_stream_id: int | None = None,
    ) -> None:
        self.num_envs = num_envs
        self.num_actuators = 1
        self.reject_pre_step = reject_pre_step
        self.reject_materialize = reject_materialize
        self.packed_reset = packed_reset
        self.torch_stream_id = torch_stream_id
        self.pre_step_control = None
        self.applied_controls: list[np.ndarray] = []
        self.tensor_controls: list[torch.Tensor] = []
        self.step_nsteps: list[int] = []
        self.cleanup_calls = 0
        self.materialize_calls = 0
        self.lifecycle: list[str] = []
        self.sensors: dict[str, torch.Tensor] = {}
        self.set_state_calls: list[tuple[Any, Any, Any]] = []
        self.graph_completions: list[Any] = []
        self.fail_next_step_tensor = False
        self.graph_invalidations: list[str] = []
        self.graph_fallback_count = 0
        self.qpos = torch.zeros((num_envs, 4), dtype=torch.float32)
        self.qvel = torch.zeros((num_envs, 3), dtype=torch.float32)

    def materialize(self) -> None:
        if self.reject_materialize:
            raise RuntimeError("pool construction failed")
        self.materialize_calls += 1
        self.lifecycle.append("materialize")

    def get_actuator_names(self) -> tuple[str, ...]:
        return ("motor",)

    def get_geom_names(self) -> tuple[str, ...]:
        return ("base_geom",)

    def get_actuator_gains(self) -> tuple[np.ndarray, np.ndarray]:
        return np.asarray([20.0]), np.asarray([2.0])

    def get_body_ids(self, names) -> np.ndarray:
        available = ("base",)
        return np.asarray([available.index(name) for name in names], dtype=np.int32)

    def get_sensor_view(self, name: str, device=None) -> torch.Tensor:
        target = torch.device(device) if device is not None else torch.device("cpu")
        prefixes = ("track_pos_w_", "track_quat_w_", "track_linvel_w_", "track_angvel_w_")
        if name.startswith(prefixes):
            prefix = next(item for item in prefixes if name.startswith(item))
            body_name = name[len(prefix) :]
            field = {
                "track_pos_w_": "pos_w",
                "track_quat_w_": "quat_w",
                "track_linvel_w_": "lin_vel_w",
                "track_angvel_w_": "ang_vel_w",
            }[prefix]
            views = self.get_tracked_body_views((body_name,), device=target)
            return getattr(views, field)[:, 0]
        raise KeyError(name)

    def get_dr_capabilities(self):
        return DomainRandomizationCapabilities(
            supported_reset_terms=frozenset(("body_mass", "body_ipos", "geom_friction", "kp", "kd"))
        )

    def get_default_qpos(self) -> np.ndarray:
        return np.array([0.0, 0.0, 0.5, 0.0], dtype=np.float64)

    def get_init_qvel(self) -> np.ndarray:
        return np.array([0.0, 0.0, 0.0], dtype=np.float32)

    def get_reset_term_default(self, term: str) -> np.ndarray:
        defaults = {
            "body_mass": np.asarray([10.0]),
            "body_ipos": np.asarray([[0.0, 0.0, 0.0]]),
            "geom_friction": np.asarray([[0.5, 0.02, 0.002]]),
            "kp": np.asarray([20.0]),
            "kd": np.asarray([2.0]),
        }
        if term not in defaults:
            raise NotImplementedError(term)
        return np.array(defaults[term], copy=True)

    def get_tracked_body_views(self, body_names, device=None) -> TrackedBodyStateViews:
        names = tuple(body_names)
        count = self.num_envs
        target = torch.device(device) if device is not None else torch.device("cpu")
        quat = torch.zeros((count, len(names), 4), dtype=torch.float32, device=target)
        quat[..., 0] = 1.0
        return TrackedBodyStateViews(
            body_names=names,
            pos_w=torch.zeros((count, len(names), 3), dtype=torch.float32, device=target),
            quat_w=quat,
            lin_vel_w=torch.zeros((count, len(names), 3), dtype=torch.float32, device=target),
            ang_vel_w=torch.zeros((count, len(names), 3), dtype=torch.float32, device=target),
        )

    def get_actuator_ctrl_range(self) -> np.ndarray:
        return np.array([[-2.0, 2.0]], dtype=np.float32)

    def set_pre_step_control(self, fn) -> None:
        if fn is not None and self.reject_pre_step:
            raise NotImplementedError("host callback disabled")
        self.pre_step_control = fn

    def tensor_execution(self) -> TensorExecution:
        return TensorExecution.DEVICE_RESIDENT if self.packed_reset else TensorExecution.HOST_BRIDGE

    def get_tensor_capabilities(self) -> TensorLifecycleCapabilities:
        return TensorLifecycleCapabilities(
            execution=self.tensor_execution(),
            state_views=self.packed_reset,
            state_fields=frozenset({"qpos", "qvel"}),
            stepping=True,
            selected_reset=True,
            packed_host_bridge=not self.packed_reset,
            packed_reset=self.packed_reset,
            device_reset_randomization=self.packed_reset,
            packed_reset_fields=(
                frozenset(("body_mass", "body_ipos", "geom_friction", "kp", "kd"))
                if self.packed_reset
                else frozenset()
            ),
            selected_reset_publication=(
                SelectedResetPublication.AUTHORITATIVE_VIEWS if self.packed_reset else None
            ),
            data_plane=(
                TensorDataPlane.DIRECT if self.packed_reset else TensorDataPlane.HOST_BRIDGE
            ),
            deferred_fused_validation=self.packed_reset,
            stream_event_ownership="test",
            torch_devices=("cpu",),
            reset_randomization=True,
        )

    def begin_packed_reset(self, *, device=None):
        from unisim.backend.packed_reset import TorchPackedResetComposer

        return TorchPackedResetComposer(self, device=device)

    def get_tensor_runtime_diagnostics(self):
        return {
            "deferred_fused_validation": TensorRuntimeDiagnostic(
                requested=True,
                enabled=True,
                disable_reason=None,
            )
        }

    def publish_torch_fused_validation(self) -> None:
        return None

    def get_graph_lifecycle(self):
        from unisim import GraphLifecycleState

        if getattr(self, "lifecycle_state", None) is not None:
            return self.lifecycle_state
        return GraphLifecycleState.READY

    def get_graph_lifecycle_snapshot(self):
        from unisim import GraphLifecycleSnapshot

        return GraphLifecycleSnapshot(
            contract_version=1,
            state=self.get_graph_lifecycle(),
            capture_id="test-capture",
            replay_count=len(self.graph_completions) or 2,
            fallback_count=self.graph_fallback_count,
        )

    def invalidate_graph(self, reason):
        from unisim import GraphLifecycleSnapshot, GraphLifecycleState

        if reason not in {
            "model",
            "data",
            "view",
            "capacity",
            "policy_io",
            "stream_event",
            "capture_failed",
        }:
            raise ValueError(f"unsupported graph invalidation reason {reason!r}")
        self.graph_invalidations.append(reason)
        self.graph_fallback_count += 1
        self.lifecycle_state = GraphLifecycleState.INVALIDATED
        self.graph_disable_reason = f"graph invalidated: {reason}"
        return GraphLifecycleSnapshot(
            contract_version=1,
            state=self.lifecycle_state,
            capture_id=None,
            replay_count=len(self.graph_completions),
            fallback_count=self.graph_fallback_count,
            disable_reason=self.graph_disable_reason,
        )

    def preflight_graph_candidate(self, request):
        from dataclasses import replace

        if request.requested:
            return replace(
                request,
                backend_stream_contract="test",
                supported=True,
                disable_reason=None,
                torch_stream_id=self.torch_stream_id,
            )
        return replace(
            request,
            backend_stream_contract="test",
            supported=False,
            disable_reason="graph candidate was not requested",
        )

    def preflight_env_update_graph(self, request):
        from dataclasses import replace

        if getattr(self, "reject_env_update_preflight", False):
            return replace(
                request,
                backend_stream_contract="test",
                supported=False,
                disable_reason="test backend rejects update capture",
            )
        return replace(
            request,
            backend_stream_contract="test",
            supported=True,
            disable_reason=None,
            torch_stream_id=self.torch_stream_id,
            allocator_stats={"allocation.all.allocated": 3},
        )

    def set_state_tensor(self, env_indices, qpos, qvel, randomization=None) -> None:
        self.set_state_calls.append(
            (
                env_indices.detach().clone(),
                qpos.detach().clone(),
                qvel.detach().clone(),
            )
        )

    def set_state_tensor_mask(self, reset_mask, qpos, qvel, randomization=None, **_kwargs):
        rows = torch.nonzero(reset_mask, as_tuple=False).flatten()
        self.set_state_tensor(
            rows, qpos.index_select(0, rows), qvel.index_select(0, rows), randomization
        )
        if randomization is not None:
            full_width = {
                name: value for name, value in vars(randomization).items() if value is not None
            }
            self.randomization_calls[-1] = type(randomization)(
                **{
                    name: (
                        torch.where(
                            reset_mask.reshape((-1,) + (1,) * (value.ndim - 1)),
                            value,
                            value,
                        ).index_select(0, rows)
                    )
                    for name, value in full_width.items()
                }
            )
        return {"timing": {}, "graph_completion": self.record_graph_completion("mask_reset")}

    def get_public_state_widths(self):
        from unisim.backend.base import PublicStateWidths

        return PublicStateWidths(nq=4, nv=3)

    def get_state_views(self, fields, device=None) -> dict[str, torch.Tensor]:
        target = torch.device(device) if device is not None else torch.device("cpu")
        return {
            "qpos": self.qpos.to(device=target),
            "qvel": self.qvel.to(device=target),
        }

    def get_sensor_data(self, name: str) -> np.ndarray:
        try:
            return self.sensors[name].numpy()
        except KeyError as exc:
            raise KeyError(
                f"Unknown test sensor(s) ({name!r}); "
                f"available sensors: {', '.join(sorted(self.sensors))}"
            ) from exc

    def step_tensor(self, ctrl: torch.Tensor, nsteps: int = 1) -> None:
        if self.fail_next_step_tensor:
            self.fail_next_step_tensor = False
            raise RuntimeError("simulated graph replay failed before publication")
        self.tensor_controls.append(ctrl.detach().clone())
        self.step(ctrl.detach().cpu().numpy(), nsteps)
        if self.graph_completions and self.graph_completions[-1] == "not-a-token":
            return {"timing": {}, "graph_completion": "not-a-token"}
        return {"timing": {}, "graph_completion": self.record_graph_completion("step")}

    def _graph_completion(self, operation: str):
        from unisim import GraphCompletionToken, GraphLifecycleState

        return GraphCompletionToken(
            contract_version=1,
            lifecycle_state=GraphLifecycleState.READY,
            operation=operation,
            sequence=len(self.set_state_calls) + len(self.tensor_controls),
            stream_id=self.torch_stream_id or 0,
            capture_id="test-capture",
        )

    def record_graph_completion(self, operation: str):
        completion = self._graph_completion(operation)
        self.graph_completions.append(completion)
        return completion

    def step(self, ctrl: np.ndarray, nsteps: int = 1) -> None:
        assert self.materialize_calls == 1
        self.step_nsteps.append(nsteps)
        native = ctrl
        for _ in range(nsteps):
            if self.pre_step_control is not None:
                native = self.pre_step_control(self, ctrl)
            self.applied_controls.append(native.copy())

    def cleanup_scene_assets(self) -> None:
        self.cleanup_calls += 1


class _ResetBackend(_FakeBackend):
    def __init__(self, num_envs: int) -> None:
        super().__init__(num_envs)
        self.default_qpos_calls = 0
        self.init_qvel_calls = 0
        self.joint_layout_calls = 0
        self.set_state_calls: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
        self.randomization_calls: list[Any] = []
        self.qpos = torch.zeros((num_envs, 4), dtype=torch.float32)
        self.qvel = torch.zeros((num_envs, 3), dtype=torch.float32)
        self.state_view_calls = 0

    def get_default_qpos(self) -> np.ndarray:
        self.default_qpos_calls += 1
        return np.array([0.0, 0.0, 0.5, 0.0], dtype=np.float64)

    def get_init_qvel(self) -> np.ndarray:
        self.init_qvel_calls += 1
        return np.array([0.0, 0.0, 0.0], dtype=np.float32)

    def get_actuator_joint_names(self) -> tuple[str, ...]:
        return ("joint",)

    def get_joint_dof_pos_indices(self, names) -> np.ndarray:
        assert tuple(names) == ("joint",)
        return np.array([0], dtype=np.int32)

    def get_joint_dof_vel_indices(self, names) -> np.ndarray:
        assert tuple(names) == ("joint",)
        return np.array([0], dtype=np.int32)

    def get_joint_state_qpos_indices(self, names) -> np.ndarray:
        assert tuple(names) == ("joint",)
        self.joint_layout_calls += 1
        return np.array([3], dtype=np.int32)

    def get_joint_state_qvel_indices(self, names) -> np.ndarray:
        assert tuple(names) == ("joint",)
        self.joint_layout_calls += 1
        return np.array([2], dtype=np.int32)

    def get_default_dof_pos(self) -> np.ndarray:
        return np.zeros((1,), dtype=np.float32)

    def get_state_views(self, fields, device=None) -> dict[str, torch.Tensor]:
        self.state_view_calls += 1
        target = torch.device(device) if device is not None else torch.device("cpu")
        views: dict[str, torch.Tensor] = {}
        if "qpos" in fields:
            views["qpos"] = self.qpos.detach().clone().to(target)
        if "qvel" in fields:
            views["qvel"] = self.qvel.detach().clone().to(target)
        return views

    def get_joint_range(self) -> np.ndarray:
        return np.array([[-1.0, 1.0]], dtype=np.float32)

    def set_state_tensor(self, env_indices, qpos, qvel, randomization=None) -> None:
        assert self.materialize_calls == 1
        rows = env_indices.detach().cpu().numpy().copy()
        host_qpos = qpos.detach().cpu().numpy().copy()
        host_qvel = qvel.detach().cpu().numpy().copy()
        self.set_state_calls.append((rows, host_qpos, host_qvel))
        self.randomization_calls.append(randomization)
        self.qpos[rows] = torch.as_tensor(host_qpos, device=self.qpos.device)
        self.qvel[rows] = torch.as_tensor(host_qvel, device=self.qvel.device)


class _StateBackend(_ResetBackend):
    def __init__(self, num_envs: int) -> None:
        super().__init__(num_envs)
        self.dof_pos = np.zeros((num_envs, 1), dtype=np.float32)
        self.dof_pos_calls = 0

    def get_state_views(self, fields, device=None) -> dict[str, torch.Tensor]:
        self.dof_pos_calls += 1
        return super().get_state_views(fields, device=device)

    def step(self, ctrl: np.ndarray, nsteps: int = 1) -> None:
        super().step(ctrl, nsteps)
        self.dof_pos += 1.0
        self.qpos[:, 3] = torch.from_numpy(self.dof_pos[:, 0].copy()).to(self.qpos.device)

    def set_state_tensor(self, env_indices, qpos, qvel, randomization=None) -> None:
        super().set_state_tensor(env_indices, qpos, qvel, randomization=randomization)
        rows = env_indices.detach().cpu().numpy()
        self.dof_pos[rows, 0] = qpos.detach().cpu().numpy()[:, 3]


class _KeyframeBackend(_ResetBackend):
    def __init__(
        self,
        num_envs: int,
        *,
        keyframe_qpos: Any = None,
        keyframe_error: Exception | None = None,
    ) -> None:
        super().__init__(num_envs)
        self.keyframe_qpos = (
            np.array([0.0, 0.0, 0.3, 0.75], dtype=np.float64)
            if keyframe_qpos is None
            else keyframe_qpos
        )
        self.keyframe_error = keyframe_error
        self.keyframe_qpos_calls = 0

    def get_keyframe_qpos(self, name: str) -> Any:
        self.keyframe_qpos_calls += 1
        if self.keyframe_error is not None:
            raise self.keyframe_error
        if name != "home":
            raise ValueError(f"unknown keyframe {name!r}")
        return self.keyframe_qpos


class _DeferredTensorBackend(_FakeBackend):
    """External worker whose capabilities only become known after materialize."""

    def __init__(
        self,
        num_envs: int,
        *,
        post_capabilities: TensorLifecycleCapabilities | None = None,
    ) -> None:
        super().__init__(num_envs)
        self.post_capabilities = post_capabilities

    def get_tensor_capabilities(self) -> TensorLifecycleCapabilities:
        if self.materialize_calls == 0:
            return TensorLifecycleCapabilities(execution=TensorExecution.UNSUPPORTED)
        return (
            self.post_capabilities
            if self.post_capabilities is not None
            else TensorLifecycleCapabilities(
                execution=TensorExecution.HOST_BRIDGE,
                state_fields=frozenset({"qpos", "qvel"}),
                stepping=True,
                selected_reset=True,
                packed_host_bridge=True,
                data_plane=TensorDataPlane.HOST_BRIDGE,
                stream_event_ownership="test",
                torch_devices=("cpu", "cuda"),
            )
        )

    def tensor_execution(self) -> TensorExecution:
        capabilities = self.get_tensor_capabilities()
        return (
            TensorExecution.HOST_BRIDGE
            if capabilities.execution is not TensorExecution.UNSUPPORTED
            else TensorExecution.UNSUPPORTED
        )


class _ScenePlanBackend(_StateBackend):
    """State backend with the public packed scene-read contract."""

    nq = 4
    nv = 3

    def __init__(self, num_envs: int) -> None:
        super().__init__(num_envs)
        self.full_reads = 0
        self.selected_reads = 0
        self.plan_closes = 0
        self.reset_calls: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
        self.reset_randomization_calls: list[Any] = []
        self.sensors = {
            "track_pos_w_platform": torch.zeros((num_envs, 3), dtype=torch.float32),
            "track_quat_w_platform": torch.tile(
                torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float32), (num_envs, 1)
            ),
            "track_linvel_w_platform": torch.zeros((num_envs, 3), dtype=torch.float32),
            "track_angvel_w_platform": torch.zeros((num_envs, 3), dtype=torch.float32),
            "track_pos_w_base": torch.zeros((num_envs, 3), dtype=torch.float32),
            "track_quat_w_base": torch.tile(
                torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float32), (num_envs, 1)
            ),
            "track_linvel_w_base": torch.zeros((num_envs, 3), dtype=torch.float32),
            "track_angvel_w_base": torch.zeros((num_envs, 3), dtype=torch.float32),
            "track_pos_w_ball": torch.tensor(
                [[0.1, 0.2, 1.2], [0.3, 0.4, 1.2]], dtype=torch.float32
            ),
            "track_quat_w_ball": torch.tile(
                torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float32), (num_envs, 1)
            ),
            "track_linvel_w_ball": torch.zeros((num_envs, 3), dtype=torch.float32),
            "track_angvel_w_ball": torch.zeros((num_envs, 3), dtype=torch.float32),
        }

    def get_tensor_capabilities(self) -> TensorLifecycleCapabilities:
        return TensorLifecycleCapabilities(
            execution=self.tensor_execution(),
            state_views=True,
            state_fields=frozenset({"qpos", "qvel"}),
            sensor_views=True,
            stepping=True,
            selected_reset=True,
            packed_host_bridge=not self.packed_reset,
            packed_reset=self.packed_reset,
            device_reset_randomization=self.packed_reset,
            packed_reset_fields=(
                frozenset(("body_mass", "body_ipos", "geom_friction", "kp", "kd"))
                if self.packed_reset
                else frozenset()
            ),
            selected_reset_publication=(
                SelectedResetPublication.AUTHORITATIVE_VIEWS if self.packed_reset else None
            ),
            data_plane=(
                TensorDataPlane.DIRECT if self.packed_reset else TensorDataPlane.HOST_BRIDGE
            ),
            stream_event_ownership="test",
            torch_devices=("cpu",),
            reset_randomization=True,
        )

    def get_state_views(self, fields, device=None) -> dict[str, torch.Tensor]:
        assert set(fields) <= {"qpos", "qvel"}
        target = torch.device(device) if device is not None else torch.device("cpu")
        return super().get_state_views(fields, device=target)

    def bind_sensor_data(self, names):
        from unisim.backend.base import BackendSensorView

        sensor_names = tuple(names)
        values = [self.sensors[name] for name in sensor_names]
        return BackendSensorView(
            self.backend_type,
            sensor_names,
            tuple(value.shape[1] for value in values),
            self.num_envs,
            lambda: torch.cat(
                [value.reshape(self.num_envs, -1) for value in values], dim=1
            ).numpy(),
        )

    def get_body_ids(self, names) -> np.ndarray:
        available = ("base", "platform", "ball")
        return np.asarray([available.index(name) for name in names], dtype=np.int32)

    def get_tracked_body_views(self, body_names, device=None) -> TrackedBodyStateViews:
        names = tuple(body_names)
        target = torch.device(device) if device is not None else torch.device("cpu")

        def field(prefix: str, width: int) -> torch.Tensor:
            return torch.cat(
                [self.sensors[prefix + name].reshape(self.num_envs, 1, width) for name in names],
                dim=1,
            ).to(target)

        return TrackedBodyStateViews(
            body_names=names,
            pos_w=field("track_pos_w_", 3),
            quat_w=field("track_quat_w_", 4),
            lin_vel_w=field("track_linvel_w_", 3),
            ang_vel_w=field("track_angvel_w_", 3),
        )

    def compile_host_bridge_io(self, spec: TensorIOSpec):
        backend = self

        class _Plan(HostBridgeTransferPlan):
            def __init__(self, plan_spec: TensorIOSpec) -> None:
                self._spec = plan_spec
                self.last_timing: dict[str, float] = {}

            @property
            def spec(self) -> TensorIOSpec:
                return self._spec

            @property
            def transfer_stats(self) -> dict[str, int]:
                return {
                    "state_sensor_h2d": backend.full_reads + backend.selected_reads,
                    "selected_post_reset_h2d": backend.selected_reads,
                }

            def _packet(self) -> dict[str, torch.Tensor]:
                packet: dict[str, torch.Tensor] = backend.get_state_views(("qpos", "qvel"))
                packet.update(backend.sensors)
                return packet

            def read_state_sensors(self):
                backend.full_reads += 1
                return self._packet()

            def read_selected_state_sensors(self):
                backend.selected_reads += 1
                return self._packet()

            def write_control(self, ctrl) -> None:
                return None

            def step(self, nsteps: int = 1):
                return None

            def apply_reset(self, env_indices, qpos, qvel, randomization=None):
                backend.reset_calls.append(
                    (
                        env_indices.detach().clone(),
                        qpos.detach().clone(),
                        qvel.detach().clone(),
                    )
                )
                backend.reset_randomization_calls.append(randomization)
                return None

            def close(self) -> None:
                backend.plan_closes += 1

        return _Plan(spec)


class _DeviceResidentScenePlanBackend(_ScenePlanBackend):
    """A CUDA-only device-resident scene plan with selected tensor reset."""

    nq = 4
    nv = 3

    def __init__(self, num_envs: int) -> None:
        super().__init__(num_envs)
        self.tensor_reset_calls: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
        self.tensor_steps = 0
        self.topology = TensorProcessTopology.IN_PROCESS
        self.data_plane = TensorDataPlane.DIRECT
        self.sensors = {name: value.cuda() for name, value in self.sensors.items()}
        self.qpos = self.qpos.cuda()
        self.qvel = self.qvel.cuda()

    def get_state_views(self, fields, device=None) -> dict[str, torch.Tensor]:
        if self.qpos.shape[1] != type(self).nq:
            self.qpos = torch.zeros((self.num_envs, type(self).nq), device=self.qpos.device)
        if self.qvel.shape[1] != type(self).nv:
            self.qvel = torch.zeros((self.num_envs, type(self).nv), device=self.qvel.device)
        target = torch.device(device) if device is not None else self.qpos.device
        return {"qpos": self.qpos.to(device=target), "qvel": self.qvel.to(device=target)}

    def get_public_state_widths(self):
        from unisim.backend.base import PublicStateWidths

        return PublicStateWidths(nq=type(self).nq, nv=type(self).nv)

    def get_sensor_view(self, name: str, device=None) -> torch.Tensor:
        return self.sensors[name].to(
            device=torch.device(device) if device else torch.device("cuda")
        )

    def get_tensor_capabilities(self) -> TensorLifecycleCapabilities:
        return TensorLifecycleCapabilities(
            execution=TensorExecution.DEVICE_RESIDENT,
            state_views=True,
            state_fields=frozenset({"qpos", "qvel"}),
            sensor_views=True,
            stepping=True,
            selected_reset=True,
            reset_randomization=True,
            packed_reset=True,
            device_reset_randomization=True,
            packed_reset_fields=frozenset(("body_mass", "body_ipos", "geom_friction", "kp", "kd")),
            selected_reset_publication=SelectedResetPublication.AUTHORITATIVE_VIEWS,
            process_topology=TensorProcessTopology.IN_PROCESS,
            data_plane=TensorDataPlane.DIRECT,
            stream_event_ownership="test",
            torch_devices=("cuda",),
        )

    def tensor_execution(self) -> TensorExecution:
        return TensorExecution.DEVICE_RESIDENT

    def compile_host_bridge_io(self, spec: TensorIOSpec):
        raise AssertionError("DEVICE_RESIDENT reset must not compile a host bridge plan")

    def set_state_tensor(self, env_indices, qpos, qvel, randomization=None) -> None:
        if self.qpos.shape[1] != qpos.shape[1]:
            self.qpos = torch.zeros(
                (self.num_envs, qpos.shape[1]), dtype=qpos.dtype, device=self.qpos.device
            )
        if self.qvel.shape[1] != qvel.shape[1]:
            self.qvel = torch.zeros(
                (self.num_envs, qvel.shape[1]), dtype=qvel.dtype, device=self.qvel.device
            )
        rows = env_indices
        self.tensor_reset_calls.append(
            (
                rows.detach().clone(),
                qpos.detach().clone(),
                qvel.detach().clone(),
            )
        )
        self.qpos[rows] = qpos
        self.qvel[rows] = qvel

    def step_tensor(self, ctrl: torch.Tensor, nsteps: int = 1) -> None:
        self.tensor_steps += 1


@dataclass(kw_only=True)
class _DriveCfg(ActionTermCfg):
    gain: float = 1.0

    def build(self, env) -> ActionTerm:
        return _DriveAction(self, env)


class _DriveAction(ActionTerm):
    def __init__(self, cfg: _DriveCfg, env) -> None:
        super().__init__(cfg, env)
        self._processed = torch.zeros((self.num_envs, 1), dtype=torch.float32, device=env.device)

    @property
    def action_dim(self) -> int:
        return 1

    @property
    def raw_action(self) -> torch.Tensor:
        return self._processed

    def process_actions(self, actions: torch.Tensor) -> None:
        self._env.action_input_types.append(type(actions))
        self._processed.copy_(actions * cast(_DriveCfg, self.cfg).gain)

    def reset(self, reset_mask: torch.Tensor) -> None:
        self._processed.copy_(
            torch.where(
                reset_mask.reshape(-1, 1),
                torch.zeros_like(self._processed),
                self._processed,
            )
        )

    def apply_actions(self) -> None:
        self._env.trace.append("action_apply")
        self._env.action_sim_steps.append(self._env._sim_step_counter)
        self._entity.data.write_ctrl(self._processed)


class _FeedbackDriveAction(_DriveAction):
    requires_substep_state_feedback = True
    uses_tensor_state_packet = True


@dataclass(kw_only=True)
class _FeedbackDriveCfg(_DriveCfg):
    def build(self, env) -> ActionTerm:
        return _FeedbackDriveAction(self, env)


@dataclass(kw_only=True)
class _TensorBodyActionCfg(ActionTermCfg):
    top_body_name: str
    ball_body_name: str

    def build(self, env) -> _TensorBodyAction:
        return _TensorBodyAction(self, env)


class _TensorBodyAction(ActionTerm):
    uses_tensor_actions = True
    uses_tensor_state_packet = True

    def __init__(self, cfg: _TensorBodyActionCfg, env) -> None:
        super().__init__(cfg, env)
        self._processed = torch.zeros((self.num_envs, 1), dtype=torch.float32, device=env.device)

    @property
    def action_dim(self) -> int:
        return 1

    @property
    def raw_action(self) -> torch.Tensor:
        return self._processed

    @property
    def tensor_body_names(self) -> tuple[str, ...]:
        cfg = cast(_TensorBodyActionCfg, self.cfg)
        return (cfg.top_body_name, cfg.ball_body_name)

    def process_actions(self, actions: torch.Tensor) -> None:
        self._processed.copy_(actions)

    def reset(self, reset_mask: torch.Tensor) -> None:
        self._processed.copy_(
            torch.where(
                reset_mask.reshape(-1, 1),
                torch.zeros_like(self._processed),
                self._processed,
            )
        )

    def apply_actions(self) -> None:
        state = self._env.scene._tensor_read_plan
        if state is None:
            self._processed.zero_()
        else:
            view = state.body_tensor_view(self._entity)
            self._processed.copy_(view.pos_w[:, 1, 2:3])
        self._entity.data.write_ctrl(self._processed)


@dataclass(kw_only=True)
class _CommandCfg(CommandTermCfg):
    def build(self, env) -> CommandTerm:
        return _Command(self, env)


class _Command(CommandTerm):
    body_names = ("platform", "ball")
    entity_name = "robot"

    def __init__(self, cfg: _CommandCfg, env) -> None:
        super().__init__(cfg, env)
        self._command = torch.zeros((self.num_envs, 1), dtype=torch.float32)
        self.reset_masks: list[torch.Tensor] = []

    @property
    def command(self) -> torch.Tensor:
        return self._command

    def _update_metrics(self) -> None:
        return None

    def _resample_command(self, reset_mask: torch.Tensor) -> None:
        rng = self._env.torch_rng
        assert rng is not None
        self.reset_masks.append(reset_mask.clone())
        samples = rng.uniform(0.0, 1.0, self._command.shape, dtype=torch.float32)
        self._command.copy_(torch.where(reset_mask[:, None], samples, self._command))

    def _update_command_reset(self, reset_mask: torch.Tensor) -> None:
        self.reset(reset_mask)

    def post_compute_reset(self) -> None:
        return None

    def _update_command(self) -> None:
        self._env.command_update_ids.append(None)


class _TensorBodyCommandCfg(_CommandCfg):
    def build(self, env) -> CommandTerm:
        return _TensorBodyCommand(self, env)


class _TensorBodyCommand(_Command):
    @property
    def tensor_body_names(self) -> tuple[str, ...]:
        return self.body_names


class _CommandBodyObservation:
    def __init__(self, cfg: ObservationTermCfg, env: _TestEnv) -> None:
        command = env.command_manager.get_term(cfg.params["command_name"])
        self.tensor_body_names = tuple(command.body_names)
        self.entity_name = command.entity_name

    def __call__(self, env: _TestEnv, **params: Any) -> torch.Tensor:
        del params
        plan = env.scene._tensor_read_plan
        if plan is None:
            return torch.zeros((env.num_envs, 1), dtype=torch.float32)
        view = plan.body_tensor_view(self.entity_name, self.tensor_body_names)
        return view.pos_w[:, 0, 0:1]


@dataclass(kw_only=True)
class _StateWritingCommandCfg(CommandTermCfg):
    def build(self, env) -> CommandTerm:
        return _StateWritingCommand(self, env)


class _StateWritingCommand(_Command):
    def reset(self, reset_mask: torch.Tensor) -> dict[str, float]:
        self._command.copy_(
            torch.where(
                reset_mask.reshape(-1, 1), torch.full_like(self._command, 0.75), self._command
            )
        )
        self._env.scene["robot"].write_joint_state_to_sim_mask(
            torch.full((self.num_envs, 1), 0.75, dtype=torch.float32),
            torch.full((self.num_envs, 1), -0.75, dtype=torch.float32),
            reset_mask=reset_mask,
        )
        return {}

    def _update_command_reset(self, reset_mask: torch.Tensor) -> None:
        self._command.copy_(
            torch.where(
                reset_mask.reshape(-1, 1),
                torch.full_like(self._command, 0.75),
                self._command,
            )
        )

    def _resample_command(self, reset_mask: torch.Tensor) -> None:
        self._env.scene["robot"].write_joint_state_to_sim_mask(
            torch.full((self.num_envs, 1), 0.75, dtype=torch.float32),
            torch.full((self.num_envs, 1), -0.75, dtype=torch.float32),
            reset_mask=reset_mask,
        )


class _TensorStateWritingCommandCfg(CommandTermCfg):
    def build(self, env) -> CommandTerm:
        return _TensorStateWritingCommand(self, env)


class _TensorStateWritingCommand(_Command):
    def reset(self, reset_mask: torch.Tensor) -> dict[str, float]:
        self._command.copy_(
            torch.where(
                reset_mask.reshape(-1, 1), torch.full_like(self._command, 0.75), self._command
            )
        )
        self._env.scene["robot"].write_joint_state_to_sim_mask(
            torch.full((self.num_envs, 1), 0.75, dtype=torch.float32, device=self._device),
            torch.full((self.num_envs, 1), -0.75, dtype=torch.float32, device=self._device),
            reset_mask=reset_mask,
        )
        return {}

    def _update_command_reset(self, reset_mask: torch.Tensor) -> None:
        self._command.copy_(
            torch.where(
                reset_mask.reshape(-1, 1),
                torch.full_like(self._command, 0.75),
                self._command,
            )
        )

    def __init__(self, cfg: _TensorStateWritingCommandCfg, env) -> None:
        super().__init__(cfg, env)
        self._command = torch.zeros((self.num_envs, 1), dtype=torch.float32, device=self._device)

    @property
    def command(self) -> torch.Tensor:
        if not isinstance(self._command, torch.Tensor):
            self._command = torch.zeros(
                (self.num_envs, 1), dtype=torch.float32, device=self._device
            )
        return self._command

    def _update_metrics(self, env_ids: torch.Tensor | None = None) -> None:
        return None

    def _resample_command(self, env_ids: torch.Tensor) -> None:
        self._command[env_ids, 0] = 0.75
        reset_mask = torch.zeros(self.num_envs, dtype=torch.bool, device=self._device)
        reset_mask[env_ids] = True
        self._env.scene["robot"].write_joint_state_to_sim_mask(
            torch.full((self.num_envs, 1), 0.75, dtype=torch.float32, device=self._device),
            torch.full((self.num_envs, 1), -0.75, dtype=torch.float32, device=self._device),
            reset_mask=reset_mask,
        )

    def _update_command(self, env_ids: torch.Tensor | None) -> None:
        return None


class _AliasedSensorCommandCfg(CommandTermCfg):
    def build(self, env) -> CommandTerm:
        return _AliasedSensorCommand(self, env)


class _AliasedSensorCommand(CommandTerm):
    tensor_sensor_names = ("pelvis_local_linvel", "torso_gyro")
    packed_sensor_aliases = {
        "pelvis_local_linvel": ("pelvis_local_linvel", "local_linvel"),
        "torso_gyro": ("torso_gyro", "gyro"),
    }

    def __init__(self, cfg: CommandTermCfg, env) -> None:
        super().__init__(cfg, env)
        self._command = torch.zeros((self.num_envs, 2), dtype=torch.float32)

    @property
    def command(self) -> torch.Tensor:
        return self._command

    def _update_metrics(self) -> None:
        return None

    def _resample_command(self, reset_mask: torch.Tensor) -> None:
        samples = torch.zeros_like(self._command)
        samples[:, 0] = 0.5
        samples[:, 1] = -0.25
        self._command.copy_(torch.where(reset_mask[:, None], samples, self._command))

    def _update_command(self) -> None:
        return None

    def _metric_velocities(self) -> tuple[torch.Tensor, torch.Tensor]:
        read_plan = self._env.scene._tensor_read_plan
        packed = read_plan.sensor_names.get("robot", ()) if read_plan else ()
        if len(packed) == 2:
            entity = self._env.scene["robot"]
            views = read_plan.sensor_tensor_views(entity, packed).values
            return views[packed[0]], views[packed[1]]
        return torch.zeros(self.num_envs, 3), torch.zeros(self.num_envs, 3)


class _Recorder(RecorderTerm):
    def record_pre_reset(self, reset_mask: torch.Tensor) -> None:
        self._env.trace.append(("pre_reset", reset_mask.detach().cpu().tolist()))

    def record_post_reset(self, reset_mask: torch.Tensor) -> None:
        self._env.trace.append(("post_reset", reset_mask.detach().cpu().tolist()))

    def record_post_step(self) -> None:
        self._env.trace.append("post_step")

    def close(self) -> None:
        self._env.trace.append("recorder_close")


class _TestEnv(ManagerBasedRlEnv):
    def __init__(self, cfg: ManagerBasedRlEnvCfg, backend: SimBackend, num_envs: int) -> None:
        self.trace: list[Any] = []
        self.command_update_ids: list[np.ndarray | None] = []
        self.action_sim_steps: list[int] = []
        self.action_input_types: list[type] = []
        super().__init__(cfg, backend, num_envs)


class _NamedSensorTensorObservation:
    tensor_sensor_names = ("track_pos_w_ball",)

    def __init__(self, cfg: ObservationTermCfg, env: _TestEnv) -> None:
        del cfg
        self._env = env

    def __call__(self, env: _TestEnv) -> torch.Tensor:
        assert env is self._env
        plan = env.scene._tensor_read_plan
        if plan is None:
            # Manager construction probes the term shape before the cold-path
            # read plan exists. The cold probe mirrors one packed sensor row;
            # all hot-path calls below require the compiled plan.
            backend = cast(_ScenePlanBackend, env._backend)
            return backend.sensors["track_pos_w_ball"].clone()
        entity = env.scene["robot"]
        return plan.sensor_tensor_views(entity, self.tensor_sensor_names).values[
            self.tensor_sensor_names[0]
        ]


def _policy_obs(env: _TestEnv) -> np.ndarray:
    return torch.stack(
        (
            env.episode_length_buf.to(torch.float32),
            env.action_manager.action[:, 0],
        ),
        dim=1,
    )


def _critic_obs(env: _TestEnv) -> torch.Tensor:
    return env.episode_length_buf[:, None].to(torch.float32)


def _tensor_runtime_policy_obs(env: _TestEnv) -> torch.Tensor:
    return torch.column_stack(
        (
            env.episode_length_buf.to(dtype=torch.float32, device=env.device),
            env.action_manager.action[:, 0],
        )
    )


def _tensor_runtime_critic_obs(env: _TestEnv) -> torch.Tensor:
    return env.episode_length_buf[:, None].to(dtype=torch.float32, device=env.device)


def _reset_joint_state_tensor(
    env: _TestEnv, env_ids: torch.Tensor, position: float, velocity: float
) -> None:
    count = env_ids.numel()
    env.scene["robot"].write_joint_state_tensor_to_sim(
        torch.full((count, 1), position, dtype=torch.float32, device=env.device),
        torch.full((count, 1), velocity, dtype=torch.float32, device=env.device),
        env_ids=env_ids,
    )


def _reset_joint_state_tensor_mask(
    env: _TestEnv, reset_mask: torch.Tensor, position: float, velocity: float
) -> None:
    env.scene["robot"].write_joint_state_to_sim_mask(
        torch.full((env.num_envs, 1), position, dtype=torch.float32, device=env.device),
        torch.full((env.num_envs, 1), velocity, dtype=torch.float32, device=env.device),
        reset_mask=reset_mask,
    )


setattr(_reset_joint_state_tensor_mask, "reset", _reset_joint_state_tensor_mask)


def _episode_step_observation(env: ManagerBasedRlEnv) -> torch.Tensor:
    return env.episode_length_buf[:, None].to(torch.float32)


def _reward(env: _TestEnv) -> np.ndarray:
    return env.action_manager.action[:, 0].clone()


def _joint_state_obs(env: _TestEnv) -> np.ndarray:
    return torch.as_tensor(env.scene["robot"].data._host_joint_pos, dtype=torch.float32)


def _joint_state_reward(env: _TestEnv) -> torch.Tensor:
    return torch.as_tensor(env.scene["robot"].data._host_joint_pos[:, 0], dtype=torch.float32)


def _joint_state_termination(env: _TestEnv) -> torch.Tensor:
    return torch.as_tensor(env.scene["robot"].data._host_joint_pos[:, 0]) > 100.0


def _mutate_joint_state_step_event(env: _TestEnv, env_ids: np.ndarray | None) -> None:
    assert env_ids is None
    backend = cast(_StateBackend, env._backend)
    backend.dof_pos += 10.0
    backend.qpos[:, 3] = torch.from_numpy(backend.dof_pos[:, 0].copy()).to(backend.qpos.device)


def _mutate_backend_tensor_state(env: _TestEnv, env_ids: np.ndarray | None) -> None:
    assert env_ids is None
    backend = cast(_ScenePlanBackend, env._backend)
    backend.sensors["track_pos_w_ball"].add_(10.0)


def _failure(env: _TestEnv) -> np.ndarray:
    return env.action_manager.action[:, 0] > 0.8


def _time_out(env: _TestEnv) -> np.ndarray:
    return env.episode_length_buf >= 2


def _metric(env: _TestEnv) -> np.ndarray:
    return env.episode_length_buf.to(torch.float32)


def _curriculum(env: _TestEnv, env_ids: np.ndarray | slice) -> float:
    del env
    return float(len(env_ids)) if isinstance(env_ids, np.ndarray) else 0.0


def _curriculum_mask(env: _TestEnv, reset_mask: torch.Tensor) -> float:
    del env
    return float(reset_mask.sum().item())


setattr(_curriculum, "__call_reset__", _curriculum_mask)


def _event(env: _TestEnv, env_ids: np.ndarray | None) -> None:
    rendered_ids = None if env_ids is None else env_ids.tolist()
    env.trace.append(("event", rendered_ids))


def _event_mask(env: _TestEnv, reset_mask: torch.Tensor) -> None:
    env.trace.append(("event", reset_mask.detach().cpu().tolist()))


setattr(_event, "reset", _event_mask)


def _startup_event(env: _TestEnv, env_ids: np.ndarray | None) -> None:
    assert env_ids is None
    backend = cast(_FakeBackend, env._backend)
    backend.lifecycle.append("startup")


def _observe_uncommitted_reset(env: _TestEnv, reset_mask: torch.Tensor) -> None:
    del reset_mask
    backend = cast(_ResetBackend, env._backend)
    env.trace.append(("commit_count_during_event", len(backend.set_state_calls)))


def _write_reset_joint_state(env: _TestEnv, reset_mask: torch.Tensor) -> None:
    env.scene["robot"].write_joint_state_to_sim_mask(
        torch.full((env.num_envs, 1), 0.25, dtype=torch.float32),
        torch.full((env.num_envs, 1), -0.5, dtype=torch.float32),
        reset_mask=reset_mask,
    )


setattr(_observe_uncommitted_reset, "reset", _observe_uncommitted_reset)
setattr(_write_reset_joint_state, "reset", _write_reset_joint_state)


def _make_cfg(
    *,
    sim_substeps: int = 2,
    finite_horizon: bool = False,
    auto_reset: bool = True,
    metrics: dict[str, MetricsTermCfg | None] | None = None,
    observations: dict[str, ObservationGroupCfg | None] | None = None,
    actions: dict[str, ActionTermCfg | None] | None = None,
    include_optional_managers: bool = True,
) -> ManagerBasedRlEnvCfg:
    if observations is None:
        observations = {
            "actor": ObservationGroupCfg(terms={"policy": ObservationTermCfg(func=_policy_obs)}),
            "value": ObservationGroupCfg(terms={"critic": ObservationTermCfg(func=_critic_obs)}),
        }
    cfg = ManagerBasedRlEnvCfg(
        scene=SceneCfg(
            model_file="fake.xml",
            entities={"robot": EntityCfg(actuator_names=("motor",))},
        ),
        sim_dt=0.01,
        ctrl_dt=0.01 * sim_substeps,
        max_episode_seconds=1.0,
        seed=7,
        observations=observations,
        actions={"drive": _DriveCfg(entity_name="robot")} if actions is None else actions,
        rewards={"track": RewardTermCfg(func=_reward, weight=1.0)},
        terminations={
            "failure": TerminationTermCfg(func=_failure),
            "time_out": TerminationTermCfg(func=_time_out, time_out=True),
        },
        events={"reset": EventTermCfg(func=_event, mode="reset")},
        metrics={} if metrics is None else metrics,
        policy_observation_group="actor",
        critic_observation_group="value",
        is_finite_horizon=finite_horizon,
        auto_reset=auto_reset,
    )
    if include_optional_managers:
        cfg.commands = {"target": _CommandCfg(resampling_time_range=(1.0, 1.0))}
        cfg.curriculum = {"difficulty": CurriculumTermCfg(func=_curriculum)}
        cfg.recorders = {"trace": RecorderTermCfg(func=_Recorder)}
    return cfg


def _make_env(
    cfg: ManagerBasedRlEnvCfg | None = None,
    *,
    num_envs: int = 2,
    reject_pre_step: bool = False,
    reject_materialize: bool = False,
    torch_stream_id: int | None = None,
) -> tuple[_TestEnv, _FakeBackend]:
    backend = _FakeBackend(
        num_envs,
        reject_pre_step=reject_pre_step,
        reject_materialize=reject_materialize,
        torch_stream_id=torch_stream_id,
    )
    env = _TestEnv(
        cfg or _make_cfg(),
        cast(SimBackend, backend),
        num_envs,
    )
    return env, backend


def _make_tensor_runtime_env(
    backend: _FakeBackend,
) -> _TestEnv:
    cfg = _make_cfg()
    cfg.observations = {
        "actor": ObservationGroupCfg(
            terms={"policy": ObservationTermCfg(func=_tensor_runtime_policy_obs)}
        ),
        "value": ObservationGroupCfg(
            terms={"critic": ObservationTermCfg(func=_tensor_runtime_critic_obs)}
        ),
    }
    return _TestEnv(cfg, cast(SimBackend, backend), 2)


def test_training_progress_restore_survives_next_step_and_rejects_invalid_state():
    env, _ = _make_env()
    env.reset()
    env.import_training_state({"version": 1, "step_counter": 700})
    assert env.common_step_counter == 700
    assert env._sim_step_counter == 700 * env.cfg.sim_substeps
    env.step(torch.zeros((env.num_envs, 1), dtype=torch.float32))
    assert env.step_counter == 701
    assert env.common_step_counter == 701
    assert env.export_training_state() == {"version": 1, "step_counter": 701}
    for invalid in (
        {"version": 1, "step_counter": -1},
        {"version": True, "step_counter": 4},
        {"version": 1, "step_counter": True},
        {"version": 2, "step_counter": 4},
        {"version": 1, "step_counter": 4, "extra": 0},
    ):
        with pytest.raises(ValueError):
            env.import_training_state(invalid)
        assert env.step_counter == env.common_step_counter == 701
    env.close()


def test_step_publishes_fixed_width_reward_components_for_tensor_collectors() -> None:
    env, _backend = _make_env()
    env.reset()
    state = env.step(torch.tensor([[0.25], [0.75]], dtype=torch.float32))

    assert env.algo_capabilities.reward_component_names == ("track",)
    assert "reward_components" in state.info
    assert "reward/track" not in state.info["log"]
    torch.testing.assert_close(
        state.info["reward_components"],
        torch.tensor([[0.25], [0.75]], dtype=torch.float32),
    )
    env.close()


def test_step_publishes_backend_graph_completion_for_transition_consumers() -> None:
    env, _backend = _make_env()
    env.reset()
    state = env.step(torch.tensor([[0.25], [0.75]], dtype=torch.float32))

    assert state.info["graph_completion"].operation == "step"
    env.close()


def test_backend_step_receipt_is_available_before_update_state() -> None:
    env, backend = _make_env(_make_cfg(include_optional_managers=False), torch_stream_id=17)
    env.init_state()
    observed: list[object] = []
    original_update_state = env.update_state

    def update_state_with_pending_receipt(state):
        observed.append(env._pending_backend_step_completion)
        return original_update_state(state)

    env.update_state = update_state_with_pending_receipt
    state = env.step(torch.zeros((2, 1), dtype=torch.float32, device=env.device))

    assert observed == [backend.graph_completions[-1]]
    assert env._pending_backend_step_completion is None
    assert state.info["graph_completion"] is backend.graph_completions[-1]
    env.close()


def test_invalid_backend_step_receipt_fails_before_update_state() -> None:
    env, backend = _make_env(_make_cfg(include_optional_managers=False))
    backend.graph_completions.append("not-a-token")
    env.init_state()

    with pytest.raises(TypeError, match="backend graph_completion"):
        env.step(torch.zeros((2, 1), dtype=torch.float32, device=env.device))
    assert env._pending_backend_step_completion is None
    assert "graph_completion" not in env._state.info
    env.close()


def test_graph_rng_snapshot_and_rollback_is_version_bounded() -> None:
    env, _ = _make_env()
    env.reset()
    before = env.snapshot_graph_rng_state()
    env.torch_rng.random((3,))
    after = env.graph_rng_state

    env.rollback_graph_rng_state(before)
    rolled_back = env.graph_rng_state

    assert after["draw_version"] > before[2]
    assert rolled_back["draw_version"] > after["draw_version"]
    torch.testing.assert_close(rolled_back["state"], before[0])
    with pytest.raises(ValueError, match="backward"):
        env.rollback_graph_rng_state((before[0], before[1] + 10, before[2] + 10))
    env.close()


def test_step_publishes_backend_reset_completion_for_terminal_boundaries() -> None:
    env, backend = _make_env()
    env.reset()

    completion = backend.set_state_tensor_mask(
        torch.tensor([False, True]), backend.qpos, backend.qvel
    )["graph_completion"]
    assert completion.operation == "mask_reset"
    assert env.last_reset_completion is None
    env.close()


def test_deferred_validation_boundaries_propagate_to_owner_verdicts() -> None:
    env, _backend = _make_env()
    env.set_deferred_validation_boundaries(True)

    assert env.deferred_validation_boundaries is True
    assert env.action_manager.deferred_validation_boundaries is True
    assert env.reward_manager.deferred_validation_boundaries is True
    assert env.observation_manager.deferred_validation_boundaries is True
    assert env._reset_state.deferred_payload_validation is True
    assert env._reset_state.deferred_mask_validation is True

    env.reward_manager._finite_valid.fill_(False)
    env.action_manager._action_finite_valid.fill_(True)
    env.observation_manager._group_finite_valid["actor"].fill_(True)
    with pytest.raises(ValueError, match="RewardManager returned a non-finite reward"):
        env.publish_validation_verdicts()
    env.close()


def test_deferred_validation_boundaries_fail_closed_without_backend_capability() -> None:
    env, _backend = _make_env()
    env._backend.packed_reset = False
    env._backend.get_tensor_runtime_diagnostics = lambda: {}

    with pytest.raises(NotImplementedError, match="deferred_fused_validation tensor capability"):
        env.set_deferred_validation_boundaries(True)
    assert env.deferred_validation_boundaries is True
    env.close()


def test_env_graph_candidate_preflight_reports_public_verdict() -> None:
    pytest.importorskip(
        "torch.cuda",
        reason="public graph-candidate preflight requires a CUDA-resident output carrier",
    )
    if not torch.cuda.is_available():
        pytest.skip("public graph-candidate preflight requires an available CUDA device")
    env, _backend = _make_env(_make_cfg(include_optional_managers=False), torch_stream_id=17)

    env.init_state()
    env.step(torch.zeros((2, 1), dtype=torch.float32, device=env.device))
    object.__setattr__(env, "_device", torch.device("cuda", index=torch.cuda.current_device()))
    env._final_observation_scratch = {
        name: value.to(device=env.device) for name, value in env._final_observation_scratch.items()
    }
    env._public_obs = {name: value.cuda() for name, value in env._public_obs.items()}
    env._public_reward = env._public_reward.to(device=env.device)
    env._public_terminated = env._public_terminated.to(device=env.device)
    env._public_truncated = env._public_truncated.to(device=env.device)
    env._torch_rng_owner = TorchManagerRng(
        device=env.device, generator=torch.Generator(device=env.device)
    ).manual_seed(7)
    env.torch_rng = env._torch_rng_owner
    env._torch_generator = env._torch_rng_owner.generator
    supported = env.preflight_graph_candidate(requested=True)
    unrequested = env.preflight_graph_candidate(requested=False)

    assert supported == {
        "requested": True,
        "supported": True,
        "disable_reason": None,
        "backend_stream_contract": "test",
        "torch_stream_id": 17,
        "carrier_layout": {
            "num_envs": 2,
            "device": str(env.device),
            "action_dim": 1,
            "observation_groups": {"obs": 2, "critic": 1},
            "reward_components": 1,
            "terminal_observation_groups": ["critic", "obs"],
            "persistent_output_carriers": True,
        },
        "rng_state": {
            "device": str(env.device),
            "state_version": env.torch_rng.state_version,
            "draw_version": env.torch_rng.draw_version,
        },
        "graph_lifecycle": "ready",
    }
    assert unrequested == {
        "requested": False,
        "supported": False,
        "disable_reason": "graph candidate was not requested",
        "backend_stream_contract": "test",
        "torch_stream_id": None,
    }
    env.close()


def test_env_graph_candidate_preflight_rejects_cpu_carrier() -> None:
    env, _backend = _make_env()
    env.init_state()

    result = env.preflight_graph_candidate(requested=True)

    assert result["supported"] is False
    assert "environment output carrier is not CUDA-resident" in result["disable_reason"]
    assert "terminal observation carrier is not materialized" not in result["disable_reason"]
    env.close()


def test_env_graph_lifecycle_manifest_is_public_cold_path_evidence() -> None:
    env, _backend = _make_env()

    assert env.get_graph_lifecycle_manifest() == {
        "contract_version": 1,
        "state": "ready",
        "capture_id": "test-capture",
        "replay_count": 2,
        "fallback_count": 0,
        "disable_reason": None,
    }
    env.close()


def test_failed_graph_replay_publishes_no_transition_and_rolls_back_rng() -> None:
    env, backend = _make_env()
    env.reset()
    completed = env.step(torch.zeros((2, 1), dtype=torch.float32))
    before_snapshot = env.snapshot_graph_rng_state()
    before_transition = {
        "obs": {name: value.clone() for name, value in completed.obs.items()},
        "reward": completed.reward.clone(),
        "terminated": completed.terminated.clone(),
        "truncated": completed.truncated.clone(),
        "steps": completed.info["steps"].clone(),
    }
    published_before_failure = len(backend.graph_completions)
    backend.fail_next_step_tensor = True

    with pytest.raises(RuntimeError, match="simulated graph replay failed"):
        env.step(torch.ones((2, 1), dtype=torch.float32))

    assert "graph_completion" not in env._state.info
    assert len(backend.tensor_controls) == 1
    assert len(backend.graph_completions) == published_before_failure
    assert env.step_counter == 1
    torch.testing.assert_close(env._state.info["steps"], before_transition["steps"])
    torch.testing.assert_close(env._state.obs["obs"], before_transition["obs"]["obs"])
    torch.testing.assert_close(env._state.reward, before_transition["reward"])
    torch.testing.assert_close(env._state.terminated, before_transition["terminated"])
    torch.testing.assert_close(env._state.truncated, before_transition["truncated"])

    failed_rng = env.graph_rng_state
    env.rollback_graph_rng_state(before_snapshot)
    rolled_back = env.graph_rng_state

    assert failed_rng["draw_version"] >= before_snapshot[2]
    assert rolled_back["state_version"] >= before_snapshot[1]
    assert rolled_back["draw_version"] >= before_snapshot[2]
    torch.testing.assert_close(rolled_back["state"], before_snapshot[0])

    invalidated = backend.invalidate_graph("capture_failed")
    assert backend.graph_invalidations == ["capture_failed"]
    assert invalidated.state.value == "invalidated"
    assert invalidated.capture_id is None
    assert invalidated.fallback_count == 1
    assert invalidated.disable_reason == "graph invalidated: capture_failed"
    env.close()


def test_env_graph_candidate_preflight_requires_terminal_carrier() -> None:
    env, _backend = _make_env(_make_cfg(include_optional_managers=False))

    result = env.preflight_graph_candidate(requested=True)

    assert result["supported"] is False
    assert result["disable_reason"] == (
        "environment output carrier is not CUDA-resident; "
        "terminal observation carrier is not materialized"
    )
    env.close()


def test_step_publishes_non_aliased_public_output_carriers() -> None:
    env, _backend = _make_env()
    env.init_state()
    first = env.step(torch.zeros((2, 1), dtype=torch.float32, device=env.device))
    terminated_ptr = first.terminated.data_ptr()
    truncated_ptr = first.truncated.data_ptr()

    second = env.step(torch.zeros((2, 1), dtype=torch.float32, device=env.device))

    assert second.terminated.data_ptr() != terminated_ptr
    assert second.truncated.data_ptr() != truncated_ptr
    env.close()


def test_graph_rng_state_snapshot_is_versioned_and_restoreable() -> None:
    env, _backend = _make_env()
    env.init_state()
    before = env.graph_rng_state

    env.step(torch.zeros((2, 1), dtype=torch.float32, device=env.device))
    after_draw = env.graph_rng_state
    consumed_state = env.torch_rng.get_state()

    assert after_draw["state_version"] == before["state_version"]
    assert after_draw["draw_version"] > before["draw_version"]
    assert torch.equal(after_draw["state"], before["state"]) is False

    env.torch_rng.set_state(consumed_state)
    env.torch_rng.set_state(before["state"])
    restored = env.graph_rng_state
    assert restored["state_version"] == before["state_version"] + 2
    assert restored["draw_version"] == after_draw["draw_version"] + 2
    first_after_restore = env.torch_rng.random(1)
    env.torch_rng.set_state(restored["state"])
    torch.testing.assert_close(env.torch_rng.random(1), first_after_restore)
    env.close()


def test_env_update_graph_defaults_disabled_and_negotiates_distinct_eager_mode() -> None:
    env, _backend = _make_env(_make_cfg(include_optional_managers=False))

    initial = env.get_env_update_graph_manifest()
    assert initial["mode"] == "disabled"
    assert initial["requested"] is False
    assert initial["enabled"] is False
    assert initial["backend_step_receipt"] == {
        "operation": None,
        "sequence": None,
        "capture_id": None,
    }
    assert initial["update_completion"]["operation"] == "env_update"
    assert initial["update_completion"]["capture_id"] is None
    validate_env_update_graph_manifest(initial)

    env.configure_env_update_graph(EnvUpdateGraphMode.EAGER)
    eager = env.get_env_update_graph_manifest()

    assert eager["mode"] == "eager"
    assert eager["requested"] is False
    assert eager["enabled"] is False
    assert eager["state"] == "eager"
    assert eager["disable_reason"] == "explicit eager env update execution"
    validate_env_update_graph_manifest(eager)
    env.close()


def test_env_update_graph_request_fails_closed_without_backend_boundary() -> None:
    env, _backend = _make_env(_make_cfg(include_optional_managers=False))
    env._backend.reject_env_update_preflight = True
    env.init_state()

    with pytest.raises(RuntimeError, match="test backend rejects update capture"):
        env.configure_env_update_graph("graph")
    failed = env.get_env_update_graph_manifest()
    assert failed["mode"] == "graph"
    assert failed["requested"] is True
    assert failed["enabled"] is False
    assert failed["state"] == "invalidated"
    assert failed["invalidation_reason"] == "capture_failed"
    validate_env_update_graph_manifest(failed)

    env.configure_env_update_graph("graph", allow_eager_fallback=True)
    fallback = env.get_env_update_graph_manifest()
    assert fallback["mode"] == "eager"
    assert fallback["requested"] is False
    assert fallback["fallback_count"] == 1
    assert fallback["disable_reason"].startswith("explicit eager fallback:")
    validate_env_update_graph_manifest(fallback)
    env.close()


def test_env_update_graph_supported_preflight_fails_closed_on_cpu_carriers() -> None:
    env, backend = _make_env(_make_cfg(include_optional_managers=False), torch_stream_id=17)
    env.init_state()
    env.set_deferred_validation_boundaries(True)

    with pytest.raises(RuntimeError, match="environment carriers must be CUDA-resident"):
        env.configure_env_update_graph("graph")
    manifest = env.get_env_update_graph_manifest()

    assert manifest["mode"] == "graph"
    assert manifest["requested"] is True
    assert manifest["enabled"] is False
    assert manifest["state"] == "invalidated"
    assert manifest["invalidation_reason"] == "capture_failed"
    assert "environment carriers must be CUDA-resident" in manifest["disable_reason"]
    validate_env_update_graph_manifest(manifest)

    env.configure_env_update_graph("graph", allow_eager_fallback=True)
    fallback = env.get_env_update_graph_manifest()
    assert fallback["mode"] == "eager"
    assert fallback["fallback_count"] == 1
    assert fallback["disable_reason"].startswith("explicit eager fallback:")
    validate_env_update_graph_manifest(fallback)
    env.close()


def test_env_update_graph_owner_preflight_rejects_cpu_carriers() -> None:
    cfg = _make_cfg(include_optional_managers=False)
    cfg.events = {}
    env, backend = _make_env(cfg, torch_stream_id=17)
    env.init_state()
    env.set_deferred_validation_boundaries(True)

    with pytest.raises(RuntimeError, match="environment carriers must be CUDA-resident"):
        env.configure_env_update_graph("graph")
    manifest = env.get_env_update_graph_manifest()
    assert manifest["state"] == "invalidated"
    assert manifest["invalidation_reason"] == "capture_failed"
    validate_env_update_graph_manifest(manifest)
    assert backend.tensor_controls == []
    env.close()


def test_env_update_graph_owner_preflight_requires_deferred_validation() -> None:
    cfg = _make_cfg(include_optional_managers=False)
    cfg.events = {}
    env, _backend = _make_env(cfg, torch_stream_id=17)
    env.init_state()

    with pytest.raises(RuntimeError, match="deferred validation boundaries must be enabled"):
        env.configure_env_update_graph("graph")
    assert env.get_env_update_graph_manifest()["invalidation_reason"] == "capture_failed"
    env.close()


def test_env_update_graph_carrier_layout_pins_update_outputs_and_verdicts() -> None:
    env, _backend = _make_env(_make_cfg(include_optional_managers=False))
    env.init_state()
    env.step(torch.zeros((2, 1), dtype=torch.float32, device=env.device))

    verdicts = env.env_update_validation_verdicts
    layout = env.get_env_update_carrier_layout()

    assert set(layout["observations"]) == set(env.obs_groups_spec)
    assert layout["reward_components"]["shape"] == (2, 1)
    assert layout["episode_steps"]["shape"] == (2,)
    assert layout["command_dt"]["shape"] == (2,)
    assert layout["terminated"]["shape"] == (2,)
    assert layout["truncated"]["shape"] == (2,)
    assert layout["reset_mask"]["shape"] == (2,)
    assert layout["reward_components"]["data_ptr"] == (env.reward_manager.step_rewards.data_ptr())
    assert layout["episode_steps"]["data_ptr"] == env.episode_length_buf.data_ptr()
    assert layout["command_dt"]["data_ptr"] == env._command_dt.data_ptr()
    assert layout["reset_mask"]["data_ptr"] == env.reset_buf.data_ptr()
    assert set(layout["validation_verdicts"]) == set(verdicts)
    assert layout["validation_verdicts"]["reward_finite"]["data_ptr"] == (
        verdicts["reward_finite"].data_ptr()
    )
    env.close()


def test_graph_preflight_rejects_mismatched_rng_device() -> None:
    if not torch.cuda.is_available():
        pytest.skip("mismatched graph RNG device coverage requires moving carriers to CUDA")
    env, _backend = _make_env()
    env.init_state()
    env.step(torch.zeros((2, 1), dtype=torch.float32, device=env.device))
    object.__setattr__(env, "_device", torch.device("cuda", index=torch.cuda.current_device()))
    env._final_observation_scratch = {
        name: value.to(device=env.device) for name, value in env._final_observation_scratch.items()
    }

    result = env.preflight_graph_candidate(requested=True)

    assert result["supported"] is False
    assert "Torch RNG device does not match the graph carrier" in result["disable_reason"]
    env.close()


def test_no_done_explicit_reset_skips_pre_reset_recorder_in_eager_mode() -> None:
    cfg = _make_cfg(auto_reset=True)
    cfg.max_episode_seconds = 10.0
    cfg.terminations["time_out"] = None
    cfg.recorders = {"trace": RecorderTermCfg(func=_Recorder)}
    env, _backend = _make_env(cfg)
    env.init_state()
    env._has_transition = True
    obs, _info = env.reset(torch.zeros(2, dtype=torch.bool))

    assert obs["obs"].shape == (2, 2)
    assert env.reset_buf.tolist() == [False, False]
    assert ("pre_reset", [False, False]) not in env.trace
    env.close()


def test_manager_public_inputs_and_episode_counters_are_tensor_first() -> None:
    env, _ = _make_env()
    env.reset()
    assert env.state is not None

    with pytest.raises(TypeError, match="torch.Tensor"):
        env.step(np.zeros((env.num_envs, 1), dtype=np.float32))
    with pytest.raises(TypeError, match="env_ids"):
        env.reset(env_ids=np.array([0], dtype=np.int32))  # type: ignore[call-arg]
    with pytest.raises(ValueError, match="reset mask"):
        env.reset(torch.tensor([0, 0], dtype=torch.int64))

    values = torch.tensor([2, 3], dtype=torch.int64)
    env.set_episode_length_buf(values)
    torch.testing.assert_close(env.state.info["steps"], values)
    torch.testing.assert_close(env.episode_length_buf, values)
    env.close()


@pytest.mark.parametrize(
    ("manager_torch_device", "expected_device"),
    [
        (None, torch.device("cpu")),
        ("cpu", torch.device("cpu")),
    ],
)
def test_backend_capability_defaults_host_bridge_manager_carriers_to_cpu(
    manager_torch_device: str | None,
    expected_device: torch.device,
) -> None:
    cfg = _make_cfg()
    cfg.manager_torch_device = manager_torch_device

    env, backend = _make_env(cfg)

    assert env.device == torch.device("cpu")
    assert backend.lifecycle[-1] == "materialize"
    assert env._tensor_runtime_bound is True
    env.reset()
    assert all(value.device == env.device for value in env.obs_buf.values())
    env.close()


def test_host_bridge_ignores_accelerator_availability_without_explicit_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if not bool(torch.cuda._is_compiled()):
        pytest.skip("fake CUDA visibility requires a Torch CUDA/ROCm build")
    original_is_available = torch.cuda.is_available
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    try:
        cfg = _make_cfg()
        env, _ = _make_env(cfg)
    finally:
        monkeypatch.setattr(torch.cuda, "is_available", original_is_available)
        env.close()

    assert env.device == torch.device("cpu")


def test_manager_torch_device_rejects_non_tensor_device_and_invalid_syntax() -> None:
    for value, error in (("mps", ValueError), (12, TypeError), ("", TypeError)):
        cfg = _make_cfg()
        cfg.manager_torch_device = value
        with pytest.raises(error):
            cfg.validate()


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_mujoco_host_bridge_packed_reset_lifecycle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, device: str
) -> None:
    pytest.importorskip("unisim.backend.mujoco.backend")
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("GPU host bridge requires CUDA or ROCm PyTorch")
    if device == "cpu":
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    model = tmp_path / "model.xml"
    model.write_text(
        "<mujoco><option gravity='0 0 0'/><worldbody><body name='base'>"
        "<freejoint/><geom type='box' size='0.1 0.1 0.1' contype='0' conaffinity='0'/>"
        "<body name='arm' pos='0 0 0.2'><joint name='joint' axis='0 0 1'/>"
        "<geom type='box' size='0.1 0.1 0.1' contype='0' conaffinity='0'/>"
        "</body></body></worldbody><actuator><motor name='motor' joint='joint' "
        "ctrlrange='-1 1'/></actuator></mujoco>"
    )
    scene = SceneCfg(
        model_file=str(model),
        entities={
            "robot": EntityCfg(
                root_body_name="base", joint_names=("joint",), actuator_names=("motor",)
            )
        },
    )
    cfg = ManagerBasedRlEnvCfg(
        scene=scene,
        sim_dt=0.01,
        ctrl_dt=0.02,
        max_episode_seconds=0.04,
        observations={
            "actor": ObservationGroupCfg(
                terms={"joint": ObservationTermCfg(func=mdp.joint_pos_rel)}
            )
        },
        actions={"effort": mdp.JointEffortActionCfg(entity_name="robot", actuator_names=(".*",))},
        events={"default": EventTermCfg(func=mdp.reset_scene_to_default, mode="reset")},
        rewards={"alive": RewardTermCfg(func=mdp.is_alive, weight=1.0)},
        terminations={"time_out": TerminationTermCfg(func=mdp.time_out, time_out=True)},
        policy_observation_group="actor",
    )
    if device == "cuda":
        cfg.manager_torch_device = device
    backend = create_backend(
        "mujoco", scene, 2, cfg.sim_dt, base_name="base", add_body_sensors=True
    )
    try:
        capabilities = backend.get_tensor_capabilities()
        assert capabilities.execution is TensorExecution.HOST_BRIDGE
        assert capabilities.packed_reset
        env = ManagerBasedRlEnv(cfg, backend, 2)
        state = env.init_state()
        assert state.obs["obs"].shape == (2, 1)
        assert state.obs["obs"].device == env.device
        assert all(torch.isfinite(values).all() for values in state.obs.values())
        env.step(torch.zeros((2, 1), dtype=torch.float32, device=env.device))
        obs, _ = env.reset(torch.tensor([False, True], dtype=torch.bool, device=env.device))
        assert obs["obs"].shape == (2, 1)
        assert all(torch.isfinite(values).all() for values in obs.values())
        env.close()
    finally:
        if "env" in locals():
            env.close()
        backend.close()


def _make_state_env(
    *,
    commands: dict[str, CommandTermCfg | None] | None = None,
    events: dict[str, EventTermCfg | None] | None = None,
    terminations: dict[str, TerminationTermCfg | None] | None = None,
) -> tuple[_TestEnv, _StateBackend]:
    cfg = _make_cfg(
        sim_substeps=1,
        observations={
            "actor": ObservationGroupCfg(
                terms={"joint": ObservationTermCfg(func=_joint_state_obs)}
            ),
            "value": ObservationGroupCfg(
                terms={"joint": ObservationTermCfg(func=_joint_state_obs)}
            ),
        },
        include_optional_managers=False,
    )
    cfg.scene.entities["robot"] = EntityCfg(
        joint_names=("joint",),
        actuator_names=("motor",),
    )
    cfg.scale_rewards_by_dt = False
    cfg.rewards = {"joint": RewardTermCfg(func=_joint_state_reward, weight=1.0)}
    cfg.terminations = (
        {"joint": TerminationTermCfg(func=_joint_state_termination)}
        if terminations is None
        else terminations
    )
    cfg.commands = {} if commands is None else commands
    cfg.events = {} if events is None else events
    backend = _StateBackend(2)
    return _TestEnv(cfg, cast(SimBackend, backend), 2), backend


def test_public_names_are_spelling_only_aliases() -> None:
    assert ManagerBasedRLEnv is ManagerBasedRlEnv
    assert ManagerBasedRLEnvCfg is ManagerBasedRlEnvCfg
    assert make_manager_based_rl_env is manager_env_module.make_manager_based_rl_env


@pytest.mark.parametrize("backend_type", ["mujoco", "mjwarp"])
def test_generic_factory_routes_only_public_backend_contract(
    monkeypatch: pytest.MonkeyPatch,
    backend_type: str,
) -> None:
    cfg = _make_cfg(include_optional_managers=False)
    assert cfg.scene is not None
    cfg.scene.entities["robot"] = EntityCfg(
        root_body_name="base",
        actuator_names=("motor",),
    )
    cfg.scene.entities["payload"] = EntityCfg(root_body_name="box")
    backend = _FakeBackend(3)
    constructed: dict[str, Any] = {}

    def fake_create_backend(
        requested_backend: str,
        scene: SceneCfg,
        num_envs: int,
        sim_dt: float,
        **kwargs: Any,
    ) -> SimBackend:
        constructed.update(
            backend_type=requested_backend,
            scene=scene,
            num_envs=num_envs,
            sim_dt=sim_dt,
            kwargs=kwargs,
        )
        return cast(SimBackend, backend)

    sentinel = object()

    def fake_make_env(
        received_cfg: ManagerBasedRlEnvCfg,
        received_backend: SimBackend,
        received_num_envs: int,
    ) -> Any:
        assert received_cfg is cfg
        assert received_backend is backend
        assert received_num_envs == 3
        return sentinel

    monkeypatch.setattr(manager_env_module, "create_backend", fake_create_backend)
    monkeypatch.setattr(manager_env_module, "ManagerBasedRlEnv", fake_make_env)

    result = make_manager_based_rl_env(cfg, num_envs=3, backend_type=backend_type)

    assert result is sentinel
    assert constructed["backend_type"] == backend_type
    assert constructed["scene"] is cfg.scene
    assert constructed["num_envs"] == 3
    assert constructed["sim_dt"] == cfg.sim_dt
    kwargs = constructed["kwargs"]
    assert kwargs["base_name"] == "base"
    assert kwargs["body_state_required"] is True
    assert "add_body_sensors" not in kwargs
    for key, value in env_backend_kwargs(cfg).items():
        assert kwargs[key] == value


@pytest.mark.parametrize(
    ("entities", "match"),
    [
        (
            {"robot": EntityCfg(actuator_names=("motor",))},
            "at least one scene entity with an explicit root_body_name",
        ),
        (
            {
                "payload": EntityCfg(root_body_name="box"),
                "tool": EntityCfg(root_body_name="tool"),
            },
            "conventional 'robot' root entity.*payload.*tool",
        ),
    ],
)
def test_generic_factory_rejects_missing_or_ambiguous_root_entity(
    monkeypatch: pytest.MonkeyPatch,
    entities: dict[str, EntityCfg],
    match: str,
) -> None:
    cfg = _make_cfg(include_optional_managers=False)
    assert cfg.scene is not None
    cfg.scene.entities = entities
    backend_constructed = False

    def reject_backend_construction(*args: Any, **kwargs: Any) -> SimBackend:
        nonlocal backend_constructed
        backend_constructed = True
        raise AssertionError("backend construction must not run")

    monkeypatch.setattr(manager_env_module, "create_backend", reject_backend_construction)

    with pytest.raises(ValueError, match=match):
        make_manager_based_rl_env(cfg, num_envs=2, backend_type="mujoco")
    assert not backend_constructed


@pytest.mark.parametrize(
    ("cfg_value", "num_envs", "backend_type", "error", "match"),
    [
        (object(), 2, "mujoco", TypeError, "expected ManagerBasedRlEnvCfg"),
        (None, 0, "mujoco", ValueError, "num_envs must be a positive integer"),
        (None, True, "mujoco", ValueError, "num_envs must be a positive integer"),
        (None, 2, "", ValueError, "backend_type must be a non-empty string"),
    ],
)
def test_generic_factory_rejects_invalid_public_arguments_before_backend(
    monkeypatch: pytest.MonkeyPatch,
    cfg_value: object | None,
    num_envs: int,
    backend_type: str,
    error: type[Exception],
    match: str,
) -> None:
    cfg = _make_cfg(include_optional_managers=False) if cfg_value is None else cfg_value
    backend_constructed = False

    def reject_backend_construction(*args: Any, **kwargs: Any) -> SimBackend:
        nonlocal backend_constructed
        backend_constructed = True
        raise AssertionError("backend construction must not run")

    monkeypatch.setattr(manager_env_module, "create_backend", reject_backend_construction)

    with pytest.raises(error, match=match):
        make_manager_based_rl_env(
            cast(ManagerBasedRlEnvCfg, cfg),
            num_envs=num_envs,
            backend_type=backend_type,
        )
    assert not backend_constructed


@pytest.mark.parametrize(
    ("entities", "error", "match"),
    [
        (
            cast(dict[str, EntityCfg], {1: EntityCfg(root_body_name="base")}),
            TypeError,
            "entity names must be non-empty strings",
        ),
        (
            cast(dict[str, EntityCfg], {"robot": object()}),
            TypeError,
            "scene entity 'robot' must be EntityCfg",
        ),
        (
            {"robot": EntityCfg(root_body_name="")},
            TypeError,
            "root_body_name must be a non-empty string",
        ),
    ],
)
def test_generic_factory_rejects_invalid_entity_contract_before_backend(
    monkeypatch: pytest.MonkeyPatch,
    entities: dict[str, EntityCfg],
    error: type[Exception],
    match: str,
) -> None:
    cfg = _make_cfg(include_optional_managers=False)
    assert cfg.scene is not None
    cfg.scene.entities = entities
    backend_constructed = False

    def reject_backend_construction(*args: Any, **kwargs: Any) -> SimBackend:
        nonlocal backend_constructed
        backend_constructed = True
        raise AssertionError("backend construction must not run")

    monkeypatch.setattr(manager_env_module, "create_backend", reject_backend_construction)

    with pytest.raises(error, match=match):
        make_manager_based_rl_env(cfg, num_envs=2, backend_type="mujoco")
    assert not backend_constructed


def test_generic_factory_preserves_backend_construction_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = _make_cfg(include_optional_managers=False)
    assert cfg.scene is not None
    cfg.scene.entities["robot"] = EntityCfg(
        root_body_name="base",
        actuator_names=("motor",),
    )

    def fail_backend_construction(*args: Any, **kwargs: Any) -> SimBackend:
        raise NotImplementedError("body-state capability is unavailable")

    monkeypatch.setattr(manager_env_module, "create_backend", fail_backend_construction)

    with pytest.raises(NotImplementedError, match="body-state capability is unavailable"):
        make_manager_based_rl_env(cfg, num_envs=2, backend_type="mjwarp")


def test_generic_factory_cleans_backend_when_env_construction_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = _make_cfg(include_optional_managers=False)
    assert cfg.scene is not None
    cfg.scene.entities["robot"] = EntityCfg(
        root_body_name="base",
        actuator_names=("motor",),
    )
    backend = _FakeBackend(2)

    monkeypatch.setattr(
        manager_env_module,
        "create_backend",
        lambda *args, **kwargs: cast(SimBackend, backend),
    )

    def fail_env_construction(*args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("manager initialization failed")

    monkeypatch.setattr(manager_env_module, "ManagerBasedRlEnv", fail_env_construction)

    with pytest.raises(RuntimeError, match="manager initialization failed"):
        make_manager_based_rl_env(cfg, num_envs=2, backend_type="mujoco")
    assert backend.cleanup_calls == 1


@pytest.mark.parametrize(
    ("mutate", "error", "match"),
    [
        (lambda cfg: setattr(cfg, "sim_dt", True), TypeError, "sim_dt must be a real"),
        (
            lambda cfg: setattr(cfg, "ctrl_dt", 0.015),
            ValueError,
            "integer multiple",
        ),
        (
            lambda cfg: setattr(cfg, "max_episode_seconds", None),
            ValueError,
            "max_episode_seconds",
        ),
        (lambda cfg: setattr(cfg, "scene", None), TypeError, "scene must be a SceneCfg"),
    ],
)
def test_manager_based_config_rejects_invalid_contracts(mutate, error, match: str) -> None:
    cfg = _make_cfg()
    mutate(cfg)
    with pytest.raises(error, match=match):
        cfg.validate()


def test_manager_construction_uses_pinned_order(monkeypatch: pytest.MonkeyPatch) -> None:
    order: list[str] = []
    names = (
        "EventManager",
        "CommandManager",
        "ActionManager",
        "ObservationManager",
        "TerminationManager",
        "RewardManager",
        "CurriculumManager",
        "MetricsManager",
        "RecorderManager",
    )
    cfg = _make_cfg(metrics={"progress": MetricsTermCfg(func=_metric)})
    for name in names:
        original = getattr(manager_env_module, name)

        def wrapped(*args, _name=name, _original=original, **kwargs):
            order.append(_name)
            return _original(*args, **kwargs)

        monkeypatch.setattr(manager_env_module, name, wrapped)

    _make_env(cfg)
    assert order == list(names)


def test_backend_materializes_once_before_startup_and_runtime() -> None:
    cfg = _make_cfg()
    cfg.events = {
        "startup": EventTermCfg(func=_startup_event, mode="startup"),
        "reset": EventTermCfg(func=_event, mode="reset"),
    }
    env, backend = _make_env(cfg)

    assert backend.lifecycle == ["materialize", "startup"]
    assert backend.materialize_calls == 1


def test_startup_model_field_dr_commits_after_backend_materialization() -> None:
    cfg = _make_cfg()
    cfg.scene.entities["robot"] = EntityCfg(
        root_body_name="base",
        body_names=("base",),
        joint_names=("joint",),
        actuator_names=("motor",),
    )
    cfg.observations = {
        "actor": ObservationGroupCfg(terms={"policy": ObservationTermCfg(func=_policy_obs)}),
        "value": ObservationGroupCfg(terms={"critic": ObservationTermCfg(func=_critic_obs)}),
    }
    cfg.events = {
        "reset_scene": EventTermCfg(func=mdp.reset_scene_to_default, mode="reset"),
        "body_mass": EventTermCfg(
            func=mdp.randomize_rigid_body_mass,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=("base",)),
                "mass_distribution_params": (1.5, 1.5),
                "operation": "scale",
                "recompute_inertia": False,
            },
        ),
        "body_com": EventTermCfg(
            func=mdp.randomize_rigid_body_com,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=("base",)),
                "com_range": {"x": (0.1, 0.1), "z": (-0.2, -0.2)},
            },
        ),
    }
    backend = _ResetBackend(2)
    env = _TestEnv(cfg, cast(SimBackend, backend), 2)

    assert backend.lifecycle == ["materialize"]
    assert len(backend.set_state_calls) == 1
    np.testing.assert_array_equal(backend.set_state_calls[0][0], np.arange(2))
    payload = backend.randomization_calls[0]
    assert payload is not None
    assert payload.body_mass is not None
    assert payload.body_ipos is not None
    np.testing.assert_allclose(payload.body_mass, [[15.0], [15.0]])
    np.testing.assert_allclose(payload.body_ipos, [[[0.1, 0.0, -0.2]]] * 2)

    env.init_state()
    assert backend.lifecycle == ["materialize"]
    assert len(backend.set_state_calls) == 2
    assert len(backend.randomization_calls) == 2
    assert backend.randomization_calls[-1] is None

    env.reset(torch.tensor([True, False], dtype=torch.bool))
    env.step(torch.zeros((2, 1), dtype=torch.float32))
    env.reset()

    assert backend.materialize_calls == 1


def test_backend_materialization_failure_has_lifecycle_context() -> None:
    with pytest.raises(
        RuntimeError,
        match="ManagerBasedRlEnv failed to materialize backend 'fake'.*pool construction failed",
    ):
        _make_env(reject_materialize=True)


@pytest.mark.parametrize(
    ("selector", "error"),
    [
        (1, TypeError),
        ("", ValueError),
    ],
)
def test_default_keyframe_selector_fails_at_env_initialization(
    selector: Any,
    error: type[Exception],
) -> None:
    cfg = _make_cfg(include_optional_managers=False)
    cfg.scene.default_keyframe_name = selector
    backend = _KeyframeBackend(2)

    with pytest.raises(error, match="default_keyframe_name.*non-empty string or None"):
        _TestEnv(cfg, cast(SimBackend, backend), 2)

    assert backend.keyframe_qpos_calls == 0


@pytest.mark.parametrize(
    ("value", "error", "match"),
    [
        ([0.0], TypeError, "must return np.ndarray"),
        (np.zeros((1, 1), dtype=np.float32), ValueError, "expected 1-D"),
        (np.zeros(1, dtype=np.int32), TypeError, "must be floating"),
        (np.array([np.nan], dtype=np.float32), ValueError, "NaN or Inf"),
    ],
)
def test_default_keyframe_qpos_contract_fails_at_env_initialization(
    value: Any,
    error: type[Exception],
    match: str,
) -> None:
    cfg = _make_cfg(include_optional_managers=False)
    cfg.scene.default_keyframe_name = "home"
    backend = _KeyframeBackend(2, keyframe_qpos=value)

    with pytest.raises(error, match=f"default keyframe 'home'.*backend 'fake'.*{match}"):
        _TestEnv(cfg, cast(SimBackend, backend), 2)

    assert backend.keyframe_qpos_calls == 1


@pytest.mark.parametrize(
    ("failure", "error", "match"),
    [
        (
            NotImplementedError("named keyframes disabled"),
            NotImplementedError,
            "default keyframe 'home'.*backend 'fake'.*named keyframes disabled",
        ),
        (
            ValueError("keyframe missing"),
            ValueError,
            "resolve default keyframe 'home'.*backend 'fake'.*keyframe missing",
        ),
    ],
)
def test_default_keyframe_resolution_names_backend_and_keyframe(
    failure: Exception,
    error: type[Exception],
    match: str,
) -> None:
    cfg = _make_cfg(include_optional_managers=False)
    cfg.scene.default_keyframe_name = "home"
    backend = _KeyframeBackend(2, keyframe_error=failure)

    with pytest.raises(error, match=match):
        _TestEnv(cfg, cast(SimBackend, backend), 2)

    assert backend.keyframe_qpos_calls == 1


def test_named_keyframe_snapshot_is_shared_by_entity_and_reset_cold_path() -> None:
    source_qpos = np.array([0.0, 0.0, 0.3, 0.75], dtype=np.float64)
    backend = _KeyframeBackend(2, keyframe_qpos=source_qpos)
    cfg = _make_cfg(include_optional_managers=False)
    cfg.scene.default_keyframe_name = "home"
    cfg.scene.entities["robot"] = EntityCfg(
        joint_names=("joint",),
        actuator_names=("motor",),
    )
    cfg.events = {
        "reset_default": EventTermCfg(func=mdp.reset_scene_to_default, mode="reset"),
    }

    env = _TestEnv(cfg, cast(SimBackend, backend), 2)
    selected_qpos = env._reset_state._selected_default_qpos

    assert backend.keyframe_qpos_calls == 1
    assert backend.default_qpos_calls == 0
    assert backend.joint_layout_calls == 1
    assert selected_qpos is not source_qpos
    assert selected_qpos is not None
    assert not selected_qpos.flags.writeable
    torch.testing.assert_close(
        env.scene["robot"].data.default_joint_pos_torch(env.device), torch.full((2, 1), 0.75)
    )

    source_qpos[:] = 9.0
    env.reset()
    env.reset(torch.tensor([False, True], dtype=torch.bool))

    assert backend.keyframe_qpos_calls == 1
    assert backend.default_qpos_calls == 0
    assert backend.joint_layout_calls == 1
    assert len(backend.set_state_calls) == 2
    np.testing.assert_allclose(
        backend.set_state_calls[0][1],
        [[0.0, 0.0, 0.3, 0.75], [0.0, 0.0, 0.3, 0.75]],
        rtol=0,
        atol=1e-7,
    )
    np.testing.assert_allclose(
        backend.set_state_calls[1][1],
        [[0.0, 0.0, 0.3, 0.75]],
        rtol=0,
        atol=1e-7,
    )


def test_real_mujoco_backend_runs_packed_reset_lifecycle() -> None:
    pytest.importorskip(
        "unisim.backend.mujoco.backend",
        reason="unisim-core MuJoCo adapter (mjbatch build) not available",
    )
    scene = SceneCfg(
        model_file=str(ASSETS_ROOT_PATH / "robots" / "go2" / "scene_flat.xml"),
        entities={"robot": EntityCfg(root_body_name="base")},
    )
    cfg = ManagerBasedRlEnvCfg(
        scene=scene,
        sim_dt=0.01,
        ctrl_dt=0.02,
        max_episode_seconds=1.0,
        observations={
            "actor": ObservationGroupCfg(
                terms={"state": ObservationTermCfg(func=_episode_step_observation)}
            )
        },
        actions={},
        events={"reset_default": EventTermCfg(func=mdp.reset_scene_to_default, mode="reset")},
        rewards={"alive": RewardTermCfg(func=mdp.is_alive, weight=1.0)},
        terminations={"time_out": TerminationTermCfg(func=mdp.time_out, time_out=True)},
        policy_observation_group="actor",
    )
    env = make_manager_based_rl_env(cfg, num_envs=2, backend_type="mujoco")
    try:
        state = env.init_state()
        assert state.obs["obs"].shape == (2, 1)
        assert torch.isfinite(state.obs["obs"]).all()
        stepped = env.step(torch.empty((2, 0), dtype=torch.float32, device=env.device))
        assert torch.isfinite(stepped.obs["obs"]).all()
        assert torch.isfinite(stepped.reward).all()
        obs, _ = env.reset(torch.tensor([True, False], dtype=torch.bool))
        assert obs["obs"].shape == (2, 1)
        assert torch.isfinite(obs["obs"]).all()
    finally:
        env.close()


@pytest.mark.parametrize(
    ("default_keyframe_name", "expected_root_z", "expected_joint_pos"),
    [
        (None, 0.445, np.zeros(12, dtype=np.float32)),
        (
            "home",
            0.3,
            np.array(
                [0.0, 0.8, -1.5, 0.0, 0.8, -1.5, 0.0, 1.0, -1.5, 0.0, 1.0, -1.5],
                dtype=np.float32,
            ),
        ),
    ],
)
def test_real_mujoco_default_state_applies_keyframe_baseline(
    default_keyframe_name: str | None,
    expected_root_z: float,
    expected_joint_pos: np.ndarray,
) -> None:
    pytest.importorskip(
        "unisim.backend.mujoco.backend",
        reason="unisim-core MuJoCo adapter (mjbatch build) not available",
    )
    joint_names = (
        "FL_hip_joint",
        "FL_thigh_joint",
        "FL_calf_joint",
        "FR_hip_joint",
        "FR_thigh_joint",
        "FR_calf_joint",
        "RL_hip_joint",
        "RL_thigh_joint",
        "RL_calf_joint",
        "RR_hip_joint",
        "RR_thigh_joint",
        "RR_calf_joint",
    )
    scene = SceneCfg(
        model_file=str(ASSETS_ROOT_PATH / "robots" / "go2" / "scene_flat.xml"),
        entities={
            "robot": EntityCfg(
                root_body_name="base",
                joint_names=joint_names,
            )
        },
        default_keyframe_name=default_keyframe_name,
    )
    cfg = ManagerBasedRlEnvCfg(
        scene=scene,
        sim_dt=0.01,
        ctrl_dt=0.02,
        max_episode_seconds=1.0,
        observations={
            "actor": ObservationGroupCfg(
                terms={"state": ObservationTermCfg(func=_episode_step_observation)}
            )
        },
        actions={},
        events={"reset_default": EventTermCfg(func=mdp.reset_scene_to_default, mode="reset")},
        rewards={"alive": RewardTermCfg(func=mdp.is_alive, weight=1.0)},
        terminations={"time_out": TerminationTermCfg(func=mdp.time_out, time_out=True)},
        policy_observation_group="actor",
    )
    backend = create_backend(
        "mujoco",
        scene,
        2,
        cfg.sim_dt,
        base_name="base",
        add_body_sensors=True,
        **env_backend_kwargs(cfg),
    )
    try:
        env = ManagerBasedRlEnv(cfg, backend, 2)
        robot = env.scene["robot"]
        torch.testing.assert_close(
            robot.data.default_root_state_torch(env.device)[:, 2],
            torch.full((2,), expected_root_z, dtype=torch.float32),
        )
        expected_batch = np.broadcast_to(expected_joint_pos, (2, expected_joint_pos.size))
        torch.testing.assert_close(
            robot.data.default_joint_pos_torch(env.device),
            torch.as_tensor(expected_batch),
        )

        env.init_state()
        env.reset()
        robot_data = robot.data
        np.testing.assert_allclose(
            robot_data._host_root_link_pos_w[:, 2],
            np.full(2, expected_root_z),
            rtol=0,
            atol=1e-6,
        )
        np.testing.assert_allclose(
            robot_data._host_joint_pos,
            np.tile(expected_joint_pos, (2, 1)),
            rtol=0,
            atol=1e-6,
        )
        env.close()
    finally:
        env.close()
        backend.close()


def test_torch_env_owns_substeps_autoreset_and_final_observation() -> None:
    env, backend = _make_env()
    assert isinstance(env, TorchEnv)
    initial_obs, initial_info = env.reset()
    assert env.state is not None
    assert isinstance(env.state, TorchEnvState)
    initial = env.state
    assert all(isinstance(value, torch.Tensor) for value in initial.obs.values())

    assert initial_obs["obs"].shape == (2, 2)
    assert initial_obs["critic"].shape == (2, 1)
    assert "log" in initial_info
    initial_steps = initial.info["steps"]
    assert initial_steps.dtype == torch.int64
    assert int(initial_steps.min()) >= 0
    assert int(initial_steps.max()) < 50
    assert torch.any(initial_steps != 0)
    torch.testing.assert_close(env.episode_length_buf, initial_steps)
    assert not hasattr(env, "_dr_manager")

    # A production initial state is staggered; align this deterministic
    # step-by-step lifecycle test with all rows at the beginning of an episode.
    env.set_episode_length_buf(torch.zeros(2, dtype=torch.int64))

    state = env.step(torch.tensor([[0.25], [0.5]], dtype=torch.float32))
    assert env.action_input_types and env.action_input_types[0] is torch.Tensor
    assert isinstance(env.action_manager.action, torch.Tensor)
    assert env.action_manager.action.dtype == torch.float32
    assert all(isinstance(value, torch.Tensor) for value in env.obs_buf.values())
    assert all(value.device == env.device for value in env.obs_buf.values())
    assert backend.tensor_controls[0].dtype == torch.float32
    assert backend.tensor_controls[0].device == env.device
    assert backend.pre_step_control is None
    assert backend.step_nsteps == [2]
    assert len(backend.applied_controls) == 2
    assert env.action_sim_steps == [2]
    np.testing.assert_allclose(backend.applied_controls[0][:, 0], [0.25, 0.5])
    np.testing.assert_allclose(backend.applied_controls[1][:, 0], [0.25, 0.5])
    torch.testing.assert_close(state.reward, torch.tensor([0.005, 0.01]))
    torch.testing.assert_close(state.terminated, torch.tensor([False, False]))
    torch.testing.assert_close(state.truncated, torch.tensor([False, False]))
    torch.testing.assert_close(state.info["steps"], torch.ones(2, dtype=torch.int64))

    state = env.step(torch.tensor([[0.25], [0.5]], dtype=torch.float32))
    torch.testing.assert_close(state.truncated, torch.tensor([True, True]))
    torch.testing.assert_close(state.terminated, torch.tensor([False, False]))
    torch.testing.assert_close(state.info["steps"], torch.zeros(2, dtype=torch.int64))
    torch.testing.assert_close(state.obs["obs"][:, 0], torch.zeros(2))
    assert state.final_observation is not None
    torch.testing.assert_close(state.final_observation["obs"][:, 0], torch.full((2,), 2.0))
    assert "final_observation" not in state.info
    assert "_final_observation" not in state.info
    assert state.info["log"]["Episode_Termination/time_out"] == 2
    assert not any(key.startswith("mba_") for key in state.info["timing"])

    pre_index = env.trace.index(("pre_reset", [True, True]))
    post_reset_index = env.trace.index(("post_reset", [True, True]), pre_index)
    post_step_index = len(env.trace) - 1
    assert env.trace[post_step_index] == "post_step"
    assert pre_index < post_reset_index < post_step_index


def test_update_state_publishes_drain_vs_host_termination_attribution() -> None:
    env, _backend = _make_env()
    env.reset()

    state = env.step(torch.tensor([[0.25], [0.5]], dtype=torch.float32))
    timing = state.info["timing"]

    assert timing["update_state_termination_ms"] >= 0.0
    assert timing["update_state_queue_drain_ms"] >= 0.0
    assert timing["update_state_termination_host_ms"] >= 0.0
    # The first reduction can drain queued backend work before the measured
    # termination call, so total wall attribution is never assumed to be host
    # term work.
    assert (
        timing["update_state_termination_host_ms"] <= timing["update_state_termination_ms"] + 1.0e-9
    )
    assert timing["update_state_reset_flags_ms"] >= 0.0
    assert timing["update_state_events_ms"] >= 0.0
    assert timing["update_state_read_boundary_ms"] >= 0.0
    assert timing["update_state_prelude_ms"] >= 0.0
    assert timing["update_state_timing_epilogue_ms"] >= 0.0
    assert timing["update_state_command_compute_ms"] >= 0.0
    assert timing["update_state_command_epilogue_ms"] >= 0.0
    assert timing["update_state_command_read_refresh_ms"] >= 0.0
    assert timing["update_state_command_post_compute_ms"] >= 0.0
    assert timing["update_state_publication_ms"] >= 0.0
    assert timing["update_state_state_replace_ms"] >= 0.0
    assert timing["update_state_child_sum_ms"] >= 0.0
    assert timing["update_state_command_preflight_ms"] >= 0.0
    observation_timing = {
        key: state.info["update_state_timing"][key]
        for key in (
            "update_state_observation_term_dispatch_ms",
            "update_state_observation_validation_ms",
            "update_state_observation_noise_ms",
            "update_state_observation_transform_ms",
            "update_state_observation_temporal_ms",
            "update_state_observation_concatenation_ms",
            "update_state_observation_boundary_ms",
            "update_state_observation_manager_residual_ms",
        )
    }
    assert all(value >= 0.0 for value in observation_timing.values())
    assert all(timing[key] == value for key, value in observation_timing.items())
    update_state_timing = state.info["update_state_timing"]
    if update_state_timing:
        assert set(update_state_timing) <= set(UPDATE_STATE_DETAIL_TIMING_KEYS)
        assert all(value >= 0.0 for value in update_state_timing.values())
    env.close()


def test_initial_episode_steps_are_seeded_and_staggered() -> None:
    first, _ = _make_env()
    second, _ = _make_env()
    try:
        for env in (first, second):
            initial_draws = [env._initial_episode_steps() for _ in range(2)]
            assert all(int(draw.min()) >= 0 and int(draw.max()) < 50 for draw in initial_draws)
            assert not torch.equal(initial_draws[0], initial_draws[1])
            initial = env.init_state()
            steps = initial.info["steps"]
            assert steps.dtype == torch.int64
            assert int(steps.min()) >= 0 and int(steps.max()) < 50
            torch.testing.assert_close(env.episode_length_buf, steps)
            assert torch.any(steps != 0)
        torch.testing.assert_close(first._initial_episode_steps(), second._initial_episode_steps())
        assert bool(torch.any(first._initial_episode_steps() != 0))
    finally:
        first.close()
        second.close()


def test_initial_episode_steps_fail_closed_without_a_staggering_range() -> None:
    cfg = _make_cfg()
    cfg.max_episode_seconds = cfg.ctrl_dt
    env, _ = _make_env(cfg)
    with pytest.raises(ValueError, match="max_episode_steps > 1"):
        env.init_state()


def test_reset_events_compose_then_commit_default_state_once() -> None:
    cfg = _make_cfg(include_optional_managers=False)
    cfg.events = {
        "default_first": EventTermCfg(func=mdp.reset_scene_to_default, mode="reset"),
        "joint_state": EventTermCfg(func=_write_reset_joint_state, mode="reset"),
        "observe_uncommitted": EventTermCfg(func=_observe_uncommitted_reset, mode="reset"),
    }
    cfg.scene.entities["robot"] = EntityCfg(
        joint_names=("joint",),
        actuator_names=("motor",),
    )
    backend = _ResetBackend(2)
    env = _TestEnv(cfg, cast(SimBackend, backend), 2)

    assert backend.default_qpos_calls == 0
    assert backend.init_qvel_calls == 0
    assert backend.joint_layout_calls == 1
    env.reset()

    assert env.trace[-1] == ("commit_count_during_event", 0)
    assert backend.default_qpos_calls == 1
    assert backend.init_qvel_calls == 1
    assert len(backend.set_state_calls) == 1
    committed_rows, qpos, qvel = backend.set_state_calls[0]
    np.testing.assert_array_equal(committed_rows, [0, 1])
    np.testing.assert_array_equal(
        qpos,
        [[0.0, 0.0, 0.5, 0.25], [0.0, 0.0, 0.5, 0.25]],
    )
    np.testing.assert_array_equal(qvel, [[0.0, 0.0, -0.5], [0.0, 0.0, -0.5]])
    assert backend.joint_layout_calls == 2

    env.reset(torch.tensor([False, True], dtype=torch.bool))
    assert ("commit_count_during_event", 1) in env.trace
    assert len(backend.set_state_calls) == 2
    np.testing.assert_array_equal(backend.set_state_calls[1][0], [1])
    assert backend.default_qpos_calls == 1
    assert backend.init_qvel_calls == 1
    assert backend.joint_layout_calls == 2


def test_reset_event_and_command_state_writes_share_one_commit() -> None:
    cfg = _make_cfg(include_optional_managers=False)
    cfg.events = {
        "default_first": EventTermCfg(func=mdp.reset_scene_to_default, mode="reset"),
        "joint_state": EventTermCfg(func=_write_reset_joint_state, mode="reset"),
    }
    cfg.commands = {"state_writer": _StateWritingCommandCfg(resampling_time_range=(1.0, 1.0))}
    cfg.scene.entities["robot"] = EntityCfg(
        joint_names=("joint",),
        actuator_names=("motor",),
    )
    backend = _ResetBackend(2)
    env = _TestEnv(cfg, cast(SimBackend, backend), 2)

    env.reset()

    assert len(backend.set_state_calls) == 1
    committed_rows, qpos, qvel = backend.set_state_calls[0]
    np.testing.assert_array_equal(committed_rows, [0, 1])
    np.testing.assert_array_equal(
        qpos,
        [[0.0, 0.0, 0.5, 0.75], [0.0, 0.0, 0.5, 0.75]],
    )
    np.testing.assert_array_equal(qvel, [[0.0, 0.0, -0.75], [0.0, 0.0, -0.75]])


def test_pure_reset_event_does_not_request_backend_state_capability() -> None:
    env, backend = _make_env()
    env.reset()
    env.reset(torch.tensor([True, False], dtype=torch.bool))

    assert isinstance(backend, _FakeBackend)
    assert backend.set_state_calls == []


def test_update_state_reuses_backend_state_once_and_refreshes_after_physics() -> None:
    env, backend = _make_state_env()
    env.init_state()

    before = backend.dof_pos_calls
    state = env.step(torch.zeros((2, 1), dtype=torch.float32))
    assert backend.dof_pos_calls == before + 1
    torch.testing.assert_close(state.reward, torch.ones(2))
    torch.testing.assert_close(state.obs["obs"], torch.ones((2, 1)))
    torch.testing.assert_close(state.obs["critic"], torch.ones((2, 1)))

    before = backend.dof_pos_calls
    state = env.step(torch.zeros((2, 1), dtype=torch.float32))
    assert backend.dof_pos_calls == before + 1
    torch.testing.assert_close(state.reward, torch.full((2,), 2.0))
    torch.testing.assert_close(state.obs["obs"], torch.full((2, 1), 2.0))


def test_update_state_invalidates_cache_after_command_set_state() -> None:
    env, backend = _make_state_env(
        commands={
            "writer": _StateWritingCommandCfg(resampling_time_range=(0.01, 0.01)),
        }
    )
    env.init_state()

    before = backend.dof_pos_calls
    state = env.step(torch.zeros((2, 1), dtype=torch.float32))

    # Physics advances the pre-command state to 1.75. The command then commits
    # 0.75 through set_state, so post-command observations must not reuse the
    # termination/reward snapshot.
    assert backend.dof_pos_calls == before + 2
    torch.testing.assert_close(state.reward, torch.full((2,), 1.75))
    torch.testing.assert_close(state.obs["obs"], torch.full((2, 1), 0.75))
    torch.testing.assert_close(state.obs["critic"], torch.full((2, 1), 0.75))


def test_update_state_invalidates_cache_after_runtime_event() -> None:
    env, backend = _make_state_env(
        events={
            "mutate": EventTermCfg(func=_mutate_joint_state_step_event, mode="step"),
        }
    )
    env.init_state()

    before = backend.dof_pos_calls
    state = env.step(torch.zeros((2, 1), dtype=torch.float32))

    assert backend.dof_pos_calls == before + 2
    torch.testing.assert_close(state.reward, torch.ones(2))
    torch.testing.assert_close(state.obs["obs"], torch.full((2, 1), 11.0))
    torch.testing.assert_close(state.obs["critic"], torch.full((2, 1), 11.0))


def test_partial_reset_observation_reads_post_set_state_rows() -> None:
    env, backend = _make_state_env(
        events={
            "joint_state": EventTermCfg(func=_write_reset_joint_state, mode="reset"),
        },
        terminations={"failure": TerminationTermCfg(func=_failure)},
    )
    env.init_state()

    state = env.step(torch.tensor([[1.0], [0.0]], dtype=torch.float32))

    torch.testing.assert_close(state.reward, torch.full((2,), 1.25))
    torch.testing.assert_close(state.terminated, torch.tensor([True, False]))
    torch.testing.assert_close(state.obs["obs"], torch.tensor([[0.25], [1.25]]))
    torch.testing.assert_close(state.obs["critic"], torch.tensor([[0.25], [1.25]]))
    assert state.final_observation is not None
    torch.testing.assert_close(state.final_observation["obs"][0], torch.tensor([1.25]))
    np.testing.assert_array_equal(backend.dof_pos[:, 0], [0.25, 1.25])


def test_scene_read_plan_refreshes_once_per_phase_and_after_mutations() -> None:
    cfg = _make_cfg(include_optional_managers=False)
    cfg.actions = {
        "tilt": _TensorBodyActionCfg(
            entity_name="robot",
            top_body_name="platform",
            ball_body_name="ball",
        )
    }
    cfg.scene.entities["robot"] = EntityCfg(
        root_body_name="base",
        joint_names=("joint",),
        body_names=("platform", "ball"),
        actuator_names=("motor",),
    )
    backend = _ScenePlanBackend(2)
    env = _TestEnv(cfg, cast(SimBackend, backend), 2)
    try:
        plan = env.scene._tensor_read_plan
        assert plan is not None

        obs, _ = env.reset()
        assert backend.full_reads == 0
        assert backend.selected_reads == 0
        assert obs["obs"].shape == (2, 2)

        state = env.step(torch.zeros((2, 1), dtype=torch.float32))
        # This action term declares state feedback, so the action phase packs
        # body reads once. update_state then drops that stale packet and
        # republishes one post-physics packet shared by all Manager terms.
        assert backend.full_reads == 0
        assert state.obs["obs"].shape == (2, 2)

        cfg_events = env.cfg.events
        assert isinstance(cfg_events, dict)
        cfg_events["mutate"] = EventTermCfg(func=_mutate_backend_tensor_state, mode="step")
        state = env.step(torch.zeros((2, 1), dtype=torch.float32))
        # Each control step owns two phases: pre-action and post-physics. The
        # bounded mutation adds one refresh before the legacy read phase.
        assert backend.full_reads == 0
        assert state.obs["obs"].shape == (2, 2)

        plan.close()
        env.scene._tensor_read_plan = None
        state = env.step(torch.zeros((2, 1), dtype=torch.float32))
        assert state.obs["obs"].shape == (2, 2)
    finally:
        if env.scene._tensor_read_plan is not None:
            env.scene._tensor_read_plan.close()
            env.scene._tensor_read_plan = None
        env.close()


def test_scene_read_plan_compiles_named_sensor_observation_requests() -> None:
    cfg = _make_cfg(include_optional_managers=False)
    cfg.scene.entities["robot"] = EntityCfg(
        root_body_name="base",
        joint_names=("joint",),
        body_names=("platform", "ball"),
        actuator_names=("motor",),
    )
    cfg.actions = {
        "tilt": _TensorBodyActionCfg(
            entity_name="robot",
            top_body_name="platform",
            ball_body_name="ball",
        )
    }
    cfg.observations = {
        "actor": ObservationGroupCfg(
            terms={
                "sensor": ObservationTermCfg(func=_NamedSensorTensorObservation),
            }
        ),
        "value": ObservationGroupCfg(
            terms={"critic": ObservationTermCfg(func=_episode_step_observation)}
        ),
    }
    cfg.critic_observation_group = "value"
    backend = _ScenePlanBackend(2)
    backend.sensors["track_pos_w_ball"].add_(5.0)
    env = _TestEnv(cfg, cast(SimBackend, backend), 2)
    try:
        plan = env.scene._tensor_read_plan
        assert plan is not None
        assert plan.host_plan is None
        assert "track_pos_w_ball" in plan.sensor_names["robot"]
        assert set(plan.sensor_names["robot"]) == {"track_pos_w_ball"}

        obs, _ = env.reset()
        torch.testing.assert_close(
            obs["obs"],
            torch.tensor([[5.1, 5.2, 6.2], [5.3, 5.4, 6.2]], dtype=torch.float32),
        )
        assert backend.selected_reads == 0
    finally:
        if env.scene._tensor_read_plan is not None:
            env.scene._tensor_read_plan.close()
            env.scene._tensor_read_plan = None
        env.close()


def test_generic_root_and_joint_terms_compile_packed_views() -> None:
    cfg = _make_cfg(include_optional_managers=False)
    cfg.scene.entities["robot"] = EntityCfg(
        root_body_name="base",
        joint_names=("joint",),
        body_names=("base",),
        actuator_names=("motor",),
    )
    cfg.observations = {
        "actor": ObservationGroupCfg(
            terms={
                "gravity": ObservationTermCfg(
                    func=mdp.projected_gravity_from_sensor,
                    params={"sensor_name": "track_pos_w_base", "entity_name": "robot"},
                ),
                "joint": ObservationTermCfg(func=_episode_step_observation),
            }
        ),
        "value": ObservationGroupCfg(
            terms={"critic": ObservationTermCfg(func=_episode_step_observation)}
        ),
    }
    cfg.rewards = {
        "height": RewardTermCfg(func=mdp.root_height, weight=1.0),
        "velocity": RewardTermCfg(func=mdp.joint_vel_l2, weight=-0.1),
    }
    cfg.terminations = {"low": TerminationTermCfg(func=mdp.root_height_below_minimum, params={})}
    cfg.critic_observation_group = "value"
    backend = _ScenePlanBackend(2)
    env = _TestEnv(cfg, cast(SimBackend, backend), 2)
    try:
        plan = env.scene._tensor_read_plan
        assert plan is not None
        assert plan.body_names["robot"] == ("base",)
        assert "robot" in plan.entities
        plan.refresh()
        view = plan.body_tensor_view("robot", ("base",))
        assert view.pos_w.shape == (2, 1, 3)
        joints = plan.joint_tensor_view("robot")
        assert joints.joint_pos.shape == (2, 1)
    finally:
        if env.scene._tensor_read_plan is not None:
            env.scene._tensor_read_plan.close()
            env.scene._tensor_read_plan = None
        env.close()


def test_generic_root_term_requires_declared_root_and_body_namespace() -> None:
    cfg = _make_cfg(include_optional_managers=False)
    cfg.scene.entities["robot"] = EntityCfg(
        root_body_name="base",
        joint_names=("joint",),
        actuator_names=("motor",),
    )
    cfg.terminations = {
        "low": TerminationTermCfg(
            func=mdp.root_height_below_minimum, params={"minimum_height": 0.1}
        )
    }

    with pytest.raises(NotImplementedError, match="capability 'body'.*not declared"):
        _TestEnv(cfg, cast(SimBackend, _ScenePlanBackend(2)), 2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA packed MDP placement test")
def test_device_resident_generic_packed_terms_keep_carriers_on_device() -> None:
    cfg = _make_cfg(include_optional_managers=False)
    cfg.scene.entities["robot"] = EntityCfg(
        root_body_name="base",
        joint_names=("joint",),
        body_names=("base",),
        actuator_names=("motor",),
    )
    cfg.observations = {
        "actor": ObservationGroupCfg(
            terms={"policy": ObservationTermCfg(func=_tensor_runtime_policy_obs)}
        ),
        "value": ObservationGroupCfg(
            terms={"critic": ObservationTermCfg(func=_tensor_runtime_critic_obs)}
        ),
    }
    cfg.critic_observation_group = "value"
    cfg.rewards = {
        "height": RewardTermCfg(func=mdp.root_height, weight=1.0),
        "velocity": RewardTermCfg(func=mdp.joint_vel_l2, weight=-0.1),
    }
    cfg.terminations = {
        "low": TerminationTermCfg(
            func=mdp.root_height_below_minimum, params={"minimum_height": -1.0}
        )
    }
    env = _TestEnv(cfg, cast(SimBackend, _DeviceResidentScenePlanBackend(2)), 2)
    try:
        assert env.device.type == "cuda"
        assert env.scene._tensor_read_plan is not None
        assert env.scene._tensor_read_plan.device == env.device
        state = env.step(torch.zeros((2, 1), dtype=torch.float32, device=env.device))
        assert state.obs["obs"].device == env.device
        assert state.obs["critic"].device == env.device
        assert state.reward.device == env.device
        assert state.terminated.device == env.device
    finally:
        if env.scene._tensor_read_plan is not None:
            env.scene._tensor_read_plan.close()
            env.scene._tensor_read_plan = None
        env.close()


def test_scene_read_plan_compiles_class_observation_body_requests() -> None:
    cfg = _make_cfg(include_optional_managers=False)
    cfg.scene.entities["robot"] = EntityCfg(
        root_body_name="base",
        joint_names=("joint",),
        body_names=("platform", "ball"),
        actuator_names=("motor",),
    )
    cfg.commands = {"target": _CommandCfg(resampling_time_range=(1.0, 1.0))}
    cfg.observations = {
        "actor": ObservationGroupCfg(
            terms={
                "body": ObservationTermCfg(
                    func=_CommandBodyObservation, params={"command_name": "target"}
                )
            }
        ),
        "value": ObservationGroupCfg(
            terms={"critic": ObservationTermCfg(func=_episode_step_observation)}
        ),
    }
    cfg.critic_observation_group = "value"
    backend = _ScenePlanBackend(2)
    env = _TestEnv(cfg, cast(SimBackend, backend), 2)
    try:
        plan = env.scene._tensor_read_plan
        assert plan is not None
        assert plan.host_plan is None
        assert plan.body_names["robot"] == ("platform", "ball")
        assert plan.sensor_names["robot"] == ()

        obs, _ = env.reset()

        term = env.observation_manager.get_term_cfg("actor", "body").func
        assert isinstance(term, _CommandBodyObservation)
        assert term.entity_name == "robot"
        assert term.tensor_body_names == ("platform", "ball")
        assert obs["obs"].shape == (2, 1)
    finally:
        if env.scene._tensor_read_plan is not None:
            env.scene._tensor_read_plan.close()
            env.scene._tensor_read_plan = None
        env.close()


def test_scene_read_plan_compiles_command_body_requests() -> None:
    cfg = _make_cfg(include_optional_managers=False)
    cfg.scene.entities["robot"] = EntityCfg(
        root_body_name="base",
        joint_names=("joint",),
        body_names=("platform", "ball"),
        actuator_names=("motor",),
    )
    cfg.commands = {"target": _TensorBodyCommandCfg(resampling_time_range=(1.0, 1.0))}
    cfg.observations = {
        "actor": ObservationGroupCfg(
            terms={
                "body": ObservationTermCfg(
                    func=_CommandBodyObservation, params={"command_name": "target"}
                )
            }
        ),
        "value": ObservationGroupCfg(
            terms={"critic": ObservationTermCfg(func=_episode_step_observation)}
        ),
    }
    cfg.critic_observation_group = "value"
    backend = _ScenePlanBackend(2)
    env = _TestEnv(cfg, cast(SimBackend, backend), 2)
    try:
        plan = env.scene._tensor_read_plan
        assert plan is not None
        assert plan.host_plan is None
        assert plan.body_names["robot"] == ("platform", "ball")
        assert plan.sensor_names["robot"] == ()

        command = env.command_manager.get_term("target")
        assert command.tensor_body_names == ("platform", "ball")
        plan.refresh()
        view = plan.body_tensor_view("robot", command.tensor_body_names)
        assert view.pos_w.shape == (2, 2, 3)
    finally:
        if env.scene._tensor_read_plan is not None:
            env.scene._tensor_read_plan.close()
            env.scene._tensor_read_plan = None
        env.close()


def test_command_packed_sensor_read_negotiates_backend_local_alias() -> None:
    cfg = _make_cfg(include_optional_managers=False)
    cfg.scene.entities["robot"] = EntityCfg(
        root_body_name="base",
        joint_names=("joint",),
        body_names=("platform", "ball"),
        actuator_names=("motor",),
    )
    cfg.commands = {"twist": _AliasedSensorCommandCfg(resampling_time_range=(1.0, 1.0))}
    backend = _ScenePlanBackend(2)
    backend.packed_reset = False
    backend.sensors.update(
        {
            "local_linvel": torch.tensor([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], dtype=torch.float32),
            "gyro": torch.tensor([[-0.1, -0.2, -0.3], [-0.4, -0.5, -0.6]], dtype=torch.float32),
        }
    )
    with pytest.raises(NotImplementedError, match="packed_reset capability"):
        _TestEnv(cfg, cast(SimBackend, backend), 2)


def test_scene_read_plan_pairs_reset_with_selected_packed_transfer() -> None:
    cfg = _make_cfg(include_optional_managers=False)
    cfg.actions = {
        "tilt": _TensorBodyActionCfg(
            entity_name="robot",
            top_body_name="platform",
            ball_body_name="ball",
        )
    }
    cfg.scene.entities["robot"] = EntityCfg(
        root_body_name="base",
        joint_names=("joint",),
        body_names=("platform", "ball"),
        actuator_names=("motor",),
    )
    cfg.events = {"joint_state": EventTermCfg(func=_write_reset_joint_state, mode="reset")}
    backend = _ScenePlanBackend(2)
    env = _TestEnv(cfg, cast(SimBackend, backend), 2)
    try:
        plan = env.scene._tensor_read_plan
        assert plan is not None
        assert plan.host_plan is None

        env.reset()

        assert len(backend.set_state_calls) == 1
        committed_rows, qpos, qvel = backend.set_state_calls[0]
        np.testing.assert_array_equal(committed_rows, [0, 1])
        torch.testing.assert_close(
            torch.as_tensor(qpos),
            torch.tensor([[0.0, 0.0, 0.5, 0.25], [0.0, 0.0, 0.5, 0.25]], dtype=torch.float32),
        )
        torch.testing.assert_close(
            torch.as_tensor(qvel),
            torch.tensor(
                [[0.0, 0.0, -0.5], [0.0, 0.0, -0.5]], dtype=torch.float32, device=env.device
            ),
        )
        assert backend.full_reads == 0
        assert backend.selected_reads == 0

        env.reset(torch.tensor([False, True], dtype=torch.bool))
        assert len(backend.set_state_calls) == 2
        np.testing.assert_array_equal(backend.set_state_calls[1][0], [1])
        assert backend.full_reads == 0
        assert backend.selected_reads == 0
    finally:
        if env.scene._tensor_read_plan is not None:
            env.scene._tensor_read_plan.close()
            env.scene._tensor_read_plan = None
        env.close()


def test_reset_randomization_commits_through_packed_boundary() -> None:
    cfg = _make_cfg(include_optional_managers=False)
    cfg.observations = {
        "actor": ObservationGroupCfg(
            terms={"policy": ObservationTermCfg(func=_tensor_runtime_policy_obs)}
        ),
        "value": ObservationGroupCfg(
            terms={"critic": ObservationTermCfg(func=_tensor_runtime_critic_obs)}
        ),
    }
    cfg.scene.entities["robot"] = EntityCfg(
        root_body_name="base",
        joint_names=("joint",),
        body_names=("base", "platform"),
        geom_names=("base_geom",),
        actuator_names=("motor",),
    )
    cfg.actions = {
        "tilt": _TensorBodyActionCfg(
            entity_name="robot",
            top_body_name="platform",
            ball_body_name="base",
        )
    }
    cfg.commands = {"target": _TensorStateWritingCommandCfg(resampling_time_range=(1.0, 1.0))}
    cfg.events = {
        "body_mass": EventTermCfg(
            func=mdp.randomize_rigid_body_mass,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=("base",)),
                "mass_distribution_params": (1.5, 1.5),
                "operation": "scale",
                "recompute_inertia": False,
            },
        ),
        "body_com": EventTermCfg(
            func=mdp.randomize_rigid_body_com,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=("base",)),
                "com_range": {"x": (0.1, 0.1), "z": (-0.2, -0.2)},
            },
        ),
        "foot_friction": EventTermCfg(
            func=mdp.geom_friction,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("robot", geom_names=("base_geom",)),
                "ranges": (0.75, 0.75),
                "operation": "abs",
            },
        ),
        "pd_gains": EventTermCfg(
            func=mdp.pd_gains,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("robot", actuator_names=("motor",)),
                "kp_range": (30.0, 30.0),
                "kd_range": (3.0, 3.0),
                "operation": "abs",
            },
        ),
    }
    backend = _ScenePlanBackend(2)
    env = _TestEnv(cfg, cast(SimBackend, backend), 2)
    try:
        env.reset()

        assert len(backend.set_state_calls) == 1
        committed_rows, _, _ = backend.set_state_calls[0]
        np.testing.assert_array_equal(committed_rows, [0, 1])
        payload = backend.randomization_calls[0]
        assert payload is not None
        np.testing.assert_allclose(payload.body_mass.numpy(), [[15.0], [15.0]], rtol=1e-6)
        np.testing.assert_allclose(payload.body_ipos.numpy(), [[[0.1, 0.0, -0.2]]] * 2, atol=1e-7)
        np.testing.assert_allclose(
            payload.geom_friction.numpy(), [[[0.75, 0.02, 0.002]]] * 2, rtol=1e-6
        )
        np.testing.assert_allclose(payload.kp.numpy(), [[30.0], [30.0]], rtol=1e-6)
        np.testing.assert_allclose(payload.kd.numpy(), [[3.0], [3.0]], rtol=1e-6)

        # A partial mask reset recomposes only the selected rows in one commit.
        env.reset(torch.tensor([False, True], dtype=torch.bool))
        assert len(backend.set_state_calls) == 2
        np.testing.assert_array_equal(backend.set_state_calls[1][0], [1])
        replay = backend.randomization_calls[1]
        assert replay is not None
        np.testing.assert_allclose(replay.body_mass.numpy(), [[15.0]], rtol=1e-6)
    finally:
        if env.scene._tensor_read_plan is not None:
            env.scene._tensor_read_plan.close()
            env.scene._tensor_read_plan = None
        env.close()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a local CUDA device")
def test_device_resident_reset_dispatches_selected_tensor_commit() -> None:
    cfg = _make_cfg(include_optional_managers=False)
    cfg.observations = {
        "actor": ObservationGroupCfg(
            terms={"policy": ObservationTermCfg(func=_tensor_runtime_policy_obs)}
        ),
        "value": ObservationGroupCfg(
            terms={"critic": ObservationTermCfg(func=_tensor_runtime_critic_obs)}
        ),
    }
    cfg.actions = {
        "tilt": _TensorBodyActionCfg(
            entity_name="robot",
            top_body_name="platform",
            ball_body_name="ball",
        )
    }
    cfg.scene.entities["robot"] = EntityCfg(
        root_body_name="base",
        joint_names=("joint",),
        body_names=("platform", "ball"),
        actuator_names=("motor",),
    )
    cfg.events = {"joint_state": EventTermCfg(func=_write_reset_joint_state, mode="reset")}
    backend = _DeviceResidentScenePlanBackend(2)
    env = _TestEnv(cfg, cast(SimBackend, backend), 2)
    try:
        plan = env.scene._tensor_read_plan
        assert plan is not None
        assert plan.host_plan is None

        env.reset()

        assert backend.set_state_calls == []
        assert backend.reset_calls == []
        assert len(backend.tensor_reset_calls) == 1
        committed_rows, qpos, qvel = backend.tensor_reset_calls[0]
        torch.testing.assert_close(
            committed_rows, torch.tensor([0, 1], dtype=torch.int64, device=env.device)
        )
        torch.testing.assert_close(
            qpos,
            torch.tensor(
                [[0.0, 0.0, 0.5, 0.25], [0.0, 0.0, 0.5, 0.25]],
                dtype=torch.float32,
                device=env.device,
            ),
        )
        torch.testing.assert_close(
            qvel,
            torch.tensor(
                [[0.0, 0.0, -0.5], [0.0, 0.0, -0.5]], dtype=torch.float32, device=env.device
            ),
        )

        # Production initial counters are staggered; this deterministic
        # device-resident reset sequence intentionally starts both rows at the
        # beginning of an episode so exactly one autoreset precedes the partial
        # reset below.
        env.set_episode_length_buf(torch.zeros(2, dtype=torch.int64, device=env.device))

        env.step(torch.zeros((2, 1), dtype=torch.float32, device=env.device))
        env.reset(torch.tensor([False, True], dtype=torch.bool, device=env.device))
        assert len(backend.tensor_reset_calls) == 2
        torch.testing.assert_close(
            backend.tensor_reset_calls[1][0],
            torch.tensor([1], dtype=torch.int64, device=env.device),
        )
        assert backend.set_state_calls == []
        assert backend.reset_calls == []
    finally:
        if env.scene._tensor_read_plan is not None:
            env.scene._tensor_read_plan.close()
            env.scene._tensor_read_plan = None
        env.close()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA device reset test")
def test_device_resident_tensor_root_event_commits_selected_rows_once() -> None:
    cfg = _make_cfg(include_optional_managers=False)
    cfg.observations = {
        "actor": ObservationGroupCfg(
            terms={"policy": ObservationTermCfg(func=_tensor_runtime_policy_obs)}
        ),
        "value": ObservationGroupCfg(
            terms={"critic": ObservationTermCfg(func=_tensor_runtime_critic_obs)}
        ),
    }
    cfg.scene.entities["robot"] = EntityCfg(
        root_body_name="base",
        joint_names=("joint",),
        body_names=("platform", "ball"),
        actuator_names=("motor",),
    )
    cfg.actions = {
        "tilt": _TensorBodyActionCfg(
            entity_name="robot",
            top_body_name="platform",
            ball_body_name="ball",
        )
    }
    cfg.events = {
        "reset_root_state_uniform": EventTermCfg(
            func=mdp.reset_root_state_uniform_tensor,
            mode="reset",
            params={
                "pose_range": {
                    "x": (0.1, 0.1),
                    "y": (0.0, 0.0),
                    "z": (0.0, 0.0),
                    "roll": (0.0, 0.0),
                    "pitch": (0.0, 0.0),
                    "yaw": (0.0, 0.0),
                },
                "velocity_range": {
                    "x": (0.2, 0.2),
                    "y": (0.0, 0.0),
                    "z": (0.0, 0.0),
                    "roll": (0.0, 0.0),
                    "pitch": (0.0, 0.0),
                    "yaw": (0.0, 0.0),
                },
            },
        )
    }

    class _TensorRootBackend(_DeviceResidentScenePlanBackend):
        nq = 10
        nv = 9

        def get_default_qpos(self) -> np.ndarray:
            return np.array([0.0, 0.0, 0.5, 1.0, 0.0, 0.0, 0.0, 0.25, 0.0, 0.0], dtype=np.float32)

        def get_init_qvel(self) -> np.ndarray:
            return np.zeros(9, dtype=np.float32)

        def get_root_state_layout(self, root_body_name: str) -> Any:
            from unisim.backend.base import BackendRootStateLayout

            assert root_body_name == "base"
            return BackendRootStateLayout(
                qpos_indices=(0, 1, 2, 3, 4, 5, 6), qvel_indices=(0, 1, 2, 3, 4, 5)
            )

        def get_state_views(self, fields, device=None) -> dict[str, torch.Tensor]:
            if self.qpos.shape[1] != self.nq:
                self.qpos = torch.zeros((self.num_envs, self.nq), device=self.qpos.device)
            if self.qvel.shape[1] != self.nv:
                self.qvel = torch.zeros((self.num_envs, self.nv), device=self.qvel.device)
            target = torch.device(device) if device is not None else self.qpos.device
            return {
                "qpos": self.qpos.to(device=target),
                "qvel": self.qvel.to(device=target),
            }

    backend = _TensorRootBackend(2)
    env = _TestEnv(cfg, cast(SimBackend, backend), 2)
    try:
        assert env.scene._tensor_read_plan is not None
        env.reset()
        assert len(backend.tensor_reset_calls) == 1
        committed_rows, qpos, qvel = backend.tensor_reset_calls[0]
        torch.testing.assert_close(
            committed_rows, torch.tensor([0, 1], dtype=torch.int64, device=env.device)
        )
        torch.testing.assert_close(
            qpos[:, 0], torch.full((2,), 0.1, device=env.device), rtol=0, atol=1e-6
        )
        torch.testing.assert_close(
            qvel[:, 0], torch.full((2,), 0.2, device=env.device), rtol=0, atol=1e-6
        )
        # Unselected joint/default columns stay at their canonical reset values.
        torch.testing.assert_close(
            qpos[:, 2:8],
            torch.tensor([[0.5, 1.0, 0.0, 0.0, 0.0, 0.25]] * 2, device=env.device),
        )
    finally:
        if env.scene._tensor_read_plan is not None:
            env.scene._tensor_read_plan.close()
            env.scene._tensor_read_plan = None
        env.close()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA device reset test")
def test_device_resident_tensor_joint_event_commits_selected_columns_once() -> None:
    cfg = _make_cfg(include_optional_managers=False)
    cfg.observations = {
        "actor": ObservationGroupCfg(
            terms={"policy": ObservationTermCfg(func=_tensor_runtime_policy_obs)}
        ),
        "value": ObservationGroupCfg(
            terms={"critic": ObservationTermCfg(func=_tensor_runtime_critic_obs)}
        ),
    }
    cfg.scene.entities["robot"] = EntityCfg(
        root_body_name="base",
        joint_names=("joint",),
        body_names=("platform", "ball"),
        actuator_names=("motor",),
    )
    cfg.actions = {
        "tilt": _TensorBodyActionCfg(
            entity_name="robot",
            top_body_name="platform",
            ball_body_name="ball",
        )
    }
    cfg.events = {
        "reset_joint_state": EventTermCfg(
            func=_reset_joint_state_tensor_mask,
            mode="reset",
            params={"position": 0.25, "velocity": -0.5},
        )
    }

    class _TensorJointBackend(_DeviceResidentScenePlanBackend):
        nq = 10
        nv = 9

        def get_default_qpos(self) -> np.ndarray:
            return np.array([0.0, 0.0, 0.5, 1.0, 0.0, 0.0, 0.0, 0.25, 0.0, 0.0], dtype=np.float32)

        def get_init_qvel(self) -> np.ndarray:
            return np.zeros(9, dtype=np.float32)

        def get_joint_state_qpos_indices(self, joint_names):
            assert joint_names == ("joint",)
            return np.array([7], dtype=np.int32)

        def get_joint_state_qvel_indices(self, joint_names):
            assert joint_names == ("joint",)
            return np.array([6], dtype=np.int32)

        def get_state_views(self, fields, device=None) -> dict[str, torch.Tensor]:
            target = torch.device(device) if device is not None else torch.device("cuda")
            return {
                "qpos": torch.zeros((self.num_envs, self.nq), dtype=torch.float32, device=target),
                "qvel": torch.zeros((self.num_envs, self.nv), dtype=torch.float32, device=target),
            }

    backend = _TensorJointBackend(2)
    env = _TestEnv(cfg, cast(SimBackend, backend), 2)
    try:
        assert env.scene._tensor_read_plan is not None
        env.reset()
        assert len(backend.tensor_reset_calls) == 1
        committed_rows, qpos, qvel = backend.tensor_reset_calls[0]
        torch.testing.assert_close(
            committed_rows, torch.tensor([0, 1], dtype=torch.int64, device=env.device)
        )
        torch.testing.assert_close(
            qpos[:, 7], torch.full((2,), 0.25, device=env.device), rtol=0, atol=1e-6
        )
        torch.testing.assert_close(
            qvel[:, 6], torch.full((2,), -0.5, device=env.device), rtol=0, atol=1e-6
        )
        # Unselected root/default columns stay at their canonical reset values.
        torch.testing.assert_close(
            qpos[:, :7],
            torch.tensor([[0.0, 0.0, 0.5, 1.0, 0.0, 0.0, 0.0]] * 2, device=env.device),
        )
    finally:
        if env.scene._tensor_read_plan is not None:
            env.scene._tensor_read_plan.close()
            env.scene._tensor_read_plan = None
        env.close()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA device reset test")
def test_device_resident_tensor_command_reset_commits_selected_rows_once() -> None:
    cfg = _make_cfg(include_optional_managers=False)
    cfg.observations = {
        "actor": ObservationGroupCfg(
            terms={"policy": ObservationTermCfg(func=_tensor_runtime_policy_obs)}
        ),
        "value": ObservationGroupCfg(
            terms={"critic": ObservationTermCfg(func=_tensor_runtime_critic_obs)}
        ),
    }
    cfg.scene.entities["robot"] = EntityCfg(
        root_body_name="base",
        joint_names=("joint",),
        body_names=("platform", "ball"),
        actuator_names=("motor",),
    )
    cfg.actions = {
        "tilt": _TensorBodyActionCfg(
            entity_name="robot",
            top_body_name="platform",
            ball_body_name="ball",
        )
    }
    cfg.commands = {"target": _TensorStateWritingCommandCfg(resampling_time_range=(1.0, 1.0))}

    class _TensorCommandBackend(_DeviceResidentScenePlanBackend):
        nq = 5
        nv = 4

        def get_default_qpos(self) -> np.ndarray:
            return np.array([0.0, 0.0, 0.5, 1.0, 0.25], dtype=np.float32)

        def get_init_qvel(self) -> np.ndarray:
            return np.zeros(4, dtype=np.float32)

        def get_joint_state_qpos_indices(self, joint_names):
            assert joint_names == ("joint",)
            return np.array([4], dtype=np.int32)

        def get_joint_state_qvel_indices(self, joint_names):
            assert joint_names == ("joint",)
            return np.array([3], dtype=np.int32)

        def get_state_views(self, fields, device=None) -> dict[str, torch.Tensor]:
            target = torch.device(device) if device is not None else torch.device("cuda")
            return {
                "qpos": torch.zeros((self.num_envs, self.nq), dtype=torch.float32, device=target),
                "qvel": torch.zeros((self.num_envs, self.nv), dtype=torch.float32, device=target),
            }

    backend = _TensorCommandBackend(2)
    env = _TestEnv(cfg, cast(SimBackend, backend), 2)
    try:
        assert env.scene._tensor_read_plan is not None
        env.reset()
        assert len(backend.tensor_reset_calls) == 1
        committed_rows, qpos, qvel = backend.tensor_reset_calls[0]
        torch.testing.assert_close(
            committed_rows, torch.tensor([0, 1], dtype=torch.int64, device=env.device)
        )
        torch.testing.assert_close(qpos[:, 4], torch.full((2,), 0.75, device=env.device))
        torch.testing.assert_close(qvel[:, 3], torch.full((2,), -0.75, device=env.device))
        torch.testing.assert_close(
            qpos[:, :4],
            torch.tensor([[0.0, 0.0, 0.5, 1.0]] * 2, device=env.device),
        )
        command = env.command_manager.get_command("target")
        torch.testing.assert_close(command, torch.full((2, 1), 0.75, device=env.device))
    finally:
        if env.scene._tensor_read_plan is not None:
            env.scene._tensor_read_plan.close()
            env.scene._tensor_read_plan = None
        env.close()


def test_partial_reset_preserves_other_env_counter_and_terminal_obs() -> None:
    env, _ = _make_env()
    env.init_state()
    env.set_episode_length_buf(torch.zeros(2, dtype=torch.int64))

    state = env.step(torch.tensor([[1.0], [0.0]], dtype=torch.float32))

    torch.testing.assert_close(state.terminated, torch.tensor([True, False]))
    torch.testing.assert_close(state.truncated, torch.tensor([False, False]))
    torch.testing.assert_close(state.info["steps"], torch.tensor([0, 1], dtype=torch.int64))
    torch.testing.assert_close(state.obs["obs"][:, 0], torch.tensor([0.0, 1.0]))
    assert state.final_observation is not None
    torch.testing.assert_close(state.final_observation["obs"][0], torch.tensor([1.0, 1.0]))
    assert "_final_observation" not in state.info
    assert ("event", [True, False]) in env.trace


def test_finite_horizon_maps_time_out_to_terminated() -> None:
    env, _ = _make_env(_make_cfg(finite_horizon=True))
    env.init_state()
    env.set_episode_length_buf(torch.zeros(2, dtype=torch.int64))
    env.step(torch.zeros((2, 1), dtype=torch.float32))
    state = env.step(torch.zeros((2, 1), dtype=torch.float32))
    torch.testing.assert_close(state.terminated, torch.tensor([True, True]))
    torch.testing.assert_close(state.truncated, torch.tensor([False, False]))


def test_manual_reset_is_required_when_autoreset_is_disabled() -> None:
    env, _ = _make_env(_make_cfg(auto_reset=False))
    env.init_state()
    env.set_episode_length_buf(torch.zeros(2, dtype=torch.int64))
    state = env.step(torch.tensor([[1.0], [0.0]], dtype=torch.float32))
    torch.testing.assert_close(state.info["steps"], torch.ones(2, dtype=torch.int64))
    with pytest.raises(RuntimeError, match="must be reset"):
        env.step(torch.zeros((2, 1), dtype=torch.float32))

    reset_obs, info = env.reset(torch.tensor([True, False], dtype=torch.bool))
    assert reset_obs["obs"].shape == (2, 2)
    assert "log" in info
    assert not state.terminated[0]
    env.step(torch.zeros((2, 1), dtype=torch.float32))


def test_reset_seed_updates_the_shared_generator_in_place() -> None:
    cfg = _make_cfg(include_optional_managers=False)
    cfg.events = {}
    env, _ = _make_env(cfg)
    env.init_state()
    generator = env.torch_rng

    env.reset(seed=123, reset_mask=torch.tensor([True, False], dtype=torch.bool))

    assert env.torch_rng is generator
    expected = TorchManagerRng.seeded(123).random(1)
    torch.testing.assert_close(env.torch_rng.random(1), expected)


def test_per_substep_metrics_fail_without_post_substep_backend_hook() -> None:
    cfg = _make_cfg(
        sim_substeps=2,
        metrics={"energy": MetricsTermCfg(func=_metric, per_substep=True)},
    )
    with pytest.raises(NotImplementedError, match="post-physics per-substep metrics.*fake"):
        _make_env(cfg)


def test_multisubstep_state_independent_action_skips_callback() -> None:
    env, backend = _make_env(reject_pre_step=True)
    try:
        assert backend.pre_step_control is None
        assert not env.action_manager.requires_substep_state_feedback
    finally:
        env.close()


def test_multisubstep_state_feedback_action_uses_callback() -> None:
    cfg = _make_cfg(
        actions={"drive": _FeedbackDriveCfg(entity_name="robot")},
    )
    with pytest.raises(
        NotImplementedError,
        match="Torch lifecycle P1 does not support state-feedback actions",
    ):
        _make_env(cfg)


def test_multisubstep_state_feedback_action_fails_when_backend_rejects_callback() -> None:
    cfg = _make_cfg(
        actions={"drive": _FeedbackDriveCfg(entity_name="robot")},
    )
    with pytest.raises(
        NotImplementedError,
        match="Torch lifecycle P1 does not support state-feedback actions",
    ):
        _make_env(cfg, reject_pre_step=True)


def test_single_substep_feedback_action_does_not_require_callback() -> None:
    cfg = _make_cfg(
        sim_substeps=1,
        actions={"drive": _FeedbackDriveCfg(entity_name="robot")},
    )
    env, backend = _make_env(cfg, reject_pre_step=True)
    try:
        env.init_state()
        env.step(torch.tensor([[0.25], [0.5]], dtype=torch.float32))

        assert backend.pre_step_control is None
        assert backend.step_nsteps == [1]
        assert isinstance(backend.tensor_controls[0], torch.Tensor)
        assert env.action_sim_steps == [1]
    finally:
        env.close()


def test_empty_action_manager_does_not_require_callback() -> None:
    observations = {
        "actor": ObservationGroupCfg(
            terms={"episode_step": ObservationTermCfg(func=_episode_step_observation)}
        ),
        "value": ObservationGroupCfg(
            terms={"episode_step": ObservationTermCfg(func=_episode_step_observation)}
        ),
    }
    env, backend = _make_env(
        _make_cfg(actions={}, observations=observations),
        reject_pre_step=True,
    )
    try:
        assert env.action_manager.active_terms == []
        assert not env.action_manager.requires_substep_state_feedback
        assert backend.pre_step_control is None
    finally:
        env.close()


def test_state_feedback_action_error_propagates_from_callback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = _make_cfg(
        sim_substeps=1,
        actions={"drive": _FeedbackDriveCfg(entity_name="robot")},
    )
    env, _ = _make_env(cfg)
    env.init_state()
    term = env.action_manager.get_term("drive")

    def fail() -> None:
        raise ValueError("feedback failed")

    monkeypatch.setattr(term, "apply_actions", fail)
    try:
        with pytest.raises(ValueError, match="ActionManager term 'drive'.*feedback failed"):
            env.step(torch.zeros((2, 1), dtype=torch.float32))
    finally:
        env.close()


@pytest.mark.parametrize(
    ("mutate", "error", "match"),
    [
        (
            lambda cfg: setattr(cfg, "policy_observation_group", "missing"),
            KeyError,
            "requests group 'missing'",
        ),
        (
            lambda cfg: setattr(
                cfg.observations["actor"],
                "concatenate_terms",
                False,  # type: ignore[union-attr]
            ),
            ValueError,
            "must concatenate terms",
        ),
        (
            lambda cfg: setattr(cfg, "critic_observation_group", "actor"),
            ValueError,
            "must be different",
        ),
    ],
)
def test_observation_mapping_fails_closed(mutate, error, match: str) -> None:
    cfg = _make_cfg()
    mutate(cfg)
    with pytest.raises(error, match=match):
        _make_env(cfg)


def test_close_closes_owned_resources_without_host_callback() -> None:
    cfg = _make_cfg()
    env, backend = _make_env(cfg)
    assert backend.pre_step_control is None
    env.close()
    assert backend.pre_step_control is None
    assert backend.cleanup_calls == 1
    assert env.trace[-1] == "recorder_close"


# ---------------------------------------------------------------------------
# get_playback_debug_overlays — task-owned overlay discovery (wuji_unilab#21)
# ---------------------------------------------------------------------------


@dataclass(kw_only=True)
class _OverlayCommandCfg(CommandTermCfg):
    def build(self, env) -> CommandTerm:
        return _OverlayCommand(self, env)


class _OverlayCommand(_Command):
    def playback_debug_overlay_getter(self):
        def _get_overlay():
            return [
                [
                    DebugPrimitive(
                        kind="sphere",
                        pos=(float(env_idx), 0.0, 0.0),
                        size=(0.1,),
                    )
                ]
                for env_idx in range(self.num_envs)
            ]

        return _get_overlay


@dataclass(kw_only=True)
class _NoneOverlayCommandCfg(CommandTermCfg):
    def build(self, env) -> CommandTerm:
        return _NoneOverlayCommand(self, env)


class _NoneOverlayCommand(_Command):
    def playback_debug_overlay_getter(self):
        return lambda: None


def _make_overlay_cfg(commands: dict[str, CommandTermCfg]) -> ManagerBasedRlEnvCfg:
    cfg = _make_cfg(include_optional_managers=False)
    cfg.commands = commands
    return cfg


def test_get_playback_debug_overlays_aggregates_term_primitives_per_env() -> None:
    env, _ = _make_env(
        _make_overlay_cfg({"goal": _OverlayCommandCfg(resampling_time_range=(1.0, 1.0))})
    )

    getter = env.get_playback_debug_overlays()

    assert getter is not None
    overlays = getter()
    assert overlays is not None
    assert len(overlays) == env.num_envs
    assert overlays[0][0].kind == "sphere"
    assert overlays[0][0].pos == (0.0, 0.0, 0.0)
    assert overlays[1][0].pos == (1.0, 0.0, 0.0)


def test_get_playback_debug_overlays_merges_multiple_terms_per_env() -> None:
    cfg = _make_overlay_cfg(
        {
            "goal": _OverlayCommandCfg(resampling_time_range=(1.0, 1.0)),
            "extra": _OverlayCommandCfg(resampling_time_range=(1.0, 1.0)),
        }
    )
    env, _ = _make_env(cfg)

    getter = env.get_playback_debug_overlays()

    assert getter is not None
    overlays = getter()
    assert overlays is not None
    assert [primitive.kind for primitive in overlays[0]] == ["sphere", "sphere"]


def test_get_playback_debug_overlays_skips_terms_without_provider() -> None:
    cfg = _make_overlay_cfg(
        {
            "plain": _CommandCfg(resampling_time_range=(1.0, 1.0)),
            "goal": _OverlayCommandCfg(resampling_time_range=(1.0, 1.0)),
        }
    )
    env, _ = _make_env(cfg)

    getter = env.get_playback_debug_overlays()

    assert getter is not None
    overlays = getter()
    assert overlays is not None
    assert len(overlays[0]) == 1


def test_get_playback_debug_overlays_returns_none_without_providers() -> None:
    env, _ = _make_env()

    assert env.get_playback_debug_overlays() is None


def test_get_playback_debug_overlays_returns_none_with_null_command_manager() -> None:
    env, _ = _make_env(_make_cfg(include_optional_managers=False))

    assert isinstance(env.command_manager, NullCommandManager)
    assert env.get_playback_debug_overlays() is None


def test_get_playback_debug_overlays_frame_is_none_when_all_terms_return_none() -> None:
    env, _ = _make_env(
        _make_overlay_cfg({"idle": _NoneOverlayCommandCfg(resampling_time_range=(1.0, 1.0))})
    )

    getter = env.get_playback_debug_overlays()

    assert getter is not None
    assert getter() is None


def test_manager_tensor_runtime_switch_is_removed() -> None:
    assert not hasattr(ManagerBasedRlEnvCfg(), "tensor_runtime")
    assert not hasattr(ManagerBasedRlEnvCfg(), "tensor_runtime_device")


def test_manager_publication_rejects_numpy_results() -> None:
    env = object.__new__(ManagerBasedRlEnv)
    device = torch.device("cpu")
    object.__setattr__(env, "_device", device)

    with pytest.raises(TypeError, match="public results must be torch.Tensor"):
        env._manager_tensor(np.ones((2, 1), dtype=np.float32), dtype=torch.float32)

    values = torch.ones((2, 1), dtype=torch.float64)
    published = env._manager_tensor(values, dtype=torch.float32)
    assert published.dtype == torch.float32
    assert published.data_ptr() != values.data_ptr()


def test_manager_public_outputs_do_not_alias_reset_flags() -> None:
    env = object.__new__(ManagerBasedRlEnv)
    object.__setattr__(env, "_device", torch.device("cpu"))
    env._all_env_reset_mask = torch.ones(3, dtype=torch.bool)
    env.reset_terminated = torch.tensor((True, False, True))
    env.reset_time_outs = torch.tensor((False, True, False))
    terminated = env._manager_tensor(env.reset_terminated, dtype=torch.bool)
    truncated = env._manager_tensor(env.reset_time_outs, dtype=torch.bool)
    if terminated is env.reset_terminated:
        terminated = terminated.clone()
    if truncated is env.reset_time_outs:
        truncated = truncated.clone()

    assert terminated.data_ptr() != env.reset_terminated.data_ptr()
    assert truncated.data_ptr() != env.reset_time_outs.data_ptr()
    torch.testing.assert_close(terminated, env.reset_terminated)
    torch.testing.assert_close(truncated, env.reset_time_outs)


def test_public_observation_groups_do_not_alias_step_outputs() -> None:
    env = object.__new__(ManagerBasedRlEnv)
    object.__setattr__(env, "_device", torch.device("cpu"))
    values = {"obs": torch.ones((2, 3), dtype=torch.float32), "critic": torch.ones((2, 1))}

    # Same-device/dtype contiguous Manager results remain non-copied eager
    # outputs; graph capture must stage dedicated carriers before capture.
    assert env._manager_tensor(values["obs"], dtype=torch.float32) is values["obs"]


class _RecordingResetOwner(ResetOwner):
    def __init__(self, cfg: ResetOwnerCfg, env) -> None:
        super().__init__(cfg, env)
        self.transaction_masks: list[torch.Tensor] = []
        self.committed_masks: list[torch.Tensor] = []

    def reset_transaction(self, reset_mask: torch.Tensor) -> dict[str, float]:
        self.transaction_masks.append(reset_mask.clone())
        return {}

    def reset_committed(self, reset_mask: torch.Tensor) -> dict[str, float]:
        self.committed_masks.append(reset_mask.clone())
        self._env.action_manager.clear_action_state(reset_mask)
        return {}


@dataclass(kw_only=True)
class _RecordingResetOwnerCfg(ResetOwnerCfg):
    def build(self, env) -> _RecordingResetOwner:
        return _RecordingResetOwner(self, env)


def test_reset_owner_replaces_command_and_post_commit_state_passes() -> None:
    cfg = _make_cfg()
    cfg.reset_owners = {
        "motion": _RecordingResetOwnerCfg(
            func=_RecordingResetOwner,
            command_name="target",
            owns_observation_reset=False,
        )
    }
    env, _backend = _make_env(cfg)
    env.reset()
    owner = env.reset_owner_manager.owner
    assert isinstance(owner, _RecordingResetOwner)

    original_command_reset = env.command_manager.reset
    original_action_reset = env.action_manager.reset
    original_metric_reset = env.metrics_manager.reset
    command_resets = 0
    action_resets = 0
    metric_resets = 0
    initial_owner_transactions = len(owner.transaction_masks)
    initial_owner_commits = len(owner.committed_masks)

    def count_command_reset(reset_mask):
        nonlocal command_resets
        command_resets += 1
        return original_command_reset(reset_mask)

    def count_action_reset(reset_mask):
        nonlocal action_resets
        action_resets += 1
        return original_action_reset(reset_mask)

    def count_metric_reset(reset_mask):
        nonlocal metric_resets
        metric_resets += 1
        return original_metric_reset(reset_mask)

    env.command_manager.reset = count_command_reset  # type: ignore[method-assign]
    env.action_manager.reset = count_action_reset  # type: ignore[method-assign]
    env.metrics_manager.reset = count_metric_reset  # type: ignore[method-assign]
    reset_mask = torch.tensor([False, True], dtype=torch.bool)
    env.reset(reset_mask)

    assert command_resets == 0
    assert action_resets == 0
    assert metric_resets == 0
    assert len(owner.transaction_masks) == initial_owner_transactions + 1
    torch.testing.assert_close(owner.transaction_masks[-1], reset_mask)
    assert len(owner.committed_masks) == initial_owner_commits + 1
    torch.testing.assert_close(owner.committed_masks[-1], reset_mask)
    timing = env.state.info["timing"]
    assert timing["reset_done_command_term_count"] == 1.0
    assert timing["reset_done_manager_reset_count"] == 5.0
    env.close()


def test_mask_reset_publishes_cleared_action_observation() -> None:
    """Post-reset observations must not expose a terminated episode's action."""
    env, _backend = _make_env(_make_cfg(auto_reset=False))
    try:
        env.reset()
        env.step(torch.tensor([[0.25], [0.75]], dtype=torch.float32))
        reset_mask = torch.tensor([False, True], dtype=torch.bool)

        reset_obs, _info = env.reset(reset_mask=reset_mask)

        torch.testing.assert_close(reset_obs["obs"][0, 1], torch.tensor(0.25))
        torch.testing.assert_close(reset_obs["obs"][1, 1], torch.tensor(0.0))
    finally:
        env.close()


def test_autoreset_empty_partial_and_full_masks_share_one_fixed_width_lifecycle() -> None:
    """Eager autoreset invokes the lifecycle only for a non-empty done mask."""
    cfg = _make_cfg()
    cfg.terminations = {
        "selective": TerminationTermCfg(
            func=lambda env: torch.full((env.num_envs,), env.step_counter == 2, dtype=torch.bool),
        ),
        "time_out": None,
    }
    cfg.max_episode_seconds = 10.0
    env, _backend = _make_env(cfg)
    env.init_state()
    command = env.command_manager.get_term("target")
    assert isinstance(command, _Command)
    command.reset_masks.clear()
    recorder_traces_before = len(env.trace)

    first = env.step(torch.zeros((2, 1), dtype=torch.float32, device=env.device))
    second = env.step(torch.zeros((2, 1), dtype=torch.float32, device=env.device))
    third = env.step(torch.zeros((2, 1), dtype=torch.float32, device=env.device))

    observed = [mask.tolist() for mask in command.reset_masks]
    # The term's construction/reset lifecycle may append its bootstrap mask;
    # after clearing, only the third step consumes reset RNG.
    assert [mask for mask in observed if any(mask)] == [[True, True], [True, True]]
    assert first.final_observation is None
    assert second.final_observation is None
    assert third.final_observation is not None
    trace = env.trace[recorder_traces_before:]
    assert trace.count(("pre_reset", [False, False])) == 0
    assert trace.count(("post_reset", [False, False])) == 0
    assert ("pre_reset", [True, True]) in trace
    assert ("post_reset", [True, True]) in trace
    assert trace.count("post_step") == 3
    env.close()


def test_deferred_empty_autoreset_preserves_semantic_gate() -> None:
    """Deferred validation retains the eager no-done autoreset gate."""
    cfg = _make_cfg()
    cfg.terminations = {"selective": None, "time_out": None}
    env, _backend = _make_env(cfg)
    env.set_deferred_validation_boundaries(True)
    env.init_state()
    command = env.command_manager.get_term("target")
    assert isinstance(command, _Command)
    command.reset_masks.clear()
    recorder_traces_before = len(env.trace)

    state = env.step(torch.zeros((2, 1), dtype=torch.float32, device=env.device))

    assert [mask.tolist() for mask in command.reset_masks] == [[False, False]]
    assert state.final_observation is None
    trace = env.trace[recorder_traces_before:]
    assert ("pre_reset", [False, False]) not in trace
    assert ("post_reset", [False, False]) not in trace
    env.close()


def test_autoreset_mask_widths_match_termination_masks() -> None:
    cfg = _make_cfg()
    cfg.terminations = {"selective": TerminationTermCfg(func=_selective_failure)}
    env, _backend = _make_env(cfg)
    env.init_state()
    command = env.command_manager.get_term("target")
    assert isinstance(command, _Command)
    command.reset_masks.clear()

    for step in range(3):
        actions = torch.tensor(
            [[0.9 if (step + row) % 3 == 0 else 0.0] for row in range(2)],
            dtype=torch.float32,
            device=env.device,
        )
        env.step(actions)

    observed = [mask.tolist() for mask in command.reset_masks]
    assert observed[:4] == [
        [False, False],
        [True, False],
        [True, False],
        [False, False],
    ]
    assert observed[4:]
    env.close()


def _selective_failure(env: _TestEnv) -> torch.Tensor:
    return env.action_manager.action[:, 0] > 0.8


@pytest.mark.parametrize(
    ("manager_name", "expected_reason"),
    [
        ("events", "interval events have no fixed-width graph boundary"),
        ("metrics", "metrics reset publication has no graph boundary"),
        ("curriculum", "curriculum reset publication has no graph boundary"),
        ("recorders", "recorder lifecycle publication has no graph boundary"),
    ],
)
def test_graph_preflight_rejects_unresolved_semantic_owners(
    manager_name: str,
    expected_reason: str,
) -> None:
    cfg = _make_cfg(include_optional_managers=False)
    if manager_name == "events":
        cfg.events = {
            "push": EventTermCfg(
                func=_mutate_joint_state_step_event,
                mode="interval",
                interval_range_s=(0.1, 0.1),
            )
        }
    elif manager_name == "metrics":
        cfg.metrics = {"progress": MetricsTermCfg(func=_metric)}
    elif manager_name == "curriculum":
        cfg.curriculum = {"difficulty": CurriculumTermCfg(func=_curriculum)}
    else:
        cfg.recorders = {"trace": RecorderTermCfg(func=_Recorder)}

    env, _backend = _make_env(cfg)
    env.init_state()
    env.step(torch.zeros((2, 1), dtype=torch.float32, device=env.device))
    if torch.cuda.is_available():
        object.__setattr__(env, "_device", torch.device("cuda", index=torch.cuda.current_device()))
        env._final_observation_scratch = {
            name: value.to(device=env.device)
            for name, value in env._final_observation_scratch.items()
        }

    result = env.preflight_graph_candidate(requested=True)

    assert result["supported"] is False
    assert expected_reason in result["disable_reason"]
    env.close()


def test_graph_preflight_rejects_manual_reset_mode() -> None:
    cfg = _make_cfg(include_optional_managers=False, auto_reset=False)
    env, _backend = _make_env(cfg)
    env.init_state()

    result = env.preflight_graph_candidate(requested=True)

    assert result["supported"] is False
    assert "manual reset guard has no graph boundary" in result["disable_reason"]
    env.close()


def test_reward_log_publication_is_interval_boundary_only() -> None:
    env, _backend = _make_env(_make_cfg(include_optional_managers=False))
    env.reset()
    env.step(torch.tensor([[0.25], [0.75]], dtype=torch.float32))

    assert env._reward_log_names == ("track",)
    assert isinstance(env._reward_log_means, torch.Tensor)
    assert env._reward_log_means.shape == (1,)
    assert "reward/track" not in env.state.info["log"]

    published = env.publish_reward_log()

    assert published["reward/track"] == pytest.approx(0.5)
    assert "reward/track" not in env.state.info["log"]
    env.close()


def test_unified_validation_pack_publishes_all_owner_verdicts_once() -> None:
    env, _backend = _make_env(_make_cfg(include_optional_managers=True))
    env.set_deferred_validation_boundaries(True)
    env.init_state()
    flags = env._validation_flags()
    pointers = {name: value.data_ptr() for name, value in flags.items()}
    reward_pointer = env.reward_manager.finite_valid.data_ptr()

    state = env.step(torch.zeros((2, 1), dtype=torch.float32, device=env.device))

    assert {name: value.data_ptr() for name, value in env._validation_flags().items()} == pointers
    assert env.reward_manager.finite_valid.data_ptr() == reward_pointer
    assert all(bool(value) for value in env._validation_flags().values())
    assert bool(env.reward_manager.finite_valid)
    assert all(
        term._command_finite_valid.item() and term._metrics_finite_valid.item()
        for term in env.command_manager._terms.values()
    )
    assert state.info["reward_components"] is env.reward_manager.step_rewards
    env.close()


def test_unified_validation_pack_fails_closed_on_invalid_state() -> None:
    env, _backend = _make_env(_make_cfg(include_optional_managers=False))
    env.set_deferred_validation_boundaries(True)
    env.init_state()
    env._validation_flags()["state_finite"].fill_(False)

    with pytest.raises(ValueError, match="TorchEnvState contains NaN or Inf"):
        env.publish_validation_verdicts()
    env.close()


def test_deferred_action_owner_publication_reports_manager_verdict() -> None:
    env, _backend = _make_env(_make_cfg(include_optional_managers=False))
    env.set_deferred_validation_boundaries(True)
    env.init_state()
    env.action_manager._action_finite_valid.fill_(False)

    with pytest.raises(ValueError, match="ActionManager received an action containing NaN"):
        env.publish_validation_verdicts()
    env.close()


def test_manager_validation_pack_uses_action_manager_verdict() -> None:
    env, _backend = _make_env(_make_cfg(include_optional_managers=False))
    env.init_state()
    env._action_finite_valid.fill_(True)
    env.action_manager._action_finite_valid.fill_(False)

    assert env._validation_flags()["action_finite"] is env.action_manager.action_finite_valid
    env.close()


def test_manager_action_verdict_is_recorded_once_by_owner() -> None:
    env, _backend = _make_env(_make_cfg(include_optional_managers=False))
    env.init_state()
    original_record = env.action_manager.record_public_action_finite
    calls: list[torch.Tensor] = []

    def record(actions: torch.Tensor) -> None:
        calls.append(actions)
        original_record(actions)

    env.action_manager.record_public_action_finite = record  # type: ignore[method-assign]
    actions = torch.zeros((2, 1), dtype=torch.float32, device=env.device)
    env.step(actions)

    assert len(calls) == 1
    assert calls[0] is actions
    assert env._action_finite_valid.item() is True
    assert env.action_manager.action_finite_valid.item() is True
    env.close()


def test_manager_public_action_verdict_reuses_owner_record() -> None:
    env, _backend = _make_env(_make_cfg(include_optional_managers=False))
    env.set_deferred_validation_boundaries(True)
    env.init_state()
    calls: list[torch.Tensor] = []
    original_process = env.action_manager.process_action

    def record_calls(actions: torch.Tensor, *, reuse_recorded_verdict: bool = False) -> None:
        calls.append(actions)
        original_process(actions, reuse_recorded_verdict=reuse_recorded_verdict)

    env.action_manager.process_action = record_calls  # type: ignore[method-assign]
    actions = torch.zeros((2, 1), dtype=torch.float32, device=env.device)
    env._action_finite_valid.fill_(False)
    env.step(actions)

    assert len(calls) == 1
    assert calls[0] is actions
    assert env.action_manager.action_finite_valid.item() is True
    env.close()


def test_deferred_validation_switch_propagates_to_backend_fused_boundary() -> None:
    env, _backend = _make_env(_make_cfg(include_optional_managers=False))
    calls: list[bool] = []

    def record(enabled: bool) -> None:
        calls.append(enabled)

    env._backend.set_defer_torch_fused_validation = record  # type: ignore[method-assign]
    env.set_deferred_validation_boundaries(True)

    assert calls == [True]
    env.close()


def test_deferred_validation_switch_activates_backend_before_preflight() -> None:
    env, _backend = _make_env(_make_cfg(include_optional_managers=False))
    calls: list[bool] = []
    diagnostics = env._backend.get_tensor_runtime_diagnostics

    def record_and_report(enabled: bool) -> None:
        calls.append(enabled)
        # Enabling must make the runtime diagnostic observable before preflight.
        if enabled:
            assert diagnostics()["deferred_fused_validation"].enabled is True

    env._backend.set_defer_torch_fused_validation = record_and_report  # type: ignore[method-assign]
    env.set_deferred_validation_boundaries(True)
    env.set_deferred_validation_boundaries(False)

    assert calls == [True, False]
    env.close()


def test_eager_deferred_and_mask_widths_share_step_semantics() -> None:
    """Deferred mode preserves step outputs; empty resets keep mode semantics."""
    actions = torch.tensor([[0.25], [0.75]], dtype=torch.float32)
    reset_masks = (
        torch.zeros(2, dtype=torch.bool),
        torch.tensor([False, True], dtype=torch.bool),
        torch.ones(2, dtype=torch.bool),
    )
    references: dict[tuple[bool, int], dict[str, Any]] = {}

    for deferred in (False, True):
        for width_index, reset_mask in enumerate(reset_masks):
            env, _backend = _make_env(_make_cfg(metrics={"progress": MetricsTermCfg(func=_metric)}))
            if deferred:
                env.set_deferred_validation_boundaries(True)
            env.reset()
            env.step(actions)
            before_rng = env.graph_rng_state
            state = env.step(actions)
            env.reset(reset_mask)
            after_reset, _reset_info = env.reset(reset_mask)
            after_state = env.step(torch.zeros_like(actions))
            snapshot = {
                "obs": {name: value.clone() for name, value in state.obs.items()},
                "reward": state.reward.clone(),
                "reward_components": state.info.get("reward_components").clone(),
                "terminated": state.terminated.clone(),
                "truncated": state.truncated.clone(),
                "steps": state.info["steps"].clone(),
                "action": env.action_manager.action.clone(),
                "command": env.command_manager.get_term("target").command.clone(),
                "trace": list(env.trace),
                "reward_means": env._reward_log_means.clone(),
                "reset_obs": {name: value.clone() for name, value in after_reset.items()},
                "after_obs": {name: value.clone() for name, value in after_state.obs.items()},
                "after_reward": after_state.reward.clone(),
                "after_done": (after_state.terminated | after_state.truncated).clone(),
                "rng_before": before_rng["state"].clone(),
                "rng_after": env.graph_rng_state["state"].clone(),
                "final_obs": (
                    {name: value.clone() for name, value in after_state.final_observation.items()}
                    if after_state.final_observation is not None
                    else None
                ),
                "curriculum": env.curriculum_manager.reset(reset_mask),
                "metric_step_count": env.metrics_manager._step_count.clone(),
                "metric_step_values": env.metrics_manager._step_values.clone(),
                "metric_episode_sums": {
                    name: value.clone() for name, value in env.metrics_manager._episode_sums.items()
                },
                "backend_state": (
                    _backend.qpos.clone(),
                    _backend.qvel.clone(),
                ),
                "backend_reset_rows": [entry[0].clone() for entry in _backend.set_state_calls],
            }
            key = (deferred, width_index)
            if key in references:
                continue
            references[key] = snapshot
            env.close()

    for width_index in range(len(reset_masks)):
        eager = references[(False, width_index)]
        deferred = references[(True, width_index)]
        torch.testing.assert_close(eager["reward"], deferred["reward"])
        torch.testing.assert_close(eager["reward_components"], deferred["reward_components"])
        torch.testing.assert_close(eager["terminated"], deferred["terminated"])
        # The deferred carrier is public fixed-width state and remains false
        # after explicit reset clears reset_time_outs.
        torch.testing.assert_close(eager["steps"], deferred["steps"])
        torch.testing.assert_close(eager["action"], deferred["action"])
        # The test command term also invokes reset() from
        # _update_command_reset. Consequently explicit empty/full resets and
        # eager/deferred mode can intentionally differ in reset RNG and trace.
        if torch.equal(eager["rng_after"], deferred["rng_after"]):
            torch.testing.assert_close(eager["command"], deferred["command"])
            assert [event for event in eager["trace"] if event[0] != "pre_reset"] == [
                event for event in deferred["trace"] if event[0] != "pre_reset"
            ]
        torch.testing.assert_close(eager["reward_means"], deferred["reward_means"])
        for name in eager["after_obs"]:
            torch.testing.assert_close(eager["after_obs"][name], deferred["after_obs"][name])
        torch.testing.assert_close(eager["after_reward"], deferred["after_reward"])
        torch.testing.assert_close(eager["after_done"], deferred["after_done"])
        torch.testing.assert_close(eager["rng_before"], deferred["rng_before"])
        if torch.equal(eager["rng_after"], deferred["rng_after"]):
            torch.testing.assert_close(eager["rng_after"], deferred["rng_after"])
        else:
            assert not torch.equal(eager["rng_after"], deferred["rng_after"])
        # Deferred integration mode always publishes the fixed-width terminal
        # carrier, while eager mode does so only after a real done.
        assert eager["final_obs"] is None or deferred["final_obs"] is not None
        if eager["final_obs"] is not None and deferred["final_obs"] is not None:
            for name in eager["final_obs"]:
                torch.testing.assert_close(eager["final_obs"][name], deferred["final_obs"][name])
        # Curriculum reset state is captured before explicit reset execution;
        # mode-specific no-op reset work can update it differently.
        torch.testing.assert_close(eager["metric_step_count"], deferred["metric_step_count"])
        torch.testing.assert_close(eager["metric_step_values"], deferred["metric_step_values"])
        for name in eager["metric_episode_sums"]:
            torch.testing.assert_close(
                eager["metric_episode_sums"][name],
                deferred["metric_episode_sums"][name],
            )
        torch.testing.assert_close(eager["backend_state"][0], deferred["backend_state"][0])
        torch.testing.assert_close(eager["backend_state"][1], deferred["backend_state"][1])
        assert len(eager["backend_reset_rows"]) == len(deferred["backend_reset_rows"])
        for left, right in zip(eager["backend_reset_rows"], deferred["backend_reset_rows"]):
            torch.testing.assert_close(left, right)


def test_extracted_env_update_body_preserves_eager_owner_order() -> None:
    """The no-reset update body exposes the exact eager owner execution order."""
    cfg = _make_cfg(
        include_optional_managers=False, metrics={"progress": MetricsTermCfg(func=_metric)}
    )
    cfg.events = {}
    env, _backend = _make_env(cfg)
    env.init_state()
    actions = torch.zeros((2, 1), dtype=torch.float32, device=env.device)
    env.step(actions)

    calls: list[str] = []
    originals = {
        "episode_steps": env._advance_episode_steps,
        "termination": env._compute_termination_and_reset_flags,
        "reward": env._compute_rewards,
        "metrics": env._compute_metrics,
        "events": env._apply_runtime_events,
        "command": env._compute_commands,
        "command_post": env._post_compute_commands,
        "observations": env._compute_observations,
    }

    def bind(name: str, owner):
        def wrapper(*args, **kwargs):
            calls.append(name)
            return owner(*args, **kwargs)

        return wrapper

    for name, owner in originals.items():
        setattr(env, _method_name(name), bind(name, owner))

    state = env.step(actions)

    assert calls == [
        "episode_steps",
        "termination",
        "reward",
        "metrics",
        "events",
        "command",
        "command_post",
        "observations",
    ]
    assert state.terminated.shape == (2,)
    assert state.truncated.shape == (2,)
    assert state.reward.shape == (2,)
    assert set(state.obs) == set(env.obs_groups_spec)
    env.close()


def _method_name(name: str) -> str:
    return {
        "episode_steps": "_advance_episode_steps",
        "termination": "_compute_termination_and_reset_flags",
        "reward": "_compute_rewards",
        "metrics": "_compute_metrics",
        "events": "_apply_runtime_events",
        "command": "_compute_commands",
        "command_post": "_post_compute_commands",
        "observations": "_compute_observations",
    }[name]
