"""Tensor-native environment lifecycle base.

`TorchEnv` is the sole Manager-Based runtime carrier. It owns the state, step,
selected-row autoreset, timeout, finite, training-state, and backend playback
contracts.
"""

from __future__ import annotations

import abc
import dataclasses
import os
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from os import PathLike
from typing import TYPE_CHECKING, Any, Optional, cast

import gymnasium as gym
import numpy as np
import torch
from unisim.backend.base import (
    BackendPlayRenderPlan,
    CameraCfg,
    DebugOverlayGetter,
    DebugPrimitive,
    SimBackend,
    TensorExecution,
    tensor_device_matches,
)

from unilab.base.backend_timing import RESET_DONE_DETAIL_TIMING_KEYS
from unilab.base.base import ABEnv, EnvCfg, EnvPlayCapabilities
from unilab.base.cpu_runtime import apply_env_cpu_runtime
from unilab.base.scene import SceneCfg
from unilab.dtype_config import get_global_dtype

if TYPE_CHECKING:
    from unilab.training.tensor_diagnostics import TensorNanGuard


@dataclass
class TorchEnvState:
    """Tensor-native vectorized environment transition.

    Terminal pre-reset observations are represented only by
    ``final_observation``. This contract does not duplicate them through
    ``info`` compatibility buffers.
    """

    obs: dict[str, torch.Tensor]
    reward: torch.Tensor
    terminated: torch.Tensor
    truncated: torch.Tensor
    info: dict[str, Any]
    final_observation: dict[str, torch.Tensor] | None = None

    def replace(self, **updates: Any) -> "TorchEnvState":
        return dataclasses.replace(self, **updates)


def _torch_dtype(dtype: np.dtype[Any]) -> torch.dtype:
    if dtype == np.dtype(np.float32):
        return torch.float32
    if dtype == np.dtype(np.float64):
        return torch.float64
    raise TypeError(f"Unsupported environment float dtype {dtype!r}")


def _cpu_time() -> float:
    process = os.times()
    return process.user + process.system


class TorchEnv(ABEnv):
    """Backend-agnostic Torch environment lifecycle.

    Concrete owners implement action transformation, tensor state update, and
    selected-row reset. The base deliberately rejects NumPy input and never
    sanitizes non-finite tensors: invalid actions, controls, observations, and
    rewards fail closed.
    """

    def __init__(
        self,
        cfg: EnvCfg,
        backend: SimBackend,
        num_envs: int,
        *,
        device: torch.device | str = "cpu",
    ):
        apply_env_cpu_runtime(cfg.cpu_ids)
        self._cfg = cfg
        self._backend = backend
        self._num_envs = num_envs
        requested_device = torch.device(device)
        if requested_device.type == "cuda" and requested_device.index is None:
            requested_device = torch.device("cuda", index=torch.cuda.current_device())
        self._device = requested_device
        self._dtype = _torch_dtype(get_global_dtype())
        self._state: Optional[TorchEnvState] = None
        self._truncated_scratch = torch.zeros((num_envs,), dtype=torch.bool, device=self._device)
        self._final_observation_scratch: dict[str, torch.Tensor] | None = None
        self._tensor_runtime_bound = False
        self.step_counter = 0
        self._autoreset = True
        self._autoreset_reset_active = False
        self._rgb_array_renderer_ready = False
        self._nan_guard: "TensorNanGuard | None" = None

    @property
    def device(self) -> torch.device:
        return self._device

    def _bind_tensor_runtime(self) -> None:
        if self._tensor_runtime_bound:
            return
        # A backend without a tensor lifecycle has only its public NumPy wire.
        # That lifecycle is legal solely for the legacy runtime request, which
        # is the default for subprocess owners such as IsaacSim. Tensor runtime
        # requests remain fail-closed below.
        if (
            getattr(self.cfg, "tensor_runtime", True) is False
            and self._backend.get_tensor_capabilities().execution is TensorExecution.UNSUPPORTED
        ):
            self._tensor_runtime_bound = True
            return
        # A device-resident backend cannot expose a public CPU lifecycle. The
        # only valid false request is a task-owned cold-path proxy used to extract
        # contracts before its direct CUDA runtime is constructed.
        if (
            getattr(self.cfg, "tensor_runtime", True) is False
            and self._backend.get_tensor_capabilities().execution is TensorExecution.DEVICE_RESIDENT
        ):
            self._tensor_runtime_bound = True
            return
        capabilities = self._backend.get_tensor_capabilities()
        execution = capabilities.execution
        if execution not in {TensorExecution.HOST_BRIDGE, TensorExecution.DEVICE_RESIDENT}:
            raise ValueError(
                f"{type(self._backend).__name__} does not declare a tensor lifecycle; "
                f"TorchEnv got {execution!r}"
            )
        if self._backend.tensor_execution() is not execution:
            raise ValueError("backend tensor execution does not match its capabilities")
        if not capabilities.stepping or not capabilities.selected_reset:
            raise ValueError("TorchEnv backend must support tensor stepping and selected reset")
        if execution is TensorExecution.HOST_BRIDGE and not capabilities.packed_host_bridge:
            raise ValueError("HOST_BRIDGE TorchEnv requires a packed host-bridge plan")
        if not tensor_device_matches(capabilities.torch_devices, self._device):
            raise ValueError(
                f"backend did not accept Torch device {str(self._device)!r}; "
                f"supported devices are {capabilities.torch_devices}"
            )
        if execution is TensorExecution.DEVICE_RESIDENT and self._device.type != "cuda":
            raise ValueError(
                f"DEVICE_RESIDENT TorchEnv requires a CUDA device; received {self._device}"
            )
        self._tensor_runtime_bound = True

    @property
    def cfg(self) -> EnvCfg:
        return self._cfg

    @property
    def num_envs(self) -> int:
        return self._num_envs

    @property
    def state(self) -> Optional[TorchEnvState]:
        return self._state

    @property
    def obs_groups_spec(self) -> dict[str, int]:
        raise NotImplementedError(
            f"{type(self).__name__} must declare observation group dimensions"
        )

    @property
    def observation_space(self) -> gym.Space:
        total = sum(self.obs_groups_spec.values())
        return gym.spaces.Box(-np.inf, np.inf, shape=(total,), dtype=np.float32)

    def init_state(self) -> TorchEnvState:
        self._bind_tensor_runtime()
        obs = {
            name: torch.zeros((self._num_envs, dim), dtype=self._dtype, device=self._device)
            for name, dim in self.obs_groups_spec.items()
        }
        self._state = TorchEnvState(
            obs=obs,
            reward=torch.zeros((self._num_envs,), dtype=self._dtype, device=self._device),
            terminated=torch.ones((self._num_envs,), dtype=torch.bool, device=self._device),
            truncated=torch.zeros((self._num_envs,), dtype=torch.bool, device=self._device),
            info={"steps": torch.zeros((self._num_envs,), dtype=torch.int64, device=self._device)},
        )
        self._reset_done_envs()
        self._clear_step_final_observation()
        return self._state

    def reset(
        self, env_indices: torch.Tensor | None = None
    ) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
        raise NotImplementedError(f"{type(self).__name__} must define a tensor reset lifecycle")

    def _normalize_reset_indices(self, env_indices: torch.Tensor | None) -> torch.Tensor:
        rows = (
            torch.arange(self._num_envs, device=self._device, dtype=torch.int64)
            if env_indices is None
            else env_indices
        )
        if not isinstance(rows, torch.Tensor):
            raise TypeError(
                f"TorchEnv reset indices must be a torch.Tensor, got {type(rows).__name__}"
            )
        if rows.ndim != 1:
            raise ValueError(f"TorchEnv reset indices must be one-dimensional, got {rows.ndim}")
        if rows.dtype not in {torch.int32, torch.int64}:
            raise TypeError(f"TorchEnv reset indices must be integer dtype, got {rows.dtype}")
        if rows.device != self._device:
            raise ValueError(f"TorchEnv reset indices device must be {self._device}")
        if rows.numel() != rows.unique().numel():
            raise ValueError("TorchEnv reset indices must be unique")
        if rows.numel() and (rows.min() < 0 or rows.max() >= self._num_envs):
            raise ValueError(f"TorchEnv reset indices must be in [0, {self._num_envs})")
        return rows.to(torch.int64)

    def step(self, actions: torch.Tensor) -> TorchEnvState:
        started = time.perf_counter()
        cpu_started = _cpu_time()
        self._validate_action(actions)
        self._bind_tensor_runtime()
        if self._state is None:
            self.init_state()
        assert self._state is not None

        phase = time.perf_counter()
        phase_cpu = _cpu_time()
        ctrl = self.apply_action(actions, self._state)
        if self._nan_guard is not None:
            bad_ctrl_ids = self._nan_guard.check_ctrl(ctrl, step=self.step_counter)
            if bad_ctrl_ids is not None:
                self._nan_guard.dump(
                    bad_ctrl_ids, self._resolve_nan_guard_model_file(), self.step_counter
                )
        self._validate_control(ctrl)
        apply_action_ms = (time.perf_counter() - phase) * 1000.0
        apply_action_cpu_ms = (_cpu_time() - phase_cpu) * 1000.0

        self._state.truncated.fill_(False)
        self._clear_step_final_observation()

        phase = time.perf_counter()
        phase_cpu = _cpu_time()
        if self._backend.get_tensor_capabilities().execution is TensorExecution.UNSUPPORTED:
            if getattr(self.cfg, "tensor_runtime", True) is not False:
                self._bind_tensor_runtime()
            host_control = ctrl.detach().cpu().numpy()
            backend_result = self._backend.step(host_control, self._cfg.sim_substeps)
        else:
            backend_result = self._backend.step_tensor(ctrl, self._cfg.sim_substeps)
        step_core_ms = (time.perf_counter() - phase) * 1000.0
        step_core_cpu_ms = (_cpu_time() - phase_cpu) * 1000.0

        phase = time.perf_counter()
        phase_cpu = _cpu_time()
        self._state = self.update_state(self._state)
        if self._nan_guard is not None:
            self._nan_guard.capture(self._tensor_diagnostics_state())
            nan_ids = self._nan_guard.check(
                self._state.obs, self._state.reward, step=self.step_counter
            )
            if nan_ids is not None:
                self._nan_guard.dump(
                    nan_ids, self._resolve_nan_guard_model_file(), self.step_counter
                )
        self._validate_state(self._state)
        update_state_ms = (time.perf_counter() - phase) * 1000.0
        update_state_cpu_ms = (_cpu_time() - phase_cpu) * 1000.0

        self._state.info["steps"] += 1
        self.step_counter += 1
        truncated = self._compute_truncated(self._state)
        self._state.truncated.logical_or_(truncated)

        phase = time.perf_counter()
        phase_cpu = _cpu_time()
        did_reset = self._autoreset and bool((self._state.terminated | self._state.truncated).any())
        if did_reset:
            self._reset_done_envs()
        reset_done_ms = (time.perf_counter() - phase) * 1000.0
        reset_done_cpu_ms = (_cpu_time() - phase_cpu) * 1000.0

        timing = self._state.info.setdefault("timing", {})
        if not did_reset:
            self._clear_reset_done_detail_timing(timing)
        timing["env_step_total_ms"] = (time.perf_counter() - started) * 1000.0
        timing["apply_action_ms"] = apply_action_ms
        timing["apply_action_cpu_ms"] = apply_action_cpu_ms
        timing["step_core_ms"] = step_core_ms
        timing["step_core_cpu_ms"] = step_core_cpu_ms
        timing["update_state_ms"] = update_state_ms
        timing["update_state_cpu_ms"] = update_state_cpu_ms
        timing["reset_done_ms"] = reset_done_ms
        timing["reset_done_cpu_ms"] = reset_done_cpu_ms
        timing["env_step_other_cpu_ms"] = max(
            (_cpu_time() - cpu_started) * 1000.0
            - apply_action_cpu_ms
            - step_core_cpu_ms
            - update_state_cpu_ms
            - reset_done_cpu_ms,
            0.0,
        )
        if isinstance(backend_result, dict):
            for key, value in backend_result.get("timing", {}).items():
                timing[f"backend_{key}"] = value
        return self._state

    def _validate_action(self, actions: torch.Tensor) -> None:
        if not isinstance(actions, torch.Tensor):
            raise TypeError(f"TorchEnv action must be a torch.Tensor, got {type(actions).__name__}")
        action_shape = self.action_space.shape
        if action_shape is None:
            raise ValueError("TorchEnv action_space must declare a vector shape")
        expected_shape = (self._num_envs, *action_shape)
        if actions.shape != expected_shape:
            raise ValueError(
                f"TorchEnv action shape must be {expected_shape}, got {tuple(actions.shape)}"
            )
        if actions.dtype != torch.float32:
            raise TypeError(f"TorchEnv action dtype must be float32, got {actions.dtype}")
        if not actions.is_contiguous():
            raise ValueError("TorchEnv action must be contiguous")
        if actions.device != self.device:
            raise ValueError(f"TorchEnv action device must be {self.device}, got {actions.device}")
        if not bool(torch.isfinite(actions).all()):
            raise ValueError("TorchEnv action contains NaN or Inf")

    def _validate_control(self, ctrl: torch.Tensor) -> None:
        if not isinstance(ctrl, torch.Tensor):
            raise TypeError(f"TorchEnv control must be a torch.Tensor, got {type(ctrl).__name__}")
        if ctrl.dtype != torch.float32:
            raise TypeError(f"TorchEnv control dtype must be float32, got {ctrl.dtype}")
        expected_shape = (self._num_envs, self._backend.num_actuators)
        if ctrl.shape != expected_shape:
            raise ValueError(
                f"TorchEnv control shape must be {expected_shape}, got {tuple(ctrl.shape)}"
            )
        if not ctrl.is_contiguous():
            raise ValueError("TorchEnv control must be contiguous")
        if ctrl.device != self.device:
            raise ValueError(f"TorchEnv control device must be {self.device}, got {ctrl.device}")
        if not bool(torch.isfinite(ctrl).all()):
            raise ValueError("TorchEnv control contains NaN or Inf")

    def _validate_state(self, state: TorchEnvState) -> None:
        expected_keys = set(self.obs_groups_spec)
        if not isinstance(state.obs, dict) or set(state.obs) != expected_keys:
            raise ValueError("TorchEnvState.obs keys do not match obs_groups_spec")
        for name, dim in self.obs_groups_spec.items():
            value = state.obs[name]
            if not isinstance(value, torch.Tensor):
                raise TypeError(f"TorchEnvState.obs[{name!r}] must be a torch.Tensor")
            if value.dtype != self._dtype:
                raise TypeError(
                    f"TorchEnvState.obs[{name!r}] dtype must be {self._dtype}, got {value.dtype}"
                )
            if value.device != self.device:
                raise ValueError(f"TorchEnvState.obs[{name!r}] device must be {self.device}")
            if value.shape != (self._num_envs, dim):
                raise ValueError(f"TorchEnvState.obs[{name!r}] has shape {tuple(value.shape)}")
            self._validate_finite_float(value, f"obs[{name!r}]")
        self._validate_vector(state.reward, "reward", self._dtype)
        self._validate_finite_float(state.reward, "reward")
        self._validate_vector(state.terminated, "terminated", torch.bool)
        self._validate_vector(state.truncated, "truncated", torch.bool)
        steps = state.info.get("steps")
        self._validate_vector(steps, "info['steps']", torch.int64)
        self._validate_final_observation(state.final_observation)

    def _validate_vector(self, value: Any, label: str, dtype: torch.dtype) -> None:
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"TorchEnvState {label} must be a torch.Tensor")
        if value.shape != (self._num_envs,):
            raise ValueError(f"TorchEnvState {label} has shape {tuple(value.shape)}")
        if value.dtype != dtype:
            raise TypeError(f"TorchEnvState {label} dtype must be {dtype}, got {value.dtype}")
        if value.device != self.device:
            raise ValueError(f"TorchEnvState {label} device must be {self.device}")

    def _validate_finite_float(self, value: torch.Tensor, label: str) -> None:
        if not value.is_floating_point():
            raise TypeError(f"TorchEnvState {label} must be floating-point")
        if not bool(torch.isfinite(value).all()):
            raise ValueError(f"TorchEnvState {label} contains NaN or Inf")

    def _validate_final_observation(
        self, final_observation: dict[str, torch.Tensor] | None
    ) -> None:
        if final_observation is None:
            return
        assert self._state is not None
        if set(final_observation) != set(self._state.obs):
            raise ValueError("TorchEnvState.final_observation keys do not match obs")
        for name, value in final_observation.items():
            expected = self._state.obs[name]
            if not isinstance(value, torch.Tensor):
                raise TypeError(f"TorchEnvState.final_observation[{name!r}] must be a torch.Tensor")
            if value.shape != expected.shape:
                raise ValueError(
                    f"TorchEnvState.final_observation[{name!r}] has shape {tuple(value.shape)}"
                )
            if value.dtype != expected.dtype:
                raise TypeError(
                    f"TorchEnvState.final_observation[{name!r}] dtype must be "
                    f"{expected.dtype}, got {value.dtype}"
                )
            if value.device != expected.device:
                raise ValueError(
                    f"TorchEnvState.final_observation[{name!r}] device must be {expected.device}"
                )
            self._validate_finite_float(value, f"final_observation[{name!r}]")

    def _compute_truncated(self, state: TorchEnvState) -> torch.Tensor:
        self._truncated_scratch.fill_(False)
        max_steps = self._cfg.max_episode_steps
        if max_steps is not None:
            torch.gt(state.info["steps"], max_steps - 1, out=self._truncated_scratch)
        return self._truncated_scratch

    def _reset_done_envs(self) -> None:
        assert self._state is not None
        reset_total_started = time.perf_counter()
        detail_timing = {key: 0.0 for key in RESET_DONE_DETAIL_TIMING_KEYS}
        done = self._state.terminated | self._state.truncated
        if not bool(done.any()):
            self._clear_reset_done_detail_timing(self._state.info.setdefault("timing", {}))
            return

        rows = done.nonzero(as_tuple=False).flatten().to(torch.int64)
        terminal_started = time.perf_counter()
        detail_timing["reset_done_count"] = float(rows.numel())
        self._state.info["steps"][rows] = 0

        final_observation = self._ensure_final_observation_scratch()
        for name, values in self._state.obs.items():
            final_observation[name].index_copy_(0, rows, values.index_select(0, rows))
        self._state.final_observation = final_observation
        detail_timing["reset_done_terminal_obs_ms"] = (
            time.perf_counter() - terminal_started
        ) * 1000.0

        reset_started = time.perf_counter()
        self._autoreset_reset_active = True
        try:
            new_obs, reset_info = self.reset(rows)
        finally:
            self._autoreset_reset_active = False
        detail_timing["reset_done_reset_call_ms"] = (time.perf_counter() - reset_started) * 1000.0
        collected = self._collect_reset_backend_timing_ms()
        detail_timing.update(
            {key: value for key, value in collected.items() if key in detail_timing}
        )

        scatter_started = time.perf_counter()
        self._validate_reset_observation(new_obs, rows)
        for name, values in new_obs.items():
            self._state.obs[name].index_copy_(0, rows, values)
        self._scatter_reset_info(reset_info, rows)
        detail_timing["reset_done_obs_scatter_ms"] = (
            time.perf_counter() - scatter_started
        ) * 1000.0
        detail_timing["reset_done_info_scatter_ms"] = 0.0
        measured = (
            detail_timing["reset_done_terminal_obs_ms"]
            + detail_timing["reset_done_reset_call_ms"]
            + detail_timing["reset_done_obs_scatter_ms"]
            + detail_timing["reset_done_info_scatter_ms"]
        )
        detail_timing["reset_done_internal_gap_ms"] = (
            time.perf_counter() - reset_total_started
        ) * 1000.0 - measured

        timing = self._state.info.setdefault("timing", {})
        self._clear_reset_done_detail_timing(timing)
        timing.update(detail_timing)

    def _ensure_final_observation_scratch(self) -> dict[str, torch.Tensor]:
        assert self._state is not None
        obs = self._state.obs
        scratch = self._final_observation_scratch
        if (
            scratch is None
            or set(scratch) != set(obs)
            or any(
                scratch[name].shape != value.shape
                or scratch[name].dtype != value.dtype
                or scratch[name].device != value.device
                for name, value in obs.items()
            )
        ):
            scratch = {name: torch.zeros_like(value) for name, value in obs.items()}
            self._final_observation_scratch = scratch
        return scratch

    def _validate_reset_observation(
        self, obs: Mapping[str, torch.Tensor], rows: torch.Tensor
    ) -> None:
        if not isinstance(obs, dict) or set(obs) != set(self.obs_groups_spec):
            raise ValueError("TorchEnv reset observations do not match obs_groups_spec")
        for name, dim in self.obs_groups_spec.items():
            value = obs[name]
            if not isinstance(value, torch.Tensor):
                raise TypeError(f"TorchEnv reset obs[{name!r}] must be a torch.Tensor")
            if value.dtype != self._dtype:
                raise TypeError(
                    f"TorchEnv reset obs[{name!r}] dtype must be {self._dtype}, got {value.dtype}"
                )
            if value.shape != (rows.numel(), dim):
                raise ValueError(
                    f"TorchEnv reset obs[{name!r}] has shape {tuple(value.shape)}; "
                    f"expected {(rows.numel(), dim)}"
                )
            if value.device != self.device:
                raise ValueError(f"TorchEnv reset obs[{name!r}] device must be {self.device}")
            self._validate_finite_float(value, f"reset obs[{name!r}]")

    def _scatter_reset_info(self, reset_info: Mapping[str, Any], rows: torch.Tensor) -> None:
        assert self._state is not None
        for key, value in reset_info.items():
            current = self._state.info.get(key)
            if isinstance(value, torch.Tensor) and value.shape and value.shape[0] == rows.numel():
                if not isinstance(current, torch.Tensor) or current.shape[1:] != value.shape[1:]:
                    full_shape = (self._num_envs, *value.shape[1:])
                    current = torch.zeros(full_shape, dtype=value.dtype, device=self.device)
                    self._state.info[key] = current
                current.index_copy_(0, rows, value.to(device=self.device))
            elif not isinstance(current, torch.Tensor):
                self._state.info[key] = value

    def _clear_reset_done_detail_timing(self, timing: dict[str, Any]) -> None:
        for key in RESET_DONE_DETAIL_TIMING_KEYS:
            timing[key] = 0.0

    def _collect_reset_backend_timing_ms(self) -> dict[str, float]:
        """Return backend-owned sub-timings from the most recent tensor reset."""
        return {}

    def _clear_step_final_observation(self) -> None:
        assert self._state is not None
        self._state.final_observation = None

    def init_play_renderer(
        self,
        render_spacing: float | None = None,
        render_offset_mode: str | None = None,
        *,
        headless: bool = False,
        capture: bool = False,
        width: int = 1280,
        height: int = 720,
        camera_kwargs: CameraCfg | Mapping[str, Any] | None = None,
    ) -> None:
        """Initialize backend-native playback rendering when available."""
        if capture:
            if not self.play_capabilities.supports_native_video_capture:
                raise NotImplementedError(
                    f"{self._backend.__class__.__name__} does not support native video capture"
                )
        elif not self.play_capabilities.supports_native_interactive_renderer:
            raise NotImplementedError(
                f"{self._backend.__class__.__name__} does not support native interactive playback"
            )
        spacing = (
            float(render_spacing) if render_spacing is not None else float(self._cfg.render_spacing)
        )
        offset_mode = (
            str(render_offset_mode)
            if render_offset_mode is not None
            else str(getattr(self._cfg, "render_offset_mode", "grid"))
        )
        self._backend.init_renderer(
            spacing=spacing,
            offset_mode=offset_mode,
            headless=bool(headless),
            capture=bool(capture),
            width=int(width),
            height=int(height),
            camera_kwargs=camera_kwargs,
        )

    def resolve_play_render_plan(
        self,
        *,
        play_render_mode: str | None,
        play_steps: int | None,
        output_video: str | PathLike[str] | None,
    ) -> BackendPlayRenderPlan:
        """Resolve high-level playback mode through the concrete backend."""
        if self._backend.backend_type == "drake" and (
            play_render_mode is None or str(play_render_mode).strip().lower() == "auto"
        ):
            play_render_mode = "record"
        return self._backend.resolve_play_render_plan(
            play_render_mode=play_render_mode,
            play_steps=play_steps,
            output_video=output_video,
        )

    def run_playback(
        self,
        *,
        initialize: Callable[[], Any],
        step: Callable[[Any], Any],
        num_steps: int | None,
        output_video: str | PathLike[str] | None = None,
        render_spacing: float | None = None,
        render_offset_mode: str | None = None,
        headless: bool | None = None,
        record_video: bool | None = None,
        frame_state_getter: Callable[[], np.ndarray] | None = None,
        camera_kwargs: CameraCfg | Mapping[str, Any] | None = None,
        debug_overlay_getter: DebugOverlayGetter | None = None,
        on_frame: Callable[[int, np.ndarray], np.ndarray | None] | None = None,
    ) -> str | None:
        """Execute playback through the concrete backend."""
        if on_frame is not None:
            raise NotImplementedError(
                f"{self.__class__.__name__} cannot forward on_frame to "
                f"{self._backend.__class__.__name__}.run_playback yet"
            )
        return cast(
            str | None,
            self._backend.run_playback(
                env=self,
                initialize=initialize,
                step=step,
                num_steps=num_steps,
                output_video=output_video,
                render_spacing=render_spacing,
                render_offset_mode=render_offset_mode,
                headless=headless,
                record_video=record_video,
                frame_state_getter=frame_state_getter,
                camera_kwargs=camera_kwargs,
                debug_overlay_getter=debug_overlay_getter,
            ),
        )

    @property
    def play_capabilities(self) -> EnvPlayCapabilities:
        capabilities = self._backend.get_play_capabilities()
        return EnvPlayCapabilities(
            supports_native_interactive_renderer=capabilities.supports_native_interactive_renderer,
            supports_physics_state_playback=capabilities.supports_physics_state_playback,
            supports_native_video_capture=capabilities.supports_native_video_capture,
            supports_debug_overlay=capabilities.supports_debug_overlay,
            supports_interactive_debug_overlay=capabilities.supports_interactive_debug_overlay,
            supports_mocap_playback=capabilities.supports_mocap_playback,
        )

    def render_play_frame(self) -> None:
        """Render one interactive playback frame through the env contract."""
        if not self.play_capabilities.supports_native_interactive_renderer:
            raise NotImplementedError(
                f"{self._backend.__class__.__name__} does not support native interactive playback"
            )
        self._backend.render()

    def render(self, mode: str = "rgb_array") -> np.ndarray:
        """Render the current state to an RGB array through the play renderer."""
        if mode != "rgb_array":
            raise NotImplementedError(
                f"{self.__class__.__name__} does not support render mode {mode!r}"
            )
        if not self.play_capabilities.supports_native_video_capture:
            raise NotImplementedError(
                f"{self._backend.__class__.__name__} does not support native video capture"
            )
        if not self._rgb_array_renderer_ready:
            self.init_play_renderer(headless=True, capture=True)
            self._rgb_array_renderer_ready = True
        return self.capture_play_video_frame()

    def capture_play_video_frame(self) -> np.ndarray:
        """Capture one detached RGB video frame through the env contract."""
        if not self.play_capabilities.supports_native_video_capture:
            raise NotImplementedError(
                f"{self._backend.__class__.__name__} does not support native video capture"
            )
        return cast(
            np.ndarray, np.asarray(self._backend.capture_video_frame(), dtype=np.uint8).copy()
        )

    def get_physics_state_snapshot(self) -> np.ndarray:
        """Return a detached physics snapshot for playback/video export."""
        if not self.play_capabilities.supports_physics_state_playback:
            raise NotImplementedError(
                f"{self._backend.__class__.__name__} does not support physics-state playback"
            )
        physics_state = cast(
            np.ndarray, np.asarray(self._backend.get_physics_state(), dtype=np.float32)
        )
        return physics_state.copy()

    def get_playback_model(self, env_index: int | None = None) -> Any:
        """Return the backend playback model for one vectorized environment."""
        return self._backend.get_playback_model(env_index)

    def get_physics_state_layout(self) -> Any:
        """Return the backend physics-state playback layout."""
        return self._backend.get_physics_state_layout()

    def get_playback_mocap_state(self, env_index: int = 0) -> tuple[np.ndarray, np.ndarray]:
        """Return detached ``(mocap_pos, mocap_quat)`` playback state."""
        if not self.play_capabilities.supports_mocap_playback:
            raise NotImplementedError(
                f"{self._backend.__class__.__name__} does not support mocap playback"
            )
        mocap_pos, mocap_quat = self._backend.get_playback_mocap_state(env_index)
        return (
            np.asarray(mocap_pos, dtype=np.float64).copy(),
            np.asarray(mocap_quat, dtype=np.float64).copy(),
        )

    def get_scene_visual_model_file(self) -> str | None:
        """Return the backend visual model file on the playback cold path."""
        return cast(str | None, self._backend.get_scene_visual_model_file())

    def set_autoreset(self, enabled: bool) -> None:
        self._autoreset = bool(enabled)

    def set_nan_guard(self, guard: TensorNanGuard) -> None:
        """Attach the owner-provided diagnostic guard.

        Detection stays device-resident. Artifact export occurs only at the
        guard's explicit abnormal-diagnostic host boundary.
        """
        from unilab.training.tensor_diagnostics import TensorNanGuard as _TensorNanGuard

        if not isinstance(guard, _TensorNanGuard):
            raise TypeError(
                f"TorchEnv nan guard must be TensorNanGuard, got {type(guard).__name__}"
            )
        self._nan_guard = guard

    def _resolve_nan_guard_model_file(self) -> str:
        """Resolve the abnormal-dump model path only on a detected failure."""
        scene = getattr(self._cfg, "scene", None)
        if isinstance(scene, SceneCfg) and scene.model_file:
            return str(scene.model_file)
        model_file = self._backend.get_scene_model_file()
        return str(model_file) if model_file else ""

    def _tensor_diagnostics_state(self) -> torch.Tensor | None:
        """Return a public backend state snapshot for abnormal diagnostics."""
        if not self._tensor_runtime_bound:
            return None
        capabilities = self._backend.get_tensor_capabilities()
        requested_fields = {"qpos", "qvel"}
        fields = tuple(sorted(requested_fields.intersection(capabilities.state_fields)))
        if not capabilities.state_views or not fields:
            return None
        views = self._backend.get_state_views(fields, device=self._device)
        values = [views[field] for field in fields]
        if not all(isinstance(value, torch.Tensor) for value in values):
            raise TypeError("backend state views must be torch.Tensor values")
        return torch.cat(
            tuple(value.reshape(self._num_envs, -1) for value in values), dim=1
        ).detach()

    def export_training_state(self) -> dict[str, Any]:
        return {"version": 1, "step_counter": self.step_counter}

    def import_training_state(self, state: Mapping[str, Any]) -> None:
        if not isinstance(state, Mapping) or set(state) != {"version", "step_counter"}:
            raise ValueError("TorchEnv training state requires version and step_counter only")
        if type(state["version"]) is not int or state["version"] != 1:
            raise ValueError("Unsupported TorchEnv training state version")
        counter = state["step_counter"]
        if type(counter) is not int or counter < 0:
            raise ValueError("TorchEnv training step_counter must be a non-negative integer")
        self.step_counter = counter

    def close(self) -> None:
        self._backend.cleanup_scene_assets()

    @abc.abstractmethod
    def apply_action(self, actions: torch.Tensor, state: TorchEnvState) -> torch.Tensor:
        """Transform an action tensor into a backend control tensor."""

    @abc.abstractmethod
    def update_state(self, state: TorchEnvState) -> TorchEnvState:
        """Compute tensor observations, reward, and termination state."""


__all__ = [
    "TorchEnv",
    "TorchEnvState",
]
