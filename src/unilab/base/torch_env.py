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
    GraphCompletionToken,
    SimBackend,
    TensorExecution,
    tensor_device_matches,
)

from unilab.base.backend_timing import (
    RESET_DONE_DETAIL_TIMING_KEYS,
)
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


def _cpu_time() -> float:
    process = os.times()
    return process.user + process.system


class TorchEnv(ABEnv):
    """Backend-agnostic Torch environment lifecycle.

    Concrete owners implement action transformation, tensor state update, and
    fixed-width mask reset. The base deliberately rejects NumPy input and never
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
        self._dtype = get_global_dtype()
        self._state: Optional[TorchEnvState] = None
        self._truncated_scratch = torch.zeros((num_envs,), dtype=torch.bool, device=self._device)
        self._final_observation_scratch: dict[str, torch.Tensor] | None = None
        self._action_finite_valid = torch.ones((), dtype=torch.bool, device=self._device)
        self._control_finite_valid = torch.ones((), dtype=torch.bool, device=self._device)
        self._state_finite_valid = torch.ones((), dtype=torch.bool, device=self._device)
        self._reset_obs_finite_valid = torch.ones((), dtype=torch.bool, device=self._device)
        self._reset_done_count = torch.zeros((), dtype=torch.int64, device=self._device)
        self._tensor_runtime_bound = False
        self._last_initial_episode_steps: torch.Tensor | None = None
        self.step_counter = 0
        self._autoreset = True
        self._autoreset_reset_active = False
        self._autoreset_scattered_reset_obs = False
        self._rgb_array_renderer_ready = False
        self._nan_guard: "TensorNanGuard | None" = None
        self._defer_validation_at_boundaries = False
        self._pending_backend_step_completion: GraphCompletionToken | None = None

    @property
    def device(self) -> torch.device:
        return self._device

    def _bind_tensor_runtime(self) -> None:
        if self._tensor_runtime_bound:
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
            raise ValueError("TorchEnv backend must support tensor stepping and mask reset")
        if execution is TensorExecution.HOST_BRIDGE and not capabilities.packed_host_bridge:
            raise ValueError("HOST_BRIDGE TorchEnv requires a packed host-bridge plan")
        if (
            execution is TensorExecution.HOST_BRIDGE
            and self._device.type == "cuda"
            and not capabilities.graph_completion_publication
        ):
            raise ValueError(
                "CUDA HOST_BRIDGE TorchEnv requires graph_completion_publication so "
                "downstream replay publication has an ordered public completion receipt"
            )
        if not tensor_device_matches(capabilities.torch_devices, self._device):
            raise ValueError(
                f"backend did not accept Torch device {str(self._device)!r}; "
                f"supported devices are {capabilities.torch_devices}"
            )
        self._tensor_runtime_bound = True

    @property
    def cfg(self) -> EnvCfg:
        return self._cfg

    @property
    def backend(self) -> SimBackend:
        """Public backend owner used by Manager terms and tensor contracts."""
        return self._backend

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
        action_shape = self.action_space.shape
        if action_shape is not None:
            # The public action carrier contract is fixed. Validate it once at
            # this cold boundary; ActionTerm.process_actions owns the per-step
            # value check on the authoritative control tensor.
            self._validate_action(
                torch.zeros(
                    (self._num_envs, *action_shape),
                    dtype=self._dtype,
                    device=self._device,
                )
            )
        initial_steps = self._initial_episode_steps()
        # Preserve the initial draw across the bootstrap reset; a subclass may
        # need it to restore randomized initial offsets after reset clears the
        # public counters.
        obs = {
            name: torch.zeros((self._num_envs, dim), dtype=self._dtype, device=self._device)
            for name, dim in self.obs_groups_spec.items()
        }
        self._state = TorchEnvState(
            obs=obs,
            reward=torch.zeros((self._num_envs,), dtype=self._dtype, device=self._device),
            terminated=torch.ones((self._num_envs,), dtype=torch.bool, device=self._device),
            truncated=torch.zeros((self._num_envs,), dtype=torch.bool, device=self._device),
            info={"steps": initial_steps},
        )
        self._last_initial_episode_steps = initial_steps.clone()
        self._reset_done_envs()
        self._clear_step_final_observation()
        return self._state

    def _initial_episode_steps(self) -> torch.Tensor:
        """Allocate the public int64 episode counter tensor before reset."""
        return torch.zeros((self._num_envs,), dtype=torch.int64, device=self._device)

    def reset(
        self, reset_mask: torch.Tensor | None = None
    ) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
        """Reset the true rows of a full-batch mask; ``None`` resets all rows."""
        raise NotImplementedError(f"{type(self).__name__} must define a tensor reset lifecycle")

    def step(self, actions: torch.Tensor) -> TorchEnvState:
        started = time.perf_counter()
        cpu_started = _cpu_time()
        phase = time.perf_counter()
        self._validate_action_contract(actions)
        self._record_action_finite(actions)
        action_validate_ms = (time.perf_counter() - phase) * 1000.0
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

        phase = time.perf_counter()
        phase_cpu = _cpu_time()
        self._pending_backend_step_completion = None
        try:
            backend_result = self._backend.step_tensor(ctrl, self._cfg.sim_substeps)
        except BaseException:
            # Do not expose the previous transition's completion receipt as if
            # this failed replay had published a new transition.
            self._state.info.pop("graph_completion", None)
            raise
        step_core_ms = (time.perf_counter() - phase) * 1000.0
        step_core_cpu_ms = (_cpu_time() - phase_cpu) * 1000.0
        if isinstance(backend_result, dict):
            pending_completion = backend_result.get("graph_completion")
            if pending_completion is not None and not isinstance(
                pending_completion, GraphCompletionToken
            ):
                self._state.info.pop("graph_completion", None)
                raise TypeError(
                    "backend graph_completion must be a GraphCompletionToken; got "
                    f"{type(pending_completion).__name__}"
                )
            # Publish the receipt to the owner before update_state. The update
            # graph needs this ordering dependency before replay, while public
            # transition consumers still read it only after the update succeeds.
            self._pending_backend_step_completion = pending_completion

        # Transition publication starts only after the backend completion
        # boundary succeeds. A failed replay therefore leaves the previous
        # public final observation and truncation carrier untouched for RNG
        # rollback and explicit eager fallback.
        self._state.truncated.fill_(False)
        self._clear_step_final_observation()

        phase = time.perf_counter()
        phase_cpu = _cpu_time()
        try:
            self._state = self.update_state(self._state)
        finally:
            self._pending_backend_step_completion = None
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
        if self._defer_validation_at_boundaries:
            self.publish_validation_verdicts()
        update_state_ms = (time.perf_counter() - phase) * 1000.0
        update_state_cpu_ms = (_cpu_time() - phase_cpu) * 1000.0

        self._state.info["steps"] += 1
        self.step_counter += 1
        truncated = self._compute_truncated(self._state)
        self._state.truncated.logical_or_(truncated)

        phase = time.perf_counter()
        phase_cpu = _cpu_time()
        # Preserve the semantic autoreset gate in both modes: an empty done
        # mask must not consume reset RNG or publish a terminal carrier.
        did_reset = self._autoreset and bool((self._state.terminated | self._state.truncated).any())
        if did_reset:
            self._reset_done_envs()
        reset_done_ms = (time.perf_counter() - phase) * 1000.0
        reset_done_cpu_ms = (_cpu_time() - phase_cpu) * 1000.0

        timing = self._state.info.setdefault("timing", {})
        if not did_reset:
            self._clear_reset_done_detail_timing(timing)
        timing["env_step_total_ms"] = (time.perf_counter() - started) * 1000.0
        timing["action_validate_ms"] = action_validate_ms
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
            completion = backend_result.get("graph_completion")
            if completion is not None:
                if "graph_completion" in self._state.info:
                    # Preserve the independent owner update receipt while keeping
                    # the backend dependency publicly inspectable.
                    self._state.info["backend_step_completion"] = completion
                else:
                    self._state.info["graph_completion"] = completion
        return self._state

    def _validate_action(self, actions: torch.Tensor) -> None:
        self._validate_action_contract(actions)
        self._record_action_finite(actions)
        valid = self._action_finite_valid
        if self._defer_validation_at_boundaries:
            return
        if not bool(valid):
            raise ValueError("TorchEnv action contains NaN or Inf")

    def _validate_action_contract(self, actions: torch.Tensor) -> None:
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
        valid = torch.isfinite(ctrl).all()
        self._control_finite_valid.copy_(valid)
        if self._defer_validation_at_boundaries:
            return
        if not bool(valid):
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
        self._validate_vector(state.reward, "reward", self._dtype)
        self._validate_vector(state.terminated, "terminated", torch.bool)
        self._validate_vector(state.truncated, "truncated", torch.bool)
        steps = state.info.get("steps")
        self._validate_vector(steps, "info['steps']", torch.int64)
        self._validate_final_observation(state.final_observation)
        # Aggregate all public floating outputs into one persistent verdict.
        # Structural checks above own dtype/shape/device validation; this path
        # neither allocates a temporary concatenation nor synchronizes in
        # deferred-validation mode.
        for value in state.obs.values():
            self._require_floating_float(value)
        self._require_floating_float(state.reward)
        valid = self._state_finite_valid
        valid.fill_(True)
        for value in (state.reward, *state.obs.values()):
            torch.logical_and(valid, torch.isfinite(value).all(), out=valid)
        if self._defer_validation_at_boundaries:
            return
        if not bool(valid):
            for label, value in (
                *((f"obs[{name!r}]", value) for name, value in state.obs.items()),
                ("reward", state.reward),
            ):
                if not bool(torch.isfinite(value).all()):
                    raise ValueError(f"TorchEnvState {label} contains NaN or Inf")

    def _validate_finite_float(self, value: torch.Tensor, label: str) -> None:
        if not value.is_floating_point():
            raise TypeError(f"TorchEnvState {label} must be floating-point")
        if not bool(torch.isfinite(value).all()):
            raise ValueError(f"TorchEnvState {label} contains NaN or Inf")

    def _record_action_finite(self, actions: torch.Tensor) -> None:
        """Record the public action finite verdict on its persistent carrier."""
        self._action_finite_valid.copy_(torch.isfinite(actions).all())

    def _validation_flags(self) -> dict[str, torch.Tensor]:
        """Return persistent device-resident verdicts for interval publication.

        These flags are diagnostic carriers, not authorization to skip the
        fail-closed checks above. They retain the latest aggregate result for a
        completion boundary without allocating a new scalar per validation.
        """
        return {
            "action_finite": self._action_finite_valid,
            "control_finite": self._control_finite_valid,
            "state_finite": self._state_finite_valid,
            "reset_obs_finite": self._reset_obs_finite_valid,
        }

    def publish_validation_verdicts(self) -> None:
        """Fail closed on persistent finite verdicts at a completion boundary."""
        flags = self._validation_flags()
        failed = [name for name, verdict in flags.items() if not bool(verdict)]
        if failed:
            labels = {
                "action_finite": "TorchEnv action contains NaN or Inf",
                "control_finite": "TorchEnv control contains NaN or Inf",
                "state_finite": "TorchEnvState contains NaN or Inf",
                "reset_obs_finite": "TorchEnv mask reset obs contains NaN or Inf",
            }
            messages = [labels[name] for name in failed if name in labels]
            raise ValueError("; ".join(messages))

    @property
    def deferred_validation_boundaries(self) -> bool:
        """Whether eager finite verdicts may defer to step/reset completion."""
        return self._defer_validation_at_boundaries

    def set_deferred_validation_boundaries(self, enabled: bool) -> None:
        """Select completion-boundary publication for graph-candidate runs.

        When enabled, eager structural checks remain synchronous, while finite
        verdicts stay on their persistent device carriers until an explicit
        completion boundary publishes them. This is an opt-in integration-only
        contract and does not sanitize invalid values.
        """
        if not isinstance(enabled, bool):
            raise TypeError(f"deferred validation boundaries must be bool, got {type(enabled)}")
        self._defer_validation_at_boundaries = enabled

    def _require_floating_float(self, value: torch.Tensor) -> None:
        if not value.is_floating_point():
            raise TypeError("TorchEnvState floating outputs must be floating-point")

    def _validate_vector(self, value: Any, label: str, dtype: torch.dtype) -> None:
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"TorchEnvState {label} must be a torch.Tensor")
        if value.shape != (self._num_envs,):
            raise ValueError(f"TorchEnvState {label} has shape {tuple(value.shape)}")
        if value.dtype != dtype:
            raise TypeError(f"TorchEnvState {label} dtype must be {dtype}, got {value.dtype}")
        if value.device != self.device:
            raise ValueError(f"TorchEnvState {label} device must be {self.device}")

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
        """Run the fixed-shape mask-driven autoreset lifecycle.

        The graph-candidate empty-mask path is fixed-width. Eager execution
        retains the semantic no-done gate; backends and managers own mask
        semantics and no reset env-id row carrier is materialized.
        """
        assert self._state is not None
        reset_total_started = time.perf_counter()
        detail_timing = {key: 0.0 for key in RESET_DONE_DETAIL_TIMING_KEYS}
        done = self._state.terminated | self._state.truncated
        if not self._defer_validation_at_boundaries and not bool(done.any()):
            self._clear_reset_done_detail_timing(self._state.info.setdefault("timing", {}))
            return
        self._reset_done_count.copy_(done.sum())

        terminal_started = time.perf_counter()
        detail_timing["reset_done_count"] = float(self._reset_done_count)
        self._state.info["steps"].masked_fill_(done, 0)
        final_observation = self._ensure_final_observation_scratch()
        for name, values in self._state.obs.items():
            final_observation[name].copy_(values)
        self._state.final_observation = final_observation
        detail_timing["reset_done_terminal_obs_ms"] = (
            time.perf_counter() - terminal_started
        ) * 1000.0

        reset_started = time.perf_counter()
        self._autoreset_reset_active = True
        self._autoreset_scattered_reset_obs = False
        try:
            new_obs, reset_info = self.reset(done)
        finally:
            self._autoreset_reset_active = False
        detail_timing["reset_done_reset_call_ms"] = (time.perf_counter() - reset_started) * 1000.0
        collected = self._collect_reset_backend_timing_ms()
        detail_timing.update(
            {key: value for key, value in collected.items() if key in detail_timing}
        )

        scatter_started = time.perf_counter()
        if not self._autoreset_scattered_reset_obs:
            self._validate_reset_observation_mask(new_obs)
            for name, values in new_obs.items():
                self._state.obs[name].copy_(
                    torch.where(done[:, None], values, self._state.obs[name])
                )
        self._scatter_reset_info_mask(reset_info, done)
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

    def _validate_reset_observation_mask(self, obs: Mapping[str, torch.Tensor]) -> None:
        if not isinstance(obs, dict) or set(obs) != set(self.obs_groups_spec):
            raise ValueError("TorchEnv mask reset observations do not match obs_groups_spec")
        for name, dim in self.obs_groups_spec.items():
            value = obs[name]
            if not isinstance(value, torch.Tensor):
                raise TypeError(f"TorchEnv mask reset obs[{name!r}] must be a torch.Tensor")
            if value.dtype != self._dtype:
                raise TypeError(
                    f"TorchEnv mask reset obs[{name!r}] dtype must be {self._dtype}; "
                    f"got {value.dtype}"
                )
            if value.shape != (self._num_envs, dim):
                raise ValueError(
                    f"TorchEnv mask reset obs[{name!r}] has shape {tuple(value.shape)}; "
                    f"expected {(self._num_envs, dim)}"
                )
            if value.device != self.device:
                raise ValueError(f"TorchEnv mask reset obs[{name!r}] device must be {self.device}")
            self._require_floating_float(value)
        valid = self._reset_obs_finite_valid
        valid.fill_(True)
        for value in obs.values():
            torch.logical_and(valid, torch.isfinite(value).all(), out=valid)
        if self._defer_validation_at_boundaries:
            return
        if not bool(valid):
            for name, value in obs.items():
                if not bool(torch.isfinite(value).all()):
                    raise ValueError(f"TorchEnv mask reset obs[{name!r}] contains NaN or Inf")

    def _scatter_reset_info_mask(
        self, reset_info: Mapping[str, Any], reset_mask: torch.Tensor
    ) -> None:
        assert self._state is not None
        for key, value in reset_info.items():
            current = self._state.info.get(key)
            if not isinstance(value, torch.Tensor) or value.shape[:1] != (self._num_envs,):
                continue
            if not isinstance(current, torch.Tensor) or current.shape != value.shape:
                current = torch.zeros_like(value, device=self.device)
                self._state.info[key] = current
            selected_reset_mask = reset_mask if value.ndim == 1 else reset_mask[:, None]
            current.copy_(torch.where(selected_reset_mask, value.to(device=self.device), current))

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

    def get_backend_runtime_diagnostics(self) -> dict[str, Any]:
        """Serialize the backend's tensor runtime diagnostics on the cold path.

        This is the collection boundary consumed by training runtimes (the
        uni_rl off-policy collector publishes it into the run manifest). The
        payload is plain JSON: ``available`` and ``backend_type`` plus, when
        available, one ``diagnostics`` entry per backend optimization with
        ``requested``/``enabled``/``disable_reason``. Backends that declare
        no optional tensor runtime optimization, or whose diagnostic probe
        fails, record an explicit unavailable marker instead of a null field.
        """
        payload: dict[str, Any] = {"backend_type": str(self._backend.backend_type)}
        try:
            diagnostics = self._backend.get_tensor_runtime_diagnostics()
        except Exception as exc:
            payload["available"] = False
            payload["unavailable_reason"] = f"{type(exc).__name__}: {exc}"
            return payload
        if not diagnostics:
            payload["available"] = False
            payload["unavailable_reason"] = (
                f"backend {self._backend.backend_type} declares no tensor runtime diagnostics"
            )
            return payload
        payload["available"] = True
        payload["diagnostics"] = {
            name: dataclasses.asdict(diagnostic) for name, diagnostic in diagnostics.items()
        }
        return payload

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
