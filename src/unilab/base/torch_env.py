"""Tensor-native environment lifecycle base.

`TorchEnv` is the sole Manager-Based runtime carrier. It owns the state, step,
selected-row autoreset, timeout, finite, training-state, and backend playback
contracts.
"""

from __future__ import annotations

import abc
import dataclasses
import os
import sys
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from os import PathLike
from typing import TYPE_CHECKING, Any, Optional, cast

import gymnasium as gym
import numpy as np
import torch
from unisim.backend.base import (
    CameraCfg,
    CompletionToken,
    DebugOverlayGetter,
    DebugPrimitive,
    PlayRenderPlan,
    SimBackend,
    TensorExecution,
    torch_device_matches,
)
from unisim.backend.graph_capture import DeviceVerdicts, ExternalGraphCapture

from unilab.base.backend_timing import (
    RESET_DONE_DETAIL_TIMING_KEYS,
)
from unilab.base.autoreset_graph import AutoresetGraph
from unilab.base.base import ABEnv, EnvCfg, EnvPlayCapabilities
from unilab.base.cpu_runtime import apply_cpu_confinement
from unilab.base.reset_selection import ResetSelection
from unilab.base.scene import SceneCfg
from unilab.dtype_config import get_global_dtype

if TYPE_CHECKING:
    from unilab.training.tensor_diagnostics import NanGuard


@dataclass
class TorchEnvState:
    """Tensor-native vectorized environment transition.

    Terminal pre-reset observations are represented only by
    ``final_observation``; ``info`` never carries duplicate
    terminal-observation entries.
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


# Backends whose ``step_tensor`` publishes an event-backed asynchronous step
# receipt (a unisim ``CompletionToken`` carried in ``info["completion"]``).
# Event-backed Genesis and MJWarp CUDA paths publish a receipt per step;
# synchronous host-bridge producers publish none.
_EVENT_BACKED_STEP_RECEIPT_BACKENDS = frozenset({"genesis", "mjwarp"})


class TorchEnv(ABEnv):
    """Backend-agnostic Torch environment lifecycle.

    Concrete owners implement action transformation, tensor state update, and
    selected-row (``env_ids``) reset. The base deliberately rejects NumPy input and never
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
        apply_cpu_confinement(cfg.cpu_ids)
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
        self._pending_control_finite_check: torch.Tensor | bool | None = None
        self._reset_obs_finite_valid = torch.ones((), dtype=torch.bool, device=self._device)
        self._reset_done_count = torch.zeros((), dtype=torch.int64, device=self._device)
        self._tensor_runtime_bound = False
        self._last_initial_episode_steps: torch.Tensor | None = None
        self.step_counter = 0
        self._autoreset = True
        self._autoreset_reset_active = False
        self._autoreset_scattered_reset_obs = False
        self._rgb_array_renderer_ready = False
        self._nan_guard: "NanGuard | None" = None
        self._pending_backend_step_completion: CompletionToken | None = None
        self._defer_finite_validation = bool(getattr(cfg, "defer_finite_validation", True))
        interval = getattr(cfg, "finite_validation_interval", 1)
        if isinstance(interval, bool) or not isinstance(interval, int) or interval <= 0:
            raise ValueError("finite_validation_interval must be a positive integer")
        self._finite_validation_interval = interval
        self._finite_verdict = torch.ones((), dtype=torch.bool, device=self._device)
        # Named deferred fail-closed checks (``require``); published together
        # with ``_finite_verdict`` by the single read in poll_finite_validation.
        self._verdicts = DeviceVerdicts()
        self._configure_fixed_shape_step(cfg)

    def _configure_fixed_shape_step(self, cfg: EnvCfg) -> None:
        """Resolve the opt-in sync-free / graph-capturable step switches.

        - ``masked_autoreset``: autoreset runs every step as a fixed-shape
          ``reset(ResetSelection)`` under the device done mask (no ``any()``
          gate, no ``nonzero``); requires deferred finite validation.
        - ``persistent_step_buffers``: the working step carriers (observations,
          reward, terminated, truncated) are allocated once and updated with
          ``copy_``; consumers receive a ring of ``published_buffer_depth``
          persistent slots (``0`` publishes the working carriers themselves).
        - ``autoreset_cuda_graph``: the masked autoreset segment is captured into
          one CUDA graph after ``autoreset_graph_warmup`` eager executions and
          replayed afterwards; requires both switches above, a CUDA device and
          a backend declaring ``external_graph_capture``.
        """

        def flag(name: str) -> bool:
            value = getattr(cfg, name, False)
            if not isinstance(value, bool):
                raise TypeError(f"{name} must be bool, got {type(value).__name__}")
            return value

        def bounded_int(name: str, default: int, minimum: int) -> int:
            value = getattr(cfg, name, default)
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}, got {value!r}")
            return value

        self._masked_autoreset = flag("masked_autoreset")
        self._persistent_step_buffers = flag("persistent_step_buffers")
        self._autoreset_cuda_graph = flag("autoreset_cuda_graph")
        self._published_buffer_depth = bounded_int("published_buffer_depth", 2, 0)
        self._autoreset_graph_warmup = bounded_int("autoreset_graph_warmup", 2, 1)
        capture_mode = getattr(cfg, "autoreset_graph_capture_mode", "thread_local")
        if capture_mode not in ("global", "thread_local", "relaxed"):
            raise ValueError(
                "autoreset_graph_capture_mode must be 'global', 'thread_local' or 'relaxed'; "
                f"got {capture_mode!r}"
            )
        self._autoreset_graph_capture_mode = capture_mode
        if self._masked_autoreset and not self._defer_finite_validation:
            raise ValueError(
                "masked_autoreset requires defer_finite_validation: its fail-closed checks "
                "are published at the deferred validation boundary"
            )
        if self._autoreset_cuda_graph:
            if not (self._masked_autoreset and self._persistent_step_buffers):
                raise ValueError(
                    "autoreset_cuda_graph requires masked_autoreset and persistent_step_buffers"
                )
            if self._device.type != "cuda":
                raise ValueError(
                    f"autoreset_cuda_graph requires a CUDA env device, got {self._device}"
                )
        self._reset_selection: ResetSelection | None = (
            ResetSelection.allocate(self._num_envs, self._device)
            if self._masked_autoreset
            else None
        )
        # Device mirror of ``step_counter`` for reset participants that record
        # the trigger step inside a fixed-shape (and possibly captured) reset.
        self._step_counter_tensor = torch.zeros((), dtype=torch.int64, device=self._device)
        self._active_graph_capture: ExternalGraphCapture | None = None
        self._autoreset_graph: AutoresetGraph | None = None
        self._publication_ring: list[TorchEnvState] = []
        self._publication_cursor = 0
        self._published_state: TorchEnvState | None = None

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
        if self._backend.get_tensor_execution() is not execution:
            raise ValueError("backend tensor execution does not match its capabilities")
        if not capabilities.stepping or not capabilities.selected_reset:
            raise ValueError("TorchEnv backend must support tensor stepping and mask reset")
        if execution is TensorExecution.HOST_BRIDGE and not capabilities.packed_host_bridge:
            raise ValueError("HOST_BRIDGE TorchEnv requires a packed host-bridge plan")
        if not torch_device_matches(capabilities.torch_devices, self._device):
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
    def produces_completion_receipts(self) -> bool:
        """RL env-contract declaration of the backend's step receipts.

        ``True`` when the bound backend publishes an event-backed async
        completion receipt on every ``step`` (replay CUDA ingress then fails
        closed without one). ``False`` for synchronous producers whose step
        outputs are already ordered at return, where no receipt exists and
        none is awaited. This is a cold-path declaration; ``step`` keeps
        publishing whatever receipt the backend returns.
        """
        return self._backend.backend_type in _EVENT_BACKED_STEP_RECEIPT_BACKENDS

    @property
    def num_envs(self) -> int:
        return self._num_envs

    @property
    def state(self) -> Optional[TorchEnvState]:
        """The latest published transition.

        With ``persistent_step_buffers`` this is a publication slot whose
        tensors stay unchanged for ``published_buffer_depth - 1`` further
        steps (or, at depth 0, the working carriers that the next step
        overwrites in place).
        """
        if self._published_state is not None:
            return self._published_state
        return self._state

    @property
    def masked_autoreset(self) -> bool:
        return self._masked_autoreset

    @property
    def persistent_step_buffers(self) -> bool:
        return self._persistent_step_buffers

    @property
    def step_counter_tensor(self) -> torch.Tensor:
        """0-d int64 device mirror of ``step_counter`` (fixed address)."""
        return self._step_counter_tensor

    @property
    def autoreset_graph(self) -> AutoresetGraph | None:
        return self._autoreset_graph

    def require(self, ok: torch.Tensor, name: str, message: str) -> None:
        """Fail-closed check of a 0-d bool device verdict.

        With deferred finite validation the verdict is AND-ed into the named
        persistent flag ``name`` (no host read, CUDA-graph safe) and published
        at the next ``poll_finite_validation`` boundary, which raises
        ``ValueError(message)``. Otherwise it is read now and raises at once.
        """
        if self._defer_finite_validation:
            self._verdicts.record(name, ok, message)
            return
        if not bool(ok):
            raise ValueError(message)

    def record_host_transition(self, name: str, transition: Callable[[], None]) -> None:
        """Run a host-state update and replay it after autoreset graph replays.

        See ``unilab.base.host_transitions``. Outside a capture this simply
        runs ``transition``.
        """
        capture = self._active_graph_capture
        if capture is None:
            transition()
            return
        capture.record_host_transition(f"env.{name}", transition)

    def reset_selection_blockers(self) -> list[str]:
        """Reset participants that cannot take a ``ResetSelection``."""
        return [f"{type(self).__qualname__} does not implement a ResetSelection reset"]

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
        if self._masked_autoreset:
            blockers = self.reset_selection_blockers()
            if blockers:
                raise NotImplementedError(
                    "masked_autoreset needs every reset participant to accept a "
                    "ResetSelection; unsupported: " + "; ".join(blockers)
                )
        # terminated is statically all-ones here; the bootstrap reset must not
        # re-read the done mask from the device. The bootstrap is a cold path
        # and always uses explicit row ids.
        self._reset_done_envs(done_active=True)
        self._clear_step_final_observation()
        if self._autoreset_cuda_graph and self._autoreset_graph is None:
            self._autoreset_graph = self._build_autoreset_graph()
        if self._persistent_step_buffers:
            self._allocate_publication_ring()
            self._publish_step_state()
        state = self.state
        assert state is not None
        return state

    def _initial_episode_steps(self) -> torch.Tensor:
        """Allocate the public int64 episode counter tensor before reset."""
        return torch.zeros((self._num_envs,), dtype=torch.int64, device=self._device)

    def reset(
        self, env_ids: torch.Tensor | None = None
    ) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
        """Reset the selected environment rows; ``None`` resets all rows.

        The public selector follows the IsaacLab/mjlab ``env_ids`` row-index
        form; the v1.4 fixed-width bool mask form is removed.
        """
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
        # Action terms publish into one device-resident control verdict. Clear
        # both publication carriers before each new action is processed.
        self._control_finite_valid.fill_(True)
        self._pending_control_finite_check = None
        ctrl = self.apply_action(actions, self._state)
        guard_defer = self._defer_finite_validation
        if self._nan_guard is not None:
            guard_defer = self._defer_finite_validation or self._nan_guard.defer_validation
        self._validate_control_with_nan_guard(ctrl, defer=guard_defer)
        apply_action_ms = (time.perf_counter() - phase) * 1000.0
        apply_action_cpu_ms = (_cpu_time() - phase_cpu) * 1000.0

        phase = time.perf_counter()
        phase_cpu = _cpu_time()
        self._pending_backend_step_completion = None
        # A receipt describes only the transition being published. Remove the
        # previous one before the backend call so unsupported or failed steps
        # cannot expose stale completion metadata.
        self._state.info.pop("completion", None)
        backend_result = self.advance_physics(ctrl)
        step_core_ms = (time.perf_counter() - phase) * 1000.0
        step_core_cpu_ms = (_cpu_time() - phase_cpu) * 1000.0
        pending_completion = self._backend_step_completion(backend_result)
        # Publish the receipt before update_state so public transition consumers
        # can only read it after the update boundary succeeds.
        self._pending_backend_step_completion = pending_completion

        try:
            self._gate_backend_step_completion(pending_completion)
            self._state.truncated.fill_(False)
            self._clear_step_final_observation()

            phase = time.perf_counter()
            phase_cpu = _cpu_time()
            self._state = self.update_state(self._state)
        finally:
            self._pending_backend_step_completion = None
        if self._nan_guard is not None:
            transition_step = self.step_counter + 1
            if self._nan_guard.capture_due(transition_step):
                self._nan_guard.capture(self._tensor_diagnostics_state(), step=transition_step)
            nan_ids = self._nan_guard.check(
                self._state.obs,
                self._state.reward,
                step=transition_step,
                defer=guard_defer,
            )
            if nan_ids is not None:
                self._nan_guard.dump(nan_ids, self._resolve_nan_guard_model_file(), transition_step)
            if not self._defer_finite_validation and self._nan_guard.defer_validation:
                deferred_nan_ids = self._nan_guard.poll(step=self.step_counter, force=True)
                if deferred_nan_ids is not None:
                    self._nan_guard.dump(
                        deferred_nan_ids,
                        self._resolve_nan_guard_model_file(),
                        transition_step,
                    )
                    raise ValueError(
                        "NanGuard detected NaN or Inf for environments "
                        f"{deferred_nan_ids[:10].detach().cpu().tolist()}"
                    )
        self._validate_state(self._state)
        update_state_ms = (time.perf_counter() - phase) * 1000.0
        update_state_cpu_ms = (_cpu_time() - phase_cpu) * 1000.0
        if self.produces_completion_receipts:
            # An event-backed asynchronous step drains its device work at the
            # first update-state host read; re-attribute that measured window
            # to the backend step phase so update_state reports manager work.
            drain_ms = self._collect_step_core_drain_ms(self._state)
            if drain_ms > 0.0:
                step_core_ms += drain_ms
                update_state_ms = max(update_state_ms - drain_ms, 0.0)

        self._state.info["steps"] += 1
        self.step_counter += 1
        if self._masked_autoreset:
            self._step_counter_tensor.fill_(self.step_counter)
        truncated = self._compute_truncated(self._state)
        self._state.truncated.logical_or_(truncated)

        phase = time.perf_counter()
        phase_cpu = _cpu_time()
        # Preserve the semantic autoreset gate: an empty done mask must not
        # consume reset RNG or publish a terminal carrier. When update_state
        # made no host read of the step outputs, this gate is the first
        # synchronization point after an event-backed asynchronous backend
        # step, so its wait belongs to the backend step phase rather than to
        # reset bookkeeping.
        # The fixed-shape autoreset has no gate: it runs every step under the
        # device done mask, so nothing here reads the device.
        did_reset = self._autoreset and (
            self._masked_autoreset
            or bool((self._state.terminated | self._state.truncated).any())
        )
        step_core_ms += (time.perf_counter() - phase) * 1000.0
        step_core_cpu_ms += (_cpu_time() - phase_cpu) * 1000.0

        phase = time.perf_counter()
        phase_cpu = _cpu_time()
        if did_reset:
            if self._masked_autoreset:
                self._run_masked_autoreset()
            else:
                self._reset_done_envs(done_active=True)
        reset_done_ms = (time.perf_counter() - phase) * 1000.0
        reset_done_cpu_ms = (_cpu_time() - phase_cpu) * 1000.0
        self.poll_finite_validation()

        timing = self._state.info.setdefault("timing", {})
        if not did_reset or self._masked_autoreset:
            # The fixed-shape path has no host row count or per-phase split.
            self._clear_reset_done_detail_timing(timing)
            if did_reset:
                timing["reset_done_reset_call_ms"] = reset_done_ms
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
            completion = backend_result.get("completion")
            if completion is not None:
                self._state.info["completion"] = completion
        if self._persistent_step_buffers:
            self._publish_step_state()
        state = self.state
        assert state is not None
        return state

    @property
    def requires_substep_state_feedback(self) -> bool:
        """Whether controls must be recomputed from state between physics substeps."""
        return False

    def apply_substep_action(self) -> torch.Tensor:
        """Refresh state-dependent control before the next physics substep.

        ``TorchEnv`` consumes the previous substep's asynchronous completion,
        with a stream-ordered read gate, before calling this owner hook.
        """
        raise NotImplementedError(
            f"{type(self).__name__} must implement apply_substep_action when "
            "requires_substep_state_feedback is True"
        )

    def _backend_step_completion(self, result: dict[str, Any] | None) -> CompletionToken | None:
        """Validate one backend step receipt before a consumer can use it."""
        if result is None:
            if self.produces_completion_receipts:
                raise ValueError(
                    f"event-backed backend {self._backend.backend_type!r} step did not "
                    "publish a completion receipt"
                )
            return None
        if not isinstance(result, dict):
            raise TypeError(
                f"backend step result must be a mapping or None; got {type(result).__name__}"
            )
        completion = result.get("completion")
        if completion is None and self.produces_completion_receipts:
            raise ValueError(
                f"event-backed backend {self._backend.backend_type!r} step did not "
                "publish a completion receipt"
            )
        if completion is not None and not isinstance(completion, CompletionToken):
            raise TypeError(
                f"backend completion must be a CompletionToken; got {type(completion).__name__}"
            )
        return completion

    def _gate_backend_step_completion(self, completion: CompletionToken | None) -> None:
        """Order the current Torch consumer stream after an async step."""
        if completion is None:
            return
        stream = torch.cuda.current_stream(self._device) if self._device.type == "cuda" else None
        # The receipt owns the public dependency insertion. Passing no stream is
        # valid only for already-complete test receipts; metadata-only tokens
        # fail closed inside their implementation.
        completion.wait_before_read(stream)

    def _step_physics(self, ctrl: torch.Tensor) -> dict[str, Any] | None:
        if not self.requires_substep_state_feedback:
            result = cast(
                dict[str, Any] | None,
                self._backend.step_tensor(ctrl, self._cfg.sim_substeps),
            )
            self._backend_step_completion(result)
            return result

        backend_result: dict[str, Any] | None = None
        backend_timing: dict[str, Any] = {}
        final_completion: CompletionToken | None = None
        final_substep = self._cfg.sim_substeps - 1
        publishes_receipts: bool | None = None
        for substep in range(self._cfg.sim_substeps):
            if substep > 0:
                ctrl = self.apply_substep_action()
                self._validate_control_with_nan_guard(
                    ctrl,
                    defer=self._defer_finite_validation
                    or (self._nan_guard is not None and self._nan_guard.defer_validation),
                )
            result = self._backend.step_tensor(ctrl, 1)
            completion = self._backend_step_completion(result)
            result_publishes_receipts = completion is not None
            if publishes_receipts is None:
                publishes_receipts = result_publishes_receipts
            elif publishes_receipts != result_publishes_receipts:
                raise ValueError(
                    "backend completion receipt presence changed between physics substeps"
                )
            if result is not None:
                backend_result = result
                for key, value in result.get("timing", {}).items():
                    backend_timing[key] = backend_timing.get(key, 0.0) + value
            if substep < final_substep:
                # The next action is both a state consumer and a device-control
                # writer. Gate it on this substep before re-entering Manager code.
                self._gate_backend_step_completion(completion)
            else:
                final_completion = completion

        if backend_result is None:
            return None
        return {"timing": backend_timing, "completion": final_completion}

    def advance_physics(self, ctrl: torch.Tensor) -> dict[str, Any] | None:
        """Advance one control interval with an already validated control.

        Owners may drive intermediate physics steps through public backend APIs.
        They must advance exactly sim_substeps, may refresh the control in place
        between chunks, and leave rewards, episode counters and resets to the
        outer step lifecycle. A returned completion receipt must cover the
        entire control interval. The default delegates to :meth:`_step_physics`.
        """
        return self._step_physics(ctrl)

    def _validate_action(self, actions: torch.Tensor) -> None:
        self._validate_action_contract(actions)
        self._record_action_finite(actions)
        if not self._defer_finite_validation and not bool(self._action_finite_valid):
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

    def _validate_control_with_nan_guard(self, ctrl: torch.Tensor, *, defer: bool) -> None:
        """Validate one control, including state-feedback substeps, on the same guard path."""
        guard = self._nan_guard
        if guard is not None:
            bad_ctrl_ids = guard.check_ctrl(ctrl, step=self.step_counter, defer=defer)
            if bad_ctrl_ids is not None:
                guard.dump(bad_ctrl_ids, self._resolve_nan_guard_model_file(), self.step_counter)
        try:
            self._validate_control(ctrl)
        except ValueError:
            # A deferred diagnostic guard attached to a strict environment
            # still needs its abnormal control dump before the strict raise.
            if guard is not None and not self._defer_finite_validation:
                ctrl_nan_ids = guard.poll(step=self.step_counter, force=True)
                if ctrl_nan_ids is not None:
                    guard.dump(
                        ctrl_nan_ids, self._resolve_nan_guard_model_file(), self.step_counter
                    )
            raise

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
        # Check the final tensor after every action term and encoder-bias
        # adjustment. A term's earlier verdict cannot certify this value.
        valid = self._control_finite_valid
        valid.logical_and_(torch.isfinite(ctrl).all())
        verdict = self._publish_control_finite_verdict()
        if isinstance(verdict, torch.Tensor):
            valid.logical_and_(verdict)
        elif verdict is False:
            valid.fill_(False)
        if self._defer_finite_validation:
            self._finite_verdict.logical_and_(valid)
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
        # neither allocates a temporary concatenation nor a new scalar per step.
        for value in state.obs.values():
            self._require_floating_float(value)
        self._require_floating_float(state.reward)
        valid = self._state_finite_valid
        valid.fill_(True)
        for value in (state.reward, *state.obs.values()):
            torch.logical_and(valid, torch.isfinite(value).all(), out=valid)
        if self._defer_finite_validation:
            self._finite_verdict.logical_and_(valid)
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
        if self._defer_finite_validation:
            self._finite_verdict.logical_and_(self._action_finite_valid)

    def record_control_finite_verdict(self, verdict: torch.Tensor) -> None:
        """Accumulate an action term's device verdict before control publication."""
        if (
            not isinstance(verdict, torch.Tensor)
            or verdict.dtype != torch.bool
            or verdict.ndim != 0
        ):
            raise TypeError("TorchEnv control finite verdict must be a scalar bool tensor")
        if verdict.device != self.device:
            raise ValueError(f"TorchEnv control finite verdict device must be {self.device}")
        self._control_finite_valid.logical_and_(verdict)

    def _publish_control_finite_verdict(self) -> torch.Tensor | bool | None:
        """Publish and consume the manager control verdict, if recorded."""
        verdict = self._pending_control_finite_check
        self._pending_control_finite_check = None
        return verdict

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
            if self._defer_finite_validation:
                self._require_floating_float(value)
                self._record_finite_validation(value, f"final_observation[{name!r}]")
            else:
                self._validate_finite_float(value, f"final_observation[{name!r}]")

    def _compute_truncated(self, state: TorchEnvState) -> torch.Tensor:
        self._truncated_scratch.fill_(False)
        max_steps = self._cfg.max_episode_steps
        if max_steps is not None:
            torch.gt(state.info["steps"], max_steps - 1, out=self._truncated_scratch)
        return self._truncated_scratch

    def _reset_done_envs(self, *, done_active: bool = False) -> None:
        """Run the selected-row autoreset lifecycle.

        Eager execution retains the semantic no-done gate. Callers that already
        proved the done mask non-empty on the host (the step autoreset gate,
        the all-terminated bootstrap) pass ``done_active=True`` so the gate is
        not re-read from the device. The public reset contract speaks
        ``env_ids``; this boundary materializes the done rows once, and the
        concrete env owns the ids→mask conversion at the backend transaction
        edge (backend inputs stay fixed-width masks).
        """
        assert self._state is not None
        reset_total_started = time.perf_counter()
        detail_timing = {key: 0.0 for key in RESET_DONE_DETAIL_TIMING_KEYS}
        done = self._state.terminated | self._state.truncated
        if not done_active and not bool(done.any()):
            self._clear_reset_done_detail_timing(self._state.info.setdefault("timing", {}))
            return
        self._reset_done_count.copy_(done.sum())
        done_env_ids = torch.nonzero(done, as_tuple=False).flatten()

        terminal_started = time.perf_counter()
        # The nonzero above already published the done-row count to the host;
        # reuse it instead of reading the device counter back for timing.
        detail_timing["reset_done_count"] = float(done_env_ids.numel())
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
            new_obs, reset_info = self.reset(done_env_ids)
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

    def _run_masked_autoreset(self) -> None:
        """Execute the fixed-shape autoreset, eagerly or by graph replay."""
        graph = self._autoreset_graph
        if graph is not None:
            graph.run()
        else:
            self._masked_autoreset_body()

    def _masked_autoreset_body(self) -> None:
        """Fixed-shape autoreset: device work plus declared host transitions.

        Every tensor written here has a fixed address (persistent working
        carriers, the persistent ``ResetSelection`` mask, final-observation
        scratch), nothing reads the device from the host, and host state that
        the rest of the step changes is updated only through
        ``record_host_transition``. That makes the body capturable as one CUDA
        graph (``autoreset_cuda_graph``) with no eager epilogue.
        """
        assert self._state is not None
        selection = self._reset_selection
        assert selection is not None
        state = self._state
        done = selection.mask
        torch.logical_or(state.terminated, state.truncated, out=done)
        self._reset_done_count.copy_(done.sum())
        state.info["steps"].masked_fill_(done, 0)
        final_observation = self._ensure_final_observation_scratch()
        for name, values in state.obs.items():
            final_observation[name].copy_(values)
        # step() clears final_observation before update_state on every step.
        self.record_host_transition("final_observation", self._publish_final_observation)
        self._autoreset_reset_active = True
        self._autoreset_scattered_reset_obs = False
        try:
            new_obs, reset_info = self.reset(selection)  # type: ignore[arg-type]
        finally:
            self._autoreset_reset_active = False
        if not self._autoreset_scattered_reset_obs:
            self._validate_reset_observation_mask(new_obs)
            for name, values in new_obs.items():
                state.obs[name].copy_(torch.where(done[:, None], values, state.obs[name]))
        self._scatter_reset_info_mask(reset_info, done)

    def _publish_final_observation(self) -> None:
        assert self._state is not None
        self._state.final_observation = self._final_observation_scratch

    def _build_autoreset_graph(self) -> AutoresetGraph:
        capabilities = self._backend.get_tensor_capabilities()
        if not getattr(capabilities, "external_graph_capture", False):
            raise NotImplementedError(
                f"autoreset_cuda_graph requires backend '{self._backend.backend_type}' to "
                "declare the external_graph_capture tensor capability"
            )
        return AutoresetGraph(
            backend=self._backend,
            device=self._device,
            body=self._masked_autoreset_body,
            bind_capture=self._bind_graph_capture,
            generators=self._graph_generators(),
            warmup=self._autoreset_graph_warmup,
            capture_error_mode=self._autoreset_graph_capture_mode,
        )

    def _bind_graph_capture(self, capture: ExternalGraphCapture | None) -> None:
        self._active_graph_capture = capture

    def _graph_generators(self) -> list[torch.Generator]:
        """CUDA generators consumed by the autoreset segment (besides the default)."""
        return []

    def _allocate_publication_ring(self) -> None:
        assert self._state is not None
        if self._publication_ring or self._published_buffer_depth == 0:
            return
        state = self._state
        self._publication_ring = [
            TorchEnvState(
                obs={name: torch.empty_like(value) for name, value in state.obs.items()},
                reward=torch.empty_like(state.reward),
                terminated=torch.empty_like(state.terminated),
                truncated=torch.empty_like(state.truncated),
                info=state.info,
            )
            for _ in range(self._published_buffer_depth)
        ]

    def _publish_step_state(self) -> None:
        """Publish the working carriers into the next persistent ring slot."""
        state = self._state
        assert state is not None
        if self._published_buffer_depth == 0:
            self._published_state = state
            return
        slot = self._publication_ring[self._publication_cursor]
        self._publication_cursor = (self._publication_cursor + 1) % self._published_buffer_depth
        for name, value in state.obs.items():
            slot.obs[name].copy_(value)
        slot.reward.copy_(state.reward)
        slot.terminated.copy_(state.terminated)
        slot.truncated.copy_(state.truncated)
        slot.info = state.info
        slot.final_observation = state.final_observation
        self._published_state = slot

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
        if self._defer_finite_validation:
            self._finite_verdict.logical_and_(valid)
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

    def _collect_step_core_drain_ms(self, state: TorchEnvState) -> float:
        """Return the update-state window that drained an async backend step.

        Only consulted when the backend publishes event-backed completion
        receipts (``produces_completion_receipts``); synchronous backends drain
        nothing, so the default is zero. An env whose update-state sub-phase
        hosts the first read of asynchronous step outputs overrides this to
        report that measured window.
        """
        del state
        return 0.0

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
        self._backend.bind_renderer(
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
    ) -> PlayRenderPlan:
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

    @property
    def defer_finite_validation(self) -> bool:
        """Whether finite checks accumulate on the device between polls."""
        return self._defer_finite_validation

    @property
    def finite_validation_interval(self) -> int:
        """Number of steps between deferred finite-validation polls."""
        return self._finite_validation_interval

    def _record_finite_validation(self, value: torch.Tensor, label: str = "") -> None:
        """Accumulate one finite verdict without a scalar host read."""
        del label
        if self._defer_finite_validation:
            self._finite_verdict.logical_and_(torch.isfinite(value).all())

    def record_finite_validation(self, value: torch.Tensor, label: str = "") -> None:
        """Allow a manager to add a finite verdict to the shared boundary."""
        self._record_finite_validation(value, label)

    def poll_finite_validation(self, *, force: bool = False) -> bool:
        """Publish all deferred verdicts at one fixed host boundary."""
        if not self._defer_finite_validation:
            return True
        if not force and self.step_counter % self._finite_validation_interval != 0:
            return True

        verdicts = [self._finite_verdict]
        managers: list[Any] = []
        for name in (
            "action_manager",
            "reward_manager",
            "observation_manager",
            "command_manager",
            "metrics_manager",
        ):
            manager = getattr(self, name, None)
            verdict = getattr(manager, "finite_verdict", None)
            if isinstance(verdict, torch.Tensor):
                verdicts.append(verdict)
                managers.append(manager)

        guard = self._nan_guard
        guard_bad: torch.Tensor | None = None
        if guard is not None:
            bad_verdict = guard.device_verdict
            if isinstance(bad_verdict, torch.Tensor):
                guard_bad = bad_verdict
                verdicts.append(torch.logical_not(bad_verdict))
        named_sets = self._named_verdict_sets()
        for named in named_sets:
            verdicts.extend(named.device_flags())

        valid = bool(torch.stack(verdicts).all().detach().cpu().item())
        if valid:
            self._finite_verdict.fill_(True)
            for named in named_sets:
                named.clear()
            for manager in managers:
                reset = getattr(manager, "reset_finite_verdict", None)
                if callable(reset):
                    reset()
                else:
                    manager.finite_verdict.fill_(True)
            if guard is not None and guard_bad is not None:
                guard.reset_verdict()
            return True

        # Detailed scans and diagnostics are reserved for the abnormal path.
        for named in named_sets:
            # Raises ValueError naming every failed fail-closed check.
            named.flush()
        base_valid = bool(self._finite_verdict.detach().cpu().item())
        self._finite_verdict.fill_(True)
        if guard is not None and guard_bad is not None:
            nan_ids = guard.poll(step=self.step_counter, force=True)
            if nan_ids is not None:
                guard.dump(nan_ids, self._resolve_nan_guard_model_file(), self.step_counter)
                raise ValueError(
                    "NanGuard detected NaN or Inf at validation boundary for environments "
                    f"{nan_ids[:10].detach().cpu().tolist()}"
                )
        handled_nonfatal = False
        for manager in managers:
            poll = getattr(manager, "poll_finite_validation", None)
            if callable(poll) and poll() is False:
                handled_nonfatal = True
        if handled_nonfatal and base_valid:
            return True
        raise ValueError("TorchEnv deferred finite validation detected NaN or Inf")

    def _named_verdict_sets(self) -> list[DeviceVerdicts]:
        """Env-owned and backend-owned named verdicts, for the single poll read."""
        sets = [self._verdicts]
        backend_verdicts = getattr(self._backend, "tensor_validation_verdicts", None)
        if callable(backend_verdicts):
            backend_set = backend_verdicts()
            if isinstance(backend_set, DeviceVerdicts):
                sets.append(backend_set)
        return [named for named in sets if len(named)]

    def set_nan_guard(self, guard: NanGuard) -> None:
        """Attach the owner-provided diagnostic guard.

        Detection stays device-resident. Artifact export occurs only at the
        guard's explicit abnormal-diagnostic host boundary.
        """
        from unilab.training.tensor_diagnostics import NanGuard as _NanGuard

        if not isinstance(guard, _NanGuard):
            raise TypeError(f"TorchEnv nan guard must be NanGuard, got {type(guard).__name__}")
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
        ``unilab.rl`` off-policy collector publishes it into the run
        manifest). The
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
        try:
            # A short final cycle still owns device verdicts. Preserve an
            # existing training error when close runs from a finally block.
            if sys.exc_info()[0] is None:
                self.poll_finite_validation(force=True)
        finally:
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
