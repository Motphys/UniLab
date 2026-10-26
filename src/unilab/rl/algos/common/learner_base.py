"""Algorithm-agnostic learner base pieces shared by the off-policy SAC learners.

``LearnerBaseMixin`` centralizes the pieces of ``SACLearner`` and
``FlashSACLearner`` that are identical between the two: AMP dtype/scaler
resolution, autocast context, observation-normalizer updates, gradient-sync
plumbing, and loss-method compilation. The mixin only declares the attribute
contract; consuming learners own the attributes.
"""

from __future__ import annotations

import copy
import os
import warnings
from collections.abc import Callable, Iterable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, cast

import torch
import torch.nn as nn
import torch.optim as optim

from unilab.rl.algos.common.normalization import EmpiricalNormalization
from unilab.rl.runtime.device import sync_device
from unilab.rl.runtime.offpolicy.warmup import (
    OffPolicyWarmupContext,
    build_offpolicy_warmup_batch,
    capture_rng_state,
    restore_rng_state,
)


def polyak_update_target(target: nn.Module, source: nn.Module, tau: float | torch.Tensor) -> None:
    """Polyak-average ``target`` parameters toward ``source`` in place.

    ``tau`` may be a device scalar.  The CUDA whole-cycle graph uses that form
    to keep the finite-update gate on-device.
    """
    with torch.no_grad():
        target_params = cast(list[torch.Tensor], list(target.parameters()))
        source_params = cast(list[torch.Tensor], list(source.parameters()))
        if isinstance(tau, torch.Tensor):
            # Tensor-weight foreach_lerp_ performs a prohibited host sync while
            # CUDA graph capture is active.  Per-parameter lerp_ is capture-safe
            # and preserves the finite-gated device scalar.
            for tgt, src in zip(target_params, source_params, strict=True):
                tgt.lerp_(src, tau)
        else:
            try:
                torch._foreach_lerp_(target_params, source_params, tau)
            except RuntimeError:
                for tgt, src in zip(target_params, source_params, strict=True):
                    tgt.lerp_(src, tau)


def fused_adam_supported(device_type: str) -> bool:
    """Whether this torch build provides a fused Adam/AdamW kernel for the device.

    Fused kernels collapse the per-parameter host loop into one launch; on MPS
    the single-tensor fallback additionally performs a blocking `.item()` per
    parameter per step, which dominates learner time there.  CUDA has always
    been supported; MPS fused kernels ship with torch >= 2.6.  Other devices
    keep the default path.
    """
    if device_type == "cuda":
        return True
    if device_type != "mps":
        return False
    try:
        from torch.utils._foreach_utils import (  # pyright: ignore[reportPrivateImportUsage]
            _get_fused_kernels_supported_devices,
        )
    except ImportError:
        return False
    return device_type in _get_fused_kernels_supported_devices()


_MPS_BF16_AUTOCAST_SUPPORTED: bool | None = None


@dataclass(frozen=True)
class AmpPolicy:
    """Per-device AMP enablement and validation policy."""

    enabled: bool
    dtypes: frozenset[torch.dtype]
    auto_dtype: torch.dtype | None
    grad_scaler_dtypes: frozenset[torch.dtype] = frozenset()
    explicit_only: bool = False
    requires_runtime_probe: bool = False
    compile_mutually_exclusive: bool = False

    def manifest(self, amp_dtype: str) -> dict[str, object]:
        return {
            "enabled": self.enabled,
            "requested_dtype": amp_dtype,
            "supported_dtypes": [str(dtype).removeprefix("torch.") for dtype in self.dtypes],
            "grad_scaler_dtypes": [
                str(dtype).removeprefix("torch.") for dtype in self.grad_scaler_dtypes
            ],
            "auto_dtype": (
                str(self.auto_dtype).removeprefix("torch.") if self.auto_dtype is not None else None
            ),
            "explicit_only": self.explicit_only,
            "requires_runtime_probe": self.requires_runtime_probe,
            "evidence": "policy",
        }


_AMP_POLICIES: dict[str, AmpPolicy] = {
    "cuda": AmpPolicy(
        enabled=True,
        dtypes=frozenset((torch.float16, torch.bfloat16)),
        auto_dtype=torch.bfloat16,
        grad_scaler_dtypes=frozenset((torch.float16,)),
    ),
    "mps": AmpPolicy(
        enabled=True,
        dtypes=frozenset((torch.bfloat16,)),
        auto_dtype=None,
        explicit_only=True,
        requires_runtime_probe=True,
        compile_mutually_exclusive=True,
    ),
    "xpu": AmpPolicy(
        enabled=True,
        dtypes=frozenset((torch.bfloat16,)),
        auto_dtype=torch.bfloat16,
    ),
    "cpu": AmpPolicy(enabled=False, dtypes=frozenset(), auto_dtype=None),
}


@dataclass(frozen=True)
class AmpSelection:
    """Resolved AMP enablement for one learner construction."""

    policy: AmpPolicy
    use_amp: bool
    requested_dtype: str

    @property
    def dtype(self) -> torch.dtype | None:
        if not self.use_amp:
            return None
        normalized = self.requested_dtype.strip().lower()
        if normalized == "fp16":
            return torch.float16
        if normalized == "bf16":
            return torch.bfloat16
        return self.policy.auto_dtype

    def manifest(self) -> dict[str, object]:
        resolved_dtype = self.dtype
        return {
            **self.policy.manifest(self.requested_dtype),
            "use_amp": self.use_amp,
            "resolved_dtype": (
                str(resolved_dtype).removeprefix("torch.") if resolved_dtype is not None else None
            ),
        }


def resolve_amp_policy(device_type: str) -> AmpPolicy:
    """Return the per-device AMP policy, failing closed for unknown devices."""
    try:
        return _AMP_POLICIES[device_type]
    except KeyError as exc:
        raise ValueError(f"Unsupported learner AMP device type: {device_type!r}") from exc


def resolve_amp_selection(
    *,
    device_type: str,
    use_amp: bool,
    amp_dtype: str,
    learner_name: str,
) -> AmpSelection:
    """Resolve AMP enablement from the single cross-learner policy table."""
    policy = resolve_amp_policy(device_type)
    normalized_amp_dtype = amp_dtype.strip().lower()
    if normalized_amp_dtype not in {"auto", "fp16", "bf16"}:
        raise ValueError(f"Unsupported {learner_name} AMP dtype: {amp_dtype!r}")
    enabled = bool(use_amp and policy.enabled)
    if enabled and policy.explicit_only and normalized_amp_dtype == "auto":
        enabled = False
    resolved_dtype = {
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
    }.get(normalized_amp_dtype, policy.auto_dtype)
    if enabled and resolved_dtype not in policy.dtypes:
        supported = ", ".join(
            str(dtype).removeprefix("torch.") for dtype in sorted(policy.dtypes, key=str)
        )
        raise ValueError(
            f"{learner_name} {device_type.upper()} AMP requires one of the explicitly "
            f"validated dtypes {supported}; got amp_dtype={amp_dtype!r}"
        )
    return AmpSelection(
        policy=policy,
        use_amp=enabled,
        requested_dtype=amp_dtype,
    )


def validate_amp_compile_paths(
    *,
    device_type: str,
    use_compile: bool,
    use_amp: bool,
    learner_name: str,
) -> None:
    """Reject initially unsupported compile/AMP combinations."""
    if resolve_amp_policy(device_type).compile_mutually_exclusive and use_compile and use_amp:
        raise ValueError(
            f"{learner_name} MPS torch.compile and BF16 AMP are initially mutually "
            "exclusive; enable exactly one path"
        )


def mps_bf16_autocast_supported() -> bool:
    """Probe whether this Torch/MPS runtime can execute BF16 autocast ops.

    The result is cached for the process.  A failed probe reports
    "unsupported", it does not raise: callers decide whether to fail closed
    for an explicit BF16 request.
    """
    global _MPS_BF16_AUTOCAST_SUPPORTED
    if _MPS_BF16_AUTOCAST_SUPPORTED is not None:
        return _MPS_BF16_AUTOCAST_SUPPORTED
    supported = False
    if torch.backends.mps.is_available():
        try:
            probe = torch.nn.Sequential(
                torch.nn.Linear(8, 8, device="mps"),
                torch.nn.LayerNorm(8, device="mps"),
                torch.nn.Linear(8, 8, device="mps"),
            )
            values = torch.ones(8, 8, device="mps", requires_grad=True)
            with torch.autocast(device_type="mps", dtype=torch.bfloat16, enabled=True):
                outputs = probe(values)
                loss = outputs.square().mean()
            loss.backward()
            sync_device("mps")
            supported = (
                outputs.dtype == torch.bfloat16
                and bool(torch.isfinite(outputs).all().item())
                and all(
                    bool(torch.isfinite(parameter.grad).all().item())
                    for parameter in probe.parameters()
                    if parameter.grad is not None
                )
            )
        except (NotImplementedError, RuntimeError, TypeError, ValueError):
            supported = False
    _MPS_BF16_AUTOCAST_SUPPORTED = supported
    return supported


def resolve_finite_check_flags(device_type: str, device_gated: bool) -> tuple[bool, bool]:
    """Return ``(host_finite_checks, metrics_finite_checks)`` for a learner.

    ``device_gated`` marks the CUDA compiled path whose fused kernels skip
    non-finite steps on-device.  MPS has no equivalent (its fused AdamW kernel
    ignores ``found_inf``), so host checks there are deferred to the
    metrics-reading update of each iteration — every check is a blocking MPS
    sync, and a non-finite loss is detected at that same sync boundary.
    """
    host_finite_checks = not device_gated
    return host_finite_checks, host_finite_checks and device_type == "mps"


class LearnerBaseMixin:
    """Shared AMP / gradient-sync / obs-normalizer / compile plumbing.

    Attribute contract provided by the consuming learner:
    """

    device: Any
    use_amp: bool
    _amp_dtype: torch.dtype
    scaler: Any | None
    obs_normalizer: EmpiricalNormalization | nn.Identity
    _gradient_sync: Callable[[Iterable[torch.Tensor]], None] | None
    _host_finite_checks: bool
    _metrics_finite_checks: bool

    def _finite_check_ok(self, loss: torch.Tensor, read_metrics: bool) -> bool:
        """Host-side finite guard for one loss.

        On MPS the check is deferred to the metrics-reading update of each
        iteration (``_metrics_finite_checks``); every other device keeps the
        original per-update check semantics.
        """
        if not self._host_finite_checks:
            return True
        if self._metrics_finite_checks and not read_metrics:
            return True
        return bool(torch.isfinite(loss))

    @staticmethod
    def _resolve_amp_dtype(amp_dtype: str, device_type: str) -> torch.dtype:
        normalized = amp_dtype.lower()
        if normalized == "auto":
            return torch.bfloat16
        if normalized == "fp16":
            return torch.float16
        if normalized == "bf16":
            return torch.bfloat16
        raise ValueError("amp_dtype must be one of: auto, fp16, bf16")

    @staticmethod
    def _should_use_grad_scaler(
        use_amp: bool,
        device_type: str,
        amp_dtype: torch.dtype,
    ) -> bool:
        """Whether the selected AMP dtype needs loss scaling on this device."""
        return bool(use_amp) and amp_dtype in resolve_amp_policy(device_type).grad_scaler_dtypes

    def _autocast(self):
        return torch.autocast(
            device_type=torch.device(self.device).type,
            dtype=self._amp_dtype,
            enabled=self.use_amp,
        )

    @torch.no_grad()
    def _update_obs_normalizer(self, obs: torch.Tensor) -> None:
        if isinstance(self.obs_normalizer, nn.Identity):
            return
        normalizer = cast(EmpiricalNormalization, self.obs_normalizer)
        normalizer.update(obs)

    def _sync_gradients(self, parameters: Iterable[torch.Tensor]) -> None:
        if self._gradient_sync is not None:
            self._gradient_sync(parameters)


class WholeCycleLearnerMixin(LearnerBaseMixin):
    """Shared state-preserving machinery for whole-cycle off-policy learners."""

    obs_dim: int
    critic_obs_dim: int
    action_dim: int
    _device_type: str
    _compile_full_update_cycle: bool
    tau: float
    _target_tau: torch.Tensor
    _optimizer_found_inf: torch.Tensor
    _pending_actor_metric_values: torch.Tensor | None
    _pending_cycle_critic_metric_values: torch.Tensor | None
    _pending_cycle_metric_values: torch.Tensor | None
    _update_cycle_graph: torch.cuda.CUDAGraph | None
    _update_cycle_graph_cache_key: tuple[object, ...] | None
    _update_cycle_static_batch: dict[str, torch.Tensor] | None
    _update_cycle_graph_metric_values: torch.Tensor | None

    @property
    def use_update_cycle(self) -> bool:
        raise NotImplementedError

    def get_state_dict(self) -> dict[str, Any]:
        raise NotImplementedError

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        del state_dict
        raise NotImplementedError

    def update_actor(
        self, batch: dict[str, torch.Tensor], *, read_metrics: bool = True
    ) -> dict[str, float]:
        del batch, read_metrics
        raise NotImplementedError

    def update_critic(
        self, batch: dict[str, torch.Tensor], *, read_metrics: bool = True
    ) -> dict[str, float]:
        del batch, read_metrics
        raise NotImplementedError

    def _update_cycle_graph_key(
        self,
        large_batch: dict[str, torch.Tensor],
        *,
        updates_per_step: int,
        policy_frequency: int,
        target_frequency: int,
        policy_before_critic: bool,
    ) -> tuple[object, ...]:
        del large_batch, updates_per_step, policy_frequency, target_frequency, policy_before_critic
        raise NotImplementedError

    def _whole_cycle_optimizers(self) -> tuple[optim.Optimizer, ...]:
        raise NotImplementedError

    def _whole_cycle_modules(self) -> tuple[nn.Module, ...]:
        raise NotImplementedError

    def _whole_cycle_schedulers(self) -> tuple[Any, ...]:
        return ()

    def _whole_cycle_finite_gate(self) -> torch.Tensor:
        raise NotImplementedError

    def _whole_cycle_target_modules(self) -> tuple[nn.Module, nn.Module]:
        raise NotImplementedError

    def _whole_cycle_warmup_hyperparameters(self) -> tuple[str, ...]:
        return ("lr",)

    def _materialize_optimizer_hyperparameters(self) -> tuple[str, ...]:
        return ("lr",)

    def _begin_update_cycle_metrics(self) -> None:
        raise NotImplementedError

    def _prepare_training_methods_for_collection(
        self, large_batch: dict[str, torch.Tensor], batch_size: int
    ) -> None:
        del large_batch, batch_size

    def _clear_warmup_pending_state(self) -> None:
        """Drop warmup-staged pending metric values after the state restore."""
        self._pending_actor_metric_values = None
        self._pending_cycle_critic_metric_values = None
        self._pending_cycle_metric_values = None

    def _synchronize_after_warmup(self) -> None:
        """Synchronize the learner device after the warmup state restore."""
        sync_device(self.device)

    def _prepare_update_cycle_warmup(self, *, updates_per_step: int, policy_frequency: int) -> None:
        for optimizer in self._whole_cycle_optimizers():
            for group in optimizer.param_groups:
                for name in self._whole_cycle_warmup_hyperparameters():
                    group[name] = 0.0

    @contextmanager
    def _whole_cycle_target_update_range(self) -> Iterator[None]:
        yield

    def _before_update_cycle_capture(
        self,
        saved_groups: Sequence[dict[str, Any]],
        *,
        updates_per_step: int,
        policy_frequency: int,
    ) -> tuple[Any, ...]:
        del saved_groups, updates_per_step, policy_frequency
        return ()

    def _finish_update_cycle_capture(self, saved_groups: Sequence[dict[str, Any]]) -> None:
        del saved_groups

    @property
    def _update_cycle_capture_error_mode(self) -> str | None:
        return None

    @property
    def _whole_cycle_capture_fallback(self) -> bool:
        return False

    def _whole_cycle_capture_failure_message(self) -> str:
        return (
            f"{type(self).__name__} whole-cycle CUDA graph capture failed; "
            "falling back to eager update-cycle execution for the rest of training."
        )

    @staticmethod
    def _snapshot_optimizer_groups(optimizers: tuple[optim.Optimizer, ...]) -> list[dict[str, Any]]:
        return [
            {name: value for name, value in group.items() if name != "params"}
            for optimizer in optimizers
            for group in optimizer.param_groups
        ]

    @staticmethod
    def _restore_optimizer_groups(
        optimizers: tuple[optim.Optimizer, ...],
        saved_groups: Sequence[dict[str, Any]],
    ) -> None:
        cursor = 0
        for optimizer in optimizers:
            for group in optimizer.param_groups:
                group.update(saved_groups[cursor])
                cursor += 1

    def prepare_for_collection(self, warmup_context: OffPolicyWarmupContext) -> None:
        """Exercise learner cold paths while preserving all training state."""
        large_batch = warmup_context.replay_batch or build_offpolicy_warmup_batch(
            warmup_context,
            obs_dim=self.obs_dim,
            critic_obs_dim=self.critic_obs_dim,
            action_dim=self.action_dim,
            device=self.device,
        )
        batch_size = max(1, int(warmup_context.batch_size))
        self._prepare_training_methods_for_collection(large_batch, batch_size)
        if self.use_update_cycle:
            self._ensure_update_cycle_graph(
                large_batch,
                updates_per_step=warmup_context.updates_per_step,
                policy_frequency=warmup_context.policy_frequency,
                target_frequency=warmup_context.target_frequency,
                policy_before_critic=warmup_context.policy_before_critic,
            )
            return

        saved_state = copy.deepcopy(self.get_state_dict())
        saved_scaler = copy.deepcopy(self.scaler.state_dict()) if self.scaler is not None else None
        saved_found_inf = self._optimizer_found_inf.detach().clone()
        saved_finite_gate = self._whole_cycle_finite_gate().detach().clone()
        rng_state = capture_rng_state(self.device)
        try:
            for update_idx in range(max(1, int(warmup_context.updates_per_step))):
                start = update_idx * batch_size
                batch = {
                    key: value[start : start + batch_size] for key, value in large_batch.items()
                }
                do_actor_update = update_idx % max(1, warmup_context.policy_frequency) == 0
                if warmup_context.policy_before_critic and do_actor_update:
                    self.update_actor(batch, read_metrics=False)
                self.update_critic(batch, read_metrics=False)
                if not warmup_context.policy_before_critic and do_actor_update:
                    self.update_actor(batch, read_metrics=False)
                if update_idx % max(1, warmup_context.target_frequency) == 0:
                    self.soft_update_target()
        finally:
            self.load_state_dict(saved_state)
            if self.scaler is not None and saved_scaler is not None:
                self.scaler.load_state_dict(saved_scaler)
            self._optimizer_found_inf.copy_(saved_found_inf)
            self._whole_cycle_finite_gate().copy_(saved_finite_gate)
            self._clear_warmup_pending_state()
            restore_rng_state(rng_state)
            for optimizer in self._whole_cycle_optimizers():
                self._zero_optimizer_gradients(optimizer)
            self._synchronize_after_warmup()

    def _materialize_capturable_optimizer_state(self) -> None:
        """Create fused optimizer state before whole-cycle graph capture."""
        optimizers = self._whole_cycle_optimizers()
        saved_groups = self._snapshot_optimizer_groups(optimizers)
        try:
            for optimizer in optimizers:
                for group in optimizer.param_groups:
                    for name in self._materialize_optimizer_hyperparameters():
                        group[name] = 0.0
                    for parameter in group["params"]:
                        parameter.grad = torch.zeros_like(parameter)
                optimizer.step()
        finally:
            self._restore_optimizer_groups(optimizers, saved_groups)
            for optimizer in optimizers:
                for group in optimizer.param_groups:
                    for parameter in group["params"]:
                        if parameter.grad is not None:
                            parameter.grad.zero_()
            for optimizer in optimizers:
                for state in optimizer.state.values():
                    for value in state.values():
                        if isinstance(value, torch.Tensor):
                            value.zero_()

    def _zero_optimizer_gradients(self, optimizer: optim.Optimizer) -> None:
        """Zero gradients, retaining stable buffers required by compiled backward."""
        if not self._compile_full_update_cycle:
            if not getattr(self, "use_compile", False):
                optimizer.zero_grad(set_to_none=True)
                return
            for group in optimizer.param_groups:
                for parameter in group["params"]:
                    if parameter.grad is None:
                        parameter.grad = torch.zeros_like(parameter)
        gradients = [
            parameter.grad
            for group in optimizer.param_groups
            for parameter in group["params"]
            if parameter.grad is not None
        ]
        if gradients:
            torch._foreach_zero_(gradients)

    def _run_update_cycle_core(
        self,
        large_batch: dict[str, torch.Tensor],
        *,
        updates_per_step: int,
        policy_frequency: int,
        target_frequency: int,
        policy_before_critic: bool,
    ) -> None:
        self._begin_update_cycle_metrics()
        batch_size = int(next(iter(large_batch.values())).shape[0]) // updates_per_step
        for update_idx in range(updates_per_step):
            start = update_idx * batch_size
            end = start + batch_size
            batch = {key: value[start:end] for key, value in large_batch.items()}
            do_actor_update = update_idx % policy_frequency == 0
            if policy_before_critic and do_actor_update:
                self.update_actor(batch, read_metrics=False)
            self.update_critic(batch, read_metrics=False)
            if not policy_before_critic and do_actor_update:
                self.update_actor(batch, read_metrics=False)
            if update_idx % target_frequency == 0:
                self.soft_update_target()

        critic_values = self._pending_cycle_critic_metric_values
        actor_values = self._pending_actor_metric_values
        if critic_values is None:
            return
        self._pending_cycle_metric_values = (
            torch.cat((critic_values, actor_values)) if actor_values is not None else critic_values
        )

    def soft_update_target(self) -> None:
        """Polyak-average the target critic, gated by critic-update finiteness."""
        with self._whole_cycle_target_update_range():
            tau: float | torch.Tensor
            if self._device_type == "cuda":
                tau = self._target_tau * self._whole_cycle_finite_gate().to(self._target_tau.dtype)
            else:
                tau = self.tau
            target, source = self._whole_cycle_target_modules()
            polyak_update_target(target, source, tau)

    def _warm_update_cycle_graph(
        self,
        large_batch: dict[str, torch.Tensor],
        *,
        updates_per_step: int,
        policy_frequency: int,
        target_frequency: int,
        policy_before_critic: bool,
    ) -> None:
        """Dry-run captured kernels without changing training state."""
        modules = self._whole_cycle_modules()
        optimizers = self._whole_cycle_optimizers()
        schedulers = self._whole_cycle_schedulers()
        saved_models = [copy.deepcopy(module.state_dict()) for module in modules]
        saved_optimizers = [copy.deepcopy(optimizer.state_dict()) for optimizer in optimizers]
        saved_schedulers = [copy.deepcopy(scheduler.state_dict()) for scheduler in schedulers]
        saved_groups = self._snapshot_optimizer_groups(optimizers)
        saved_obs = (
            copy.deepcopy(self.obs_normalizer.state_dict())
            if not isinstance(self.obs_normalizer, nn.Identity)
            else None
        )
        cpu_rng_state = torch.random.get_rng_state()
        cuda_rng_state = torch.cuda.get_rng_state(self.device)
        stream = torch.cuda.Stream(device=self.device)
        self._prepare_update_cycle_warmup(
            updates_per_step=updates_per_step,
            policy_frequency=policy_frequency,
        )
        try:
            self._capture_update_cycle = True
            stream.wait_stream(torch.cuda.current_stream(self.device))
            with torch.cuda.stream(stream):
                self._run_update_cycle_core(
                    large_batch,
                    updates_per_step=updates_per_step,
                    policy_frequency=policy_frequency,
                    target_frequency=target_frequency,
                    policy_before_critic=policy_before_critic,
                )
            torch.cuda.current_stream(self.device).wait_stream(stream)
            sync_device(self.device)
        finally:
            self._capture_update_cycle = False
            self._restore_optimizer_groups(optimizers, saved_groups)
            for module, state_dict in zip(modules, saved_models, strict=True):
                module.load_state_dict(state_dict)
            for optimizer, state_dict in zip(optimizers, saved_optimizers, strict=True):
                optimizer.load_state_dict(state_dict)
            for scheduler, state_dict in zip(schedulers, saved_schedulers, strict=True):
                scheduler.load_state_dict(state_dict)
            if saved_obs is not None:
                self.obs_normalizer.load_state_dict(saved_obs)
            torch.random.set_rng_state(cpu_rng_state)
            torch.cuda.set_rng_state(cuda_rng_state, self.device)
            for optimizer in optimizers:
                self._zero_optimizer_gradients(optimizer)
            sync_device(self.device)

    def _ensure_update_cycle_graph(
        self,
        large_batch: dict[str, torch.Tensor],
        *,
        updates_per_step: int,
        policy_frequency: int,
        target_frequency: int,
        policy_before_critic: bool,
    ) -> None:
        key = self._update_cycle_graph_key(
            large_batch,
            updates_per_step=updates_per_step,
            policy_frequency=policy_frequency,
            target_frequency=target_frequency,
            policy_before_critic=policy_before_critic,
        )
        if self._update_cycle_graph_cache_key == key and self._update_cycle_graph is not None:
            assert self._update_cycle_static_batch is not None
            for input_key, tensor in self._update_cycle_static_batch.items():
                tensor.copy_(large_batch[input_key])
            return

        self._update_cycle_graph_cache_key = key
        self._update_cycle_graph = None
        self._update_cycle_static_batch = {
            key: value.detach().clone() for key, value in large_batch.items()
        }
        static_batch = self._update_cycle_static_batch
        optimizers = self._whole_cycle_optimizers()
        saved_groups = self._snapshot_optimizer_groups(optimizers)
        capture_state = self._before_update_cycle_capture(
            saved_groups,
            updates_per_step=updates_per_step,
            policy_frequency=policy_frequency,
        )
        graph: torch.cuda.CUDAGraph | None = None
        try:
            self._warm_update_cycle_graph(
                static_batch,
                updates_per_step=updates_per_step,
                policy_frequency=policy_frequency,
                target_frequency=target_frequency,
                policy_before_critic=policy_before_critic,
            )
            graph = torch.cuda.CUDAGraph()
            self._prepare_update_cycle_capture(*capture_state)
            self._capture_update_cycle = True
            error_mode = self._update_cycle_capture_error_mode
            if error_mode is None:
                graph_context = torch.cuda.graph(graph)
            else:
                graph_context = torch.cuda.graph(graph, capture_error_mode=error_mode)
            with torch.cuda.device(self.device), graph_context:
                self._run_update_cycle_core(
                    static_batch,
                    updates_per_step=updates_per_step,
                    policy_frequency=policy_frequency,
                    target_frequency=target_frequency,
                    policy_before_critic=policy_before_critic,
                )
        except torch.AcceleratorError:
            if not self._whole_cycle_capture_fallback:
                raise
            self._update_cycle_capture_failed = True
            self._update_cycle_graph = None
            warnings.warn(self._whole_cycle_capture_failure_message(), stacklevel=2)
            return
        finally:
            self._capture_update_cycle = False
            self._finish_update_cycle_capture(saved_groups)
        assert graph is not None
        self._update_cycle_graph = graph
        self._update_cycle_graph_metric_values = self._pending_cycle_metric_values

    def _prepare_update_cycle_capture(self, *capture_state: Any) -> None:
        del capture_state
