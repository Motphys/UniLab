"""Finite-state diagnostics for the tensor-native environment runtime.

The guard's detection path stays in Torch: it never converts observations,
rewards, controls, or physics snapshots to host arrays. Deferred mode copies
only a scalar verdict at :meth:`NanGuard.poll`; row/state host conversion and
artifact writing happen in :meth:`NanGuard.dump`, an explicit abnormal boundary.
"""

from __future__ import annotations

import logging
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np
import torch

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class NanGuardCfg:
    """Pickleable configuration for the tensor finite-state guard."""

    enabled: bool = False
    buffer_size: int = 100
    max_envs_to_dump: int = 5
    output_dir: str | None = None
    # Zero disables snapshots; one retains the debug behavior. Production
    # profiles use 32 to avoid cloning the full physics state every step.
    capture_interval: int = 1
    # Zero means the owner must invoke poll explicitly at its cycle boundary.
    check_interval: int = 1
    defer_validation: bool = False


def _bad_row_mask(value: torch.Tensor, num_envs: int) -> torch.Tensor:
    """Return a one-D boolean mask without introducing a host synchronization."""
    if value.ndim == 0 or value.shape[0] != num_envs:
        raise ValueError(
            f"tensor diagnostics leading dimension must be {num_envs}, got {tuple(value.shape)}"
        )
    if not value.is_floating_point():
        raise TypeError("tensor diagnostics finite checks require floating-point tensors")
    return ~torch.isfinite(value).reshape(num_envs, -1).all(dim=1)


def _snapshot(value: torch.Tensor) -> torch.Tensor:
    return value.detach().to(device="cpu", copy=True)


class NanGuard:
    """Detect bad tensor rows on-device and export dumps at an explicit host boundary."""

    def __init__(
        self,
        cfg: NanGuardCfg,
        num_envs: int,
        supports_state_playback: bool,
    ) -> None:
        if num_envs <= 0:
            raise ValueError("tensor diagnostics require a positive environment count")
        if cfg.buffer_size <= 0:
            raise ValueError("tensor diagnostics buffer_size must be positive")
        if cfg.max_envs_to_dump < 0:
            raise ValueError("tensor diagnostics max_envs_to_dump must be non-negative")
        for label, interval in (
            ("capture_interval", cfg.capture_interval),
            ("check_interval", cfg.check_interval),
        ):
            if isinstance(interval, bool) or not isinstance(interval, int) or interval < 0:
                raise ValueError(f"tensor diagnostics {label} must be non-negative (integer)")
        self._cfg = cfg
        self._num_envs = num_envs
        self._supports_state_playback = supports_state_playback
        self._buffer: list[torch.Tensor] = []
        self._buffer_idx = 0
        self._buffer_full = False
        self._dumped = False
        # Both latches live on the first input's device. They retain all bad
        # rows until a single explicit host poll closes the validation interval.
        self._device: torch.device | None = None
        self._device_bad_rows: torch.Tensor | None = None
        self._device_verdict: torch.Tensor | None = None
        self._capture_counter = 0

    def _ensure_device_state(self, device: torch.device) -> None:
        device = torch.device(device)
        if self._device_bad_rows is None or self._device_verdict is None:
            self._device = device
            self._device_bad_rows = torch.zeros((self._num_envs,), dtype=torch.bool, device=device)
            self._device_verdict = torch.zeros((), dtype=torch.bool, device=device)
        elif self._device != device:
            raise ValueError(
                "tensor diagnostics inputs must share one device: "
                f"expected {self._device}, got {device}"
            )

    def _record_bad_rows(self, bad_rows: torch.Tensor) -> None:
        if bad_rows.dtype != torch.bool or bad_rows.ndim != 1:
            raise ValueError("tensor diagnostics bad-row mask must be one-dimensional bool")
        if bad_rows.shape[0] != self._num_envs:
            raise ValueError(
                f"tensor diagnostics bad-row mask must have {self._num_envs} rows, "
                f"got {bad_rows.shape[0]}"
            )
        self._ensure_device_state(bad_rows.device)
        assert self._device_bad_rows is not None
        assert self._device_verdict is not None
        self._device_bad_rows.logical_or_(bad_rows)
        self._device_verdict.logical_or_(bad_rows.any())

    @property
    def device_verdict(self) -> torch.Tensor | None:
        """Persistent device-side bad flag, without a host transfer."""
        return self._device_verdict

    @property
    def defer_validation(self) -> bool:
        return self._cfg.defer_validation

    def capture_due(self, step: int) -> bool:
        """Allow the owner to skip even the backend state read on ordinary steps."""
        interval = self._cfg.capture_interval
        return self._cfg.enabled and interval > 0 and step % interval == 0

    def reset_verdict(self) -> None:
        if self._device_bad_rows is not None:
            self._device_bad_rows.zero_()
        if self._device_verdict is not None:
            self._device_verdict.zero_()

    def _validate_finite_float(self, value: torch.Tensor, label: str) -> None:
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"{label} must be a torch.Tensor, got {type(value).__name__}")
        if not value.is_floating_point():
            raise TypeError(f"{label} must be a floating-point tensor, got {value.dtype}")

    def capture(self, physics_state: torch.Tensor | None, step: int | None = None) -> None:
        """Retain a detached device tensor snapshot for a later abnormal dump."""
        if physics_state is None:
            return
        self._validate_finite_float(physics_state, "physics_state")
        if physics_state.ndim < 2 or physics_state.shape[0] != self._num_envs:
            raise ValueError(
                "physics_state must have shape "
                f"({self._num_envs}, ...), got {tuple(physics_state.shape)}"
            )
        if self._cfg.capture_interval == 0:
            return
        if step is None:
            step = self._capture_counter
            self._capture_counter += 1
        else:
            self._capture_counter = max(self._capture_counter, step + 1)
        if step % self._cfg.capture_interval != 0:
            return
        self._ensure_device_state(physics_state.device)
        snapshot = physics_state.detach().clone()
        if not self._buffer_full and len(self._buffer) < self._cfg.buffer_size:
            self._buffer.append(snapshot)
        else:
            self._buffer_full = True
            self._buffer[self._buffer_idx] = snapshot
        self._buffer_idx = (self._buffer_idx + 1) % self._cfg.buffer_size

    def check(
        self,
        obs: Mapping[str, torch.Tensor],
        reward: torch.Tensor,
        step: int = 0,
        *,
        defer: bool | None = None,
    ) -> torch.Tensor | None:
        """Latch bad rows on-device; publish them only at an explicit poll."""
        if not self._cfg.enabled:
            return None
        if not isinstance(obs, Mapping):
            raise TypeError(f"observations must be a mapping, got {type(obs).__name__}")
        masks: list[torch.Tensor] = []
        for name, value in obs.items():
            self._validate_finite_float(value, f"obs[{name!r}]")
            masks.append(_bad_row_mask(value, self._num_envs))
        self._validate_finite_float(reward, "reward")
        masks.append(_bad_row_mask(reward, self._num_envs))
        if any(mask.device != reward.device for mask in masks):
            raise ValueError("tensor diagnostics inputs must share one device")
        bad_rows = masks[0]
        for mask in masks[1:]:
            bad_rows.logical_or_(mask)
        self._record_bad_rows(bad_rows)
        should_defer = self._cfg.defer_validation if defer is None else bool(defer)
        return None if should_defer else self.poll(step=step, force=defer is False)

    def check_ctrl(
        self, ctrl: torch.Tensor, step: int = 0, *, defer: bool | None = None
    ) -> torch.Tensor | None:
        """Latch bad control rows on-device; publish at a poll boundary."""
        if not self._cfg.enabled:
            return None
        self._validate_finite_float(ctrl, "ctrl")
        self._record_bad_rows(_bad_row_mask(ctrl, self._num_envs))
        should_defer = self._cfg.defer_validation if defer is None else bool(defer)
        return None if should_defer else self.poll(step=step, force=defer is False)

    def poll(self, step: int = 0, *, force: bool = False) -> torch.Tensor | None:
        """Read one scalar verdict at a cycle boundary; materialize rows on error."""
        if not self._cfg.enabled:
            return None
        if (
            not force
            and self._cfg.defer_validation
            and (self._cfg.check_interval == 0 or step % self._cfg.check_interval != 0)
        ):
            return None
        if self._device_verdict is None or self._device_bad_rows is None:
            return None
        if not bool(self._device_verdict.detach().to(device="cpu").item()):
            return None
        nan_ids = torch.nonzero(self._device_bad_rows, as_tuple=False).flatten().to(torch.int64)
        logger.warning(
            "NanGuard: NaN/Inf detected at step %d (envs=%d, sample_ids=%s)",
            step,
            nan_ids.numel(),
            nan_ids[: min(5, nan_ids.numel())].detach().cpu().tolist(),
        )
        self.reset_verdict()
        return nan_ids

    def dump(
        self,
        nan_env_ids: torch.Tensor,
        model_file: str,
        step: int,
    ) -> str | None:
        """Export one abnormal-state artifact at an explicit host boundary."""
        if self._dumped:
            return None
        self._dumped = True
        if not isinstance(nan_env_ids, torch.Tensor):
            raise TypeError(f"nan_env_ids must be a torch.Tensor, got {type(nan_env_ids).__name__}")
        if nan_env_ids.ndim != 1:
            raise ValueError("nan_env_ids must be one-dimensional")
        host_nan_ids = _snapshot(nan_env_ids).to(dtype=torch.int64)

        output_dir = Path(self._cfg.output_dir or "/tmp/unilab/nan_dumps")
        output_dir.mkdir(parents=True, exist_ok=True)
        dump_env_ids = host_nan_ids[: self._cfg.max_envs_to_dump]

        if self._buffer_full:
            ordered = self._buffer[self._buffer_idx :] + self._buffer[: self._buffer_idx]
        else:
            ordered = list(self._buffer)
        states = (
            torch.stack(ordered, dim=0)
            if ordered
            else torch.empty((0, self._num_envs, 0), dtype=torch.float32)
        )
        host_states = _snapshot(states).numpy()
        if host_states.ndim >= 3 and dump_env_ids.numel() > 0:
            host_states = host_states[:, dump_env_ids.numpy()]

        metadata = {
            "num_envs_total": self._num_envs,
            "nan_env_ids": host_nan_ids.numpy(),
            "dumped_env_ids": dump_env_ids.numpy(),
            "buffer_size": self._cfg.buffer_size,
            "buffer_len": len(ordered),
            "detection_step": step,
            "timestamp": time.time(),
            "model_file": model_file,
            "supports_state_playback": self._supports_state_playback,
        }

        ts = time.strftime("%Y%m%d_%H%M%S")
        dump_name = f"nan_dump_{ts}_step{step}"
        npz_path = output_dir / f"{dump_name}.npz"
        np.savez(
            str(npz_path),
            states=host_states,
            **{f"meta_{key}": value for key, value in metadata.items()},
        )

        if model_file and Path(model_file).is_file():
            model_dst = output_dir / f"{dump_name}_model{Path(model_file).suffix}"
            shutil.copy2(model_file, model_dst)

        latest_link = output_dir / "nan_dump_latest.npz"
        latest_link.unlink(missing_ok=True)
        try:
            latest_link.symlink_to(npz_path.name)
        except OSError:
            pass

        logger.info(
            "NanGuard: dump written to %s (step=%d, envs=%d)",
            npz_path,
            step,
            host_nan_ids.numel(),
        )
        return str(npz_path)
