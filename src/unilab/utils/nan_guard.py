"""NaN/Inf guard for env-layer numerical anomaly detection and state dumping."""

from __future__ import annotations

import logging
import shutil
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class NanGuardCfg:
    enabled: bool = False
    buffer_size: int = 100
    max_envs_to_dump: int = 5
    output_dir: str | None = None


class NanGuard:
    def __init__(
        self,
        cfg: NanGuardCfg,
        num_envs: int,
        supports_state_playback: bool,
    ) -> None:
        self._cfg = cfg
        self._num_envs = num_envs
        self._supports_state_playback = supports_state_playback
        self._buffer: list[np.ndarray] = []
        self._buffer_idx: int = 0
        self._buffer_full: bool = False
        self._dumped: bool = False

    def capture(self, physics_state: np.ndarray | None) -> None:
        if physics_state is None:
            return
        if not self._buffer_full and len(self._buffer) < self._cfg.buffer_size:
            self._buffer.append(physics_state)
        else:
            self._buffer_full = True
            self._buffer[self._buffer_idx] = physics_state
        self._buffer_idx = (self._buffer_idx + 1) % self._cfg.buffer_size

    def check(self, obs: dict[str, np.ndarray], reward: np.ndarray) -> np.ndarray | None:
        if not self._cfg.enabled or self._dumped:
            return None
        bad_mask = np.zeros(self._num_envs, dtype=bool)
        for v in obs.values():
            bad_mask |= ~np.all(np.isfinite(v), axis=tuple(range(1, v.ndim)))
        bad_mask |= ~np.isfinite(reward)
        if not np.any(bad_mask):
            return None
        return np.flatnonzero(bad_mask).astype(np.int32)

    def dump(
        self,
        nan_env_ids: np.ndarray,
        model_file: str,
        step: int,
    ) -> str | None:
        if self._dumped:
            return None
        self._dumped = True

        output_dir = Path(self._cfg.output_dir or "/tmp/unilab/nan_dumps")
        output_dir.mkdir(parents=True, exist_ok=True)

        dump_env_ids = nan_env_ids[: self._cfg.max_envs_to_dump]

        if self._buffer_full:
            ordered = self._buffer[self._buffer_idx :] + self._buffer[: self._buffer_idx]
        else:
            ordered = list(self._buffer)

        states = np.stack(ordered, axis=0) if ordered else np.array([])
        if states.ndim >= 3 and dump_env_ids.size > 0:
            states = states[:, dump_env_ids]

        metadata = {
            "num_envs_total": self._num_envs,
            "nan_env_ids": nan_env_ids,
            "dumped_env_ids": dump_env_ids,
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
            states=states,
            **{f"meta_{k}": v for k, v in metadata.items()},
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

        logger.warning(
            "NaN guard triggered at step %d for %d envs. Dump: %s",
            step,
            len(nan_env_ids),
            npz_path,
        )
        return str(npz_path)
