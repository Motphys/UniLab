"""Base async runner for multi-process RL training."""

from __future__ import annotations

import multiprocessing as mp
from abc import ABC, abstractmethod
from typing import Any, Callable

_SPAWN_CTX = mp.get_context("spawn")


class AsyncRunner(ABC):
    """Base class for async RL algorithms.

    Manages:
    - Shared memory allocation/cleanup
    - Collector process lifecycle
    - Training loop skeleton
    """

    def __init__(
        self,
        env_name: str,
        env_cfg_overrides: dict,
        rl_cfg: dict,
        *,
        device: str | None = None,
        collector_device: str | None = None,
        num_envs: int = 4096,
        **kwargs,
    ):
        self.env_name = env_name
        self.env_cfg_overrides = env_cfg_overrides
        self.rl_cfg = rl_cfg
        self.device = device or self._get_default_device()
        self.collector_device = collector_device or self.device
        self.num_envs = num_envs
        self.extra_kwargs = kwargs

        self._collector_process: Any = None
        self._stop_event = _SPAWN_CTX.Event()
        self._shared_resources: list = []

    @abstractmethod
    def _get_default_device(self) -> str:
        """Get default device (backend-specific)."""
        ...

    @abstractmethod
    def _build_learner(self) -> Any: ...

    @abstractmethod
    def _collector_fn(self, stop_event: Any, **kwargs) -> None: ...

    @abstractmethod
    def learn(
        self, max_iterations: int, save_interval: int = 50, log_dir: str = "logs"
    ) -> None: ...

    def _start_collector(self, target_fn: Callable, kwargs: dict) -> None:
        collector_kwargs = {**self.extra_kwargs, **kwargs}
        self._collector_process = _SPAWN_CTX.Process(
            target=target_fn, kwargs=collector_kwargs, daemon=True
        )
        self._collector_process.start()

    def close(self) -> None:
        self._stop_event.set()
        if self._collector_process is not None and self._collector_process.is_alive():
            self._collector_process.join(timeout=10)
            if self._collector_process.is_alive():
                self._collector_process.terminate()
                self._collector_process.join(timeout=5)
        for resource in self._shared_resources:
            if hasattr(resource, "cleanup"):
                resource.cleanup()
            elif hasattr(resource, "close"):
                resource.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
