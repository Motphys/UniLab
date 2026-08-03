"""CUDA stream and event ownership for the device managed runtime."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

import torch


@dataclass(frozen=True)
class DeviceRuntimeSynchronization:
    """Cold-created synchronization primitives reused for the runtime lifetime."""

    device: torch.device
    task_stream: torch.cuda.Stream
    control_event: torch.cuda.Event
    reset_event: torch.cuda.Event
    output_event: torch.cuda.Event
    episode_length_input_event: torch.cuda.Event
    cold_init_event: torch.cuda.Event

    @classmethod
    def create(cls, device: torch.device) -> DeviceRuntimeSynchronization:
        return cls(
            device=device,
            task_stream=cast(torch.cuda.Stream, torch.cuda.Stream(device=device)),
            control_event=cast(torch.cuda.Event, torch.cuda.Event(enable_timing=False)),
            reset_event=cast(torch.cuda.Event, torch.cuda.Event(enable_timing=False)),
            output_event=cast(torch.cuda.Event, torch.cuda.Event(enable_timing=False)),
            episode_length_input_event=cast(
                torch.cuda.Event, torch.cuda.Event(enable_timing=False)
            ),
            cold_init_event=cast(torch.cuda.Event, torch.cuda.Event(enable_timing=False)),
        )

    def publish_cold_initialization(self) -> None:
        """Order constructor-stream tensor writes before the first task operation."""

        cold_init_stream = torch.cuda.current_stream(self.device)
        if cold_init_stream != self.task_stream:
            self.cold_init_event.record(cold_init_stream)
            self.task_stream.wait_event(cast(Any, self.cold_init_event))

    def copy_episode_lengths(self, *, values: torch.Tensor, target: torch.Tensor) -> None:
        """Copy a temporary schedule without releasing its storage too early."""

        producer_stream = torch.cuda.current_stream(self.device)
        with torch.cuda.stream(self.task_stream):
            if producer_stream != self.task_stream:
                self.episode_length_input_event.record(producer_stream)
                self.task_stream.wait_event(cast(Any, self.episode_length_input_event))
            target.copy_(values, non_blocking=True)
            values.record_stream(self.task_stream)


__all__ = ["DeviceRuntimeSynchronization"]
