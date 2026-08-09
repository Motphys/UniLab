"""Replay pipeline abstraction."""

from unilab.ipc.replay_pipelines.base import ReplayPipeline, ReplayTickMetadata
from unilab.ipc.replay_pipelines.cpu_pinned_double_buffer import (
    CPUPinnedDoubleBufferReplayPipeline,
)
from unilab.ipc.replay_pipelines.gpu_resident import GPUResidentReplayPipeline

__all__ = [
    "ReplayPipeline",
    "ReplayTickMetadata",
    "CPUPinnedDoubleBufferReplayPipeline",
    "GPUResidentReplayPipeline",
]
