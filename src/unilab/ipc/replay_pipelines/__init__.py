"""Replay pipeline abstraction."""

from unilab.ipc.replay_pipelines.base import ReplayPipeline, ReplayTickMetadata
from unilab.ipc.replay_pipelines.cpu_pinned_double_buffer import (
    CPUPinnedDoubleBufferReplayPipeline,
)
from unilab.ipc.replay_pipelines.gpu_resident import GPUResidentReplayPipeline
from unilab.ipc.replay_pipelines.multi_gpu_gpu_resident import (
    MultiGPUGPUResidentReplayPipeline,
)

__all__ = [
    "ReplayPipeline",
    "ReplayTickMetadata",
    "CPUPinnedDoubleBufferReplayPipeline",
    "GPUResidentReplayPipeline",
    "MultiGPUGPUResidentReplayPipeline",
]
