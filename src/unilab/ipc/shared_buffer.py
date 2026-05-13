"""Base class for device-adaptive shared memory buffers."""

import torch


class SharedBufferBase:
    """Device-adaptive shared memory buffer base class."""

    def __init__(self, capacity: int, device: str, defer_gpu: bool = False):
        del defer_gpu
        self.capacity = capacity
        self.device = device
        self.ptr = torch.zeros(1, dtype=torch.int64).share_memory_()
