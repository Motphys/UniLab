"""Base class for device-adaptive shared memory buffers."""

import torch


class SharedBufferBase:
    """Device-adaptive shared memory buffer base class."""

    def __init__(self, capacity: int, device: str):
        self.capacity = capacity
        self.device = device
        self.ptr = torch.zeros(1, dtype=torch.int64).share_memory_()

        if device == "cuda":
            self._gpu_synced_ptr = 0
            self._cuda_stream = torch.cuda.Stream()
        else:
            self._gpu_synced_ptr = None  # type: ignore[assignment]
            self._cuda_stream = None  # type: ignore[assignment]
