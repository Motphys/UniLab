"""Base class for device-adaptive shared memory buffers."""

import torch


class SharedBufferBase:
    """Device-adaptive shared memory buffer base class."""

    def __init__(self, capacity: int, device: str, defer_gpu: bool = False):
        self.capacity = capacity
        self.device = device
        self.ptr = torch.zeros(1, dtype=torch.int64).share_memory_()

        if device == "cuda":
            self._gpu_synced_ptr = 0
            # defer_gpu=True: skip CUDA stream creation so the buffer is picklable
            # (used in multi-GPU mode where workers call init_local_gpu_cache later)
            self._cuda_stream = None if defer_gpu else torch.cuda.Stream()  # type: ignore[assignment]
        else:
            self._gpu_synced_ptr = None  # type: ignore[assignment]
            self._cuda_stream = None  # type: ignore[assignment]
