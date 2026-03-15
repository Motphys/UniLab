"""Hardware monitoring utilities for performance profiling."""

from typing import Dict

import torch

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False


class HardwareMonitor:
    """Monitor CPU, GPU, memory usage."""

    def __init__(self):
        self.has_psutil = HAS_PSUTIL
        if self.has_psutil:
            self.process = psutil.Process()

        self.has_cuda = torch.cuda.is_available()
        if self.has_cuda:
            try:
                import pynvml
                pynvml.nvmlInit()
                self.nvml_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                self.has_nvml = True
            except Exception:
                self.has_nvml = False
        else:
            self.has_nvml = False

    def get_metrics(self) -> Dict[str, float]:
        """Get current hardware metrics."""
        metrics = {}

        # CPU & Memory (requires psutil)
        if self.has_psutil:
            metrics["cpu_percent"] = self.process.cpu_percent()
            metrics["cpu_count"] = psutil.cpu_count()
            mem = self.process.memory_info()
            metrics["memory_rss_mb"] = mem.rss / 1024 / 1024
            metrics["memory_percent"] = self.process.memory_percent()

        # GPU
        if self.has_cuda:
            metrics["gpu_memory_allocated_mb"] = torch.cuda.memory_allocated() / 1024 / 1024
            metrics["gpu_memory_reserved_mb"] = torch.cuda.memory_reserved() / 1024 / 1024

            if self.has_nvml:
                import pynvml
                util = pynvml.nvmlDeviceGetUtilizationRates(self.nvml_handle)
                metrics["gpu_utilization"] = util.gpu
                metrics["gpu_memory_utilization"] = util.memory

        return metrics
