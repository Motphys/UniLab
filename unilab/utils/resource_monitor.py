"""Mac resource monitor for CPU and GPU utilization.

Uses psutil for CPU/memory and subprocess calls for Apple GPU metrics.
Runs a background thread that samples at configurable intervals.
"""

import threading
import time

try:
    import psutil
except ImportError:
    psutil = None


class ResourceMonitor:
    """Background monitor that periodically samples CPU, memory, and GPU stats."""

    def __init__(self, interval: float = 2.0):
        self.interval = interval
        self._running = False
        self._thread = None
        self._stats = {
            "cpu_percent": 0.0,
            "mem_used_gb": 0.0,
            "mem_total_gb": 0.0,
            "mps_mem_gb": 0.0,
            "gpu_power": "N/A",
        }

    def start(self):
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=5.0)

    def get_stats(self) -> dict:
        return self._stats.copy()

    def _loop(self):
        while self._running:
            try:
                self._sample()
            except Exception:
                pass
            time.sleep(self.interval)

    def _sample(self):
        # CPU
        if psutil:
            self._stats["cpu_percent"] = psutil.cpu_percent(interval=0.5)
            mem = psutil.virtual_memory()
            self._stats["mem_used_gb"] = mem.used / (1024 ** 3)
            self._stats["mem_total_gb"] = mem.total / (1024 ** 3)

        # MPS Memory
        try:
            import torch
            if torch.backends.mps.is_available():
                self._stats["mps_mem_gb"] = torch.mps.current_allocated_memory() / (1024 ** 3)
        except Exception:
            pass

        # Apple GPU — try to get power/frequency info via ioreg
        self._stats["gpu_power"] = self._get_apple_gpu_info()

    @staticmethod
    def _get_apple_gpu_info() -> str:
        """Attempt to read Apple GPU utilization via sysctl or powermetrics.

        powermetrics requires root, so we fall back to a simpler approach.
        """
        import subprocess

        # Try sysctl for basic GPU info (doesn't require root)
        try:
            result = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True, text=True, timeout=2,
            )
            chip = result.stdout.strip() if result.returncode == 0 else ""
        except Exception:
            chip = ""

        # Try ioreg for GPU utilization data
        try:
            result = subprocess.run(
                ["ioreg", "-r", "-c", "AGXAccelerator", "-d", "1"],
                capture_output=True, text=True, timeout=2,
            )
            if result.returncode == 0 and "gpu-core-count" in result.stdout.lower():
                # Parse basic info — just indicate GPU is being used
                for line in result.stdout.split("\n"):
                    if "PerformanceStatistics" in line or "gpu-core-count" in line:
                        return f"Active ({chip})" if chip else "Active"
        except Exception:
            pass

        if chip:
            return f"Available ({chip})"
        return "N/A"


def print_resource_line():
    """One-shot: print a single resource status line to stdout."""
    if psutil is None:
        print("[Resource] psutil not installed, cannot monitor")
        return

    cpu = psutil.cpu_percent(interval=1.0)
    mem = psutil.virtual_memory()
    gpu_info = ResourceMonitor._get_apple_gpu_info()
    print(
        f"[Resource] CPU: {cpu:.1f}% | "
        f"Mem: {mem.used / (1024**3):.1f}/{mem.total / (1024**3):.1f} GB | "
        f"GPU: {gpu_info}"
    )
