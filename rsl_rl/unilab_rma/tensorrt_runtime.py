"""Optional C++ inference bridge; no TensorRT Python package is required.

The worker persists for an evaluation run. Physics, observation/history assembly,
episode resets and optional rendering continue to belong to the UniLab caller.
"""

from __future__ import annotations

import os
import select
import struct
import subprocess
import time
from pathlib import Path

import numpy as np
import torch


class TensorRTWorker:
    def __init__(self, worker: Path, engine: Path, cfg, *, timeout_s: float = 30.0):
        if os.name != "posix":
            raise RuntimeError("Run the TensorRT simulation bridge inside Linux/WSL")
        if (cfg.obs_dim, cfg.history_length, cfg.action_dim) != (49, 50, 12):
            raise ValueError("C++ worker requires observation=49, history=50, action=12")
        if not np.isfinite(timeout_s) or timeout_s <= 0:
            raise ValueError("Worker timeout must be finite and positive")
        worker, engine = Path(worker).resolve(), Path(engine).resolve()
        if not worker.is_file():
            raise FileNotFoundError(f"Build with WITH_TENSORRT=ON first: {worker}")
        if not engine.is_file():
            raise FileNotFoundError(f"TensorRT engine not found: {engine}")
        self.timeout_s = timeout_s
        self._process = subprocess.Popen(
            [str(worker), "--engine", str(engine)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=None,
            bufsize=0,
        )
        self._closed = False
        try:
            os.set_blocking(self._process.stdin.fileno(), False)
            os.set_blocking(self._process.stdout.fileno(), False)
            magic, obs, history, action = struct.unpack("<4sIII", self._read_exact(16))
            if (magic, obs, history, action) != (b"RMA1", 49, 50, 12):
                raise RuntimeError("TensorRT worker protocol/dimension mismatch")
        except BaseException:
            self.close()
            raise

    def _read_exact(self, size: int) -> bytes:
        fd = self._process.stdout.fileno()
        deadline = time.monotonic() + self.timeout_s
        result = bytearray()
        while len(result) < size:
            remaining = deadline - time.monotonic()
            if remaining <= 0 or not select.select([fd], [], [], remaining)[0]:
                raise TimeoutError("TensorRT worker timed out while reading output")
            try:
                part = os.read(fd, size - len(result))
            except BlockingIOError:
                continue
            if not part:
                raise RuntimeError("TensorRT worker closed output; check its stderr message")
            result.extend(part)
        return bytes(result)

    def _write_all(self, payload: bytes) -> None:
        fd = self._process.stdin.fileno()
        pending = memoryview(payload)
        deadline = time.monotonic() + self.timeout_s
        while pending:
            remaining = deadline - time.monotonic()
            if remaining <= 0 or not select.select([], [fd], [], remaining)[1]:
                raise TimeoutError("TensorRT worker timed out while writing input")
            try:
                written = os.write(fd, pending)
            except BlockingIOError:
                continue
            except BrokenPipeError as error:
                raise RuntimeError(
                    "TensorRT worker closed input; check its stderr message"
                ) from error
            pending = pending[written:]

    def infer(self, observation: torch.Tensor, history: torch.Tensor) -> torch.Tensor:
        if self._closed:
            raise RuntimeError("TensorRT worker is closed")
        if observation.ndim != 2 or observation.shape[1] != 49:
            raise ValueError("Expected observation shape [num_envs, 49]")
        count = observation.shape[0]
        if count < 1 or history.shape != (count, 50, 49):
            raise ValueError("Expected history shape [num_envs, 50, 49]")
        obs = observation.detach().to(device="cpu", dtype=torch.float32).contiguous().numpy()
        hist = history.detach().to(device="cpu", dtype=torch.float32).contiguous().numpy()
        if not np.isfinite(obs).all() or not np.isfinite(hist).all():
            raise ValueError("TensorRT inputs must be finite")
        actions = np.empty((count, 12), dtype=np.float32)
        # Fixed-batch-one engine: process each environment independently. Read
        # each reply before the next request so pipe capacity cannot deadlock.
        for index in range(count):
            payload = np.concatenate((obs[index], hist[index].reshape(-1))).astype(
                "<f4", copy=False
            )
            self._write_all(payload.tobytes())
            actions[index] = np.frombuffer(self._read_exact(12 * 4), dtype="<f4")
        if not np.isfinite(actions).all():
            raise RuntimeError("TensorRT worker returned non-finite actions")
        return torch.from_numpy(actions).to(observation.device)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        process = self._process
        try:
            process.stdin.close()
            try:
                process.wait(timeout=1.0)
            except subprocess.TimeoutExpired:
                process.terminate()
                try:
                    process.wait(timeout=1.0)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=1.0)
        finally:
            process.stdout.close()
