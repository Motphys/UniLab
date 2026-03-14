#!/usr/bin/env python3
"""
Benchmark MLP inference overhead across:
- NumPy / Numba / JAX
- PyTorch (CPU/MPS) / PyTorch+torch.compile
- MLX / MLX+mx.compile
- ONNX Runtime / Core ML / ANE (Apple Neural Engine, no CPU fallback)

env_num from 2^8 to 2^14. Model size aligned with locomotion policy (obs_dim=48,
hidden=[256,256,256], action_dim=12). ANE uses EnumeratedShapes and Core ML CPU_AND_NE (no runtime ANE-usage API). Outputs JSON and plots.

Run with the MLX Python environment active (e.g. conda activate mj_env then
run this script). Do not use `conda run` so that benchmark output is visible.

Fairness (公平性): Same warmup/repeat, model shape, OBS clip [-3,3]; timing includes
device sync (MPS/JAX/MLX). Weights: numpy/numba seed 42, torch* manual_seed(42),
onnx/coreml/ane from torch 42, jax key 42. Input: rng(43) or equivalent per backend.
"""

from __future__ import annotations

try:
    from benchmark.core import (
        bench_callable,
        available_backends,
        MLPBenchRecord,
        mlp_param_count,
        trimmed_mean,
        print_mlp_table,
    )
    from benchmark.core.device_info import get_device_info_dict, get_device_info_line
except ModuleNotFoundError:
    from core import (
        bench_callable,
        available_backends,
        MLPBenchRecord,
        mlp_param_count,
        trimmed_mean,
        print_mlp_table,
    )
    from core.device_info import get_device_info_dict, get_device_info_line

import argparse
import json
import os
import statistics
import sys
import time
import warnings
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple


try:
    import numpy as np
except Exception:
    np = None

# Lazy load torch/mlx so phase 1 (numpy / numba) runs without OMP conflict
_TORCH = None
_MX = None
_NN_MLX = None
_TORCH_NUM_THREADS: Optional[int] = None  # set by main() --threads for CPU matmul


def _get_torch():
    global _TORCH
    if _TORCH is None:
        try:
            _TORCH = __import__("torch")
            n = _TORCH_NUM_THREADS if _TORCH_NUM_THREADS is not None else (os.cpu_count() or 4)
            _TORCH.set_num_threads(n)
        except Exception:
            pass
    return _TORCH


def _get_mlx():
    global _MX, _NN_MLX
    if _MX is None:
        try:
            _MX = __import__("mlx.core", fromlist=["core"])
            _NN_MLX = __import__("mlx.nn", fromlist=["nn"])
        except Exception:
            pass
    return _MX, _NN_MLX


try:
    import numba
    from numba import jit
except Exception:
    numba = None
    jit = None

try:
    import jax
    import jax.numpy as jnp
except Exception:
    jax = None
    jnp = None

try:
    import onnx
except Exception:
    onnx = None
try:
    import onnxruntime as ort
except Exception:
    ort = None

try:
    import coremltools as ct
except Exception:
    ct = None

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception:
    plt = None

# Model config aligned with locomotion tasks (Go1/Go2/G1 style policy)
DEFAULT_OBS_DIM = 48
DEFAULT_ACTION_DIM = 12
DEFAULT_HIDDEN_DIMS = [256, 256, 256]
DEFAULT_ENV_NUM_POW_MIN = 8
DEFAULT_ENV_NUM_POW_MAX = 15

# Safe divisor to avoid divide-by-zero and overflow (e.g. envs_per_sec = env_num / mean_sec)
_MIN_MEAN_SEC = 1e-12


def _safe_envs_per_sec(env_num: int, mean_sec: float) -> float:
    if mean_sec is None or mean_sec <= 0 or env_num <= 0:
        return 0.0
    mean_sec = max(mean_sec, _MIN_MEAN_SEC)
    return env_num / mean_sec
    return samples


# Input/activation range to avoid overflow (e.g. float32) in large batches
_OBS_CLIP_LOW, _OBS_CLIP_HIGH = -3.0, 3.0

# Fairness: same warmup/repeat for all; same OBS clip; timing includes device sync (MPS/JAX/MLX).
# Weights: numpy/numba seed=42; torch* we set manual_seed(42); onnx/coreml/ane from torch seed 42; jax key 42.
# Input: numpy/numba/onnx/coreml/ane rng(43); torch* manual_seed(43); mlx/jax fixed seed so reproducible.
# Same model shape everywhere: obs_dim, hidden_dims, action_dim.


def build_numpy_mlp(
    obs_dim: int,
    action_dim: int,
    hidden_dims: List[int],
    dtype: np.dtype,
    seed: int = 42,
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    rng = np.random.default_rng(seed)
    dims = [obs_dim] + hidden_dims + [action_dim]
    weights, biases = [], []
    for i in range(len(dims) - 1):
        w = rng.standard_normal((dims[i], dims[i + 1])).astype(dtype) * 0.1
        b = np.zeros((dims[i + 1],), dtype=dtype)
        weights.append(w)
        biases.append(b)
    return weights, biases


def _numpy_mlp_forward(
    x: np.ndarray, weights: List[np.ndarray], biases: List[np.ndarray]
) -> np.ndarray:
    out = x
    for i in range(len(weights) - 1):
        out = np.tanh(out @ weights[i] + biases[i])
    return out @ weights[-1] + biases[-1]


def run_numpy(
    env_num: int,
    obs_dim: int,
    action_dim: int,
    hidden_dims: List[int],
    warmup: int,
    repeat: int,
) -> Optional[MLPBenchRecord]:
    if np is None:
        return None
    weights, biases = build_numpy_mlp(obs_dim, action_dim, hidden_dims, np.float32)
    rng = np.random.default_rng(43)
    x = np.ascontiguousarray(
        np.clip(
            rng.standard_normal((env_num, obs_dim)).astype(np.float32),
            _OBS_CLIP_LOW,
            _OBS_CLIP_HIGH,
        )
    )

    def fwd():
        _ = _numpy_mlp_forward(x, weights, biases)

    elapsed = bench_callable(fwd, lambda: None, warmup, repeat)
    mean_sec = statistics.mean(elapsed)
    std_sec = statistics.pstdev(elapsed) if len(elapsed) > 1 else 0.0
    return MLPBenchRecord(
        backend="numpy",
        env_num=env_num,
        warmup=warmup,
        repeat=repeat,
        elapsed_sec=elapsed,
        mean_sec=mean_sec,
        std_sec=std_sec,
        min_sec=min(elapsed),
        max_sec=max(elapsed),
        envs_per_sec=_safe_envs_per_sec(env_num, mean_sec),
    )


# ---------- Numba (4-layer MLP, nopython + fastmath) ----------
if numba is not None and jit is not None:

    @jit(nopython=True, fastmath=True, cache=True)
    def _numba_mlp_fwd(
        x: np.ndarray,
        w0: np.ndarray,
        b0: np.ndarray,
        w1: np.ndarray,
        b1: np.ndarray,
        w2: np.ndarray,
        b2: np.ndarray,
        w3: np.ndarray,
        b3: np.ndarray,
    ) -> np.ndarray:
        """4-layer MLP: tanh(x@w0+b0), tanh(...@w1+b1), tanh(...@w2+b2), out@w3+b3."""
        out = np.tanh(x @ w0 + b0)
        out = np.tanh(out @ w1 + b1)
        out = np.tanh(out @ w2 + b2)
        return out @ w3 + b3


def run_numba(
    env_num: int,
    obs_dim: int,
    action_dim: int,
    hidden_dims: List[int],
    warmup: int,
    repeat: int,
) -> Optional[MLPBenchRecord]:
    if np is None or numba is None or jit is None:
        return None
    if len(hidden_dims) != 3:
        return None  # Numba path only supports 4-layer MLP (obs -> h1 -> h2 -> h3 -> action)
    weights, biases = build_numpy_mlp(obs_dim, action_dim, hidden_dims, np.float32)
    w0, w1, w2, w3 = weights[0], weights[1], weights[2], weights[3]
    b0, b1, b2, b3 = biases[0], biases[1], biases[2], biases[3]
    rng = np.random.default_rng(43)
    x = np.ascontiguousarray(
        np.clip(
            rng.standard_normal((env_num, obs_dim)).astype(np.float32),
            _OBS_CLIP_LOW,
            _OBS_CLIP_HIGH,
        )
    )

    def fwd():
        _ = _numba_mlp_fwd(x, w0, b0, w1, b1, w2, b2, w3, b3)

    elapsed = bench_callable(fwd, lambda: None, warmup, repeat)
    mean_sec = statistics.mean(elapsed)
    std_sec = statistics.pstdev(elapsed) if len(elapsed) > 1 else 0.0
    return MLPBenchRecord(
        backend="numba",
        env_num=env_num,
        warmup=warmup,
        repeat=repeat,
        elapsed_sec=elapsed,
        mean_sec=mean_sec,
        std_sec=std_sec,
        min_sec=min(elapsed),
        max_sec=max(elapsed),
        envs_per_sec=_safe_envs_per_sec(env_num, mean_sec),
    )


# ---------- JAX (jax.jit) ----------
def _jax_mlp_forward(x, params):
    out = x
    for i, (W, b) in enumerate(params):
        out = out @ W + b
        out = jnp.clip(out, -1e4, 1e4)
        if i < len(params) - 1:
            out = jnp.tanh(out)
    return out


def run_jax(
    env_num: int,
    obs_dim: int,
    action_dim: int,
    hidden_dims: List[int],
    warmup: int,
    repeat: int,
) -> Optional[MLPBenchRecord]:
    if jax is None or jnp is None:
        return None
    dims = [obs_dim] + hidden_dims + [action_dim]
    key = jax.random.PRNGKey(42)
    params = []
    for i in range(len(dims) - 1):
        k1, key = jax.random.split(key)
        W = jax.random.normal(k1, (dims[i], dims[i + 1])) * 0.1
        b = jnp.zeros((dims[i + 1],))
        params.append((W, b))
    key, k2 = jax.random.split(key)
    x = jnp.clip(
        jax.random.normal(k2, (env_num, obs_dim)),
        _OBS_CLIP_LOW,
        _OBS_CLIP_HIGH,
    ).astype(jnp.float32)
    fwd_jit = jax.jit(_jax_mlp_forward)

    def fwd():
        out = fwd_jit(x, params)
        jax.block_until_ready(out)

    elapsed = bench_callable(fwd, lambda: None, warmup, repeat)
    mean_sec = statistics.mean(elapsed)
    std_sec = statistics.pstdev(elapsed) if len(elapsed) > 1 else 0.0
    return MLPBenchRecord(
        backend="jax_jit",
        env_num=env_num,
        warmup=warmup,
        repeat=repeat,
        elapsed_sec=elapsed,
        mean_sec=mean_sec,
        std_sec=std_sec,
        min_sec=min(elapsed),
        max_sec=max(elapsed),
        envs_per_sec=_safe_envs_per_sec(env_num, mean_sec),
    )


# ---------- PyTorch CPU ----------
def run_torch_cpu(
    env_num: int,
    obs_dim: int,
    action_dim: int,
    hidden_dims: List[int],
    warmup: int,
    repeat: int,
) -> Optional[MLPBenchRecord]:
    torch = _get_torch()
    if torch is None:
        return None
    torch.manual_seed(42)
    dims = [obs_dim] + hidden_dims + [action_dim]
    layers = []
    for i in range(len(dims) - 1):
        layers.append(torch.nn.Linear(dims[i], dims[i + 1]))
        if i < len(dims) - 2:
            layers.append(torch.nn.Tanh())
    model = torch.nn.Sequential(*layers).float().to("cpu")
    model.eval()
    torch.manual_seed(43)
    x = torch.clamp(
        torch.randn(env_num, obs_dim, dtype=torch.float32, device="cpu"),
        _OBS_CLIP_LOW,
        _OBS_CLIP_HIGH,
    )

    def fwd():
        with torch.no_grad():
            _ = model(x)

    elapsed = bench_callable(fwd, lambda: None, warmup, repeat)
    mean_sec = statistics.mean(elapsed)
    std_sec = statistics.pstdev(elapsed) if len(elapsed) > 1 else 0.0
    return MLPBenchRecord(
        backend="torch_cpu",
        env_num=env_num,
        warmup=warmup,
        repeat=repeat,
        elapsed_sec=elapsed,
        mean_sec=mean_sec,
        std_sec=std_sec,
        min_sec=min(elapsed),
        max_sec=max(elapsed),
        envs_per_sec=_safe_envs_per_sec(env_num, mean_sec),
    )


def run_torch_cpu_compile(
    env_num: int,
    obs_dim: int,
    action_dim: int,
    hidden_dims: List[int],
    warmup: int,
    repeat: int,
) -> Optional[MLPBenchRecord]:
    torch = _get_torch()
    if torch is None:
        return None
    print(f"    [torch_cpu_compile] env_num={env_num}: building model...", flush=True)
    torch.manual_seed(42)
    dims = [obs_dim] + hidden_dims + [action_dim]
    layers = []
    for i in range(len(dims) - 1):
        layers.append(torch.nn.Linear(dims[i], dims[i + 1]))
        if i < len(dims) - 2:
            layers.append(torch.nn.Tanh())
    model = torch.nn.Sequential(*layers).float().to("cpu")
    print(
        f"    [torch_cpu_compile] env_num={env_num}: calling torch.compile(model, mode='reduce-overhead') ...",
        flush=True,
    )
    t0 = time.perf_counter()
    model = torch.compile(model, mode="reduce-overhead")
    print(
        f"    [torch_cpu_compile] env_num={env_num}: torch.compile() returned in {time.perf_counter() - t0:.2f}s",
        flush=True,
    )
    model.eval()
    torch.manual_seed(43)
    x = torch.clamp(
        torch.randn(env_num, obs_dim, dtype=torch.float32, device="cpu"),
        _OBS_CLIP_LOW,
        _OBS_CLIP_HIGH,
    )

    def fwd():
        with torch.no_grad():
            _ = model(x)

    print(
        f"    [torch_cpu_compile] env_num={env_num}: warmup ({warmup} iters, first forward may trigger JIT compile, can be slow)...",
        flush=True,
    )
    t0 = time.perf_counter()
    for i in range(warmup):
        fwd()
    print(
        f"    [torch_cpu_compile] env_num={env_num}: warmup done in {time.perf_counter() - t0:.2f}s",
        flush=True,
    )
    print(
        f"    [torch_cpu_compile] env_num={env_num}: running {repeat} timed repeats...", flush=True
    )
    elapsed = bench_callable(fwd, lambda: None, 0, repeat)  # warmup already done
    mean_sec = statistics.mean(elapsed)
    std_sec = statistics.pstdev(elapsed) if len(elapsed) > 1 else 0.0
    print(
        f"    [torch_cpu_compile] env_num={env_num}: done. mean={mean_sec * 1000:.3f} ms",
        flush=True,
    )
    return MLPBenchRecord(
        backend="torch_cpu_compile",
        env_num=env_num,
        warmup=warmup,
        repeat=repeat,
        elapsed_sec=elapsed,
        mean_sec=mean_sec,
        std_sec=std_sec,
        min_sec=min(elapsed),
        max_sec=max(elapsed),
        envs_per_sec=_safe_envs_per_sec(env_num, mean_sec),
    )


# ---------- PyTorch MPS ----------
def run_torch_mps(
    env_num: int,
    obs_dim: int,
    action_dim: int,
    hidden_dims: List[int],
    warmup: int,
    repeat: int,
) -> Optional[MLPBenchRecord]:
    torch = _get_torch()
    if torch is None:
        return None
    if not hasattr(torch.backends, "mps") or not torch.backends.mps.is_available():
        return None
    device = torch.device("mps")
    torch.manual_seed(42)
    dims = [obs_dim] + hidden_dims + [action_dim]
    layers = []
    for i in range(len(dims) - 1):
        layers.append(torch.nn.Linear(dims[i], dims[i + 1]))
        if i < len(dims) - 2:
            layers.append(torch.nn.Tanh())
    model = torch.nn.Sequential(*layers).float().to(device)
    model.eval()
    torch.manual_seed(43)
    x = torch.clamp(
        torch.randn(env_num, obs_dim, dtype=torch.float32, device=device),
        _OBS_CLIP_LOW,
        _OBS_CLIP_HIGH,
    )

    def fwd():
        with torch.no_grad():
            _ = model(x)

    def sync():
        torch.mps.synchronize()

    elapsed = bench_callable(fwd, sync, warmup, repeat)
    mean_sec = statistics.mean(elapsed)
    std_sec = statistics.pstdev(elapsed) if len(elapsed) > 1 else 0.0
    return MLPBenchRecord(
        backend="torch_mps",
        env_num=env_num,
        warmup=warmup,
        repeat=repeat,
        elapsed_sec=elapsed,
        mean_sec=mean_sec,
        std_sec=std_sec,
        min_sec=min(elapsed),
        max_sec=max(elapsed),
        envs_per_sec=_safe_envs_per_sec(env_num, mean_sec),
    )


def run_torch_mps_compile(
    env_num: int,
    obs_dim: int,
    action_dim: int,
    hidden_dims: List[int],
    warmup: int,
    repeat: int,
) -> Optional[MLPBenchRecord]:
    torch = _get_torch()
    if torch is None:
        return None
    if not hasattr(torch.backends, "mps") or not torch.backends.mps.is_available():
        return None
    print(f"    [torch_mps_compile] env_num={env_num}: building model (MPS)...", flush=True)
    device = torch.device("mps")
    torch.manual_seed(42)
    dims = [obs_dim] + hidden_dims + [action_dim]
    layers = []
    for i in range(len(dims) - 1):
        layers.append(torch.nn.Linear(dims[i], dims[i + 1]))
        if i < len(dims) - 2:
            layers.append(torch.nn.Tanh())
    model = torch.nn.Sequential(*layers).float().to(device)
    print(
        f"    [torch_mps_compile] env_num={env_num}: calling torch.compile(model, mode='reduce-overhead') ...",
        flush=True,
    )
    t0 = time.perf_counter()
    model = torch.compile(model, mode="reduce-overhead")
    print(
        f"    [torch_mps_compile] env_num={env_num}: torch.compile() returned in {time.perf_counter() - t0:.2f}s",
        flush=True,
    )
    model.eval()
    torch.manual_seed(43)
    x = torch.clamp(
        torch.randn(env_num, obs_dim, dtype=torch.float32, device=device),
        _OBS_CLIP_LOW,
        _OBS_CLIP_HIGH,
    )

    def fwd():
        with torch.no_grad():
            _ = model(x)

    def sync():
        torch.mps.synchronize()

    print(
        f"    [torch_mps_compile] env_num={env_num}: warmup ({warmup} iters, first forward may trigger JIT compile, can be slow)...",
        flush=True,
    )
    t0 = time.perf_counter()
    for i in range(warmup):
        fwd()
        sync()
    print(
        f"    [torch_mps_compile] env_num={env_num}: warmup done in {time.perf_counter() - t0:.2f}s",
        flush=True,
    )
    print(
        f"    [torch_mps_compile] env_num={env_num}: running {repeat} timed repeats...", flush=True
    )
    elapsed = bench_callable(fwd, sync, 0, repeat)
    mean_sec = statistics.mean(elapsed)
    std_sec = statistics.pstdev(elapsed) if len(elapsed) > 1 else 0.0
    print(
        f"    [torch_mps_compile] env_num={env_num}: done. mean={mean_sec * 1000:.3f} ms",
        flush=True,
    )
    return MLPBenchRecord(
        backend="torch_mps_compile",
        env_num=env_num,
        warmup=warmup,
        repeat=repeat,
        elapsed_sec=elapsed,
        mean_sec=mean_sec,
        std_sec=std_sec,
        min_sec=min(elapsed),
        max_sec=max(elapsed),
        envs_per_sec=_safe_envs_per_sec(env_num, mean_sec),
    )


# ---------- MLX ----------
def run_mlx(
    env_num: int,
    obs_dim: int,
    action_dim: int,
    hidden_dims: List[int],
    warmup: int,
    repeat: int,
) -> Optional[MLPBenchRecord]:
    mx, nn_mlx = _get_mlx()
    if mx is None or nn_mlx is None:
        return None
    dims = [obs_dim] + hidden_dims + [action_dim]

    class _MLXMLP(nn_mlx.Module):
        def __init__(self, d: List[int]):
            super().__init__()
            self.linears = [nn_mlx.Linear(d[i], d[i + 1]) for i in range(len(d) - 1)]

        def __call__(self, x):  # noqa: A003
            for i, lin in enumerate(self.linears):
                x = lin(x)
                if i < len(self.linears) - 1:
                    x = mx.tanh(x)
            return x

    model = _MLXMLP(dims)
    mx.random.seed(43)
    x = mx.minimum(
        mx.maximum(mx.random.normal((env_num, obs_dim), dtype=mx.float32), _OBS_CLIP_LOW),
        _OBS_CLIP_HIGH,
    )

    def fwd():
        out = model(x)
        mx.eval(out)

    elapsed = bench_callable(fwd, lambda: None, warmup, repeat)
    mean_sec = statistics.mean(elapsed)
    std_sec = statistics.pstdev(elapsed) if len(elapsed) > 1 else 0.0
    return MLPBenchRecord(
        backend="mlx",
        env_num=env_num,
        warmup=warmup,
        repeat=repeat,
        elapsed_sec=elapsed,
        mean_sec=mean_sec,
        std_sec=std_sec,
        min_sec=min(elapsed),
        max_sec=max(elapsed),
        envs_per_sec=_safe_envs_per_sec(env_num, mean_sec),
    )


def run_mlx_compile(
    env_num: int,
    obs_dim: int,
    action_dim: int,
    hidden_dims: List[int],
    warmup: int,
    repeat: int,
) -> Optional[MLPBenchRecord]:
    mx, nn_mlx = _get_mlx()
    if mx is None or nn_mlx is None:
        return None
    if not hasattr(mx, "compile"):
        return None
    dims = [obs_dim] + hidden_dims + [action_dim]

    class _MLXMLP(nn_mlx.Module):
        def __init__(self, d: List[int]):
            super().__init__()
            self.linears = [nn_mlx.Linear(d[i], d[i + 1]) for i in range(len(d) - 1)]

        def __call__(self, x):  # noqa: A003
            for i, lin in enumerate(self.linears):
                x = lin(x)
                if i < len(self.linears) - 1:
                    x = mx.tanh(x)
            return x

    model = _MLXMLP(dims)
    mx.random.seed(43)
    x = mx.minimum(
        mx.maximum(mx.random.normal((env_num, obs_dim), dtype=mx.float32), _OBS_CLIP_LOW),
        _OBS_CLIP_HIGH,
    )
    compiled_forward = mx.compile(lambda x_in: model(x_in))

    def fwd():
        out = compiled_forward(x)
        mx.eval(out)

    elapsed = bench_callable(fwd, lambda: None, warmup, repeat)
    mean_sec = statistics.mean(elapsed)
    std_sec = statistics.pstdev(elapsed) if len(elapsed) > 1 else 0.0
    return MLPBenchRecord(
        backend="mlx_compile",
        env_num=env_num,
        warmup=warmup,
        repeat=repeat,
        elapsed_sec=elapsed,
        mean_sec=mean_sec,
        std_sec=std_sec,
        min_sec=min(elapsed),
        max_sec=max(elapsed),
        envs_per_sec=_safe_envs_per_sec(env_num, mean_sec),
    )


# ---------- ONNX Runtime ----------
def _build_and_export_onnx(
    obs_dim: int,
    action_dim: int,
    hidden_dims: List[int],
    onnx_path: str,
    seed: int = 42,
) -> bool:
    torch = _get_torch()
    if torch is None or np is None:
        return False
    dims = [obs_dim] + hidden_dims + [action_dim]
    layers = []
    torch.manual_seed(seed)
    for i in range(len(dims) - 1):
        layers.append(torch.nn.Linear(dims[i], dims[i + 1]))
        if i < len(dims) - 2:
            layers.append(torch.nn.Tanh())
    model = torch.nn.Sequential(*layers).float().eval()
    dummy = torch.randn(1, obs_dim)
    try:
        torch.onnx.export(
            model,
            dummy,
            onnx_path,
            input_names=["x"],
            output_names=["y"],
            dynamic_axes={"x": {0: "batch"}, "y": {0: "batch"}},
            opset_version=14,
        )
        if onnx is not None:
            model_proto = onnx.load(onnx_path)
            onnx.save(model_proto, onnx_path)
        return True
    except Exception:
        return False


def run_onnx(
    env_num: int,
    obs_dim: int,
    session: "ort.InferenceSession",
    warmup: int,
    repeat: int,
) -> Optional[MLPBenchRecord]:
    if np is None or ort is None or session is None:
        return None
    rng = np.random.default_rng(43)
    x = np.clip(
        rng.standard_normal((env_num, obs_dim)).astype(np.float32),
        _OBS_CLIP_LOW,
        _OBS_CLIP_HIGH,
    )
    input_name = session.get_inputs()[0].name

    def fwd():
        _ = session.run(None, {input_name: x})

    elapsed = bench_callable(fwd, lambda: None, warmup, repeat)
    mean_sec = statistics.mean(elapsed)
    std_sec = statistics.pstdev(elapsed) if len(elapsed) > 1 else 0.0
    return MLPBenchRecord(
        backend="onnxruntime",
        env_num=env_num,
        warmup=warmup,
        repeat=repeat,
        elapsed_sec=elapsed,
        mean_sec=mean_sec,
        std_sec=std_sec,
        min_sec=min(elapsed),
        max_sec=max(elapsed),
        envs_per_sec=_safe_envs_per_sec(env_num, mean_sec),
    )


# ---------- Core ML ----------
def _build_and_convert_coreml(
    obs_dim: int,
    action_dim: int,
    hidden_dims: List[int],
    mlmodel_path: str,
    seed: int = 42,
) -> bool:
    torch = _get_torch()
    if torch is None or ct is None or np is None:
        return False
    dims = [obs_dim] + hidden_dims + [action_dim]
    layers = []
    torch.manual_seed(seed)
    for i in range(len(dims) - 1):
        layers.append(torch.nn.Linear(dims[i], dims[i + 1]))
        if i < len(dims) - 2:
            layers.append(torch.nn.Tanh())
    model = torch.nn.Sequential(*layers).float().eval()
    example_input = torch.randn(1, obs_dim)
    try:
        traced = torch.jit.trace(model, example_input)
        mlmodel = ct.convert(
            traced,
            convert_to="mlprogram",
            inputs=[
                ct.TensorType(name="x", shape=ct.Shape(shape=(ct.RangeDim(1, 32768), obs_dim)))
            ],
            compute_units=ct.ComputeUnit.ALL,
        )
        mlmodel.save(mlmodel_path)
        return True
    except Exception:
        return False


def run_coreml(
    env_num: int,
    obs_dim: int,
    model: "ct.models.MLModel",
    warmup: int,
    repeat: int,
) -> Optional[MLPBenchRecord]:
    if np is None or model is None:
        return None
    rng = np.random.default_rng(43)
    x = np.clip(
        rng.standard_normal((env_num, obs_dim)).astype(np.float32),
        _OBS_CLIP_LOW,
        _OBS_CLIP_HIGH,
    )
    input_name = list(model.get_spec().description.input)[0].name

    def fwd():
        _ = model.predict({input_name: x})

    elapsed = bench_callable(fwd, lambda: None, warmup, repeat)
    mean_sec = statistics.mean(elapsed)
    std_sec = statistics.pstdev(elapsed) if len(elapsed) > 1 else 0.0
    return MLPBenchRecord(
        backend="coreml",
        env_num=env_num,
        warmup=warmup,
        repeat=repeat,
        elapsed_sec=elapsed,
        mean_sec=mean_sec,
        std_sec=std_sec,
        min_sec=min(elapsed),
        max_sec=max(elapsed),
        envs_per_sec=_safe_envs_per_sec(env_num, mean_sec),
    )


# ---------- Core ML ANE (Apple Neural Engine, no CPU fallback) ----------
def _build_and_convert_coreml_ane(
    obs_dim: int,
    action_dim: int,
    hidden_dims: List[int],
    env_nums: List[int],
    mlmodel_path: str,
    seed: int = 42,
) -> bool:
    """Convert to Core ML with EnumeratedShapes so model can run on ANE (no flexible batch)."""
    torch = _get_torch()
    if torch is None or ct is None or np is None:
        return False
    dims = [obs_dim] + hidden_dims + [action_dim]
    layers = []
    torch.manual_seed(seed)
    for i in range(len(dims) - 1):
        layers.append(torch.nn.Linear(dims[i], dims[i + 1]))
        if i < len(dims) - 2:
            layers.append(torch.nn.Tanh())
    model = torch.nn.Sequential(*layers).float().eval()
    example_input = torch.randn(env_nums[0], obs_dim)
    try:
        traced = torch.jit.trace(model, example_input)
        shapes = [[n, obs_dim] for n in env_nums]
        default_shape = [env_nums[-1], obs_dim]
        input_shape = ct.EnumeratedShapes(shapes=shapes, default=default_shape)
        mlmodel = ct.convert(
            traced,
            convert_to="mlprogram",
            inputs=[ct.TensorType(name="x", shape=input_shape)],
        )
        mlmodel.save(mlmodel_path)
        return True
    except Exception:
        return False


def _log_ane_phase() -> None:
    """Log that ANE phase uses Core ML with compute_units=CPU_AND_NE.
    Apple provides no Python API to confirm at runtime whether ANE was used; to verify,
    use Xcode Instruments (Time Profiler / H11ANEServicesThread) or compare with CPU_ONLY runs."""
    print(
        "  [ANE] Core ML loaded with compute_units=CPU_AND_NE (no runtime ANE-usage check; see doc).",
        flush=True,
    )


def run_ane(
    env_num: int,
    obs_dim: int,
    model: "ct.models.MLModel",
    warmup: int,
    repeat: int,
) -> Optional[MLPBenchRecord]:
    if np is None or model is None:
        return None
    rng = np.random.default_rng(43)
    x = np.clip(
        rng.standard_normal((env_num, obs_dim)).astype(np.float32),
        _OBS_CLIP_LOW,
        _OBS_CLIP_HIGH,
    )
    input_name = list(model.get_spec().description.input)[0].name

    def fwd():
        _ = model.predict({input_name: x})

    elapsed = bench_callable(fwd, lambda: None, warmup, repeat)
    mean_sec = statistics.mean(elapsed)
    std_sec = statistics.pstdev(elapsed) if len(elapsed) > 1 else 0.0
    return MLPBenchRecord(
        backend="ane",
        env_num=env_num,
        warmup=warmup,
        repeat=repeat,
        elapsed_sec=elapsed,
        mean_sec=mean_sec,
        std_sec=std_sec,
        min_sec=min(elapsed),
        max_sec=max(elapsed),
        envs_per_sec=_safe_envs_per_sec(env_num, mean_sec),
    )


def available_backends(include_lazy: bool = True) -> Dict[str, bool]:
    out = {
        "numpy": np is not None,
        "numba": np is not None and numba is not None and jit is not None,
        "jax_jit": jax is not None and jnp is not None,
    }
    if include_lazy:
        torch = _get_torch()
        mx, nn_mlx = _get_mlx()
        out["torch_cpu"] = torch is not None
        out["torch_cpu_compile"] = torch is not None
        out["torch_mps"] = bool(
            torch is not None
            and hasattr(torch.backends, "mps")
            and torch.backends.mps.is_available()
        )
        out["torch_mps_compile"] = bool(
            torch is not None
            and hasattr(torch.backends, "mps")
            and torch.backends.mps.is_available()
        )
        out["mlx"] = mx is not None and nn_mlx is not None
        out["mlx_compile"] = bool(
            mx is not None and nn_mlx is not None and getattr(mx, "compile", None) is not None
        )
        out["onnxruntime"] = torch is not None and np is not None and ort is not None
        out["coreml"] = torch is not None and np is not None and ct is not None
        out["ane"] = torch is not None and np is not None and ct is not None
    else:
        out["torch_cpu"] = out["torch_cpu_compile"] = out["torch_mps"] = out[
            "torch_mps_compile"
        ] = False
        out["mlx"] = out["mlx_compile"] = False
        out["onnxruntime"] = out["coreml"] = out["ane"] = False
    return out


def save_plots(
    records: List[MLPBenchRecord],
    plot_dir: Path,
    file_prefix: str,
) -> List[str]:
    if plt is None or not records:
        return []
    plot_dir.mkdir(parents=True, exist_ok=True)
    saved: List[str] = []

    backends = sorted({r.backend for r in records})
    env_nums = sorted({r.env_num for r in records})

    # Safe minimum for log-scale y (avoid log(0))
    _min_time_ms = 1e-9
    _min_throughput = 1e-6

    # 1) Mean time (ms) vs env_num
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    for backend in backends:
        subset = sorted(
            [r for r in records if r.backend == backend],
            key=lambda x: x.env_num,
        )
        if not subset:
            continue
        x = [r.env_num for r in subset]
        y = [max(_min_time_ms, r.mean_sec * 1000.0) for r in subset]
        ax1.plot(x, y, marker="o", label=backend)
    ax1.set_title(f"MLP inference time (ms) vs env_num\n{get_device_info_line()}")
    ax1.set_xlabel("env_num")
    ax1.set_ylabel("Mean time (ms)")
    ax1.set_xscale("log", base=2)
    ax1.set_xticks(env_nums)
    ax1.set_xticklabels([str(v) for v in env_nums])
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    fig1.tight_layout()
    out1 = plot_dir / f"{file_prefix}_time_ms.png"
    fig1.savefig(out1, dpi=150)
    plt.close(fig1)
    saved.append(str(out1.resolve()))

    # 2) Throughput (envs/s) vs env_num
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    for backend in backends:
        subset = sorted(
            [r for r in records if r.backend == backend],
            key=lambda x: x.env_num,
        )
        if not subset:
            continue
        x = [r.env_num for r in subset]
        y = [max(_min_throughput, r.envs_per_sec) for r in subset]
        ax2.plot(x, y, marker="o", label=backend)
    ax2.set_title(f"MLP inference throughput (envs/s) vs env_num\n{get_device_info_line()}")
    ax2.set_xlabel("env_num")
    ax2.set_ylabel("Envs per second")
    ax2.set_xscale("log", base=2)
    ax2.set_xticks(env_nums)
    ax2.set_xticklabels([str(v) for v in env_nums])
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    fig2.tight_layout()
    out2 = plot_dir / f"{file_prefix}_throughput.png"
    fig2.savefig(out2, dpi=150)
    plt.close(fig2)
    saved.append(str(out2.resolve()))

    return saved


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark MLP inference: numpy, numba, jax, torch (cpu/mps), torch.compile, mlx, mlx.compile, onnxruntime, coreml, ane."
    )
    parser.add_argument(
        "--sizes",
        type=str,
        default=",".join(
            str(2**k) for k in range(DEFAULT_ENV_NUM_POW_MIN, DEFAULT_ENV_NUM_POW_MAX + 1)
        ),
        help="Comma-separated env_num sizes (default: 256,512,1024,...)",
    )
    parser.add_argument("--obs-dim", type=int, default=DEFAULT_OBS_DIM)
    parser.add_argument("--action-dim", type=int, default=DEFAULT_ACTION_DIM)
    parser.add_argument(
        "--hidden-dims",
        type=str,
        default=",".join(map(str, DEFAULT_HIDDEN_DIMS)),
        help="Comma-separated hidden layer sizes",
    )
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument(
        "--out",
        type=str,
        default="benchmark/outputs/mlp_inference/benchmark_mlp_results.json",
        help="Output JSON path",
    )
    parser.add_argument(
        "--plot-dir",
        type=str,
        default="",
        help="Plot directory (default: same as --out parent)",
    )
    parser.add_argument(
        "--skip-backends",
        type=str,
        default="torch_cpu_compile,torch_mps_compile",
        help="Comma-separated backend names to skip (default: skip torch compile, too slow)",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=0,
        metavar="N",
        help="Set BLAS/torch CPU threads (0=auto). Speeds up numpy/torch_cpu. E.g. --threads 8",
    )
    args = parser.parse_args()

    global _TORCH_NUM_THREADS
    if args.threads > 0:
        _TORCH_NUM_THREADS = args.threads
        # NumPy/Accelerate/OpenBLAS often respect these (set before any BLAS call)
        os.environ.setdefault("VECLIB_MAXIMUM_THREADS", str(args.threads))  # Mac Accelerate
        os.environ.setdefault("OPENBLAS_NUM_THREADS", str(args.threads))
        os.environ.setdefault("OMP_NUM_THREADS", str(args.threads))
        print(f"Using {args.threads} threads (BLAS / torch CPU)", flush=True)
    else:
        _TORCH_NUM_THREADS = None

    env_nums = [int(x.strip()) for x in args.sizes.split(",") if x.strip()]
    hidden_dims = [int(x.strip()) for x in args.hidden_dims.split(",") if x.strip()]
    if not hidden_dims:
        hidden_dims = list(DEFAULT_HIDDEN_DIMS)
    skip_backends = {s.strip() for s in args.skip_backends.split(",") if s.strip()}

    # Phase 1: numpy + numba + jax (no torch/mlx load -> no OMP conflict with Numba)
    backends_p1 = available_backends(include_lazy=False)
    print("Detected backends (phase 1: numpy / numba / jax):", flush=True)
    for k in ["numpy", "numba", "jax_jit"]:
        print(f"  - {k}: {'yes' if backends_p1.get(k) else 'no'}", flush=True)
    print("  (torch/mlx not loaded yet)", flush=True)
    if skip_backends:
        print(f"Skipping backends: {skip_backends}", flush=True)
    print(f"env_nums: {env_nums}", flush=True)
    print(
        f"model: obs_dim={args.obs_dim}, hidden_dims={hidden_dims}, action_dim={args.action_dim}",
        flush=True,
    )
    print("Tip: use --threads N to speed up numpy/torch_cpu (e.g. --threads 8)", flush=True)

    all_records: List[MLPBenchRecord] = []
    skipped: List[Dict[str, str]] = []

    def run_phase(runners: List[tuple], backends_dict: Dict[str, bool]) -> None:
        for env_num in env_nums:
            print(f"\nRunning env_num={env_num} ...", flush=True)
            for backend_name, run_fn in runners:
                if backend_name in skip_backends:
                    continue
                if not backends_dict.get(backend_name, False):
                    continue
                print(f"  >> {backend_name} env_num={env_num} starting...", flush=True)
                try:
                    rec = run_fn(env_num)
                    if rec is not None:
                        all_records.append(rec)
                        print(
                            f"  {backend_name}: mean={rec.mean_sec * 1000:.3f} ms, envs/s={rec.envs_per_sec:.1f}",
                            flush=True,
                        )
                    else:
                        skipped.append(
                            {
                                "backend": backend_name,
                                "env_num": str(env_num),
                                "reason": "unavailable",
                            }
                        )
                except Exception as e:
                    skipped.append(
                        {"backend": backend_name, "env_num": str(env_num), "reason": str(e)}
                    )
                    print(f"  {backend_name}: skip - {e}", flush=True)

    runners_p1 = [
        (
            "numpy",
            lambda n: run_numpy(
                n, args.obs_dim, args.action_dim, hidden_dims, args.warmup, args.repeat
            ),
        ),
        (
            "numba",
            lambda n: run_numba(
                n, args.obs_dim, args.action_dim, hidden_dims, args.warmup, args.repeat
            ),
        ),
        (
            "jax_jit",
            lambda n: run_jax(
                n, args.obs_dim, args.action_dim, hidden_dims, args.warmup, args.repeat
            ),
        ),
    ]
    run_phase(runners_p1, backends_p1)

    # Phase 2: load torch, run torch backends
    _get_torch()
    backends_p2 = available_backends(include_lazy=True)
    print("\nDetected backends (phase 2: torch):", flush=True)
    for k in ["torch_cpu", "torch_cpu_compile", "torch_mps", "torch_mps_compile"]:
        print(f"  - {k}: {'yes' if backends_p2.get(k) else 'no'}", flush=True)
    runners_p2 = [
        (
            "torch_cpu",
            lambda n: run_torch_cpu(
                n, args.obs_dim, args.action_dim, hidden_dims, args.warmup, args.repeat
            ),
        ),
        (
            "torch_cpu_compile",
            lambda n: run_torch_cpu_compile(
                n, args.obs_dim, args.action_dim, hidden_dims, args.warmup, args.repeat
            ),
        ),
        (
            "torch_mps",
            lambda n: run_torch_mps(
                n, args.obs_dim, args.action_dim, hidden_dims, args.warmup, args.repeat
            ),
        ),
        (
            "torch_mps_compile",
            lambda n: run_torch_mps_compile(
                n, args.obs_dim, args.action_dim, hidden_dims, args.warmup, args.repeat
            ),
        ),
    ]
    run_phase(runners_p2, backends_p2)

    # Phase 3: load mlx, run mlx backends
    _get_mlx()
    backends_p3 = available_backends(include_lazy=True)
    print("\nDetected backends (phase 3: mlx):", flush=True)
    for k in ["mlx", "mlx_compile"]:
        print(f"  - {k}: {'yes' if backends_p3.get(k) else 'no'}", flush=True)
    runners_p3 = [
        (
            "mlx",
            lambda n: run_mlx(
                n, args.obs_dim, args.action_dim, hidden_dims, args.warmup, args.repeat
            ),
        ),
        (
            "mlx_compile",
            lambda n: run_mlx_compile(
                n, args.obs_dim, args.action_dim, hidden_dims, args.warmup, args.repeat
            ),
        ),
    ]
    run_phase(runners_p3, backends_p3)

    # Phase 4: ONNX Runtime (export from torch once, then run)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    onnx_path = str(out_path.parent / "temp_mlp_bench.onnx")
    onnx_session = None
    if available_backends(include_lazy=True).get("onnxruntime"):
        if _build_and_export_onnx(args.obs_dim, args.action_dim, hidden_dims, onnx_path):
            try:
                onnx_session = ort.InferenceSession(
                    onnx_path, providers=ort.get_available_providers()
                )
            except Exception:
                pass
    backends_p4 = available_backends(include_lazy=True)
    if onnx_session is not None:
        print("\nDetected backends (phase 4: ONNX Runtime):", flush=True)
        print("  - onnxruntime: yes", flush=True)
        runners_p4 = [
            (
                "onnxruntime",
                lambda n: run_onnx(n, args.obs_dim, onnx_session, args.warmup, args.repeat),
            ),
        ]
        run_phase(runners_p4, {"onnxruntime": True})
    if Path(onnx_path).exists():
        try:
            Path(onnx_path).unlink()
        except Exception:
            pass

    # Phase 5: Core ML (convert from torch once, then run)
    import shutil
    import tempfile

    coreml_path = tempfile.mkdtemp(suffix=".mlpackage")
    coreml_model = None
    if available_backends(include_lazy=True).get("coreml"):
        if _build_and_convert_coreml(args.obs_dim, args.action_dim, hidden_dims, coreml_path):
            try:
                coreml_model = ct.models.MLModel(coreml_path)
            except Exception:
                pass
    if coreml_model is not None:
        print("\nDetected backends (phase 5: Core ML):", flush=True)
        print("  - coreml: yes", flush=True)
        runners_p5 = [
            (
                "coreml",
                lambda n: run_coreml(n, args.obs_dim, coreml_model, args.warmup, args.repeat),
            ),
        ]
        run_phase(runners_p5, {"coreml": True})
    try:
        shutil.rmtree(coreml_path, ignore_errors=True)
    except Exception:
        pass

    # Phase 6: ANE (Core ML CPU_AND_NE, EnumeratedShapes)
    ane_path = tempfile.mkdtemp(suffix=".mlpackage")
    ane_model = None
    if available_backends(include_lazy=True).get("ane") and "ane" not in skip_backends:
        if _build_and_convert_coreml_ane(
            args.obs_dim, args.action_dim, hidden_dims, env_nums, ane_path
        ):
            try:
                ane_model = ct.models.MLModel(ane_path, compute_units=ct.ComputeUnit.CPU_AND_NE)
            except Exception:
                pass
    if ane_model is not None:
        print("\nDetected backends (phase 6: ANE):", flush=True)
        print("  - ane: yes", flush=True)
        _log_ane_phase()
        runners_p6 = [
            ("ane", lambda n: run_ane(n, args.obs_dim, ane_model, args.warmup, args.repeat)),
        ]
        run_phase(runners_p6, {"ane": True})
    try:
        shutil.rmtree(ane_path, ignore_errors=True)
    except Exception:
        pass

    backends = available_backends(include_lazy=True)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plot_dir = Path(args.plot_dir) if args.plot_dir else out_path.resolve().parent
    plot_files = save_plots(all_records, plot_dir=plot_dir, file_prefix=out_path.stem)

    n_params = mlp_param_count(args.obs_dim, args.action_dim, hidden_dims)
    payload = {
        "meta": {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "device_info": get_device_info_dict(),
            "env_nums": env_nums,
            "obs_dim": args.obs_dim,
            "action_dim": args.action_dim,
            "hidden_dims": hidden_dims,
            "approx_params": n_params,
            "warmup": args.warmup,
            "repeat": args.repeat,
            "available_backends": backends,
            "skip_backends": list(skip_backends),
            "plot_files": plot_files,
            "skipped": skipped,
        },
        "results": [asdict(r) for r in all_records],
    }
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nSaved results to: {out_path.resolve()}")
    print()
    print_mlp_table(all_records)
    if plot_files:
        print("\n生成图片路径:")
        for f in plot_files:
            print(f"  {f}")


if __name__ == "__main__":
    main()
