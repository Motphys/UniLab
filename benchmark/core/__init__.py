"""Core benchmark utilities for standardized benchmarking."""

from .backends import available_backends, numpy_dtype, torch_dtype, mlx_dtype, sync_backend
from .device_info import get_device_info_dict, get_device_info_line
from .mlp_utils import MLPBenchRecord, env_nums_pow2, mlp_param_count, trimmed_mean, print_mlp_table
from .output import print_table, save_json
from .plotting import save_line_plot
from .record import BenchRecord
from .runner import bench_callable, summarize
from .utils import parse_sizes, pow2_sizes, parse_dtypes, normalize_dtypes

__all__ = [
    "available_backends",
    "numpy_dtype",
    "torch_dtype",
    "mlx_dtype",
    "sync_backend",
    "get_device_info_dict",
    "get_device_info_line",
    "BenchRecord",
    "MLPBenchRecord",
    "bench_callable",
    "summarize",
    "parse_sizes",
    "pow2_sizes",
    "parse_dtypes",
    "normalize_dtypes",
    "print_table",
    "save_json",
    "save_line_plot",
    "env_nums_pow2",
    "mlp_param_count",
    "trimmed_mean",
    "print_mlp_table",
]
