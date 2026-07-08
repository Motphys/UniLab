"""Torch CPU thread budget helpers for off-policy training.

Off-policy training runs Torch in multiple processes: the learner, the CPU
collector, and optional multi-GPU learner workers. PyTorch defaults each process
to a host-sized thread pool, so every role needs an explicit budget.
"""

from __future__ import annotations

import os
import sys
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from typing import Any, cast

_AUTO = "auto"
_BLAS_THREAD_ENV_KEYS = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)


def _cfg_get(cfg: Any, key: str, default: Any) -> Any:
    if cfg is None:
        return default
    if isinstance(cfg, Mapping):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def _as_positive_int(value: Any, *, field: str) -> int:
    try:
        resolved = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"training.torch_threads.{field} must be a positive int or 'auto'"
        ) from exc
    if resolved < 1:
        raise ValueError(f"training.torch_threads.{field} must be >= 1, got {value!r}")
    return resolved


def _resolve_role_threads(value: Any, *, role: str, cpu_count: int) -> int:
    if value is None or str(value).lower() == _AUTO:
        if role == "collector":
            return min(4, max(1, cpu_count // 16))
        return min(8, max(1 if cpu_count < 4 else 2, cpu_count // 8))
    return _as_positive_int(value, field=f"{role}_num_threads")


def _resolve_interop_threads(value: Any, *, role: str) -> int:
    if value is None or str(value).lower() == _AUTO:
        return 1
    return _as_positive_int(value, field=f"{role}_num_interop_threads")


def _resolve_compile_threads(value: Any, *, cpu_count: int) -> int:
    if value is None or str(value).lower() == _AUTO:
        return min(2, max(1, cpu_count // 32))
    return _as_positive_int(value, field="compile_threads")


def resolve_torch_thread_runtime(
    cfg: Any,
    *,
    cpu_count: int | None = None,
) -> dict[str, Any]:
    """Resolve ``training.torch_threads`` into a spawn-safe runtime manifest."""
    cpu_total = max(1, int(cpu_count or os.cpu_count() or 1))
    enabled = bool(_cfg_get(cfg, "enabled", True))
    set_env_vars = bool(_cfg_get(cfg, "set_env_vars", True))
    if not enabled:
        return {
            "enabled": False,
            "cpu_count": cpu_total,
            "set_env_vars": set_env_vars,
        }
    learner_threads = _resolve_role_threads(
        _cfg_get(cfg, "learner_num_threads", _AUTO),
        role="learner",
        cpu_count=cpu_total,
    )
    collector_threads = _resolve_role_threads(
        _cfg_get(cfg, "collector_num_threads", _AUTO),
        role="collector",
        cpu_count=cpu_total,
    )
    learner_interop = _resolve_interop_threads(
        _cfg_get(cfg, "learner_num_interop_threads", 1),
        role="learner",
    )
    collector_interop = _resolve_interop_threads(
        _cfg_get(cfg, "collector_num_interop_threads", 1),
        role="collector",
    )
    compile_threads = _resolve_compile_threads(
        _cfg_get(cfg, "compile_threads", _AUTO),
        cpu_count=cpu_total,
    )
    return {
        "enabled": enabled,
        "cpu_count": cpu_total,
        "set_env_vars": set_env_vars,
        "compile_threads": compile_threads,
        "learner": {
            "num_threads": learner_threads,
            "num_interop_threads": learner_interop,
        },
        "collector": {
            "num_threads": collector_threads,
            "num_interop_threads": collector_interop,
        },
    }


def _set_torch_interop_threads(torch_module: Any, num_threads: int, *, role: str) -> None:
    get_num_interop_threads = getattr(torch_module, "get_num_interop_threads", None)
    set_num_interop_threads = getattr(torch_module, "set_num_interop_threads", None)
    if not callable(set_num_interop_threads):
        return
    try:
        if callable(get_num_interop_threads):
            current_threads = cast(Any, get_num_interop_threads())
            if int(current_threads) == int(num_threads):
                return
        set_num_interop_threads(int(num_threads))
    except RuntimeError as exc:
        print(
            "[offpolicy.thread_budget] unable to set "
            f"{role} torch inter-op threads to {num_threads}: {exc}",
            file=sys.stderr,
        )


def _set_torch_compile_threads(torch_module: Any, num_threads: int) -> None:
    os.environ["TORCHINDUCTOR_COMPILE_THREADS"] = str(int(num_threads))
    try:
        config = getattr(getattr(torch_module, "_inductor", None), "config", None)
        if config is None:
            from torch._inductor import config as imported_config  # type: ignore[import-not-found]

            config = imported_config

        setattr(config, "compile_threads", int(num_threads))
    except Exception as exc:
        print(
            "[offpolicy.thread_budget] unable to set torch inductor compile_threads "
            f"to {num_threads}: {exc}",
            file=sys.stderr,
        )


def _thread_env_values(runtime: Mapping[str, Any], *, role: str) -> dict[str, str]:
    if role not in ("learner", "collector"):
        raise ValueError(f"Unsupported torch thread budget role: {role!r}")
    role_cfg = runtime[role]
    num_threads = str(int(role_cfg["num_threads"]))
    values = {key: num_threads for key in _BLAS_THREAD_ENV_KEYS}
    values["TORCH_NUM_THREADS"] = num_threads
    values["TORCHINDUCTOR_COMPILE_THREADS"] = str(int(runtime["compile_threads"]))
    return values


@contextmanager
def torch_thread_env(runtime: Mapping[str, Any] | None, *, role: str) -> Iterator[None]:
    """Temporarily expose a role's thread budget to child process startup."""
    if (
        not runtime
        or not bool(runtime.get("enabled", True))
        or not bool(runtime.get("set_env_vars", True))
    ):
        yield
        return

    values = _thread_env_values(runtime, role=role)
    previous = {key: os.environ.get(key) for key in values}
    try:
        os.environ.update(values)
        yield
    finally:
        for key, old_value in previous.items():
            if old_value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = old_value


def apply_torch_thread_runtime(
    runtime: Mapping[str, Any] | None,
    *,
    role: str,
    torch_module: Any | None = None,
) -> dict[str, Any]:
    """Apply a resolved runtime manifest in the current process.

    Returns the role manifest used by tests and status logging.
    """
    if not runtime or not bool(runtime.get("enabled", True)):
        return {"enabled": False, "role": role}
    if role not in ("learner", "collector"):
        raise ValueError(f"Unsupported torch thread budget role: {role!r}")

    role_cfg = dict(runtime[role])
    num_threads = int(role_cfg["num_threads"])
    num_interop_threads = int(role_cfg["num_interop_threads"])
    compile_threads = int(runtime["compile_threads"])

    if bool(runtime.get("set_env_vars", True)):
        os.environ.update(_thread_env_values(runtime, role=role))

    torch_runtime = torch_module
    if torch_runtime is None:
        import torch as imported_torch

        torch_runtime = imported_torch

    set_num_threads = getattr(torch_runtime, "set_num_threads", None)
    if callable(set_num_threads):
        set_num_threads(num_threads)
    _set_torch_interop_threads(torch_runtime, num_interop_threads, role=role)
    _set_torch_compile_threads(torch_runtime, compile_threads)

    return {
        "enabled": True,
        "role": role,
        "num_threads": num_threads,
        "num_interop_threads": num_interop_threads,
        "compile_threads": compile_threads,
    }


def format_torch_thread_runtime(runtime: Mapping[str, Any] | None) -> str:
    if not runtime or not bool(runtime.get("enabled", True)):
        return "Torch thread budget: disabled"
    learner = runtime["learner"]
    collector = runtime["collector"]
    return (
        "Torch thread budget: "
        f"learner={learner['num_threads']} intra/{learner['num_interop_threads']} inter, "
        f"collector={collector['num_threads']} intra/{collector['num_interop_threads']} inter, "
        f"compile_workers={runtime['compile_threads']}"
    )
