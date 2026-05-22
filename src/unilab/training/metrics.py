"""Shared training metric helpers."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any

import torch
from omegaconf import OmegaConf


def _json_ready(value: Any) -> Any:
    if OmegaConf.is_config(value):
        return OmegaConf.to_container(value, resolve=True)
    if isinstance(value, dict):
        return {str(k): _json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def stable_config_hash(cfg: Any) -> str:
    """Return a stable SHA256 hash for a resolved config-like object."""
    payload = json.dumps(_json_ready(cfg), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def build_seed_bundle(cfg: Any) -> dict[str, int | str | None]:
    """Collect the deterministic seed knobs currently visible to training."""
    algo_seed = OmegaConf.select(cfg, "algo.seed") if OmegaConf.is_config(cfg) else None
    return {
        "algo_seed": int(algo_seed) if algo_seed is not None else None,
        "torch_initial_seed": int(torch.initial_seed()),
        "cuda_initial_seed": (
            int(torch.cuda.initial_seed()) if torch.cuda.is_available() else None
        ),
    }


@dataclass
class RunFingerprint:
    config_hash: str
    seed_bundle: dict[str, int | str | None]


def build_run_fingerprint(cfg: Any) -> RunFingerprint:
    return RunFingerprint(config_hash=stable_config_hash(cfg), seed_bundle=build_seed_bundle(cfg))


def writer_add_scalar(writer: Any, key: str, value: Any, step: int) -> None:
    writer.add_scalar(key, float(value), step)


def writer_add_text(writer: Any, key: str, value: str, step: int) -> None:
    if hasattr(writer, "add_text"):
        writer.add_text(key, value, step)
