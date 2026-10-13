"""Hydra owner resolution for the public collector graph execution mode."""

from __future__ import annotations

from typing import Any

from omegaconf import OmegaConf

from unilab.base.collector_graph import CollectorGraphMode, resolve_collector_graph_mode


def resolve_training_collector_graph_mode(
    training_cfg: Any, *, default: CollectorGraphMode = CollectorGraphMode.DISABLED
) -> CollectorGraphMode:
    """Resolve the cold-path training mode before runner construction."""
    value = OmegaConf.select(training_cfg, "collector_graph", default=None)
    if value is None:
        return default
    return resolve_collector_graph_mode(value)


__all__ = ["resolve_training_collector_graph_mode"]
