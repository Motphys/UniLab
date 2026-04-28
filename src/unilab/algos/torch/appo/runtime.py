"""Runtime resolution helpers for APPO script assembly.

This module keeps entrypoint scripts generic: they resolve an APPO runtime bundle
from owner config and then call the returned runner/play entrypoints without
knowing which concrete runtime implementation is active.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class APPORuntime:
    """Resolved APPO runtime entrypoints consumed by the generic script.

    Args:
        runner_cls: Runner class used for APPO training mode.
        play_fn: Play-mode callable used for checkpoint playback.

    Returns:
        Immutable APPO runtime bundle selected from owner config.
    """

    runner_cls: type[Any]
    play_fn: Callable[..., str | None]


def resolve_appo_runtime(
    rl_cfg: dict[str, Any],
    *,
    default_play_fn: Callable[..., str | None],
) -> APPORuntime:
    """Resolve the APPO runtime bundle from owner config.

    Args:
        rl_cfg: Resolved algorithm config dictionary from Hydra composition.
        default_play_fn: Generic APPO play function used when no custom runtime
            resolver is selected by the owner config.

    Returns:
        ``APPORuntime`` containing the train and play entrypoints for the
        selected APPO runtime.
    """
    runtime_resolver = rl_cfg.get("runtime_resolver")
    if runtime_resolver in (None, ""):
        from unilab.algos.torch.appo.runner import APPORunner

        return APPORuntime(runner_cls=APPORunner, play_fn=default_play_fn)

    from rsl_rl.utils import resolve_callable

    resolver = resolve_callable(str(runtime_resolver))
    runtime = resolver(rl_cfg)
    if runtime is None:
        raise ValueError(
            f"APPO runtime resolver {runtime_resolver!r} returned None for rl_cfg runtime selection."
        )

    runner_cls = getattr(runtime, "runner_cls", None)
    play_fn = getattr(runtime, "play_fn", None)
    if runner_cls is None or play_fn is None:
        raise TypeError(
            f"APPO runtime resolver {runtime_resolver!r} must return an object with "
            "'runner_cls' and 'play_fn' attributes."
        )
    return APPORuntime(runner_cls=runner_cls, play_fn=play_fn)
