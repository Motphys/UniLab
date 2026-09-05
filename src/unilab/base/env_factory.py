"""Picklable ``EnvFactory`` adapters bridging uni_rl's injected env contract
to the UniLab env registry.

uni_rl runners and collectors never construct environments themselves; the
caller injects an ``EnvFactory`` (see ``uni_rl.env_contract``). Because
collectors run in ``multiprocessing`` spawn subprocesses, the factory must be
picklable by reference — the adapters below are module-level functions bound
with ``functools.partial`` (never closures or lambdas).
"""

from __future__ import annotations

from collections.abc import Mapping
from functools import partial
from typing import Any, cast

from uni_rl.env_contract import EnvFactory, EnvProtocol

from unilab.base.process_device import bind_genesis_process_device


def make_registry_env(
    task_name: str,
    sim_backend: str,
    num_envs: int,
    env_cfg_override: Mapping[str, Any] | None = None,
) -> EnvProtocol:
    """``EnvFactory`` implementation backed by the UniLab env registry.

    Top-level on purpose: ``functools.partial`` bindings of this function stay
    picklable for spawn-based collector subprocesses. ``ensure_registries``
    runs here because a spawned subprocess is a fresh interpreter that never
    executed the parent process's registry bootstrap.
    """
    from unilab.base import registry
    from unilab.base.registry import ensure_registries

    ensure_registries()
    # Genesis owns a process-wide session and reads ``torch.cuda.current_device``
    # before/while ``gs.init`` runs.  Off-policy/APPO collectors are fresh
    # spawn processes, so the parent-side binding cannot reach them; carry the
    # explicit cold-path id in the opaque override and bind immediately before
    # registry construction.  Newer unisim-core versions repeat this check in
    # GenesisBackend itself, making this compatibility guard idempotent.
    if sim_backend == "genesis" and env_cfg_override is not None:
        genesis_device_id = env_cfg_override.get("genesis_device_id")
        if genesis_device_id is not None:
            if (
                isinstance(genesis_device_id, bool)
                or not isinstance(genesis_device_id, int)
                or genesis_device_id < 0
            ):
                raise ValueError(
                    "genesis_device_id must be a non-negative integer or None, "
                    f"got {genesis_device_id!r}"
                )
            bind_genesis_process_device(f"cuda:{genesis_device_id}")
    # ABEnv satisfies EnvProtocol at runtime (reset/set_nan_guard live on
    # NpEnv); the declared ABEnv type predates the uni_rl protocol.
    return cast(
        "EnvProtocol",
        registry.make(
            task_name,
            sim_backend=sim_backend,
            num_envs=num_envs,
            env_cfg_override=dict(env_cfg_override) if env_cfg_override is not None else None,
        ),
    )


def registry_env_factory(task_name: str, sim_backend: str) -> EnvFactory:
    """Bind a registry task/backend pair into a picklable ``EnvFactory``."""
    return partial(make_registry_env, task_name, sim_backend)


__all__ = ["make_registry_env", "registry_env_factory"]
