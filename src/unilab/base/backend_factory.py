"""UniLab owner-layer assembly for UniSim physics backends.

This module owns only UniLab concerns: resolving hosted robot assets and
translating :class:`EnvCfg` backend options into the public ``unisim``
factory. Physics implementations and their public contract live in the
``unisim-core`` distribution.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import unisim
from unisim.backend.base import SimBackend

from unilab.assets.hub import ensure_robot_assets_for_paths
from unilab.base.process_device import bind_genesis_process_device

if TYPE_CHECKING:
    from unilab.base.base import EnvCfg
    from unilab.base.scene import SceneCfg


def _legacy_genesis_device_option_error(exc: TypeError) -> bool:
    """Identify an old UniSim adapter rejecting the optional device keyword.

    UniSim 1.1 reports unknown backend options from ``GenesisBackend`` while
    other compatible releases may expose Python's usual ``unexpected keyword``
    wording.  Keep the compatibility retry narrowly scoped to those messages;
    constructor errors from the actual Genesis runtime must still propagate.
    """

    message = str(exc).lower()
    mentions_device = "genesis_device_id" in message or "device_id" in message
    rejects_keyword = (
        "does not accept backend options" in message
        or "unexpected keyword argument" in message
        or "unexpected keyword" in message
    )
    return mentions_device and rejects_keyword


def env_backend_kwargs(cfg: "EnvCfg") -> dict[str, Any]:
    """Translate ``EnvCfg`` backend knobs into UniSim adapter options."""
    result: dict[str, Any] = {
        "post_step_forward_sensor": cfg.post_step_forward_sensor,
        "motrix_max_iterations": cfg.motrix_max_iterations,
        "chunk_size": cfg.chunk_size,
        "adaptive_chunk_size": cfg.adaptive_chunk_size,
        "cpu_ids": cfg.cpu_ids,
        "bench_nsteps": cfg.sim_substeps,
        "mjwarp_nconmax": cfg.mjwarp_nconmax,
        "mjwarp_njmax": cfg.mjwarp_njmax,
        "newton_device": cfg.newton_device,
        "newton_nconmax": cfg.newton_nconmax,
        "newton_njmax": cfg.newton_njmax,
        "newton_capacity_check_steps": cfg.newton_capacity_check_steps,
        "genesis_integrator": cfg.genesis_integrator,
        "genesis_constraint_solver": cfg.genesis_constraint_solver,
        "genesis_friction_cone": cfg.genesis_friction_cone,
        "genesis_solver_iterations": cfg.genesis_solver_iterations,
        "drake_backend_mode": cfg.drake_backend_mode,
        "drake_nthread": cfg.drake_nthread,
        "isaacgym_device_id": cfg.isaacgym_device_id,
        "isaacgym_worker_timeout_s": cfg.isaacgym_worker_timeout_s,
        "isaacsim_device_id": cfg.isaacsim_device_id,
        "isaacsim_worker_timeout_s": cfg.isaacsim_worker_timeout_s,
        "isaacsim_render_mode": cfg.isaacsim_render_mode,
        "isaacsim_render_width": cfg.isaacsim_render_width,
        "isaacsim_render_height": cfg.isaacsim_render_height,
    }
    # Keep the optional key absent for legacy unisim-core releases that do not
    # know about Genesis' explicit device argument.  Once a rank selects a
    # device the key is added below and ``create_backend`` supplies a narrow
    # compatibility fallback for those releases.
    if cfg.genesis_device_id is not None:
        result["genesis_device_id"] = cfg.genesis_device_id
    return result


def create_backend(
    backend_type: str,
    scene: "SceneCfg",
    num_envs: int,
    sim_dt: float,
    *,
    body_state_required: bool = False,
    **kwargs: Any,
) -> SimBackend:
    """Prepare UniLab-owned assets and construct a UniSim backend."""
    if scene is None:
        raise ValueError("SceneCfg must be provided")
    ensure_robot_assets_for_paths(
        [scene.model_file, scene.visual_model_file, *scene.fragment_files]
    )
    if backend_type != "newton":
        # Keep the owner translation forward-compatible with unisim-core
        # releases that predate the Newton adapter and therefore do not pop
        # these optional kwargs in their shared factory.
        for key in (
            "newton_device",
            "newton_nconmax",
            "newton_njmax",
            "newton_capacity_check_steps",
        ):
            kwargs.pop(key, None)
    # Newton reconstructs body state from its compiled articulation and does
    # not accept MuJoCo's synthetic body-sensor injection.  Keep this
    # capability translation at the owner/backend boundary so env code remains
    # backend-agnostic.
    kwargs["body_state_required"] = body_state_required and backend_type != "newton"
    if backend_type == "genesis" and kwargs.get("genesis_device_id") is not None:
        # Bind before any unisim-core Genesis constructor can call gs.init.
        # New unisim-core releases repeat this idempotently; old releases do
        # not accept the keyword, so the retry below still gets the correct
        # process-wide device.  Binding a non-zero id pins
        # CUDA_VISIBLE_DEVICES (Quadrants only honors the first visible
        # device), so forward the *post-pin* in-process index.
        genesis_device_id = kwargs["genesis_device_id"]
        if (
            isinstance(genesis_device_id, bool)
            or not isinstance(genesis_device_id, int)
            or genesis_device_id < 0
        ):
            raise ValueError(
                "genesis_device_id must be a non-negative integer or None, "
                f"got {genesis_device_id!r}"
            )
        bound = bind_genesis_process_device(f"cuda:{genesis_device_id}")
        kwargs["genesis_device_id"] = int(bound.rsplit(":", 1)[1])
    try:
        return unisim.create_backend(backend_type, scene, num_envs, sim_dt, **kwargs)
    except TypeError as exc:
        if backend_type != "genesis" or "genesis_device_id" not in kwargs:
            raise
        # unisim-core < 1.2 has no Genesis device field and reports the
        # unknown option from GenesisBackend.  Retry only for that precise
        # capability error; unrelated constructor TypeErrors must propagate.
        if not _legacy_genesis_device_option_error(exc):
            raise
        legacy_kwargs = dict(kwargs)
        legacy_kwargs.pop("genesis_device_id", None)
        return unisim.create_backend(backend_type, scene, num_envs, sim_dt, **legacy_kwargs)


__all__ = ["SimBackend", "create_backend", "env_backend_kwargs"]
