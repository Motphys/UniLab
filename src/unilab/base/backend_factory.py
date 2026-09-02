"""UniLab owner-layer assembly for UniSim physics backends.

This module owns only UniLab concerns: resolving hosted robot assets and
translating :class:`EnvCfg` backend options into the public ``unisim``
factory. Physics implementations and their public contract live in the
``unisim-core`` distribution.
"""

from __future__ import annotations

from types import MethodType
from typing import TYPE_CHECKING, Any

import numpy as np
import unisim
from unisim.backend.base import BackendRootStateLayout, SimBackend

from unilab.assets.hub import ensure_robot_assets_for_paths

if TYPE_CHECKING:
    from unilab.base.base import EnvCfg
    from unilab.base.scene import SceneCfg


def env_backend_kwargs(cfg: "EnvCfg") -> dict[str, Any]:
    """Translate ``EnvCfg`` backend knobs into UniSim adapter options."""
    return {
        "post_step_forward_sensor": cfg.post_step_forward_sensor,
        "motrix_max_iterations": cfg.motrix_max_iterations,
        "chunk_size": cfg.chunk_size,
        "adaptive_chunk_size": cfg.adaptive_chunk_size,
        "cpu_ids": cfg.cpu_ids,
        "bench_nsteps": cfg.sim_substeps,
        "mjwarp_nconmax": cfg.mjwarp_nconmax,
        "mjwarp_njmax": cfg.mjwarp_njmax,
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
    kwargs["body_state_required"] = body_state_required
    backend = unisim.create_backend(backend_type, scene, num_envs, sim_dt, **kwargs)
    # unisim-core 0.1.12 predates Drake's root-state layout contract.  Keep
    # Drake reset semantics working while that optional adapter is upgraded by
    # installing the contract on the concrete instance only.  This is a cold
    # path shim: Entity materialization calls it once and caches the result;
    # reset/step never inspect model metadata.
    if str(backend_type).lower() == "drake":
        _install_drake_root_state_layout(backend)
    return backend


def _install_drake_root_state_layout(backend: SimBackend) -> None:
    """Backfill Drake's floating-root layout for older unisim-core wheels."""
    if callable(getattr(type(backend), "get_root_state_layout", None)) and (
        type(backend).get_root_state_layout is not SimBackend.get_root_state_layout
    ):
        return

    def get_root_state_layout(self: Any, root_body_name: str) -> BackendRootStateLayout:
        runtime = getattr(self, "_runtime", None)
        if runtime is None or not hasattr(runtime, "model_info"):
            raise NotImplementedError(
                "DrakeBackend does not expose root-state layout metadata for "
                f"body {root_body_name!r}"
            )
        info = runtime.model_info()
        body_names = tuple(getattr(info, "joint_body_names", ()))
        joint_qpos_adr = np.asarray(getattr(info, "joint_qpos_adr", ()), dtype=np.intp)
        joint_qvel_adr = np.asarray(getattr(info, "joint_qvel_adr", ()), dtype=np.intp)
        qpos_dims = np.asarray(getattr(info, "joint_qpos_dim", ()), dtype=np.intp)
        qvel_dims = np.asarray(getattr(info, "joint_qvel_dim", ()), dtype=np.intp)
        matches = [
            index
            for index, body_name in enumerate(body_names)
            if str(body_name) == str(root_body_name)
            and index < len(qpos_dims)
            and index < len(qvel_dims)
            and int(qpos_dims[index]) == 7
            and int(qvel_dims[index]) == 6
        ]
        if len(matches) != 1:
            if not matches:
                raise NotImplementedError(
                    "backend 'drake' capability 'root-state layout' requires body "
                    f"'{root_body_name}' to own exactly one floating free joint"
                )
            raise NotImplementedError(
                "backend 'drake' capability 'root-state layout' requires body "
                f"'{root_body_name}' to own exactly one floating free joint; found {len(matches)}"
            )
        index = matches[0]
        qpos_start = int(joint_qpos_adr[index])
        qvel_start = int(joint_qvel_adr[index])
        return BackendRootStateLayout(
            qpos_indices=tuple(range(qpos_start, qpos_start + 7)),
            qvel_indices=tuple(range(qvel_start, qvel_start + 6)),
        )

    backend.get_root_state_layout = MethodType(get_root_state_layout, backend)  # type: ignore[attr-defined]


__all__ = ["SimBackend", "create_backend", "env_backend_kwargs"]
