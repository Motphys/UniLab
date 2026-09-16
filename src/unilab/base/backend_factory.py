"""UniLab owner-layer assembly for UniSim physics backends.

This module owns only UniLab concerns: resolving hosted robot assets and
translating :class:`EnvCfg` backend options into the public ``unisim``
factory. Physics implementations and their public contract live in the
``unisim-core`` distribution.
"""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING, Any

import unisim
from unisim.backend.base import SimBackend

from unilab.assets.hub import ensure_robot_assets_for_paths, resolve_superdex_robot_asset
from unilab.base.process_device import bind_genesis_process_device

if TYPE_CHECKING:
    from unilab.base.base import EnvCfg
    from unilab.base.scene import SceneCfg


def env_backend_kwargs(cfg: "EnvCfg") -> dict[str, Any]:
    """Translate ``EnvCfg`` backend knobs into UniSim adapter options."""
    result: dict[str, Any] = {
        "superdex_num_workers": cfg.superdex_num_workers,
        "superdex_assets_root": cfg.superdex_assets_root,
        "superdex_effort_limits": cfg.superdex_effort_limits,
        "superdex_allow_contact_approximation": cfg.superdex_allow_contact_approximation,
        "motrix_max_iterations": cfg.motrix_max_iterations,
        "cpu_ids": cfg.cpu_ids,
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
        "superdex_execution_mode": cfg.superdex_execution_mode,
    }
    # Forward the explicit Genesis device id only when a rank selected one;
    # when absent, unisim-core's factory default applies and Genesis picks
    # its own device.
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
    superdex_assets_root = kwargs.pop("superdex_assets_root", None)
    if backend_type == "superdex" and scene.model_file.endswith(".superdex_bot"):
        scene = replace(
            scene,
            model_file=resolve_superdex_robot_asset(
                scene.model_file, assets_root=superdex_assets_root
            ),
        )
    if backend_type != "superdex":
        kwargs.pop("superdex_num_workers", None)
        kwargs.pop("superdex_execution_mode", None)
        kwargs.pop("superdex_effort_limits", None)
        kwargs.pop("superdex_allow_contact_approximation", None)
    ensure_robot_assets_for_paths(
        [scene.model_file, scene.visual_model_file, *scene.fragment_files]
    )
    if backend_type == "drake":
        # unisim-core 1.4.2 dropped the Drake-branch filtering of MuJoCo
        # root-body options; Drake derives root state from its own plant and
        # DrakeBackend rejects the keywords.  Pop them at the owner boundary.
        kwargs.pop("base_name", None)
        kwargs.pop("push_body_name", None)
    # Newton reconstructs body state from its compiled articulation and does
    # not accept MuJoCo's synthetic body-sensor injection.  Keep this
    # capability translation at the owner/backend boundary so env code remains
    # backend-agnostic.
    kwargs["body_state_required"] = body_state_required and backend_type not in {
        "newton",
        "superdex",
    }
    if backend_type == "genesis" and kwargs.get("genesis_device_id") is not None:
        # Bind before any unisim-core Genesis constructor can call gs.init.
        # Binding a non-zero id pins CUDA_VISIBLE_DEVICES (Quadrants only
        # honors the first visible device), so forward the *post-pin*
        # in-process index.
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
    return unisim.create_backend(backend_type, scene, num_envs, sim_dt, **kwargs)


__all__ = ["SimBackend", "create_backend", "env_backend_kwargs"]
