"""UniLab owner-layer assembly for UniSim physics backends.

This module owns only UniLab concerns: resolving hosted robot assets and
translating :class:`EnvCfg` backend options into the public ``unisim``
factory. Physics implementations and their public contract live in the
``unisim-core`` distribution.
"""

from __future__ import annotations

import inspect
import sys
from dataclasses import replace
from typing import TYPE_CHECKING, Any

import unisim
from unisim.backend.base import SimBackend, TensorExecution, validate_tensor_device
from unisim.support import get_tensor_platform_profiles

from unilab.assets.hub import ensure_robot_assets_for_paths, resolve_superdex_robot_asset
from unilab.base.process_device import bind_genesis_process_device

if TYPE_CHECKING:
    from unilab.base.base import EnvCfg
    from unilab.base.scene import SceneCfg


_CUDA_BACKEND_INT_DEVICE_FIELDS = {
    "genesis": "genesis_device_id",
    "isaacgym": "isaacgym_device_id",
    "isaacsim": "isaacsim_device_id",
}
_EXTERNAL_WORKER_DEVICE_FIELDS = {
    "isaacgym": "isaacgym_device_id",
    "isaacsim": "isaacsim_device_id",
}


def _requested_cuda_only_device(backend_type: str, kwargs: dict[str, Any]) -> str:
    if backend_type == "mjwarp":
        return "cuda"
    if backend_type == "newton":
        device = kwargs.get("newton_device")
        return "cuda" if device is None else str(device)

    field = _CUDA_BACKEND_INT_DEVICE_FIELDS[backend_type]
    device_id = kwargs.get(field)
    if device_id is None:
        # Genesis follows its current/default process device when unset.  The
        # Isaac workers instead default their payload to ordinal zero, so an
        # omitted owner field remains an explicit zero request.
        return "cuda:0" if backend_type in _EXTERNAL_WORKER_DEVICE_FIELDS else "cuda"
    if isinstance(device_id, bool) or not isinstance(device_id, int) or device_id < 0:
        raise ValueError(f"{field} must be a non-negative integer or None, got {device_id!r}")
    return f"cuda:{device_id}"


def _torch_cuda_runtime_state() -> tuple[dict[str, Any], int | None]:
    try:
        import torch

        available = bool(torch.cuda.is_available())
        count = int(torch.cuda.device_count()) if available else 0
        current = int(torch.cuda.current_device()) if available else None
        state = {
            "available": available,
            "visible_count": count,
            "current": current,
            "torch": torch.__version__,
            "torch_hip": torch.version.hip,
            "platform": sys.platform,
        }
        return state, current
    except Exception as exc:
        return {
            "available": False,
            "visible_count": 0,
            "current": None,
            "torch": f"import failed: {exc}",
            "torch_hip": "unknown",
            "platform": sys.platform,
        }, None


def _fail_closed_cuda_only(
    backend_type: str,
    requested_device: str,
    runtime_state: dict[str, Any],
    reason: str,
    next_step: str,
) -> RuntimeError:
    return RuntimeError(
        f"{backend_type} is a CUDA-only tensor backend and this request is unsupported: "
        f"{reason}. requested_device={requested_device!r}, "
        f"cuda_available={runtime_state['available']}, "
        f"visible_cuda_devices={runtime_state['visible_count']}, "
        f"current_cuda_device={runtime_state['current']}, "
        f"torch={runtime_state['torch']}, torch_hip={runtime_state['torch_hip']}, "
        f"platform={runtime_state['platform']}. {next_step}"
    )


def _validate_cuda_only_backend_platform(backend_type: str, kwargs: dict[str, Any]) -> None:
    """Fail closed before cold-path work on a non-CUDA platform or device.

    UniSim's SDK-free public inventory owns the long-term platform matrix.  This
    final owner-layer choke point checks only the static boundary, so optional
    SDK discovery and task capability negotiation remain in UniSim.
    """

    profile = get_tensor_platform_profiles().get(backend_type)
    if profile is None or profile.execution is not TensorExecution.DEVICE_RESIDENT:
        return

    try:
        requested_device = _requested_cuda_only_device(backend_type, kwargs)
    except ValueError as exc:
        runtime_state, _ = _torch_cuda_runtime_state()
        raise _fail_closed_cuda_only(
            backend_type,
            "<invalid>",
            runtime_state,
            str(exc),
            "Use a non-negative integer backend device id.",
        ) from exc

    runtime_state, current_device = _torch_cuda_runtime_state()
    try:
        validate_tensor_device(
            profile.torch_devices,
            requested_device,
            current_device=current_device,
            label=f"{backend_type} tensor runtime",
        )
    except ValueError as exc:
        raise _fail_closed_cuda_only(
            backend_type,
            requested_device,
            runtime_state,
            str(exc),
            "Use a Linux CUDA process; CPU, MPS, ROCm, and hidden fallbacks are unsupported.",
        ) from exc

    if sys.platform == "darwin":
        raise _fail_closed_cuda_only(
            backend_type,
            requested_device,
            runtime_state,
            "macOS has no supported CUDA runtime",
            "Use a CPU-authoritative host-bridge backend or a Linux CUDA runtime.",
        )
    if runtime_state["torch_hip"] not in (None, "None", ""):
        raise _fail_closed_cuda_only(
            backend_type,
            requested_device,
            runtime_state,
            "this Torch build reports ROCm/HIP rather than CUDA",
            "Install a Linux CUDA Torch build or use a CPU-authoritative host-bridge backend.",
        )
    if not runtime_state["available"]:
        raise _fail_closed_cuda_only(
            backend_type,
            requested_device,
            runtime_state,
            "Torch CUDA is unavailable",
            "Check driver/runtime installation and CUDA_VISIBLE_DEVICES before construction.",
        )

    requested_parts = requested_device.lower().split(":", 1)
    explicit_index = int(requested_parts[1]) if len(requested_parts) == 2 else None
    requested_index = current_device if explicit_index is None else explicit_index
    if requested_index is None or requested_index >= int(runtime_state["visible_count"]):
        raise _fail_closed_cuda_only(
            backend_type,
            requested_device,
            runtime_state,
            "the requested CUDA ordinal is out of range",
            "Use an index below torch.cuda.device_count() after CUDA_VISIBLE_DEVICES remapping.",
        )

    field = _EXTERNAL_WORKER_DEVICE_FIELDS.get(backend_type)
    if field is not None:
        payload_id = kwargs.get(field)
        payload_index = 0 if payload_id is None else int(payload_id)
        if payload_index != current_device:
            raise _fail_closed_cuda_only(
                backend_type,
                requested_device,
                runtime_state,
                (
                    "the external worker payload device does not match the learner's "
                    f"current Torch CUDA device ({payload_index} != {current_device})"
                ),
                "Bind the rank learner and its Isaac worker to the same CUDA ordinal "
                "before construction.",
            )
        import torch

        torch.cuda.set_device(payload_index)


def _validate_isaacsim_tensor_cuda_ipc_runtime() -> None:
    """Fail closed before constructing an IsaacSim release without CUDA IPC.

    ``isaacsim_tensor_cuda_ipc`` is an M9 candidate contract.  A production
    owner must not reach a released unisim-core that silently ignores or does
    not understand the constructor option.  The public adapter constructor is
    the compatibility boundary; no backend-private implementation state is
    inspected.
    """
    from unisim.backend.isaacsim import IsaacSimBackend

    parameter = inspect.signature(IsaacSimBackend.__init__).parameters.get("tensor_cuda_ipc")
    if parameter is None or parameter.kind is inspect.Parameter.VAR_KEYWORD:
        raise RuntimeError(
            "isaacsim_tensor_cuda_ipc requires a unisim-core IsaacSim backend with the "
            "tensor_cuda_ipc constructor contract; the installed unisim-core IsaacSim "
            "backend does not provide it"
        )


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
        "isaacsim_solver_position_iteration_count": cfg.isaacsim_solver_position_iteration_count,
        "isaacsim_solver_velocity_iteration_count": cfg.isaacsim_solver_velocity_iteration_count,
        "isaacsim_bounce_threshold_velocity": cfg.isaacsim_bounce_threshold_velocity,
        "isaacsim_contact_offset": cfg.isaacsim_contact_offset,
        "isaacsim_rest_offset": cfg.isaacsim_rest_offset,
        "isaacsim_max_depenetration_velocity": cfg.isaacsim_max_depenetration_velocity,
        "superdex_execution_mode": cfg.superdex_execution_mode,
    }
    # Forward the GPU buffer capacities only when set: unisim-core releases
    # before unilabsim/unisim#292 do not pop these keys, and forwarding None
    # would leak them into other backends' constructors; setting them against
    # an older unisim-core fails closed at the factory.
    if cfg.isaacsim_gpu_max_rigid_contact_count is not None:
        result["isaacsim_gpu_max_rigid_contact_count"] = cfg.isaacsim_gpu_max_rigid_contact_count
    if cfg.isaacsim_gpu_max_rigid_patch_count is not None:
        result["isaacsim_gpu_max_rigid_patch_count"] = cfg.isaacsim_gpu_max_rigid_patch_count
    # Forward the tensor opt-in only when enabled so releases without the M9
    # constructor never see an unknown keyword on the legacy default path.
    if cfg.isaacsim_tensor_cuda_ipc:
        result["isaacsim_tensor_cuda_ipc"] = True
    if cfg.isaacsim_share_friction_materials:
        result["share_friction_materials"] = True
    # Forward the explicit Genesis device id only when a rank selected one;
    # when absent, unisim-core's factory default applies and Genesis picks
    # its own device.
    if cfg.genesis_device_id is not None:
        result["genesis_device_id"] = cfg.genesis_device_id
    result["newton_use_cuda_graph"] = cfg.newton_use_cuda_graph
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
    _validate_cuda_only_backend_platform(backend_type, kwargs)
    if backend_type == "isaacsim" and kwargs.get("isaacsim_tensor_cuda_ipc", False):
        _validate_isaacsim_tensor_cuda_ipc_runtime()
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
    paths = [scene.model_file, scene.visual_model_file, *scene.fragment_files]
    paths.extend(
        entity.source.model_file for entity in scene.entity_assets if entity.source is not None
    )
    if scene.entity_variant is not None:
        paths.extend(source.model_file for source in scene.entity_variant.plan.variants)
    ensure_robot_assets_for_paths(list(dict.fromkeys(paths)))
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
