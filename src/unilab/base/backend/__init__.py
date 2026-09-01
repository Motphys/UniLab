from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any, cast

from unilab.assets.hub import ensure_robot_assets_for_paths
from unilab.base.scene import SceneCfg

from .base import BackendRootStateLayout, BackendSensorView, RenderClosedError, SimBackend

if TYPE_CHECKING:
    from unilab.base.base import EnvCfg

    from .mujoco.xml import materialize_scene_visual_override


def env_backend_kwargs(cfg: "EnvCfg") -> dict:
    """Bundle EnvCfg-level backend tuning fields for create_backend(**...)."""
    return {
        "post_step_forward_sensor": cfg.post_step_forward_sensor,
        "motrix_max_iterations": cfg.motrix_max_iterations,
        "chunk_size": cfg.chunk_size,
        "adaptive_chunk_size": cfg.adaptive_chunk_size,
        "cpu_ids": cfg.cpu_ids,
        "bench_nsteps": cfg.sim_substeps,
        "mjwarp_nconmax": cfg.mjwarp_nconmax,
        "mjwarp_njmax": cfg.mjwarp_njmax,
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


_MUJOCO_XML_EXPORTS = frozenset(
    {
        "add_sensor",
        "create_discardvisual_xml",
        "get_named_bodies",
        "get_named_body_ids",
        "inject_mujoco_tracking_sensors",
        "materialize_mujoco_hfield_attached_scene",
        "materialize_scene_fragments",
        "materialize_scene_visual_override",
        "processed_xml",
    }
)
_MUJOCO_MOTION_EXPORT_EXPORTS = frozenset({"compute_tracking_fk"})
_MOTRIX_SCENE_EXPORTS = frozenset(
    {
        "add_motrix_tracking_frame_sensors",
        "materialize_motrix_hfield_attached_scene",
        "materialize_motrix_scene",
    }
)


def _load_mujoco_backend() -> Any:
    from .mujoco.backend import MuJoCoBackend

    return MuJoCoBackend


def _load_motrix_backend() -> tuple[Any, bool]:
    from .motrix.backend import MOTRIX_AVAILABLE, MotrixBackend

    return MotrixBackend, bool(MOTRIX_AVAILABLE)


def _load_mjwarp_backend() -> Any:
    """Load the independent optional mujoco-warp backend on demand."""
    from .mjwarp.backend import MjwarpBackend

    return MjwarpBackend


def _mjwarp_available() -> bool:
    from .mjwarp.dependencies import mjwarp_dependencies_available

    return mjwarp_dependencies_available()


def _load_motrix_scene_export(name: str) -> Any:
    from .motrix import scene

    return getattr(scene, name)


def _load_drake_backend() -> Any:
    from .drake.backend import DrakeBackend

    return DrakeBackend


def _drake_available() -> bool:
    from .drake.backend import ensure_drake_batch_available

    return ensure_drake_batch_available()[0]


def _load_isaacgym_backend() -> Any:
    from .isaacgym.backend import IsaacGymBackend

    return IsaacGymBackend


def _load_isaacsim_backend() -> Any:
    """Load IsaacSim lazily so importing UniLab never starts Kit."""
    from .isaacsim.backend import IsaacSimBackend

    return IsaacSimBackend


def create_backend(
    backend_type: str,
    scene: SceneCfg,
    num_envs: int,
    sim_dt: float,
    *,
    body_state_required: bool = False,
    **kwargs,
) -> SimBackend:
    """Create a simulation backend.

    Args:
        backend_type: ``"mujoco"``, ``"mjwarp"``, ``"motrix"``, ``"drake"``,
            ``"isaacgym"``, or ``"isaacsim"``.
        scene: SceneCfg for either static or composed scenes.
        num_envs: Number of environments.
        sim_dt: Simulation timestep.
        body_state_required: Whether the caller requires public body-state views.
            Backend adapters decide whether satisfying this request requires extra
            cold-path scene materialization.
        **kwargs: Additional backend options such as ``position_actuator_gains``,
            ``iterations``, or ``motrix_max_iterations``.

    Returns:
        SimBackend instance.
    """
    if scene is None:
        raise ValueError("SceneCfg must be provided")
    if not isinstance(body_state_required, bool):
        raise TypeError(
            "create_backend body_state_required must be bool, "
            f"got {type(body_state_required).__name__}"
        )

    # Cold path: fetch HF-hosted robot meshes/textures referenced by this
    # scene before any backend parses the XML (no-op when cached locally).
    ensure_robot_assets_for_paths(
        [scene.model_file, scene.visual_model_file, *scene.fragment_files]
    )

    position_actuator_gains = kwargs.pop("position_actuator_gains", None)
    motrix_max_iterations = kwargs.pop("motrix_max_iterations", None)
    post_step_forward_sensor = kwargs.pop("post_step_forward_sensor", None)
    iterations = kwargs.pop("iterations", None)
    chunk_size = kwargs.pop("chunk_size", None)
    adaptive_chunk_size = kwargs.pop("adaptive_chunk_size", False)
    cpu_ids = kwargs.pop("cpu_ids", None)
    bench_nsteps = kwargs.pop("bench_nsteps", 1)
    mjwarp_nconmax = kwargs.pop("mjwarp_nconmax", None)
    mjwarp_njmax = kwargs.pop("mjwarp_njmax", None)
    drake_backend_mode = kwargs.pop("drake_backend_mode", "batch")
    drake_nthread = kwargs.pop("drake_nthread", None)
    isaacgym_device_id = kwargs.pop("isaacgym_device_id", None)
    isaacgym_worker_timeout_s = kwargs.pop("isaacgym_worker_timeout_s", None)
    isaacsim_device_id = kwargs.pop("isaacsim_device_id", None)
    isaacsim_worker_timeout_s = kwargs.pop("isaacsim_worker_timeout_s", None)
    isaacsim_render_mode = kwargs.pop("isaacsim_render_mode", None)
    isaacsim_render_width = kwargs.pop("isaacsim_render_width", 1280)
    isaacsim_render_height = kwargs.pop("isaacsim_render_height", 720)
    if backend_type == "mujoco":
        MuJoCoBackend = _load_mujoco_backend()
        if body_state_required:
            kwargs["add_body_sensors"] = True
        if position_actuator_gains is not None:
            kwargs["position_actuator_gains"] = position_actuator_gains
        if post_step_forward_sensor is not None:
            kwargs["post_step_forward_sensor"] = post_step_forward_sensor
        kwargs["iterations"] = iterations
        kwargs["chunk_size"] = chunk_size
        kwargs["adaptive_chunk_size"] = adaptive_chunk_size
        kwargs["cpu_ids"] = cpu_ids
        kwargs["bench_nsteps"] = bench_nsteps
        return cast(SimBackend, MuJoCoBackend(scene, num_envs, sim_dt, **kwargs))
    if backend_type == "mjwarp":
        MjwarpBackend = _load_mjwarp_backend()
        if body_state_required:
            kwargs["add_body_sensors"] = True
        if position_actuator_gains is not None:
            raise ValueError(
                "mjwarp does not accept position_actuator_gains in the host compatibility "
                "profile; configure the model on the cold path instead."
            )
        ignored_non_defaults = {
            key: value
            for key, value, default in (
                ("post_step_forward_sensor", post_step_forward_sensor, None),
                ("iterations", iterations, None),
                ("chunk_size", chunk_size, None),
                ("adaptive_chunk_size", adaptive_chunk_size, False),
                ("cpu_ids", cpu_ids, None),
                ("bench_nsteps", bench_nsteps, 1),
            )
            if value != default
        }
        if ignored_non_defaults:
            rendered = ", ".join(f"{key}={value!r}" for key, value in ignored_non_defaults.items())
            warnings.warn(
                "mjwarp ignores non-default MuJoCo-only backend options: " + rendered,
                UserWarning,
                stacklevel=2,
            )
        # These generic EnvCfg fields are routed only to the MuJoCo pool.
        del post_step_forward_sensor, iterations, chunk_size, adaptive_chunk_size, cpu_ids
        del bench_nsteps
        kwargs["nconmax"] = mjwarp_nconmax
        kwargs["njmax"] = mjwarp_njmax
        return cast(SimBackend, MjwarpBackend(scene, num_envs, sim_dt, **kwargs))
    if backend_type == "motrix":
        MotrixBackend, motrix_available = _load_motrix_backend()
        if not motrix_available:
            raise ImportError("MotrixSim not available, install motrixsim package")
        if body_state_required:
            kwargs["add_body_sensors"] = True
        if motrix_max_iterations is not None:
            kwargs["max_iterations"] = motrix_max_iterations
        return cast(SimBackend, MotrixBackend(scene, num_envs, sim_dt, **kwargs))
    if backend_type == "drake":
        DrakeBackend = _load_drake_backend()
        # DrakeUni is a generic batch engine. Task-level body names and scalar
        # gain overrides are consumed by other backends, but Drake reads bodies,
        # actuators, and sensors from the model contract itself.
        kwargs.pop("base_name", None)
        kwargs.pop("push_body_name", None)
        kwargs.pop("add_body_sensors", None)
        kwargs["drake_backend_mode"] = drake_backend_mode
        if drake_nthread is not None:
            kwargs["nthread"] = drake_nthread
        return cast(SimBackend, DrakeBackend(scene, num_envs, sim_dt, **kwargs))
    if backend_type == "isaacgym":
        IsaacGymBackend = _load_isaacgym_backend()
        # Body states are always available from the rigid-body state tensor;
        # the flag exists for backends that need extra scene materialization.
        kwargs.pop("add_body_sensors", None)
        if position_actuator_gains is not None:
            raise ValueError(
                "isaacgym runs torque-mode dofs only; position_actuator_gains has no "
                "IsaacGym equivalent in the subprocess profile."
            )
        ignored_non_defaults = {
            key: value
            for key, value, default in (
                ("post_step_forward_sensor", post_step_forward_sensor, None),
                ("iterations", iterations, None),
                ("chunk_size", chunk_size, None),
                ("adaptive_chunk_size", adaptive_chunk_size, False),
                ("cpu_ids", cpu_ids, None),
                ("bench_nsteps", bench_nsteps, 1),
            )
            if value != default
        }
        if ignored_non_defaults:
            rendered = ", ".join(f"{key}={value!r}" for key, value in ignored_non_defaults.items())
            warnings.warn(
                "isaacgym ignores non-default MuJoCo-only backend options: " + rendered,
                UserWarning,
                stacklevel=2,
            )
        if isaacgym_device_id is not None:
            kwargs["device_id"] = isaacgym_device_id
        if isaacgym_worker_timeout_s is not None:
            kwargs["worker_timeout_s"] = isaacgym_worker_timeout_s
        return cast(SimBackend, IsaacGymBackend(scene, num_envs, sim_dt, **kwargs))
    if backend_type == "isaacsim":
        IsaacSimBackend = _load_isaacsim_backend()
        # Accept the backend-native spellings for direct factory callers while
        # keeping EnvCfg's explicit ``isaacsim_*`` fields canonical.
        direct_render_width = kwargs.pop("render_width", None)
        direct_render_height = kwargs.pop("render_height", None)
        # IsaacLab's implicit actuator path owns the gains in the worker.  A
        # host-side scalar override would silently diverge from the MJCF
        # contract, so reject it at the factory boundary.
        kwargs.pop("add_body_sensors", None)
        if position_actuator_gains is not None:
            raise ValueError(
                "isaacsim uses IsaacLab implicit position actuators; configure gains in the "
                "scene/owner contract rather than position_actuator_gains."
            )
        ignored_non_defaults = {
            key: value
            for key, value, default in (
                ("post_step_forward_sensor", post_step_forward_sensor, None),
                ("iterations", iterations, None),
                ("chunk_size", chunk_size, None),
                ("adaptive_chunk_size", adaptive_chunk_size, False),
                ("cpu_ids", cpu_ids, None),
                ("bench_nsteps", bench_nsteps, 1),
            )
            if value != default
        }
        if ignored_non_defaults:
            rendered = ", ".join(f"{key}={value!r}" for key, value in ignored_non_defaults.items())
            warnings.warn(
                "isaacsim ignores non-default MuJoCo-only backend options: " + rendered,
                UserWarning,
                stacklevel=2,
            )
        if isaacsim_device_id is not None:
            kwargs["device_id"] = isaacsim_device_id
        if isaacsim_worker_timeout_s is not None:
            kwargs["worker_timeout_s"] = isaacsim_worker_timeout_s
        if isaacsim_render_mode is not None:
            kwargs["render_mode"] = isaacsim_render_mode
        kwargs["render_width"] = (
            isaacsim_render_width if direct_render_width is None else direct_render_width
        )
        kwargs["render_height"] = (
            isaacsim_render_height if direct_render_height is None else direct_render_height
        )
        return cast(SimBackend, IsaacSimBackend(scene, num_envs, sim_dt, **kwargs))
    raise ValueError(f"Unknown backend: {backend_type}")


def __getattr__(name: str):
    if name == "MuJoCoBackend":
        return _load_mujoco_backend()
    if name == "MotrixBackend":
        return _load_motrix_backend()[0]
    if name == "MOTRIX_AVAILABLE":
        return _load_motrix_backend()[1]
    if name == "MjwarpBackend":
        return _load_mjwarp_backend()
    if name == "MJWARP_AVAILABLE":
        return _mjwarp_available()
    if name == "DrakeBackend":
        return _load_drake_backend()
    if name == "DRAKE_AVAILABLE":
        return _drake_available()
    if name == "IsaacGymBackend":
        return _load_isaacgym_backend()
    if name == "IsaacSimBackend":
        return _load_isaacsim_backend()
    if name in _MUJOCO_XML_EXPORTS:
        from .mujoco import xml

        return getattr(xml, name)
    if name in _MUJOCO_MOTION_EXPORT_EXPORTS:
        from .mujoco import motion_export

        return getattr(motion_export, name)
    if name in _MOTRIX_SCENE_EXPORTS:
        return _load_motrix_scene_export(name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "SimBackend",
    "BackendSensorView",
    "RenderClosedError",
    "MuJoCoBackend",
    "MjwarpBackend",
    "MotrixBackend",
    "DrakeBackend",
    "IsaacGymBackend",
    "IsaacSimBackend",
    "DRAKE_AVAILABLE",
    "MJWARP_AVAILABLE",
    "add_sensor",
    "compute_tracking_fk",
    "create_discardvisual_xml",
    "create_backend",
    "get_named_bodies",
    "get_named_body_ids",
    "inject_mujoco_tracking_sensors",
    "add_motrix_tracking_frame_sensors",
    "materialize_motrix_hfield_attached_scene",
    "materialize_motrix_scene",
    "materialize_mujoco_hfield_attached_scene",
    "materialize_scene_fragments",
    "materialize_scene_visual_override",
    "processed_xml",
]
