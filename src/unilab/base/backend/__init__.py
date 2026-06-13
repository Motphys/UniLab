import os
import sys
from typing import Any, cast

from unilab.base.scene import SceneCfg

from .base import SimBackend

_MUJOCO_XML_EXPORTS = frozenset(
    {
        "add_sensor",
        "create_discardvisual_xml",
        "get_named_body_ids",
        "inject_mujoco_tracking_sensors",
        "materialize_mujoco_hfield_attached_scene",
        "materialize_scene_fragments",
        "materialize_scene_visual_override",
        "processed_xml",
    }
)
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


def _load_motrix_scene_export(name: str) -> Any:
    from .motrix import scene

    return getattr(scene, name)


def _load_drake_backend() -> tuple[Any, bool]:
    from .drake.backend import DRAKE_AVAILABLE, DrakeBackend

    return DrakeBackend, bool(DRAKE_AVAILABLE)


def _load_native_drake_backend() -> tuple[Any, bool, ImportError | None]:
    from .drake.backend_native import (
        NATIVE_DRAKE_AVAILABLE,
        NATIVE_DRAKE_IMPORT_ERROR,
        NativeDrakeBackend,
    )

    return NativeDrakeBackend, bool(NATIVE_DRAKE_AVAILABLE), NATIVE_DRAKE_IMPORT_ERROR


def _pydrake_loaded() -> bool:
    return any(name == "pydrake" or name.startswith("pydrake.") for name in sys.modules)


def create_backend(
    backend_type: str,
    scene: SceneCfg,
    num_envs: int,
    sim_dt: float,
    **kwargs,
) -> SimBackend:
    """Create a simulation backend.

    Args:
        backend_type: ``"mujoco"`` or ``"motrix"``.
        scene: SceneCfg for either static or composed scenes.
        num_envs: Number of environments.
        sim_dt: Simulation timestep.
        **kwargs: Additional backend options such as ``position_actuator_gains``
            or ``motrix_max_iterations``.

    Returns:
        SimBackend instance.
    """
    if scene is None:
        raise ValueError("SceneCfg must be provided")

    position_actuator_gains = kwargs.pop("position_actuator_gains", None)
    motrix_max_iterations = kwargs.pop("motrix_max_iterations", None)
    post_step_forward_sensor = kwargs.pop("post_step_forward_sensor", None)
    drake_backend_mode = kwargs.pop("drake_backend_mode", "auto")
    drake_nthread = kwargs.pop("drake_nthread", None)
    if backend_type == "mujoco":
        MuJoCoBackend = _load_mujoco_backend()
        if position_actuator_gains is not None:
            kwargs["position_actuator_gains"] = position_actuator_gains
        if post_step_forward_sensor is not None:
            kwargs["post_step_forward_sensor"] = post_step_forward_sensor
        return cast(SimBackend, MuJoCoBackend(scene, num_envs, sim_dt, **kwargs))
    if backend_type == "motrix":
        MotrixBackend, motrix_available = _load_motrix_backend()
        if not motrix_available:
            raise ImportError("MotrixSim not available, install motrixsim package")
        if motrix_max_iterations is not None:
            kwargs["max_iterations"] = motrix_max_iterations
        return cast(SimBackend, MotrixBackend(scene, num_envs, sim_dt, **kwargs))
    if backend_type == "drake":
        mode = str(drake_backend_mode or "auto").strip().lower()
        if mode == "auto":
            mode = os.environ.get("UNILAB_DRAKE_BACKEND", "pydrake").strip().lower()
        if mode in {"native", "native_pool", "drakeuni"}:
            if _pydrake_loaded():
                raise ImportError(
                    "Native Drake backend cannot be loaded after pydrake has already "
                    "been imported in this process. Start a fresh process for "
                    "drake_backend_mode='native', or select drake_backend_mode='pydrake'."
                )
            DrakeBackend, drake_available, import_error = _load_native_drake_backend()
        elif mode in {"pydrake", "python"}:
            DrakeBackend, drake_available = _load_drake_backend()
            import_error = None
        else:
            raise ValueError(
                "drake_backend_mode must be one of auto, pydrake, python, "
                f"native, native_pool, drakeuni; got {drake_backend_mode!r}"
            )
        if not drake_available:
            message = f"Drake backend mode {mode!r} is not available"
            if import_error is not None:
                message = f"{message}: {import_error}"
            raise ImportError(message) from import_error
        if position_actuator_gains is not None:
            kwargs["position_actuator_gains"] = position_actuator_gains
        if mode in {"native", "native_pool", "drakeuni"} and drake_nthread is not None:
            kwargs["nthread"] = drake_nthread
        return cast(SimBackend, DrakeBackend(scene, num_envs, sim_dt, **kwargs))
    raise ValueError(f"Unknown backend: {backend_type}")


def __getattr__(name: str):
    if name == "MuJoCoBackend":
        return _load_mujoco_backend()
    if name == "MotrixBackend":
        return _load_motrix_backend()[0]
    if name == "MOTRIX_AVAILABLE":
        return _load_motrix_backend()[1]
    if name == "DrakeBackend":
        return _load_drake_backend()[0]
    if name == "DRAKE_AVAILABLE":
        return _load_drake_backend()[1]
    if name in _MUJOCO_XML_EXPORTS:
        from .mujoco import xml

        return getattr(xml, name)
    if name in _MOTRIX_SCENE_EXPORTS:
        return _load_motrix_scene_export(name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "SimBackend",
    "MuJoCoBackend",
    "MotrixBackend",
    "DrakeBackend",
    "DRAKE_AVAILABLE",
    "add_sensor",
    "create_discardvisual_xml",
    "create_backend",
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
