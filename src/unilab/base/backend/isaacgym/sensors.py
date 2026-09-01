"""Compatibility surface for shared MJCF subprocess metadata helpers."""

from unilab.base.backend.subprocess_ipc import sensors as _sensors
from unilab.base.backend.subprocess_ipc.sensors import (
    KIND_CONTACT_FOUND,
    KIND_FRAMEPOS,
    KIND_FRAMEQUAT,
    KIND_FRAMEZAXIS,
    KIND_GYRO,
    KIND_LOCAL_LINVEL,
    SUPPORTED_KINDS,
    ActuatorSpec,
    SceneMetadata,
    SceneSensorSpec,
    SiteFrame,
    UnsupportedSensorSpec,
)


def scan_scene_metadata(model_file: str, *, backend_label: str = "isaacgym") -> SceneMetadata:
    """Preserve the historical IsaacGym diagnostic label for direct callers."""
    return _sensors.scan_scene_metadata(model_file, backend_label=backend_label)


__all__ = _sensors.__all__
