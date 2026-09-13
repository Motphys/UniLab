"""Backend-owned domain-randomization plan types re-exported for tasks.

The legacy UniLab provider/manager protocol was removed. Manager-Based tasks
declare reset and interval behavior through Hydra event terms and submit curated
UniSim plans; they do not implement a second reset protocol.
"""

from unisim.dr.interval import (
    INTERVAL_TERM_BODY_ANGULAR_VELOCITY_DELTA,
    INTERVAL_TERM_BODY_FORCE,
    INTERVAL_TERM_BODY_LINEAR_VELOCITY_DELTA,
    INTERVAL_TERM_BODY_TORQUE,
    INTERVAL_TERM_PUSH,
    IntervalTermOp,
)
from unisim.dr.types import (
    DomainRandomizationCapabilities,
    GeomSizeOverride,
    InitRandomizationPlan,
    IntervalRandomizationPlan,
    ModelVariantSpec,
    ResetPlan,
    ResetRandomizationPayload,
)

__all__ = [
    "INTERVAL_TERM_BODY_ANGULAR_VELOCITY_DELTA",
    "INTERVAL_TERM_BODY_FORCE",
    "INTERVAL_TERM_BODY_LINEAR_VELOCITY_DELTA",
    "INTERVAL_TERM_BODY_TORQUE",
    "INTERVAL_TERM_PUSH",
    "DomainRandomizationCapabilities",
    "GeomSizeOverride",
    "InitRandomizationPlan",
    "IntervalRandomizationPlan",
    "IntervalTermOp",
    "ModelVariantSpec",
    "ResetPlan",
    "ResetRandomizationPayload",
]
