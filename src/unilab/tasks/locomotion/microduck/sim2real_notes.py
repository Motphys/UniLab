"""Known deployment gaps for the MicroDuck Manager-Based task."""

from __future__ import annotations

from .deploy_contract import MICRODUCK_ACTOR_OBS_DIM, MICRODUCK_OBS_SEGMENTS

POLLEN_RUNTIME_OBS_DIM = MICRODUCK_ACTOR_OBS_DIM

# Known gaps against pollen-robotics/microduck_rl:
# 1. BAM XL330 M6 voltage-control actuator; UniLab uses XML position PD.
# 2. Optional backlash hinges are not represented in the current model.
# 3. Observation delay/noise envelopes still need hardware calibration.


def obs_layout_for_export() -> list[tuple[str, int]]:
    """Return the stable actor observation layout used by deploy tooling."""
    return list(MICRODUCK_OBS_SEGMENTS)
