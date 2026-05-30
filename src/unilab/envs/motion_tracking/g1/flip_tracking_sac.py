"""SAC variants of the G1 flip / wall-flip tracking tasks.

These pair the flip-specialized configs (scene / motion / zeroed reset
randomization, from :mod:`flip_tracking`) with the asymmetric actor-critic
observation layout of :class:`G1MotionTrackingSACEnv` (``base_lin_vel``
appended to the critic obs).

The flip behavior lives entirely in the cfg dataclasses and the SAC obs
augmentation lives entirely in the env class, so the two compose by simple
inheritance — no obs override is duplicated here. Registered under separate
names so the PPO flip pipeline (``G1FlipTracking`` / ``G1WallFlipTracking``)
is untouched, and sit at the same level as ``G1WBTObs``.
"""

from __future__ import annotations

from dataclasses import dataclass

from unilab.base import registry

from .flip_tracking import G1FlipTrackingCfg, G1WallFlipTrackingCfg
from .tracking_sac import G1MotionTrackingSACEnv


@registry.envcfg("G1FlipTrackingSAC")
@dataclass
class G1FlipTrackingSACCfg(G1FlipTrackingCfg):
    """Flip tracking cfg for SAC (identical fields, separate registry entry)."""


@registry.env("G1FlipTrackingSAC", sim_backend="mujoco")
@registry.env("G1FlipTrackingSAC", sim_backend="motrix")
class G1FlipTrackingSACEnv(G1MotionTrackingSACEnv):
    """Flip tracking env with the SAC asymmetric actor-critic obs layout.

    The motrix backend is registered for sim2sim eval/playback only.
    """

    _cfg: G1FlipTrackingSACCfg


@registry.envcfg("G1WallFlipTrackingSAC")
@dataclass
class G1WallFlipTrackingSACCfg(G1WallFlipTrackingCfg):
    """Wall-flip tracking cfg for SAC (identical fields, separate registry entry)."""


@registry.env("G1WallFlipTrackingSAC", sim_backend="mujoco")
@registry.env("G1WallFlipTrackingSAC", sim_backend="motrix")
class G1WallFlipTrackingSACEnv(G1MotionTrackingSACEnv):
    """Wall-flip tracking env with the SAC asymmetric actor-critic obs layout.

    The motrix backend is registered for sim2sim eval/playback only.
    """

    _cfg: G1WallFlipTrackingSACCfg
