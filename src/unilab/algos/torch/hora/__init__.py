from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .distill import HoraDistillationTrainer
from .models import HoraActorModel, HoraCriticModel, HoraSharedActorCritic
from .ppo import HoraPPO

if TYPE_CHECKING:
    from .appo import HoraAPPORunner, play_hora_appo

__all__ = [
    "HoraActorModel",
    "HoraAPPORunner",
    "HoraCriticModel",
    "HoraDistillationTrainer",
    "HoraPPO",
    "HoraSharedActorCritic",
    "play_hora_appo",
]


def __getattr__(name: str) -> Any:
    if name in {"HoraAPPORunner", "play_hora_appo"}:
        from .appo import HoraAPPORunner, play_hora_appo

        exports = {
            "HoraAPPORunner": HoraAPPORunner,
            "play_hora_appo": play_hora_appo,
        }
        return exports[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
