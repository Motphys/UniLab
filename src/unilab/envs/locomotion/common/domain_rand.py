from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class DomainRandConfig:
    randomize_base_mass: bool = False
    added_mass_range: list[float] = field(default_factory=lambda: [-1.5, 1.5])

    random_com: bool = False
    com_offset_x: list[float] = field(default_factory=lambda: [-0.05, 0.05])

    push_robots: bool = False
    push_interval: int = 750  # step
    max_force: list[float] = field(default_factory=lambda: [1.0, 1.0, 0.5])
