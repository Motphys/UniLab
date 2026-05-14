from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class DomainRandConfig:
    randomize_base_mass: bool = False
    added_mass_range: list[float] = field(default_factory=lambda: [-1.5, 1.5])

    randomize_body_mass: bool = False
    body_mass_multiplier_range: list[float] = field(default_factory=lambda: [0.9, 1.1])

    random_com: bool = False
    com_offset_x: list[float] = field(default_factory=lambda: [-0.05, 0.05])

    randomize_gravity: bool = False
    gravity_range: list[list[float]] = field(
        default_factory=lambda: [[0.0, 0.0, -9.81], [0.0, 0.0, -9.81]]
    )

    randomize_ground_friction: bool = False
    ground_friction_multiplier_range: list[float] = field(default_factory=lambda: [0.8, 1.2])

    randomize_dof_armature: bool = False
    dof_armature_multiplier_range: list[float] = field(default_factory=lambda: [0.8, 1.2])

    push_robots: bool = False
    push_interval: int = 750  # step
    max_force: list[float] = field(default_factory=lambda: [1.0, 1.0, 0.5])
    push_body_name: str | None = None
