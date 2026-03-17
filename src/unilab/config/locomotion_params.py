"""Locomotion task registry.

Config factory functions (ppo_config, appo_config, offpolicy_config) have been
removed. Configurations are now managed via Hydra YAML files in conf/.
KNOWN_TASKS is kept for legacy routing utilities.
"""

KNOWN_TASKS: frozenset[str] = frozenset(
    {
        "Go1JoystickFlatTerrain",
        "Go2JoystickFlatTerrain",
        "G1JoystickFlatTerrain",
        "G1WalkTaskMjSAC",
        "G1MotionTracking",
    }
)
