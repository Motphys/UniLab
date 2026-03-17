"""Manipulation task registry.

Config factory functions have been removed.
Configurations are now managed via Hydra YAML files in conf/.
KNOWN_TASKS is kept for legacy routing utilities.
"""

KNOWN_TASKS: set = {
    "AllegroInhandRotation",
    "AllegroInhandRotationSac",
}
