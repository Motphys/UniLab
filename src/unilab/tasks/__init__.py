"""Production task registry bootstrap.

Concrete task implementations are moving from :mod:`unilab.envs` into this
package under issue #1112.  Until each task family moves, this explicit list
records its legacy module as the last remaining consumer of that path.  The
registry imports these leaf modules directly, so registration stays explicit
and deterministic throughout the migration.
"""

__unilab_registry_modules__ = (
    "unilab.tasks.locomotion.go1",
    "unilab.tasks.locomotion.go2",
    "unilab.tasks.locomotion.go2w",
    "unilab.tasks.locomotion.g1",
    "unilab.tasks.locomotion.go2_arm",
    "unilab.tasks.locomotion.a2",
    "unilab.tasks.manipulation.allegro_inhand",
    "unilab.tasks.manipulation.sharpa_inhand",
    "unilab.tasks.manipulation.stewart",
    "unilab.envs.motion_tracking.g1",
    "unilab.tasks.motion_tracking.x2",
)

__all__ = ["__unilab_registry_modules__"]
