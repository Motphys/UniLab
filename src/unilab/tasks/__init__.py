"""Production task registry bootstrap.

Concrete task implementations live in this package.  The registry imports the
explicit leaf-module list directly, so registration stays deterministic and
does not depend on package discovery or import order.
"""

__unilab_registry_modules__ = (
    "unilab.tasks.locomotion.go1",
    "unilab.tasks.locomotion.go2",
    "unilab.tasks.locomotion.go2w",
    "unilab.tasks.locomotion.g1",
    "unilab.tasks.locomotion.go2_arm",
    "unilab.tasks.locomotion.a2",
    "unilab.tasks.locomotion.t800",
    "unilab.tasks.manipulation.allegro_inhand",
    "unilab.tasks.manipulation.sharpa_inhand",
    "unilab.tasks.manipulation.stewart",
    "unilab.tasks.motion_tracking.g1",
    "unilab.tasks.motion_tracking.x2",
)

__all__ = ["__unilab_registry_modules__"]
