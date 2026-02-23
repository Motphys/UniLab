try:
    from unilab.envs.locomotion.go2 import joystick  # noqa: F401
except ImportError:
    pass

try:
    from unilab.envs.locomotion.go2 import go2_loco  # noqa: F401
except ImportError:
    pass
