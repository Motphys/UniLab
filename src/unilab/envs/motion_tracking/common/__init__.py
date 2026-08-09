"""Robot-agnostic motion-tracking engine and owner modules.

This package holds the motion-tracking task engine (:class:`MotionTrackingEnv`
/ :class:`MotionTrackingDeployEnv`) and its per-concern owner modules (config,
rewards, observations, terminations, transforms, reset, domain randomization,
motion loading). Per-robot profiles live under ``g1/`` and
``x2/`` and only carry robot-specific defaults and thin registry subclasses.
"""
