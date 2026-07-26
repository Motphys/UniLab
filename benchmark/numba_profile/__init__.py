"""Standalone numba-fusion profile for the ``g1_motion_tracking`` update_state.

Faithfully replicates the real task's reward + termination (no MuJoCo/Motrix
needed — the target is the backend-agnostic Env overhead) and implements the
structured numba scheme discussed for issues #663 / #665.  See README.md.
"""
