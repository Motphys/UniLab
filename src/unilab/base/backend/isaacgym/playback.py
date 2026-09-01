"""Compatibility import for the shared subprocess playback helper."""

from unilab.base.backend.subprocess_ipc.playback import run_subprocess_playback

run_isaacgym_playback = run_subprocess_playback

__all__ = ["run_isaacgym_playback"]
