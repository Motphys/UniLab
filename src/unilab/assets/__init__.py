"""UniLab assets module - centralized management of robot models and motion files."""

from pathlib import Path

# Root directory of all assets (robots, motions, etc.)
ASSETS_ROOT_PATH = Path(__file__).parent

__all__ = ["ASSETS_ROOT_PATH"]
