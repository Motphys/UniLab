"""UniLab package initialization."""

from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

try:
    __version__ = version("unilab")
except PackageNotFoundError:
    __version__ = "0.0.0"

ROOT_PATH = Path(__file__).resolve().parent
