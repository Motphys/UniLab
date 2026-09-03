import importlib
from pathlib import Path

import unilab.utils

ALLOWED_UTILS_API = {"get_default_device", "to_numpy", "to_torch"}
ALLOWED_UTILS_MODULES = {
    "__init__",
    "checkpoint",
    "device",
    "geometry",
    "monitoring",
    "nan_guard",
    "nan_viz",
    "reward",
    "rotation",
    "seed",
    "sim2sim",
    "tensor",
}


def test_utils_api_is_whitelisted() -> None:
    assert set(unilab.utils.__all__) == ALLOWED_UTILS_API


def test_utils_directory_is_whitelisted() -> None:
    modules = {path.stem for path in Path("src/unilab/utils").glob("*.py")}
    assert modules == ALLOWED_UTILS_MODULES


def test_repo_has_no_package_level_utils_imports() -> None:
    current_file = Path(__file__).resolve()
    for root in (Path("src"), Path("tests"), Path("scripts")):
        for path in root.rglob("*.py"):
            if path.resolve() == current_file:
                continue
            assert "from unilab.utils import" not in path.read_text(encoding="utf-8"), path


def test_algos_torch_common_no_longer_reexports_utils_primitives() -> None:
    common = importlib.import_module("unilab.algos.common")
    assert "get_default_device" not in common.__all__
    assert "to_numpy" not in common.__all__
    assert "to_torch" not in common.__all__
