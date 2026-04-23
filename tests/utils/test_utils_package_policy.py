import importlib
import sys
import warnings
from pathlib import Path

import unilab.utils

ALLOWED_UTILS_API = {"get_default_device", "to_numpy", "to_torch"}


def test_utils_api_is_whitelisted() -> None:
    assert set(unilab.utils.__all__) == ALLOWED_UTILS_API


def test_repo_has_no_package_level_utils_imports() -> None:
    current_file = Path(__file__).resolve()
    for root in (Path("src"), Path("tests"), Path("scripts")):
        for path in root.rglob("*.py"):
            if path.resolve() == current_file:
                continue
            assert "from unilab.utils import" not in path.read_text(encoding="utf-8"), path


def test_each_utils_shim_is_importable_and_warns_with_removal_target() -> None:
    shim_modules = sorted(
        f"unilab.utils.{path.stem}"
        for path in Path("src/unilab/utils").glob("*.py")
        if path.stem not in {"__init__", "device", "tensor"}
    )

    for module_name in shim_modules:
        sys.modules.pop(module_name, None)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", DeprecationWarning)
            module = importlib.import_module(module_name)
        assert module is not None
        assert any(item.category is DeprecationWarning for item in caught), module_name
        assert any("0.2.0" in str(item.message) for item in caught), module_name
