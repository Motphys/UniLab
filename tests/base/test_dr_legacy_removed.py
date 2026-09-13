from __future__ import annotations

import importlib
import subprocess
import sys
from pathlib import Path

import pytest


def test_legacy_dr_protocol_is_removed() -> None:
    import unilab.base.np_env as np_env

    assert not hasattr(np_env.NpEnv, "_init_domain_randomization")
    with pytest.raises(ModuleNotFoundError, match="No module named 'unilab.dr'"):
        importlib.import_module("unilab.dr")

    source = Path(np_env.__file__).read_text(encoding="utf-8")
    assert "_dr_manager" not in source
    assert "unilab.dr" not in source


def test_fresh_import_graph_does_not_load_legacy_dr_manager() -> None:
    code = "\n".join(
        [
            "import sys",
            "import unilab",
            "assert 'unilab.dr' not in sys.modules",
            "assert 'unilab.dr.manager' not in sys.modules",
            "assert 'unilab.dr.provider' not in sys.modules",
        ]
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr
