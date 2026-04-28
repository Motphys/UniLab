from __future__ import annotations

import importlib
import sys

from rsl_rl.utils import resolve_callable


def test_hora_package_import_keeps_appo_lazy() -> None:
    sys.modules.pop("unilab.algos.torch.hora", None)
    sys.modules.pop("unilab.algos.torch.hora.appo", None)

    importlib.import_module("unilab.algos.torch.hora")

    assert "unilab.algos.torch.hora.appo" not in sys.modules


def test_resolve_callable_loads_hora_ppo_from_package_export() -> None:
    resolved = resolve_callable("unilab.algos.torch.hora:HoraPPO")

    from unilab.algos.torch.hora.ppo import HoraPPO

    assert resolved is HoraPPO
